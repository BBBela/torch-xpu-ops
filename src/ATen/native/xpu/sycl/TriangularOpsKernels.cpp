/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ceil_div.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <limits>

#include <ATen/native/xpu/sycl/TriangularOpsKernels.h>

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

namespace at::native::xpu {

using namespace at::xpu;

// 2D kernel for all matrices - avoids uint32 overflow
template <
    typename scalar_t,
    typename IndexType,
    bool upper,
    int elements_per_thread,
    bool inplace>
struct ApplyTriuTrilKernel2DFunctor {
  void operator()(sycl::nd_item<2> item) const {
    // Get 2D coordinates: row and column block
    IndexType row = item.get_global_id(0);
    IndexType col_block = item.get_global_id(1);
    
    if (row >= matrix_rows_ || col_block >= col_blocks_) {
      return;
    }
    
    auto dims = self_info_.dims;
    constexpr IndexType cols_per_block = elements_per_thread;
    IndexType col_start = col_block * cols_per_block;
    
    // Compute batch offset (everything except last 2 dimensions)
    IndexType batch_size = c10::multiply_integers(self_info_.sizes, self_info_.sizes + dims - 2);
    
    for (IndexType batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      // Compute batch offsets for all dimensions except the last 2
      IndexType self_batch_offset = 0, result_batch_offset = 0;
      IndexType running_batch = batch_idx;
      
      for (int i = dims - 3; i >= 0; --i) {
        IndexType dim_idx = running_batch % self_info_.sizes[i];
        running_batch /= self_info_.sizes[i];
        self_batch_offset += dim_idx * self_info_.strides[i];
        result_batch_offset += dim_idx * result_info_.strides[i];
      }
      
      // Add row and column offsets
      IndexType self_offset = self_batch_offset + 
                              row * self_info_.strides[dims - 2] + 
                              col_start * self_info_.strides[dims - 1];
      IndexType result_offset = result_batch_offset + 
                                row * result_info_.strides[dims - 2] + 
                                col_start * result_info_.strides[dims - 1];
      
      if constexpr (inplace) {
        // Early exit optimization for inplace
        bool all_masked = upper ? (col_start - row >= k_) : (col_start + cols_per_block - row <= k_);
        if (all_masked) continue;
        
        // Process elements with unrolled loop for better performance
#pragma unroll
        for (int i = 0; i < cols_per_block; ++i) {
          IndexType col = col_start + i;
          if (col >= self_info_.sizes[dims - 1]) break;
          
          bool mask = upper ? (col - row >= k_) : (col - row <= k_);
          if (!mask) {
            result_info_.data[result_offset + i * result_info_.strides[dims - 1]] = scalar_t(0);
          }
        }
      } else {
        // Early exit optimizations for non-inplace case
        bool all_zero = upper ? (col_start + cols_per_block - 1 - row < k_) : (col_start - row > k_);
        bool all_copy = upper ? (col_start - row >= k_) : (col_start + cols_per_block - 1 - row <= k_);
        
        if (all_zero) {
          // All elements in this block should be zero
#pragma unroll
          for (int i = 0; i < cols_per_block; ++i) {
            IndexType col = col_start + i;
            if (col < self_info_.sizes[dims - 1]) {
              result_info_.data[result_offset + i * result_info_.strides[dims - 1]] = scalar_t(0);
            }
          }
        } else if (all_copy) {
          // All elements in this block should be copied directly
#pragma unroll
          for (int i = 0; i < cols_per_block; ++i) {
            IndexType col = col_start + i;
            if (col < self_info_.sizes[dims - 1]) {
              result_info_.data[result_offset + i * result_info_.strides[dims - 1]] = 
                  self_info_.data[self_offset + i * self_info_.strides[dims - 1]];
            }
          }
        } else {
          // Mixed case - need per-element masking
          // Load data into local array for better memory coalescing
          scalar_t values[elements_per_thread];
          bool masks[elements_per_thread];
          
          // Compute masks and load values in unrolled loops
#pragma unroll
          for (int i = 0; i < cols_per_block; ++i) {
            IndexType col = col_start + i;
            if (col < self_info_.sizes[dims - 1]) {
              masks[i] = upper ? (col - row >= k_) : (col - row <= k_);
              values[i] = masks[i] ? self_info_.data[self_offset + i * self_info_.strides[dims - 1]] : scalar_t(0);
            } else {
              values[i] = scalar_t(0);
            }
          }
          
          // Store results with unrolled loop
#pragma unroll
          for (int i = 0; i < cols_per_block; ++i) {
            IndexType col = col_start + i;
            if (col < self_info_.sizes[dims - 1]) {
              result_info_.data[result_offset + i * result_info_.strides[dims - 1]] = values[i];
            }
          }
        }
      }
    }
  }
  
  ApplyTriuTrilKernel2DFunctor(
      at::xpu::detail::TensorInfo<scalar_t, IndexType> result_info,
      at::xpu::detail::TensorInfo<const scalar_t, IndexType> self_info,
      const int64_t k,
      const IndexType matrix_rows,
      const IndexType col_blocks)
      : result_info_(result_info),
        self_info_(self_info),
        k_(k),
        matrix_rows_(matrix_rows),
        col_blocks_(col_blocks) {}

 private:
  at::xpu::detail::TensorInfo<scalar_t, IndexType> result_info_;
  at::xpu::detail::TensorInfo<const scalar_t, IndexType> self_info_;
  const int64_t k_;
  const IndexType matrix_rows_;
  const IndexType col_blocks_;
};

template <typename scalar_t, typename IndexType, bool upper, int elements_per_thread>
void apply_triu_tril_2d(
    const Tensor& result,
    const Tensor& self,
    const int64_t k) {
  auto sizes = self.sizes();
  auto dims = sizes.size();
  
  IndexType matrix_rows = sizes[dims - 2];
  IndexType matrix_cols = sizes[dims - 1];
  
  constexpr int64_t max_uint32 = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
  constexpr int64_t max_dim_size = max_uint32 / 2; // Conservative limit
  
  // Ensure matrix_rows fits in uint32
  if (matrix_rows > max_dim_size) {
    TORCH_CHECK(false, "Matrix has too many rows (", matrix_rows, ") for 2D kernel. Maximum supported: ", max_dim_size);
  }
  
  int64_t col_blocks = (matrix_cols + elements_per_thread - 1) / elements_per_thread;
  
  // Use 2D grid: rows x column_blocks
  int64_t local_range_x = 16; // rows per workgroup
  int64_t local_range_y = 16; // column blocks per workgroup
  
  int64_t global_range_x = ((matrix_rows + local_range_x - 1) / local_range_x) * local_range_x;
  int64_t global_range_y = ((col_blocks + local_range_y - 1) / local_range_y) * local_range_y;
  
  // Final safety checks
  TORCH_CHECK(global_range_x <= max_uint32, "Computed global_range_x (", global_range_x, ") exceeds uint32 maximum");
  TORCH_CHECK(global_range_y <= max_uint32, "Computed global_range_y (", global_range_y, ") exceeds uint32 maximum");

  auto result_info =
      at::xpu::detail::getTensorInfo<scalar_t, IndexType>(result);
  auto self_info =
      at::xpu::detail::getTensorInfo<const scalar_t, IndexType>(self);
      
  BOOL_SWITCH(self.is_same(result), inplace, [&] {
    ApplyTriuTrilKernel2DFunctor<
        scalar_t,
        IndexType,
        upper,
        elements_per_thread,
        inplace>
        kfn(result_info, self_info, k, matrix_rows, col_blocks);
    sycl_kernel_submit(
        sycl::range<2>(global_range_x, global_range_y),
        sycl::range<2>(local_range_x, local_range_y),
        getCurrentSYCLQueue(),
        kfn);
  });
}

template <typename scalar_t, typename IndexType, bool upper>
void apply_triu_tril(
    const Tensor& result,
    const Tensor& self,
    const int64_t k) {
  constexpr int base_elements_per_thread =
      sizeof(scalar_t) < 8 ? 8 / sizeof(scalar_t) : 1;
  
  // Always use 2D kernel for all matrix sizes
  apply_triu_tril_2d<scalar_t, IndexType, upper, base_elements_per_thread>(result, self, k);
}

#define TRIU_TRIL_LAMBDA(upper)                                   \
  [&] {                                                           \
    if (canUse32BitIndexMath(self)) {                             \
      apply_triu_tril<scalar_t, int32_t, upper>(result, self, k); \
    } else {                                                      \
      apply_triu_tril<scalar_t, int64_t, upper>(result, self, k); \
    }                                                             \
  }

void tril_kernel(const Tensor& result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::ComplexHalf,
      self.scalar_type(),
      "tril_xpu",
      TRIU_TRIL_LAMBDA(false));
}

void triu_kernel(const Tensor& result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return;
  }
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::ComplexHalf,
      self.scalar_type(),
      "triu_xpu",
      TRIU_TRIL_LAMBDA(true));
}

} // namespace at::native::xpu
