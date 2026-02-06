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

template <
    typename scalar_t,
    typename IndexType,
    bool upper,
    bool inplace>
struct ApplyTriuTrilKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    IndexType linear_idx = item.get_global_id(0) * elements_per_thread_;
    if (linear_idx >= N_padded_) {
      return;
    }
    auto dims = self_info_.dims;

    // Compute column index amd row index
    IndexType col = linear_idx % last_dim_padded_;
    linear_idx /= last_dim_padded_;
    IndexType row = linear_idx % self_info_.sizes[dims - 2];

    if constexpr (inplace) {
      bool mask_all_true =
          upper ? (col - row >= k_) : (col + elements_per_thread_ - row <= k_);
      if (mask_all_true)
        return;
    }

    // Compute offset
    IndexType self_offset = 0, result_offset = 0;
    self_offset += self_info_.strides[dims - 1] * col;
    result_offset += result_info_.strides[dims - 1] * col;
    linear_idx /= self_info_.sizes[dims - 2];
    self_offset += self_info_.strides[dims - 2] * row;
    result_offset += result_info_.strides[dims - 2] * row;

    // Compute remaining offsets
    IndexType running_index;
    for (int i = dims - 3; i >= 0; --i) {
      running_index = linear_idx % self_info_.sizes[i];
      linear_idx /= self_info_.sizes[i];
      self_offset += running_index * self_info_.strides[i];
      result_offset += running_index * result_info_.strides[i];
    }

    if constexpr (inplace) {
      for (int i = 0;
           i < elements_per_thread_ && col + i < self_info_.sizes[dims - 1];
           i++) {
        bool mask = upper ? (col + i - row >= k_) : (col + i - row <= k_);
        if (!mask)
          result_info_
              .data[result_offset + i * result_info_.strides[dims - 1]] =
              scalar_t(0);
      }
    } else {
      // Process elements one by one without buffering
      for (int i = 0;
           i < elements_per_thread_ && col + i < self_info_.sizes[dims - 1];
           i++) {
        bool mask = upper ? (col + i - row >= k_) : (col + i - row <= k_);
        scalar_t value = mask ? 
            self_info_.data[self_offset + i * self_info_.strides[dims - 1]] : 
            scalar_t(0);
        result_info_.data[result_offset + i * result_info_.strides[dims - 1]] = value;
      }
    }
  }
  ApplyTriuTrilKernelFunctor(
      at::xpu::detail::TensorInfo<scalar_t, IndexType> result_info,
      at::xpu::detail::TensorInfo<const scalar_t, IndexType> self_info,
      const int64_t k,
      const int64_t N_padded,
      const IndexType last_dim_padded,
      const int elements_per_thread)
      : result_info_(result_info),
        self_info_(self_info),
        k_(k),
        N_padded_(N_padded),
        last_dim_padded_(last_dim_padded),
        elements_per_thread_(elements_per_thread) {}

 private:
  at::xpu::detail::TensorInfo<scalar_t, IndexType> result_info_;
  at::xpu::detail::TensorInfo<const scalar_t, IndexType> self_info_;
  const int64_t k_;
  const int64_t N_padded_;
  const IndexType last_dim_padded_;
  const int elements_per_thread_;
};

template <typename scalar_t, typename IndexType, bool upper>
void apply_triu_tril(
    const Tensor& result,
    const Tensor& self,
    const int64_t k) {
  constexpr int base_elements_per_thread =
      sizeof(scalar_t) < 8 ? 8 / sizeof(scalar_t) : 1;
  
  auto sizes = self.sizes();
  int64_t total_elements = c10::multiply_integers(sizes.begin(), sizes.end());
  
  // Calculate elements_per_thread to keep thread count within uint32 range
  // Use conservative limit to account for padding and local_range alignment
  constexpr int64_t max_threads = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) / 2;
  
  int elements_per_thread = base_elements_per_thread;
  while (total_elements / elements_per_thread > max_threads) {
    elements_per_thread *= 2;
  }
  
  // Kernel setup and launch
  int64_t last_dim_padded =
      round_up<int64_t>(sizes.back(), elements_per_thread);
  int64_t N_padded =
      c10::multiply_integers(sizes.begin(), sizes.end() - 1) * last_dim_padded;

  int64_t local_range = syclMaxWorkItemsPerSubSlice();
  int64_t global_range =
      ((N_padded / elements_per_thread + local_range - 1) / local_range) *
      local_range;

  auto result_info =
      at::xpu::detail::getTensorInfo<scalar_t, IndexType>(result);
  auto self_info =
      at::xpu::detail::getTensorInfo<const scalar_t, IndexType>(self);
  BOOL_SWITCH(self.is_same(result), inplace, [&] {
    ApplyTriuTrilKernelFunctor<
        scalar_t,
        IndexType,
        upper,
        inplace>
        kfn(result_info, self_info, k, N_padded, last_dim_padded, elements_per_thread);
    sycl_kernel_submit(
        sycl::range<1>(global_range),
        sycl::range<1>(local_range),
        getCurrentSYCLQueue(),
        kfn);
  });
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
