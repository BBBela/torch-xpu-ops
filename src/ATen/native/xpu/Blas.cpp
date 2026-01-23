/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/native/xpu/Blas.h>
#if defined(USE_ONEMKL_XPU)
#include <ATen/native/xpu/mkl/BlasImpl.h>
#endif
#include <torch/library.h>

namespace at::native {

Tensor& mm_complex_out_xpu(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  c10::DeviceGuard guard(self.device());
  TORCH_CHECK(
      self.is_complex(),
      "mm_complex_out_xpu expects self to be a complex datatype.");

  if (out.numel() == 0) {
    return out;
  }

#if defined(USE_ONEMKL_XPU)
  return at::native::xpu::mm_complex_out_xpu_mkl(self, mat2, out);
#else
  TORCH_CHECK(
      false,
      "Complex datatype matmul is not supported in oneDNN. Please include oneMKL library in compilation.");
#endif // USE_ONEMKL_XPU
}

Tensor& bmm_complex_out_xpu(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  c10::DeviceGuard guard(self.device());
  TORCH_CHECK(
      self.is_complex(),
      "bmm_complex_out_xpu expects self to be a complex datatype.");

  if (out.numel() == 0) {
    return out;
  }

#if defined(USE_ONEMKL_XPU)
  return at::native::xpu::bmm_complex_out_xpu_mkl(self, mat2, out);
#else
  TORCH_CHECK(
      false,
      "Complex datatype matmul is not supported in oneDNN. Please include oneMKL library in compilation.");
#endif // USE_ONEMKL_XPU
}

Tensor& addmm_complex_out_xpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  c10::DeviceGuard guard(self.device());
  TORCH_CHECK(
      self.is_complex(),
      "addmm_complex_out_xpu expects self to be a complex datatype.");

  if (out.numel() == 0) {
    return out;
  }
  if (mat1.numel() == 0) {
    if (beta.toComplexDouble() == 0.0) {
      out.zero_();
    } else {
      if (!self.is_same(out)) {
        out.copy_(self);
      }
      out.mul_(beta);
    }
    return out;
  }

#if defined(USE_ONEMKL_XPU)
  return at::native::xpu::addmm_complex_out_xpu_mkl(
      self, mat1, mat2, beta, alpha, out);
#else
  TORCH_CHECK(
      false,
      "Complex datatype matmul is not supported in oneDNN. Please include oneMKL library in compilation.");
#endif // USE_ONEMKL_XPU
}

Tensor& baddbmm_complex_out_xpu(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  c10::DeviceGuard guard(self.device());
  TORCH_CHECK(
      self.is_complex(),
      "baddbmm_complex_out_xpu expects self to be a complex datatype.");

  if (out.numel() == 0) {
    return out;
  }
  if (batch1.numel() == 0) {
    if (beta.toComplexDouble() == 0.0) {
      out.zero_();
    } else {
      if (!self.is_same(out)) {
        out.copy_(self);
      }
      out.mul_(beta);
    }
    return out;
  }

#if defined(USE_ONEMKL_XPU)
  return at::native::xpu::baddbmm_complex_out_xpu_mkl(
      self, batch1, batch2, beta, alpha, out);
#else
  TORCH_CHECK(
      false,
      "Complex datatype matmul is not supported in oneDNN. Please include oneMKL library in compilation.");
#endif // USE_ONEMKL_XPU
}

// XPU implementation of mv using mm for better numerical stability
Tensor mv_xpu(const Tensor& self, const Tensor& vec) {
  TORCH_CHECK(self.dim() == 2, "mv expects a matrix");
  TORCH_CHECK(vec.dim() == 1, "mv expects a vector");
  TORCH_CHECK(self.size(1) == vec.size(0), "size mismatch");
  
  // Use direct mm instead of decomposition to avoid precision issues
  // mv(A, v) = mm(A, v.unsqueeze(-1)).squeeze(-1)
  Tensor vec_reshaped = vec.view({vec.size(0), 1});
  Tensor result_2d = at::mm(self, vec_reshaped);
  return result_2d.view({self.size(0)});
}

// Register XPU implementations
TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("mv", TORCH_FN(mv_xpu));
}

} // namespace at::native
