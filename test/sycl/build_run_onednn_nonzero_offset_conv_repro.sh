#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"

src="${script_dir}/onednn_nonzero_offset_conv_repro.cpp"
out="${script_dir}/onednn_nonzero_offset_conv_repro"

dnnl_lib="${repo_root}/build/xpu_mkldnn_proj-prefix/src/xpu_mkldnn_proj-build/src/libdnnl.a"
if [[ ! -f "${dnnl_lib}" ]]; then
  dnnl_lib="${repo_root}/torch/lib/libdnnl.a"
fi

cxx="${CXX:-/opt/intel/oneapi/compiler/2025.3/bin/icpx}"

echo "[build] compiler: ${cxx}"
"${cxx}" \
  -std=c++17 \
  -O2 \
  -fsycl \
  -fopenmp \
  -I"${repo_root}/torch/include" \
  "${src}" \
  "${dnnl_lib}" \
  -L/lib/x86_64-linux-gnu -ldl -lpthread -lm -lgomp -l:libOpenCL.so.1 \
  -o "${out}"

echo "[run] ${out} --device-index 0"
DNNL_VERBOSE=1 ONEDNN_VERBOSE=1 "${out}" --device-index 0

echo
echo "Tip: override tolerance and device"
echo "  ${out} --device-index 0 --atol 1e-4"
