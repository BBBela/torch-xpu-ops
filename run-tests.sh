# PYTORCH_DEBUG_XPU_FALLBACK=1 PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=0 PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v --tb=no --disable-warnings \
# ~/pytorch/third_party/torch-xpu-ops/test/xpu/test_decomp_xpu.py::TestDecompXPU::test_quick_addmv_xpu_float64 \
# ~/pytorch/third_party/torch-xpu-ops/test/xpu/test_decomp_xpu.py::TestDecompXPU::test_comprehensive_baddbmm_xpu_float64 \
# ~/pytorch/third_party/torch-xpu-ops/test/xpu/test_decomp_xpu.py::TestDecompXPU::test_comprehensive_nn_functional_instance_norm_xpu_float64 \
# ~/pytorch/third_party/torch-xpu-ops/test/xpu/test_decomp_xpu.py::TestDecompXPU::test_quick_baddbmm_xpu_float64 \
# ~/pytorch/third_party/torch-xpu-ops/test/xpu/test_decomp_xpu.py::TestDecompXPU::test_quick_core_backward_baddbmm_xpu_float64

# PYTORCH_DEBUG_XPU_FALLBACK=1 PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=0 PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v --tb=no --disable-warnings \




PYTORCH_DEBUG_XPU_FALLBACK=1 PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=0 PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v --tb=no --disable-warnings \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_forward_mode_AD_addmv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_forward_mode_AD_mv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_forward_mode_AD_addmm_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_forward_mode_AD_matmul_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_forward_mode_AD___rmatmul___xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_forward_mode_AD_baddbmm_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_forward_mode_AD_addbmm_xpu_float64 \
\
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_inplace_forward_mode_AD_addmm_decomposed_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_inplace_forward_mode_AD_addmv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_inplace_forward_mode_AD_addmm_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_inplace_forward_mode_AD_baddbmm_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_inplace_forward_mode_AD_addbmm_xpu_float64 \
\
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_fn_fwgrad_bwgrad___rmatmul___xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_fn_fwgrad_bwgrad_mv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_fn_fwgrad_bwgrad_matmul_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_fn_fwgrad_bwgrad_addmv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_fwd_gradients_xpu.py::TestFwdGradientsXPU::test_fn_fwgrad_bwgrad_addr_xpu_float64 \
\
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_grad_matmul_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_grad___rmatmul___xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_grad_addmm_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_grad_cdist_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_grad_mv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_grad_addmv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_grad_baddbmm_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_grad_addbmm_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_grad_nn_functional_multi_head_attention_forward_xpu_float64 \
\
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_inplace_gradgrad_addr_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_inplace_gradgrad_addmv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_inplace_grad_addmm_decomposed_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_inplace_grad_addbmm_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_inplace_grad_addmm_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_inplace_grad_addmv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_inplace_grad_baddbmm_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_gradgrad_addr_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_gradgrad_matmul_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_gradgrad___rmatmul___xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_gradgrad_mv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_ops_gradients_xpu.py::TestBwdGradientsXPU::test_fn_gradgrad_addmv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_decomp_xpu.py::TestDecompXPU::test_quick_core_backward_mv_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_autograd_xpu.py::TestAutogradDeviceTypeXPU::test_mv_grad_stride_0_xpu \
\
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_modules_xpu.py::TestModuleXPU::test_grad_nn_MultiheadAttention_eval_mode_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_modules_xpu.py::TestModuleXPU::test_grad_nn_MultiheadAttention_train_mode_xpu_float64 \
\
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_modules_xpu.py::TestModuleXPU::test_gradgrad_nn_MultiheadAttention_eval_mode_xpu_float64 \
~/pytorch/third_party/torch-xpu-ops/test/xpu/test_modules_xpu.py::TestModuleXPU::test_gradgrad_nn_MultiheadAttention_train_mode_xpu_float64


PYTORCH_DEBUG_XPU_FALLBACK=1 PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=0 PYTORCH_TEST_WITH_SLOW=1 python -m pytest -v --tb=no --disable-warnings \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_forward_mode_AD___rmatmul___xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_forward_mode_AD_addmm_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_forward_mode_AD_addmv_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_forward_mode_AD_baddbmm_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_forward_mode_AD_matmul_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_forward_mode_AD_mv_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_inplace_forward_mode_AD_addmv_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_forward_mode_AD_addbmm_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_inplace_forward_mode_AD_addbmm_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_inplace_forward_mode_AD_addmm_decomposed_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_inplace_forward_mode_AD_addmm_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_inplace_forward_mode_AD_baddbmm_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_fn_fwgrad_bwgrad_matmul_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_fn_fwgrad_bwgrad_mv_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_fn_fwgrad_bwgrad_addmv_xpu_float64 \
~/pytorch/test/test_ops_fwd_gradients.py::TestFwdGradientsXPU::test_fn_fwgrad_bwgrad_addr_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_grad___rmatmul___xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_grad_addmm_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_grad_addmv_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_grad_baddbmm_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_grad_cdist_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_grad_matmul_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_grad_mv_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_grad_nn_functional_multi_head_attention_forward_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_inplace_grad_addmv_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_grad_addbmm_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_gradgrad___rmatmul___xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_inplace_grad_addbmm_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_inplace_grad_addmm_decomposed_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_inplace_grad_addmm_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_inplace_grad_baddbmm_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_gradgrad_matmul_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_gradgrad_mv_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_gradgrad_addmv_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_fn_gradgrad_addr_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_inplace_gradgrad_addmv_xpu_float64 \
~/pytorch/test/test_ops_gradients.py::TestBwdGradientsXPU::test_inplace_gradgrad_addr_xpu_float64
