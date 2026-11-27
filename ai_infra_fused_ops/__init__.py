"""
fused-cuda-triton-inductor: PyTorch 2.x TorchInductor ATen Post-Grad Fusion Pass + Native CUDA C++ (.cu) & Triton Kernels
"""

from .compiler_pass import (
    Llama3DecoderStack,
    Llama3TransformerBlock,
    OfficialHFQwen2StyleModel,
    disable_inductor_fusion_pass,
    eager_add_rmsnorm_fp32,
    eager_gqa_rope_qk,
    eager_rmsnorm_fp32,
    eager_rope_single,
    eager_swiglu,
    enable_inductor_fusion_pass,
    get_compile_counters,
    hf_official_rmsnorm,
    inductor_fusion_pass,
    reset_compile_counters,
)
from .cuda_ext import (
    cuda_fused_rmsnorm_residual,
    cuda_rmsnorm,
    inspect_cuda_ext_ptx_stats,
)
from .custom_ops import (
    fused_gqa_rope_op,
    fused_rmsnorm_op,
    fused_rmsnorm_residual_op,
    fused_swiglu_op,
    get_runtime_counters,
    reset_runtime_counters,
    rope_single_op,
)
from .kernels import (
    triton_fused_gqa_rope_qk,
    triton_fused_linear_cross_entropy,
    triton_fused_rmsnorm_residual,
    triton_fused_swiglu,
    triton_rmsnorm,
    triton_rope_single,
)

__all__ = [
    "cuda_fused_rmsnorm_residual",
    "cuda_rmsnorm",
    "inspect_cuda_ext_ptx_stats",
    "triton_fused_rmsnorm_residual",
    "triton_rmsnorm",
    "triton_rope_single",
    "triton_fused_swiglu",
    "triton_fused_gqa_rope_qk",
    "triton_fused_linear_cross_entropy",
    "fused_rmsnorm_residual_op",
    "fused_rmsnorm_op",
    "rope_single_op",
    "fused_gqa_rope_op",
    "fused_swiglu_op",
    "reset_runtime_counters",
    "get_runtime_counters",
    "eager_rmsnorm_fp32",
    "eager_add_rmsnorm_fp32",
    "hf_official_rmsnorm",
    "eager_rope_single",
    "eager_gqa_rope_qk",
    "eager_swiglu",
    "inductor_fusion_pass",
    "enable_inductor_fusion_pass",
    "disable_inductor_fusion_pass",
    "reset_compile_counters",
    "get_compile_counters",
    "Llama3TransformerBlock",
    "Llama3DecoderStack",
    "OfficialHFQwen2StyleModel",
]
