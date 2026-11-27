"""PyTorch 2.x CustomOp 注册中心与运行期原子计数器。

- `fused_rmsnorm_residual` / `fused_rmsnorm`：Python 侧注册 Schema 与 Meta 推导，
  由 `csrc/fused_add_rmsnorm_cuda.cu` 通过 `TORCH_LIBRARY_IMPL` 直连 `c10::Dispatcher`（绕过 GIL）；
- `fused_gqa_rope` / `rope_single` / `fused_swiglu`：绑定至 Triton 核函数并统计调用次数。
"""

import torch
import torch.nn.functional as F

from .cuda_ext import get_cuda_rmsnorm_ext
from .kernels import (
    triton_fused_gqa_rope_qk,
    triton_fused_swiglu,
    triton_rope_single,
)

_PY_RUNTIME_COUNTERS: dict[str, int] = {
    "fused_gqa_rope": 0,
    "rope_single": 0,
    "fused_swiglu": 0,
}


def reset_runtime_counters() -> None:
    for k in _PY_RUNTIME_COUNTERS:
        _PY_RUNTIME_COUNTERS[k] = 0
    if torch.cuda.is_available():
        get_cuda_rmsnorm_ext().reset_rmsnorm_calls()


def get_runtime_counters() -> dict[str, int]:
    add_calls = 0
    solo_calls = 0
    if torch.cuda.is_available():
        ext = get_cuda_rmsnorm_ext()
        add_calls = int(ext.get_rmsnorm_add_calls())
        solo_calls = int(ext.get_rmsnorm_solo_calls())
    return {
        "fused_rmsnorm_residual": add_calls,
        "fused_rmsnorm": solo_calls,
        "fused_gqa_rope": _PY_RUNTIME_COUNTERS["fused_gqa_rope"],
        "rope_single": _PY_RUNTIME_COUNTERS["rope_single"],
        "fused_swiglu": _PY_RUNTIME_COUNTERS["fused_swiglu"],
    }


def _ref_rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


if not hasattr(torch.ops, "ai_infra") or not hasattr(
    torch.ops.ai_infra, "fused_rmsnorm"
):
    _AI_INFRA_LIB = torch.library.Library("ai_infra", "DEF")

    _AI_INFRA_LIB.define(
        "fused_rmsnorm_residual(Tensor x, Tensor residual, Tensor weight, float eps=1e-6) -> (Tensor, Tensor)"
    )
    _AI_INFRA_LIB.define(
        "fused_rmsnorm(Tensor x, Tensor weight, float eps=1e-6) -> Tensor"
    )
    _AI_INFRA_LIB.define("rope_single(Tensor x, Tensor cos, Tensor sin) -> Tensor")
    _AI_INFRA_LIB.define(
        "fused_gqa_rope(Tensor q, Tensor k, Tensor cos, Tensor sin) -> (Tensor, Tensor)"
    )
    _AI_INFRA_LIB.define("fused_swiglu(Tensor gate_up) -> Tensor")

    # Triton 算子的 CUDA 实现（RMSNorm 双算子的 CUDA 实现直接由 `.cu` 内的 C++ `TORCH_LIBRARY_IMPL` 注册！）
    @torch.library.impl(_AI_INFRA_LIB, "rope_single", "CUDA")
    def _rope_single_cuda(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        _PY_RUNTIME_COUNTERS["rope_single"] += 1
        return triton_rope_single(x, cos, sin)

    @torch.library.impl(_AI_INFRA_LIB, "fused_gqa_rope", "CUDA")
    def _rope_gqa_cuda(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _PY_RUNTIME_COUNTERS["fused_gqa_rope"] += 1
        return triton_fused_gqa_rope_qk(q, k, cos, sin)

    @torch.library.impl(_AI_INFRA_LIB, "fused_swiglu", "CUDA")
    def _swiglu_cuda(gate_up: torch.Tensor) -> torch.Tensor:
        _PY_RUNTIME_COUNTERS["fused_swiglu"] += 1
        return triton_fused_swiglu(gate_up)

    # ---------------- CPU 回退实现 ----------------
    @torch.library.impl(_AI_INFRA_LIB, "fused_rmsnorm_residual", "CPU")
    def _rmsnorm_add_cpu(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        up_res = x + residual
        acc = up_res.to(torch.float32)
        var = acc.pow(2).mean(-1, keepdim=True)
        out = (acc * torch.rsqrt(var + eps) * weight.to(torch.float32)).to(up_res.dtype)
        return out, up_res

    @torch.library.impl(_AI_INFRA_LIB, "fused_rmsnorm", "CPU")
    def _rmsnorm_solo_cpu(
        x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        acc = x.to(torch.float32)
        var = acc.pow(2).mean(-1, keepdim=True)
        return (acc * torch.rsqrt(var + eps) * weight.to(torch.float32)).to(x.dtype)

    @torch.library.impl(_AI_INFRA_LIB, "rope_single", "CPU")
    def _rope_single_cpu(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return (x * cos) + (_ref_rotate_half(x) * sin)

    @torch.library.impl(_AI_INFRA_LIB, "fused_gqa_rope", "CPU")
    def _rope_gqa_cpu(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_out = (q * cos) + (_ref_rotate_half(q) * sin)
        k_out = (k * cos) + (_ref_rotate_half(k) * sin)
        return q_out, k_out

    @torch.library.impl(_AI_INFRA_LIB, "fused_swiglu", "CPU")
    def _swiglu_cpu(gate_up: torch.Tensor) -> torch.Tensor:
        g, u = gate_up.chunk(2, dim=-1)
        return (F.silu(g.float()) * u.float()).to(gate_up.dtype)

    # ---------------- Meta / FakeTensor 符号形状推导 ----------------
    @torch.library.impl(_AI_INFRA_LIB, "fused_rmsnorm_residual", "Meta")
    def _rmsnorm_add_meta(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch._check(x.shape == residual.shape)
        torch._check(x.shape[-1] == weight.shape[0])
        return torch.empty_like(x), torch.empty_like(residual)

    @torch.library.impl(_AI_INFRA_LIB, "fused_rmsnorm", "Meta")
    def _rmsnorm_solo_meta(
        x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        torch._check(x.shape[-1] == weight.shape[0])
        return torch.empty_like(x)

    @torch.library.impl(_AI_INFRA_LIB, "rope_single", "Meta")
    def _rope_single_meta(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return torch.empty_like(x)

    @torch.library.impl(_AI_INFRA_LIB, "fused_gqa_rope", "Meta")
    def _rope_gqa_meta(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch._check(q.shape[-1] == k.shape[-1])
        return torch.empty_like(q), torch.empty_like(k)

    @torch.library.impl(_AI_INFRA_LIB, "fused_swiglu", "Meta")
    def _swiglu_meta(gate_up: torch.Tensor) -> torch.Tensor:
        torch._check(gate_up.shape[-1] % 2 == 0)
        return torch.empty(
            (*gate_up.shape[:-1], gate_up.shape[-1] // 2),
            dtype=gate_up.dtype,
            device=gate_up.device,
        )

    if torch.cuda.is_available():
        get_cuda_rmsnorm_ext()


def fused_rmsnorm_residual_op(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.ai_infra.fused_rmsnorm_residual(x, residual, weight, eps)


def fused_rmsnorm_op(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return torch.ops.ai_infra.fused_rmsnorm(x, weight, eps)


def rope_single_op(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return torch.ops.ai_infra.rope_single(x, cos, sin)


def fused_gqa_rope_op(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.ai_infra.fused_gqa_rope(q, k, cos, sin)


def fused_swiglu_op(gate_up: torch.Tensor) -> torch.Tensor:
    return torch.ops.ai_infra.fused_swiglu(gate_up)
