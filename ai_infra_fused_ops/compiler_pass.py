"""
基于 `torch._inductor.pattern_matcher` 的两阶段推理图（Inference-Only `post_grad`）ATen 图算融合 Pass

核心设计与边界界定：
1. 适用范围声明：本 Pass 基于 `pm.fwd_only` 构建，专为推理阶段（`torch.inference_mode()` / `torch.no_grad()`）设计；
   训练阶段（需保留前向中间张量供反向图消费）的大词表显存与吞吐瓶颈由 `triton_fused_linear_cross_entropy` 自定义 `autograd.Function` 承担。
2. 覆盖三大主流 RMSNorm 写法方言（均覆盖 `Add + RMSNorm` 双输出/单输出及首层独立 `RMSNorm`，支持 FP16/BF16 与 `eps=1e-6/1e-5`）：
   - 方言 A：显式 FP32 权重提升 `(acc * rsqrt(var + eps) * w.float()).to(dtype)`；
   - 方言 B：HuggingFace 官方 `LlamaRMSNorm` / `Qwen2RMSNorm` 标准写法 `w * (acc * rsqrt(var + eps)).to(dtype)`（及乘法交换律形式）；
   - 方言 C：PyTorch 2.4+ 原生算子 `F.rms_norm(x, (N,), w, eps)`（覆盖 2D 与 3D 输入）。
3. 两阶段 GQA RoPE 重写与动态传递闭包防死环保证：
   - Stage 1 将单分支 7 节点 RoPE 规约为 `ai_infra::rope_single`；
   - Stage 2 在每次成功横向合并一对兄弟 `(nq, nk) -> fused_gqa_rope` 后，**立即执行 `pm.stable_topological_sort(graph)` 并动态重算全图祖先位掩码 `anc_mask`**，
     严密防止多组 `(cos_a, sin_a)` / `(cos_b, sin_b)` 交叉依赖场景下因静态掩码过期导致的 `P1 <-> P2` 有向死环。
4. 提供异常安全的上下文管理器 `with inductor_fusion_pass():`，确保退出作用域或发生异常时自动还原全局 `inductor_config.post_grad_custom_post_pass`。
"""

import contextlib
import operator
from typing import Any, Dict, Iterator, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._inductor.config as inductor_config
from torch._inductor import pattern_matcher as pm
from torch._inductor.custom_graph_pass import CustomGraphPass

from .cuda_ext import ensure_linux_cuda_toolchain
from . import custom_ops as _custom_ops  # noqa: F401

ensure_linux_cuda_toolchain()

COMPILE_MATCH_COUNTERS: Dict[str, int] = {
    "fused_rmsnorm_residual": 0,
    "fused_rmsnorm": 0,
    "fused_gqa_rope": 0,
    "rope_single": 0,
    "fused_swiglu": 0,
}

_PASSES_INITIALIZED = False
_RMSNORM_ADD_DUAL_PASS = pm.PatternMatcherPass()
_RMSNORM_ADD_SINGLE_PASS = pm.PatternMatcherPass()
_RMSNORM_SOLO_PASS = pm.PatternMatcherPass()
_ROPE_SINGLE_PASS = pm.PatternMatcherPass()
_SWIGLU_PASS = pm.PatternMatcherPass()


def reset_compile_counters() -> None:
    for k in COMPILE_MATCH_COUNTERS:
        COMPILE_MATCH_COUNTERS[k] = 0


def get_compile_counters() -> Dict[str, int]:
    return dict(COMPILE_MATCH_COUNTERS)


# ==============================================================================
# 1. 多方言参考算子定义（覆盖 FP32-Weight、HuggingFace 官方 Qwen2/Llama 与原生 F.rms_norm）
# ==============================================================================

def eager_rmsnorm_fp32(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    acc = x.to(torch.float32)
    var = acc.pow(2).mean(-1, keepdim=True)
    return (acc * torch.rsqrt(var + eps) * weight.to(torch.float32)).to(x.dtype)


def eager_add_rmsnorm_fp32(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    up_res = x + residual
    return eager_rmsnorm_fp32(up_res, weight, eps), up_res


def hf_official_rmsnorm(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """与 HuggingFace `transformers` 官方 `Qwen2RMSNorm.forward` / `LlamaRMSNorm.forward` 逐行一致"""
    input_dtype = hidden_states.dtype
    hidden_fp32 = hidden_states.to(torch.float32)
    variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
    normed = hidden_fp32 * torch.rsqrt(variance + eps)
    return weight * normed.to(input_dtype)


def _hf_official_rmsnorm_commuted(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_fp32 = hidden_states.to(torch.float32)
    variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
    normed = hidden_fp32 * torch.rsqrt(variance + eps)
    return normed.to(input_dtype) * weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def eager_rope_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def eager_gqa_rope_qk(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = eager_rope_single(q, cos, sin)
    k_embed = eager_rope_single(k, cos, sin)
    return q_embed, k_embed


def eager_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


# ==============================================================================
# 2. 注册 Stage-1 ATen Post-Grad 纵向模式与 Stage-2 动态闭包横向融合 Pass
# ==============================================================================

def _register_norm_dialect(
    norm_fn,
    sample_shape: Tuple[int, ...],
    dev: torch.device,
    eps_val: float,
) -> None:
    n_dim = sample_shape[-1]

    def _add_dual_pat(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor):
        u = x + res
        return norm_fn(u, w, eps_val), u

    def _add_dual_rep(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor):
        return torch.ops.ai_infra.fused_rmsnorm_residual(x, res, w, eps_val)

    def _add_single_pat(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor):
        return norm_fn(x + res, w, eps_val)

    def _add_single_rep(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor):
        out, _ = torch.ops.ai_infra.fused_rmsnorm_residual(x, res, w, eps_val)
        return out

    def _solo_pat(x: torch.Tensor, w: torch.Tensor):
        return norm_fn(x, w, eps_val)

    def _solo_rep(x: torch.Tensor, w: torch.Tensor):
        return torch.ops.ai_infra.fused_rmsnorm(x, w, eps_val)

    pm.register_replacement(
        _add_dual_pat,
        _add_dual_rep,
        [
            torch.empty(sample_shape, device=dev, dtype=torch.float16),
            torch.empty(sample_shape, device=dev, dtype=torch.float16),
            torch.empty((n_dim,), device=dev, dtype=torch.float16),
        ],
        pm.fwd_only,
        _RMSNORM_ADD_DUAL_PASS,
    )
    pm.register_replacement(
        _add_single_pat,
        _add_single_rep,
        [
            torch.empty(sample_shape, device=dev, dtype=torch.float16),
            torch.empty(sample_shape, device=dev, dtype=torch.float16),
            torch.empty((n_dim,), device=dev, dtype=torch.float16),
        ],
        pm.fwd_only,
        _RMSNORM_ADD_SINGLE_PASS,
    )
    pm.register_replacement(
        _solo_pat,
        _solo_rep,
        [
            torch.empty(sample_shape, device=dev, dtype=torch.float16),
            torch.empty((n_dim,), device=dev, dtype=torch.float16),
        ],
        pm.fwd_only,
        _RMSNORM_SOLO_PASS,
    )


def _torch_native_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), w, eps)


def _init_inductor_passes_once() -> None:
    global _PASSES_INITIALIZED
    if _PASSES_INITIALIZED or not torch.cuda.is_available():
        return

    dev = torch.device("cuda:0")

    for eps_val in (1e-6, 1e-5):
        # 方言 A: FP32 权重归一化；方言 B1/B2: HuggingFace 官方 Qwen2/Llama RMSNorm
        for fn in (eager_rmsnorm_fp32, hf_official_rmsnorm, _hf_official_rmsnorm_commuted):
            _register_norm_dialect(fn, (2, 16, 128), dev, eps_val)
        # 方言 C: PyTorch 原生 F.rms_norm（因 ATen 分解为正数维度轴 [1] / [2]，同时注册 2D 与 3D 模板）
        for shape in ((16, 128), (2, 16, 128)):
            _register_norm_dialect(_torch_native_rmsnorm, shape, dev, eps_val)

    def _swiglu_pat(gate_up: torch.Tensor):
        return eager_swiglu(gate_up)

    def _swiglu_rep(gate_up: torch.Tensor):
        return torch.ops.ai_infra.fused_swiglu(gate_up)

    pm.register_replacement(
        _swiglu_pat,
        _swiglu_rep,
        [torch.empty((2, 16, 256), device=dev, dtype=torch.float16)],
        pm.fwd_only,
        _SWIGLU_PASS,
    )

    def _rope_single_pat(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return eager_rope_single(x, cos, sin)

    def _rope_single_rep(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return torch.ops.ai_infra.rope_single(x, cos, sin)

    for head_dim in (64, 128):
        pm.register_replacement(
            _rope_single_pat,
            _rope_single_rep,
            [
                torch.empty((2, 16, 8, head_dim), device=dev, dtype=torch.float16),
                torch.empty((1, 16, 1, head_dim), device=dev, dtype=torch.float16),
                torch.empty((1, 16, 1, head_dim), device=dev, dtype=torch.float16),
            ],
            pm.fwd_only,
            _ROPE_SINGLE_PASS,
        )

    _PASSES_INITIALIZED = True


def _compute_anc_mask(graph: torch.fx.Graph):
    nodes = list(graph.nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    anc_mask: Dict[torch.fx.Node, int] = {}
    for i, n in enumerate(nodes):
        m = 1 << i
        for inp in n.all_input_nodes:
            m |= anc_mask[inp]
        anc_mask[n] = m
    return nodes, node_idx, anc_mask


def _fuse_sibling_qk_rope(graph: torch.fx.Graph) -> Tuple[int, int]:
    """
    Stage 2: 动态传递闭包更新的横向兄弟 RoPE 合并。
    每次成功合并一对节点 `(nq, nk)` 后，立即对修改后的计算图执行 `pm.stable_topological_sort(graph)`
    并重算全图 `anc_mask`，从而将合并产生的新传递闭包边（如 `C -> B` 与 `A -> D` 合并 `(A, B)` 后形成的 `C -> P1 -> D`）
    实时反映到后续可达性判定中，彻底阻断多 `(cos, sin)` 交叉依赖死环。
    """
    rope_target = torch.ops.ai_infra.rope_single.default
    init_ropes = [n for n in graph.nodes if n.op == "call_function" and n.target == rope_target]
    if len(init_ropes) < 2:
        return 0, len(init_ropes)

    gqa_fused = 0

    while True:
        nodes, node_idx, anc_mask = _compute_anc_mask(graph)
        rope_nodes = [n for n in nodes if n.op == "call_function" and n.target == rope_target]
        if len(rope_nodes) < 2:
            break

        merged_this_iter = False
        for i, n1 in enumerate(rope_nodes):
            x1, cos1, sin1 = n1.args[0], n1.args[1], n1.args[2]
            v1 = n1.meta.get("val")
            if not isinstance(v1, torch.Tensor) or v1.ndim != 4:
                continue

            for j in range(i + 1, len(rope_nodes)):
                n2 = rope_nodes[j]
                x2, cos2, sin2 = n2.args[0], n2.args[1], n2.args[2]
                if cos1 is not cos2 or sin1 is not sin2:
                    continue

                # 在最新计算图传递闭包上校验严格反链（互不可达）
                if ((anc_mask[n2] >> node_idx[n1]) & 1) != 0 or ((anc_mask[n1] >> node_idx[n2]) & 1) != 0:
                    continue

                v2 = n2.meta.get("val")
                if not isinstance(v2, torch.Tensor) or v2.ndim != 4:
                    continue
                if (v1.shape[0], v1.shape[1], v1.shape[3], v1.dtype) != (v2.shape[0], v2.shape[1], v2.shape[3], v2.dtype):
                    continue

                if v1.shape[2] >= v2.shape[2]:
                    nq, xq, vq = n1, x1, v1
                    nk, xk, vk = n2, x2, v2
                else:
                    nq, xq, vq = n2, x2, v2
                    nk, xk, vk = n1, x1, v1

                with graph.inserting_after(n2):
                    fused = graph.call_function(
                        torch.ops.ai_infra.fused_gqa_rope.default,
                        args=(xq, xk, cos1, sin1),
                    )
                    fused.meta["val"] = (vq, vk)
                    q_out = graph.call_function(operator.getitem, args=(fused, 0))
                    q_out.meta["val"] = vq
                    k_out = graph.call_function(operator.getitem, args=(fused, 1))
                    k_out.meta["val"] = vk

                nq.replace_all_uses_with(q_out)
                nk.replace_all_uses_with(k_out)
                graph.erase_node(nq)
                graph.erase_node(nk)

                # 立即恢复拓扑序并在下一轮迭代重算传递闭包位掩码！
                pm.stable_topological_sort(graph)
                gqa_fused += 1
                merged_this_iter = True
                break

            if merged_this_iter:
                break

        if not merged_this_iter:
            break

    rem_single = sum(1 for n in graph.nodes if n.op == "call_function" and n.target == rope_target)
    return gqa_fused, rem_single


def _ai_infra_post_grad_pass(graph: torch.fx.Graph) -> None:
    """挂载至 `torch._inductor.config.post_grad_custom_post_pass` 的推理图重写入口"""
    # 若检测到 AOTAutograd 反向图（含 tangent 占位符），直接跳过，严格限定于前向推理图
    if any(n.op == "placeholder" and "tangent" in n.name for n in graph.nodes):
        return

    c_norm_dual = _RMSNORM_ADD_DUAL_PASS.apply(graph)
    if c_norm_dual > 0:
        graph.eliminate_dead_code()

    c_norm_single = _RMSNORM_ADD_SINGLE_PASS.apply(graph)
    if c_norm_single > 0:
        graph.eliminate_dead_code()

    c_norm_solo = _RMSNORM_SOLO_PASS.apply(graph)
    c_swiglu = _SWIGLU_PASS.apply(graph)
    c_rope_raw = _ROPE_SINGLE_PASS.apply(graph)

    c_rope_gqa = 0
    c_rope_rem = c_rope_raw
    if c_rope_raw > 0:
        pm.stable_topological_sort(graph)
        graph.eliminate_dead_code()
        c_rope_gqa, c_rope_rem = _fuse_sibling_qk_rope(graph)

    c_norm_add = c_norm_dual + c_norm_single
    COMPILE_MATCH_COUNTERS["fused_rmsnorm_residual"] += c_norm_add
    COMPILE_MATCH_COUNTERS["fused_rmsnorm"] += c_norm_solo
    COMPILE_MATCH_COUNTERS["fused_gqa_rope"] += c_rope_gqa
    COMPILE_MATCH_COUNTERS["rope_single"] += c_rope_rem
    COMPILE_MATCH_COUNTERS["fused_swiglu"] += c_swiglu

    if (c_norm_add + c_norm_solo + c_swiglu + c_rope_raw) > 0:
        pm.stable_topological_sort(graph)
        graph.eliminate_dead_code()
        graph.lint()


class AIInfraPostGradPass(CustomGraphPass):
    """
    继承 TorchInductor 官方 `CustomGraphPass` 接口：
    - `uuid()` 返回 `None` 显式告知 Inductor `FXGraphCache` 绕过磁盘/内存缓存复用，
      既彻底隔离默认 `torch.compile` 与自定义 Pass 的编译缓存（杜绝缓存串扰），
      又保证每次 `torch.compile` 均真实触发 `COMPILE_MATCH_COUNTERS` 统计。
    """
    def __call__(self, graph: torch.fx.Graph) -> None:
        _ai_infra_post_grad_pass(graph)

    def uuid(self) -> Optional[Any]:
        return None


_PASS_INSTANCE = AIInfraPostGradPass()
_PREV_FX_CACHE_STATE: Optional[bool] = None


def enable_inductor_fusion_pass() -> None:
    """开启自定义 TorchInductor Post-Grad 融合 Pass（推荐使用 `with inductor_fusion_pass():` 上下文管理器）"""
    global _PREV_FX_CACHE_STATE
    _init_inductor_passes_once()
    if _PREV_FX_CACHE_STATE is None and hasattr(inductor_config, "fx_graph_cache"):
        _PREV_FX_CACHE_STATE = bool(inductor_config.fx_graph_cache)
    if hasattr(inductor_config, "fx_graph_cache"):
        inductor_config.fx_graph_cache = False
    inductor_config.post_grad_custom_post_pass = _PASS_INSTANCE


def disable_inductor_fusion_pass() -> None:
    """关闭自定义 Pass，恢复原生默认 TorchInductor 及缓存配置"""
    global _PREV_FX_CACHE_STATE
    inductor_config.post_grad_custom_post_pass = None
    if _PREV_FX_CACHE_STATE is not None and hasattr(inductor_config, "fx_graph_cache"):
        inductor_config.fx_graph_cache = _PREV_FX_CACHE_STATE
        _PREV_FX_CACHE_STATE = None


@contextlib.contextmanager
def inductor_fusion_pass() -> Iterator[None]:
    """异常安全的上下文管理器：在作用域内启用自定义融合 Pass 并隔离 FXGraphCache，退出或异常时自动还原先前配置"""
    _init_inductor_passes_once()
    prev_pass = inductor_config.post_grad_custom_post_pass
    has_cache_cfg = hasattr(inductor_config, "fx_graph_cache")
    prev_cache = bool(inductor_config.fx_graph_cache) if has_cache_cfg else False
    inductor_config.post_grad_custom_post_pass = _PASS_INSTANCE
    if has_cache_cfg:
        inductor_config.fx_graph_cache = False
    try:
        yield
    finally:
        inductor_config.post_grad_custom_post_pass = prev_pass
        if has_cache_cfg:
            inductor_config.fx_graph_cache = prev_cache


# ==============================================================================
# 3. 测试与评测模型（含标准 LLaMA-3 Stack 与 HuggingFace 官方写法 Qwen2.5 真实权重层）
# ==============================================================================

class Llama3TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2048,
        intermediate_dim: int = 5632,
        num_q_heads: int = 16,
        num_kv_heads: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert hidden_dim % num_q_heads == 0
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_q_heads
        self.eps = eps

        kv_dim = num_kv_heads * self.head_dim
        self.attn_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.ffn_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            up_res_1 = x
            norm_1 = eager_rmsnorm_fp32(x, self.attn_norm_weight, self.eps)
        else:
            norm_1, up_res_1 = eager_add_rmsnorm_fp32(x, residual, self.attn_norm_weight, self.eps)

        bsz, seq_len, _ = norm_1.shape
        q = self.q_proj(norm_1).view(bsz, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(norm_1).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(norm_1).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q_rot, k_rot = eager_gqa_rope_qk(q, k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q_rot.transpose(1, 2),
            k_rot.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=(seq_len > 1),
            enable_gqa=True,
        )
        attn_proj = self.o_proj(attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim))

        norm_2, up_res_2 = eager_add_rmsnorm_fp32(attn_proj, up_res_1, self.ffn_norm_weight, self.eps)

        gate_up = self.gate_up_proj(norm_2)
        mlp_act = eager_swiglu(gate_up)
        mlp_out = self.down_proj(mlp_act)

        return mlp_out, up_res_2


class Llama3DecoderStack(nn.Module):
    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 2048,
        intermediate_dim: int = 5632,
        num_q_heads: int = 16,
        num_kv_heads: int = 4,
        max_seq_len: int = 4096,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.head_dim = hidden_dim // num_q_heads
        self.eps = eps
        self.layers = nn.ModuleList(
            [
                Llama3TransformerBlock(
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm_weight = nn.Parameter(torch.ones(hidden_dim))

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().view(1, max_seq_len, 1, self.head_dim), persistent=False)
        self.register_buffer("sin_cached", emb.sin().view(1, max_seq_len, 1, self.head_dim), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        cos = self.cos_cached[:, :seq_len, :, :].to(hidden_states.dtype)
        sin = self.sin_cached[:, :seq_len, :, :].to(hidden_states.dtype)

        residual: Optional[torch.Tensor] = None
        x = hidden_states
        for layer in self.layers:
            x, residual = layer(x, residual, cos, sin)

        assert residual is not None
        final_out, _ = eager_add_rmsnorm_fp32(x, residual, self.final_norm_weight, self.eps)
        return final_out


class OfficialHFQwen2StyleModel(nn.Module):
    """
    严格按照 HuggingFace `transformers.models.qwen2.modeling_qwen2` 官方源码结构编写的模型：
    - 使用 `hf_official_rmsnorm`（先 `.to(input_dtype)` 再乘 `self.weight`）；
    - 使用标准 HuggingFace `residual = hidden_states; hidden_states = residual + attn_out` 隐式残差流（而非手动传递元组）；
    - 支持直接加载本机 HuggingFace 缓存中的 `Qwen2.5-0.5B-Instruct` 真实 `model.safetensors` 权重（`14Q / 2KV, D=64, H=896`）。
    """
    def __init__(
        self,
        num_layers: int = 2,
        hidden_size: int = 896,
        intermediate_size: int = 4864,
        num_q_heads: int = 14,
        num_kv_heads: int = 2,
        eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_q_heads
        self.eps = eps

        kv_dim = num_kv_heads * self.head_dim
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict(
                {
                    "input_layernorm": nn.ParameterList([nn.Parameter(torch.ones(hidden_size))]),
                    "q_proj": nn.Linear(hidden_size, hidden_size, bias=qkv_bias),
                    "k_proj": nn.Linear(hidden_size, kv_dim, bias=qkv_bias),
                    "v_proj": nn.Linear(hidden_size, kv_dim, bias=qkv_bias),
                    "o_proj": nn.Linear(hidden_size, hidden_size, bias=False),
                    "post_attention_layernorm": nn.ParameterList([nn.Parameter(torch.ones(hidden_size))]),
                    "gate_up_proj": nn.Linear(hidden_size, 2 * intermediate_size, bias=False),
                    "down_proj": nn.Linear(intermediate_size, hidden_size, bias=False),
                }
            )
            self.layers.append(layer)

        self.norm_weight = nn.Parameter(torch.ones(hidden_size))

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        t = torch.arange(2048, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().view(1, 2048, 1, self.head_dim), persistent=False)
        self.register_buffer("sin_cached", emb.sin().view(1, 2048, 1, self.head_dim), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        cos = self.cos_cached[:, :seq_len, :, :].to(hidden_states.dtype)
        sin = self.sin_cached[:, :seq_len, :, :].to(hidden_states.dtype)

        for layer in self.layers:
            residual = hidden_states
            hidden_states = hf_official_rmsnorm(hidden_states, layer["input_layernorm"][0], self.eps)  # type: ignore[index]

            q = layer["q_proj"](hidden_states).view(bsz, seq_len, self.num_q_heads, self.head_dim)  # type: ignore[index]
            k = layer["k_proj"](hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)  # type: ignore[index]
            v = layer["v_proj"](hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)  # type: ignore[index]

            q, k = eager_gqa_rope_qk(q, k, cos, sin)
            attn_out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=(seq_len > 1),
                enable_gqa=True,
            )
            attn_out = layer["o_proj"](attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size))  # type: ignore[index]
            hidden_states = residual + attn_out

            residual = hidden_states
            hidden_states = hf_official_rmsnorm(hidden_states, layer["post_attention_layernorm"][0], self.eps)  # type: ignore[index]
            mlp_out = layer["down_proj"](eager_swiglu(layer["gate_up_proj"](hidden_states)))  # type: ignore[index]
            hidden_states = residual + mlp_out

        return hf_official_rmsnorm(hidden_states, self.norm_weight, self.eps)
