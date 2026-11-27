"""Triton 高性能算子实现集。

包含：
- `triton_fused_rmsnorm_residual` / `triton_rmsnorm`：单 Pass 融合残差加法与 RMSNorm；
- `triton_fused_swiglu`：2D 分块单 Pass 融合 SwiGLU；
- `triton_fused_gqa_rope_qk` / `triton_rope_single`：双模式单发射 GQA RoPE（含双向广播越界防护）；
- `triton_fused_linear_cross_entropy`：分块 GEMM + Triton Online-Softmax 原位前反向交叉熵
  （FP16 `2^-4` 预缩放防下溢、零 `.item()` 同步、兼容 `torch.compile(fullgraph=True)`）。
"""

import torch
import triton
import triton.language as tl

# ==============================================================================
# 1. Fused Add + RMSNorm & Standalone RMSNorm (Triton 版)
# ==============================================================================


@triton.jit
def _fused_rmsnorm_residual_kernel(
    X_ptr,
    Residual_ptr,
    W_ptr,
    Out_ptr,
    Updated_Res_ptr,
    stride_x_m,
    stride_res_m,
    stride_out_m,
    stride_up_res_m,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row_idx * stride_x_m + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    res = tl.load(
        Residual_ptr + row_idx * stride_res_m + cols, mask=mask, other=0.0
    ).to(tl.float32)

    row_up_ptr = Updated_Res_ptr + row_idx * stride_up_res_m + cols
    up_rounded = (x + res).to(row_up_ptr.dtype.element_ty)
    tl.store(row_up_ptr, up_rounded, mask=mask)

    accum = up_rounded.to(tl.float32)
    mean_sq = tl.sum(accum * accum, axis=0) / N
    rsqrt = tl.rsqrt(mean_sq + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (accum * rsqrt) * w
    row_out_ptr = Out_ptr + row_idx * stride_out_m + cols
    tl.store(row_out_ptr, out.to(row_out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_solo_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    stride_x_m,
    stride_out_m,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row_idx * stride_x_m + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    mean_sq = tl.sum(x * x, axis=0) / N
    rsqrt = tl.rsqrt(mean_sq + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (x * rsqrt) * w
    row_out_ptr = Out_ptr + row_idx * stride_out_m + cols
    tl.store(row_out_ptr, out.to(row_out_ptr.dtype.element_ty), mask=mask)


def triton_fused_rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda and residual.is_cuda and weight.is_cuda
    assert x.shape == residual.shape and x.shape[-1] == weight.shape[0]

    orig_shape = x.shape
    N = orig_shape[-1]
    M = x.numel() // N

    x_2d = x.view(M, N) if x.is_contiguous() else x.contiguous().view(M, N)
    res_2d = (
        residual.view(M, N)
        if residual.is_contiguous()
        else residual.contiguous().view(M, N)
    )
    w_1d = weight if weight.is_contiguous() else weight.contiguous()

    out_2d = torch.empty_like(x_2d)
    up_res_2d = torch.empty_like(res_2d)

    if M == 0:
        return out_2d.view(orig_shape), up_res_2d.view(orig_shape)

    block_size = triton.next_power_of_2(N)
    warps = 8 if N >= 2048 else 4

    _fused_rmsnorm_residual_kernel[(M,)](
        x_2d,
        res_2d,
        w_1d,
        out_2d,
        up_res_2d,
        x_2d.stride(0),
        res_2d.stride(0),
        out_2d.stride(0),
        up_res_2d.stride(0),
        N=N,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=warps,
    )
    return out_2d.view(orig_shape), up_res_2d.view(orig_shape)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and x.shape[-1] == weight.shape[0]
    orig_shape = x.shape
    N = orig_shape[-1]
    M = x.numel() // N

    x_2d = x.view(M, N) if x.is_contiguous() else x.contiguous().view(M, N)
    w_1d = weight if weight.is_contiguous() else weight.contiguous()
    out_2d = torch.empty_like(x_2d)
    if M == 0:
        return out_2d.view(orig_shape)

    block_size = triton.next_power_of_2(N)
    _rmsnorm_solo_kernel[(M,)](
        x_2d,
        w_1d,
        out_2d,
        x_2d.stride(0),
        out_2d.stride(0),
        N=N,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=8 if N >= 2048 else 4,
    )
    return out_2d.view(orig_shape)


# ==============================================================================
# 2. Fused SwiGLU (Triton 版)
# ==============================================================================


@triton.jit
def _fused_swiglu_kernel(
    GateUp_ptr,
    Out_ptr,
    stride_gu_m,
    stride_out_m,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    cols = col_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N

    row_base = GateUp_ptr + row_idx * stride_gu_m
    gate = tl.load(row_base + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(row_base + N + cols, mask=mask, other=0.0).to(tl.float32)

    out = (gate * tl.sigmoid(gate)) * up
    out_ptr = Out_ptr + row_idx * stride_out_m + cols
    tl.store(out_ptr, out.to(out_ptr.dtype.element_ty), mask=mask)


def triton_fused_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    assert gate_up.is_cuda and gate_up.shape[-1] % 2 == 0
    orig_prefix = gate_up.shape[:-1]
    two_n = gate_up.shape[-1]
    N = two_n // 2
    M = gate_up.numel() // two_n

    gu_2d = (
        gate_up.view(M, two_n)
        if gate_up.is_contiguous()
        else gate_up.contiguous().view(M, two_n)
    )
    out_2d = torch.empty((M, N), dtype=gate_up.dtype, device=gate_up.device)
    if M == 0:
        return out_2d.view(*orig_prefix, N)

    block_n = 1024
    warps = 4 if M <= 32 else 8
    grid = (M, triton.cdiv(N, block_n))
    _fused_swiglu_kernel[grid](
        gu_2d,
        out_2d,
        gu_2d.stride(0),
        out_2d.stride(0),
        N=N,
        BLOCK_N=block_n,
        num_warps=warps,
    )
    return out_2d.view(*orig_prefix, N)


# ==============================================================================
# 3. Single-Launch Multi-Head Heterogeneous GQA Q+K RoPE & Single-Branch RoPE
# ==============================================================================


def _prepare_rope_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    B: int,
    S: int,
    D: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    严格校验并规范化 `cos` / `sin` 布局，正确计算 Batch 维与 Seq 维广播步长（修复 `S_cos=1` 时的显存越界读漏洞）。
    合法形状：`[B_or_1, S_or_1, 1, D]`、`[B_or_1, S_or_1, D]` 或 `[S_or_1, D]`。
    """
    assert cos.is_cuda and sin.is_cuda, "cos/sin must be on CUDA"
    assert cos.shape == sin.shape, f"cos/sin shape mismatch: {cos.shape} vs {sin.shape}"
    assert cos.dtype == dtype and sin.dtype == dtype, (
        f"cos/sin dtype must match {dtype}"
    )

    if cos.ndim == 4:
        assert cos.shape[2] == 1, (
            f"4D cos/sin head dim (axis 2) must be 1 for broadcast, got {cos.shape}"
        )
        cos_3d = cos.squeeze(2)
        sin_3d = sin.squeeze(2)
    elif cos.ndim == 3:
        cos_3d, sin_3d = cos, sin
    elif cos.ndim == 2:
        cos_3d = cos.unsqueeze(0)
        sin_3d = sin.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported cos/sin ndim={cos.ndim}, shape={cos.shape}")

    Bc, Sc, Dc = cos_3d.shape
    assert Dc == D, f"cos/sin last dim {Dc} != head_dim {D}"
    assert Bc in (1, B), f"cos/sin batch dim {Bc} incompatible with B={B}"
    assert Sc in (1, S), f"cos/sin seq dim {Sc} incompatible with S={S}"

    cos_c = cos_3d if cos_3d.is_contiguous() else cos_3d.contiguous()
    sin_c = sin_3d if sin_3d.is_contiguous() else sin_3d.contiguous()

    stride_cb = 0 if Bc == 1 else cos_c.stride(0)
    stride_cs = 0 if Sc == 1 else cos_c.stride(1)
    return cos_c, sin_c, stride_cb, stride_cs


@triton.jit
def _rope_single_kernel(
    X_ptr,
    Cos_ptr,
    Sin_ptr,
    Out_ptr,
    stride_xb,
    stride_xs,
    stride_xh,
    stride_cb,
    stride_cs,
    S,
    HALF_D: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    h_idx = tl.program_id(1)

    b_idx = tok_idx // S
    s_idx = tok_idx % S

    half_offs = tl.arange(0, HALF_D)
    cos_base = Cos_ptr + b_idx * stride_cb + s_idx * stride_cs
    sin_base = Sin_ptr + b_idx * stride_cb + s_idx * stride_cs

    cos1 = tl.load(cos_base + half_offs).to(tl.float32)
    cos2 = tl.load(cos_base + HALF_D + half_offs).to(tl.float32)
    sin1 = tl.load(sin_base + half_offs).to(tl.float32)
    sin2 = tl.load(sin_base + HALF_D + half_offs).to(tl.float32)

    x_in = X_ptr + b_idx * stride_xb + s_idx * stride_xs + h_idx * stride_xh
    x_out = Out_ptr + b_idx * stride_xb + s_idx * stride_xs + h_idx * stride_xh
    x1 = tl.load(x_in + half_offs).to(tl.float32)
    x2 = tl.load(x_in + HALF_D + half_offs).to(tl.float32)
    o1 = x1 * cos1 - x2 * sin1
    o2 = x2 * cos2 + x1 * sin2
    tl.store(x_out + half_offs, o1.to(Out_ptr.dtype.element_ty))
    tl.store(x_out + HALF_D + half_offs, o2.to(Out_ptr.dtype.element_ty))


@triton.jit
def _fused_gqa_rope_decode_kernel(
    Q_ptr,
    K_ptr,
    Cos_ptr,
    Sin_ptr,
    Q_out_ptr,
    K_out_ptr,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_cb,
    stride_cs,
    S,
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    HALF_D: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    head_combined = tl.program_id(1)

    b_idx = tok_idx // S
    s_idx = tok_idx % S

    half_offs = tl.arange(0, HALF_D)
    cos_base = Cos_ptr + b_idx * stride_cb + s_idx * stride_cs
    sin_base = Sin_ptr + b_idx * stride_cb + s_idx * stride_cs

    cos1 = tl.load(cos_base + half_offs).to(tl.float32)
    cos2 = tl.load(cos_base + HALF_D + half_offs).to(tl.float32)
    sin1 = tl.load(sin_base + half_offs).to(tl.float32)
    sin2 = tl.load(sin_base + HALF_D + half_offs).to(tl.float32)

    if head_combined < Hq:
        q_in = Q_ptr + b_idx * stride_qb + s_idx * stride_qs + head_combined * stride_qh
        q_out = (
            Q_out_ptr
            + b_idx * stride_qb
            + s_idx * stride_qs
            + head_combined * stride_qh
        )
        x1 = tl.load(q_in + half_offs).to(tl.float32)
        x2 = tl.load(q_in + HALF_D + half_offs).to(tl.float32)
        o1 = x1 * cos1 - x2 * sin1
        o2 = x2 * cos2 + x1 * sin2
        tl.store(q_out + half_offs, o1.to(Q_out_ptr.dtype.element_ty))
        tl.store(q_out + HALF_D + half_offs, o2.to(Q_out_ptr.dtype.element_ty))
    else:
        kv_h = head_combined - Hq
        k_in = K_ptr + b_idx * stride_kb + s_idx * stride_ks + kv_h * stride_kh
        k_out = K_out_ptr + b_idx * stride_kb + s_idx * stride_ks + kv_h * stride_kh
        x1 = tl.load(k_in + half_offs).to(tl.float32)
        x2 = tl.load(k_in + HALF_D + half_offs).to(tl.float32)
        o1 = x1 * cos1 - x2 * sin1
        o2 = x2 * cos2 + x1 * sin2
        tl.store(k_out + half_offs, o1.to(K_out_ptr.dtype.element_ty))
        tl.store(k_out + HALF_D + half_offs, o2.to(K_out_ptr.dtype.element_ty))


@triton.jit
def _fused_gqa_rope_prefill_kernel(
    Q_ptr,
    K_ptr,
    Cos_ptr,
    Sin_ptr,
    Q_out_ptr,
    K_out_ptr,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_cb,
    stride_cs,
    S,
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_HKV: tl.constexpr,
    HALF_D: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    b_idx = tok_idx // S
    s_idx = tok_idx % S

    d_offs = tl.arange(0, HALF_D)[None, :]
    cos_base = Cos_ptr + b_idx * stride_cb + s_idx * stride_cs
    sin_base = Sin_ptr + b_idx * stride_cb + s_idx * stride_cs

    cos1 = tl.load(cos_base + d_offs).to(tl.float32)
    cos2 = tl.load(cos_base + HALF_D + d_offs).to(tl.float32)
    sin1 = tl.load(sin_base + d_offs).to(tl.float32)
    sin2 = tl.load(sin_base + HALF_D + d_offs).to(tl.float32)

    hq_offs = tl.arange(0, BLOCK_HQ)[:, None]
    hq_mask = hq_offs < Hq
    q_tok_in = Q_ptr + b_idx * stride_qb + s_idx * stride_qs + hq_offs * stride_qh
    q_tok_out = Q_out_ptr + b_idx * stride_qb + s_idx * stride_qs + hq_offs * stride_qh

    qx1 = tl.load(q_tok_in + d_offs, mask=hq_mask, other=0.0).to(tl.float32)
    qx2 = tl.load(q_tok_in + HALF_D + d_offs, mask=hq_mask, other=0.0).to(tl.float32)
    qo1 = qx1 * cos1 - qx2 * sin1
    qo2 = qx2 * cos2 + qx1 * sin2
    tl.store(q_tok_out + d_offs, qo1.to(Q_out_ptr.dtype.element_ty), mask=hq_mask)
    tl.store(
        q_tok_out + HALF_D + d_offs, qo2.to(Q_out_ptr.dtype.element_ty), mask=hq_mask
    )

    hkv_offs = tl.arange(0, BLOCK_HKV)[:, None]
    hkv_mask = hkv_offs < Hkv
    k_tok_in = K_ptr + b_idx * stride_kb + s_idx * stride_ks + hkv_offs * stride_kh
    k_tok_out = K_out_ptr + b_idx * stride_kb + s_idx * stride_ks + hkv_offs * stride_kh

    kx1 = tl.load(k_tok_in + d_offs, mask=hkv_mask, other=0.0).to(tl.float32)
    kx2 = tl.load(k_tok_in + HALF_D + d_offs, mask=hkv_mask, other=0.0).to(tl.float32)
    ko1 = kx1 * cos1 - kx2 * sin1
    ko2 = kx2 * cos2 + kx1 * sin2
    tl.store(k_tok_out + d_offs, ko1.to(K_out_ptr.dtype.element_ty), mask=hkv_mask)
    tl.store(
        k_tok_out + HALF_D + d_offs, ko2.to(K_out_ptr.dtype.element_ty), mask=hkv_mask
    )


def triton_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    assert x.ndim == 4 and x.is_cuda
    B, S, H, D = x.shape
    assert D > 0 and (D & (D - 1)) == 0, "Head dim 必须为 2 的幂次"

    x_c = x if x.is_contiguous() else x.contiguous()
    cos_c, sin_c, stride_cb, stride_cs = _prepare_rope_cos_sin(
        cos, sin, B, S, D, x.dtype
    )

    out = torch.empty_like(x_c)
    if B * S * H == 0:
        return out

    _rope_single_kernel[(B * S, H)](
        x_c,
        cos_c,
        sin_c,
        out,
        x_c.stride(0),
        x_c.stride(1),
        x_c.stride(2),
        stride_cb,
        stride_cs,
        S,
        HALF_D=D // 2,
        num_warps=2 if D <= 128 else 4,
    )
    return out


def triton_fused_gqa_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.ndim == 4 and k.ndim == 4 and q.is_cuda and k.is_cuda
    B, S, Hq, D = q.shape
    Bk, Sk, Hkv, Dk = k.shape
    assert (Bk, Sk, Dk) == (B, S, D) and q.dtype == k.dtype
    assert D > 0 and (D & (D - 1)) == 0, "Head dim 必须为 2 的幂次 (如 64, 128)"

    q_c = q if q.is_contiguous() else q.contiguous()
    k_c = k if k.is_contiguous() else k.contiguous()
    cos_c, sin_c, stride_cb, stride_cs = _prepare_rope_cos_sin(
        cos, sin, B, S, D, q.dtype
    )

    q_out = torch.empty_like(q_c)
    k_out = torch.empty_like(k_c)

    num_tokens = B * S
    if num_tokens == 0:
        return q_out, k_out

    if num_tokens <= 32:
        _fused_gqa_rope_decode_kernel[(num_tokens, Hq + Hkv)](
            q_c,
            k_c,
            cos_c,
            sin_c,
            q_out,
            k_out,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            k_c.stride(0),
            k_c.stride(1),
            k_c.stride(2),
            stride_cb,
            stride_cs,
            S,
            Hq=Hq,
            Hkv=Hkv,
            HALF_D=D // 2,
            num_warps=2 if D <= 128 else 4,
        )
    else:
        block_hq = triton.next_power_of_2(Hq)
        block_hkv = triton.next_power_of_2(Hkv)
        _fused_gqa_rope_prefill_kernel[(num_tokens,)](
            q_c,
            k_c,
            cos_c,
            sin_c,
            q_out,
            k_out,
            q_c.stride(0),
            q_c.stride(1),
            q_c.stride(2),
            k_c.stride(0),
            k_c.stride(1),
            k_c.stride(2),
            stride_cb,
            stride_cs,
            S,
            Hq=Hq,
            Hkv=Hkv,
            BLOCK_HQ=block_hq,
            BLOCK_HKV=block_hkv,
            HALF_D=D // 2,
            num_warps=4 if block_hq * (D // 2) <= 1024 else 8,
        )
    return q_out, k_out


# ==============================================================================
# 4. Fused Linear + Cross-Entropy (Chunked GEMM + Triton Online-Softmax In-Place Grad)
# ==============================================================================


@triton.jit
def _online_ce_fwd_bwd_inplace_kernel(
    Logits_ptr,
    Targets_ptr,
    Losses_ptr,
    stride_m,
    V,
    ignore_index,
    PRESCALE: tl.constexpr,
    HAS_GRAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = Logits_ptr + row * stride_m
    target = tl.load(Targets_ptr + row)

    if target == ignore_index:
        tl.store(Losses_ptr + row, 0.0)
        if HAS_GRAD:
            for v_start in range(0, V, BLOCK_V):
                offs = v_start + tl.arange(0, BLOCK_V)
                tl.store(
                    row_ptr + offs,
                    tl.zeros([BLOCK_V], dtype=Logits_ptr.dtype.element_ty),
                    mask=offs < V,
                )
        return

    # Pass 1: Online LogSumExp
    m_i = -float("inf")
    d_i = 0.0
    for v_start in range(0, V, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        z = tl.load(row_ptr + offs, mask=offs < V, other=-float("inf")).to(tl.float32)
        m_tile = tl.max(z, axis=0)
        m_new = tl.maximum(m_i, m_tile)
        d_i = d_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(z - m_new), axis=0)
        m_i = m_new

    lse = m_i + tl.log(d_i)
    z_target = tl.load(row_ptr + target).to(tl.float32)
    tl.store(Losses_ptr + row, lse - z_target)

    # Pass 2: In-place Softmax Gradient Overwrite (不提前除以 valid_cnt，避免 FP16 非目标词梯度下溢为 0！)
    if HAS_GRAD:
        for v_start in range(0, V, BLOCK_V):
            offs = v_start + tl.arange(0, BLOCK_V)
            mask = offs < V
            z = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            prob = tl.exp(z - lse)
            grad = tl.where(offs == target, prob - 1.0, prob) * PRESCALE
            tl.store(row_ptr + offs, grad.to(Logits_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _scale_grad_out_of_place_kernel(
    In_ptr,
    Out_ptr,
    GradOut_ptr,
    ValidCnt_ptr,
    n_elements,
    INV_PRESCALE: tl.constexpr,
    IS_MEAN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    在反向传播尾端寄存器内以 FP32 完成 `(raw_grad * inv_prescale / valid_cnt) * grad_output` 并输出至新张量：
    - 零临时 FP32 显存分配；
    - 零 `.item()` Host-Device 同步阻塞；
    - 不原地修改 `ctx.saved_tensors`，安全支持 `retain_graph=True` 多次反向传播。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    g_up = tl.load(GradOut_ptr).to(tl.float32)
    if IS_MEAN:
        vcnt = tl.load(ValidCnt_ptr).to(tl.float32)
        scale = tl.where(vcnt > 0.0, (g_up * INV_PRESCALE) / vcnt, 0.0)
    else:
        scale = g_up * INV_PRESCALE

    val = tl.load(In_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = val * scale
    tl.store(Out_ptr + offs, out.to(Out_ptr.dtype.element_ty), mask=mask)


def _apply_final_grad_scale(
    raw_grad: torch.Tensor,
    grad_output: torch.Tensor,
    valid_cnt_t: torch.Tensor,
    inv_prescale: float,
    is_mean: bool,
) -> torch.Tensor:
    out = torch.empty_like(raw_grad)
    numel = raw_grad.numel()
    if numel == 0:
        return out
    g_contig = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    block = 1024
    grid = (triton.cdiv(numel, block),)
    _scale_grad_out_of_place_kernel[grid](
        raw_grad,
        out,
        g_contig,
        valid_cnt_t,
        numel,
        INV_PRESCALE=inv_prescale,
        IS_MEAN=is_mean,
        BLOCK_SIZE=block,
        num_warps=4,
    )
    return out


class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
        chunk_size: int = 256,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert hidden.is_cuda and weight.is_cuda and targets.is_cuda
        assert reduction in ("mean", "sum")

        orig_h_shape = hidden.shape
        H = orig_h_shape[-1]
        M = hidden.numel() // H
        V = weight.shape[0]
        assert weight.shape[1] == H and targets.numel() == M

        h_2d = (
            hidden.view(M, H)
            if hidden.is_contiguous()
            else hidden.contiguous().view(M, H)
        )
        w_2d = weight if weight.is_contiguous() else weight.contiguous()
        # 使用 reshape + contiguous 支持 Causal LM 标准偏移切片 `labels[:, 1:]` 及任意步长视图
        t_1d = targets.reshape(M).to(torch.int64).contiguous()

        need_h_grad = hidden.requires_grad
        need_w_grad = weight.requires_grad
        has_grad = bool(need_h_grad or need_w_grad)
        is_mean = reduction == "mean"

        # FP16 采用 2^-4 (0.0625) 纯指数移位预缩放：既保留非目标词概率 (~4.9e-7 >> 5.96e-8) 零下溢，又防止极端同目标累加溢出 (>65504)
        prescale = 0.0625 if h_2d.dtype == torch.float16 else 1.0
        inv_prescale = 16.0 if h_2d.dtype == torch.float16 else 1.0

        valid_cnt_t = (t_1d != ignore_index).sum(dtype=torch.int64)

        losses = torch.empty((M,), dtype=torch.float32, device=hidden.device)
        grad_h = torch.empty_like(h_2d) if need_h_grad else None
        grad_w = torch.empty_like(w_2d) if need_w_grad else None

        if M > 0:
            actual_chunk = min(M, chunk_size)
            logits_buf = torch.empty(
                (actual_chunk, V), dtype=h_2d.dtype, device=hidden.device
            )
            block_v = (
                8192
                if V >= 8192
                else (2048 if V >= 2048 else triton.next_power_of_2(V))
            )
            warps = 16 if block_v >= 8192 else 8

            for start in range(0, M, actual_chunk):
                end = min(start + actual_chunk, M)
                cur_m = end - start
                h_slice = h_2d[start:end]
                buf_slice = logits_buf[:cur_m]

                torch.matmul(h_slice, w_2d.t(), out=buf_slice)

                _online_ce_fwd_bwd_inplace_kernel[(cur_m,)](
                    buf_slice,
                    t_1d[start:end],
                    losses[start:end],
                    buf_slice.stride(0),
                    V,
                    ignore_index,
                    PRESCALE=prescale,
                    HAS_GRAD=has_grad,
                    BLOCK_V=block_v,
                    num_warps=warps,
                )

                if grad_h is not None:
                    torch.matmul(buf_slice, w_2d, out=grad_h[start:end])
                if grad_w is not None:
                    if start == 0:
                        torch.matmul(buf_slice.t(), h_slice, out=grad_w)
                    else:
                        grad_w.addmm_(buf_slice.t(), h_slice)

            del logits_buf
        elif grad_w is not None:
            grad_w.zero_()

        ctx.h_shape = orig_h_shape
        ctx.need_h_grad = need_h_grad
        ctx.need_w_grad = need_w_grad
        ctx.inv_prescale = inv_prescale
        ctx.is_mean = is_mean

        saved = [valid_cnt_t]
        if grad_h is not None:
            saved.append(grad_h)
        if grad_w is not None:
            saved.append(grad_w)
        ctx.save_for_backward(*saved)

        total_loss = (
            losses.sum()
            if M > 0
            else torch.zeros((), dtype=torch.float32, device=hidden.device)
        )
        if is_mean:
            denom = valid_cnt_t.clamp(min=1).to(torch.float32)
            out_loss = torch.where(
                valid_cnt_t > 0, total_loss / denom, torch.zeros_like(total_loss)
            )
        else:
            out_loss = total_loss
        return out_loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        valid_cnt_t = saved[0]
        idx = 1
        gh = None
        gw = None

        if ctx.need_h_grad:
            raw_gh = saved[idx]
            idx += 1
            scaled_gh = _apply_final_grad_scale(
                raw_gh, grad_output, valid_cnt_t, ctx.inv_prescale, ctx.is_mean
            )
            gh = scaled_gh.view(ctx.h_shape)

        if ctx.need_w_grad:
            raw_gw = saved[idx]
            gw = _apply_final_grad_scale(
                raw_gw, grad_output, valid_cnt_t, ctx.inv_prescale, ctx.is_mean
            )

        return gh, gw, None, None, None, None


def triton_fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    chunk_size: int = 256,
    reduction: str = "mean",
) -> torch.Tensor:
    return FusedLinearCrossEntropyFunction.apply(  # type: ignore[return-value]
        hidden, weight, targets, ignore_index, chunk_size, reduction
    )
