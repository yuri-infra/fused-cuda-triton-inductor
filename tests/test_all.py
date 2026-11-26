"""
项目三严苛验证与红方回归测试套件（共 8 项，全覆盖内存边界、动态传递闭包防死环、FP16 防下溢、C++ 原生分发与真实 HF 权重）
"""

import glob
import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._inductor.config as inductor_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_infra_fused_ops import (
    cuda_fused_rmsnorm_residual,
    cuda_rmsnorm,
    inspect_cuda_ext_ptx_stats,
    triton_fused_rmsnorm_residual,
    triton_rmsnorm,
    triton_rope_single,
    triton_fused_gqa_rope_qk,
    triton_fused_linear_cross_entropy,
    fused_rmsnorm_residual_op,
    fused_rmsnorm_op,
    rope_single_op,
    fused_gqa_rope_op,
    fused_swiglu_op,
    reset_runtime_counters,
    get_runtime_counters,
    eager_rmsnorm_fp32,
    eager_add_rmsnorm_fp32,
    eager_rope_single,
    eager_gqa_rope_qk,
    inductor_fusion_pass,
    disable_inductor_fusion_pass,
    reset_compile_counters,
    get_compile_counters,
    Llama3DecoderStack,
    OfficialHFQwen2StyleModel,
)


class TestProject3RigorousSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available(), "需要可用的 NVIDIA CUDA GPU"
        torch.manual_seed(42)
        cls.device = torch.device("cuda:0")

    def tearDown(self):
        disable_inductor_fusion_pass()

    def test_01_native_cuda_cpp_fp16_bf16_fp32_and_ptx_audit(self):
        """验证原生 CUDA C++ (.cu) 与 Triton 在 FP16/BF16/FP32、对齐/奇数维度 (3075) 下的精度及 PTX 向量化指令实证"""
        shapes = [
            (1, 2048),      # Decode M=1 寄存器单 Pass 路径
            (4, 3075),      # 非向量长度整数倍维度，覆盖标量尾部回退分支
            (16, 4096),     # 标准 LLaMA-3 维度
            (2, 32, 4096),  # 3D 张量
        ]
        for dtype, tol in [(torch.float16, 2e-3), (torch.bfloat16, 1.6e-2), (torch.float32, 1e-5)]:
            for shape in shapes:
                n = shape[-1]
                x = torch.randn(*shape, dtype=dtype, device=self.device)
                res = torch.randn(*shape, dtype=dtype, device=self.device)
                w = torch.randn(n, dtype=dtype, device=self.device)

                ref_o, ref_r = eager_add_rmsnorm_fp32(x, res, w, 1e-6)
                cu_o, cu_r = cuda_fused_rmsnorm_residual(x, res, w, 1e-6)
                tr_o, tr_r = triton_fused_rmsnorm_residual(x, res, w, 1e-6)

                self.assertTrue(torch.allclose(ref_o, cu_o, atol=tol, rtol=tol), f".cu add+norm mismatch on {dtype} {shape}")
                self.assertTrue(torch.allclose(ref_r, cu_r, atol=tol, rtol=tol))
                self.assertTrue(torch.allclose(ref_o, tr_o, atol=tol, rtol=tol), f"triton add+norm mismatch on {dtype} {shape}")
                self.assertTrue(torch.allclose(ref_r, tr_r, atol=tol, rtol=tol))

                ref_solo = eager_rmsnorm_fp32(x, w, 1e-6)
                cu_solo = cuda_rmsnorm(x, w, 1e-6)
                tr_solo = triton_rmsnorm(x, w, 1e-6)
                self.assertTrue(torch.allclose(ref_solo, cu_solo, atol=tol, rtol=tol), f".cu solo mismatch on {dtype} {shape}")
                self.assertTrue(torch.allclose(ref_solo, tr_solo, atol=tol, rtol=tol))

        ptx_stats = inspect_cuda_ext_ptx_stats()
        self.assertGreater(ptx_stats["ld_v4_u32"], 0, f"PTX 未检测到 128-bit 向量化加载指令: {ptx_stats}")
        self.assertGreater(ptx_stats["st_v4_u32"], 0, f"PTX 未检测到 128-bit 向量化存储指令: {ptx_stats}")
        self.assertGreater(ptx_stats["shfl_bfly"], 0, f"PTX 未检测到 shfl.sync.bfly 指令: {ptx_stats}")

    def test_02_rope_broadcast_oob_guard_and_fp64_accuracy(self):
        """【P0-1 红方回归】覆盖 cos/sin 在 Batch 与 Seq 维全组合广播（含 [1, 1, 1, D]），严防 S_cos=1 越界读"""
        for B, S in [(2, 16), (2, 48)]:  # 同时覆盖 Decode 核 (B*S=32) 与 Prefill 2D 核 (B*S=96)
            for dtype in (torch.float16, torch.bfloat16):
                q = torch.randn(B, S, 16, 64, dtype=dtype, device=self.device)
                k = torch.randn(B, S, 4, 64, dtype=dtype, device=self.device)

                for cos_shape in [
                    (1, 1, 1, 64),   # Batch 与 Seq 同时广播（原 P0-1 越界复现形状！）
                    (B, 1, 1, 64),   # 仅 Seq 广播
                    (1, S, 1, 64),   # 仅 Batch 广播
                    (B, S, 1, 64),   # 无广播
                    (S, 64),         # 2D 位置表输入
                ]:
                    # 特意在 cos/sin 后方紧邻分配大幅垃圾噪声显存，若发生越界读会立刻引爆误差
                    buf_angles = torch.full((16 * B * S * 64,), 9999.0, dtype=torch.float32, device=self.device)
                    numel_c = 1
                    for dim_v in cos_shape:
                        numel_c *= dim_v
                    valid_angles = torch.randn(numel_c, dtype=torch.float32, device=self.device)
                    buf_angles[:numel_c].copy_(valid_angles)
                    sub_view = buf_angles[:numel_c].view(*cos_shape)
                    cos = sub_view.cos().to(dtype)
                    sin = sub_view.sin().to(dtype)

                    cos_ref = cos.view(1, S, 1, 64) if cos.ndim == 2 else cos
                    sin_ref = sin.view(1, S, 1, 64) if sin.ndim == 2 else sin

                    gt_q, gt_k = eager_gqa_rope_qk(q.double(), k.double(), cos_ref.double(), sin_ref.double())
                    eager_q, eager_k = eager_gqa_rope_qk(q, k, cos_ref, sin_ref)
                    tri_q, tri_k = triton_fused_gqa_rope_qk(q, k, cos, sin)
                    solo_q = triton_rope_single(q, cos, sin)

                    eager_err_q = (eager_q.double() - gt_q).abs().max().item()
                    tri_err_q = (tri_q.double() - gt_q).abs().max().item()
                    solo_err_q = (solo_q.double() - gt_q).abs().max().item()
                    eager_err_k = (eager_k.double() - gt_k).abs().max().item()
                    tri_err_k = (tri_k.double() - gt_k).abs().max().item()

                    self.assertLessEqual(tri_err_q, eager_err_q + 1e-5, f"OOB/Accuracy fail on {cos_shape}")
                    self.assertLessEqual(solo_err_q, eager_err_q + 1e-5, f"Solo OOB/Accuracy fail on {cos_shape}")
                    self.assertLessEqual(tri_err_k, eager_err_k + 1e-5)

        # 非法广播形状必须显式抛出异常拒绝执行
        q_bad = torch.randn(2, 16, 8, 64, dtype=torch.float16, device=self.device)
        cos_bad = torch.randn(1, 3, 1, 64, dtype=torch.float16, device=self.device)
        with self.assertRaises(AssertionError):
            triton_rope_single(q_bad, cos_bad, cos_bad)

    def test_03_fused_linear_ce_shifted_retain_fp16_underflow_and_vram(self):
        """【P0-3 & P0-4 红方回归】验证非连续 labels[:,1:]、retain_graph=True、FP16 防梯度下溢与修正基准后的真实显存削减"""
        # (a) Causal LM 非连续偏移切片 labels[:, 1:] + retain_graph=True 二次反传 + fullgraph=True
        B_s, S_s, H_s, V_s = 2, 33, 256, 4096
        h_seq = torch.randn(B_s, S_s, H_s, dtype=torch.bfloat16, device=self.device)
        labels_full = torch.randint(0, V_s, (B_s, S_s), device=self.device)
        labels_full[:, -1] = -100

        shift_h = h_seq[:, :-1, :].detach().requires_grad_(True)      # 非连续 3D 切片!
        shift_labels = labels_full[:, 1:]                             # 非连续 2D 切片!
        w_sm = (torch.randn(V_s, H_s, dtype=torch.bfloat16, device=self.device) * 0.02).detach().requires_grad_(True)
        self.assertFalse(shift_labels.is_contiguous())

        loss_1 = triton_fused_linear_cross_entropy(shift_h, w_sm, shift_labels, ignore_index=-100, chunk_size=32)
        loss_1.backward(retain_graph=True)
        g_once = w_sm.grad.detach().clone()  # type: ignore[union-attr]
        loss_1.backward()  # 第二次反传：若原地修改 saved_tensors 会立刻崩溃！
        g_twice = w_sm.grad.detach().clone()  # type: ignore[union-attr]
        self.assertTrue(torch.allclose(g_twice, g_once * 2.0, atol=1e-2, rtol=1e-2))

        @torch.compile(backend="inductor", fullgraph=True)
        def fullgraph_ce_fn(x, w, y):
            return triton_fused_linear_cross_entropy(x, w, y, ignore_index=-100, chunk_size=32)

        shift_h2 = h_seq[:, :-1, :].detach().requires_grad_(True)
        w_sm2 = w_sm.detach().clone().requires_grad_(True)
        loss_fg = fullgraph_ce_fn(shift_h2, w_sm2, shift_labels)
        loss_fg.backward()
        self.assertAlmostEqual(loss_1.item(), loss_fg.item(), places=2)

        # (b) FP16 128,256 大词表非目标词权重梯度防下溢测试（对比 FP32 黄金梯度转 FP16）
        M_f16, H_f16, V_f16 = 4096, 512, 128256
        h_f16 = torch.randn(M_f16, H_f16, dtype=torch.float16, device=self.device, requires_grad=True)
        w_f16 = (torch.randn(V_f16, H_f16, dtype=torch.float16, device=self.device) * 0.02).detach().requires_grad_(True)
        t_f16 = torch.randint(0, V_f16, (M_f16,), device=self.device)

        loss_f16 = triton_fused_linear_cross_entropy(h_f16, w_f16, t_f16, chunk_size=512)
        loss_f16.backward()
        fus_gw_f16 = w_f16.grad.detach()  # type: ignore[union-attr]
        fus_zero_frac = (fus_gw_f16 == 0).float().mean().item()

        # 旧版提前乘 1/4096 会导致 96.85% 元素下溢为 0；延迟缩放后零元素占比应贴近 IEEE-754 FP16 理论极限 (~11%，严格 < 25%)！
        self.assertLess(
            fus_zero_frac,
            0.25,
            f"FP16 大词表 w.grad 发生异常下溢归零: zero_frac={fus_zero_frac:.4f}",
        )
        del h_f16, w_f16, fus_gw_f16, loss_f16
        torch.cuda.empty_cache()

        # (c) BF16 128,256 大词表精度与【修正清零梯度基准后】真实显存增量对比
        M, H, V = 4096, 1024, 128256
        dtype = torch.bfloat16
        h_base = torch.randn(M, H, dtype=dtype, device=self.device)
        w_base = torch.randn(V, H, dtype=dtype, device=self.device) * 0.02
        targets = torch.randint(0, V, (M,), device=self.device)
        targets[0:32] = -100

        with torch.no_grad():
            fp32_gt_loss = F.cross_entropy(
                F.linear(h_base.float(), w_base.float()), targets, ignore_index=-100
            )

        @torch.compile(backend="inductor")
        def compiled_baseline_ce(h, w, t):
            return F.cross_entropy(F.linear(h, w), t, ignore_index=-100)

        _h_w = h_base.clone().requires_grad_(True)
        _w_w = w_base.clone().requires_grad_(True)
        (compiled_baseline_ce(_h_w, _w_w, targets) * 0.37).backward()
        del _h_w, _w_w
        torch.cuda.empty_cache()

        h_ref = h_base.clone().requires_grad_(True)
        w_ref = w_base.clone().requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_b0 = torch.cuda.memory_allocated(self.device)  # 此时 h_ref.grad / w_ref.grad 均为 None！
        loss_ref = compiled_baseline_ce(h_ref, w_ref, targets)
        (loss_ref * 0.37).backward()
        torch.cuda.synchronize()
        compile_peak_delta = torch.cuda.max_memory_allocated(self.device) - mem_b0
        grad_leaf_bytes = h_ref.grad.nbytes + w_ref.grad.nbytes  # type: ignore[union-attr]

        ref_gh = h_ref.grad.detach().clone()  # type: ignore[union-attr]
        ref_gw = w_ref.grad.detach().clone()  # type: ignore[union-attr]
        del h_ref, w_ref, loss_ref
        torch.cuda.empty_cache()

        h_fus = h_base.clone().requires_grad_(True)
        w_fus = w_base.clone().requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_f0 = torch.cuda.memory_allocated(self.device)  # 此时 h_fus.grad / w_fus.grad 均为 None！
        loss_fus = triton_fused_linear_cross_entropy(
            h_fus, w_fus, targets, ignore_index=-100, chunk_size=256, reduction="mean"
        )
        (loss_fus * 0.37).backward()
        torch.cuda.synchronize()
        fused_peak_delta = torch.cuda.max_memory_allocated(self.device) - mem_f0

        self.assertLess(abs(loss_fus.float().item() - fp32_gt_loss.item()), 1.5e-2)
        self.assertTrue(torch.allclose(ref_gh, h_fus.grad, atol=2e-2, rtol=2e-2))  # type: ignore[arg-type]
        self.assertTrue(torch.allclose(ref_gw, w_fus.grad, atol=3e-2, rtol=3e-2))  # type: ignore[arg-type]

        compile_extra_act = compile_peak_delta - grad_leaf_bytes
        fused_extra_act = fused_peak_delta - grad_leaf_bytes
        self.assertLess(fused_peak_delta, compile_peak_delta * 0.45)
        self.assertLess(fused_extra_act, compile_extra_act * 0.30)

    def test_04_custom_ops_meta_and_cpp_dispatcher_counters(self):
        """验证全部 5 个注册 CustomOp 的 Meta 形状推导及 C++ 原子运行期计数器"""
        x = torch.empty((2, 16, 2048), dtype=torch.float16, device="meta")
        r = torch.empty((2, 16, 2048), dtype=torch.float16, device="meta")
        w = torch.empty((2048,), dtype=torch.float16, device="meta")
        o, ur = fused_rmsnorm_residual_op(x, r, w, 1e-6)
        self.assertEqual(o.shape, (2, 16, 2048))
        self.assertEqual(ur.device.type, "meta")
        self.assertEqual(fused_rmsnorm_op(x, w, 1e-6).shape, (2, 16, 2048))

        q = torch.empty((2, 16, 16, 128), dtype=torch.float16, device="meta")
        k = torch.empty((2, 16, 4, 128), dtype=torch.float16, device="meta")
        c = torch.empty((1, 16, 1, 128), dtype=torch.float16, device="meta")
        self.assertEqual(rope_single_op(q, c, c).shape, (2, 16, 16, 128))
        qo, ko = fused_gqa_rope_op(q, k, c, c)
        self.assertEqual(qo.shape, (2, 16, 16, 128))
        self.assertEqual(ko.shape, (2, 16, 4, 128))
        self.assertEqual(fused_swiglu_op(torch.empty((2, 16, 11264), dtype=torch.float16, device="meta")).shape, (2, 16, 5632))

        reset_runtime_counters()
        xc = torch.randn(2, 8, 512, dtype=torch.float16, device=self.device)
        rc = torch.randn(2, 8, 512, dtype=torch.float16, device=self.device)
        wc = torch.ones(512, dtype=torch.float16, device=self.device)
        _ = fused_rmsnorm_residual_op(xc, rc, wc, 1e-6)
        _ = fused_rmsnorm_op(xc, wc, 1e-6)
        cnts = get_runtime_counters()
        self.assertEqual(cnts["fused_rmsnorm_residual"], 1)
        self.assertEqual(cnts["fused_rmsnorm"], 1)

    def test_05_real_torch_compile_llama3_fp16_and_bf16_counters(self):
        """在标准 `torch.compile(backend="inductor")` 下验证 FP16/BF16 4 层 LLaMA-3 编译期与运行期双计数器精确匹配"""
        num_layers = 4
        expected = {
            "fused_rmsnorm_residual": 2 * num_layers,
            "fused_rmsnorm": 1,
            "fused_gqa_rope": num_layers,
            "rope_single": 0,
            "fused_swiglu": num_layers,
        }

        for dtype, hdim, idim, eps, tol in [
            (torch.float16, 1024, 2816, 1e-6, 5e-3),
            (torch.bfloat16, 2048, 5632, 1e-5, 2.5e-2),
        ]:
            torch._dynamo.reset()
            reset_compile_counters()
            reset_runtime_counters()

            model = (
                Llama3DecoderStack(
                    num_layers=num_layers,
                    hidden_dim=hdim,
                    intermediate_dim=idim,
                    num_q_heads=16,
                    num_kv_heads=4,
                    eps=eps,
                )
                .to(device=self.device, dtype=dtype)
                .eval()
            )
            inp = torch.randn(2, 32, hdim, dtype=dtype, device=self.device)

            with inductor_fusion_pass(), torch.inference_mode():
                compiled_model = torch.compile(model, backend="inductor", dynamic=False)
                ref_out = model(inp)
                comp_out_1 = compiled_model(inp)

                self.assertEqual(get_compile_counters(), expected)

                reset_runtime_counters()
                comp_out_2 = compiled_model(inp)
                torch.cuda.synchronize()
                self.assertEqual(get_runtime_counters(), expected)

            self.assertTrue(torch.allclose(ref_out, comp_out_1, atol=tol, rtol=tol))
            self.assertTrue(torch.allclose(ref_out, comp_out_2, atol=tol, rtol=tol))

    def test_06_escape_safety_and_dual_cos_cross_dependency_cycle_guard(self):
        """【P0-2 红方回归】覆盖 (a) var 逃逸安全；(b) 串联 RoPE 反链拦截；(c) 双 (cos_a, cos_b) 交叉依赖防死环测试"""
        # (a) 内部 var 逃逸测试
        torch._dynamo.reset()
        reset_compile_counters()

        class EscapingVarMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(256))

            def forward(self, x, res):
                u = x + res
                a = u.to(torch.float32)
                var = a.pow(2).mean(-1, keepdim=True)
                out = (a * torch.rsqrt(var + 1e-6) * self.w.to(torch.float32)).to(u.dtype)
                return out, u, var

        mod = EscapingVarMod().to(self.device).half().eval()
        x = torch.randn(2, 16, 256, dtype=torch.float16, device=self.device)
        r = torch.randn(2, 16, 256, dtype=torch.float16, device=self.device)
        with inductor_fusion_pass(), torch.inference_mode():
            c_mod = torch.compile(mod, backend="inductor")
            o1, u1, v1 = mod(x, r)
            o2, u2, v2 = c_mod(x, r)

        hits_a = get_compile_counters()
        self.assertEqual(hits_a["fused_rmsnorm_residual"], 0)
        self.assertEqual(hits_a["fused_rmsnorm"], 0)
        self.assertTrue(torch.allclose(o1, o2, atol=2e-3, rtol=2e-3))
        self.assertTrue(torch.allclose(v1, v2, atol=1e-3, rtol=1e-3))

        # (b) 单 (cos, sin) 串联依赖拦截
        torch._dynamo.reset()
        reset_compile_counters()
        reset_runtime_counters()

        class SerialDependentRopeMod(nn.Module):
            def forward(self, q, cos, sin):
                y1 = eager_rope_single(q, cos, sin)
                y2 = eager_rope_single(y1 * 1.01, cos, sin)
                return y1 + y2

        s_mod = SerialDependentRopeMod().to(self.device).half().eval()
        q = torch.randn(2, 16, 8, 64, dtype=torch.float16, device=self.device)
        ang = torch.randn(1, 16, 1, 64, dtype=torch.float32, device=self.device)
        cos, sin = ang.cos().half(), ang.sin().half()
        with inductor_fusion_pass(), torch.inference_mode():
            cs_mod = torch.compile(s_mod, backend="inductor")
            ref_y = s_mod(q, cos, sin)
            _ = cs_mod(q, cos, sin)
            reset_runtime_counters()
            comp_y = cs_mod(q, cos, sin)
            torch.cuda.synchronize()

        self.assertEqual(get_compile_counters()["fused_gqa_rope"], 0)
        self.assertEqual(get_compile_counters()["rope_single"], 2)
        self.assertEqual(get_runtime_counters()["rope_single"], 2)
        self.assertTrue(torch.allclose(ref_y, comp_y, atol=2e-3, rtol=2e-3))

        # (c) 【P0-2 红方复现攻击用例】双 (cos_a, sin_a) 与 (cos_b, sin_b) 交叉依赖：A(a)->D(b), C(b)->B(a)
        # 若使用静态 anc_mask 会先后合并 (A,B)->P1 与 (C,D)->P2 产生 P1<->P2 死环崩溃！
        torch._dynamo.reset()
        reset_compile_counters()
        reset_runtime_counters()

        class DualCosCrossCycleMod(nn.Module):
            def forward(self, q1, q2, cos_a, sin_a, cos_b, sin_b):
                a = eager_rope_single(q1, cos_a, sin_a)
                c = eager_rope_single(q2, cos_b, sin_b)
                d = eager_rope_single(a * 0.9, cos_b, sin_b)   # A -> D
                b = eager_rope_single(c * 1.1, cos_a, sin_a)   # C -> B
                return b + d

        dc_mod = DualCosCrossCycleMod().to(self.device).half().eval()
        q1 = torch.randn(2, 16, 8, 64, dtype=torch.float16, device=self.device)
        q2 = torch.randn(2, 16, 8, 64, dtype=torch.float16, device=self.device)
        ang_b = torch.randn(1, 16, 1, 64, dtype=torch.float32, device=self.device)
        cos_b, sin_b = ang_b.cos().half(), ang_b.sin().half()

        with inductor_fusion_pass(), torch.inference_mode():
            cdc_mod = torch.compile(dc_mod, backend="inductor")
            gt_dc = dc_mod.double()(q1.double(), q2.double(), cos.double(), sin.double(), cos_b.double(), sin_b.double())
            dc_mod.half()
            ref_dc = dc_mod(q1, q2, cos, sin, cos_b, sin_b)
            _ = cdc_mod(q1, q2, cos, sin, cos_b, sin_b)
            reset_runtime_counters()
            comp_dc = cdc_mod(q1, q2, cos, sin, cos_b, sin_b)
            torch.cuda.synchronize()

        hits_c = get_compile_counters()
        calls_c = get_runtime_counters()
        # 合并第一对 (A,B) 后动态传递闭包立即识别出 C -> P1 -> D 可达，安全保留 (C,D) 为 2 个独立 rope_single！
        self.assertEqual(hits_c["fused_gqa_rope"], 1, f"应仅合并第一对兄弟节点: {hits_c}")
        self.assertEqual(hits_c["rope_single"], 2, f"第二对因动态传递闭包可达应保留为单节点: {hits_c}")
        self.assertEqual(calls_c["fused_gqa_rope"], 1)
        self.assertEqual(calls_c["rope_single"], 2)
        eager_dc_err = (ref_dc.double() - gt_dc).abs().max().item()
        comp_dc_err = (comp_dc.double() - gt_dc).abs().max().item()
        self.assertLessEqual(comp_dc_err, eager_dc_err + 1e-3)

    def test_07_hf_official_qwen2_real_weights_and_native_f_rms_norm(self):
        """【P1-2 验证】测试 PyTorch 原生 F.rms_norm 及 HuggingFace 官方写法 Qwen2.5-0.5B 真实 safetensors 权重全命中"""
        # (a) PyTorch 原生 F.rms_norm (solo + residual add, 3D & 2D)
        torch._dynamo.reset()
        reset_compile_counters()
        reset_runtime_counters()

        class NativeFRmsNormMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Parameter(torch.ones(512))
                self.w2 = nn.Parameter(torch.ones(512))

            def forward(self, x):
                n1 = F.rms_norm(x, (512,), self.w1, 1e-6)
                u = x + n1
                n2 = F.rms_norm(u, (512,), self.w2, 1e-6)
                return n2 + u

        nf_mod = NativeFRmsNormMod().to(self.device).to(torch.bfloat16).eval()
        x_nf = torch.randn(2, 16, 512, dtype=torch.bfloat16, device=self.device)
        with inductor_fusion_pass(), torch.inference_mode():
            cnf = torch.compile(nf_mod, backend="inductor")
            ref_nf = nf_mod(x_nf)
            comp_nf = cnf(x_nf)

        hits_nf = get_compile_counters()
        self.assertEqual(hits_nf["fused_rmsnorm"], 1, f"F.rms_norm solo 未命中: {hits_nf}")
        self.assertEqual(hits_nf["fused_rmsnorm_residual"], 1, f"F.rms_norm residual 未命中: {hits_nf}")
        self.assertTrue(torch.allclose(ref_nf, comp_nf, atol=2e-2, rtol=2e-2))

        # (b) HuggingFace 官方 Qwen2 写法 + 本机 Qwen2.5-0.5B-Instruct 真实权重（若本机已缓存则加载真实权重）
        torch._dynamo.reset()
        reset_compile_counters()
        reset_runtime_counters()

        hf_model = OfficialHFQwen2StyleModel(num_layers=2).to(device=self.device, dtype=torch.bfloat16).eval()
        sf_candidates = glob.glob(
            os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/model.safetensors")
        )
        if sf_candidates:
            import safetensors.torch as st
            sd = st.load_file(sf_candidates[0], device=str(self.device))
            with torch.no_grad():
                for idx in range(2):
                    pfx = f"model.layers.{idx}."
                    layer = hf_model.layers[idx]
                    layer["input_layernorm"][0].copy_(sd[pfx + "input_layernorm.weight"])  # type: ignore[index]
                    layer["q_proj"].weight.copy_(sd[pfx + "self_attn.q_proj.weight"])  # type: ignore[index,union-attr]
                    layer["q_proj"].bias.copy_(sd[pfx + "self_attn.q_proj.bias"])  # type: ignore[index,union-attr]
                    layer["k_proj"].weight.copy_(sd[pfx + "self_attn.k_proj.weight"])  # type: ignore[index,union-attr]
                    layer["k_proj"].bias.copy_(sd[pfx + "self_attn.k_proj.bias"])  # type: ignore[index,union-attr]
                    layer["v_proj"].weight.copy_(sd[pfx + "self_attn.v_proj.weight"])  # type: ignore[index,union-attr]
                    layer["v_proj"].bias.copy_(sd[pfx + "self_attn.v_proj.bias"])  # type: ignore[index,union-attr]
                    layer["o_proj"].weight.copy_(sd[pfx + "self_attn.o_proj.weight"])  # type: ignore[index,union-attr]
                    layer["post_attention_layernorm"][0].copy_(sd[pfx + "post_attention_layernorm.weight"])  # type: ignore[index]
                    gate_w = sd[pfx + "mlp.gate_proj.weight"]
                    up_w = sd[pfx + "mlp.up_proj.weight"]
                    layer["gate_up_proj"].weight.copy_(torch.cat([gate_w, up_w], dim=0))  # type: ignore[index,union-attr]
                    layer["down_proj"].weight.copy_(sd[pfx + "mlp.down_proj.weight"])  # type: ignore[index,union-attr]
                hf_model.norm_weight.copy_(sd["model.norm.weight"])
            del sd

        x_qwen = torch.randn(2, 24, 896, dtype=torch.bfloat16, device=self.device)
        expected_qwen = {
            "fused_rmsnorm_residual": 4,  # 2 层 post_attn_norm + 第 1 层 input_norm + 末尾 final_norm = 4
            "fused_rmsnorm": 1,           # 第 0 层入口 input_norm = 1
            "fused_gqa_rope": 2,          # 2 层 14Q/2KV RoPE = 2
            "rope_single": 0,
            "fused_swiglu": 2,            # 2 层 SwiGLU = 2
        }
        with torch.inference_mode():
            gt_qw_fp32 = hf_model.float()(x_qwen.float())
            hf_model.to(torch.bfloat16)
            ref_qw = hf_model(x_qwen)

        with inductor_fusion_pass(), torch.inference_mode():
            c_qwen = torch.compile(hf_model, backend="inductor", dynamic=False)
            _ = c_qwen(x_qwen)
            self.assertEqual(get_compile_counters(), expected_qwen, f"HF 官方 Qwen2.5 命中不匹配: {get_compile_counters()}")

            reset_runtime_counters()
            comp_qw = c_qwen(x_qwen)
            torch.cuda.synchronize()
            self.assertEqual(get_runtime_counters(), expected_qwen)

        # Qwen2.5 真实权重输出幅值达 ~53.0 (BF16 单 ULP = 0.25)，以 FP32 真值为基准校验误差 <= Eager + 1 ULP 且余弦相似度 > 0.9999
        eager_qw_err = (ref_qw.float() - gt_qw_fp32).abs().max().item()
        comp_qw_err = (comp_qw.float() - gt_qw_fp32).abs().max().item()
        cos_sim = F.cosine_similarity(comp_qw.float().flatten(), gt_qw_fp32.flatten(), dim=0).item()
        self.assertLessEqual(comp_qw_err, eager_qw_err + 0.26)
        self.assertGreater(cos_sim, 0.9999)

    def test_08_context_manager_exception_safety_and_cache_isolation(self):
        """【P1-3 & FXGraphCache 隔离验证】确认异常自动还原配置，且同进程先跑默认 compile 后跑自定义 Pass 绝不发生缓存串扰"""
        sentinel = lambda g: None
        inductor_config.post_grad_custom_post_pass = sentinel
        try:
            with self.assertRaises(RuntimeError):
                with inductor_fusion_pass():
                    self.assertIsNot(inductor_config.post_grad_custom_post_pass, sentinel)
                    raise RuntimeError("simulated compile error")
            self.assertIs(inductor_config.post_grad_custom_post_pass, sentinel)
        finally:
            inductor_config.post_grad_custom_post_pass = None

        # 同进程不调用 dynamo.reset()，分别编译默认入口与自定义 Pass 入口并交替调用，验证 FXGraphCache 100% 隔离
        import copy
        m_base = Llama3DecoderStack(num_layers=1, hidden_dim=512, intermediate_dim=1024, num_q_heads=8, num_kv_heads=2).to(self.device).half().eval()
        m_def_raw = copy.deepcopy(m_base)
        m_cus_raw = copy.deepcopy(m_base)
        x = torch.randn(1, 8, 512, dtype=torch.float16, device=self.device)

        @torch.compile(backend="inductor")
        def run_def(inp):
            return m_def_raw(inp)

        @torch.compile(backend="inductor")
        def run_cus(inp):
            return m_cus_raw(inp)

        with torch.inference_mode():
            _ = run_def(x)
            reset_runtime_counters()
            _ = run_def(x)
            torch.cuda.synchronize()
            self.assertEqual(sum(get_runtime_counters().values()), 0)

            with inductor_fusion_pass():
                reset_compile_counters()
                _ = run_cus(x)
                self.assertEqual(get_compile_counters()["fused_rmsnorm_residual"], 2)
                self.assertEqual(get_compile_counters()["fused_gqa_rope"], 1)

            # ABA 交替执行验证无串扰、无重编译覆盖
            reset_runtime_counters()
            _ = run_def(x)
            torch.cuda.synchronize()
            self.assertEqual(sum(get_runtime_counters().values()), 0)
            _ = run_cus(x)
            torch.cuda.synchronize()
            self.assertEqual(get_runtime_counters()["fused_rmsnorm_residual"], 2)


if __name__ == "__main__":
    print("=" * 78)
    print(f"Running Project 3 Red-Team Hardened Suite on: {torch.cuda.get_device_name(0)}")
    print("=" * 78)
    unittest.main(verbosity=2)
