# `fused-cuda-triton-inductor` 实机性能验证报告（ABA 五轮中位数对照 · 修正显存基线 · 含 PTX 汇编与 D2D 带宽审计）

- **实测时间**：2026-11-19 22:09:09
- **实测硬件**：`NVIDIA GeForce RTX 5060 Laptop GPU`（CC `12.0`，`26` SMs，显存 `7.93 GB`）
- **显存带宽双标尺**：硬件标称理论峰值 `384.0 GB/s` ｜ **实测 256 MB 跨 L2 持续 D2D 拷贝上限 `312.6 GB/s`（标称值的 `81.4%`）**
- **软件栈**：PyTorch `2.13.0+cu130` + Triton `3.7.1` + 原生 CUDA C++ (`TORCH_LIBRARY_IMPL` C++ 直接注册)
- **`.cu` 扩展 `cuobjdump --dump-ptx` 实证指令统计**：128-bit 向量化加载 `ld.global(.nc).v4.u32` = **`80`** 处，128-bit 向量化存储 `st.global.v4.u32` = **`30`** 处，寄存器蝶形规约 `shfl.sync.bfly.b32` = **`140`** 处，块同步 `bar.sync` = **`28`** 处

---

## 1. 大词表 `Fused Linear + Cross-Entropy` 正面对决原生 `torch.compile`（修正梯度清零基线）

评测规模：`M=4096, H=1024, Vocab=128256 (BF16, Fwd+Bwd, 叶子梯度固定占 258.5 MB)`（注：Warmup 后已清空 `h.grad` 与 `w.grad` 再记录 `base_mem`）

| 实现方案 | 前反向中位耗时 (ms) | 总峰值显存增量 (含叶子梯度, MB) | 纯中间激活额外显存 (MB) | **相对 `compile` 速度** | **总增量节省 / 中间激活节省** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| PyTorch Eager (`F.cross_entropy(F.linear)`) | 127.52 | 3006.0 | 2747.5 | 0.90x | - |
| 原生 `torch.compile(backend='inductor')` | 114.22 | 1260.5 | 1002.0 | 1.00x (基准) | 0.0% / 0.0% |
| **本方案 (`chunk_size=256`, 同 `test_03`)** | **134.12** | **517.0** | **258.5** | **0.85x** (vs Eager 0.95x) | **-59.0% / -74.2%** |
| **本方案 (`chunk_size=1024`)** | **115.35** | **517.0** | **258.5** | **0.99x** (vs Eager 1.11x) | **-59.0% / -74.2%** |

---

## 2. `Fused Add + RMSNorm` 四方对比（5 轮 ABA 中位数，覆盖单 Pass 寄存器驻留、纯 HBM 大矩阵与 FP32 向量化）

| 工况与尺寸 | Eager (us) | 原生 `compile` (us) | Triton (us) | **原生 `.cu` (us)** | **`.cu` vs `compile`** | **`.cu` 有效带宽 (占持续 D2D 上限 / 标称峰值)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Decode 单步 `[1, 4096]` FP16 | 139.1 | 14.8 | 14.5 | **13.8** | **1.07x** (vs Eager 10.08x) | 3.0 GB/s (0.9% D2D / 0.8% 标称) |
| Decode 小批次 `[8, 4096]` FP16 | 149.2 | 13.4 | 13.6 | **14.6** | **0.92x** (vs Eager 10.23x) | 18.5 GB/s (5.9% D2D / 4.8% 标称) |
| Prefill `[4096, 4096]` FP16 (32MB 含 L2 增益) | 2375.5 | 434.9 | 421.6 | **436.3** | **1.00x** (vs Eager 5.44x) | 307.6 GB/s (98.4% D2D / 80.1% 标称) |
| 大吞吐 `[16384, 4096]` FP16 (512MB 纯 HBM 持续区) | 9583.5 | 1702.7 | 1712.6 | **1726.2** | **0.99x** (vs Eager 5.55x) | 311.0 GB/s (99.5% D2D / 81.0% 标称) |
| FP32 向量化 `[4096, 4096]` FP32 | 2075.2 | 864.3 | 858.3 | **853.1** | **1.01x** (vs Eager 2.43x) | 314.7 GB/s (100.7% D2D / 81.9% 标称) |

---

## 3. 异构头数 GQA (`16 Q / 4 KV`) Q+K RoPE 对比（5 轮 ABA 中位数）

| 工况与尺寸 (FP16) | Eager (us) | 原生 `torch.compile` (us) | **单次发射 Triton GQA RoPE (us)** | **相对 `compile` 速度** | **相对 Eager 速度** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Decode 单步 (`B=1, S=1, 16Q/4KV, D=128`) | 138.7 | 16.2 | **11.6** | **1.40x** | **11.97x** |
| Decode 批次 (`B=8, S=1, 16Q/4KV, D=128`) | 142.7 | 16.0 | **16.2** | **0.99x** | **8.81x** |
| Prefill 序列 (`B=2, S=1024, 16Q/4KV, D=128`) | 386.0 | 76.4 | **75.1** | **1.02x** | **5.14x** |

---

## 4. 8 层 LLaMA-3 GQA Decoder Stack 端到端对比（独立实例隔离缓存 + 5 轮 ABA 中位数）

- **默认路径隔离核验**：实测执行 `m_default(inp)` 时 `RUNTIME_CALL_COUNTERS` 全部严格为 `0`（零缓存串扰）
- **自定义路径编译期 `COMPILE_MATCH_COUNTERS`**：`{'fused_rmsnorm_residual': 16, 'fused_rmsnorm': 1, 'fused_gqa_rope': 8, 'rope_single': 0, 'fused_swiglu': 8}`
- **自定义路径单步运行期 `RUNTIME_CALL_COUNTERS`**：`{'fused_rmsnorm_residual': 16, 'fused_rmsnorm': 1, 'fused_gqa_rope': 8, 'rope_single': 0, 'fused_swiglu': 8}`

| 推理工况 (`Hidden=2048, 8 Layers`, ABA 5 轮中位数) | Eager (ms) | 原生 `torch.compile` (ms) | **注入本方案 Post-Grad Pass (ms)** | **vs Eager** | **vs 原生 `compile`** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Decode 单步 (`B=1, S=1`) | 4.823 | 2.726 | **2.829** | **1.70x** | **0.96x** |
| Decode 小批次 (`B=8, S=1`) | 6.862 | 2.956 | **2.864** | **2.40x** | **1.03x** |
| Prefill 长序列 (`B=2, S=512`) | 29.339 | 26.397 | **26.211** | **1.12x** | **1.01x** |

---

## 5. 客观工程边界与 Roofline 分析

1. **跨 GEMM 大词表损失算子（`Fused Linear + Cross-Entropy`）**：
   - 修正 `base_mem` 后，包含 `258.5 MB` 叶子梯度在内，原生 `torch.compile` 总峰值增量为 `1260.5 MB`（纯中间激活 `1002.0 MB`），本方案压降至 `517.0 MB`（纯中间激活 `258.5 MB`，主要为支持 `retain_graph=True` 在 `backward()` 非原地缩放 `w.grad` 产生的单份副本开销，与序列长度 `M` 无关），`chunk_size=1024` 下耗时与原生 `compile` 持平（`0.99x`）；同时针对 FP16 采用 `2^-4` 指数移位预缩放与尾端寄存器 FP32 缩放，避免了非目标词梯度下溢归零。
2. **实测持续 D2D 带宽天花板与 Roofline 收敛**：
   - 本机 RTX 5060 Laptop 256 MB 跨 L2 持续 D2D 拷贝上限实测为 `312.6 GB/s`（标称 `384.0 GB/s` 的 `81.4%`）。在 `[4096, 4096]` 与 `[16384, 4096]`（读写 `512 MB`）以及 FP32 向量化路径下，`.cu`、Triton 与原生 `compile` 均达到实测持续 D2D 上限的 `98% ~ 100%`，完整贴合硬件带宽屋顶。
3. **C++ `TORCH_LIBRARY_IMPL` 注册与整网 ABA 对照表现**：
   - 将 `fused_rmsnorm_residual` 与 `fused_rmsnorm` 迁移至 `.cu` 内部通过 `TORCH_LIBRARY_IMPL(ai_infra, CUDA, m)` 纯 C++ 注册后，消除了 17 次归一化算子的 Python 回调税；其余 16 次 Triton 自定义算子（8 次 RoPE + 8 次 SwiGLU）仍经 Python `@torch.library.impl` 分发。在 8 层整网中，注意力与线性投影矩阵乘（cuBLAS/FlashAttention）占据绝大部分 GPU 时间，因此自定义编译 Pass 相对原生 `torch.compile` 的端到端净收益集中在几个百分点量级（以表 4 的 5 轮 ABA 中位数实测值为准）。
4. **适用范围边界**：
   - `inductor_fusion_pass` 基于 `pm.fwd_only` 构建并自动跳过含 `tangent` 的反向图，专门面向推理图优化；训练场景的大词表显存优化请直接调用 `triton_fused_linear_cross_entropy`。
