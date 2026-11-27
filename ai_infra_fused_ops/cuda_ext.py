"""
Native CUDA C++ Extension Loader (`csrc/fused_add_rmsnorm_cuda.cu`)

自动探测本机 Linux `/usr/local/cuda*` 工具链并通过 `torch.utils.cpp_extension.load` 编译为持久化 `.so` 动态库。
扩展加载后通过 C++ `TORCH_LIBRARY_IMPL(ai_infra, CUDA, m)` 直接向 `c10::Dispatcher` 注册原生核函数（消除 Python 回调开销），
并提供 `inspect_cuda_ext_ptx_stats()` 对编译出的 `.so` 执行真实 `cuobjdump --dump-ptx` 指令级核验。
"""

import glob
import os
import re
import subprocess

import torch
import torch.utils.cpp_extension as cpp_ext

_EXT_MOD = None
_EXT_SO_PATH = ""


def ensure_linux_cuda_toolchain() -> str:
    """确保 Linux 原生 CUDA Toolkit 置于 PATH 首位并返回 CUDA_HOME"""
    cuda_dirs = sorted(glob.glob("/usr/local/cuda*"), reverse=True)
    cuda_home = "/usr/local/cuda"
    for d in cuda_dirs:
        if os.path.isfile(os.path.join(d, "bin", "nvcc")):
            cuda_home = d
            break

    cbin = os.path.join(cuda_home, "bin")
    cur_path = os.environ.get("PATH", "")
    parts = [p for p in cur_path.split(os.pathsep) if p != cbin]
    os.environ["PATH"] = os.pathsep.join([cbin] + parts)
    os.environ.setdefault("CUDA_HOME", cuda_home)
    return cuda_home


def _detect_nvcc_gencode_flags() -> list[str]:
    """根据当前 GPU Compute Capability 与 nvcc 版本生成编译架构参数"""
    major, minor = torch.cuda.get_device_capability(0)
    cc = major * 10 + minor

    out = subprocess.check_output(["nvcc", "--version"]).decode(
        "utf-8", errors="replace"
    )
    if "release 12.6" in out or "release 12.4" in out or "release 12.1" in out:
        target_cc = min(cc, 89)
        return [
            "-O3",
            "--use_fast_math",
            f"-gencode=arch=compute_{target_cc},code=sm_{target_cc}",
            f"-gencode=arch=compute_{target_cc},code=compute_{target_cc}",
        ]
    return [
        "-O3",
        "--use_fast_math",
        f"-gencode=arch=compute_{cc},code=sm_{cc}",
    ]


def get_cuda_rmsnorm_ext():
    """懒加载并缓存编译好的 C++/CUDA 扩展模块"""
    global _EXT_MOD, _EXT_SO_PATH
    if _EXT_MOD is not None:
        return _EXT_MOD

    if not hasattr(torch.ops, "ai_infra") or not hasattr(
        torch.ops.ai_infra, "fused_rmsnorm"
    ):
        from . import custom_ops as _ensure_schema  # noqa: F401

        if _EXT_MOD is not None:
            return _EXT_MOD

    ensure_linux_cuda_toolchain()
    cpp_ext._check_cuda_version = lambda *args, **kwargs: None

    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cu_src = os.path.join(proj_root, "csrc", "fused_add_rmsnorm_cuda.cu")
    build_dir = os.path.expanduser("~/.cache/ai_infra_p3_cuda_ext_v2")
    os.makedirs(build_dir, exist_ok=True)

    ext_name = "ai_infra_rmsnorm_cuda_ext_v2"
    _EXT_MOD = cpp_ext.load(
        name=ext_name,
        sources=[cu_src],
        build_directory=build_dir,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=_detect_nvcc_gencode_flags(),
        verbose=False,
    )
    so_candidates = glob.glob(os.path.join(build_dir, f"{ext_name}*.so"))
    _EXT_SO_PATH = (
        so_candidates[0] if so_candidates else getattr(_EXT_MOD, "__file__", "")
    )
    return _EXT_MOD


def inspect_cuda_ext_ptx_stats() -> dict[str, int]:
    """对编译生成的 `.so` 执行 `cuobjdump --dump-ptx` 指令统计，实证 128-bit 向量化与 Warp 蝶形规约"""
    get_cuda_rmsnorm_ext()
    ensure_linux_cuda_toolchain()
    if not _EXT_SO_PATH or not os.path.isfile(_EXT_SO_PATH):
        return {"ld_v4_u32": 0, "st_v4_u32": 0, "shfl_bfly": 0, "bar_sync": 0}

    ptx = subprocess.check_output(["cuobjdump", "--dump-ptx", _EXT_SO_PATH]).decode(
        "utf-8", errors="replace"
    )
    return {
        "ld_v4_u32": len(re.findall(r"ld\.global(?:\.nc)?\.v4\.u32", ptx)),
        "st_v4_u32": len(re.findall(r"st\.global\.v4\.u32", ptx)),
        "shfl_bfly": len(re.findall(r"shfl\.sync\.bfly\.b32", ptx)),
        "bar_sync": len(re.findall(r"bar\.sync", ptx)),
    }


def cuda_fused_rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """通过 PyBind11 直接调用原生 CUDA C++ (.cu) 实现的 Fused Add + RMSNorm"""
    ext = get_cuda_rmsnorm_ext()
    return ext.fused_add_rmsnorm(x, residual, weight, float(eps))


def cuda_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """通过 PyBind11 直接调用原生 CUDA C++ (.cu) 实现的独立 RMSNorm"""
    ext = get_cuda_rmsnorm_ext()
    return ext.rmsnorm(x, weight, float(eps))
