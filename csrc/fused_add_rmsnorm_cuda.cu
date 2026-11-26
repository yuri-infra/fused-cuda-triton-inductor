/*
 * Native CUDA C++ Kernels + Direct C++ `TORCH_LIBRARY_IMPL(ai_infra, CUDA, m)` Registration:
 *   1. Fused Residual Add + RMSNorm (`fused_add_rmsnorm`)
 *   2. Standalone RMSNorm (`rmsnorm`)
 * Supports FP16, BF16, and FP32 (all with 128-bit `uint4` vectorized global memory transactions).
 *
 * Key Micro-architectural Optimizations:
 * 1. 128-bit Vectorized Global Memory Access (`uint4` = 16 Bytes = 8x FP16/BF16 or 4x FP32):
 *    Emits `ld.global.nc.v4.u32` / `st.global.v4.u32` across FP16, BF16, AND FP32.
 * 2. Single-Pass Register Residency (`reg_u[MAX_VECS][VEC_ELEMS]` for aligned `N <= 8192`):
 *    Caches `x` / `x + residual` directly in hardware registers across the reduction barrier,
 *    eliminating the second global/L1 memory read pass during normalization.
 * 3. Two-Level Warp-Shuffle + Shared-Memory Reduction:
 *    `__shfl_xor_sync(0xffffffffu, val, offset)` intra-warp butterfly reduction + single
 *    `__shared__ float s_warp_sums[32]` array with two `__syncthreads()` barriers.
 * 4. Pure C++ `TORCH_LIBRARY_IMPL(ai_infra, CUDA, m)` Dispatch:
 *    Bypasses Python GIL / `@torch.library.impl` callback overhead when invoked by TorchInductor,
 *    while maintaining lock-free `std::atomic<int64_t>` runtime execution counters.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <atomic>
#include <tuple>
#include <vector>

namespace {

std::atomic<int64_t> g_rmsnorm_add_calls{0};
std::atomic<int64_t> g_rmsnorm_solo_calls{0};

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v) {
    return static_cast<float>(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v) {
    return static_cast<scalar_t>(v);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_inv_rms(
    float thread_sum_sq,
    const int warp_id,
    const int lane_id,
    const int num_warps,
    const int N,
    const float eps
) {
    float warp_sum = warp_reduce_sum(thread_sum_sq);
    __shared__ float s_warp_sums[32];
    __shared__ float s_inv_rms;

    if (lane_id == 0) {
        s_warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? s_warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            s_inv_rms = rsqrtf(val / static_cast<float>(N) + eps);
        }
    }
    __syncthreads();
    return s_inv_rms;
}

template <typename scalar_t>
__global__ void fused_add_rmsnorm_vec128_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ norm_out,
    scalar_t* __restrict__ updated_res,
    const int N,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int num_warps = block_threads >> 5;

    const scalar_t* row_x = x + static_cast<int64_t>(row) * N;
    const scalar_t* row_res = residual + static_cast<int64_t>(row) * N;
    scalar_t* row_out = norm_out + static_cast<int64_t>(row) * N;
    scalar_t* row_up = updated_res + static_cast<int64_t>(row) * N;

    float thread_sum_sq = 0.0f;

    constexpr int VEC_ELEMS = (sizeof(scalar_t) <= 4) ? (16 / sizeof(scalar_t)) : 1;
    constexpr int MAX_VECS = 4;

    const bool can_vec = (sizeof(scalar_t) == 2 || sizeof(scalar_t) == 4) &&
                         ((reinterpret_cast<uintptr_t>(row_x) & 15u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(row_res) & 15u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(row_up) & 15u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(row_out) & 15u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(weight) & 15u) == 0);

    // Fast Path: Single-Pass Register Residency (eliminates second HBM/L1 read of updated_res)
    if (can_vec && (N % VEC_ELEMS == 0) && (N <= block_threads * VEC_ELEMS * MAX_VECS)) {
        const int n_vec = N / VEC_ELEMS;
        const uint4* vec_x = reinterpret_cast<const uint4*>(row_x);
        const uint4* vec_res = reinterpret_cast<const uint4*>(row_res);
        const uint4* vec_w = reinterpret_cast<const uint4*>(weight);
        uint4* vec_up = reinterpret_cast<uint4*>(row_up);
        uint4* vec_out = reinterpret_cast<uint4*>(row_out);

        float reg_u[MAX_VECS][VEC_ELEMS];

        #pragma unroll
        for (int v = 0; v < MAX_VECS; ++v) {
            const int idx = tid + v * block_threads;
            if (idx < n_vec) {
                uint4 ux = __ldg(&vec_x[idx]);
                uint4 ur = __ldg(&vec_res[idx]);
                uint4 u_out;

                const scalar_t* sx = reinterpret_cast<const scalar_t*>(&ux);
                const scalar_t* sr = reinterpret_cast<const scalar_t*>(&ur);
                scalar_t* su = reinterpret_cast<scalar_t*>(&u_out);

                #pragma unroll
                for (int k = 0; k < VEC_ELEMS; ++k) {
                    float acc = to_float(sx[k]) + to_float(sr[k]);
                    scalar_t rounded = from_float<scalar_t>(acc);
                    su[k] = rounded;
                    float u_fp32 = to_float(rounded);
                    reg_u[v][k] = u_fp32;
                    thread_sum_sq += u_fp32 * u_fp32;
                }
                vec_up[idx] = u_out;
            }
        }

        const float inv_rms = block_reduce_inv_rms(thread_sum_sq, warp_id, lane_id, num_warps, N, eps);

        #pragma unroll
        for (int v = 0; v < MAX_VECS; ++v) {
            const int idx = tid + v * block_threads;
            if (idx < n_vec) {
                uint4 uw = __ldg(&vec_w[idx]);
                uint4 uo;

                const scalar_t* sw = reinterpret_cast<const scalar_t*>(&uw);
                scalar_t* so = reinterpret_cast<scalar_t*>(&uo);

                #pragma unroll
                for (int k = 0; k < VEC_ELEMS; ++k) {
                    float normed = (reg_u[v][k] * inv_rms) * to_float(sw[k]);
                    so[k] = from_float<scalar_t>(normed);
                }
                vec_out[idx] = uo;
            }
        }
        return;
    }

    // General Fallback Path (arbitrary N, non-multiple-of-VEC_ELEMS tail, or unaligned tensors)
    if (can_vec) {
        const int n_vec = N / VEC_ELEMS;
        const uint4* vec_x = reinterpret_cast<const uint4*>(row_x);
        const uint4* vec_res = reinterpret_cast<const uint4*>(row_res);
        uint4* vec_up = reinterpret_cast<uint4*>(row_up);

        for (int idx = tid; idx < n_vec; idx += block_threads) {
            uint4 ux = __ldg(&vec_x[idx]);
            uint4 ur = __ldg(&vec_res[idx]);
            uint4 u_out;

            const scalar_t* sx = reinterpret_cast<const scalar_t*>(&ux);
            const scalar_t* sr = reinterpret_cast<const scalar_t*>(&ur);
            scalar_t* su = reinterpret_cast<scalar_t*>(&u_out);

            #pragma unroll
            for (int k = 0; k < VEC_ELEMS; ++k) {
                float acc = to_float(sx[k]) + to_float(sr[k]);
                scalar_t rounded = from_float<scalar_t>(acc);
                su[k] = rounded;
                float u_fp32 = to_float(rounded);
                thread_sum_sq += u_fp32 * u_fp32;
            }
            vec_up[idx] = u_out;
        }

        for (int col = n_vec * VEC_ELEMS + tid; col < N; col += block_threads) {
            float acc = to_float(row_x[col]) + to_float(row_res[col]);
            scalar_t rounded = from_float<scalar_t>(acc);
            row_up[col] = rounded;
            float u_fp32 = to_float(rounded);
            thread_sum_sq += u_fp32 * u_fp32;
        }
    } else {
        for (int col = tid; col < N; col += block_threads) {
            float acc = to_float(row_x[col]) + to_float(row_res[col]);
            scalar_t rounded = from_float<scalar_t>(acc);
            row_up[col] = rounded;
            float u_fp32 = to_float(rounded);
            thread_sum_sq += u_fp32 * u_fp32;
        }
    }

    const float inv_rms = block_reduce_inv_rms(thread_sum_sq, warp_id, lane_id, num_warps, N, eps);

    if (can_vec) {
        const int n_vec = N / VEC_ELEMS;
        const uint4* vec_up = reinterpret_cast<const uint4*>(row_up);
        const uint4* vec_w = reinterpret_cast<const uint4*>(weight);
        uint4* vec_out = reinterpret_cast<uint4*>(row_out);

        for (int idx = tid; idx < n_vec; idx += block_threads) {
            uint4 uu = vec_up[idx];
            uint4 uw = __ldg(&vec_w[idx]);
            uint4 uo;

            const scalar_t* su = reinterpret_cast<const scalar_t*>(&uu);
            const scalar_t* sw = reinterpret_cast<const scalar_t*>(&uw);
            scalar_t* so = reinterpret_cast<scalar_t*>(&uo);

            #pragma unroll
            for (int k = 0; k < VEC_ELEMS; ++k) {
                float normed = (to_float(su[k]) * inv_rms) * to_float(sw[k]);
                so[k] = from_float<scalar_t>(normed);
            }
            vec_out[idx] = uo;
        }

        for (int col = n_vec * VEC_ELEMS + tid; col < N; col += block_threads) {
            float normed = (to_float(row_up[col]) * inv_rms) * to_float(weight[col]);
            row_out[col] = from_float<scalar_t>(normed);
        }
    } else {
        for (int col = tid; col < N; col += block_threads) {
            float normed = (to_float(row_up[col]) * inv_rms) * to_float(weight[col]);
            row_out[col] = from_float<scalar_t>(normed);
        }
    }
}

template <typename scalar_t>
__global__ void rmsnorm_vec128_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ norm_out,
    const int N,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int num_warps = block_threads >> 5;

    const scalar_t* row_x = x + static_cast<int64_t>(row) * N;
    scalar_t* row_out = norm_out + static_cast<int64_t>(row) * N;

    float thread_sum_sq = 0.0f;

    constexpr int VEC_ELEMS = (sizeof(scalar_t) <= 4) ? (16 / sizeof(scalar_t)) : 1;
    constexpr int MAX_VECS = 4;

    const bool can_vec = (sizeof(scalar_t) == 2 || sizeof(scalar_t) == 4) &&
                         ((reinterpret_cast<uintptr_t>(row_x) & 15u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(row_out) & 15u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(weight) & 15u) == 0);

    // Fast Path: Single-Pass Register Residency
    if (can_vec && (N % VEC_ELEMS == 0) && (N <= block_threads * VEC_ELEMS * MAX_VECS)) {
        const int n_vec = N / VEC_ELEMS;
        const uint4* vec_x = reinterpret_cast<const uint4*>(row_x);
        const uint4* vec_w = reinterpret_cast<const uint4*>(weight);
        uint4* vec_out = reinterpret_cast<uint4*>(row_out);

        float reg_x[MAX_VECS][VEC_ELEMS];

        #pragma unroll
        for (int v = 0; v < MAX_VECS; ++v) {
            const int idx = tid + v * block_threads;
            if (idx < n_vec) {
                uint4 ux = __ldg(&vec_x[idx]);
                const scalar_t* sx = reinterpret_cast<const scalar_t*>(&ux);
                #pragma unroll
                for (int k = 0; k < VEC_ELEMS; ++k) {
                    float val = to_float(sx[k]);
                    reg_x[v][k] = val;
                    thread_sum_sq += val * val;
                }
            }
        }

        const float inv_rms = block_reduce_inv_rms(thread_sum_sq, warp_id, lane_id, num_warps, N, eps);

        #pragma unroll
        for (int v = 0; v < MAX_VECS; ++v) {
            const int idx = tid + v * block_threads;
            if (idx < n_vec) {
                uint4 uw = __ldg(&vec_w[idx]);
                uint4 uo;
                const scalar_t* sw = reinterpret_cast<const scalar_t*>(&uw);
                scalar_t* so = reinterpret_cast<scalar_t*>(&uo);
                #pragma unroll
                for (int k = 0; k < VEC_ELEMS; ++k) {
                    so[k] = from_float<scalar_t>((reg_x[v][k] * inv_rms) * to_float(sw[k]));
                }
                vec_out[idx] = uo;
            }
        }
        return;
    }

    if (can_vec) {
        const int n_vec = N / VEC_ELEMS;
        const uint4* vec_x = reinterpret_cast<const uint4*>(row_x);
        for (int idx = tid; idx < n_vec; idx += block_threads) {
            uint4 ux = __ldg(&vec_x[idx]);
            const scalar_t* sx = reinterpret_cast<const scalar_t*>(&ux);
            #pragma unroll
            for (int k = 0; k < VEC_ELEMS; ++k) {
                float v = to_float(sx[k]);
                thread_sum_sq += v * v;
            }
        }
        for (int col = n_vec * VEC_ELEMS + tid; col < N; col += block_threads) {
            float v = to_float(row_x[col]);
            thread_sum_sq += v * v;
        }
    } else {
        for (int col = tid; col < N; col += block_threads) {
            float v = to_float(row_x[col]);
            thread_sum_sq += v * v;
        }
    }

    const float inv_rms = block_reduce_inv_rms(thread_sum_sq, warp_id, lane_id, num_warps, N, eps);

    if (can_vec) {
        const int n_vec = N / VEC_ELEMS;
        const uint4* vec_x = reinterpret_cast<const uint4*>(row_x);
        const uint4* vec_w = reinterpret_cast<const uint4*>(weight);
        uint4* vec_out = reinterpret_cast<uint4*>(row_out);

        for (int idx = tid; idx < n_vec; idx += block_threads) {
            uint4 ux = __ldg(&vec_x[idx]);
            uint4 uw = __ldg(&vec_w[idx]);
            uint4 uo;

            const scalar_t* sx = reinterpret_cast<const scalar_t*>(&ux);
            const scalar_t* sw = reinterpret_cast<const scalar_t*>(&uw);
            scalar_t* so = reinterpret_cast<scalar_t*>(&uo);

            #pragma unroll
            for (int k = 0; k < VEC_ELEMS; ++k) {
                float normed = (to_float(sx[k]) * inv_rms) * to_float(sw[k]);
                so[k] = from_float<scalar_t>(normed);
            }
            vec_out[idx] = uo;
        }
        for (int col = n_vec * VEC_ELEMS + tid; col < N; col += block_threads) {
            float normed = (to_float(row_x[col]) * inv_rms) * to_float(weight[col]);
            row_out[col] = from_float<scalar_t>(normed);
        }
    } else {
        for (int col = tid; col < N; col += block_threads) {
            float normed = (to_float(row_x[col]) * inv_rms) * to_float(weight[col]);
            row_out[col] = from_float<scalar_t>(normed);
        }
    }
}

inline int select_block_threads(const int N, const size_t elem_size) {
    if (elem_size == 4) {
        // FP32: VEC_ELEMS=4, MAX_VECS=4 -> 256 threads covers N <= 4096, 512 threads covers N <= 8192
        if (N <= 2048) return 128;
        if (N <= 4096) return 256;
        return 512;
    }
    // FP16 / BF16: VEC_ELEMS=8, MAX_VECS=4 -> 128 threads covers N <= 4096, 256 threads covers N <= 8192
    if (N <= 2048) return 128;
    if (N <= 8192) return 256;
    return 512;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> cuda_fused_add_rmsnorm_tuple(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps
) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda() && weight.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual shapes must match");
    TORCH_CHECK(x.scalar_type() == residual.scalar_type() && x.scalar_type() == weight.scalar_type(),
                "x, residual, and weight dtypes must match");

    auto x_c = x.is_contiguous() ? x : x.contiguous();
    auto res_c = residual.is_contiguous() ? residual : residual.contiguous();
    auto w_c = weight.is_contiguous() ? weight : weight.contiguous();

    const int N = static_cast<int>(x_c.size(-1));
    TORCH_CHECK(w_c.numel() == N, "weight length must match last dimension N");
    const int M = static_cast<int>(x_c.numel() / N);

    auto norm_out = torch::empty_like(x_c);
    auto updated_res = torch::empty_like(res_c);

    if (M == 0) {
        return std::make_tuple(norm_out, updated_res);
    }

    const int threads = select_block_threads(N, x_c.element_size());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x_c.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x_c.scalar_type(),
        "cuda_fused_add_rmsnorm",
        [&] {
            fused_add_rmsnorm_vec128_kernel<scalar_t><<<M, threads, 0, stream>>>(
                x_c.data_ptr<scalar_t>(),
                res_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                norm_out.data_ptr<scalar_t>(),
                updated_res.data_ptr<scalar_t>(),
                N,
                static_cast<float>(eps)
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    return std::make_tuple(norm_out, updated_res);
}

torch::Tensor cuda_rmsnorm_impl(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps
) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight dtypes must match");

    auto x_c = x.is_contiguous() ? x : x.contiguous();
    auto w_c = weight.is_contiguous() ? weight : weight.contiguous();

    const int N = static_cast<int>(x_c.size(-1));
    TORCH_CHECK(w_c.numel() == N, "weight length must match last dimension N");
    const int M = static_cast<int>(x_c.numel() / N);

    auto norm_out = torch::empty_like(x_c);
    if (M == 0) {
        return norm_out;
    }

    const int threads = select_block_threads(N, x_c.element_size());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x_c.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x_c.scalar_type(),
        "cuda_rmsnorm",
        [&] {
            rmsnorm_vec128_kernel<scalar_t><<<M, threads, 0, stream>>>(
                x_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                norm_out.data_ptr<scalar_t>(),
                N,
                static_cast<float>(eps)
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    return norm_out;
}

// Pure C++ `c10::Dispatcher` targets (zero Python callback / zero GIL overhead inside TorchInductor!)
std::tuple<torch::Tensor, torch::Tensor> cuda_fused_add_rmsnorm_op(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps
) {
    g_rmsnorm_add_calls.fetch_add(1, std::memory_order_relaxed);
    return cuda_fused_add_rmsnorm_tuple(x, residual, weight, eps);
}

torch::Tensor cuda_rmsnorm_op(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps
) {
    g_rmsnorm_solo_calls.fetch_add(1, std::memory_order_relaxed);
    return cuda_rmsnorm_impl(x, weight, eps);
}

int64_t get_rmsnorm_add_calls() {
    return g_rmsnorm_add_calls.load(std::memory_order_relaxed);
}

int64_t get_rmsnorm_solo_calls() {
    return g_rmsnorm_solo_calls.load(std::memory_order_relaxed);
}

void reset_rmsnorm_calls() {
    g_rmsnorm_add_calls.store(0, std::memory_order_relaxed);
    g_rmsnorm_solo_calls.store(0, std::memory_order_relaxed);
}

TORCH_LIBRARY_IMPL(ai_infra, CUDA, m) {
    m.impl("fused_rmsnorm_residual", &cuda_fused_add_rmsnorm_op);
    m.impl("fused_rmsnorm", &cuda_rmsnorm_op);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_rmsnorm", &cuda_fused_add_rmsnorm_tuple, "Native CUDA C++ Fused Residual Add + RMSNorm");
    m.def("rmsnorm", &cuda_rmsnorm_impl, "Native CUDA C++ Standalone RMSNorm");
    m.def("get_rmsnorm_add_calls", &get_rmsnorm_add_calls, "Get C++ atomic call count for fused_rmsnorm_residual");
    m.def("get_rmsnorm_solo_calls", &get_rmsnorm_solo_calls, "Get C++ atomic call count for fused_rmsnorm");
    m.def("reset_rmsnorm_calls", &reset_rmsnorm_calls, "Reset C++ atomic call counters");
}
