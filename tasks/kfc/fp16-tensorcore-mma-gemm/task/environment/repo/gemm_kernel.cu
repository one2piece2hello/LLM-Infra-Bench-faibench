// gemm_kernel.cu — fp16 matrix-multiply kernel (THIS FILE is the one you edit).
//
// Computes C[M,N] = A[M,K] @ B[K,N] with half-precision (fp16) inputs and
// float32 accumulation, storing a half-precision result. It is invoked from the
// fixed Python wrapper gemm.py, which validates the contract and JIT-compiles
// this translation unit with nvcc.
//
// ---------------------------------------------------------------------------
// FROZEN BASELINE (candidate start state): a straightforward shared-memory
// tiled GEMM on the GPU's general arithmetic path. It is CORRECT for arbitrary
// M, N, K (edges are guarded) but leaves most of the device's fp16 throughput
// on the table. Make it FASTER on the GPU while keeping the result within the
// verifier's tolerance of a high-precision reference. You may use any GPU
// technique available in the image; accumulate in float32 for numerical
// stability and keep the fp16 output.
//
// Forbidden: do NOT delegate the multiply to a prebuilt vendor matrix-multiply
// library or a framework matrix-multiply primitive — implement the multiply
// yourself. The exact banned symbols are listed in the task instructions; the
// scoring harness scans THIS file for them and scores the task 0 on a hit, so do
// not name them here even in comments.
// ---------------------------------------------------------------------------
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef KB_TILE
#define KB_TILE 16
#endif

__global__ void gemm_tiled_kernel(const __half* __restrict__ A,
                                  const __half* __restrict__ B,
                                  __half* __restrict__ C,
                                  int M, int N, int K) {
    __shared__ float As[KB_TILE][KB_TILE];
    __shared__ float Bs[KB_TILE][KB_TILE];
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * KB_TILE + ty;
    const int col = blockIdx.x * KB_TILE + tx;

    float acc = 0.0f;
    const int ntiles = (K + KB_TILE - 1) / KB_TILE;
    for (int t = 0; t < ntiles; ++t) {
        const int aCol = t * KB_TILE + tx;
        const int bRow = t * KB_TILE + ty;
        As[ty][tx] = (row < M && aCol < K) ? __half2float(A[row * K + aCol]) : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? __half2float(B[bRow * N + col]) : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KB_TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];   // float32 accumulate
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}

torch::Tensor gemm_fp16(torch::Tensor A, torch::Tensor B) {
    // A: (M,K) half, B: (K,N) half; both CUDA + contiguous (checked in gemm.py).
    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));
    auto C = torch::empty({M, N}, A.options());
    if (M == 0 || N == 0) {
        return C;
    }
    const c10::cuda::OptionalCUDAGuard device_guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(KB_TILE, KB_TILE);
    const dim3 grid((N + KB_TILE - 1) / KB_TILE, (M + KB_TILE - 1) / KB_TILE);
    gemm_tiled_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fp16", &gemm_fp16,
          "fp16 GEMM C=A@B (frozen shared-memory tiled baseline; float32 accumulate)");
}
