// gemm_kernel.cu — dense fp32 matrix-multiply kernel (THIS FILE is the one you edit).
//
// Computes C[M,N] = A[M,K] @ B[K,N] with float32 inputs and float32
// accumulation. It is invoked from the fixed Python wrapper gemm.py, which
// validates the contract and JIT-compiles this translation unit with nvcc.
//
// ---------------------------------------------------------------------------
// FROZEN BASELINE (candidate start state): a straightforward shared-memory
// tiled GEMM. One thread block is assigned to each output tile and walks the
// entire inner dimension K by itself, accumulating in float32. It is CORRECT for
// arbitrary M, N, K (edges are guarded). When the output is large this keeps the
// device busy; but when the output has FEW rows and columns while the inner
// dimension K is very LARGE, only a handful of blocks are launched, so most of
// the GPU stays idle and each block's long serial walk down the inner dimension
// dominates the runtime. Make it FASTER on the GPU for those
// few-output / large-inner-dimension shapes while keeping float32 accumulation
// and staying within the verifier's tolerance of a high-precision reference.
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

#ifndef KB_TILE
#define KB_TILE 16
#endif

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
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
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KB_TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];   // float32 accumulate
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor gemm_forward(torch::Tensor A, torch::Tensor B) {
    // A: (M,K) float32, B: (K,N) float32; both CUDA + contiguous (checked in gemm.py).
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
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_forward", &gemm_forward,
          "fp32 GEMM C=A@B (frozen shared-memory tiled baseline; float32 accumulate)");
}
