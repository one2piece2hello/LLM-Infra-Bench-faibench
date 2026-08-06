// sgemm_kernel.cu — single-precision matrix-multiply kernel (THIS FILE is the one
// you edit).
//
// Computes D[M,N] = alpha * (A[M,K] @ B[K,N]) + beta * C[M,N] with float32 inputs,
// float32 accumulation and a float32 result. It is invoked from the fixed Python
// wrapper sgemm.py, which validates the contract and JIT-compiles this translation
// unit with nvcc. The input C is read (for the beta term) but not modified; the
// result is written to a fresh output buffer.
//
// ---------------------------------------------------------------------------
// FROZEN BASELINE (candidate start state): the most direct possible kernel — ONE
// GPU thread per output element. Each thread walks the full length-K inner
// dimension, re-reading one row of A and one column of B straight from global
// memory, and never shares or reuses any value with its neighbours. It is CORRECT
// for arbitrary M, N, K (including sizes that are not multiples of any tile, K=1,
// and large K) but it re-reads the same operands from global memory over and over,
// so the device spends almost all of its time moving data rather than computing.
// Make it FASTER on the GPU while keeping the result within the verifier's
// tolerance of a high-precision reference. You may use any GPU technique available
// in the image; keep the accumulation in float32.
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

#ifndef KB_BLOCK
#define KB_BLOCK 16
#endif

__global__ void sgemm_naive_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   const float* __restrict__ C,
                                   float* __restrict__ D,
                                   int M, int N, int K,
                                   float alpha, float beta) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];   // float32 accumulate
        }
        const int idx = row * N + col;
        D[idx] = alpha * acc + beta * C[idx];
    }
}

torch::Tensor sgemm(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                    double alpha, double beta) {
    // A: (M,K), B: (K,N), C: (M,N) — all float32, CUDA, contiguous (checked in sgemm.py).
    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));
    auto D = torch::empty({M, N}, A.options());
    if (M == 0 || N == 0) {
        return D;
    }
    const c10::cuda::OptionalCUDAGuard device_guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(KB_BLOCK, KB_BLOCK);
    const dim3 grid((N + KB_BLOCK - 1) / KB_BLOCK, (M + KB_BLOCK - 1) / KB_BLOCK);
    sgemm_naive_kernel<<<grid, block, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        D.data_ptr<float>(),
        M, N, K,
        static_cast<float>(alpha), static_cast<float>(beta));
    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm", &sgemm,
          "fp32 GEMM D = alpha*(A@B) + beta*C (frozen naive one-thread-per-output baseline; float32 accumulate)");
}
