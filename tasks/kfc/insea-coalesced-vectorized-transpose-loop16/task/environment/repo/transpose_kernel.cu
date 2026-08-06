// transpose_kernel.cu — 2-D matrix transpose kernel (THIS FILE is the one you edit).
//
// Produces y[N, M] = the transpose of x[M, N]: y[j, i] == x[i, j], with x and y
// both row-major contiguous. Elements are float32 or float16 and are copied
// unchanged (this is pure data movement — there is no arithmetic). It is invoked
// from the fixed Python wrapper transpose.py, which validates the contract and
// JIT-compiles this translation unit with nvcc.
//
// ---------------------------------------------------------------------------
// FROZEN BASELINE (candidate start state): a straightforward element-by-element
// reorder — one GPU thread produces one output element by reading the source
// element at its computed position and writing it to the transposed position. It
// is CORRECT for arbitrary M, N (non-square and sizes that are not a multiple of
// any block width included; edges are guarded), but the pattern in which it
// touches global memory drives the data across the memory bus far below the rate
// the hardware can sustain. This operation is MEMORY-BOUND: the run time is
// dominated by how the reads and writes land in global memory, not by any
// computation. Make it FASTER on the GPU while producing exactly the same output.
// You may use any GPU technique available in the image.
//
// Forbidden: do NOT delegate the reorder to a built-in framework/library reorder
// primitive that already produces the transposed layout for you — implement the
// reorder yourself. The exact banned symbols are listed in the task
// instructions; the scoring harness scans THIS file for them and scores the task
// 0 on a hit, so do not name them here even in comments.
// ---------------------------------------------------------------------------
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef KB_BW
#define KB_BW 16
#endif

// One thread per output element. Thread (tx, ty) handles source position
// (row, col) = (blockIdx.y*KB_BW+ty, blockIdx.x*KB_BW+tx) and writes it to the
// transposed slot in y. Correct for every M, N; both indices are guarded.
template <typename scalar_t>
__global__ void transpose_elementwise_kernel(const scalar_t* __restrict__ in,
                                             scalar_t* __restrict__ out,
                                             int M, int N) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;   // in [0, N)
    const int row = blockIdx.y * blockDim.y + threadIdx.y;   // in [0, M)
    if (row < M && col < N) {
        // in has shape (M, N); out has shape (N, M): out[col, row] = in[row, col]
        out[static_cast<long long>(col) * M + row] =
            in[static_cast<long long>(row) * N + col];
    }
}

template <typename scalar_t>
static void launch_elementwise(const torch::Tensor& x, torch::Tensor& y,
                              int M, int N) {
    const c10::cuda::OptionalCUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 block(KB_BW, KB_BW);
    const dim3 grid((N + KB_BW - 1) / KB_BW, (M + KB_BW - 1) / KB_BW);
    transpose_elementwise_kernel<scalar_t><<<grid, block, 0, stream>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), M, N);
}

// Host entry: x is (M, N), contiguous, CUDA, float32 or float16 (checked in
// transpose.py). Returns y (N, M) of the same dtype.
torch::Tensor transpose_2d(torch::Tensor x) {
    const int M = static_cast<int>(x.size(0));
    const int N = static_cast<int>(x.size(1));
    auto y = torch::empty({N, M}, x.options());
    if (M == 0 || N == 0) {
        return y;
    }
    if (x.scalar_type() == at::kFloat) {
        launch_elementwise<float>(x, y, M, N);
    } else if (x.scalar_type() == at::kHalf) {
        launch_elementwise<at::Half>(x, y, M, N);
    } else {
        TORCH_CHECK(false, "unsupported dtype for transpose_2d");
    }
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_2d", &transpose_2d,
          "2-D matrix transpose y[N,M]=x[M,N] (frozen element-by-element baseline)");
}
