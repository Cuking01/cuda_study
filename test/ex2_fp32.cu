#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef ITERS
#define ITERS 4096
#endif

#ifndef REPEAT
#define REPEAT 256
#endif

#define CUDA_CHECK(x) do {                                      \
    cudaError_t err = (x);                                      \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        std::exit(1);                                           \
    }                                                           \
} while (0)

__device__ __forceinline__ float ex2_approx_ftz_f32(float x) {
    float y;
    asm volatile(
        "ex2.approx.ftz.f32 %0, %1;"
        : "=f"(y)
        : "f"(x)
    );
    return y;
}

__global__ void bench_ex2_kernel(float *out, uint64_t *clk_out, int active_threads) {
    const int tid = threadIdx.x;

    float x0 = 0.50f + static_cast<float>(tid) * 0.0001f;
    float x1 = 0.25f + static_cast<float>(tid) * 0.0002f;
    float x2 = -0.50f + static_cast<float>(tid) * 0.0003f;
    float x3 = -1.00f + static_cast<float>(tid) * 0.0004f;

    uint64_t start = 0;
    uint64_t stop = 0;

    if (tid < active_threads) {
        asm volatile("bar.sync 0;");
        start = clock64();

#pragma unroll 1
        for (int r = 0; r < REPEAT; ++r) {
#pragma unroll 64
            for (int i = 0; i < ITERS; ++i) {
                x0 = ex2_approx_ftz_f32(x0);
                x1 = ex2_approx_ftz_f32(x1);
                x2 = ex2_approx_ftz_f32(x2);
                x3 = ex2_approx_ftz_f32(x3);
            }
        }

        stop = clock64();
        asm volatile("bar.sync 0;");

        out[tid * 4 + 0] = x0;
        out[tid * 4 + 1] = x1;
        out[tid * 4 + 2] = x2;
        out[tid * 4 + 3] = x3;

        if (tid == 0) {
            clk_out[0] = stop - start;
        }
    }
}

static void run_case(int threads) {
    float *d_out = nullptr;
    uint64_t *d_clk = nullptr;
    uint64_t h_clk = 0;

    CUDA_CHECK(cudaMalloc(&d_out, threads * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_clk, sizeof(uint64_t)));

    CUDA_CHECK(cudaMemset(d_out, 0, threads * 4 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_clk, 0, sizeof(uint64_t)));

    bench_ex2_kernel<<<1, threads>>>(d_out, d_clk, threads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_clk, d_clk, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    const double inst_per_thread =
        static_cast<double>(REPEAT) * static_cast<double>(ITERS) * 4.0;

    const double inst =
        inst_per_thread * static_cast<double>(threads);

    printf("%3d threads: cycles = %llu, "
           "f32 ex2 inst/cycle = %.3f\n",
           threads,
           static_cast<unsigned long long>(h_clk),
           inst / static_cast<double>(h_clk));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_clk));
}

int main() {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("GPU: %s\n", prop.name);
    printf("SM version: %d.%d\n", prop.major, prop.minor);
    printf("instruction: ex2.approx.ftz.f32\n");
    printf("ITERS=%d REPEAT=%d\n\n", ITERS, REPEAT);

    for (int threads = 32; threads <= 128; threads += 32) {
        run_case(threads);
    }
    run_case(256);
    run_case(512);
    run_case(1024);
    return 0;
}
