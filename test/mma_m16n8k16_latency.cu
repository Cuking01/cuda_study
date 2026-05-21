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

__device__ __forceinline__ void mma_m16n8k16_f32_f16(
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1,
    float& c0,
    float& c1,
    float& c2,
    float& c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__global__ __launch_bounds__(32, 1) void bench_mma_m16n8k16_latency_kernel(
    float* out,
    uint64_t* clk_out
) {
    const int tid = threadIdx.x;
    const uint32_t lane = static_cast<uint32_t>(tid & 31);

    uint32_t a0 = 0x3c003800u ^ (lane * 0x00010001u);
    uint32_t a1 = 0x34003000u + (lane * 0x00030003u);
    uint32_t a2 = 0x2c002800u ^ (lane * 0x00050005u);
    uint32_t a3 = 0x24002000u + (lane * 0x00070007u);
    uint32_t b0 = 0x3a003600u ^ (lane * 0x00090009u);
    uint32_t b1 = 0x32002e00u + (lane * 0x000b000bu);

    float c0 = 0.125f + static_cast<float>(tid) * 0.001f;
    float c1 = 0.250f + static_cast<float>(tid) * 0.001f;
    float c2 = 0.375f + static_cast<float>(tid) * 0.001f;
    float c3 = 0.500f + static_cast<float>(tid) * 0.001f;

    asm volatile("bar.sync 0;" ::: "memory");
    uint64_t start = clock64();

#pragma unroll 1
    for (int r = 0; r < REPEAT; ++r) {
#pragma unroll 64
        for (int i = 0; i < ITERS; ++i) {
            mma_m16n8k16_f32_f16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
        }
    }

    uint64_t stop = clock64();
    asm volatile("bar.sync 0;" ::: "memory");

    out[tid * 4 + 0] = c0;
    out[tid * 4 + 1] = c1;
    out[tid * 4 + 2] = c2;
    out[tid * 4 + 3] = c3;

    if (tid == 0) {
        clk_out[0] = stop - start;
    }
}

static void run_once()
{
    float* d_out = nullptr;
    uint64_t* d_clk = nullptr;
    float h_out[32 * 4]{};
    uint64_t h_clk = 0;

    CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_out)));
    CUDA_CHECK(cudaMalloc(&d_clk, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    CUDA_CHECK(cudaMemset(d_clk, 0, sizeof(uint64_t)));

    bench_mma_m16n8k16_latency_kernel<<<1, 32>>>(d_out, d_clk);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_clk, d_clk, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (float value : h_out) {
        checksum += static_cast<double>(value);
    }

    const double mma_count = static_cast<double>(ITERS) * static_cast<double>(REPEAT);
    printf("cycles = %llu\n", static_cast<unsigned long long>(h_clk));
    printf("mma instructions per warp = %.0f\n", mma_count);
    printf("latency = %.4f cycles / mma.m16n8k16\n", static_cast<double>(h_clk) / mma_count);
    printf("checksum = %.9e\n", checksum);

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_clk));
}

int main()
{
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("GPU: %s\n", prop.name);
    printf("SM version: %d.%d\n", prop.major, prop.minor);
    printf("launch: grid=1 block=32\n");
    printf("instruction: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n");
    printf("ITERS=%d REPEAT=%d\n\n", ITERS, REPEAT);

    run_once();
    return 0;
}
