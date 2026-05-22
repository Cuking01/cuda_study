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

constexpr unsigned kFullMask = 0xffffffffu;

enum class ShflOp {
    Index,
    Up,
    Down,
    Xor,
};

#define CUDA_CHECK(x) do {                                      \
    cudaError_t err = (x);                                      \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        std::exit(1);                                           \
    }                                                           \
} while (0)

template <ShflOp Op>
__device__ __forceinline__ uint32_t shfl(uint32_t value, int lane_id) {
    if (Op == ShflOp::Index) {
        return __shfl_sync(kFullMask, value, (lane_id + 1) & 31);
    }
    if (Op == ShflOp::Up) {
        return __shfl_up_sync(kFullMask, value, 1);
    }
    if (Op == ShflOp::Down) {
        return __shfl_down_sync(kFullMask, value, 1);
    }
    return __shfl_xor_sync(kFullMask, value, 1);
}

template <ShflOp Op>
__global__ __launch_bounds__(32, 1) void bench_shfl_latency_kernel(
    uint32_t* out,
    uint64_t* clk_out
) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    uint32_t value = 0x12340000u + static_cast<uint32_t>(tid);

    asm volatile("bar.sync 0;" ::: "memory");
    const uint64_t start = clock64();

#pragma unroll 1
    for (int r = 0; r < REPEAT; ++r) {
#pragma unroll 64
        for (int i = 0; i < ITERS; ++i) {
            value = shfl<Op>(value, lane_id);
        }
    }

    const uint64_t stop = clock64();
    asm volatile("bar.sync 0;" ::: "memory");

    out[tid] = value;
    if (tid == 0) {
        clk_out[0] = stop - start;
    }
}

template <ShflOp Op>
__global__ void bench_shfl_throughput_kernel(uint32_t* out, uint64_t* clk_out) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;

    uint32_t x0 = 0x12340000u + static_cast<uint32_t>(tid);
    uint32_t x1 = 0x23450000u ^ static_cast<uint32_t>(tid * 3);
    uint32_t x2 = 0x34560000u + static_cast<uint32_t>(tid * 5);
    uint32_t x3 = 0x45670000u ^ static_cast<uint32_t>(tid * 7);

    asm volatile("bar.sync 0;" ::: "memory");
    uint64_t start = 0;
    if (tid == 0) {
        start = clock64();
    }

#pragma unroll 1
    for (int r = 0; r < REPEAT; ++r) {
#pragma unroll 64
        for (int i = 0; i < ITERS; ++i) {
            x0 = shfl<Op>(x0, lane_id);
            x1 = shfl<Op>(x1, lane_id);
            x2 = shfl<Op>(x2, lane_id);
            x3 = shfl<Op>(x3, lane_id);
        }
    }

    uint64_t stop = 0;
    if (tid == 0) {
        stop = clock64();
    }
    asm volatile("bar.sync 0;" ::: "memory");

    out[tid * 4 + 0] = x0;
    out[tid * 4 + 1] = x1;
    out[tid * 4 + 2] = x2;
    out[tid * 4 + 3] = x3;
    if (tid == 0) {
        clk_out[0] = stop - start;
    }
}

static const char* shfl_name(ShflOp op) {
    if (op == ShflOp::Index) {
        return "__shfl_sync";
    }
    if (op == ShflOp::Up) {
        return "__shfl_up_sync";
    }
    if (op == ShflOp::Down) {
        return "__shfl_down_sync";
    }
    return "__shfl_xor_sync";
}

template <ShflOp Op>
static void run_latency_case() {
    uint32_t* d_out = nullptr;
    uint64_t* d_clk = nullptr;
    uint32_t h_out[32]{};
    uint64_t h_clk = 0;

    CUDA_CHECK(cudaMalloc(&d_out, sizeof(h_out)));
    CUDA_CHECK(cudaMalloc(&d_clk, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    CUDA_CHECK(cudaMemset(d_clk, 0, sizeof(uint64_t)));

    bench_shfl_latency_kernel<Op><<<1, 32>>>(d_out, d_clk);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_clk, d_clk, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    uint64_t checksum = 0;
    for (uint32_t value : h_out) {
        checksum += value;
    }

    const double inst =
        static_cast<double>(ITERS) * static_cast<double>(REPEAT);
    printf("latency: cycles = %llu, cycles/warp shfl = %.4f, checksum = 0x%llx\n",
           static_cast<unsigned long long>(h_clk),
           static_cast<double>(h_clk) / inst,
           static_cast<unsigned long long>(checksum));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_clk));
}

template <ShflOp Op>
static void run_throughput_case(int threads) {
    uint32_t* d_out = nullptr;
    uint64_t* d_clk = nullptr;
    uint64_t h_clk = 0;

    CUDA_CHECK(cudaMalloc(&d_out, threads * 4 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_clk, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_out, 0, threads * 4 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_clk, 0, sizeof(uint64_t)));

    bench_shfl_throughput_kernel<Op><<<1, threads>>>(d_out, d_clk);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_clk, d_clk, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    const double warp_inst =
        static_cast<double>(ITERS) * static_cast<double>(REPEAT) * 4.0 *
        static_cast<double>(threads / 32);
    printf("throughput: %4d threads, cycles = %llu, warp shfl inst/cycle = %.3f\n",
           threads,
           static_cast<unsigned long long>(h_clk),
           warp_inst / static_cast<double>(h_clk));

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_clk));
}

template <ShflOp Op>
static void run_throughput_cases() {
    for (int threads = 32; threads <= 128; threads += 32) {
        run_throughput_case<Op>(threads);
    }
    run_throughput_case<Op>(256);
    run_throughput_case<Op>(512);
    run_throughput_case<Op>(1024);
}

template <ShflOp Op>
static void run_case() {
    printf("\nfunction: %s\n", shfl_name(Op));
    run_latency_case<Op>();
    run_throughput_cases<Op>();
}

int main() {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("GPU: %s\n", prop.name);
    printf("SM version: %d.%d\n", prop.major, prop.minor);
    printf("ITERS=%d REPEAT=%d\n", ITERS, REPEAT);
    printf("latency launch: grid=1 block=32\n");
    printf("throughput launch: grid=1, block threads shown per case\n");
    printf("throughput uses four independent shuffle dependency chains per lane\n");

    run_case<ShflOp::Index>();
    run_case<ShflOp::Up>();
    run_case<ShflOp::Down>();
    run_case<ShflOp::Xor>();
    return 0;
}
