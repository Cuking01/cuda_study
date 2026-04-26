#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(err));       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

static double bytes_to_gib(size_t bytes) {
    return static_cast<double>(bytes) / 1024.0 / 1024.0 / 1024.0;
}

static double bytes_to_gb(size_t bytes) {
    return static_cast<double>(bytes) / 1000.0 / 1000.0 / 1000.0;
}

static void print_device_info(int device) {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::printf("Device %d: %s\n", device, prop.name);
    std::printf("PCI domain:bus:device = %04x:%02x:%02x\n",
                prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    std::printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    std::printf("Global memory: %.2f GiB\n", bytes_to_gib(prop.totalGlobalMem));
    std::printf("\n");
}

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static double bandwidth_gbps(size_t bytes, float ms) {
    return bytes_to_gb(bytes) / (static_cast<double>(ms) / 1000.0);
}

static void touch_host_buffer(unsigned char* ptr, size_t bytes) {
    const size_t step = 4096;
    for (size_t i = 0; i < bytes; i += step) {
        ptr[i] = static_cast<unsigned char>(i);
    }
    ptr[bytes - 1] = 123;
}

static double run_h2d(unsigned char* h_src,
                      unsigned char* d_dst,
                      size_t bytes,
                      int warmup,
                      int iters,
                      cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, stream));
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = elapsed_ms(start, stop);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return bandwidth_gbps(bytes * static_cast<size_t>(iters), ms);
}

static double run_d2h(unsigned char* h_dst,
                      unsigned char* d_src,
                      size_t bytes,
                      int warmup,
                      int iters,
                      cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(h_dst, d_src, bytes, cudaMemcpyDeviceToHost, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(h_dst, d_src, bytes, cudaMemcpyDeviceToHost, stream));
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = elapsed_ms(start, stop);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return bandwidth_gbps(bytes * static_cast<size_t>(iters), ms);
}

static double run_bidirectional(unsigned char* h0,
                                unsigned char* h1,
                                unsigned char* d0,
                                unsigned char* d1,
                                size_t bytes,
                                int warmup,
                                int iters,
                                cudaStream_t h2d_stream,
                                cudaStream_t d2h_stream) {
    for (int i = 0; i < warmup; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(d0, h0, bytes, cudaMemcpyHostToDevice, h2d_stream));
        CHECK_CUDA(cudaMemcpyAsync(h1, d1, bytes, cudaMemcpyDeviceToHost, d2h_stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(h2d_stream));
    CHECK_CUDA(cudaStreamSynchronize(d2h_stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, h2d_stream));

    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaMemcpyAsync(d0, h0, bytes, cudaMemcpyHostToDevice, h2d_stream));
        CHECK_CUDA(cudaMemcpyAsync(h1, d1, bytes, cudaMemcpyDeviceToHost, d2h_stream));
    }

    CHECK_CUDA(cudaEventRecord(stop, h2d_stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaStreamSynchronize(d2h_stream));

    float ms = elapsed_ms(start, stop);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    size_t total_bytes = bytes * static_cast<size_t>(iters) * 2;
    return bandwidth_gbps(total_bytes, ms);
}

int main(int argc, char** argv) {
    int device = 0;
    size_t mib = 1024;
    int iters = 50;
    int warmup = 10;

    if (argc >= 2) {
        device = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        mib = static_cast<size_t>(std::atoll(argv[2]));
    }
    if (argc >= 4) {
        iters = std::atoi(argv[3]);
    }

    const size_t bytes = mib * 1024ULL * 1024ULL;

    CHECK_CUDA(cudaSetDevice(device));
    print_device_info(device);

    std::printf("Single transfer size: %zu MiB\n", mib);
    std::printf("Warmup iterations: %d\n", warmup);
    std::printf("Measured iterations: %d\n", iters);
    std::printf("Total transferred per one-way test: %.2f GiB\n",
                bytes_to_gib(bytes * static_cast<size_t>(iters)));
    std::printf("\n");

    unsigned char* h0 = nullptr;
    unsigned char* h1 = nullptr;
    unsigned char* d0 = nullptr;
    unsigned char* d1 = nullptr;

    CHECK_CUDA(cudaHostAlloc(&h0, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h1, bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaMalloc(&d0, bytes));
    CHECK_CUDA(cudaMalloc(&d1, bytes));

    touch_host_buffer(h0, bytes);
    touch_host_buffer(h1, bytes);

    cudaStream_t stream0, stream1;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

    CHECK_CUDA(cudaMemcpyAsync(d0, h0, bytes, cudaMemcpyHostToDevice, stream0));
    CHECK_CUDA(cudaMemcpyAsync(d1, h1, bytes, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream0));
    CHECK_CUDA(cudaStreamSynchronize(stream1));

    double h2d = run_h2d(h0, d0, bytes, warmup, iters, stream0);
    double d2h = run_d2h(h1, d1, bytes, warmup, iters, stream0);
    double bidir = run_bidirectional(h0, h1, d0, d1, bytes, warmup, iters, stream0, stream1);

    std::printf("Results:\n");
    std::printf("  H2D              : %.2f GB/s\n", h2d);
    std::printf("  D2H              : %.2f GB/s\n", d2h);
    std::printf("  Bidirectional sum: %.2f GB/s\n", bidir);
    std::printf("\n");

    std::printf("Reference:\n");
    std::printf("  PCIe 4.0 x16 theoretical raw-ish one-way payload: about 31.5 GB/s\n");
    std::printf("  PCIe 5.0 x8  theoretical raw-ish one-way payload: about 31.5 GB/s\n");
    std::printf("  Real measured one-way bandwidth is usually lower, often ~26-30 GB/s if healthy.\n");
    std::printf("\n");

    CHECK_CUDA(cudaStreamDestroy(stream0));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaFree(d0));
    CHECK_CUDA(cudaFree(d1));
    CHECK_CUDA(cudaFreeHost(h0));
    CHECK_CUDA(cudaFreeHost(h1));

    return 0;
}
