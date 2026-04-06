#include "tool.h"

__global__ static void test_wavefront_1_impl(const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    __syncthreads();

    float4 tmp=a[tid];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_1(const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_1_impl<<<1,32>>>(in,tmp);
    gpu_sync();
    tmp.to_host(out);
}

__global__ static void test_wavefront_2_impl(const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    __syncthreads();

    float4 tmp=a[tid/2];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_2(const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_2_impl<<<1,32>>>(in,tmp);
    gpu_sync();
    tmp.to_host(out);
}

__global__ static void test_wavefront_3_impl(const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    __syncthreads();

    float4 tmp=a[tid%2];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_3(const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_3_impl<<<1,32>>>(in,tmp);
    gpu_sync();
    tmp.to_host(out);
}

__global__ static void test_wavefront_4_impl(const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    __syncthreads();

    float4 tmp=a[tid/4];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_4(const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_4_impl<<<1,32>>>(in,tmp);
    gpu_sync();
    tmp.to_host(out);
}

__global__ static void test_wavefront_5_impl(const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    __syncthreads();

    float4 tmp=a[tid%4];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_5(const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_5_impl<<<1,32>>>(in,tmp);
    gpu_sync();
    tmp.to_host(out);
}


__global__ static void test_wavefront_6_impl(const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    __syncthreads();

    float4 tmp=a[tid/8];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_6(const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_6_impl<<<1,32>>>(in,tmp);
    gpu_sync();
    tmp.to_host(out);
}

__global__ static void test_wavefront_7_impl(const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    __syncthreads();

    float4 tmp=a[tid%8];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_7(const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_7_impl<<<1,32>>>(in,tmp);
    gpu_sync();
    tmp.to_host(out);
}

__global__ static void test_wavefront_8_impl(const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    __syncthreads();

    float4 tmp=a[tid/16];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_8(const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_8_impl<<<1,32>>>(in,tmp);
    gpu_sync();
    tmp.to_host(out);
}

__global__ static void test_wavefront_9_impl(const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    __syncthreads();

    float4 tmp=a[tid%16];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_9(const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_9_impl<<<1,32>>>(in,tmp);
    gpu_sync();
    tmp.to_host(out);
}


__global__ static void test_wavefront_10_impl(const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    __syncthreads();

    float4 tmp=a[tid/32];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_10(const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_10_impl<<<1,32>>>(in,tmp);
    gpu_sync();
    tmp.to_host(out);
}

__global__ static void test_wavefront_11_impl(const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    __syncthreads();

    float4 tmp=a[tid%32];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_11(const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_11_impl<<<1,32>>>(in,tmp);
    gpu_sync();
    tmp.to_host(out);
}

__global__ static void test_wavefront_x_impl(const unsigned int*idx,const float* in,float*out)
{
    unsigned int tid=threadIdx.x;

    __shared__ float4 a[32];

    a[tid]=tid[(float4*)in];
    unsigned int i=idx[tid];
    __syncthreads();

    float4 tmp=a[i];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_wavefront_x(const unsigned int*idx,const float* in,float*out)
{
    GPU_Data<float> tmp(32);
    test_wavefront_x_impl<<<1,32>>>(idx,in,tmp);
    gpu_sync();
    tmp.to_host(out);
}

