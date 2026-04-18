#include "tool.h"
#include<stdint.h>

__device__ __forceinline__ static void copy_16B_prefetch(void*dst,const void* src)
{
	uint32_t smem_int_ptr=__cvta_generic_to_shared(dst);

	asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
	                 :: "r"(smem_int_ptr),
	                 "l"(src),
	                 "n"(16));
}

__device__ __forceinline__ static void copy_16B(void*dst,const void* src)
{
	uint32_t smem_int_ptr=__cvta_generic_to_shared(dst);

	asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
	                 :: "r"(smem_int_ptr),
	                 "l"(src),
	                 "n"(16));
}

__device__ __forceinline__ static void commit_group()
{
	asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ static void wait_all()
{
	asm volatile("cp.async.wait_all;\n" ::);
}

__global__ static void test_cp_sync_1_impl(const float* in,float*out)
{
    __align__(128) __shared__ float4 a[32];

    unsigned int tid=threadIdx.x;
    int idx=tid;
    
    a[tid]=idx[(const float4*)in];
    __syncthreads();

    float4 tmp=a[tid];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_cp_sync_1(const float* in,float*out)
{
    GPU_Data<float> in_dev(in,128);
    GPU_Data<float> out_dev(32);
    test_cp_sync_1_impl<<<1,32>>>(in_dev,out_dev); 
    gpu_sync();
    out_dev.to_host(out);
}

__global__ static void test_cp_sync_2_impl(const float* in,float*out)
{
    __align__(128) __shared__ float4 a[32];

    unsigned int tid=threadIdx.x;
    int idx=(tid&24)+(tid%8+1)%8;
    
    a[tid]=idx[(const float4*)in];
    __syncthreads();

    float4 tmp=a[tid];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_cp_sync_2(const float* in,float*out)
{
    GPU_Data<float> in_dev(in,128);
    GPU_Data<float> out_dev(32);
    test_cp_sync_2_impl<<<1,32>>>(in_dev,out_dev); 
    gpu_sync();
    out_dev.to_host(out);
}

__global__ static void test_cp_async_prefetch_1_impl(const float* in,float*out)
{
    __align__(128) __shared__ float4 a[32];

    unsigned int tid=threadIdx.x;
    unsigned int idx=tid;
    
    copy_16B_prefetch(a+tid,(const float4*)in+idx);
    commit_group();
    wait_all();
    __syncthreads();

    float4 tmp=a[tid];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_cp_async_prefetch_1(const float* in,float*out)
{
    GPU_Data<float> in_dev(in,128);
    GPU_Data<float> out_dev(32);
    test_cp_async_prefetch_1_impl<<<1,32>>>(in_dev,out_dev); 
    gpu_sync();
    out_dev.to_host(out);
}

__global__ static void test_cp_async_prefetch_2_impl(const float* in,float*out)
{
    __align__(128) __shared__ float4 a[32];

    unsigned int tid=threadIdx.x;
    unsigned int idx=(tid&24)+(tid%8+1)%8;
    
    copy_16B_prefetch(a+tid,(const float4*)in+idx);
    commit_group();
    wait_all();
    __syncthreads();

    float4 tmp=a[tid];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_cp_async_prefetch_2(const float* in,float*out)
{
    GPU_Data<float> in_dev(in,128);
    GPU_Data<float> out_dev(32);
    test_cp_async_prefetch_2_impl<<<1,32>>>(in_dev,out_dev); 
    gpu_sync();
    out_dev.to_host(out);
}

__global__ static void test_cp_async_1_impl(const float* in,float*out)
{
    __align__(128) __shared__ float4 a[32];

    unsigned int tid=threadIdx.x;
    unsigned int idx=tid;
    
    copy_16B(a+tid,(const float4*)in+idx);
    commit_group();
    wait_all();
    __syncthreads();

    float4 tmp=a[tid];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_cp_async_1(const float* in,float*out)
{
    GPU_Data<float> in_dev(in,128);
    GPU_Data<float> out_dev(32);
    test_cp_async_1_impl<<<1,32>>>(in_dev,out_dev); 
    gpu_sync();
    out_dev.to_host(out);
}

__global__ static void test_cp_async_2_impl(const float* in,float*out)
{
    __align__(128) __shared__ float4 a[32];

    unsigned int tid=threadIdx.x;
    unsigned int idx=(tid&24)+(tid%8+1)%8;
    
    copy_16B(a+tid,(const float4*)in+idx);
    commit_group();
    wait_all();
    __syncthreads();

    float4 tmp=a[tid];
    out[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void test_cp_async_2(const float* in,float*out)
{
    GPU_Data<float> in_dev(in,128);
    GPU_Data<float> out_dev(32);
    test_cp_async_2_impl<<<1,32>>>(in_dev,out_dev); 
    gpu_sync();
    out_dev.to_host(out);
}

