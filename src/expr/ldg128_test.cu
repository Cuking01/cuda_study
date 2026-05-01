#include "tool.h"

__global__ static void ldg128_test_impl_v1(float* gmem,float*result)
{
    __align__(128) __shared__ float smem[128];
    unsigned int tid = threadIdx.x;
    unsigned int id=tid;
    
    float4 tmp = *(float4*)(gmem+id*4);
    __syncthreads();
    result[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}


void ldg128_test_v1(float* hmem)
{
    GPU_Data<float> gmem(hmem,128);
    GPU_Data<float> result(32);
    float tmp[32];
    ldg128_test_impl_v1<<<1,32>>>(gmem,result);
    result.to_host(tmp);

    gpu_sync();
    
    for(int i=0;i<32;i++)
        printf("result[%d]=%f\n",i,tmp[i]);
    fflush(stdout);
}

__global__ static void ldg128_test_impl_v2(float* gmem,float*result,unsigned int stride)
{
    __align__(128) __shared__ float smem[128];
    unsigned int tid = threadIdx.x;
    unsigned int id=tid%4*stride+tid/4*4;
    
    float4 tmp = *(float4*)(gmem+id);
    __syncthreads();
    result[tid]=tmp.x+tmp.y+tmp.z+tmp.w;
}

void ldg128_test_v2(float* hmem,unsigned int stride)
{
    GPU_Data<float> gmem(hmem,4*stride);
    GPU_Data<float> result(32);
    float tmp[32];
    ldg128_test_impl_v2<<<1,32>>>(gmem,result,stride);
    result.to_host(tmp);

    gpu_sync();
    
    for(int i=0;i<32;i++)
        printf("result[%d]=%f\n",i,tmp[i]);
    fflush(stdout);
}

