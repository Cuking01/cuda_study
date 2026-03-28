#include "tool.h"

static __global__ void test_lsu_v1_impl(int k,float* out)
{
	__shared__ float a[270*32];

	float sum=0;
    if(threadIdx.x<32)
    {
        for(int i=0;i<270;i++)
            a[threadIdx.x*270+i]=i;
    }
    __syncthreads();

    const float* const b=a+(threadIdx.x&127)*65;

    for(int i=0;i<k;i++)
    {
        #pragma unroll 32
        for(int j=0;j<64;j++)
            sum+=b[j];
    }
    out[threadIdx.x]=sum;
}

void test_lsu_v1(int k,int t,float* out)
{
    GPU_Data<float> tmp(t);
    test_lsu_v1_impl<<<1,t>>>(k,tmp);
    gpu_sync();
    tmp.to_host(out);
}



