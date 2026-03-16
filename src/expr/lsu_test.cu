#include "tool.h"

static __global__ void test_lsu_v1_impl(int k,float* out)
{
	__shared__ float a[258*33];

	float sum=0;
    if(threadIdx.x==0)
    {
        for(int i=0;i<32;i++)
            a[i]=a[32+i]=i;
    }
    __syncthreads();

    const float* const b=a+(threadIdx.x&255)*33;

    for(int i=0;i<k;i++)
    {
        #pragma unroll 32
        for(int j=0;j<32;j++)
            sum+=b[j];
    }
    out[threadIdx.x]=sum;
}

void test_lsu_v1(int k,int t,float* out)
{
    GPU_Data<float> tmp(t);
    test_lsu_v1_impl<<<1,t>>>(k,tmp);
    tmp.to_host(out);
}



