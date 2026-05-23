#pragma once

__device__ __forceinline__ static void pack(float* dst, const float* src)
{
    *(half2*)dst=__floats2half2_rn(src[0],src[1]);
};

__device__ __forceinline__ static float max_x4(float* src0,float* src1)
{
    float t0=fmaxf(src0[0],src1[0]);
    float t1=fmaxf(src0[1],src1[1]);
    return fmaxf(t0,t1);
}

__device__ __forceinline__ static float sum_x4(float* src0,float* src1)
{
    float t0=src0[0]+src1[0];
    float t1=src0[1]+src1[1];
    return t0+t1;
}

__device__ __forceinline__ static void butterfly_max_x4(float&mx)
{
    mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));
    mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
}

__device__ __forceinline__ static void non_butterfly_max_x4(float&mx)
{
    float t1=__shfl_xor_sync(0xffffffff,mx,1);
    float t2=__shfl_xor_sync(0xffffffff,mx,2);
    float t3=__shfl_xor_sync(0xffffffff,mx,3);
    t1=fmaxf(mx,t1);
    t2=fmaxf(t2,t3);
    mx=fmaxf(t1,t2);
}

__device__ __forceinline__ static void butterfly_sum_x4(float&sum)
{
    sum+=__shfl_xor_sync(0xffffffff,sum,1);
    sum+=__shfl_xor_sync(0xffffffff,sum,2);
}

__device__ __forceinline__ static void non_butterfly_sum_x4(float&sum)
{
    float t1=__shfl_xor_sync(0xffffffff,sum,1);
    float t2=__shfl_xor_sync(0xffffffff,sum,2);
    float t3=__shfl_xor_sync(0xffffffff,sum,3);
    sum=(sum+t1)+(t2+t3);
}

__device__ __forceinline__ static float blend(bool mask,float a,float b)
{
    return mask?a:b;
}

__device__ __forceinline__ static float blend_x4(u2 lid,float a,float b,float c,float d)
{
    a=blend(lid%4==0,a,b);
    c=blend(lid%4==2,c,d);
    return blend(lid%4<2,a,c);
}
