#pragma once

__device__ static void print_8x8(const half* a)
{
    for(int i=0;i<32;i++)
    {
        if(threadIdx.x%32==i)
        {
            printf("%f ",__half2float(a[0]));
            printf("%f ",__half2float(a[1]));
            if(i%4==3)printf("\n");
        }
    }
    if(threadIdx.x%32==0)printf("\n");
}
