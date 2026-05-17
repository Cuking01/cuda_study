
#include "fa/fa_cudnn.cu"
#include <stdio.h>

__global__ void hello_fa_impl()
{
    printf("hello fa\n");
}

void hello_fa()
{
    hello_fa_impl<<<1,1>>>();
}

