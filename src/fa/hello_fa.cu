#include<stdio.h>

__global__ void hello_fa_impl()
{
    printf("hello flash attention\n");
}

void hello_fa()
{
    hello_fa_impl<<<1,1>>>();
}

