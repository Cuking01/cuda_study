#pragma once

#include<cuda_fp16.h>
#include<stdint.h>
using u2=uint32_t;

void hello_fa();
void fa_cudnn(cudaStream_t stream, const half* q, const half* k, const half* v, half* o, int n, int heads);
