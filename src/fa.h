#pragma once

#include<cuda_fp16.h>
#include "type.h"

void hello_fa();
void fa_cudnn(cudaStream_t stream, const half* q, const half* k, const half* v, half* o, u2 n, u2 heads);
void fa_v1(cudaStream_t stream, const half* q, const half* k, const half* v, half* o, u2 n, u2 heads);
