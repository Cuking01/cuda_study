#pragma once

#include<cuda_runtime.h>

void hello_sgemm();
void nop(cudaStream_t stream);

void sgemm_v1(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k);
void sgemm_v2(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k);
void sgemm_v3(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k);
void sgemm_v4(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k);
void sgemm_cublas(cudaStream_t stream,const float* a, const float* b, float* c, int N, int M, int K);
