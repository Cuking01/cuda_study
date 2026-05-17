#pragma once

#include<cuda_fp16.h>
#include<stdint.h>
using u2=uint32_t;
void hello_hgemm();
void hgemm_v1(cudaStream_t stream,const half* a, const half* b, half* c, u2 n, u2 m, u2 k);
void hgemm_v2(cudaStream_t stream,const half* a, const half* b, half* c, u2 n, u2 m, u2 k);
void hgemm_v3(cudaStream_t stream,const half* a, const half* b, half* c, u2 n, u2 m, u2 k);
void hgemm_v4(cudaStream_t stream,const half* a, const half* b, half* c, u2 n, u2 m, u2 k);
void hgemm_v5(cudaStream_t stream,const half* a, const half* b, half* c, u2 n, u2 m, u2 k);
void hgemm_cublas(cudaStream_t stream, const half* a, const half* b, half* c, u2 N, u2 M, u2 K);
