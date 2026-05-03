#include<cuda_fp16.h>

void hello_hgemm();
void hgemm_v1(cudaStream_t stream,const half* a, const half* b, half* c, int n, int m, int k);
void hgemm_cublas(cudaStream_t stream, const half* a, const half* b, half* c, int N, int M, int K);
