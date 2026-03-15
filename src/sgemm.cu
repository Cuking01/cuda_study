
#include<stdio.h>
#include<cassert>
#include<cuda_runtime.h>
#include "tool.h"

__global__ static void hello_sgemm_impl()
{
	printf("Hello SGEMM!\n");
}

void hello_sgemm()
{
	hello_sgemm_impl<<<1,1>>>();
	cudaDeviceSynchronize();
}

__global__ static void sgemm_v1_impl(float* a, float* b, float* c, int n, int m, int k)
{
	const int tx=threadIdx.x;
	const int ty=threadIdx.y;
	
	__shared__ float as[32][33];
	__shared__ float bs[32][33];
	
	float tmp=0;

	const float *a_local=a+blockIdx.y*32*m;
	const float *b_local=b+blockIdx.x*32;

	const int k32=k*32;

	for(int i=0;i<m;i+=32)
	{
		as[ty][tx]=a_local[ty*m+tx];
		bs[ty][tx]=b_local[ty*k+tx];

		a_local+=32;
		b_local+=k32;
		__syncthreads();

		#pragma unroll 2
		for(int j=0;j<32;j++)
			tmp+=as[ty][j]*bs[j][tx];
		
		__syncthreads();
	}

	c[(blockIdx.y*32+ty)*k+blockIdx.x*32+tx]=tmp;
}

__global__ static void sgemm_v2_impl(float* a, float* b, float* c, int n, int m, int k)
{
	const int tx=threadIdx.x;
	const int ty=threadIdx.y;
	
	__shared__ float as[32][33];
	__shared__ float bs[32][33];
	
	float tmp=0;

	const float *a_local=a+blockIdx.y*32*m;
	const float *b_local=b+blockIdx.x*32;

	const int k32=k*32;

	for(int i=0;i<m;i+=32)
	{
		as[ty][tx]=a_local[ty*m+tx];
		bs[ty][tx]=b_local[ty*k+tx];

		a_local+=32;
		b_local+=k32;
		__syncthreads();

		#pragma unroll 2
		for(int j=0;j<32;j++)
		{
			tmp+=as[ty][j]*bs[j][tx];
			__syncthreads();
		}
			

		// __syncthreads();

		// #pragma unroll 2
		// for(int j=16;j<32;j++)
		// 	tmp+=as[ty][j]*bs[j][tx];
		
		__syncthreads();
	}

	c[(blockIdx.y*32+ty)*k+blockIdx.x*32+tx]=tmp;
}

template<auto sgemm_impl>
void sgemm_interface(const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%32==0&&n%32==0&&k%32==0,"m,n,k must be divisible by 32");
	
	GPU_Data<float> gpu_a(a,n*m), gpu_b(b,m*k), gpu_c(n*k);

	dim3 grid(k/32,n/32);
	dim3 block(32,32);
	sgemm_impl<<<grid,block>>>(gpu_a,gpu_b,gpu_c,n,m,k);
	
	gpu_c.to_host(c);
	gpu_sync();
}

void sgemm_v1(const float* a, const float* b, float* c, int n, int m, int k)
{
	sgemm_interface<sgemm_v1_impl>(a,b,c,n,m,k);
}

void sgemm_v2(const float* a, const float* b, float* c, int n, int m, int k)
{
	sgemm_interface<sgemm_v2_impl>(a,b,c,n,m,k);
}
