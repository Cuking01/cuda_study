
#include<stdio.h>
#include<cassert>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cuda_runtime_api.h>
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

struct Block_32x32_4x4
{
	struct alignas(16) Block_4x32
	{
		float x[4][32];
		float _[4];  //用于错开板块

		using T=float(*)[32];
		__device__ operator T()
		{
			return x;
		}
	};

	Block_4x32 x[8];

	__device__ Block_4x32& operator[](int i)
	{
		return x[i];
	}
};

__global__ static void sgemm_v2_impl(float* a, float* b, float* c, int n, int m, int k)
{
	const int tx=threadIdx.x;
	const int ty=threadIdx.y;
	
	__shared__ Block_32x32_4x4 as;
	__shared__ Block_32x32_4x4 bs;
	
	float4 a_4x4[4];
	float4 b_4x4[4];
	float4 c_4x4[4];

	//把c初始化为0
	#pragma unroll 4
	for(int j=0;j<4;j++)
		c_4x4[j]=make_float4(0,0,0,0);

	const float *a_local=a+blockIdx.y*32*m;
	const float *b_local=b+blockIdx.x*32;

	const int k32=k*32;

	for(int i=0;i<m;i+=32)
	{
		#pragma unroll 4
		for(int j=0;j<4;j++)
		{
			tx[(float4*)as[ty][j]]=tx[(float4*)(a_local+(ty*4+j)*m)];
			tx[(float4*)bs[ty][j]]=tx[(float4*)(b_local+(ty*4+j)*k)];
		}
		a_local+=32;
		b_local+=k32;
		__syncthreads();

		for(int l=0;l<8;l++)
		{	
			for(int j=0;j<4;j++)
			{
				a_4x4[j]=l[(float4*)as[ty][j]];
				b_4x4[j]=tx[(float4*)bs[l][j]];
			}
			
			#pragma unroll 4
			for(int j=0;j<4;j++)
			{
				c_4x4[j].x+=a_4x4[j].x*b_4x4[0].x;
				c_4x4[j].y+=a_4x4[j].x*b_4x4[0].y;
				c_4x4[j].z+=a_4x4[j].x*b_4x4[0].z;
				c_4x4[j].w+=a_4x4[j].x*b_4x4[0].w;

				c_4x4[j].x+=a_4x4[j].y*b_4x4[1].x;
				c_4x4[j].y+=a_4x4[j].y*b_4x4[1].y;
				c_4x4[j].z+=a_4x4[j].y*b_4x4[1].z;
				c_4x4[j].w+=a_4x4[j].y*b_4x4[1].w;

				c_4x4[j].x+=a_4x4[j].z*b_4x4[2].x;
				c_4x4[j].y+=a_4x4[j].z*b_4x4[2].y;
				c_4x4[j].z+=a_4x4[j].z*b_4x4[2].z;
				c_4x4[j].w+=a_4x4[j].z*b_4x4[2].w;

				c_4x4[j].x+=a_4x4[j].w*b_4x4[3].x;
				c_4x4[j].y+=a_4x4[j].w*b_4x4[3].y;
				c_4x4[j].z+=a_4x4[j].w*b_4x4[3].z;
				c_4x4[j].w+=a_4x4[j].w*b_4x4[3].w;
			}
		}

		__syncthreads();
	}

	#pragma unroll 4
	for(int j=0;j<4;j++)
		tx[(float4*)(c+(blockIdx.y*32+ty*4+j)*k+blockIdx.x*32)]=c_4x4[j];
}

struct Block_4x32x32_4x4
{
	struct alignas(16) Block_32x32_4x4
	{

		struct alignas(16) Block_4x4
		{
			float x[4][4];
			__device__ operator float4*(){return (float4*)x;}
		};

		Block_4x4 x[8][8];

		using T=Block_4x4(*)[8];
		__device__ operator T(){return x;}
	};

	Block_32x32_4x4 x[4];

	__device__ operator Block_32x32_4x4*()
	{
		return x;
	}
};

__global__ static void sgemm_v3_impl(float* a, float* b, float* c, int N, int M, int K)
{
	const int tx=threadIdx.x;
	const int ty=threadIdx.y;
	
	__shared__ Block_4x32x32_4x4 as;
	__shared__ Block_4x32x32_4x4 bs;
	
	float4 a_4x4[4];
	float4 b_4x4[4];
	float4 c_4x4[4];

	//把c初始化为0
	#pragma unroll 4
	for(int j=0;j<4;j++)
		c_4x4[j]=make_float4(0,0,0,0);

	const float *a_local=a+blockIdx.y*128*M;
	const float *b_local=b+blockIdx.x*128;

	const int k128=K*128;

	for(int i=0;i<M;i+=128)
	{
		#pragma unroll 1
		for(int bstep_id=0;bstep_id<4;bstep_id++)
		{
			as[ty/8][ty%8][tx%8][tx/8]=(tx%8)[(float4*)(a_local+bstep_id*32+(ty*4+tx/8)*M)];
			bs[ty/8][ty%8][tx%8][tx/8]=(tx%8)[(float4*)(b_local+(bstep_id*32+ty%8*4+tx/8)*K+ty/8*32)];

			__syncthreads();

			#pragma unroll 1
			for(int l=0;l<8;l++)
			{	
				for(int j=0;j<4;j++)
				{
					a_4x4[j]=as[ty/8][ty%8][l][j];
					b_4x4[j]=bs[tx/8][l][tx%8][j];
				}

				for(int j=0;j<4;j++)
				{
					c_4x4[j].x+=a_4x4[j].x*b_4x4[0].x;
					c_4x4[j].y+=a_4x4[j].x*b_4x4[0].y;
					c_4x4[j].z+=a_4x4[j].x*b_4x4[0].z;
					c_4x4[j].w+=a_4x4[j].x*b_4x4[0].w;

					c_4x4[j].x+=a_4x4[j].y*b_4x4[1].x;
					c_4x4[j].y+=a_4x4[j].y*b_4x4[1].y;
					c_4x4[j].z+=a_4x4[j].y*b_4x4[1].z;
					c_4x4[j].w+=a_4x4[j].y*b_4x4[1].w;

					c_4x4[j].x+=a_4x4[j].z*b_4x4[2].x;
					c_4x4[j].y+=a_4x4[j].z*b_4x4[2].y;
					c_4x4[j].z+=a_4x4[j].z*b_4x4[2].z;
					c_4x4[j].w+=a_4x4[j].z*b_4x4[2].w;

					c_4x4[j].x+=a_4x4[j].w*b_4x4[3].x;
					c_4x4[j].y+=a_4x4[j].w*b_4x4[3].y;
					c_4x4[j].z+=a_4x4[j].w*b_4x4[3].z;
					c_4x4[j].w+=a_4x4[j].w*b_4x4[3].w;
				}
			}

			__syncthreads();
		}
		
		a_local+=128;  
		b_local+=k128;

	}

	#pragma unroll 4
	for(int j=0;j<4;j++)
		tx[(float4*)(c+(blockIdx.y*128+ty*4+j)*K+blockIdx.x*128)]=c_4x4[j];
}

template<auto sgemm_impl,int block_size=32>
void sgemm_interface(const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%block_size==0&&n%block_size==0&&k%block_size==0,"m,n,k must be divisible by block_size");
	
	GPU_Data<float> gpu_a(a,n*m), gpu_b(b,m*k), gpu_c(n*k);

	dim3 grid(k/block_size,n/block_size);
	dim3 block(block_size,block_size);
	sgemm_impl<<<grid,block>>>(gpu_a,gpu_b,gpu_c,n,m,k);
	
	gpu_c.to_host(c);
	gpu_sync();
}

void sgemm_v1(const float* a, const float* b, float* c, int n, int m, int k)
{
	sgemm_interface<sgemm_v1_impl>(a,b,c,n,m,k);
}

// void sgemm_v2(const float* a, const float* b, float* c, int n, int m, int k)
// {
// 	assert_throw(m%32==0&&n%32==0&&k%32==0,"m,n,k must be divisible by 32");
	
// 	GPU_Data<float> gpu_a(a,n*m), gpu_b(b,m*k), gpu_c(n*k);

// 	dim3 grid(k/32,n/32);
// 	dim3 block(8,8);
// 	sgemm_v2_impl<<<grid,block>>>(gpu_a,gpu_b,gpu_c,n,m,k);
	
// 	gpu_c.to_host(c);
// 	gpu_sync();
// }

void sgemm_v2(const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%32==0&&n%32==0&&k%32==0,"m,n,k must be divisible by 32");
	
	GPU_Data<float> gpu_a(a,n*m), gpu_b(b,m*k), gpu_c(n*k);

	dim3 grid(k/32,n/32);
	dim3 block(8,8);
	sgemm_v2_impl<<<grid,block>>>(gpu_a,gpu_b,gpu_c,n,m,k);
	
	gpu_c.to_host(c);
	gpu_sync();
}

void sgemm_v3(const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");
	
	GPU_Data<float> gpu_a(a,n*m), gpu_b(b,m*k), gpu_c(n*k);

	dim3 grid(k/128,n/128);
	dim3 block(32,32);
	sgemm_v3_impl<<<grid,block>>>(gpu_a,gpu_b,gpu_c,n,m,k);
	
	gpu_c.to_host(c);
	gpu_sync();
}

void sgemm_cublas(const float* a, const float* b, float* c, int N, int M, int K)
{
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cublasSetStream(handle,stream);
	float alpha=1.0f;
	float beta=0.0f;
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,b,N,a,K,&beta,c,N);
	cudaStreamSynchronize(stream);
	cublasDestroy(handle);
	cudaStreamDestroy(stream);
}
