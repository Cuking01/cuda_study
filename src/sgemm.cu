
#include<stdio.h>
#include<cassert>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cuda_runtime_api.h>
#include<cooperative_groups.h>
#include<cuda/pipeline>

#include"tool.h"

__global__ static void hello_sgemm_impl()
{
	printf("Hello SGEMM!\n");
}

void hello_sgemm()
{
	hello_sgemm_impl<<<1,1>>>();
	cudaDeviceSynchronize();
}

__global__ static void nop_impl(int n,int*p)
{
	int x=0;
	for(int i=0;i<n;i++)
		x=p[x];
	p[0]=x;
}

void nop(cudaStream_t stream)
{
	std::vector<int> p{1,0};
	GPU_Data<int> p_gpu(p);

	nop_impl<<<1,1,0,stream>>>(100000,p_gpu);
}

__global__ static void sgemm_v1_impl(const float* a, const float* b, float* c, int n, int m, int k)
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

__global__ static void sgemm_v2_impl(const float* a, const float* b, float* c, int n, int m, int k)
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

__global__ static void sgemm_v3_impl(const float* a, const float* b, float* c, int N, int M, int K)
{
	const unsigned int tx=threadIdx.x;
	const unsigned int ty=threadIdx.y;
	
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

struct Block_4x32x32_8x8
{
	struct alignas(16) Block_32x32_8x8
	{

		float4 x[32][8];

		using T=float4(*)[8];
		__device__ operator T(){return x;}
	};

	Block_32x32_8x8 x[4];

	__device__ operator Block_32x32_8x8*()
	{
		return x;
	}
};

__global__ static void sgemm_v4_impl(const float* a,const float* b, float* c, int N, int M, int K)
{
	const unsigned int tx=threadIdx.x;
	const unsigned int ty=threadIdx.y;
	
	__shared__ Block_4x32x32_8x8 as;
	__shared__ Block_4x32x32_8x8 bs;
	
	float4 a_8x8[8][2];
	float4 b_8x8[8][2];
	float4 c_8x8[8][2];

	//把c初始化为0
	#pragma unroll 4
	for(int j=0;j<8;j++)
		c_8x8[j][0]=c_8x8[j][1]=make_float4(0,0,0,0);

	const float *a_local=a+blockIdx.y*128*M;
	const float *b_local=b+blockIdx.x*128;

	const unsigned int k128=K*128;

	for(int i=0;i<M;i+=128)
	{
		#pragma unroll 1
		for(int bstep_id=0;bstep_id<4;bstep_id++)
		{
			for(int j=0;j<4;j++)
			{
				as[ty/4][ty%4*8+tx/8*4+j][tx%8]=(tx%8)[(float4*)(a_local+bstep_id*32+(ty*8+tx/8*4+j)*M)];
				bs[ty/4][ty%4*8+tx/8*4+j][tx%8]=(tx%8)[(float4*)(b_local+(bstep_id*32+ty%4*8+tx/8*4+j)*K+ty/4*32)];
			}

			

			__syncthreads();

			#pragma unroll 1
			for(int l=0;l<4;l++)
			{	
				for(int j=0;j<8;j++)
				{
					a_8x8[j][0]=as[ty/4][ty%4*8+j][l*2+0];
					a_8x8[j][1]=as[ty/4][ty%4*8+j][l*2+1];
					b_8x8[j][0]=bs[tx/4][l*8+j][tx%4*2+0];
					b_8x8[j][1]=bs[tx/4][l*8+j][tx%4*2+1];
				}

				for(int j=0;j<8;j++)
				{
					c_8x8[j][0].x+=a_8x8[j][0].x*b_8x8[0][0].x;
					c_8x8[j][0].y+=a_8x8[j][0].x*b_8x8[0][0].y;
					c_8x8[j][0].z+=a_8x8[j][0].x*b_8x8[0][0].z;
					c_8x8[j][0].w+=a_8x8[j][0].x*b_8x8[0][0].w;
					c_8x8[j][1].x+=a_8x8[j][0].x*b_8x8[0][1].x;
					c_8x8[j][1].y+=a_8x8[j][0].x*b_8x8[0][1].y;
					c_8x8[j][1].z+=a_8x8[j][0].x*b_8x8[0][1].z;
					c_8x8[j][1].w+=a_8x8[j][0].x*b_8x8[0][1].w;

					c_8x8[j][0].x+=a_8x8[j][0].y*b_8x8[1][0].x;
					c_8x8[j][0].y+=a_8x8[j][0].y*b_8x8[1][0].y;
					c_8x8[j][0].z+=a_8x8[j][0].y*b_8x8[1][0].z;
					c_8x8[j][0].w+=a_8x8[j][0].y*b_8x8[1][0].w;
					c_8x8[j][1].x+=a_8x8[j][0].y*b_8x8[1][1].x;
					c_8x8[j][1].y+=a_8x8[j][0].y*b_8x8[1][1].y;
					c_8x8[j][1].z+=a_8x8[j][0].y*b_8x8[1][1].z;
					c_8x8[j][1].w+=a_8x8[j][0].y*b_8x8[1][1].w;

					c_8x8[j][0].x+=a_8x8[j][0].z*b_8x8[2][0].x;
					c_8x8[j][0].y+=a_8x8[j][0].z*b_8x8[2][0].y;
					c_8x8[j][0].z+=a_8x8[j][0].z*b_8x8[2][0].z;
					c_8x8[j][0].w+=a_8x8[j][0].z*b_8x8[2][0].w;
					c_8x8[j][1].x+=a_8x8[j][0].z*b_8x8[2][1].x;
					c_8x8[j][1].y+=a_8x8[j][0].z*b_8x8[2][1].y;
					c_8x8[j][1].z+=a_8x8[j][0].z*b_8x8[2][1].z;
					c_8x8[j][1].w+=a_8x8[j][0].z*b_8x8[2][1].w;

					c_8x8[j][0].x+=a_8x8[j][0].w*b_8x8[3][0].x;
					c_8x8[j][0].y+=a_8x8[j][0].w*b_8x8[3][0].y;
					c_8x8[j][0].z+=a_8x8[j][0].w*b_8x8[3][0].z;
					c_8x8[j][0].w+=a_8x8[j][0].w*b_8x8[3][0].w;
					c_8x8[j][1].x+=a_8x8[j][0].w*b_8x8[3][1].x;
					c_8x8[j][1].y+=a_8x8[j][0].w*b_8x8[3][1].y;
					c_8x8[j][1].z+=a_8x8[j][0].w*b_8x8[3][1].z;
					c_8x8[j][1].w+=a_8x8[j][0].w*b_8x8[3][1].w;

					c_8x8[j][0].x+=a_8x8[j][1].x*b_8x8[4][0].x;
					c_8x8[j][0].y+=a_8x8[j][1].x*b_8x8[4][0].y;
					c_8x8[j][0].z+=a_8x8[j][1].x*b_8x8[4][0].z;
					c_8x8[j][0].w+=a_8x8[j][1].x*b_8x8[4][0].w;
					c_8x8[j][1].x+=a_8x8[j][1].x*b_8x8[4][1].x;
					c_8x8[j][1].y+=a_8x8[j][1].x*b_8x8[4][1].y;
					c_8x8[j][1].z+=a_8x8[j][1].x*b_8x8[4][1].z;
					c_8x8[j][1].w+=a_8x8[j][1].x*b_8x8[4][1].w;

					c_8x8[j][0].x+=a_8x8[j][1].y*b_8x8[5][0].x;
					c_8x8[j][0].y+=a_8x8[j][1].y*b_8x8[5][0].y;
					c_8x8[j][0].z+=a_8x8[j][1].y*b_8x8[5][0].z;
					c_8x8[j][0].w+=a_8x8[j][1].y*b_8x8[5][0].w;
					c_8x8[j][1].x+=a_8x8[j][1].y*b_8x8[5][1].x;
					c_8x8[j][1].y+=a_8x8[j][1].y*b_8x8[5][1].y;
					c_8x8[j][1].z+=a_8x8[j][1].y*b_8x8[5][1].z;
					c_8x8[j][1].w+=a_8x8[j][1].y*b_8x8[5][1].w;

					c_8x8[j][0].x+=a_8x8[j][1].z*b_8x8[6][0].x;
					c_8x8[j][0].y+=a_8x8[j][1].z*b_8x8[6][0].y;
					c_8x8[j][0].z+=a_8x8[j][1].z*b_8x8[6][0].z;
					c_8x8[j][0].w+=a_8x8[j][1].z*b_8x8[6][0].w;
					c_8x8[j][1].x+=a_8x8[j][1].z*b_8x8[6][1].x;
					c_8x8[j][1].y+=a_8x8[j][1].z*b_8x8[6][1].y;
					c_8x8[j][1].z+=a_8x8[j][1].z*b_8x8[6][1].z;
					c_8x8[j][1].w+=a_8x8[j][1].z*b_8x8[6][1].w;

					c_8x8[j][0].x+=a_8x8[j][1].w*b_8x8[7][0].x;
					c_8x8[j][0].y+=a_8x8[j][1].w*b_8x8[7][0].y;
					c_8x8[j][0].z+=a_8x8[j][1].w*b_8x8[7][0].z;
					c_8x8[j][0].w+=a_8x8[j][1].w*b_8x8[7][0].w;
					c_8x8[j][1].x+=a_8x8[j][1].w*b_8x8[7][1].x;
					c_8x8[j][1].y+=a_8x8[j][1].w*b_8x8[7][1].y;
					c_8x8[j][1].z+=a_8x8[j][1].w*b_8x8[7][1].z;
					c_8x8[j][1].w+=a_8x8[j][1].w*b_8x8[7][1].w;
				}
			}

			__syncthreads();
		}
		
		a_local+=128;  
		b_local+=k128;

	}

	#pragma unroll 4
	for(int j=0;j<8;j++)
	{
		(tx*2+0)[(float4*)(c+(blockIdx.y*128+ty*8+j)*K+blockIdx.x*128)]=c_8x8[j][0];
		(tx*2+1)[(float4*)(c+(blockIdx.y*128+ty*8+j)*K+blockIdx.x*128)]=c_8x8[j][1];
	}
		
}

namespace sgemm_v5_impl
{

using u2=uint32_t;

__device__ __forceinline__ static void LS(
	cuda::pipeline<cuda::thread_scope_block>&pipe,
	const float* A_global,float* A_shared,u2 M,u2 M2,u2 M3,
	const float* B_global,float* B_shared,u2 K,u2 K2,u2 K3
)
{
	pipe.producer_acquire();
	cuda::memcpy_async(A_shared+0,A_global+0,cuda::aligned_size_t<16>(16),pipe);
	cuda::memcpy_async(A_shared+32,A_global+M,cuda::aligned_size_t<16>(16),pipe);
	cuda::memcpy_async(A_shared+64,A_global+M2,cuda::aligned_size_t<16>(16),pipe);
	cuda::memcpy_async(A_shared+96,A_global+M3,cuda::aligned_size_t<16>(16),pipe);
	cuda::memcpy_async(B_shared+0,B_global+0,cuda::aligned_size_t<16>(16),pipe);
	cuda::memcpy_async(B_shared+32,B_global+K,cuda::aligned_size_t<16>(16),pipe);
	cuda::memcpy_async(B_shared+64,B_global+K2,cuda::aligned_size_t<16>(16),pipe);
	cuda::memcpy_async(B_shared+96,B_global+K3,cuda::aligned_size_t<16>(16),pipe);
	pipe.producer_commit();
}

__device__ __forceinline__ static void LR(const float* as_local,float(*ar)[4],const float* bs_local,float(*br)[8])
{
	#pragma unroll 8
	for(int j=0;j<8;j++)
		*(float4*)ar[j]=*(float4*)(as_local+j*32);
	#pragma unroll 4
	for(int j=0;j<4;j++)
	{
		*(float4*)(br[j]+0)=*(float4*)(bs_local+j*32+0);
		*(float4*)(br[j]+4)=*(float4*)(bs_local+j*32+4);
	}
}

__device__ __forceinline__ static void CR(const float(*ar)[4],const float(*br)[8],float(*cr)[8])
{
	#pragma unroll 4
	for(int k=0;k<4;k++)
	{
		#pragma unroll 8
		for(int i=0;i<8;i++)
		{
			#pragma unroll 8
			for(int j=0;j<8;j++)
			{
				cr[i][j]+=ar[i][k]*br[k][j];
			}
		}
	}
}

__global__ static void v5_impl(const float* a,const float* b, float* c, u2 N, u2 M, u2 K)
{
	const u2 tx=threadIdx.x;
	const u2 ty=threadIdx.y;
	
	extern __shared__ __align__(16) float smem[];
	
	//128行32列，每8行填充4个
	const u2 as_size=128*32+16*4;
	const u2 bs_size=128*32;
	float (*as)[as_size]=(float(*)[as_size])(smem+0);
	float (*bs)[bs_size]=(float(*)[bs_size])(smem+as_size*2);
	
	__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> shared_state;
	auto block = cooperative_groups::this_thread_block();
	auto pipe = cuda::make_pipeline(block, &shared_state);

	float ar[2][8][4];
	float br[2][4][8];
	float cr[8][8];

	//把c初始化为0
	#pragma unroll 4
	for(int j=0;j<8;j++)
		0[(float4*)cr[j]]=1[(float4*)cr[j]]=make_float4(0,0,0,0);

	const float *a_local=a+blockIdx.y*128*M;
	const float *b_local=b+blockIdx.x*128;

	const u2 LS_M4=M*4;
	const u2 LS_M3=M*3;
	const u2 LS_M2=M*2;
	const u2 LS_A_Row4=tx/8+ty*2;
	const float* LS_a=a_local+LS_A_Row4*LS_M4+tx%8*4;
	float* as_cur=as[0];
	const float*LR_as;

	const u2 LS_K32=K*32;
	const u2 LS_K4=K*4;
	const u2 LS_K3=K*3;
	const u2 LS_K2=K*2;
	const u2 LS_B_Row4=tx/8+ty%4*2;
	const float* LS_b=b_local+LS_B_Row4*LS_K4+ty/4*32+tx%8*4;
	float* bs_cur=bs[0];
	const float*LR_bs;

	#define CALL_LS LS(pipe, LS_a,as_cur+LS_A_Row4*4*32+ty*4+tx%8*4,M,LS_M2,LS_M3, LS_b,bs_cur+LS_B_Row4*(4*32)+ty/4*32*32+tx%8*4,K,LS_K2,LS_K3); LS_a+=32; LS_b+=LS_K32;
	#define CALL_LR(stage) LR(LR_as,ar[stage],LR_bs,br[stage]); LR_as+=4; LR_bs+=4*32;
	#define CALL_CR(stage) CR(ar[stage],br[stage],cr);
		
	CALL_LS;
	as_cur=as[1];
	bs_cur=bs[1];
	
	for(int i=0;i<M;i+=32)
	{
		if(i<M-32) {CALL_LS;}
		
		as_cur=as[i/32&1];
		bs_cur=bs[i/32&1];

		pipe.consumer_wait();

		LR_as=as_cur+tx*(8*32+4);
		LR_bs=bs_cur+ty%4*8+ty/4*32*32;

		CALL_LR(0);

		#pragma unroll 1
		for(int j=0;j<3;j++)
		{
			CALL_LR(1);
			CALL_CR(0);
			CALL_LR(0);
			CALL_CR(1);
		}

		CALL_LR(1);
		CALL_CR(0);
		CALL_CR(1);
		
		pipe.consumer_release();
	}

	#undef CALL_LS
	#undef CALL_LR
	#undef CALL_CR

	for(int i=0;i<8;i++)
	{
		(ty*2+0)[(float4*)(c+(blockIdx.y*128+tx*8+i)*K+blockIdx.x*128)]=0[(float4*)cr[i]];
		(ty*2+1)[(float4*)(c+(blockIdx.y*128+tx*8+i)*K+blockIdx.x*128)]=1[(float4*)cr[i]];
	}
}

};

__device__ __forceinline__ static void copy_16B(void*dst,const void* src)
{
	uint32_t smem_int_ptr=__cvta_generic_to_shared(dst);

	asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
	                 :: "r"(smem_int_ptr),
	                 "l"(src),
	                 "n"(16));
}

__device__ __forceinline__ static void commit_group()
{
	asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ static void wait1()
{
	asm volatile("cp.async.wait_group %0;\n" :: "n"(1));
}

namespace sgemm_v6_impl
{

using u2=uint32_t;

__device__ __forceinline__ static void LS(
	const float* A_global,float* A_shared,u2 M,u2 M2,u2 M3,
	const float* B_global,float* B_shared,u2 K,u2 K2,u2 K3
)
{
	copy_16B(A_shared+0,A_global+0);
	copy_16B(A_shared+32,A_global+M);
	copy_16B(A_shared+64,A_global+M2);
	copy_16B(A_shared+96,A_global+M3);
	copy_16B(B_shared+0,B_global+0);
	copy_16B(B_shared+32,B_global+K);
	copy_16B(B_shared+64,B_global+K2);
	copy_16B(B_shared+96,B_global+K3);
	commit_group();
}

__device__ __forceinline__ static void LR(const float* as_local,float(*ar)[4],const float* bs_local,float(*br)[8])
{
	#pragma unroll 8
	for(int j=0;j<8;j++)
		*(float4*)ar[j]=*(float4*)(as_local+j*32);
	#pragma unroll 4
	for(int j=0;j<4;j++)
	{
		*(float4*)(br[j]+0)=*(float4*)(bs_local+j*32+0);
		*(float4*)(br[j]+4)=*(float4*)(bs_local+j*32+4);
	}
}

__device__ __forceinline__ static void CR(const float(*ar)[4],const float(*br)[8],float(*cr)[8])
{
	#pragma unroll 4
	for(int k=0;k<4;k++)
	{
		#pragma unroll 8
		for(int i=0;i<8;i++)
		{
			#pragma unroll 8
			for(int j=0;j<8;j++)
			{
				cr[i][j]+=ar[i][k]*br[k][j];
			}
		}
	}
}

__global__ static void v6_impl(const float* a,const float* b, float* c, u2 N, u2 M, u2 K)
{
	const u2 tx=threadIdx.x;
	const u2 ty=threadIdx.y;
	
	extern __shared__ __align__(16) float smem[];
	
	//128行32列，每8行填充4个
	const u2 as_size=128*32+16*4;
	const u2 bs_size=128*32;
	float (*as)[as_size]=(float(*)[as_size])(smem+0);
	float (*bs)[bs_size]=(float(*)[bs_size])(smem+as_size*2);

	float ar[2][8][4];
	float br[2][4][8];
	float cr[8][8];

	//把c初始化为0
	#pragma unroll 4
	for(int j=0;j<8;j++)
		0[(float4*)cr[j]]=1[(float4*)cr[j]]=make_float4(0,0,0,0);

	const float *a_local=a+blockIdx.y*128*M;
	const float *b_local=b+blockIdx.x*128;

	const u2 LS_M4=M*4;
	const u2 LS_M3=M*3;
	const u2 LS_M2=M*2;
	const u2 LS_A_Row4=tx/8+ty*2;
	const float* LS_a=a_local+LS_A_Row4*LS_M4+tx%8*4;
	float* as_cur=as[0];
	const float*LR_as;

	const u2 LS_K32=K*32;
	const u2 LS_K4=K*4;
	const u2 LS_K3=K*3;
	const u2 LS_K2=K*2;
	const u2 LS_B_Row4=tx/8+ty%4*2;
	const float* LS_b=b_local+LS_B_Row4*LS_K4+ty/4*32+tx%8*4;
	float* bs_cur=bs[0];
	const float*LR_bs;

	#define CALL_LS LS(LS_a,as_cur+LS_A_Row4*4*32+ty*4+tx%8*4,M,LS_M2,LS_M3, LS_b,bs_cur+LS_B_Row4*(4*32)+ty/4*32*32+tx%8*4,K,LS_K2,LS_K3); LS_a+=32; LS_b+=LS_K32;
	#define CALL_LR(stage) LR(LR_as,ar[stage],LR_bs,br[stage]); LR_as+=4; LR_bs+=4*32;
	#define CALL_CR(stage) CR(ar[stage],br[stage],cr);
		
	CALL_LS;
	as_cur=as[1];
	bs_cur=bs[1];
	
	for(int i=0;i<M;i+=32)
	{
		if(i<M-32) {CALL_LS;}
		
		as_cur=as[i/32&1];
		bs_cur=bs[i/32&1];

		LR_as=as_cur+tx*(8*32+4);
		LR_bs=bs_cur+ty%4*8+ty/4*32*32;

		wait1();
		__syncthreads();

		CALL_LR(0);

		#pragma unroll 1
		for(int j=0;j<3;j++)
		{
			CALL_LR(1);
			CALL_CR(0);
			CALL_LR(0);
			CALL_CR(1);
		}

		CALL_LR(1);
		CALL_CR(0);
		CALL_CR(1);

		__syncthreads();
	}

	for(int i=0;i<8;i++)
	{
		(ty*2+0)[(float4*)(c+(blockIdx.y*128+tx*8+i)*K+blockIdx.x*128)]=0[(float4*)cr[i]];
		(ty*2+1)[(float4*)(c+(blockIdx.y*128+tx*8+i)*K+blockIdx.x*128)]=1[(float4*)cr[i]];
	}

	#undef CALL_LS
	#undef CALL_LR
	#undef CALL_CR
}

};

namespace sgemm_v7_impl
{

using u2=uint32_t;

__device__ __forceinline__ static void LS(
	const float* A_global,float* A_shared,u2 M,u2 M2,u2 M3,
	const float* B_global,float* B_shared,u2 K,u2 K2,u2 K3
)
{
	copy_16B(A_shared+0,A_global+0);
	copy_16B(A_shared+32,A_global+M);
	copy_16B(A_shared+64,A_global+M2);
	copy_16B(A_shared+96,A_global+M3);
	copy_16B(B_shared+0,B_global+0);
	copy_16B(B_shared+32,B_global+K);
	copy_16B(B_shared+64,B_global+K2);
	copy_16B(B_shared+96,B_global+K3);
	commit_group();
}

__device__ __forceinline__ static void LR(const float* as_local,float(*ar)[4],const float* bs_local,float(*br)[8])
{
	#pragma unroll 8
	for(int j=0;j<8;j++)
		*(float4*)ar[j]=*(float4*)(as_local+j*32);
	#pragma unroll 4
	for(int j=0;j<4;j++)
	{
		*(float4*)(br[j]+0)=*(float4*)(bs_local+j*32+0);
		*(float4*)(br[j]+4)=*(float4*)(bs_local+j*32+4);
	}
}

__device__ __forceinline__ static void CR(const float(*ar)[4],const float(*br)[8],float(*cr)[8])
{
	#pragma unroll 4
	for(int k=0;k<4;k++)
	{
		#pragma unroll 8
		for(int i=0;i<8;i++)
		{
			#pragma unroll 8
			for(int j=0;j<8;j++)
			{
				cr[i][j]+=ar[i][k]*br[k][j];
			}
		}
	}
}

__global__ static void v7_impl(const float* a,const float* b, float* c, u2 N, u2 M, u2 K)
{
	const u2 tx=threadIdx.x;
	const u2 ty=threadIdx.y;
	
	extern __shared__ __align__(16) float smem[];
	
	//128行32列，每8行填充4个
	const u2 as_size=128*32+16*4;
	const u2 bs_size=128*32;
	float (*as)[as_size]=(float(*)[as_size])(smem+0);
	float (*bs)[bs_size]=(float(*)[bs_size])(smem+as_size*2);

	float ar[2][8][4];
	float br[2][4][8];
	float cr[8][8];

	//把c初始化为0
	#pragma unroll 4
	for(int j=0;j<8;j++)
		0[(float4*)cr[j]]=1[(float4*)cr[j]]=make_float4(0,0,0,0);

	const float *a_local=a+blockIdx.y*128*M;
	const float *b_local=b+blockIdx.x*128;

	const u2 LS_M4=M*4;
	const u2 LS_M3=M*3;
	const u2 LS_M2=M*2;
	const u2 LS_A_Row4=tx/8+ty*2;
	const float* LS_a=a_local+LS_A_Row4*LS_M4+tx%8*4;
	float* as_cur=as[0];
	const float*LR_as;

	const u2 LS_K32=K*32;
	const u2 LS_K4=K*4;
	const u2 LS_K3=K*3;
	const u2 LS_K2=K*2;
	const u2 LS_B_Row4=tx/8+ty%4*2;
	const float* LS_b=b_local+LS_B_Row4*LS_K4+ty/4*32+tx%8*4;
	float* bs_cur=bs[0];
	const float*LR_bs;

	#define CALL_LS LS(LS_a,as_cur+LS_A_Row4*4*32+ty*4+tx%8*4,M,LS_M2,LS_M3, LS_b,bs_cur+LS_B_Row4*(4*32)+ty/4*32*32+tx%8*4,K,LS_K2,LS_K3); LS_a+=32; LS_b+=LS_K32;
	#define CALL_LR(stage) LR(LR_as,ar[stage],LR_bs,br[stage]); LR_as+=4; LR_bs+=4*32;
	#define CALL_CR(stage) CR(ar[stage],br[stage],cr);
		
	CALL_LS;
	as_cur=as[1];
	bs_cur=bs[1];
	
	for(int i=0;i<M;i+=32)
	{
		if(i<M-32) {CALL_LS;}
		
		as_cur=as[i/32&1];
		bs_cur=bs[i/32&1];

		LR_as=as_cur+(tx/2+ty%2*8)*(8*32+4);
		LR_bs=bs_cur+ty%4/2*16+tx%2*8+ty/4*32*32;

		wait1();
		__syncthreads();

		CALL_LR(0);

		#pragma unroll 1
		for(int j=0;j<3;j++)
		{
			CALL_LR(1);
			CALL_CR(0);
			CALL_LR(0);
			CALL_CR(1);
		}

		CALL_LR(1);
		CALL_CR(0);
		CALL_CR(1);

		__syncthreads();
	}

	for(int i=0;i<8;i++)
	{
		(ty/2*4+tx%2*2+0)[(float4*)(c+(blockIdx.y*128+tx/2*8+ty%2*64+i)*K+blockIdx.x*128)]=0[(float4*)cr[i]];
		(ty/2*4+tx%2*2+1)[(float4*)(c+(blockIdx.y*128+tx/2*8+ty%2*64+i)*K+blockIdx.x*128)]=1[(float4*)cr[i]];
	}

	#undef CALL_LS
	#undef CALL_LR
	#undef CALL_CR
}

};

void sgemm_v1(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%32==0&&n%32==0&&k%32==0,"m,n,k must be divisible by 32");
	dim3 grid(k/32,n/32);
	dim3 block(32,32);
	sgemm_v1_impl<<<grid,block,0,stream>>>(a,b,c,n,m,k);
}

void sgemm_v2(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%32==0&&n%32==0&&k%32==0,"m,n,k must be divisible by 32");
	
	dim3 grid(k/32,n/32);
	dim3 block(8,8);
	sgemm_v2_impl<<<grid,block,0,stream>>>(a,b,c,n,m,k);
}

void sgemm_v3(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

	dim3 grid(k/128,n/128);
	dim3 block(32,32);
	sgemm_v3_impl<<<grid,block,0,stream>>>(a,b,c,n,m,k);
}

void sgemm_v4(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

	dim3 grid(k/128,n/128);
	dim3 block(16,16);
	sgemm_v4_impl<<<grid,block,0,stream>>>(a,b,c,n,m,k);

}

void sgemm_v5(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

	const unsigned int smem_size=64*1024+16*16+1024;
	cudaFuncSetAttribute(
    sgemm_v5_impl::v5_impl,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size);

	dim3 grid(k/128,n/128);
	dim3 block(16,16);
	sgemm_v5_impl::v5_impl<<<grid,block,smem_size,stream>>>(a,b,c,n,m,k);

	process_error();
}

void sgemm_v6(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

	const unsigned int smem_size=64*1024+16*16+1024;
	cudaFuncSetAttribute(
    sgemm_v6_impl::v6_impl,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size);

	dim3 grid(k/128,n/128);
	dim3 block(16,16);
	sgemm_v6_impl::v6_impl<<<grid,block,smem_size,stream>>>(a,b,c,n,m,k);

	process_error();
}

void sgemm_v7(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

	const unsigned int smem_size=64*1024+16*16+1024;
	cudaFuncSetAttribute(
    sgemm_v7_impl::v7_impl,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size);

	dim3 grid(k/128,n/128);
	dim3 block(16,16);
	sgemm_v7_impl::v7_impl<<<grid,block,smem_size,stream>>>(a,b,c,n,m,k);

	process_error();
}

void sgemm_cublas(cudaStream_t stream,const float* a, const float* b, float* c, int N, int M, int K)
{
	static cublasHandle_t handle=[&](){
		cublasHandle_t handle;
		cublasCreate(&handle);
		return handle;
	}();
	cublasSetStream(handle,stream);
	float alpha=1.0f;
	float beta=0.0f;
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,b,N,a,K,&beta,c,N);
}
