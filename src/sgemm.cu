
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

template<typename T>
__device__ __forceinline__ static void dev_swap(T& a,T& b)
{
	T tmp=a;
	a=b;
	b=tmp;
}

namespace sgemm_v7_impl
{

using u2=uint32_t;

__device__ __forceinline__ static void LS(
	const float* A_global,float4* A_t,u2 M,u2 M2,u2 M3,
	const float* B_global,float4* B_t,u2 K,u2 K2,u2 K3
)
{
	A_t[0]=*(float4*)(A_global+0);
	A_t[1]=*(float4*)(A_global+M);
	A_t[2]=*(float4*)(A_global+M2);
	A_t[3]=*(float4*)(A_global+M3);

	B_t[0]=*(float4*)(B_global+0);
	B_t[1]=*(float4*)(B_global+K);
	B_t[2]=*(float4*)(B_global+K2);
	B_t[3]=*(float4*)(B_global+K3);
	
	// copy_16B(A_shared+0,A_global+0);
	// copy_16B(A_shared+32,A_global+M);
	// copy_16B(A_shared+64,A_global+M2);
	// copy_16B(A_shared+96,A_global+M3);
	// copy_16B(B_shared+0,B_global+0);
	// copy_16B(B_shared+128,B_global+K);
	// copy_16B(B_shared+256,B_global+K2);
	// copy_16B(B_shared+384,B_global+K3);
	// commit_group();
}

__device__ __forceinline__ static void CR(float(*cr)[8],const float(*ar)[4],const float(*br)[8],int k)
{
	#pragma unroll 8
	for(int i=0;i<8;i++)
	{
		#pragma unroll 4
		for(int j=0;j<4;j++)
		{
			cr[i][(j+k%2*4)]+=ar[i][k/2]*br[k/2][j+k%2*4];
			// if(threadIdx.x==0&&threadIdx.y==0&&i==0&&j==0&&k%2==0)
			// {
			// 	printf("ar[%d][%d]=%f, br[%d][%d]=%f\n",i,k/2,ar[i][k/2],k/2,k%2*4,br[k/2][k%2*4]);
			// }
		}
			
	}
}

__device__ __forceinline__ static void LCR(const float* as_local0,const float* as_local1,float(*ar)[8][4],const float* bs_local0,const float* bs_local1,float(*br)[4][8],float(*cr)[8])
{
	#pragma unroll 8
	for(int k=0;k<8;k++)
	{
		*(float4*)ar[0][k]=*(float4*)(as_local0+k*32);
		*(float4*)(br[0][k/2]+k%2*4)=*(float4*)(bs_local0+k*64);

		CR(cr,ar[1],br[1],k);
	}
	
	#pragma unroll 8
	for(int k=0;k<8;k++)
	{
		*(float4*)ar[1][k]=*(float4*)(as_local1+k*32);
		*(float4*)(br[1][k/2]+k%2*4)=*(float4*)(bs_local1+k*64);

		CR(cr,ar[0],br[0],k);
	}
}

__global__ static void v7_impl(const float* a,const float* b, float* c, u2 N, u2 M, u2 K)
{
	const u2 tx=threadIdx.x;
	const u2 ty=threadIdx.y;
	const u2 tid=tx+ty*16;
	const u2 bid=blockIdx.y*blockDim.x+blockIdx.x;
	extern __shared__ __align__(128) float smem[];
	
	const u2 as_size=128*32;
	const u2 bs_size=32*128;
	float (*as)[as_size]=(float(*)[as_size])(smem+0);
	float (*bs)[bs_size]=(float(*)[bs_size])(smem+as_size*2);

	float4 at[4],bt[4];

	float ar[2][8][4]={0};
	float br[2][4][8]={0};
	float cr[8][8]={0};

	const float *a_local=a+blockIdx.y*128*M;
	const float *b_local=b+blockIdx.x*128;

	const u2 LS_M4=M*4;
	const u2 LS_M3=M*3;
	const u2 LS_M2=M*2;
	const float* LS_a=a_local+tid/8*LS_M4+(tid%8^ty%2)*4;
	float* as_cur=as[0];
	float* as_lst=as[1];
	const u2 LR_as_offset0=(tx%2*8+ty/2*16)*32+tx%2*4;
	const u2 LR_as_offset1=(tx%2*8+ty/2*16)*32+(1-tx%2)*4;

	const float* LR_as0_base_cur=as_cur+LR_as_offset0;
	const float* LR_as1_base_cur=as_cur+LR_as_offset1;
	const float* LR_as0_base_lst=as_lst+LR_as_offset0;
	const float* LR_as1_base_lst=as_lst+LR_as_offset1;

	const u2 LS_K32=K*32;
	const u2 LS_K4=K*4;
	const u2 LS_K3=K*3;
	const u2 LS_K2=K*2;
	const float* LS_b=b_local+tid/32*LS_K4+tid%32*4;
	float* bs_cur=bs[0];
	float* bs_lst=bs[1];
	const u2 LR_bs_offset0=tx/2*4+ty%2*32;//+tx%2*4*128;
	const u2 LR_bs_offset1=tx/2*4+ty%2*32+4*128;//+(1-tx%2)*4*128;

	const float* LR_bs0_base_cur=bs_cur+LR_bs_offset0;
	const float* LR_bs1_base_cur=bs_cur+LR_bs_offset1;
	const float* LR_bs0_base_lst=bs_lst+LR_bs_offset0;
	const float* LR_bs1_base_lst=bs_lst+LR_bs_offset1;


	#define CALL_LS LS(  \
		LS_a, at,  M,LS_M2,LS_M3,       \
		LS_b, bt, K,LS_K2,LS_K3);
	
	#define WRITE_SMEM   \
		do{   \
			float* const as_st=as_cur+(tid/8*(4*32)  +tid%8*4);  \
			float* const bs_st=bs_cur+(tid/32*(4*128)+tid%32*4); \
			*(float4*)(as_st+0)=*(float4*)(at+0);  \
			*(float4*)(as_st+32)=*(float4*)(at+1); \
			*(float4*)(as_st+64)=*(float4*)(at+2); \
			*(float4*)(as_st+96)=*(float4*)(at+3); \
                                                   \
			*(float4*)(bs_st+0)=*(float4*)(bt+0);  \
			*(float4*)(bs_st+128)=*(float4*)(bt+1);\
			*(float4*)(bs_st+256)=*(float4*)(bt+2);\
			*(float4*)(bs_st+384)=*(float4*)(bt+3);\
		}while(0)
		
	CALL_LS;
	WRITE_SMEM;
	dev_swap(as_cur,as_lst);
	dev_swap(bs_cur,bs_lst);
	
	__syncthreads();
	for(int i=0,LS_s=0;i<M;i+=32,LS_s^=1)
	{
		if(i<M-32) 
		{
			LS_a+=32;
			LS_b+=LS_K32;
			CALL_LS;
		}
		
		const float* LR_as0=LR_as0_base_cur;
		const float* LR_as1=LR_as1_base_cur;
		const float* LR_bs0=LR_bs0_base_cur;
		const float* LR_bs1=LR_bs1_base_cur;

		for(int j=0;j<4;j++)
		{                                                             
			LCR(LR_as0,LR_as1,ar,LR_bs0,LR_bs1,br,cr);
			LR_as0+=8;
			LR_bs0+=8*128;
			LR_as1+=8;
			LR_bs1+=8*128;
		}

		WRITE_SMEM;

		dev_swap(LR_as0_base_cur,LR_as0_base_lst);
		dev_swap(LR_as1_base_cur,LR_as1_base_lst);
		dev_swap(LR_bs0_base_cur,LR_bs0_base_lst);
		dev_swap(LR_bs1_base_cur,LR_bs1_base_lst);
		dev_swap(as_cur,as_lst);
		dev_swap(bs_cur,bs_lst);

		__syncthreads();
	}

	#pragma unroll 8
	for(int i=0;i<8;i++)
		CR(cr,ar[1],br[1],i);
	
	float* const WL_c=c+(blockIdx.y*128+ty/2*16+tx%2*8)*K+blockIdx.x*128+tx/2*4+ty%2*32;

	// #pragma unroll 8
	// for(int i=0;i<8;i++)
	// {
	// 	#pragma unroll
	// 	for(int j=0;j<8;j+=2)
	// 	{
	// 		float tmp=cr[i][j];
	// 		cr[i][j]=cr[i][j+1];
	// 		cr[i][j+1]=tmp;
	// 	}
	// }

	for(int i=0;i<8;i++)
	{
		*(float4*)(WL_c+i*K)=*(float4*)(cr[i]+0);
		*(float4*)(WL_c+i*K+64)=*(float4*)(cr[i]+4);
	}

	#undef CALL_LS
	#undef WRITE_SMEM
}

};

namespace sgemm_v8_impl
{

using u2=uint32_t;

__device__ __forceinline__ static void LS(
	const float* A_global,float4* A_t,u2 M,u2 M64,u2 M65,
	const float* B_global,float4* B_t,u2 K,u2 K2,u2 K3
)
{
	A_t[0]=*(float4*)(A_global+0);
	A_t[1]=*(float4*)(A_global+M);
	A_t[2]=*(float4*)(A_global+M64);
	A_t[3]=*(float4*)(A_global+M65);

	B_t[0]=*(float4*)(B_global+0);
	B_t[1]=*(float4*)(B_global+K);
	B_t[2]=*(float4*)(B_global+K2);
	B_t[3]=*(float4*)(B_global+K3);
}

__device__ __forceinline__ static void CR(float(*cr)[8],const float(*ar)[4],const float *br,int k)
{
	#pragma unroll 8
	for(int i=0;i<8;i++)
	{
		#pragma unroll 8
		for(int j=0;j<8;j++)
		{
			cr[i][j]+=ar[i][k]*br[j];
			// if(threadIdx.x==0&&threadIdx.y==0&&i==0&&j==0)
			// {
			// 	printf("ar[%d][%d]=%f, br[%d]=%f\n",i,k,ar[i][k],k,br[j]);
			// }
		}
			
	}
}

__device__ __forceinline__ static void LCR(const float* as_local0,const float* as_local1,float(*ar)[8][4],const float* bs_local0,const float* bs_local1,float(*br)[8],float(*cr)[8])
{
	#pragma unroll 4
	for(int k=0;k<4;k++)
	{
		*(float4*)ar[k<1][2*((k+3)%4)]=*(float4*)(as_local0+(k*64));
		*(float4*)ar[k<1][2*((k+3)%4)+1]=*(float4*)(as_local0+(k*64+32));

		*(float4*)(br[k%2]+0)=*(float4*)(bs_local0+(k*128));
		*(float4*)(br[k%2]+4)=*(float4*)(bs_local0+(k*128+64));

		CR(cr,ar[k>=1],br[(k+1)%2],(k+3)%4);
	}
	
	#pragma unroll 4
	for(int k=0;k<4;k++)
	{
		*(float4*)ar[k>=1][2*((k+3)%4)]=*(float4*)(as_local1+(k*64));
		*(float4*)ar[k>=1][2*((k+3)%4)+1]=*(float4*)(as_local1+(k*64+32));

		*(float4*)(br[k%2]+0)=*(float4*)(bs_local1+(k*128));
		*(float4*)(br[k%2]+4)=*(float4*)(bs_local1+(k*128+64));

		CR(cr,ar[k<1],br[(k+1)%2],(k+3)%4);
	}
}

__device__ __forceinline__ static void LCR_tail(const float* as_local0,const float* as_local1,float(*ar)[8][4],const float* bs_local0,const float* bs_local1,float(*br)[8],float(*cr)[8])
{
	#pragma unroll 4
	for(int k=0;k<4;k++)
	{
		if(k<1)
		{
			*(float4*)ar[k<1][2*((k+3)%4)]=*(float4*)(as_local0+(k*64));
			*(float4*)ar[k<1][2*((k+3)%4)+1]=*(float4*)(as_local0+(k*64+32));
		}

		*(float4*)(br[k%2]+0)=*(float4*)(bs_local0+(k*128));
		*(float4*)(br[k%2]+4)=*(float4*)(bs_local0+(k*128+64));

		CR(cr,ar[k>=1],br[(k+1)%2],(k+3)%4);
	}

	CR(cr,ar[1],br[1],3);
}

__global__ static void v8_impl(const float* a,const float* b, float* c, u2 N, u2 M, u2 K)
{
	const u2 tx=threadIdx.x;
	const u2 ty=threadIdx.y;
	const u2 tid=tx+ty*16;
	const u2 bid=blockIdx.y*blockDim.x+blockIdx.x;
	extern __shared__ __align__(128) float smem[];
	
	const u2 as_size=128*32;
	const u2 bs_size=32*128;

	float (*as)[as_size]=(float(*)[as_size])(smem);
	float (*bs)[bs_size]=(float(*)[bs_size])(smem+as_size*2);

	float4 at[4],bt[4];

	float ar[2][8][4]={0};
	float br[2][8]={0};
	float cr[8][8]={0};

	const float *a_local=a+blockIdx.y*128*M;
	const float *b_local=b+blockIdx.x*128;

	const u2 LS_M65=M*65;
	const u2 LS_M64=M*64;
	const u2 LS_M4=M*4;
	const u2 LS_M3=M*3;
	const u2 LS_M2=M*2;
	const float* LS_a=a_local+tid/8*LS_M2+tid%8*4;
	float* as_cur=as[0];
	float* as_lst=as[1];
	
	const u2 LS_as_offset=tid/8*(2*32) +((tid%8+(tid/8%4==3))^(tid/32%2))*4 +2*32 -(tid/8%4==3?8*32:0) +(tid%32==31?127*32:0);

	const u2 LR_as_offset0=(tx%2*8+ty/2*16)*32+tx%2*4;
	const u2 LR_as_offset1=(tx%2*8+ty/2*16)*32+(1-tx%2)*4;

	const float* LR_as0_base_cur=as_cur+LR_as_offset0;
	const float* LR_as1_base_cur=as_cur+LR_as_offset1;
	const float* LR_as0_base_lst=as_lst+LR_as_offset0;
	const float* LR_as1_base_lst=as_lst+LR_as_offset1;

	as_lst=as_lst-(tid%32==31?32*256:0);

	const u2 LS_K32=K*32;
	const u2 LS_K4=K*4;
	const u2 LS_K3=K*3;
	const u2 LS_K2=K*2;
	const float* LS_b=b_local+tid/32*LS_K4+tid%32*4;

	float* bs_cur=bs[0];
	float* bs_lst=bs[1];

	const u2 LR_bs_offset0=tx/2*4+ty%2*32;//+tx%2*4*128;
	const u2 LR_bs_offset1=tx/2*4+ty%2*32+4*128;//+(1-tx%2)*4*128;

	const float* LR_bs0_base_cur=bs_cur+LR_bs_offset0;
	const float* LR_bs1_base_cur=bs_cur+LR_bs_offset1;
	const float* LR_bs0_base_lst=bs_lst+LR_bs_offset0;
	const float* LR_bs1_base_lst=bs_lst+LR_bs_offset1;

	const u2 LS_bs_offset=tid/32*(4*128)+tid%32*4+4*128;

	bs_lst=bs_lst-(tid>=224?64*128:0);

	#define CALL_LS LS(  \
		LS_a, at,  M,LS_M64,LS_M65,       \
		LS_b, bt, K,LS_K2,LS_K3);
	
	#define WRITE_SMEM   \
		do{   \
			float* const as_st=as_cur+LS_as_offset;  \
			float* const bs_st=bs_cur+LS_bs_offset; \
			*(float4*)(as_st+0)=*(float4*)(at+0);  \
			*(float4*)(as_st+32)=*(float4*)(at+1); \
			*(float4*)(as_st+64*32)=*(float4*)(at+2); \
			*(float4*)(as_st+65*32)=*(float4*)(at+3); \
                                                   \
			*(float4*)(bs_st+0)=*(float4*)(bt+0);  \
			*(float4*)(bs_st+128)=*(float4*)(bt+1);\
			*(float4*)(bs_st+256)=*(float4*)(bt+2);\
			*(float4*)(bs_st+384)=*(float4*)(bt+3);\
		}while(0)
		
	if(tid<128)*(float4*)(bs[0]+tid*4)=make_float4(0,0,0,0);
	if(tid<32)*(float4*)(as[0]+(tx*8+ty)*32+tx%2*4)=make_float4(0,0,0,0);

	__syncthreads();

	CALL_LS;


	WRITE_SMEM;
	dev_swap(as_cur,as_lst);
	dev_swap(bs_cur,bs_lst);
	
	__syncthreads();

	for(int i=0;i<M;i+=32)
	{
		if(i<M-32) 
		{
			LS_a+=32;
			LS_b+=LS_K32;
			CALL_LS;
		}
		
		const float* LR_as0=LR_as0_base_cur;
		const float* LR_as1=LR_as1_base_cur;
		const float* LR_bs0=LR_bs0_base_cur;
		const float* LR_bs1=LR_bs1_base_cur;

		for(int j=0;j<4;j++)
		{                                                             
			LCR(LR_as0,LR_as1,ar,LR_bs0,LR_bs1,br,cr);
			LR_as0+=8;
			LR_bs0+=8*128;
			LR_as1+=8;
			LR_bs1+=8*128;
		}
		__syncthreads();

		WRITE_SMEM;

		dev_swap(LR_as0_base_cur,LR_as0_base_lst);
		dev_swap(LR_as1_base_cur,LR_as1_base_lst);
		dev_swap(LR_bs0_base_cur,LR_bs0_base_lst);
		dev_swap(LR_bs1_base_cur,LR_bs1_base_lst);
		dev_swap(as_cur,as_lst);
		dev_swap(bs_cur,bs_lst);
		
		__syncthreads();
	}

	LCR_tail(LR_as0_base_cur,LR_as1_base_cur,ar,LR_bs0_base_cur,LR_bs1_base_cur,br,cr);
	
	float* const WL_c=c+(blockIdx.y*128+ty/2*16+tx%2*8)*K+blockIdx.x*128+tx/2*4+ty%2*32;

	// #pragma unroll 8
	// for(int i=0;i<8;i++)
	// {
	// 	#pragma unroll
	// 	for(int j=0;j<8;j+=2)
	// 	{
	// 		float tmp=cr[i][j];
	// 		cr[i][j]=cr[i][j+1];
	// 		cr[i][j+1]=tmp;
	// 	}
	// }

	for(int i=0;i<8;i++)
	{
		*(float4*)(WL_c+i*K)=*(float4*)(cr[i]+0);
		*(float4*)(WL_c+i*K+64)=*(float4*)(cr[i]+4);
	}

	#undef CALL_LS
	#undef WRITE_SMEM
}

};


namespace sgemm_v9_impl
{
using u2=uint32_t;
__device__ __forceinline__ void stmatrix_x4(const u2* src,u2*dst)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	uint32_t smem_int_ptr=__cvta_generic_to_shared(dst);
	asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(smem_int_ptr),
                 "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
#endif
}

template<typename T>
__device__ __forceinline__ void swap(T& a,T& b)
{
	T tmp=a;
	a=b;
	b=tmp;
}

__device__ __forceinline__ void CR(float(*cr)[8],const float*ar,const float*br)
{
	#pragma unroll 8
	for(int i=0;i<8;i++)
	{
		#pragma unroll 8
		for(int j=0;j<8;j++)
		{
			cr[i][j]+=ar[i]*br[j];
			// if(threadIdx.x==0&&threadIdx.y==0&&i==0&&j==0)
			// {
			// 	printf("ar[%d]=%f br[%d]=%f\n",i,ar[i],j,br[j]);
			// }
		}
	}
}

__global__ void v9_impl(const float* a,const float* b, float* c, u2 N, u2 M, u2 K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	const u2 tx=threadIdx.x;
	const u2 ty=threadIdx.y;
	const u2 tid=tx+ty*16;

	extern __shared__ __align__(128) float smem[];
	
	const u2 as_size=128*(32+4);
	const u2 bs_size=32*128;

	float (*as)[as_size]=(float(*)[as_size])(smem);
	float (*bs)[bs_size]=(float(*)[bs_size])(smem+as_size*2);

	float4 at[4],bt[4];

	float ar[2][8]={0};
	float br[2][8]={0};
	float cr[8][8]={0};

	float* const WL_c=c+(blockIdx.y*128+ty/2*16+tx%2*8)*K+blockIdx.x*128+tx/2*4+ty%2*32;
	bool stage0=true;

	const float *a_local=a+blockIdx.y*128*M;
	const float *b_local=b+blockIdx.x*128;

	const u2 M32=M*32;
	const u2 M64=M*64;
	const u2 M96=M*96;

	const u2 K2=K*2;
	const u2 K3=K*3;
	const u2 K8=K*8;
	const u2 K16=K*16;
	const u2 K24=K*24;
	const u2 K32=K*32;

	const float* LS_a=a_local+(tid/32*4+tid%4)*M+tid%32/4*4;
	const u2 LS_as_offset=tid%32*36+tid/32*4;           
	const float* LS_as_cur=as[0]+LS_as_offset;
	const float* LR_as_base_cur=as[0]+tx%2*8+ty%4/2*16+ty/4*32*36;

	const float* LS_b=b_local+tid/32*K*4+tid%32*4;
	const u2 LS_bs_offset=tid/32*128+tid%32*4;
	const float* LS_bs_cur=bs[0]+LS_bs_offset;
	const float* LR_bs_base_cur=bs[0]+tx/2*4+ty%2*32;

	at[0]=*(float4*)(LS_a+0);
	at[1]=*(float4*)(LS_a+M32);
	at[2]=*(float4*)(LS_a+M64);
	at[3]=*(float4*)(LS_a+M96);

	bt[0]=*(float4*)(LS_b+0);
	bt[1]=*(float4*)(LS_b+K);
	bt[2]=*(float4*)(LS_b+K2);
	bt[3]=*(float4*)(LS_b+K3);

	stmatrix_x4((u2*)(at+0),(u2*)(LS_as_cur+0*36));
	stmatrix_x4((u2*)(at+1),(u2*)(LS_as_cur+32*36));
	stmatrix_x4((u2*)(at+2),(u2*)(LS_as_cur+64*36));
	stmatrix_x4((u2*)(at+3),(u2*)(LS_as_cur+96*36));

	*(float4*)(LS_bs_cur+0)=bt[0];
	*(float4*)(LS_bs_cur+8*128)=bt[1];
	*(float4*)(LS_bs_cur+16*128)=bt[2];
	*(float4*)(LS_bs_cur+24*128)=bt[3];

	LS_as_cur+=as_size;
	LS_bs_cur+=bs_size;
	LS_a+=32;
	LS_b+=K32;

	__syncthreads();
	for(int i=0;i<M;i+=32)
	{
		if(i<M-32) 
		{
			at[0]=*(float4*)(LS_a+0);
			at[1]=*(float4*)(LS_a+M32);
			at[2]=*(float4*)(LS_a+M64);
			at[3]=*(float4*)(LS_a+M96);

			bt[0]=*(float4*)(LS_b+0);
			bt[1]=*(float4*)(LS_b+K);
			bt[2]=*(float4*)(LS_b+K2);
			bt[3]=*(float4*)(LS_b+K3);
		}

		const float* LR_as_cur=LR_as_base_cur;
		const float* LR_bs_cur=LR_bs_base_cur;

		for(int j=0;j<32;j+=2)
		{
			*(float4*)(ar[0]+0)=*(float4*)(LR_as_cur+0);
			*(float4*)(ar[0]+4)=*(float4*)(LR_as_cur+4);
			*(float4*)(br[0]+0)=*(float4*)(LR_bs_cur+0);
			*(float4*)(br[0]+4)=*(float4*)(LR_bs_cur+64);

			CR(cr,ar[1],br[1]);

			*(float4*)(ar[1]+0)=*(float4*)(LR_as_cur+36);
			*(float4*)(ar[1]+4)=*(float4*)(LR_as_cur+40);
			*(float4*)(br[1]+0)=*(float4*)(LR_bs_cur+128);
			*(float4*)(br[1]+4)=*(float4*)(LR_bs_cur+192);

			CR(cr,ar[0],br[0]);

			LR_as_cur+=72;
			LR_bs_cur+=256;
		}

		stmatrix_x4((u2*)(at+0),(u2*)(LS_as_cur+0*32));
		stmatrix_x4((u2*)(at+1),(u2*)(LS_as_cur+32*36));
		stmatrix_x4((u2*)(at+2),(u2*)(LS_as_cur+64*36));
		stmatrix_x4((u2*)(at+3),(u2*)(LS_as_cur+96*36));

		*(float4*)(LS_bs_cur+0)=bt[0];
		*(float4*)(LS_bs_cur+8*128)=bt[1];
		*(float4*)(LS_bs_cur+16*128)=bt[2];
		*(float4*)(LS_bs_cur+24*128)=bt[3];

		if(stage0)LS_as_cur-=as_size;
		else LS_as_cur+=as_size;
		if(stage0)LR_as_base_cur+=as_size;
		else LR_as_base_cur-=as_size;

		if(stage0)LS_bs_cur-=bs_size;
		else LS_bs_cur+=bs_size;
		if(stage0)LR_bs_base_cur+=bs_size;
		else LR_bs_base_cur-=bs_size;
		
		LS_a+=32;
		LS_b+=K32;
		stage0=!stage0;
		__syncthreads();
	}

	CR(cr,ar[1],br[1]);

	// #pragma unroll 8
	// for(int i=0;i<8;i++)
	// {
	// 	#pragma unroll
	// 	for(int j=0;j<8;j+=2)
	// 	{
	// 		float tmp=cr[i][j];
	// 		cr[i][j]=cr[i][j+1];
	// 		cr[i][j+1]=tmp;
	// 	}
	// }

	for(int i=0;i<8;i++)
	{
		*(float4*)(WL_c+i*K)=*(float4*)(cr[i]+0);
		*(float4*)(WL_c+i*K+64)=*(float4*)(cr[i]+4);
	}

#endif
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

void sgemm_v8(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

	const unsigned int smem_size=64*1024+16*16+1024;
	cudaFuncSetAttribute(
    sgemm_v8_impl::v8_impl,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size);

	dim3 grid(k/128,n/128);
	dim3 block(16,16);
	sgemm_v8_impl::v8_impl<<<grid,block,smem_size,stream>>>(a,b,c,n,m,k);

	process_error();
}

void sgemm_v9(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{

	cudaDeviceProp prop;
    int device = 0;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 9) {
        printf("sgemm_v9 not supported on this device\n");
        return;
    }
	assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

	const unsigned int smem_size=68*1024+16*16+1024;
	cudaFuncSetAttribute(
    sgemm_v9_impl::v9_impl,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size);

	dim3 grid(k/128,n/128);
	dim3 block(16,16);
	sgemm_v9_impl::v9_impl<<<grid,block,smem_size,stream>>>(a,b,c,n,m,k);

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


#define SWIZZLE_A(x, y) ((y) ^ ((x >> 2) << 3))
#define FLOAT4(x) (float4&)x
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_bcf_swizzling_dbf_rw_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; 
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / 32;       // 0~8
    int load_b_col = (tid % 32) * 4; // 0,4,8,12,16,20,24,28...

    int t_row_in_warp = (lane_id / 16) * 8;
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base; // 0~3
    // int c_col_1 = c_col_base + 64;

    // double buffer
    __shared__ float As_T[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    float sum[TM][TN] = {0.f};

    const float *a_ptr = a + (by * BM + load_a_row) * k + load_a_col;
    // float *a_ptr_64 = a + (by * BM + load_a_row + 64) * k + load_a_col;
    const float *b_ptr = b + load_b_row * n + bx * BN + load_b_col;
    // float *b_ptr_8 = b + (load_b_row + 8) * n + bx * BN + load_b_col;

    float4 tmp_a0 = FLOAT4(a_ptr[0]);
    float4 tmp_a1 = FLOAT4(a_ptr[64 * k]);
    float4 tmp_b0 = FLOAT4(b_ptr[0]);
    float4 tmp_b1 = FLOAT4(b_ptr[8 * n]);

    As_T[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
    As_T[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
    As_T[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
    As_T[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

    As_T[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
    As_T[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
    As_T[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
    As_T[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

    FLOAT4(Bs[0][load_b_row][load_b_col]) = tmp_b0;
    FLOAT4(Bs[0][load_b_row + 8][load_b_col]) = tmp_b1;

    __syncthreads();

    int write_idx = 1;
    int read_idx = 0;
    for (int bk = BK; bk < k; bk += BK) {
        a_ptr += BK;
        b_ptr += BK * n;

        tmp_a0 = FLOAT4(a_ptr[0]);
        tmp_a1 = FLOAT4(a_ptr[64 * k]);
        tmp_b0 = FLOAT4(b_ptr[0]);
        tmp_b1 = FLOAT4(b_ptr[8 * n]);

#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }

        As_T[write_idx][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As_T[write_idx][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As_T[write_idx][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As_T[write_idx][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

        As_T[write_idx][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
        As_T[write_idx][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
        As_T[write_idx][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
        As_T[write_idx][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

        FLOAT4(Bs[write_idx][load_b_row][load_b_col]) = tmp_b0;
        FLOAT4(Bs[write_idx][load_b_row + 8][load_b_col]) = tmp_b1;

        __syncthreads();
        write_idx ^= 1;
        read_idx ^= 1;
    }
#pragma unroll
    for (int i = 0; i < BK; i++) {
        float reg_a[TM], reg_b[TN];

        FLOAT4(reg_a[0]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row)]);
        FLOAT4(reg_a[4]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

        FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
        FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

#pragma unroll
        for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < TN; ++n_idx) {
                sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0 + 64]) = FLOAT4(sum[i][4]);
    }
}


void sgemm_zhihu(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

	dim3 grid(k/128,n/128);
	dim3 block(256);
	sgemm_at_bcf_swizzling_dbf_rw_kernel<<<grid,block,0,stream>>>(a,b,c,n,k,m);

}
