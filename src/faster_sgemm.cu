#include<stdio.h>
#include<cassert>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cuda_runtime_api.h>
#include<cooperative_groups.h>
#include<cuda/pipeline>

#include"tool.h"
#include"ptx.h"
#include"type.h"
#include"barrier.h"

__global__ static void sgemm_v10_impl(const float* a,const __grid_constant__ CUtensorMap tensor_map_B, float* c, u2 N, u2 M, u2 K)
{
    u2 tid=threadIdx.x;
    u2 bid=blockIdx.x+gridDim.x*blockIdx.y;

    constexpr u2 as_size=16*128;
    constexpr u2 bs_size=16*128;

    __shared__ __align__(1024) float as[2][as_size];
    __shared__ __align__(1024) float bs[2][bs_size];

    __shared__ __align__(128) uint64_t full_B[2];

    barrier<1,384> full_A0;
    barrier<2,384> full_A1;
    barrier<3,384> empty0;
    barrier<4,384> empty1;

    if(tid==0)
    {
        mbarrier_init(full_B[0]);
        mbarrier_init(full_B[1]);
    }
    __syncthreads();

    auto producer=[&]()
    {
        float tmp_A[8];

        const float* LS_A=a+(tid/16*4+tid%4)*M +tid%16/4*4;
        float*const STS_A_offset=(tid%4*4+tid%32/8)*128+(tid/32^tid%4)*8+tid/4%2*4;
        auto LD_A=[&](u2 stage,u2 offset)
        {
            *(float4*)(tmp_A+0)=*(float4*)(LS_A+0);
            *(float4*)(tmp_A+4)=*(float4*)(LS_A+64*M);
            LS_A+=32;
        };
        
        auto STS_A=[&](u2 stage,u2 offset)
        {
            *(float4*)(as[stage]+STS_A_offset)=*(float4*)(tmp_A+0);
            *(float4*)(as[stage]+STS_A_offset+64)=*(float4*)(tmp_A+4);
            if(stage==0)full_A0.arrive();
            else full_A1.arrive();
        };

        auto LD_B=[&](u2 stage,u2 offset)
        {

        };

        LD_A(0,0);
        LD_B(0,0);

        LD_A(1,32);
        LD_B(1,32);

        for(u2 i=64;i<M;i+=64)
        {
            bar0.sync();
            LD_A(0,i);
            LD_B(0,i);

            bar1.sync();
            LD_A(1,i+32);
            LD_B(1,i+32);
        }
    };
}

void sgemm_v10(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

	dim3 grid(k/128,n/128);
	dim3 block(256);
	sgemm_v10_impl<<<grid,block,0,stream>>>(a,b,c,n,m,k);
}
