#include<stdio.h>
#include<cassert>
#include<cuda_runtime.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cooperative_groups.h>
#include<cuda/pipeline>

#include"tool.h"
#include"ptx.h"
#include"type.h"
#include"barrier.h"

__global__ static void sgemm_v10_impl(const float* a,const __grid_constant__ CUtensorMap tensor_map_B, float* c, u2 N, u2 M, u2 K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const u2 bx=blockIdx.x;
    const u2 by=blockIdx.y;

    constexpr u2 as_size=16*128;
    constexpr u2 bs_size=16*128;

    __shared__ __align__(1024) float as[2][as_size];
    __shared__ __align__(1024) float bs[2][bs_size];

    __shared__ __align__(128) uint64_t full_B[2];

    barrier<1,384> full_A0;
    barrier<2,384> full_A1;
    barrier<3,384> empty0;
    barrier<4,384> empty1;

    if(threadIdx.x==0)
    {
        mbarrier_init(full_B[0]);
        mbarrier_init(full_B[1]);
    }
    __syncthreads();

    auto producer=[&]()
    {
        const u2 tid=threadIdx.x-256;
        float tmp_A[16];

        const float* LS_A=a+(tid/16*4+tid%4)*M +tid%16/4*4;
        const u2 STS_A_offset=(tid%4*4+tid%32/8)*128+(tid/32^tid%4)*8+tid/4%2*4;
        auto LD_A=[&](u2 stage,u2 offset)
        {
            *(float4*)(tmp_A+0)=*(float4*)(LS_A+0);
            *(float4*)(tmp_A+4)=*(float4*)(LS_A+32*M);
            *(float4*)(tmp_A+8)=*(float4*)(LS_A+64*M);
            *(float4*)(tmp_A+12)=*(float4*)(LS_A+96*M);
            LS_A+=32;
        };
        
        auto STS_A=[&](u2 stage)
        {
            *(float4*)(as[stage]+STS_A_offset)=*(float4*)(tmp_A+0);
            *(float4*)(as[stage]+STS_A_offset+32)=*(float4*)(tmp_A+4);
            *(float4*)(as[stage]+STS_A_offset+64)=*(float4*)(tmp_A+8);
            *(float4*)(as[stage]+STS_A_offset+96)=*(float4*)(tmp_A+12);
            if(stage==0)full_A0.arrive();
            else full_A1.arrive();
        };

        auto LD_B=[&](u2 stage,u2 offset)
        {
            if(tid==0)
            {
                mbarrier_arrive_expect_tx(full_B[stage], bs_size*4);
                tma_load(full_B[stage],bs[stage],tensor_map_B,bx*128,offset);
            }
        };

        LD_A(0,0);
        LD_B(0,0);
        STS_A(0);
        LD_A(1,32);
        LD_B(1,32);
        STS_A(1);

        for(u2 i=32;i<M;i+=32)
        {
            empty0.sync();
            LD_A(0,i);
            LD_B(0,i);
            STS_A(0);

            empty1.sync();
            LD_A(1,i+32);
            LD_B(1,i+32);
            STS_A(1);
        }
    };

    auto consumer=[&]()
    {
        const u2 tid=threadIdx.x;
        u2 phase[2]={0,0};

        auto WL=[&](u2 stage)
        {
            mbarrier_wait(full_B[stage],phase[stage]);
            phase[stage]^=1;
            if(stage==0)full_A0.sync();
            else full_A1.sync();
        };

        auto C=[&](u2 stage)
        {
            auto print=[&](float*p)
            {
                for(int i=0;i<16;i++)
                {
                    for(int j=0;j<32;j++)
                    {
                        printf("%f ",p[i*128+j]);
                        if(j%8==7)printf("\n");
                    }
                    printf("\n");
                }   
                printf("\n");
            };
            if(tid==0&&bx==0&&by==0)
            {
                print(as[stage]);
                print(bs[stage]);
            }
            __syncwarp();
            if(stage==0) empty0.arrive();
            else empty1.arrive();
        };

        for(int i=0;i<M;i+=32)
        {
            WL(0);
            C(0);
            WL(1);
            C(1);
        }
    };

    if(threadIdx.x<256)
    {
        consumer();
    }   
    else 
    {
        producer();
    }
#endif
}

void sgemm_v10(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
{
	assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

    CUtensorMap tma_desc_B;

    uint64_t globalDimB[2] = {(u2)k,(u2)m};
    uint64_t globalStrideB[1] = {k*sizeof(float)};
    uint32_t boxDimB[2] = {128,16};
    uint32_t elementStrideB[2] = {1,1};

    CUresult res = cuTensorMapEncodeTiled(
        &tma_desc_B,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,
        (void*)b,
        globalDimB,
        globalStrideB,
        boxDimB,
        elementStrideB,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );



	dim3 grid(k/128,n/128);
	dim3 block(384);
	sgemm_v10_impl<<<grid,block,0,stream>>>(a,tma_desc_B,c,n,m,k);
}
