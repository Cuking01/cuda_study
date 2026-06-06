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
            LS_A+=16;
        };
        
        auto STS_A=[&](u2 stage)
        {
            stmatrix_x4(as[stage]+STS_A_offset,tmp_A+0);
            stmatrix_x4(as[stage]+STS_A_offset+32,tmp_A+4);
            stmatrix_x4(as[stage]+STS_A_offset+64,tmp_A+8);
            stmatrix_x4(as[stage]+STS_A_offset+96,tmp_A+12);
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
        LD_A(1,16);
        LD_B(1,16);
        STS_A(1);

        for(u2 i=32;i<M;i+=32)
        {
            empty0.sync();
            LD_A(0,i);
            LD_B(0,i);
            STS_A(0);

            empty1.sync();
            LD_A(1,i+16);
            LD_B(1,i+16);
            STS_A(1);
        }
    };

    auto consumer=[&]()
    {
        const u2 tid=threadIdx.x;
        u2 phase[2]={0,0};

        float ar[8],br[8];
        float cr[8][8]={0.0};

        auto WL=[&](u2 stage)
        {
            mbarrier_wait(full_B[stage],phase[stage]);
            phase[stage]^=1;
            if(stage==0)full_A0.sync();
            else full_A1.sync();
        };

        auto C=[&](u2 stage)
        {
            for(int i=0;i<16;i++)
            {
                *(float4*)(ar+0)=*(float4*)(as[stage]+i*128+(tid/32*2+tid%2^i/4)*8);
                *(float4*)(ar+4)=*(float4*)(as[stage]+i*128+(tid/32*2+tid%2^i/4)*8+4);
                *(float4*)(br+0)=*(float4*)(bs[stage]+i*128+tid%32/2*4);
                *(float4*)(br+4)=*(float4*)(bs[stage]+i*128+tid%32/2*4+64);

                #pragma unroll 8
                for(int j=0;j<8;j++)
                {
                    #pragma unroll 8
                    for(int k=0;k<8;k++)
                    {
                        cr[j][k]+=ar[j]*br[k];
                        // if(j==0&&k==0&&bx==0&&by==0&&tid==1)
                        // {
                        //     printf("ar[%d]=%f, br[%d]=%f\n",j,ar[j],k,br[k]);
                        // }
                    }
                        
                }
            }
            if(stage==0) empty0.arrive();
            else empty1.arrive();
        };

        auto WRITE_BACK=[&]()
        {
            float*const STG_C=c+(by*128ull+tid/32*16+tid%2*8)*K+bx*128ull+tid%32/2*4;
            #pragma unroll 8
            for(u3 i=0;i<8;i++)
            {
                *(float4*)(STG_C+i*K+0)=*(float4*)(cr[i]+0);
                *(float4*)(STG_C+i*K+64)=*(float4*)(cr[i]+4);
            }
            
        };

        for(int i=0;i<M;i+=32)
        {
            WL(0);
            C(0);
            WL(1);
            C(1);
        }
        WRITE_BACK();
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


__global__ static void sgemm_v11_impl(const float* a,const __grid_constant__ CUtensorMap tensor_map_B, float* c, u2 N, u2 M, u2 K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const u2 tid=threadIdx.x;
    const u2 bx=blockIdx.x;
    const u2 by=blockIdx.y;

    constexpr u2 as_size=16*128;
    constexpr u2 bs_size=16*128;

    __shared__ __align__(1024) float as[2][as_size];
    __shared__ __align__(1024) float bs[2][bs_size];

    __shared__ __align__(128) uint64_t full[2];

    if(tid==0)
    {
        mbarrier_init(full[0],257);
        mbarrier_init(full[1],257);
    }
    __syncthreads();

    float tmp_A[8];

    const float* LS_A=a+(tid/16*4+tid%4)*M +tid%16/4*4;
    const u2 STS_A_offset=(tid%4*4+tid%32/8)*128+(tid/32^tid%4)*8+tid/4%2*4;
    auto LD_A=[&](u2 stage,u2 offset)
    {
        *(float4*)(tmp_A+0)=*(float4*)(LS_A+0);
        *(float4*)(tmp_A+4)=*(float4*)(LS_A+64*M);
        LS_A+=16;
    };
        
    auto STS_A=[&](u2 stage)
    {
        stmatrix_x4(as[stage]+STS_A_offset,tmp_A+0);
        stmatrix_x4(as[stage]+STS_A_offset+64,tmp_A+4);
        mbarrier_arrive(full[stage]);
    };

    auto LD_B=[&](u2 stage,u2 offset)
    {
        if(tid==0)
        {
            mbarrier_arrive_expect_tx(full[stage], bs_size*4);
            tma_load(full[stage],bs[stage],tensor_map_B,bx*128,offset);
        }
    };
    
    u2 phase[2]={0,0};

    float ar[8],br[8];
    float cr[8][8]={0.0};

    auto WL=[&](u2 stage)
    {
        mbarrier_wait(full[stage],phase[stage]);
        phase[stage]^=1;
    };

    auto C=[&](u2 stage)
    {
        for(int i=0;i<4;i++)
        {
            const float* LS_as=as[stage]+i*512+(tid/32*2+tid%2^i)*8;
            const float* LS_bs=bs[stage]+i*512+tid%32/2*4;
            #pragma unroll 2
            for(int j=0;j<4;j++)
            {
                *(float4*)(ar+0)=*(float4*)(LS_as);
                *(float4*)(ar+4)=*(float4*)(LS_as+4);
                *(float4*)(br+0)=*(float4*)(LS_bs);
                *(float4*)(br+4)=*(float4*)(LS_bs+64);

                #pragma unroll 8
                for(int j=0;j<8;j++)
                {
                    #pragma unroll 8
                    for(int k=0;k<8;k++)
                        cr[j][k]+=ar[j]*br[k];
                }
                LS_as+=128;
                LS_bs+=128;
            }
        }
    };

    auto WRITE_BACK=[&]()
    {
        float* STG_C=c+(by*128ull+tid/32*16+tid%2*8)*K+bx*128ull+tid%32/2*4;
        #pragma unroll 8
        for(u3 i=0;i<8;i++)
        {
            *(float4*)(STG_C+0)=*(float4*)(cr[i]+0);
            *(float4*)(STG_C+64)=*(float4*)(cr[i]+4);
            STG_C+=K;
        }
        
    };

    LD_A(0,0);
    LD_B(0,0);
    STS_A(0);

    for(int i=0;i<M;i+=32)
    {
        LD_A(1,i+16);
        LD_B(1,i+16);
        WL(0);
        C(0);
        __syncthreads();
        
        STS_A(1);
        if(i<M-32)
        {
            LD_A(0,i+32);
            LD_B(0,i+32);
        }
        WL(1);
        C(1);
        __syncthreads();
        
        if(i<M-32)STS_A(0);
    }
    WRITE_BACK();
#endif
}

void sgemm_v11(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k)
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
	dim3 block(256);
	sgemm_v11_impl<<<grid,block,0,stream>>>(a,tma_desc_B,c,n,m,k);
}
