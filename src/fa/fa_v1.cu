#include<stdio.h>
#include<cassert>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cooperative_groups.h>
#include<cuda/pipeline>
#include<cuda_fp16.h>

#include "fa_tma.h"
#include "tool.h"
#include "ptx.h"
#include "type.h"
#include "debug.h"

__device__ __forceinline__ static void pack(float* dst, const float* src)
{
    *(half2*)dst=__floats2half2_rn(src[0],src[1]);
};

__global__ __launch_bounds__(384,1) static void fa_v1_impl(
    const half* q,
    const __grid_constant__ CUtensorMap tensor_map_K,
    const __grid_constant__ CUtensorMap tensor_map_V,
    half* o,
    u2 n,
    u2 heads
){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const u2 head_id=blockIdx.x;
    const u2 task_id=gridDim.y-blockIdx.y-1;
    const u2 tid=threadIdx.x;

    extern __shared__ __align__(1024) half smem[];
    half* const ks=smem;
    half* const vs=smem+128*128;
    half(*const ostmp)[8][256]=(half(*)[8][256])smem;
    __shared__ __align__(128) uint64_t full_k,full_v;

    if(tid==0)
    {
        mbarrier_init(full_k);
        mbarrier_init(full_v);
    }
    __syncthreads();

    auto producer=[&]()
    {
        auto TMA_LSK=[&](u2 offset,u2 hid)
        {
            if(tid==256)
            {
                mbarrier_arrive_expect_tx(full_k, 128*128*2);
                tma_load(full_k,ks,tensor_map_K,0,offset,hid,0);
            }
        };

        auto TMA_LSV=[&](u2 offset,u2 hid)
        {
            if(tid==256)
            {
                mbarrier_arrive_expect_tx(full_v, 128*128*2);
                tma_load(full_v,vs,tensor_map_V,0,offset,hid,0);
            }
        };

        auto WCK=[&]()
        {
            barrier_sync(0,384);
        };
        auto WCV=[&]()
        {
            barrier_sync(1,384);
        };

        TMA_LSK(0,head_id);
        TMA_LSV(0,head_id);

        for(u2 i=1;i<=task_id;i++)
        {
            WCK();
            TMA_LSK(i*128,head_id);
            WCV();
            TMA_LSV(i*128,head_id);
        }
    };

    auto consumer=[&]()
    {
        const u2 wid=tid/32;
        const u2 lid=tid%32;
        const half* const LR_ks=ks+(lid/16*8+lid%8)*64+(lid/8%2^lid%8)*8;
        const half* const LR_vs=vs+(lid/8%2*8+lid%8)*64+(lid/16^lid%8)*8;
        half* const ST_o=o+128ull*head_id*n+128ull*128*task_id+128ull*16*wid+128ull*(lid/8%2*8+lid%8)+lid/16*8;

        u2 phase_k=0,phase_v=0;
        float oreg[16][4]={0};
        float sr[16][4]={0};

        half2 qr[8][4];
        half2 kr[2][4];
        half2 (*const vr)[4]=kr;

        auto TMA_WLK=[&]()
        {
            mbarrier_wait(full_k,phase_k);
            phase_k^=1;
        };
        auto TMA_WLV=[&]()
        {
            mbarrier_wait(full_v,phase_v);
            phase_v^=1;
        };
        auto TMA_WL=[&](int kv)
        {
            if(kv==0)TMA_WLK();
            else TMA_WLV();
        };

        auto LOAD_QR=[&]()
        {
            const half*q_tlocal=q+(task_id*128u+wid*16u+lid/4)*128u+lid%4*2;
            for(int i=0;i<8;i++)
            {
                qr[i][0]=*(half2*)(q_tlocal+i*16u);
                qr[i][1]=*(half2*)(q_tlocal+i*16u+8u*128u);
                qr[i][2]=*(half2*)(q_tlocal+i*16u+8u);
                qr[i][3]=*(half2*)(q_tlocal+i*16u+8u*128u+8u);
            }
        };

        auto print=[&](half* p)
        {
            if(!(tid==0&&task_id==3&&head_id==0))return;

            for(int i=0;i<16;i++)
            {
                for(int j=0;j<128;j++)
                {
                    printf("%f ",__half2float(p[i*128+j]));
                    if(j%8==7)printf("\n");
                }
                printf("\n");
            }
            printf("\n");
        };

        auto LR_next=[&](int kv,int i,int j)
        {

            if(j==7)j=0,i++;
            if(i==8)i=0,kv^=1,TMA_WL(kv);
            if(kv==0)
                ldmatrix_x4(kr[j&1],LR_ks+((j/4*128+i*16)*64+(j%4*2)*8));
            else
                ldmatrix_x4_trans(vr[j&1],LR_vs+((i/4*128+j*16)*64+(i%4*2)*8));
        };

        auto CS=[&]<bool tail=false>()
        {
            #pragma unroll 8
            for(int i=0;i<8;i++)
            {
                #pragma unroll 4
                for(int j=0;j<4;j++)
                    sr[i*2+0][j]=sr[i*2+1][j]=0.0;
                #pragma unroll 8
                for(int j=0;j<8;j++)
                {
                    LR_next(0,i,j);
                    mma(sr[i*2+0],qr[j],kr[j&1]);
                    mma(sr[i*2+1],qr[j],kr[j&1]+2);
                }
                pack(sr[i]+0,sr[i*2+0]+0); pack(sr[i]+1,sr[i*2+0]+2);
                pack(sr[i]+2,sr[i*2+1]+0); pack(sr[i]+3,sr[i*2+1]+2);

            }
            barrier_arrive(0,384);
        };

        auto CO=[&]<bool tail=false>()
        {
            #pragma unroll 8
            for(int i=0;i<8;i++)
            {
                #pragma unroll 8
                for(int j=0;j<8;j++)
                {
                    LR_next(1,i,j);
                    mma(oreg[i*2+0],sr[j],vr[j&1]);
                    mma(oreg[i*2+1],sr[j],vr[j&1]+2);
                }
            }
            barrier_arrive(1,384);
        };

        auto WRITE_BACK=[&]()
        {
            #pragma unroll 8
            for(int i=0;i<8;i++)
            {
                pack(oreg[i]+0,oreg[i*2+0]+0); pack(oreg[i]+1,oreg[i*2+0]+2);
                pack(oreg[i]+2,oreg[i*2+1]+0); pack(oreg[i]+3,oreg[i*2+1]+2);

                stmatrix_x4(ostmp[wid][i]+lid*8,oreg[i]);
                *(float4*)(ST_o+i*16)=*(float4*)(ostmp[wid][i]+lid*8);
            }
        };

        LOAD_QR();

        LR_next(1,7,7);
        for(u2 i=0;i<task_id;i++)
        {
            CS();
            CO();
        }
        
        WRITE_BACK();
    };

    if(tid<256)
    {
        barrier_sync(2,384);
        set_reg_inc<240>();
        consumer();
    }
    else
    {
        set_reg_dec<24>();
        barrier_arrive(2,384);
        producer();
    }

#endif
}

void fa_v1(cudaStream_t stream, const half* q, const half* k, const half* v, half* o, u2 n, u2 heads)
{
    assert_throw(n%128==0,"n must be divisible by 128");

	const unsigned int smem_size=64*1024;
	cudaFuncSetAttribute(
    fa_v1_impl,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size);

    FA_TMA_Desc desc_K((half*)k,n,heads);
    FA_TMA_Desc desc_V((half*)v,n,heads);

	dim3 grid(heads,n/128);
	dim3 block(384);
	fa_v1_impl<<<grid,block,smem_size,stream>>>(q,desc_K.get(),desc_V.get(),o,n,heads);

	gpu_sync();
}
