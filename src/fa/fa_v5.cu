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
#include "helper.h"
#include "barrier.h"

__global__ __launch_bounds__(384,1) static void fa_v5_impl(
    const half* q,
    const __grid_constant__ CUtensorMap tensor_map_K,
    const __grid_constant__ CUtensorMap tensor_map_V,
    half* o,
    u2 n
){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const u2 head_id=blockIdx.y;
    const u2 task_id=gridDim.x-blockIdx.x-1;
    const u2 tid=threadIdx.x;

    extern __shared__ __align__(1024) half smem[];
    half* const ks=smem;
    half* const vs=smem+128*128;
    half(*const ostmp)[8][256]=(half(*)[8][256])smem;
    __shared__ __align__(128) uint64_t full_k,full_v;

    barrier<0,384> bar_k;
    barrier<1,384> bar_v;

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
            bar_k.sync();
        };
        auto WCV=[&]()
        {
            bar_v.sync();
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
        const half* const LR_ks=ks+(lid/16*8+lid%8)*64;
        const half* const LR_vs=vs+(lid/8%2*8+lid%8)*64;
        half* const ST_o=o+128ull*head_id*n+128ull*128*task_id+128ull*16*wid+128ull*(lid/8%2*8+lid%8)+lid/16*8;

        u2 phase_k=0,phase_v=0;
        float oreg[16][4]={0};
        float sr[16][4]={0};

        half2 qr[8][4];
        half2 kr[2][4];
        half2 (*const vr)[4]=kr;

        float gmx[2]={-1e30,-1e30};
        float gsum[2]={0.0,0.0};

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
            const half*q_tlocal=q+((u3)head_id*n+task_id*128ull+wid*16ull+lid/4)*128u+lid%4*2;
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
            if(!(tid==0&&task_id==0&&head_id==0))return;

            for(int i=0;i<128;i++)
            {
                printf("%d:\n",i);
                for(int j=0;j<128;j++)
                {
                    printf("%f ",__half2float(p[i*128+j]));
                    if(j%8==7)printf("\n");
                    if(j%64==63)printf("******\n");
                }
                printf("\n");
            }
            printf("\n");
        };

        auto LR_next=[&](int kv,int i,int j)
        {
            j++;
            if(j==8)j=0,i++;
            if(i==8)i=0,kv^=1,TMA_WL(kv);
            if(kv==0)
                ldmatrix_x4(kr[j&1],LR_ks+((j/4*128+i*16)*64+(j%4*2+lid/8%2^lid%8)*8));
            else
                ldmatrix_x4_trans(vr[j&1],LR_vs+((i/4*128+j*16)*64+(i%4*2+lid/16^lid%8)*8));
        };

        auto CP=[&](bool tail=false)
        {
            constexpr float scale=0.127517430824598685;  //sqrt(1/128)/ln(2)
            float mx[2]={gmx[0],gmx[1]};
            float lmx[8][2];
            float sum[2]={0};

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

                #pragma unroll 4
                for(int j=0;j<4;j++)
                {
                    sr[i*2+0][j]*=scale;
                    sr[i*2+1][j]*=scale;
                    if(tail)
                    {
                        if(i*16+j%2+lid%4*2>wid*16+j/2*8+lid/4)sr[i*2+0][j]=-1e30;
                        if(i*16+8+j%2+lid%4*2>wid*16+j/2*8+lid/4)sr[i*2+1][j]=-1e30;
                    }
                }

                #pragma unroll 2
                for(int j=0;j<2;j++)
                {
                    mx[j]=fmaxf(mx[j],max_x4(sr[i*2+0]+j*2,sr[i*2+1]+j*2));
                    butterfly_max_x4(mx[j]);
                    lmx[i][j]=mx[j];
                }
                
                #pragma unroll 4
                for(int j=0;j<4;j++)
                {
                    sr[i*2+0][j]=ex2(sr[i*2+0][j]-mx[j/2]);
                    sr[i*2+1][j]=ex2(sr[i*2+1][j]-mx[j/2]);
                }

            }

            bar_k.arrive();

            float dm[2]={gmx[0]-mx[0],gmx[1]-mx[1]};
            float o_lst_scale[2];
            gmx[0]=mx[0];
            gmx[1]=mx[1];

            #pragma unroll 8
            for(int i=0;i<8;i++)
            {
                float scale[2];
                if(i<7)
                {
                    scale[0]=ex2(lmx[i][0]-mx[0]);
                    scale[1]=ex2(lmx[i][1]-mx[1]);
                }

                #pragma unroll 2
                for(int j=0;j<2;j++)
                {
                    #pragma unroll 4
                    for(int k=0;k<4;k++)
                    {
                        if(i<7)sr[i*2+j][k]*=scale[k/2];
                        sum[k/2]+=sr[i*2+j][k];
                    }
                }
            }

            #pragma unroll 2
            for(int j=0;j<2;j++)
            {
                butterfly_sum_x4(sum[j]);
                o_lst_scale[j]=ex2(dm[j]);
                gsum[j]=sum[j]+gsum[j]*o_lst_scale[j];
            }
            
            #pragma unroll 16
            for(int i=0;i<16;i++)
            {
                #pragma unroll 4
                for(int j=0;j<4;j++)
                    oreg[i][j]*=o_lst_scale[j/2];
                
                pack(sr[i/2]+(i%2)*2+0,sr[i]+0);
                pack(sr[i/2]+(i%2)*2+1,sr[i]+2);
            }
        };

        auto CO=[&](bool tail=false)
        {
            #pragma unroll 8
            for(int i=0;i<8;i++)
            {
                #pragma unroll 8
                for(int j=0;j<8;j++)
                {
                    if(!(tail&&i==7&&j==7))LR_next(1,i,j);
                    mma(oreg[i*2+0],sr[j],vr[j&1]);
                    mma(oreg[i*2+1],sr[j],vr[j&1]+2);
                }
            }
            bar_v.arrive();
        };

        auto WRITE_BACK=[&]()
        {
            float o_scale[2]={1.0/gsum[0],1.0/gsum[1]};

            #pragma unroll 16
            for(int i=0;i<16;i++)
            {
                #pragma unroll 4
                for(int j=0;j<4;j++)
                    oreg[i][j]*=o_scale[j/2];
            }

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
            CP();
            CO();
        }

        CP(true);
        CO(true);
        barrier_sync(3,256);
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

void fa_v5(cudaStream_t stream, const half* q, const half* k, const half* v, half* o, u2 n, u2 heads)
{
    assert_throw(n%128==0,"n must be divisible by 128");

	const unsigned int smem_size=64*1024;
	cudaFuncSetAttribute(
    fa_v5_impl,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size);

    FA_TMA_Desc desc_K((half*)k,n,heads);
    FA_TMA_Desc desc_V((half*)v,n,heads);

	dim3 grid(n/128,heads);
	dim3 block(384);
	fa_v5_impl<<<grid,block,smem_size,stream>>>(q,desc_K.get(),desc_V.get(),o,n);

	gpu_sync();
}
