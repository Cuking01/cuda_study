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



__global__ static void fa_v1_impl(
    const half* q,
    const __grid_constant__ CUtensorMap tensor_map_K,
    const __grid_constant__ CUtensorMap tensor_map_V,
    const half* o,
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
            barrier_sync(0,256+32);
        };
        auto WCV=[&]()
        {
            barrier_sync(1,256+32);
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

        u2 phase_k=0,phase_v=0;
        half2 qr[8][4];

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

        auto load_qr=[&]()
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

        load_qr();

        for(u2 i=0;i<=task_id;i++)
        {
            TMA_WLK();
            barrier_arrive(0,256+32);
            TMA_WLV();
            barrier_arrive(1,256+32);
        }
        
    };

    if(tid<256)consumer();
    else producer();

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
	dim3 block(288);
	fa_v1_impl<<<grid,block,smem_size,stream>>>(q,desc_K.get(),desc_V.get(),o,n,heads);

	process_error();
}
