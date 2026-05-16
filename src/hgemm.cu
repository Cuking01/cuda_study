
#include<stdio.h>
#include<cassert>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cooperative_groups.h>
#include<cuda/pipeline>
#include<cuda_fp16.h>

#include"tool.h"

__global__ static void hello_hgemm_impl()
{
	printf("Hello HGEMM!\n");
}

void hello_hgemm()
{
	hello_hgemm_impl<<<1,1>>>();
	gpu_sync();
}

using u2=uint32_t;
using half8=uint4;

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
__device__ __forceinline__ static void wait_all()
{
	asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ static void barrier_sync(u2 bar,u2 arrive_count)
{
	asm volatile("barrier.sync %0, %1;\n" :: "r"(bar), "r"(arrive_count) : "memory");
}

__device__ __forceinline__ static void barrier_arrive(u2 bar,u2 arrive_count)
{
	asm volatile("barrier.arrive %0, %1;\n" :: "r"(bar), "r"(arrive_count) : "memory");
}

__device__ __forceinline__ static void ldmatrix_x4(void*dst,const void*src)
{
    u2 src_smem = __cvta_generic_to_shared(src);
    u2* dst_u2=(u2*)dst;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
               "{%0, %1, %2, %3}, [%4];"
               : "=r"(dst_u2[0]), "=r"(dst_u2[1]), "=r"(dst_u2[2]),
                 "=r"(dst_u2[3])
               : "r"(src_smem));
}

__device__ __forceinline__ static void ldmatrix_x4_trans(void*dst,const void*src)
{
    u2 src_smem = __cvta_generic_to_shared(src);
    u2* dst_u2=(u2*)dst;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.trans.b16 "
               "{%0, %1, %2, %3}, [%4];"
               : "=r"(dst_u2[0]), "=r"(dst_u2[1]), "=r"(dst_u2[2]),
                 "=r"(dst_u2[3])
               : "r"(src_smem));
}

__device__ __forceinline__ static void stmatrix_x4(void*dst,const void*src)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    u2 dst_smem = __cvta_generic_to_shared(dst);
    const u2* src_u2=(const u2*)src;
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(dst_smem),
                 "r"(src_u2[0]), "r"(src_u2[1]), "r"(src_u2[2]), "r"(src_u2[3]));
#endif
}

__device__ __forceinline__ static void mma(void* cr,const void*ar,const void*br)
{
    float* cr_f=(float*)cr;
    u2* ar_u2=(u2*)ar;
    u2* br_u2=(u2*)br;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(cr_f[0]), "+f"(cr_f[1]), "+f"(cr_f[2]),
                   "+f"(cr_f[3])
                 : "r"(ar_u2[0]), "r"(ar_u2[1]), "r"(ar_u2[2]),
                   "r"(ar_u2[3]), "r"(br_u2[0]), "r"(br_u2[1]));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
__device__ __forceinline__ void mbarrier_init(uint64_t& mbar,u2 arrive_count=1)
{
    u2 mbar_s = __cvta_generic_to_shared(&mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_s), "r"(arrive_count) : "memory");

}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t&mbar,u2 sz)
{
    u2 mbar_s = __cvta_generic_to_shared(&mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mbar_s), "r"(sz) : "memory");
}

__device__ __forceinline__ void tma_load(uint64_t&mbar,half*dst,const CUtensorMap&desc,u2 x,u2 y)
{
    u2 mbar_s = __cvta_generic_to_shared(&mbar);
    u2 dst_s=__cvta_generic_to_shared(dst);
    constexpr uint64_t cache_hint=0x1000000000000000; // normal
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0],[%1, {%2, %3}],[%4],%5;"
        :
        : "r"(dst_s), "l"(&desc), "r"(x), "r"(y), "r"(mbar_s), "l"(cache_hint)
        : "memory"
    );
}

__device__ __forceinline__ void tma_load(uint64_t&mbar,half*dst,const CUtensorMap&desc,u2 x,u2 y,u2 z)
{
    u2 mbar_s = __cvta_generic_to_shared(&mbar);
    u2 dst_s=__cvta_generic_to_shared(dst);
    constexpr uint64_t cache_hint=0x1000000000000000; // normal
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0],[%1, {%2, %3, %4}],[%5],%6;"
        :
        : "r"(dst_s), "l"(&desc), "r"(x), "r"(y), "r"(z), "r"(mbar_s), "l"(cache_hint)
        : "memory"
    );
}

__device__ __forceinline__ void mbarrier_wait(uint64_t&mbar,u2 phase)
{
    u2 mbar_s = __cvta_generic_to_shared(&mbar);
    constexpr uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(mbar_s), "r"(phase), "r"(ticks)
        : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t&mbar)
{
    u2 mbar_s = __cvta_generic_to_shared(&mbar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mbar_s) : "memory");
}

template<u2 n>
__device__ __forceinline__ void set_reg()
{
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(n) : "memory");
}
#endif

template<typename T>
__device__ __forceinline__ static void swap(T& a,T& b)
{
	T tmp=a;
	a=b;
	b=tmp;
}

__device__ static void print_8x8(const half* a)
{
    for(int i=0;i<32;i++)
    {
        if(threadIdx.x%32==i)
        {
            printf("%f ",__half2float(a[0]));
            printf("%f ",__half2float(a[1]));
            if(i%4==3)printf("\n");
        }
    }
    if(threadIdx.x%32==0)printf("\n");
}

namespace hgemm_v1_impl
{

__global__ static void v1_impl(const half*a,const half*b,half*c,u2 N,u2 M,u2 K)
{
    const u2 tid=threadIdx.x;
    const u2 wid=tid/32;
    const u2 lid=tid%32;
    const u2 raw_bid=blockIdx.y*gridDim.x+blockIdx.x;

    auto calc_bx_by=[&]()
    {
        static constexpr u2 max_Bh=8;
        static constexpr u2 max_Bw=8;
        struct RT_Type{
            u2 x,y;
        };
        
        // const u2 By=raw_bid/(gridDim.x*max_Bh);
        // const u2 max_By=(gridDim.y+max_Bh-1)/max_Bh-1;
        // u2 Bh;
        // if(By<max_By||gridDim.y%max_Bh==0)Bh=max_Bh;
        // else Bh=gridDim.y-By*max_Bh;
        
        // const u2 Bx=raw_bid%(gridDim.x*max_Bh)/(max_Bh*max_Bw);
        // const u2 max_Bx=(gridDim.x+max_Bw-1)/max_Bw-1;
        // u2 Bw;
        // if(Bx<max_Bx||gridDim.x%max_Bw==0)Bw=max_Bw;
        // else Bw=gridDim.x-Bx*max_Bw;

        // u2 bid=raw_bid-By*gridDim.x*max_Bh-Bx*max_Bw*max_Bh;
        // u2 y=By*max_Bh+bid/Bw;
        // u2 x=Bx*max_Bw+bid%Bw;
        
        // return RT_Type{gridDim.x-x-1,y};

        return RT_Type{blockIdx.x,blockIdx.y};
    };

    const auto [bx,by]=calc_bx_by();

    // for(int i=0;i<gridDim.y*gridDim.x;i++)
    // {
    //     if(i==raw_bid&&tid==0)
    //     {
    //         printf("%d %u %u\n",i,by,bx);
    //     }
    // }

    constexpr u2 as_size=128*32;
    constexpr u2 bs_size=32*128;

    __shared__ __align__(128) half as[2][as_size];
    __shared__ __align__(128) half bs[2][bs_size];

    bool stage0=true;
    half ar[4][8];
    half br[4][4];
    float cr[4][4][4]={0};

    const u2 M64=M*64;
    const u2 K16=K*16;

    // const half* LS_a=a +by*128u*M +tid/4*M +tid%4*8;
    // half* LS_as_cur=as[0]+tid/8*64+(tid%8^tid/8%4)*8;
    const half* LS_a=a +by*128u*M +tid/4*M +(tid%4^tid/8%4)*8;
    half* LS_as_cur=as[0]+tid/8*64+(tid%8)*8;
    const half* LR_as_cur0=as[0] +wid/4*64*32 +lid/8%2*8*32 +lid%8*32 +((lid/16+0)^lid%8/2)*8;
    const half* LR_as_cur1=as[0] +wid/4*64*32 +lid/8%2*8*32 +lid%8*32 +((lid/16+2)^lid%8/2)*8;

    // const half* LS_b=b +bx*128u +tid/16*K +tid%16*8;
    // half* LS_bs_cur=bs[0]+tid/16*128+(tid%16^tid/16%8)*8;
    const half* LS_b=b +bx*128u +tid/16*K +(tid%16^tid/16%8)*8;
    half* LS_bs_cur=bs[0]+tid/16*128+(tid%16)*8;

    const half* LR_bs_cur0=bs[0] +lid/8%2*8*128 +lid%8*128 +((lid/16+0+wid%4*4)^lid%8)*8;
    const half* LR_bs_cur1=bs[0] +lid/8%2*8*128 +lid%8*128 +((lid/16+2+wid%4*4)^lid%8)*8;

    auto LS=[&]()
    {
        copy_16B(LS_as_cur+0,LS_a+0);
        copy_16B(LS_as_cur+64*32,LS_a+M64);
        copy_16B(LS_bs_cur+0,LS_b+0);
        copy_16B(LS_bs_cur+16*128,LS_b+K16);
        commit_group();
    };
    
    LS();
    LS_as_cur+=as_size;
    LS_bs_cur+=bs_size;
    LS_a+=32;
    LS_b+=32*K;
    wait_all();
    __syncthreads();

    for(int i=0;i<M;i+=32)
    {
        if(i<M-32)LS();

        ldmatrix_x4(ar[0],LR_as_cur0+0*32);
        ldmatrix_x4(ar[1],LR_as_cur0+16*32);
        ldmatrix_x4(ar[2],LR_as_cur0+32*32);
        ldmatrix_x4(ar[3],LR_as_cur0+48*32);

        ldmatrix_x4_trans(br[0],LR_bs_cur0+0*128);
        ldmatrix_x4_trans(br[2],LR_bs_cur1+0*128);

        #pragma unroll 4
        for(int j=0;j<4;j++)
        {
            #pragma unroll 4
            for(int k=0;k<4;k++)
                mma(cr[j][k],ar[j],br[k]);
        }

        ldmatrix_x4(ar[0],LR_as_cur1+0*32);
        ldmatrix_x4(ar[1],LR_as_cur1+16*32);
        ldmatrix_x4(ar[2],LR_as_cur1+32*32);
        ldmatrix_x4(ar[3],LR_as_cur1+48*32);

        ldmatrix_x4_trans(br[0],LR_bs_cur0+16*128);
        ldmatrix_x4_trans(br[2],LR_bs_cur1+16*128);

        #pragma unroll 4
        for(int j=0;j<4;j++)
        {
            #pragma unroll 4
            for(int k=0;k<4;k++)
                mma(cr[j][k],ar[j],br[k]);
        }

        if(stage0)LS_as_cur-=as_size;
        else LS_as_cur+=as_size;
        if(stage0)LS_bs_cur-=bs_size;
        else LS_bs_cur+=bs_size;
        if(stage0)LR_as_cur0+=as_size;
        else LR_as_cur0-=as_size;
        if(stage0)LR_as_cur1+=as_size;
        else LR_as_cur1-=as_size;
        if(stage0)LR_bs_cur0+=bs_size;
        else LR_bs_cur0-=bs_size;
        if(stage0)LR_bs_cur1+=bs_size;
        else LR_bs_cur1-=bs_size;
        LS_a+=32;
        LS_b+=32*K;
        stage0=!stage0;

        if(i<M-32)
        {
            wait_all();
            __syncthreads();
        }
    }

    half* const STG_c=c +by*128u*K +bx*128u +wid/4*64u*K +wid%4*32u +lid/4*K +lid%4*2;

    for(int i=0;i<4;i++)
    {
        half* const STG_c_row=STG_c+i*16u*K;
        half* const STG_c_row2=STG_c_row+8u*K;
        #pragma unroll 4
        for(int j=0;j<4;j++)
        {
            half2 tmp=__floats2half2_rn(cr[i][j][0],cr[i][j][1]);
            *(half2*)(STG_c_row+j*8u)=tmp;
            half2 tmp2=__floats2half2_rn(cr[i][j][2],cr[i][j][3]);
            *(half2*)(STG_c_row2+j*8u)=tmp2;
        }
    }
}
};

namespace hgemm_v2_impl
{

__global__ static void v2_impl(const half*a,const half*b,half*c,u2 N,u2 M,u2 K)
{
    const u2 tid=threadIdx.x;
    const u2 wid=tid/32;
    const u2 lid=tid%32;
    const u2 raw_bid=blockIdx.y*gridDim.x+blockIdx.x;

    auto calc_bx_by=[&]()
    {
        static constexpr u2 max_Bh=8;
        static constexpr u2 max_Bw=8;
        struct RT_Type{
            u2 x,y;
        };
        
        // const u2 By=raw_bid/(gridDim.x*max_Bh);
        // const u2 max_By=(gridDim.y+max_Bh-1)/max_Bh-1;
        // u2 Bh;
        // if(By<max_By||gridDim.y%max_Bh==0)Bh=max_Bh;
        // else Bh=gridDim.y-By*max_Bh;
        
        // const u2 Bx=raw_bid%(gridDim.x*max_Bh)/(max_Bh*max_Bw);
        // const u2 max_Bx=(gridDim.x+max_Bw-1)/max_Bw-1;
        // u2 Bw;
        // if(Bx<max_Bx||gridDim.x%max_Bw==0)Bw=max_Bw;
        // else Bw=gridDim.x-Bx*max_Bw;

        // u2 bid=raw_bid-By*gridDim.x*max_Bh-Bx*max_Bw*max_Bh;
        // u2 y=By*max_Bh+bid/Bw;
        // u2 x=Bx*max_Bw+bid%Bw;
        
        // return RT_Type{gridDim.x-x-1,y};

        return RT_Type{blockIdx.x,blockIdx.y};
    };

    const auto [bx,by]=calc_bx_by();

    // for(int i=0;i<gridDim.y*gridDim.x;i++)
    // {
    //     if(i==raw_bid&&tid==0)
    //     {
    //         printf("%d %u %u\n",i,by,bx);
    //     }
    // }

    constexpr u2 as_size=128*32;
    constexpr u2 bs_size=32*128;

    __shared__ __align__(128) half as[2][as_size];
    __shared__ __align__(128) half bs[2][bs_size];

    bool stage0=true;
    half ar[4][8];
    half br[4][4];
    float cr[4][4][4]={0};

    const u2 M64=M*64;
    const u2 K16=K*16;

    // const half* LS_a=a +by*128u*M +tid/4*M +tid%4*8;
    // half* LS_as_cur=as[0]+tid/8*64+(tid%8^tid/8%4)*8;
    const half* LS_a=a +by*128u*M +tid/4*M +(tid%4^tid/8%4)*8;
    half* LS_as_cur=as[0]+tid/8*64+(tid%8)*8;
    const half* LR_as_cur0=as[0] +wid/4*64*32 +lid/8%2*8*32 +lid%8*32 +((lid/16+0)^lid%8/2)*8;
    const half* LR_as_cur1=as[0] +wid/4*64*32 +lid/8%2*8*32 +lid%8*32 +((lid/16+2)^lid%8/2)*8;

    // const half* LS_b=b +bx*128u +tid/16*K +tid%16*8;
    // half* LS_bs_cur=bs[0]+tid/16*128+(tid%16^tid/16%8)*8;
    const half* LS_b=b +bx*128u +tid/16*K +(tid%16^tid/16%8)*8;
    half* LS_bs_cur=bs[0]+tid/16*128+(tid%16)*8;

    const half* LR_bs_cur0=bs[0] +lid/8%2*8*128 +lid%8*128 +((lid/16+0+wid%4*4)^lid%8)*8;
    const half* LR_bs_cur1=bs[0] +lid/8%2*8*128 +lid%8*128 +((lid/16+2+wid%4*4)^lid%8)*8;

    auto LS=[&]()
    {
        copy_16B(LS_as_cur+0,LS_a+0);
        copy_16B(LS_as_cur+64*32,LS_a+M64);
        copy_16B(LS_bs_cur+0,LS_b+0);
        copy_16B(LS_bs_cur+16*128,LS_b+K16);
        commit_group();
    };
    
    LS();
    LS_as_cur+=as_size;
    LS_bs_cur+=bs_size;
    LS_a+=32;
    LS_b+=32*K;
    wait_all();
    __syncthreads();

    for(int i=0;i<M;i+=32)
    {
        if(i<M-32)LS();

        ldmatrix_x4(ar[0],LR_as_cur0+0*32);
        ldmatrix_x4(ar[1],LR_as_cur0+16*32);
        ldmatrix_x4(ar[2],LR_as_cur0+32*32);
        ldmatrix_x4(ar[3],LR_as_cur0+48*32);

        ldmatrix_x4_trans(br[0],LR_bs_cur0+0*128);
        ldmatrix_x4_trans(br[2],LR_bs_cur1+0*128);

        #pragma unroll 4
        for(int j=0;j<4;j++)
        {
            #pragma unroll 4
            for(int k=0;k<4;k++)
                mma(cr[j][k],ar[j],br[k]);
        }

        ldmatrix_x4(ar[0],LR_as_cur1+0*32);
        ldmatrix_x4(ar[1],LR_as_cur1+16*32);
        ldmatrix_x4(ar[2],LR_as_cur1+32*32);
        ldmatrix_x4(ar[3],LR_as_cur1+48*32);

        ldmatrix_x4_trans(br[0],LR_bs_cur0+16*128);
        ldmatrix_x4_trans(br[2],LR_bs_cur1+16*128);

        #pragma unroll 4
        for(int j=0;j<4;j++)
        {
            #pragma unroll 4
            for(int k=0;k<4;k++)
                mma(cr[j][k],ar[j],br[k]);
        }

        if(stage0)LS_as_cur-=as_size;
        else LS_as_cur+=as_size;
        if(stage0)LS_bs_cur-=bs_size;
        else LS_bs_cur+=bs_size;
        if(stage0)LR_as_cur0+=as_size;
        else LR_as_cur0-=as_size;
        if(stage0)LR_as_cur1+=as_size;
        else LR_as_cur1-=as_size;
        if(stage0)LR_bs_cur0+=bs_size;
        else LR_bs_cur0-=bs_size;
        if(stage0)LR_bs_cur1+=bs_size;
        else LR_bs_cur1-=bs_size;
        LS_a+=32;
        LS_b+=32*K;
        stage0=!stage0;

        if(i<M-32)
        {
            wait_all();
            __syncthreads();
        }
    }

    half* const STG_c=c +by*128u*K +bx*128u +wid/4*64u*K +wid%4*32u +lid/4*K +lid%4*8;

    #pragma unroll 4
    for(int i=0;i<4;i++)
    {
        half* const STG_c_row=STG_c+i*16u*K;
        half* const STG_c_row2=STG_c_row+8u*K;
        half2 t[4];
        half2 z[4];
        
        #pragma unroll 4
        for(int j=0;j<4;j++)
            t[j]=__floats2half2_rn(cr[i][j][0],cr[i][j][1]);

        auto shift=[&](half2 x,int j)
        {
            if(j==0)return x;
            else if(j<0)return __shfl_down_sync(0xffffffff,x,-j);
            else return __shfl_up_sync(0xffffffff,x,j);
        };

        auto restruct=[&](int j)
        {
            half2 z[4];
            #pragma unroll 4
            for(int k=0;k<4;k++)
                z[k]=shift(t[k],k-j);
            
            half2 res;
            if(lid%4==0)res=z[0];
            else if(lid%4==1)res=z[1];
            else if(lid%4==2)res=z[2];
            else res=z[3];
            return res;
        };
        
        #pragma unroll 4
        for(int j=0;j<4;j++)
            z[j]=restruct(j);

        *(half8*)STG_c_row=*(half8*)z;

        #pragma unroll 4
        for(int j=0;j<4;j++)
            t[j]=__floats2half2_rn(cr[i][j][2],cr[i][j][3]);

        #pragma unroll 4
        for(int j=0;j<4;j++)
            z[j]=restruct(j);

        *(half8*)STG_c_row2=*(half8*)z;
    }
}
};

namespace hgemm_v3_impl
{

__global__ static void v3_impl(const __grid_constant__ CUtensorMap tensor_map_A,const __grid_constant__ CUtensorMap tensor_map_B,half*c,u2 M,u2 K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const u2 tid=threadIdx.x;
    const u2 wid=tid/32;
    const u2 lid=tid%32;
    const u2 bx=blockIdx.x;
    const u2 by=blockIdx.y;

    constexpr u2 as_size=128*32;
    constexpr u2 bs_size=32*128;

    __shared__ __align__(128) half as[2][as_size];
    __shared__ __align__(128) half bs[2][bs_size];
    __shared__ __align__(128) uint64_t mbar[2];

    bool stage0=true;
    half ar[4][8];
    half br[4][4];
    float cr[4][4][4]={0};
    u2 phase[2]={0,0};

    if(tid==0)
    {
        mbarrier_init(mbar[0]);
        mbarrier_init(mbar[1]);
    }

    __syncthreads();

    auto TMA_LS=[&](int stage,u2 offset)
    {
        if(tid==0)
        {
            mbarrier_arrive_expect_tx(mbar[stage], as_size*2+bs_size*2);
            tma_load(mbar[stage],as[stage],tensor_map_A,offset,by*128);
            tma_load(mbar[stage],bs[stage],tensor_map_B,0,offset,bx*2);
        }
    };

    auto TMA_WL=[&](int stage)
    {
        mbarrier_wait(mbar[stage],phase[stage]);
        phase[stage]^=1;
    };

    TMA_LS(0,0);

    const half* LR_as_cur[2];
    LR_as_cur[0]=as[0] +wid/4*64*32 +lid/8%2*8*32 +lid%8*32 +((lid/16+0)^lid%8/2)*8;
    LR_as_cur[1]=as[0] +wid/4*64*32 +lid/8%2*8*32 +lid%8*32 +((lid/16+2)^lid%8/2)*8;

    const half* LR_bs_cur[2];
    LR_bs_cur[0]=bs[0] +wid%4/2*32*64 +lid/8%2*8*64 +lid%8*64 +((lid/16+0+wid%2*4)^lid%8)*8;
    LR_bs_cur[1]=bs[0] +wid%4/2*32*64 +lid/8%2*8*64 +lid%8*64 +((lid/16+2+wid%2*4)^lid%8)*8;

    for(int i=0,stage=0;i<M;i+=32,stage^=1,stage0=!stage0)
    {
        if(i<M-32)TMA_LS(stage^1,i+32);
        TMA_WL(stage);

        auto LR=[&](int j)
        {
            #pragma unroll 4
            for(int k=0;k<4;k++)
                ldmatrix_x4(ar[k],LR_as_cur[j]+k*16*32);
            #pragma unroll 2
            for(int k=0;k<2;k++)
                ldmatrix_x4_trans(br[k*2],LR_bs_cur[k]+j*16*64);
        };

        auto CR=[&]()
        {
            #pragma unroll 4
            for(int j=0;j<4;j++)
            {
                #pragma unroll 4
                for(int k=0;k<4;k++)
                    mma(cr[j][k],ar[j],br[k]);
            }
        };

        auto Switch=[&](bool p,auto&v,u2 offset)
        {
            if(p)v+=offset;
            else v-=offset;
        };

        LR(0); CR(); LR(1); CR();

        Switch(stage0,LR_as_cur[0],as_size);
        Switch(stage0,LR_as_cur[1],as_size);
        Switch(stage0,LR_bs_cur[0],bs_size);
        Switch(stage0,LR_bs_cur[1],bs_size);
        
        __syncthreads();
    }

    half* const STG_c=c +by*128u*K +bx*128u +wid/4*64u*K +wid%4*32u +lid/4*K +lid%4*8;

    #pragma unroll 4
    for(int i=0;i<4;i++)
    {
        half* const STG_c_row=STG_c+i*16u*K;
        half* const STG_c_row2=STG_c_row+8u*K;
        half2 t[4];
        half2 z[4];
        
        #pragma unroll 4
        for(int j=0;j<4;j++)
            t[j]=__floats2half2_rn(cr[i][j][0],cr[i][j][1]);

        auto shift=[&](half2 x,int j)
        {
            if(j==0)return x;
            else if(j<0)return __shfl_down_sync(0xffffffff,x,-j);
            else return __shfl_up_sync(0xffffffff,x,j);
        };

        auto restruct=[&](int j)
        {
            half2 z[4];
            #pragma unroll 4
            for(int k=0;k<4;k++)
                z[k]=shift(t[k],k-j);
            
            half2 res;
            if(lid%4==0)res=z[0];
            else if(lid%4==1)res=z[1];
            else if(lid%4==2)res=z[2];
            else res=z[3];
            return res;
        };
        
        #pragma unroll 4
        for(int j=0;j<4;j++)
            z[j]=restruct(j);

        *(half8*)STG_c_row=*(half8*)z;

        #pragma unroll 4
        for(int j=0;j<4;j++)
            t[j]=__floats2half2_rn(cr[i][j][2],cr[i][j][3]);

        #pragma unroll 4
        for(int j=0;j<4;j++)
            z[j]=restruct(j);

        *(half8*)STG_c_row2=*(half8*)z;
    }
#endif
}
};

namespace hgemm_v4_impl
{

__global__ static void v4_impl(const __grid_constant__ CUtensorMap tensor_map_A,const __grid_constant__ CUtensorMap tensor_map_B,half*c,u2 M,u2 K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const u2 tid=threadIdx.x;
    const u2 bx=blockIdx.x;
    const u2 by=blockIdx.y;

    constexpr u2 as_size=128*32;
    constexpr u2 bs_size=32*128;

    __shared__ __align__(128) half as[2][as_size];
    __shared__ __align__(128) half bs[2][bs_size];
    __shared__ __align__(128) uint64_t full[2];
    __shared__ __align__(128) uint64_t empty[2];

    if(tid==0)
    {
        mbarrier_init(full[0]);
        mbarrier_init(full[1]);
        mbarrier_init(empty[0],256);
        mbarrier_init(empty[1],256);
    }

    __syncthreads();

    auto producer=[&]()
    {
        u2 phase[2]={0,0};
        auto TMA_LS=[&](int stage,u2 offset)
        {
            mbarrier_arrive_expect_tx(full[stage], as_size*2+bs_size*2);
            tma_load(full[stage],as[stage],tensor_map_A,offset,by*128);
            tma_load(full[stage],bs[stage],tensor_map_B,0,offset,bx*2);
        };

        auto WC=[&](int stage)
        {
            mbarrier_wait(empty[stage],phase[stage]);
            phase[stage]^=1;
        };
        TMA_LS(0,0);
        TMA_LS(1,32);
        for(int i=64,stage=0;i<M;i+=32,stage^=1)
        {
            WC(stage);
            TMA_LS(stage,i);
        }
    };

    auto consumer=[&]()
    {
        const u2 wid=tid/32;
        const u2 lid=tid%32;
        half ar[4][8];
        half br[4][4];
        float cr[4][4][4]={0};
        u2 phase[2]={0,0};
        bool stage0=true;

        const half* LR_as_cur[2];
        LR_as_cur[0]=as[0] +wid/4*64*32 +lid/8%2*8*32 +lid%8*32 +((lid/16+0)^lid%8/2)*8;
        LR_as_cur[1]=as[0] +wid/4*64*32 +lid/8%2*8*32 +lid%8*32 +((lid/16+2)^lid%8/2)*8;

        const half* LR_bs_cur[2];
        LR_bs_cur[0]=bs[0] +wid%4/2*32*64 +lid/8%2*8*64 +lid%8*64 +((lid/16+0+wid%2*4)^lid%8)*8;
        LR_bs_cur[1]=bs[0] +wid%4/2*32*64 +lid/8%2*8*64 +lid%8*64 +((lid/16+2+wid%2*4)^lid%8)*8;

        auto TMA_WL=[&](int stage)
        {
            mbarrier_wait(full[stage],phase[stage]);
            phase[stage]^=1;
        };

        auto C=[&](int stage)
        {
            auto LR=[&](int j)
            {
                #pragma unroll 4
                for(int k=0;k<4;k++)
                    ldmatrix_x4(ar[k],LR_as_cur[j]+k*16*32);
                #pragma unroll 2
                for(int k=0;k<2;k++)
                    ldmatrix_x4_trans(br[k*2],LR_bs_cur[k]+j*16*64);
            };

            auto CR=[&]()
            {
                #pragma unroll 4
                for(int j=0;j<4;j++)
                {
                    #pragma unroll 4
                    for(int k=0;k<4;k++)
                        mma(cr[j][k],ar[j],br[k]);
                }
            };

            LR(0); CR(); LR(1); CR();

            mbarrier_arrive(empty[stage]);
        };

        for(int i=0,stage=0;i<M;i+=32,stage^=1,stage0=!stage0)
        {
            TMA_WL(stage);
            C(stage);

            auto Switch=[&](bool p,auto&v,u2 offset)
            {
                if(p)v+=offset;
                else v-=offset;
            };

            Switch(stage0,LR_as_cur[0],as_size);
            Switch(stage0,LR_as_cur[1],as_size);
            Switch(stage0,LR_bs_cur[0],bs_size);
            Switch(stage0,LR_bs_cur[1],bs_size);
        }

        half* const STG_c=c +by*128u*K +bx*128u +wid/4*64u*K +wid%4*32u +lid/4*K +lid%4*8;

        #pragma unroll 4
        for(int i=0;i<4;i++)
        {
            half* const STG_c_row=STG_c+i*16u*K;
            half* const STG_c_row2=STG_c_row+8u*K;
            half2 t[4];
            half2 z[4];
            
            #pragma unroll 4
            for(int j=0;j<4;j++)
                t[j]=__floats2half2_rn(cr[i][j][0],cr[i][j][1]);

            auto shift=[&](half2 x,int j)
            {
                if(j==0)return x;
                else if(j<0)return __shfl_down_sync(0xffffffff,x,-j);
                else return __shfl_up_sync(0xffffffff,x,j);
            };

            auto restruct=[&](int j)
            {
                half2 z[4];
                #pragma unroll 4
                for(int k=0;k<4;k++)
                    z[k]=shift(t[k],k-j);
                
                half2 res;
                if(lid%4==0)res=z[0];
                else if(lid%4==1)res=z[1];
                else if(lid%4==2)res=z[2];
                else res=z[3];
                return res;
            };
            
            #pragma unroll 4
            for(int j=0;j<4;j++)
                z[j]=restruct(j);

            *(half8*)STG_c_row=*(half8*)z;

            #pragma unroll 4
            for(int j=0;j<4;j++)
                t[j]=__floats2half2_rn(cr[i][j][2],cr[i][j][3]);

            #pragma unroll 4
            for(int j=0;j<4;j++)
                z[j]=restruct(j);

            *(half8*)STG_c_row2=*(half8*)z;
        }
    };

    if(tid==256)producer();
    else consumer();
#endif
}
};

namespace hgemm_v5_impl
{

__global__ static void v5_impl(const __grid_constant__ CUtensorMap tensor_map_A,const __grid_constant__ CUtensorMap tensor_map_B,half*c,u2 M,u2 K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const u2 tid=threadIdx.x;
    const u2 bx=blockIdx.x;
    const u2 by=blockIdx.y;

    constexpr u2 as_size=128*32;
    constexpr u2 bs_size=32*128;

    __shared__ __align__(128) half as[2][as_size];
    __shared__ __align__(128) half bs[2][bs_size];
    __shared__ __align__(128) uint64_t full[2];

    if(tid==0)
    {
        mbarrier_init(full[0]);
        mbarrier_init(full[1]);
    }

    __syncthreads();

    auto producer=[&]()
    {
        auto TMA_LS=[&](int stage,u2 offset)
        {
            if(tid==256)
            {
                mbarrier_arrive_expect_tx(full[stage], as_size*2+bs_size*2);
                tma_load(full[stage],as[stage],tensor_map_A,offset,by*128);
                tma_load(full[stage],bs[stage],tensor_map_B,0,offset,bx*2);
            }
        };

        auto WC=[&](int stage)
        {
            barrier_sync(stage,256+32);
        };
        TMA_LS(0,0);
        TMA_LS(1,32);
        for(int i=64;i<M;i+=64)
        {
            WC(0);
            TMA_LS(0,i);
            WC(1);
            TMA_LS(1,i+32);
        }
    };

    auto consumer=[&]()
    {
        const u2 wid=tid/32;
        const u2 lid=tid%32;
        half ar[4][8];
        half br[4][4];
        float cr[4][4][4]={0};
        u2 phase[2]={0,0};
        bool stage0=true;

        const half* LR_as_cur[2];
        LR_as_cur[0]=as[0] +wid/4*64*32 +lid/8%2*8*32 +lid%8*32 +((lid/16+0)^lid%8/2)*8;
        LR_as_cur[1]=as[0] +wid/4*64*32 +lid/8%2*8*32 +lid%8*32 +((lid/16+2)^lid%8/2)*8;

        const half* LR_bs_cur[2];
        LR_bs_cur[0]=bs[0] +wid%4/2*32*64 +lid/8%2*8*64 +lid%8*64 +((lid/16+0+wid%2*4)^lid%8)*8;
        LR_bs_cur[1]=bs[0] +wid%4/2*32*64 +lid/8%2*8*64 +lid%8*64 +((lid/16+2+wid%2*4)^lid%8)*8;

        auto TMA_WL=[&](int stage)
        {
            mbarrier_wait(full[stage],phase[stage]);
            phase[stage]^=1;
        };

        auto C=[&](int stage)
        {
            auto LR=[&](int j)
            {
                #pragma unroll 4
                for(int k=0;k<4;k++)
                    ldmatrix_x4(ar[k],LR_as_cur[j]+(k*16*32+stage*as_size));
                #pragma unroll 2
                for(int k=0;k<2;k++)
                    ldmatrix_x4_trans(br[k*2],LR_bs_cur[k]+(j*16*64+stage*bs_size));
            };

            auto CR=[&]()
            {
                #pragma unroll 4
                for(int j=0;j<4;j++)
                {
                    #pragma unroll 4
                    for(int k=0;k<4;k++)
                        mma(cr[j][k],ar[j],br[k]);
                }
            };

            LR(0); CR(); LR(1); CR();

            barrier_arrive(stage,256+32);
        };

        for(int i=0;i<M;i+=64)
        {
            TMA_WL(0);
            C(0);
            TMA_WL(1);
            C(1);
        }

        half* const STG_c=c +by*128u*K +bx*128u +wid/4*64u*K +wid%4*32u +lid/4*K +lid%4*8;

        #pragma unroll 4
        for(int i=0;i<4;i++)
        {
            half* const STG_c_row=STG_c+i*16u*K;
            half* const STG_c_row2=STG_c_row+8u*K;
            half2 t[4];
            half2 z[4];
            
            #pragma unroll 4
            for(int j=0;j<4;j++)
                t[j]=__floats2half2_rn(cr[i][j][0],cr[i][j][1]);

            auto shift=[&](half2 x,int j)
            {
                if(j==0)return x;
                else if(j<0)return __shfl_down_sync(0xffffffff,x,-j);
                else return __shfl_up_sync(0xffffffff,x,j);
            };

            auto restruct=[&](int j)
            {
                half2 z[4];
                #pragma unroll 4
                for(int k=0;k<4;k++)
                    z[k]=shift(t[k],k-j);
                
                half2 res;
                if(lid%4==0)res=z[0];
                else if(lid%4==1)res=z[1];
                else if(lid%4==2)res=z[2];
                else res=z[3];
                return res;
            };
            
            #pragma unroll 4
            for(int j=0;j<4;j++)
                z[j]=restruct(j);

            *(half8*)STG_c_row=*(half8*)z;

            #pragma unroll 4
            for(int j=0;j<4;j++)
                t[j]=__floats2half2_rn(cr[i][j][2],cr[i][j][3]);

            #pragma unroll 4
            for(int j=0;j<4;j++)
                z[j]=restruct(j);

            *(half8*)STG_c_row2=*(half8*)z;
        }
    };

    if(tid>=256)producer();
    else consumer();
#endif
}
};

void hgemm_v1(cudaStream_t stream,const half* a, const half* b, half* c, u2 n, u2 m, u2 k)
{
    assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

	dim3 grid(k/128,n/128);
	dim3 block(256);
	hgemm_v1_impl::v1_impl<<<grid,block,0,stream>>>(a,b,c,n,m,k);
}

void hgemm_v2(cudaStream_t stream,const half* a, const half* b, half* c, u2 n, u2 m, u2 k)
{
    assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");

	dim3 grid(k/128,n/128);
	dim3 block(256);
	hgemm_v2_impl::v2_impl<<<grid,block,0,stream>>>(a,b,c,n,m,k);
}

void hgemm_v3(cudaStream_t stream,const half* a, const half* b, half* c, u2 n, u2 m, u2 k)
{
    assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");


    CUtensorMap tma_desc_A;

    uint64_t globalDimA[2] = {m,n};
    uint64_t globalStrideA[1] = {m*sizeof(__half)};
    uint32_t boxDimA[2] = {32,128};
    uint32_t elementStrideA[2] = {1,1};

    CUresult res = cuTensorMapEncodeTiled(
        &tma_desc_A,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        (void*)a,
        globalDimA,
        globalStrideA,
        boxDimA,
        elementStrideA,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    CUtensorMap tma_desc_B;

    uint64_t globalDimB[3] = {64,m,k/64};
    uint64_t globalStrideB[2] = {k*sizeof(__half),128};
    uint32_t boxDimB[3] = {64,32,2};
    uint32_t elementStrideB[3] = {1,1,1};

    res = cuTensorMapEncodeTiled(
        &tma_desc_B,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        3,
        (void*)b,
        globalDimB,
        globalStrideB,
        boxDimB,
        elementStrideB,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    
	dim3 grid(k/128,n/128);
	dim3 block(256);
	hgemm_v3_impl::v3_impl<<<grid,block,0,stream>>>(tma_desc_A,tma_desc_B,c,m,k);
}

void hgemm_v4(cudaStream_t stream,const half* a, const half* b, half* c, u2 n, u2 m, u2 k)
{
    assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");


    CUtensorMap tma_desc_A;

    uint64_t globalDimA[2] = {m,n};
    uint64_t globalStrideA[1] = {m*sizeof(__half)};
    uint32_t boxDimA[2] = {32,128};
    uint32_t elementStrideA[2] = {1,1};

    CUresult res = cuTensorMapEncodeTiled(
        &tma_desc_A,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        (void*)a,
        globalDimA,
        globalStrideA,
        boxDimA,
        elementStrideA,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    CUtensorMap tma_desc_B;

    uint64_t globalDimB[3] = {64,m,k/64};
    uint64_t globalStrideB[2] = {k*sizeof(__half),128};
    uint32_t boxDimB[3] = {64,32,2};
    uint32_t elementStrideB[3] = {1,1,1};

    res = cuTensorMapEncodeTiled(
        &tma_desc_B,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        3,
        (void*)b,
        globalDimB,
        globalStrideB,
        boxDimB,
        elementStrideB,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    
	dim3 grid(k/128,n/128);
	dim3 block(257);
	hgemm_v4_impl::v4_impl<<<grid,block,0,stream>>>(tma_desc_A,tma_desc_B,c,m,k);
}

void hgemm_v5(cudaStream_t stream,const half* a, const half* b, half* c, u2 n, u2 m, u2 k)
{
    assert_throw(m%128==0&&n%128==0&&k%128==0,"m,n,k must be divisible by 128");


    CUtensorMap tma_desc_A;

    uint64_t globalDimA[2] = {m,n};
    uint64_t globalStrideA[1] = {m*sizeof(__half)};
    uint32_t boxDimA[2] = {32,128};
    uint32_t elementStrideA[2] = {1,1};

    CUresult res = cuTensorMapEncodeTiled(
        &tma_desc_A,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,
        (void*)a,
        globalDimA,
        globalStrideA,
        boxDimA,
        elementStrideA,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    CUtensorMap tma_desc_B;

    uint64_t globalDimB[3] = {64,m,k/64};
    uint64_t globalStrideB[2] = {k*sizeof(__half),128};
    uint32_t boxDimB[3] = {64,32,2};
    uint32_t elementStrideB[3] = {1,1,1};

    res = cuTensorMapEncodeTiled(
        &tma_desc_B,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        3,
        (void*)b,
        globalDimB,
        globalStrideB,
        boxDimB,
        elementStrideB,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    
	dim3 grid(k/128,n/128);
	dim3 block(256+32);
	hgemm_v5_impl::v5_impl<<<grid,block,0,stream>>>(tma_desc_A,tma_desc_B,c,m,k);
}

void hgemm_cublas(cudaStream_t stream, const half* a, const half* b, half* c, u2 N, u2 M, u2 K)
{
    static cublasHandle_t handle = []() {
        cublasHandle_t handle;
        cublasCreate(&handle);
        return handle;
    }();
    cublasSetStream(handle, stream);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha, b, CUDA_R_16F, N,
                 a, CUDA_R_16F, K,
                 &beta, c, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
