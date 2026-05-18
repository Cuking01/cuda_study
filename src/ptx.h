#pragma once
#include "type.h"

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

__device__ __forceinline__ void tma_load(uint64_t&mbar,half*dst,const CUtensorMap&desc,u2 x,u2 y,u2 z,u2 w)
{
    u2 mbar_s = __cvta_generic_to_shared(&mbar);
    u2 dst_s=__cvta_generic_to_shared(dst);
    constexpr uint64_t cache_hint=0x1000000000000000; // normal
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0],[%1, {%2, %3, %4, %5}],[%6],%7;"
        :
        : "r"(dst_s), "l"(&desc), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_s), "l"(cache_hint)
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
