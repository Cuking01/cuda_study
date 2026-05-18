#include<cuda.h>
#include<cuda_fp16.h>
#include "type.h"

struct FA_TMA_Desc
{
    CUtensorMap tma_desc;

    FA_TMA_Desc(half*p,u2 n,u2 head_num)
    {
        u3 globalDim[4] = {64,n,head_num,2};
        u3 globalStride[3] = {256,256ull*n,128};
        u2 boxDim[4] = {64,128,1,2};
        u2 elementStride[4] = {1,1,1,1};

        CUresult res = cuTensorMapEncodeTiled(
        &tma_desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        4,
        (void*)p,
        globalDim,
        globalStride,
        boxDim,
        elementStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if(res!=0)
        {
            throw std::runtime_error("create tensor map failed\n");
            return;
        }
    }

    CUtensorMap& get()
    {
        return tma_desc;
    }
};
