#pragma once

#include "type.h"
#include "ptx.h"

template<u2 bar_id,u2 arrive_count>
struct barrier
{
    __device__ __forceinline__ void arrive() const
    {
        barrier_arrive(bar_id,arrive_count);
    }
    __device__ __forceinline__ void sync() const
    {
        barrier_sync(bar_id,arrive_count);
    }
};
