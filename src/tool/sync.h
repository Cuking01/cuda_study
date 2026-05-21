#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include "error.h"

inline void gpu_sync()
{
    process_error();
    cudaError_t err=cudaDeviceSynchronize();
    if(err!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(err));
}

