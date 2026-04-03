#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include "error.h"

inline void gpu_sync()
{
    cudaDeviceSynchronize();
    process_error();
}

