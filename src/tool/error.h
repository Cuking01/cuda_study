#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

inline void process_error(){
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

#define assert_throw(cond,msg) do{if(!(cond)) throw std::runtime_error(msg);}while(0)
