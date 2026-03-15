#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

void process_error(){
    if(cudaGetLastError()!=cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(cudaGetLastError()));
}

#define assert_throw(cond,msg) do{if(!(cond)) throw std::runtime_error(msg);}while(0)
