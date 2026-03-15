#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>



void* gpu_malloc(size_t size){
    void* data;
    cudaError_t err = cudaMalloc(&data,size);
    if(err!=cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
    return data;
}

void gpu_free(void* data){
    cudaError_t err = cudaFree(data);
    if(err!=cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

void gpu_memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind){
    cudaError_t err = cudaMemcpy(dst,src,size,kind);
    if(err!=cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

template<typename T>
struct GPU_Data
{
    T* data;
    size_t size;

    GPU_Data(const T* host_data, size_t size) :size(size) 
    {
        data = (T*)gpu_malloc(size*sizeof(T));
        gpu_memcpy(data,host_data,size*sizeof(T),cudaMemcpyHostToDevice);
    }

    GPU_Data(const std::vector<T>& host_data, size_t size) :size(size) 
    {
        data = (T*)gpu_malloc(size*sizeof(T));
        gpu_memcpy(data,host_data.data(),size*sizeof(T),cudaMemcpyHostToDevice);
    }

    GPU_Data(size_t size) :size(size) 
    {
        data = (T*)gpu_malloc(size*sizeof(T));
    }

    GPU_Data(const GPU_Data& other) :size(other.size)
    {
        data = (T*)gpu_malloc(size*sizeof(T));
        gpu_memcpy(data,other.data,size*sizeof(T),cudaMemcpyDeviceToDevice);
    }

    GPU_Data(GPU_Data&& other) :size(other.size)
    {
        data = other.data;
        other.data = nullptr;
        other.size = 0;
    }

    void operator=(const GPU_Data& other)
    {
        if(this!=&other)
            gpu_memcpy(data,other.data,size*sizeof(T),cudaMemcpyDeviceToDevice);
    }

    void operator=(GPU_Data&& other)
    {
        if(this!=&other)
        {
            gpu_free(data);
            data = other.data;
            other.data = nullptr;
            other.size = 0;
        }
    }

    void to_host(T* host_data)
    {
        gpu_memcpy(host_data,data,size*sizeof(T),cudaMemcpyDeviceToHost);
    }

    std::vector<T> to_host()
    {
        std::vector<T> host_data(size);
        gpu_memcpy(host_data.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost);
        return host_data;
    }

    operator T*()
    {
        return data;
    }

    ~GPU_Data()
    {
        gpu_free(data);
    }
};


