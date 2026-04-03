#pragma once

#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<utility>


struct Event
{
    cudaEvent_t event;
    Event(){cudaEventCreate(&event);}
    ~Event()
    {
        cudaEventDestroy(event);
    }

    operator cudaEvent_t&()
    {
        return event;
    }

    operator const cudaEvent_t&() const
    {
        return event;
    }
};

float event_duration(const Event& event_start,const Event& event_end)
{
    float ms;
    cudaEventElapsedTime(&ms,event_start,event_end);
    return ms;
}

struct Stream
{
    cudaStream_t stream;
    Stream(){cudaStreamCreate(&stream);}
    ~Stream()
    {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    Stream* record(Event& event)
    {
        cudaEventRecord(event,stream);
        return this;
    }

    Stream* wait(Event& event)
    {
        cudaEventSynchronize(event);
        return this;
    }

    Stream* synchronize()
    {
        cudaStreamSynchronize(stream);
        return this;
    }

    template<typename Kernal_Func,typename... Args>
    Stream* run(Kernal_Func func,dim3 grid,dim3 block,Args&&... args)
    {
        func<<<grid,block,0,stream>>>(std::forward<Args>(args)...);
        return this;
    }

    template<typename Kernal_Func,typename... Args>
    Stream* run_with_shared(Kernal_Func func,dim3 grid,dim3 block,size_t shared_size,Args&&... args)
    {
        func<<<grid,block,shared_size,stream>>>(std::forward<Args>(args)...);
        return this;
    }

    Stream* operator->()
    {
        return this;
    }
};
