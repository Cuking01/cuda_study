#include "cp_async_test.h"
#include <iostream>

int main()
{
    float in[128];

    for(int i=0;i<128;i++)
        in[i]=i;
    
    float out[32];

    test_cp_sync_1(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_cp_sync_2(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_cp_async_prefetch_1(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_cp_async_prefetch_2(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_cp_async_1(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_cp_async_2(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    return 0;
}