#include <iostream>
#include "wavefront_test.h"

int main()
{
    float in[128];
    float out[32];
    for(int i=0;i<128;i++)
    {
        in[i]=i;
    }
    test_wavefront_1(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;
    
    test_wavefront_2(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_wavefront_3(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;
    
    test_wavefront_4(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_wavefront_5(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_wavefront_6(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_wavefront_7(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;
}
