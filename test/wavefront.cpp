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

    test_wavefront_8(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_wavefront_9(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_wavefront_10(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    test_wavefront_11(in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    unsigned int idx[32]={
         0,12,23,25,23,25, 0,12,    //0 1 4 7
        10, 3,30, 5,10, 3,30, 5,    //2 3 5 6
         1, 5,15,22,15, 1, 5,15,    //1 5 6 7
         2,20,24,27,24,27,20, 2     //0 2 3 4
    };
    test_wavefront_x(idx,in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    unsigned int idx2[32]={
        2,3,4,5,5,4,3,2,
        6,7,7,7,6,6,6,6,
        0,2,1,1,4,3,4,1,
        6,5,7,5,6,7,6,5,
    };
    test_wavefront_x(idx2,in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;
    

    unsigned int idx3[32]={
        0,0,0,0,0,0,0,8,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
    };
    test_wavefront_x(idx3,in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    unsigned int idx4[32]={
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,8,0,0,0,
    };
    test_wavefront_x(idx4,in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    unsigned int idx5[32]={
        0,0,0,0,0,0,0,0,
        8,8,8,8,8,8,8,8,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
    };
    test_wavefront_x(idx5,in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    unsigned int idx6[32]={
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
    };
    test_wavefront_x(idx6,in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;
    
    unsigned int idx7[32]={
        0,0,0,0,1,1,1,1,
        0,0,0,0,1,1,1,1,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,
    };
    test_wavefront_x(idx7,in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    unsigned int idx8[32]={
         0, 0,12,12,23,23,25,25,    //0 1 4 7
        10,10, 3, 3,30,30, 5, 5,    //2 3 5 6
         1, 1, 5, 5,15,15,22,22,    //1 5 6 7
         2, 2,20,20,24,24,27,27,     //0 2 3 4
    };
    test_wavefront_x(idx8,in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    unsigned int idx9[32]={
         0,12, 0,12,23,25,23,25,    //0 1 4 7
        10, 3,10, 3,30, 5,30, 5,    //2 3 5 6
         1, 1, 5, 5,15,15,22,22,    //1 5 6 7
         2, 2,20,20,24,24,27,27,     //0 2 3 4
    };
    test_wavefront_x(idx9,in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;

    unsigned int idx10[32]={
        6,3,6,3,6,3,6,3,
        6,3,6,3,6,3,6,3,
        6,3,6,3,6,3,6,3,
        6,3,6,3,6,3,6,3,
    };
    test_wavefront_x(idx10,in,out);
    for(int i=0;i<32;i++)
        std::cout<<out[i]<<std::endl;
}
