#include "lsu_test.h"
#include <stdio.h>
#include <time.h>
#include <vector>

void test_t(int t)
{
    int k=1<<20;
    std::vector<float> out(t);

    int start=clock();
    test_lsu_v1(k,t,out.data());
    int end=clock();
    
    printf("t: %d\n",t);
    printf("time: %f ms\n",(end-start)*1.0/CLOCKS_PER_SEC*1000);

    // for(int i=0;i<t;i++)
    //     printf("%d %f\n",i,out[i]);
}

int main()
{
    test_t(32);
    test_t(128);
    test_t(256);
    test_t(64);
    test_t(32);
    test_t(64);
    test_t(128);
    test_t(256);
    test_t(512);
    test_t(1024);
}

