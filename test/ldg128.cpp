#include<stdio.h>
#include"ldg128_test.h"

int main()
{
    float a[16384];
    for(int i=0;i<16384;i++)
        a[i]=i;
    ldg128_test_v1(a);
    ldg128_test_v2(a,32);
    ldg128_test_v2(a,64);
    ldg128_test_v2(a,128);
    ldg128_test_v2(a,256);
    ldg128_test_v2(a,512);
    ldg128_test_v2(a,1024);
    ldg128_test_v2(a,2048);
    ldg128_test_v2(a,4096);
}
