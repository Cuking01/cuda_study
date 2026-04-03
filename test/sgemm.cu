
#include<stdio.h>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <exception>
#include <stdexcept>
#include <time.h>
#include <format>
#include "sgemm.h"

void random_init(std::vector<float>& data, int seed)
{
	//mt19937随机数生成器,传入种子
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(0,1);
	for(auto& x:data)
		x=dist(rng);

	// for(int i=0;i<data.size();i++)
	// 	data[i]=i/100000.0;
}

void sgemm_ref(const float* a, const float* b, float* c, int n, int m, int k)
{
	for(int i=0;i<n;i++)
		for(int j=0;j<k;j++)
		{
			float sum=0;
			for(int l=0;l<m;l++)
				sum+=a[i*m+l]*b[l*k+j];
			c[i*k+j]=sum;
		}
}

bool check_equal(float a,float b)
{
	//绝对误差+相对误差
	return fabs(a-b)<1e-6||fabs(a-b)/(std::max(fabs(a),fabs(b)))<1e-4;
}

typedef void (*sgemm_func)(const float* a, const float* b, float* c, int n, int m, int k);

void test_correctness(sgemm_func sgemm,std::string name,int n,int m,int k)
{
	std::vector<float> a(n*m), b(m*k), c(n*k), c_ref(n*k);
	random_init(a,1);
	random_init(b,2);
	
	printf("test correctness %s, n=%d, m=%d, k=%d\n",name.c_str(),n,m,k);

	sgemm(a.data(),b.data(),c.data(),n,m,k);
	sgemm_ref(a.data(),b.data(),c_ref.data(),n,m,k);
	
	for(int i=0;i<n*k;i++)
		if(!check_equal(c[i],c_ref[i]))
		{
			printf("test correctness not passed %s, n=%d, m=%d, k=%d, failed at c[%d]\n\n",name.c_str(),n,m,k,i);
			return;
		}

	printf("test correctness %s, n=%d, m=%d, k=%d, passed\n\n",name.c_str(),n,m,k);
}

void test_speed(sgemm_func sgemm,std::string name,int n,int m,int k,int times=1)
{
	printf("test %s, n=%d, m=%d, k=%d, %d times\n",name.c_str(),n,m,k,times);
	
	std::vector<float> a(n*m), b(m*k), c(n*k), c_ref(n*k);
	random_init(a,1);
	random_init(b,2);

	int st=clock();
	for(int i=0;i<times;i++)
		sgemm(a.data(),b.data(),c.data(),n,m,k);
	int ed=clock();
	printf("%s avg time: %d us\n%f Tflops\n\n",name.c_str(),(ed-st)/times,2.0*n*m*k*times/(ed-st)/1e6);
	
}



int main()
{
	//先打印一下顺便预热
	hello_sgemm();
	test_stream(128,256,512);

	
	test_correctness(sgemm_v1,"sgemm_v1",128,256,512);
	test_correctness(sgemm_v2,"sgemm_v2",128,256,512);
	//test_correctness(sgemm_v3,"sgemm_v3",128,128,128);
	test_correctness(sgemm_v3,"sgemm_v3",128,256,512);
	test_correctness(sgemm_v4,"sgemm_v4",128,256,512);
	test_correctness(sgemm_cublas,"sgemm_cublas",256,256,256);
	
	test_speed(sgemm_v2,"sgemm_v2",2048,2048,2048,10);
	test_speed(sgemm_v1,"sgemm_v1",4096,4096,4096,10);
	test_speed(sgemm_v2,"sgemm_v2",4096,4096,4096,10);
	test_speed(sgemm_v3,"sgemm_v3",4096,4096,4096,10);
	test_speed(sgemm_v4,"sgemm_v4",4096,4096,4096,10);

	test_speed(sgemm_v2,"sgemm_v2",8192,8192,8192,10);
	test_speed(sgemm_v2,"sgemm_v2",16384,16384,16384,1);
	test_speed(sgemm_v4,"sgemm_v4",16384,16384,16384,1);
	test_speed(sgemm_cublas,"sgemm_cublas",16384,16384,16384,1);

	test_stream(16384,16384,16384);

	test_speed(sgemm_cublas,"sgemm_cublas",8192,8192,8192,10);
}
