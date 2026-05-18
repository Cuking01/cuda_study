
#include<stdio.h>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <exception>
#include <stdexcept>
#include <time.h>
#include <format>
#include "tool.h"
#include "sgemm.h"

#define FIXED_DATA

void random_init(std::vector<float>& data, int seed)
{
#ifndef FIXED_DATA
	//mt19937随机数生成器,传入种子
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(0,1);
	for(auto& x:data)
		x=dist(rng);
#else
	for(int i=0;i<data.size();i++)
		data[i]=i/100000.0;
#endif
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

struct Stat
{
	double avg;
	double stddev;
};

Stat calc_stat(const std::vector<double>& values)
{
	double sum=0;
	for(double value:values)
		sum+=value;

	double avg=sum/values.size();
	double sq_sum=0;
	for(double value:values)
	{
		double diff=value-avg;
		sq_sum+=diff*diff;
	}

	double stddev=values.size()>1?sqrt(sq_sum/(values.size()-1)):0.0;
	return {avg,stddev};
}

typedef void (*sgemm_func)(cudaStream_t stream,const float* a, const float* b, float* c, int n, int m, int k);

void test_correctness(sgemm_func sgemm,std::string name,int n,int m,int k)
{
	std::vector<float> a(n*m), b(m*k), c(n*k), c_ref(n*k);
	random_init(a,1);
	random_init(b,2);
	GPU_Data<float> a_gpu(a), b_gpu(b), c_gpu(c);
	
	printf("test correctness %s, n=%d, m=%d, k=%d\n",name.c_str(),n,m,k);

	Stream stream;
	stream->run_any(sgemm,a_gpu,b_gpu,c_gpu,n,m,k);
	sgemm_ref(a.data(),b.data(),c_ref.data(),n,m,k);
	stream->synchronize();
	c_gpu.to_host(c.data());

	for(int i=0;i<n*k;i++)
		if(!check_equal(c[i],c_ref[i]))
		{
			printf("test correctness not passed %s, n=%d, m=%d, k=%d, failed at c[%d]\nc[%d]=%f, c_ref[%d]=%f\n",name.c_str(),n,m,k,i,i,c[i],i,c_ref[i]);
			return;
		}

	printf("test correctness %s, n=%d, m=%d, k=%d, passed\n\n",name.c_str(),n,m,k);
}

void test_speed(sgemm_func sgemm,std::string name,int n,int m,int k,int times=1)
{
	printf("test %s, n=%d, m=%d, k=%d, %d times\n",name.c_str(),n,m,k,times);
	
	std::vector<float> a(n*m), b(m*k), c(n*k);
	random_init(a,1);
	random_init(b,2);

	GPU_Data<float> a_gpu(a), b_gpu(b), c_gpu(c);

	Stream stream;
	Event start,end;

	double flops=2.0*n*m*k;
	std::vector<double> time_ms;
	std::vector<double> tflops;
	time_ms.reserve(times);
	tflops.reserve(times);

	for(int i=0;i<times;i++)
	{
		stream->nop()->record(start)->run_any(sgemm,a_gpu,b_gpu,c_gpu,n,m,k)->record(end)->synchronize();
		cudaDeviceSynchronize();
		double elapsed=event_duration(start,end);
		time_ms.push_back(elapsed);
		tflops.push_back(flops/elapsed/1e9);
	}

	Stat time_stat=calc_stat(time_ms);
	Stat tflops_stat=calc_stat(tflops);

	if(time_ms.size()>1)
	{
		printf("%s avg time: %f +/- %f ms (3stddev)\n%f +/- %f Tflops (3stddev)\n\n",
			name.c_str(),
			time_stat.avg,
			3.0*time_stat.stddev,
			tflops_stat.avg,
			3.0*tflops_stat.stddev
		);
	}
	else
	{
		printf("%s avg time: %f ms\n%f Tflops\n\n",
			name.c_str(),
			time_stat.avg,
			tflops_stat.avg
		);
	}
}



int main()
{
	//先打印一下顺便预热
	hello_sgemm();

	test_correctness(sgemm_v1,"sgemm_v1",128,256,512);
	test_correctness(sgemm_v2,"sgemm_v2",128,256,512);
	test_correctness(sgemm_v3,"sgemm_v3",128,256,512);
	test_correctness(sgemm_v4,"sgemm_v4",128,256,512);
	test_correctness(sgemm_v5,"sgemm_v5",128,256,512);
	test_correctness(sgemm_v6, "sgemm_v6",128,256,512);
	//test_correctness(sgemm_v7, "sgemm_v7",128,128,128);
	test_correctness(sgemm_v7, "sgemm_v7",128,256,512);
	test_correctness(sgemm_v8, "sgemm_v8",128,256,512);
	test_correctness(sgemm_v9, "sgemm_v9",128,256,512);

	test_correctness(sgemm_zhihu, "sgemm_zhihu",128,256,512);

	test_speed(sgemm_v9, "sgemm_v9",2048,2048,2048,1);

	// test_correctness(sgemm_cublas,"sgemm_cublas",128,256,512);
	
	// test_correctness(sgemm_v5,"sgemm_v5",1024,1024,1024);
	// test_correctness(sgemm_v6, "sgemm_v6",1024,1024,1024);
	
	test_speed(sgemm_v1,"sgemm_v1",4096,4096,4096,10);
	test_speed(sgemm_v2,"sgemm_v2",4096,4096,4096,10);
	test_speed(sgemm_v3,"sgemm_v3",4096,4096,4096,10);
	test_speed(sgemm_v4,"sgemm_v4",4096,4096,4096,10);
	test_speed(sgemm_v5,"sgemm_v5",4096,4096,4096,10);
	test_speed(sgemm_v6,"sgemm_v6",4096,4096,4096,10);
	test_speed(sgemm_v7,"sgemm_v7",4096,4096,4096,10);
	test_speed(sgemm_v8, "sgemm_v8",4096,4096,4096,10);
	test_speed(sgemm_v9, "sgemm_v9",4096,4096,4096,10);
	test_speed(sgemm_cublas,"sgemm_cublas",4096,4096,4096,10);
	test_speed(sgemm_cublas,"sgemm_cublas",4096,4096,4096,10);
	test_speed(sgemm_zhihu, "sgemm_zhihu",4096,4096,4096,10);

	test_speed(sgemm_v7,"sgemm_v7",8192,8192,8192,10);
	test_speed(sgemm_cublas,"sgemm_cublas",8192,8192,8192,10);
	
	// test_speed(sgemm_v4,"sgemm_v4",16384,16384,16384,1);
	// test_speed(sgemm_v5,"sgemm_v5",16384,16384,16384,1);
	// test_speed(sgemm_cublas,"sgemm_cublas",16384,16384,16384,1);

}
