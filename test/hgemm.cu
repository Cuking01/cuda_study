#include <stdio.h>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <exception>
#include <stdexcept>
#include <time.h>
#include <format>
#include <cuda_fp16.h>

#include "tool.h"
#include "hgemm.h"

//#define FIXED_DATA

void random_init(std::vector<__half>& data, int seed)
{
#ifndef FIXED_DATA
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(0, 1);

	for (auto& x : data)
		x = __float2half(dist(rng));
#else
	for (int i = 0; i < data.size(); i++)
		data[i] = __float2half(i*1.0f);
#endif
}

void hgemm_ref(const __half* a, const __half* b, __half* c, int n, int m, int k)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < k; j++)
		{
			float sum = 0.0f;

			for (int l = 0; l < m; l++)
			{
				float av = __half2float(a[i * m + l]);
				float bv = __half2float(b[l * k + j]);
				sum += av * bv;
			}

			c[i * k + j] = __float2half(sum);
		}
}

bool check_equal(__half a, __half b)
{
	float af = __half2float(a);
	float bf = __half2float(b);

	float diff = fabsf(af - bf);
	float scale = std::max(fabsf(af), fabsf(bf));

	return diff < 1e-2f || diff / scale < 1e-2f;
}

typedef void (*hgemm_func)(
	cudaStream_t stream,
	const __half* a,
	const __half* b,
	__half* c,
	u2 n,
	u2 m,
	u2 k
);

void test_correctness(hgemm_func hgemm, std::string name, int n, int m, int k)
{
	std::vector<half> a(n * m), b(m * k), c(n * k), c_ref(n * k);

	random_init(a, 1);
	random_init(b, 2);

	GPU_Data<half> a_gpu(a), b_gpu(b), c_gpu(c);

	printf("test correctness %s, n=%d, m=%d, k=%d\n", name.c_str(), n, m, k);

	Stream stream;
	stream->run_any(hgemm, a_gpu, b_gpu, c_gpu, n, m, k);

	hgemm_ref(a.data(), b.data(), c_ref.data(), n, m, k);

	stream->synchronize();
	c_gpu.to_host(c.data());

	for (int i = 0; i < n * k; i++)
	{
		if (!check_equal(c[i], c_ref[i]))
		{
			printf(
				"test correctness not passed %s, n=%d, m=%d, k=%d, failed at c[%d]\n"
				"c[%d]=%f, c_ref[%d]=%f\n",
				name.c_str(),
				n,
				m,
				k,
				i,
				i,
				__half2float(c[i]),
				i,
				__half2float(c_ref[i])
			);
			return;
		}
	}

	printf("test correctness %s, n=%d, m=%d, k=%d, passed\n\n", name.c_str(), n, m, k);
}

void test_speed(hgemm_func hgemm, std::string name, int n, int m, int k, int times = 1)
{
	printf("test %s, n=%d, m=%d, k=%d, %d times\n", name.c_str(), n, m, k, times);

	std::vector<__half> a(n * m), b(m * k), c(n * k);

	random_init(a, 1);
	random_init(b, 2);

	GPU_Data<__half> a_gpu(a), b_gpu(b), c_gpu(c);

	Stream stream;
	Event start, end;

	float time = 0.0f;

	for (int i = 0; i < times; i++)
	{
		stream->nop()
			->record(start)
			->run_any(hgemm, a_gpu, b_gpu, c_gpu, n, m, k)
			->record(end)
			->synchronize();

		cudaDeviceSynchronize();
		time += event_duration(start, end);
	}

	printf("%s avg time: %f ms\n%f Tflops\n\n",
		name.c_str(),
		time / times,
		2.0 * n * m * k * times / time / 1e9
	);
}

int main()
{
    hello_hgemm();
	test_speed(hgemm_cublas,"hgemm_cublas",4096,4096,4096,10);


	test_correctness(hgemm_v1,"hgemm_v1",1024,512,1024+256);
	test_correctness(hgemm_v2,"hgemm_v2",1024,512,1024+256);
	test_correctness(hgemm_v3,"hgemm_v3",1024,512,1024+256);
	test_correctness(hgemm_v4,"hgemm_v4",1024,512,1024+256);
	test_correctness(hgemm_v5,"hgemm_v5",1024,512,1024+256);

	test_speed(hgemm_v1,"hgemm_v1",4096,4096,4096,10);
	test_speed(hgemm_v2,"hgemm_v2",4096,4096,4096,10);
	test_speed(hgemm_v3,"hgemm_v3",4096,4096,4096,10);
	test_speed(hgemm_v4,"hgemm_v4",4096,4096,4096,10);
	test_speed(hgemm_v5,"hgemm_v5",4096,4096,4096,10);
	test_speed(hgemm_cublas,"hgemm_cublas",4096,4096,4096,10);

	test_speed(hgemm_v1,"hgemm_v1",6144,6144,6144,10);
	test_speed(hgemm_v2,"hgemm_v2",6144,6144,6144,10);
	test_speed(hgemm_v3,"hgemm_v3",6144,6144,6144,10);
	test_speed(hgemm_v4,"hgemm_v4",6144,6144,6144,10);
	test_speed(hgemm_v5,"hgemm_v5",6144,6144,6144,10);
	test_speed(hgemm_cublas,"hgemm_cublas",6144,6144,6144,10);

	test_speed(hgemm_v1,"hgemm_v1",8192,8192,8192,10);
	test_speed(hgemm_v2,"hgemm_v2",8192,8192,8192,10);
	test_speed(hgemm_v3,"hgemm_v3",8192,8192,8192,10);
	test_speed(hgemm_v4,"hgemm_v4",8192,8192,8192,10);
	test_speed(hgemm_v5,"hgemm_v5",8192,8192,8192,10);
	test_speed(hgemm_cublas,"hgemm_cublas",8192,8192,8192,10);
}
