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
#include "fa.h"

//#define FIXED_DATA

constexpr int HEAD_DIM = 128;

void random_init(std::vector<__half>& data, int seed)
{
#ifndef FIXED_DATA
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(-1, 1);

	for (auto& x : data)
		x = __float2half(dist(rng));
#else
	for (int i = 0; i < data.size(); i++)
		data[i] = __float2half(i * 1.0f);
#endif
}

void fa_ref(const __half* q, const __half* k, const __half* v, __half* o, int n, int heads)
{
	float scale = 1.0f / sqrtf((float)HEAD_DIM);

	for (int h = 0; h < heads; h++)
	{
		const __half* q_head = q + h * n * HEAD_DIM;
		const __half* k_head = k + h * n * HEAD_DIM;
		const __half* v_head = v + h * n * HEAD_DIM;
		__half* o_head = o + h * n * HEAD_DIM;

		for (int i = 0; i < n; i++)
		{
			std::vector<float> score(i + 1);

			float max_score = -INFINITY;

			for (int j = 0; j <= i; j++)
			{
				float sum = 0.0f;

				for (int d = 0; d < HEAD_DIM; d++)
				{
					float qv = __half2float(q_head[i * HEAD_DIM + d]);
					float kv = __half2float(k_head[j * HEAD_DIM + d]);
					sum += qv * kv;
				}

				score[j] = sum * scale;
				max_score = std::max(max_score, score[j]);
			}

			float exp_sum = 0.0f;

			for (int j = 0; j <= i; j++)
			{
				score[j] = expf(score[j] - max_score);
				exp_sum += score[j];
			}

			for (int d = 0; d < HEAD_DIM; d++)
			{
				float sum = 0.0f;

				for (int j = 0; j <= i; j++)
				{
					float p = score[j] / exp_sum;
					float vv = __half2float(v_head[j * HEAD_DIM + d]);
					sum += p * vv;
				}

				o_head[i * HEAD_DIM + d] = __float2half(sum);
			}
		}
	}
}

bool check_equal(__half a, __half b)
{
	float af = __half2float(a);
	float bf = __half2float(b);

	float diff = fabsf(af - bf);
	float scale = std::max(fabsf(af), fabsf(bf));

	return diff < 2e-2f || diff / scale < 2e-2f;
}

typedef void (*fa_func)(
	cudaStream_t stream,
	const __half* q,
	const __half* k,
	const __half* v,
	__half* o,
	int n,
	int heads
);

void test_correctness(fa_func fa, std::string name, int n, int heads)
{
	const size_t size = (size_t)heads * n * HEAD_DIM;
	std::vector<__half> q(size), k(size), v(size);
	std::vector<__half> o(size), o_ref(size);

	random_init(q, 1);
	random_init(k, 2);
	random_init(v, 3);

	GPU_Data<__half> q_gpu(q), k_gpu(k), v_gpu(v), o_gpu(o);

	printf("test correctness %s, n=%d, heads=%d, head_dim=%d\n", name.c_str(), n, heads, HEAD_DIM);

	Stream stream;
	stream->run_any(fa, q_gpu, k_gpu, v_gpu, o_gpu, n, heads);

	fa_ref(q.data(), k.data(), v.data(), o_ref.data(), n, heads);

	stream->synchronize();
	o_gpu.to_host(o.data());

	for (size_t i = 0; i < size; i++)
	{
		if (!check_equal(o[i], o_ref[i]))
		{
			printf(
				"test correctness not passed %s, n=%d, heads=%d, failed at o[%zu]\n"
				"o[%zu]=%f, o_ref[%zu]=%f\n",
				name.c_str(),
				n,
				heads,
				i,
				i,
				__half2float(o[i]),
				i,
				__half2float(o_ref[i])
			);
			return;
		}
	}

	printf("test correctness %s, n=%d, heads=%d, passed\n\n", name.c_str(), n, heads);
}

void test_speed(fa_func fa, std::string name, int n, int heads, int times = 1)
{
	printf("test %s, n=%d, heads=%d, head_dim=%d, %d times\n", name.c_str(), n, heads, HEAD_DIM, times);

	const size_t size = (size_t)heads * n * HEAD_DIM;
	std::vector<__half> q(size), k(size), v(size), o(size);

	random_init(q, 1);
	random_init(k, 2);
	random_init(v, 3);

	GPU_Data<__half> q_gpu(q), k_gpu(k), v_gpu(v), o_gpu(o);

	Stream stream;
	Event start, end;

	float time = 0.0f;

	for (int i = 0; i < times; i++)
	{
		stream->nop()
			->record(start)
			->run_any(fa, q_gpu, k_gpu, v_gpu, o_gpu, n, heads)
			->record(end)
			->synchronize();

		cudaDeviceSynchronize();
		time += event_duration(start, end);
	}

	double flops = 2.0 * HEAD_DIM * heads * n * (n + 1);

	printf("%s avg time: %f ms\n%f Tflops\n\n",
		name.c_str(),
		time / times,
		flops * times / time / 1e9
	);
}

int main()
{
	test_correctness(fa_cudnn, "fa_cudnn", 512, 4);

	test_speed(fa_cudnn, "fa_cudnn", 36*1024, 4, 10);

    test_speed(fa_cudnn, "fa_cudnn", 36*1024*2, 4, 10);
}
