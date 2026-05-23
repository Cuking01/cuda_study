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
		data[i] = __float2half((i%128+2*(i/128%128))*0.01f);
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

				score[j] = sum;
				
			}

			for(int j=0;j<=i;j++)
			{
				score[j] *= scale;
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

struct Stat
{
	double avg;
	double stddev;
};

Stat calc_stat(const std::vector<double>& values)
{
	double sum = 0.0;
	for (double value : values)
		sum += value;

	double avg = sum / values.size();
	double sq_sum = 0.0;
	for (double value : values)
	{
		double diff = value - avg;
		sq_sum += diff * diff;
	}

	double stddev = values.size() > 1 ? sqrt(sq_sum / (values.size() - 1)) : 0.0;
	return {avg, stddev};
}

typedef void (*fa_func)(
	cudaStream_t stream,
	const __half* q,
	const __half* k,
	const __half* v,
	__half* o,
	u2 n,
	u2 heads
);

void test_correctness(fa_func fa, std::string name, u2 n, u2 heads)
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

void test_speed(fa_func fa, std::string name, u2 n, u2 heads, int times = 1)
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

	double flops = 2.0 * HEAD_DIM * heads * n * (n + 1);
	std::vector<double> time_ms;
	std::vector<double> tflops;
	time_ms.reserve(times);
	tflops.reserve(times);

	for (int i = 0; i < times; i++)
	{
		stream->nop()
			->record(start)
			->run_any(fa, q_gpu, k_gpu, v_gpu, o_gpu, n, heads)
			->record(end)
			->synchronize();

		cudaDeviceSynchronize();
		double elapsed = event_duration(start, end);
		time_ms.push_back(elapsed);
		tflops.push_back(flops / elapsed / 1e9);
	}

	Stat time_stat = calc_stat(time_ms);
	Stat tflops_stat = calc_stat(tflops);

	if (time_ms.size() > 1)
	{
		printf("%s avg time: %f +/- %f ms (3stddev)\n%f +/- %f Tflops (3stddev)\n\n",
			name.c_str(),
			time_stat.avg,
			3.0 * time_stat.stddev,
			tflops_stat.avg,
			3.0 * tflops_stat.stddev
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

struct DataSet
{
	std::vector<u2> n;
	std::vector<u2> heads;

	void add(u2 n, u2 heads)
	{
		this->n.push_back(n);
		this->heads.push_back(heads);
	}
};

struct TestSet
{
	std::vector<fa_func> f;
	std::vector<std::string> name;

	void add_fun(fa_func fa, std::string name)
	{
		f.push_back(fa);
		this->name.push_back(name);
	}

	void test_correctness(u2 n, u2 heads) const
	{
		for(int i=0;i<f.size();i++)
			::test_correctness(f[i],name[i],n,heads);
	}

	void test_speed(u2 n, u2 heads, int times = 1) const
	{
		for(int i=0;i<f.size();i++)
			::test_speed(f[i],name[i],n,heads,times);
	}

	void test_correctness(const DataSet& data_set) const
	{
		for(int i=0;i<data_set.n.size();i++)
			test_correctness(data_set.n[i],data_set.heads[i]);
	}
	
	void test_speed(const DataSet& data_set, int times = 1) const
	{
		for(int i=0;i<data_set.n.size();i++)
			test_speed(data_set.n[i],data_set.heads[i],times);
	}
	
};

void cudnn_prewarm(const DataSet& data_set)
{
	puts("cudnn prewarm start\n");
	
	TestSet test_set;
	test_set.add_fun(fa_cudnn,"fa_cudnn");
	test_set.test_speed(data_set,1);

	puts("cudnn prewarm end\n");
}

int main()
{
	hello_fa();

	DataSet data_set;
	for(int i=14;i<=17;i++)
	{
		data_set.add(1<<i,6);
		data_set.add(1<<i,36);
	}

	TestSet test_set_correctness;
	test_set_correctness.add_fun(fa_v1,"fa_v1");
	test_set_correctness.add_fun(fa_v2,"fa_v2");
	test_set_correctness.add_fun(fa_v3,"fa_v3");
	test_set_correctness.add_fun(fa_v4,"fa_v4");
	test_set_correctness.add_fun(fa_cudnn,"fa_cudnn");

	test_set_correctness.test_correctness(512,4);
	test_set_correctness.test_correctness(1024,1);

	TestSet test_set_speed;
	test_set_speed.add_fun(fa_v3,"fa_v3");
	test_set_speed.add_fun(fa_v4,"fa_v4");
	test_set_speed.add_fun(fa_cudnn,"fa_cudnn");

	cudnn_prewarm(data_set);
	test_set_speed.test_speed(data_set,10);

}
