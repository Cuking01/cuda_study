#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "fa.h"
#include "tool.h"

constexpr int HEAD_DIM = 128;
constexpr int HEADS = 1;
constexpr int CPU_REF_MAX_N = 2560;
constexpr float ABS_TOL = 2.0e-2f;
constexpr float REL_TOL = 2.0e-2f;

typedef void (*fa_func)(
	cudaStream_t stream,
	const __half* q,
	const __half* k,
	const __half* v,
	__half* o,
	u2 n,
	u2 heads
);

struct FaImpl
{
	fa_func func;
	const char* name;
};

struct DataCase
{
	const char* name;
	float range;
	int seed;
	int mode;
};

struct CompareResult
{
	size_t bad = 0;
	size_t first_bad = 0;
	float first_actual = 0.0f;
	float first_expected = 0.0f;
	float max_abs = 0.0f;
	float max_rel = 0.0f;
};

void fill_case(std::vector<__half>& q, std::vector<__half>& k, std::vector<__half>& v, const DataCase& data_case)
{
	const int n = (int)(q.size() / HEAD_DIM);
	std::mt19937 rng(data_case.seed);
	std::uniform_real_distribution<float> dist(-data_case.range, data_case.range);
	std::uniform_real_distribution<float> unit(-1.0f, 1.0f);

	for (int i = 0; i < n; i++)
	{
		for (int d = 0; d < HEAD_DIM; d++)
		{
			size_t idx = (size_t)i * HEAD_DIM + d;
			float qv = 0.0f;
			float kv = 0.0f;
			float vv = unit(rng);

			if (data_case.mode == 0)
			{
				qv = dist(rng);
				kv = dist(rng);
				vv = unit(rng);
			}
			else if (data_case.mode == 1)
			{
				qv = 0.0f;
				kv = 0.0f;
				vv = ((i + d) % 17 - 8) * 0.25f;
			}
			else if (data_case.mode == 2)
			{
				qv = data_case.range;
				kv = data_case.range;
				vv = unit(rng);
			}
			else if (data_case.mode == 3)
			{
				qv = data_case.range;
				kv = (i < n / 2) ? -data_case.range : data_case.range;
				vv = unit(rng);
			}
			else if (data_case.mode == 4)
			{
				qv = data_case.range;
				kv = -data_case.range + 2.0f * data_case.range * (float)i / (float)std::max(1, n - 1);
				vv = unit(rng);
			}
			else if (data_case.mode == 5)
			{
				qv = (d % 2 == 0) ? data_case.range : -data_case.range;
				kv = ((i + d) % 2 == 0) ? data_case.range : -data_case.range;
				vv = unit(rng);
			}
			else if (data_case.mode == 6)
			{
				qv = dist(rng);
				kv = dist(rng);
				vv = (d % 2 == 0) ? 20.0f : -20.0f;
			}
			else
			{
				qv = ((i + d) % 3 == 0) ? data_case.range : -data_case.range;
				kv = ((i * 7 + d) % 5 == 0) ? -data_case.range : data_case.range;
				vv = ((i + d) % 11 - 5) * 0.5f;
			}

			q[idx] = __float2half(qv);
			k[idx] = __float2half(kv);
			v[idx] = __float2half(vv);
		}
	}
}

void fa_ref_cpu(const __half* q, const __half* k, const __half* v, __half* o, int n)
{
	const double scale = 1.0 / sqrt((double)HEAD_DIM);

	for (int i = 0; i < n; i++)
	{
		std::vector<double> score(i + 1);
		double max_score = -INFINITY;

		for (int j = 0; j <= i; j++)
		{
			double sum = 0.0;
			for (int d = 0; d < HEAD_DIM; d++)
				sum += (double)__half2float(q[i * HEAD_DIM + d]) * (double)__half2float(k[j * HEAD_DIM + d]);

			score[j] = sum * scale;
			max_score = std::max(max_score, score[j]);
		}

		double exp_sum = 0.0;
		for (int j = 0; j <= i; j++)
		{
			score[j] = exp(score[j] - max_score);
			exp_sum += score[j];
		}

		for (int d = 0; d < HEAD_DIM; d++)
		{
			double sum = 0.0;
			for (int j = 0; j <= i; j++)
				sum += score[j] / exp_sum * (double)__half2float(v[j * HEAD_DIM + d]);

			o[i * HEAD_DIM + d] = __float2half((float)sum);
		}
	}
}

void fa_ref_cudnn(const std::vector<__half>& q, const std::vector<__half>& k, const std::vector<__half>& v, std::vector<__half>& o, int n)
{
	GPU_Data<__half> q_gpu(q), k_gpu(k), v_gpu(v), o_gpu(o);
	Stream stream;
	stream->run_any(fa_cudnn, q_gpu, k_gpu, v_gpu, o_gpu, n, HEADS)->synchronize();
	o_gpu.to_host(o.data());
}

void run_fa(const FaImpl& impl, const std::vector<__half>& q, const std::vector<__half>& k, const std::vector<__half>& v, std::vector<__half>& o, int n)
{
	GPU_Data<__half> q_gpu(q), k_gpu(k), v_gpu(v), o_gpu(o);
	Stream stream;
	stream->run_any(impl.func, q_gpu, k_gpu, v_gpu, o_gpu, n, HEADS)->synchronize();
	o_gpu.to_host(o.data());
}

CompareResult compare_output(const std::vector<__half>& actual, const std::vector<__half>& expected)
{
	CompareResult result;

	for (size_t i = 0; i < actual.size(); i++)
	{
		float af = __half2float(actual[i]);
		float bf = __half2float(expected[i]);
		float diff = fabsf(af - bf);
		float scale = std::max({fabsf(af), fabsf(bf), 1.0e-6f});
		float rel = diff / scale;
		bool pass = std::isfinite(af) && (diff < ABS_TOL || rel < REL_TOL);

		if (diff > result.max_abs)
			result.max_abs = diff;
		if (rel > result.max_rel)
			result.max_rel = rel;

		if (!pass)
		{
			if (result.bad == 0)
			{
				result.first_bad = i;
				result.first_actual = af;
				result.first_expected = bf;
			}
			result.bad++;
		}
	}

	return result;
}

bool test_case(
	const FaImpl& impl,
	int n,
	const DataCase& data_case,
	const std::vector<__half>& q,
	const std::vector<__half>& k,
	const std::vector<__half>& v,
	const std::vector<__half>& expected,
	const char* ref_name)
{
	const size_t size = (size_t)HEADS * n * HEAD_DIM;
	std::vector<__half> actual(size);

	run_fa(impl, q, k, v, actual, n);
	CompareResult result = compare_output(actual, expected);

	printf(
		"test correctness %s, n=%d, case=%s, ref=%s, max_abs=%g, max_rel=%g, bad=%zu\n",
		impl.name,
		n,
		data_case.name,
		ref_name,
		result.max_abs,
		result.max_rel,
		result.bad
	);

	if (result.bad != 0)
	{
		printf(
			"failed %s, n=%d, case=%s, first_bad=%zu, actual=%f, expected=%f\n",
			impl.name,
			n,
			data_case.name,
			result.first_bad,
			result.first_actual,
			result.first_expected
		);
		return false;
	}

	return true;
}

int main()
{
	hello_fa();

	const std::vector<FaImpl> implementations = {
		{fa_v1, "fa_v1"},
		{fa_v2, "fa_v2"},
		{fa_v3, "fa_v3"},
		{fa_v4, "fa_v4"},
	};

	const std::vector<int> ns = {128, 256, 512, 1024, 1536, 2560, 4096, 8192, 16384};
	const std::vector<DataCase> data_cases = {
		{"random_range_1", 1.0f, 101, 0},
		{"random_range_3", 3.0f, 103, 0},
		{"random_range_6", 6.0f, 106, 0},
		{"random_range_10", 10.0f, 110, 0},
		{"zero_qk_pattern_v", 1.0f, 201, 1},
		{"same_large_score", 10.0f, 202, 2},
		{"half_negative_half_positive_k", 10.0f, 203, 3},
		{"monotonic_k", 10.0f, 204, 4},
		{"alternating_qk", 10.0f, 205, 5},
		{"random_qk_large_v", 10.0f, 206, 6},
		{"structured_extreme", 10.0f, 207, 7},
	};

	bool ok = true;
	for (int n : ns)
	{
		for (const DataCase& data_case : data_cases)
		{
			const size_t size = (size_t)HEADS * n * HEAD_DIM;
			std::vector<__half> q(size), k(size), v(size), expected(size);

			fill_case(q, k, v, data_case);

			const char* ref_name = n <= CPU_REF_MAX_N ? "cpu" : "cudnn";
			if (n <= CPU_REF_MAX_N)
				fa_ref_cpu(q.data(), k.data(), v.data(), expected.data(), n);
			else
				fa_ref_cudnn(q, k, v, expected, n);

			for (const FaImpl& impl : implementations)
				ok &= test_case(impl, n, data_case, q, k, v, expected, ref_name);
		}
	}

	if (ok)
		puts("all fa correctness tests passed");
	else
		puts("some fa correctness tests failed");

	return ok ? 0 : 1;
}
