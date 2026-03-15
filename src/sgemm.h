#pragma once

void hello_sgemm();

void sgemm_v1(const float* a, const float* b, float* c, int n, int m, int k);
void sgemm_v2(const float* a, const float* b, float* c, int n, int m, int k);
