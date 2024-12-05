#pragma once
#include <utility>

float maxFloat(const float *d_ptr, const std::size_t n);
double maxDouble(const double *d_ptr, const std::size_t n);
std::pair<float, float> meanStdFloat(const float *d_ptr, const std::size_t n);
std::pair<double, double> meanStdDouble(const double *d_ptr, const std::size_t n);