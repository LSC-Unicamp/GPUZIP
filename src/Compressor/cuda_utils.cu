#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>
#include <cmath>
#include <utility>
#include "cuda_utils.hpp"

template <typename T>
struct abs_transform
{
    __host__ __device__
        T
        operator()(const T &x)
    {
        return fabs(x);
    }
};

template <>
struct abs_transform<float>
{
    __host__ __device__ float operator()(const float &x)
    {
        return fabsf(x);
    }
};

template <typename T>
T maxArray(const T *d_ptr, const std::size_t n)
{
    const thrust::device_ptr<const T> dpc = thrust::device_pointer_cast(d_ptr);
    const T init = dpc[0];
    const T maximum = thrust::transform_reduce(thrust::device, dpc, dpc + n, abs_transform<T>(), init, thrust::maximum<T>());
    return maximum;
}

template <typename T>
T minArray(const T *d_ptr, const std::size_t n)
{
    const thrust::device_ptr<const T> dpc = thrust::device_pointer_cast(d_ptr);
    const T init = dpc[0];
    const T minimum = thrust::transform_reduce(thrust::device, dpc, dpc + n, abs_transform<T>(), init, thrust::minimum<T>());
    return minimum;
}

float minFloat(const float *d_ptr, const std::size_t n)
{
    return minArray<float>(d_ptr, n);
}

float maxFloat(const float *d_ptr, const std::size_t n)
{
    return maxArray<float>(d_ptr, n);
}

double minDouble(const double *d_ptr, const std::size_t n)
{
    return minArray<double>(d_ptr, n);
}

double maxDouble(const double *d_ptr, const std::size_t n)
{
    return maxArray<double>(d_ptr, n);
}

template <typename T>
struct meanstd_transform
{
    __host__ __device__
        thrust::pair<T, T>
        operator()(const T &x)
    {
        return thrust::make_pair(x, x * x);
    }
};

template <typename T>
struct meanstd_reduce
{
    __host__ __device__
        thrust::pair<T, T>
        operator()(const thrust::pair<T, T> &x, const thrust::pair<T, T> &y)
    {
        return thrust::make_pair(x.first + y.first, x.second + y.second);
    }
};

template <typename T>
std::pair<T, T> MeanStd(const T *d_ptr, const std::size_t n)
{
    const thrust::device_ptr<const T> dpc = thrust::device_pointer_cast(d_ptr);
    thrust::pair<T, T> ret = thrust::transform_reduce(
        thrust::device, dpc, dpc + n, meanstd_transform<T>(), thrust::make_pair(static_cast<T>(0), static_cast<T>(0)),
        meanstd_reduce<T>());
    const T mean = ret.first / static_cast<T>(n);
    const T var = (ret.second / static_cast<T>(n)) - mean * mean;
    const T std = std::sqrt(var);
    return std::make_pair(mean, std);
}

std::pair<float, float> meanStdFloat(const float *d_ptr, const std::size_t n)
{
    return MeanStd<float>(d_ptr, n);
}

std::pair<double, double> meanStdDouble(const double *d_ptr, const std::size_t n)
{
    return MeanStd<double>(d_ptr, n);
}