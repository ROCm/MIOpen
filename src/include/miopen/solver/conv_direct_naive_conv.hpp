/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#pragma once

#include <miopen/conv/context.hpp>

#include <string>
#include <array>
#include <algorithm>
#include <vector>
#include <cassert>

namespace miopen {

namespace solver {

bool ConvDirectNaiveConvIsAssemblyKernel(const ExecutionContext&, const ProblemDescription&);
std::string ConvDirectNaiveConvKernelName(const ProblemDescription&);
std::string ConvDirectNaiveConvKernelFile(const ConvolutionContext& ctx,
                                          const ProblemDescription& problem);
std::string ConvDirectNaiveConvCompileOption(const ConvolutionContext& ctx,
                                             const ProblemDescription& problem);
bool ConvDirectNaiveConvIsApplicableByKernelType(const ExecutionContext&,
                                                 const ProblemDescription&);

bool IsInputFp32(const ProblemDescription&);
bool IsInputFp16(const ProblemDescription&);
bool IsInputBfp16(const ProblemDescription&);
bool IsInputInt8(const ProblemDescription&);
bool IsAccFp64(const ProblemDescription&);
bool IsAccInt32(const ProblemDescription&);
bool IsOutputFp32(const ProblemDescription&);
bool IsOutputFp16(const ProblemDescription&);
bool IsOutputBfp16(const ProblemDescription&);
bool IsOutputInt8(const ProblemDescription&);
bool IsOutputInt32(const ProblemDescription&);

int GetGroupStrideIndex(const ProblemDescription& problem);

void printTensorStrides(const TensorDescriptor& inDesc,
                        const TensorDescriptor& wDesc,
                        const TensorDescriptor& outDesc);

// TODO(Amber): Uncomment when hip RTC accepts std::array
// using StrideIndexType = int;
// using Strides3D       = std::array<StrideIndexType, 3>;
// using Strides4D       = std::array<StrideIndexType, 4>;
// using Strides5D       = std::array<StrideIndexType, 5>;
// using Strides6D       = std::array<StrideIndexType, 6>;
#if 1
template <typename T, unsigned N>
class MyArray
{
    T data_[N] = {};

public:
    constexpr static const unsigned SIZE = N;
    __host__ __device__ constexpr unsigned size() const { return N; }

    __host__ __device__ const T& operator[](unsigned i) const { return data_[i]; }

    __host__ T& operator[](unsigned i) { return data_[i]; }

    __host__ __device__ MyArray()                   = default;
    __host__ __device__ MyArray(const MyArray&)     = default;
    __host__ __device__ MyArray(MyArray&&) noexcept = default;
    __host__ __device__ MyArray& operator=(const MyArray&) = default;
    __host__ __device__ MyArray& operator=(MyArray&&) noexcept = default;
    __host__ __device__ ~MyArray()                             = default;
};

using StrideIndexType = int;
using Strides5D       = MyArray<StrideIndexType, 5u>;
using Strides6D       = MyArray<StrideIndexType, 6u>;

#else

extern "C" typedef int StrideIndexType;

extern "C" typedef struct
{
    StrideIndexType v[5];
} Strides5D;

extern "C" typedef struct
{
    StrideIndexType v[6];
} Strides6D;

#endif

namespace internal {
template <unsigned N>
struct ChooseStride
{
};

template <>
struct ChooseStride<5u>
{
    using type = Strides5D;
};

template <>
struct ChooseStride<6u>
{
    using type = Strides6D;
};

} // end namespace internal

template <unsigned N, typename V>
auto MakeStrideArray(V vec)
{
    typename internal::ChooseStride<N>::type ret;
    assert(vec.size() == N);

    // MIOpen stores strides for NHWC in NCHW order, i.e. C stride in 2nd from left.
    // We sort the input stride vector so that smallest stride is at index 0. This
    // (little-endian) order is what naive convolution kernel expects for strides
    std::sort(vec.begin(), vec.end());

    for(unsigned i = 0; i < N; ++i)
    {
        ret[i] = static_cast<StrideIndexType>(vec[i]);
    }
    return ret;
}

/**
 * split the strides for C dimension in a tensor descriptor into (G, C_per_group).
 * Normally, (in packed case) num channels is a multiplying factor in the stride of
 * whatever lies to the left of C, e.g., in NCHW, N's stride contains C as a
 * factor. We output NGCHW for NCHW (and NHWGC for NHWC)
 * where the stride[G] = stride[N] / num_groups
 */
template <typename V>
V SplitStrideCtoGC(int num_groups, const V& orig_strides, int G_stride_idx)
{
    assert(G_stride_idx > 0 && G_stride_idx <= orig_strides.size());
    // (G_stride_idx - 1) is the stride index of whatever lies to the left and
    // contains C or K as a multiplying factor. We divide this value by num_groups
    // to get G_stride_val
    assert(orig_strides[G_stride_idx - 1] % num_groups == 0);

    V ret{orig_strides};
    auto G_stride_val = orig_strides[G_stride_idx - 1] / num_groups;

    ret.insert(ret.begin() + G_stride_idx, G_stride_val);

    return ret;
}

/**
 * Weight tensor has original dims: [K, C_per_group, Y, X] (2D case)
 * We return a new stride vector with strides for [G, K_per_group, C_per_group, Y, X]
 * Stride for G is computed as stride[C_per_group] * K_per_group and inserted at
 * left most position
 */
template <typename V>
V SplitWeiStrideKtoGK(int k_per_group, const V& wei_strides)
{
    V ret{wei_strides};
    ret.insert(ret.begin(), wei_strides[0] * k_per_group);
    return ret;
}

template <typename StrideArray>
void printStrideArray(const char* name, const StrideArray& sarr)
{
    printf("%s = [", name);
    for(unsigned i = 0; i < StrideArray::SIZE; ++i)
    {
        printf("%d,", sarr[i]);
    }
    printf("]\n");
}

template <typename StrideArray>
void printStrideArrays(const StrideArray& in_strides,
                       const StrideArray& wei_strides,
                       const StrideArray& out_strides)
{

    printStrideArray("in_strides", in_strides);
    printStrideArray("wei_strides", wei_strides);
    printStrideArray("out_strides", out_strides);
}

} // namespace solver
} // namespace miopen
