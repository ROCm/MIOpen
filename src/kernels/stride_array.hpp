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

/// \todo Uncomment when hip RTC accepts std::array -- amberhassaan
// #include <hip/amd_detail/amd_hip_vector_types.h>
// using StrideIndexType = int;
// using Strides3D       = std::array<StrideIndexType, 3>;
// using Strides4D       = std::array<StrideIndexType, 4>;
// using Strides5D       = std::array<StrideIndexType, 5>;
// using Strides6D       = std::array<StrideIndexType, 6>;
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

using StrideIndexType = size_t;
using Strides5D       = MyArray<StrideIndexType, 5u>;
using Strides6D       = MyArray<StrideIndexType, 6u>;

template <typename StrideArray>
__host__ __device__ void printStrideArray(const char* name, const StrideArray& sarr)
{
    printf("%s = [", name);
    for(int i = 0; i < StrideArray::SIZE; ++i)
    {
        printf("%zu,", sarr[i]);
    }
    printf("]\n");
}

template <typename StrideArray>
__host__ __device__ void printStrideArrays(const StrideArray& in_strides,
                                           const StrideArray& wei_strides,
                                           const StrideArray& out_strides)
{

    printStrideArray("in_strides", in_strides);
    printStrideArray("wei_strides", wei_strides);
    printStrideArray("out_strides", out_strides);
}
