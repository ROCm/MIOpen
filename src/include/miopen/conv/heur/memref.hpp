/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_MEMREF_HPP_
#define GUARD_MIOPEN_MEMREF_HPP_

#include <stddef.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <cassert>
namespace miopen
{

/*
 * The following definition is taken from 
 * https://github.com/llvm/llvm-project/blob/1c98f984105e552daa83ed8e92c61fba0e401410/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h#L115-L155
 * To keep it in sync with MLIR implementation
 */

template <int N>
void dropFront(int64_t arr[N], int64_t *res) {
  for (unsigned i = 1; i < N; ++i)
    *(res + i - 1) = arr[i];
}

//===----------------------------------------------------------------------===//
// Codegen-compatible structures for StridedMemRef type.
//===----------------------------------------------------------------------===//
/// StridedMemRef descriptor type with static rank.
template <typename T, int N>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
  // This operator[] is extremely slow and only for sugaring purposes.
  StridedMemRefType<T, N - 1> operator[](int64_t idx) {
    StridedMemRefType<T, N - 1> res;
    res.basePtr = basePtr;
    res.data = data;
    res.offset = offset + idx * strides[0];
    dropFront<N>(sizes, res.sizes);
    dropFront<N>(strides, res.strides);
    return res;
  }
};

/// StridedMemRef descriptor type specialized for rank 1.
template <typename T>
struct StridedMemRefType<T, 1> {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
  T &operator[](int64_t idx) { return *(data + offset + idx * strides[0]); }
};

/// StridedMemRef descriptor type specialized for rank 0.
template <typename T>
struct StridedMemRefType<T, 0> {
  T *basePtr;
  T *data;
  int64_t offset;
};

using MemRef2D = StridedMemRefType<float, 2>;
struct Tensor2D
{
    int64_t offset;
    int64_t size0;
    int64_t size1;
    int64_t stride0;
    int64_t stride1;
    std::vector<float> d;
    Tensor2D(int64_t _size0, int64_t _size1, float val)
        : offset(0),
          size0(_size0),
          size1(_size1),
          stride0(size0 * size1),
          stride1(1),
          d(std::vector<float>(size0 * size1, val))
    {
    }
    Tensor2D(int64_t _size0, int64_t _size1) : Tensor2D(_size0, _size1, 0) {}
    float* data() { return d.data(); }
    void print()
    {
        // TODO: Guard with debug
        for(auto idx0 = 0; idx0 < size0; ++idx0)
        {
            for(auto idx1 = 0; idx1 < size1; ++idx1)
            {
                std::cout << d[idx0 * stride0 + idx1 * stride1] << '\t';
            }
            std::cout << std::endl;
        }
    }
};
} // namespace miopen
#endif
