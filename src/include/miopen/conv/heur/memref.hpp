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

struct MemRef2D
{
    float* aligned;
    float* unaligned;
    int64_t offset;
    int64_t size0;
    int64_t size1;
    int64_t stride0;
    int64_t stride1;
    void print()
    {
        assert(aligned == unaligned);
        for(auto idx0 = 0; idx0 < size0; ++idx0)
        {
            for(auto idx1 = 0; idx1 < size1; ++idx1)
            {
                std::cout << aligned[idx0 * stride0 + idx1 * stride1] << '\t';
            }
            std::cout << std::endl;
        }
    }
};


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
