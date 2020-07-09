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
#ifndef CK_REDUCTION_COMMON_HPP
#define CK_REDUCTION_COMMON_HPP

#include "config.hpp"

// this enumerate should be synchronized with include/miopen/reduce_common.hpp
namespace ck {
typedef enum {
    Reduce_DirectThreadWise = 1,
    Reduce_DirectWarpWise   = 2,
    Reduce_BlockWise        = 3,
    Reduce_MultiBlock       = 4
} ckReductionMethod_t; // end of namespace ck

typedef enum {
    CK_REDUCE_TENSOR_ADD = 0,
    CK_REDUCE_TENSOR_MUL = 1,
    CK_REDUCE_TENSOR_MIN = 2,
    CK_REDUCE_TENSOR_MAX = 3,
    // CK_REDUCE_TENSOR_AMAX = 4,
    // CK_REDUCE_TENSOR_AVG =  5,
    // CK_REDUCE_TENSOR_NORM1 = 6,
    // CK_REDUCE_TENSOR_NORM2 = 7,
    // CK_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
} ckReduceTensorOp_t;

typedef enum {
    CK_NOT_PROPAGATE_NAN = 0,
    CK_PROPAGATE_NAN     = 1,
} ckNanPropagation_t;

typedef enum {
    CK_REDUCE_TENSOR_NO_INDICES        = 0,
    CK_REDUCE_TENSOR_FLATTENED_INDICES = 1,
} ckReduceTensorIndices_t;

typedef enum {
    CK_32BIT_INDICES = 0,
    CK_64BIT_INDICES = 1,
    CK_16BIT_INDICES = 2,
    CK_8BIT_INDICES  = 3,
} ckIndicesType_t;

struct float_equal
{
    template <class T>
    __device__ static inline bool apply(T x, T y)
    {
        return x <= y and x >= y;
    }

    template <class T>
    __device__ inline bool operator()(T x, T y)
    {
        return (float_equal::apply(x, y));
    };
};

struct float_equal_one
{
    template <class T>
    __device__ static inline bool apply(T x)
    {
        return x <= type_convert<T>{}(1.0f) and x >= type_convert<T>{}(1.0f);
    }

    template <class T>
    __device__ inline bool operator()(T x)
    {
        return (float_equal_one::apply(x));
    };
};

struct float_equal_zero
{
    template <class T>
    __device__ static inline bool apply(T x)
    {
        return x <= type_convert<T>{}(0.0f) and x >= type_convert<T>{}(0.0f);
    }

    template <class T>
    __device__ inline bool operator()(T x)
    {
        return (float_equal_zero::apply(x));
    };
};

}; // end of namespace ck

#endif
