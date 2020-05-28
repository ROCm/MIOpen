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
#ifndef _CK_REDUCTION_COMMON_HPP_
#define _CK_REDUCTION_COMMON_HPP_ 1

#include <half.hpp>

using float16 = half_float::half;

// this enumerate should be synchronized with include/miopen/reduce_common.hpp
namespace ck {
typedef  enum {
      CK_Reduce_DirectThreadWise=1, 
      CK_Reduce_DirectWarpWise=2,
      CK_Reduce_BlockWise=3,
      CK_Reduce_MultiBlock=4
} ckReductionMethod_t; // end of namespace ck 

// this enumerate should be synchronized with include/miopen.h
typedef enum {
    CK_REDUCE_TENSOR_ADD = 0, 
    CK_REDUCE_TENSOR_MUL = 1, 
    CK_REDUCE_TENSOR_MIN = 2, 
    CK_REDUCE_TENSOR_MAX = 3, 
    //CK_REDUCE_TENSOR_AMAX = 4,
    //CK_REDUCE_TENSOR_AVG =  5, 
    //CK_REDUCE_TENSOR_NORM1 = 6, 
    //CK_REDUCE_TENSOR_NORM2 = 7, 
    //CK_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
} ckReduceTensorOp_t; 

// this enumerate should be synchronized with include/miopen.h
typedef enum {
    CK_NOT_PROPAGATE_NAN = 0,
    CK_PROPAGATE_NAN     = 1,
} ckNanPropagation_t;

// this enumerate should be synchronized with include/miopen.h
typedef enum {
    CK_REDUCE_TENSOR_NO_INDICES        = 0,
    CK_REDUCE_TENSOR_FLATTENED_INDICES = 1,
} ckReduceTensorIndices_t;

// this enumerate should be synchronized with include/miopen.h
typedef enum {
    CK_32BIT_INDICES = 0,
    CK_64BIT_INDICES = 1,
    CK_16BIT_INDICES = 2,
    CK_8BIT_INDICES  = 3,
} ckIndicesType_t;

// this enumerate should be synchronized with include/miopen.h
typedef enum {
    ckHalf  = 0,
    ckFloat = 1, 
    ckInt32 = 2, 
    ckInt8  = 3, 
    ckInt8x4 = 4,
    ckBFloat16 = 5, 
    ckDouble = 6,
} ckDataType_t; 

template <ckDataType_t typeNum> 
struct get_type_from_type_enum; 

template <>
struct get_type_from_type_enum<ckHalf> 
{
    using type = float16; 
}; 

template <>
struct get_type_from_type_enum<ckFloat>
{
    using type = float; 
}; 

template <>
struct get_type_from_type_enum<ckDouble>
{
    using type = double; 
}; 

template <>
struct get_type_from_type_enum<ckInt32>
{
    using type = int; 
}; 

}; // end of namespace ck
#endif 

