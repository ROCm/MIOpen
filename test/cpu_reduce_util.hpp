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
#ifndef GUARD_CPU_REDUCE_UTIL_HPP
#define GUARD_CPU_REDUCE_UTIL_HPP

#include <half.hpp>
#include <limits>
#include <cmath>
#include <cassert>
#include <miopen/miopen.h>
#include <miopen/reduce_common.hpp>

#include <miopen/bfloat16.hpp>

namespace reduce {

template <typename T>
static inline bool IsNan(T x)
{
    // C++ isnan() is used for float, double and half_float::half
    return (std::isnan(x));
};

template <typename T>
static inline bool float_equal_one(T x)
{
    (void)x;
    static_assert(static_cast<T>(0), "float_equal_one() is not implemented for this data type");
    return false;
};

template <>
inline bool float_equal_one<float>(float x)
{
    return x == 1.0f;
};

template <>
inline bool float_equal_one<double>(double x)
{
    return x == 1.0;
};

template <>
inline bool float_equal_one<half_float::half>(half_float::half x)
{
    return x == convert_type<half_float::half>(1.0f);
};

template <typename T>
static inline bool float_equal_zero(T x)
{
    (void)x;
    static_assert(static_cast<T>(0), "float_equal_zero() is not implemented for this data type");
    return false;
};

template <>
inline bool float_equal_zero<float>(float x)
{
    return x == 0.0f;
};

template <>
inline bool float_equal_zero<double>(double x)
{
    return x == 0.0;
};

template <>
inline bool float_equal_zero<half_float::half>(half_float::half x)
{
    return x == convert_type<half_float::half>(0.0f);
};

template <typename T>
static inline T Sqrt(T a)
{
    return sqrt(a);
};

template <>
inline float Sqrt<float>(float a)
{
    return sqrtf(a);
};

template <>
inline half_float::half Sqrt<half_float::half>(half_float::half a)
{
    return half_float::sqrt(a);
};

template <typename T>
static inline T Abs(T a)
{
    return abs(a);
};

template <>
inline float Abs<float>(float a)
{
    return fabsf(a);
};

template <>
inline double Abs<double>(double a)
{
    return fabs(a);
};

template <>
inline half_float::half Abs<half_float::half>(half_float::half a)
{
    return half_float::abs(a);
};

template <typename compType>
static inline std::function<void(compType&)> PreUnaryOpFn(miopenReduceTensorOp_t op_, int divider)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_NORM1:
        return ([&, divider](compType& a_) {
            a_ = Abs<compType>(a_) / convert_type<compType>(static_cast<float>(divider));
        });
    case MIOPEN_REDUCE_TENSOR_NORM2:
        return ([&, divider](compType& a_) {
            a_ = a_ * a_ / convert_type<compType>(static_cast<float>(divider));
        });
    case MIOPEN_REDUCE_TENSOR_AMAX: return ([&](compType& a_) { a_ = Abs<compType>(a_); });

    case MIOPEN_REDUCE_TENSOR_AVG:
    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_MUL:
    case MIOPEN_REDUCE_TENSOR_MIN:
    case MIOPEN_REDUCE_TENSOR_MAX: return ([&](compType& a_) { (void)a_; });
    }

    return (std::function<void(compType&)>{});
};

template <typename compType>
static inline std::function<void(compType&)> PosUnaryOpFn(miopenReduceTensorOp_t op_, int divider)
{
    (void)divider;
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_NORM2: return ([&](compType& a_) { a_ = Sqrt<compType>(a_); });

    case MIOPEN_REDUCE_TENSOR_AVG:
        return ([&, divider](compType& a_) {
            a_ = a_ / convert_type<compType>(static_cast<float>(divider));
        });

    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_NORM1:
    case MIOPEN_REDUCE_TENSOR_MUL:
    case MIOPEN_REDUCE_TENSOR_MIN:
    case MIOPEN_REDUCE_TENSOR_MAX:
    case MIOPEN_REDUCE_TENSOR_AMAX: return ([&](compType& a_) { (void)a_; });
    }

    return (std::function<void(compType&)>{});
};

template <typename compType>
static inline std::function<void(compType&, compType)> ReduceOpFn(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_AVG:
    case MIOPEN_REDUCE_TENSOR_NORM1:
    case MIOPEN_REDUCE_TENSOR_NORM2: return ([&](compType& a_, compType b_) { a_ = a_ + b_; });

    case MIOPEN_REDUCE_TENSOR_MUL: return ([&](compType& a_, compType b_) { a_ = a_ * b_; });

    case MIOPEN_REDUCE_TENSOR_MIN:
        return ([&](compType& a_, compType b_) {
            if(a_ > b_)
                a_ = b_;
        });

    case MIOPEN_REDUCE_TENSOR_MAX:
    case MIOPEN_REDUCE_TENSOR_AMAX:
        return ([&](compType& a_, compType b_) {
            if(a_ < b_)
                a_ = b_;
        });
    }

    return (std::function<void(compType&, compType)>{});
};

template <typename compType>
static inline std::function<void(compType&, compType, bool& changed)>
ReduceOpFn2(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_MIN:
        return ([&](compType& a_, compType b_, bool& changed) {
            if(a_ > b_)
            {
                a_      = b_;
                changed = true;
            }
            else
                changed = false;
        });

    case MIOPEN_REDUCE_TENSOR_MAX:
    case MIOPEN_REDUCE_TENSOR_AMAX:
        return ([&](compType& a_, compType b_, bool& changed) {
            if(a_ < b_)
            {
                a_      = b_;
                changed = true;
            }
            else
                changed = false;
        });

    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_MUL:
    case MIOPEN_REDUCE_TENSOR_AVG:
    case MIOPEN_REDUCE_TENSOR_NORM1:
    case MIOPEN_REDUCE_TENSOR_NORM2: return (std::function<void(compType&, compType, bool&)>{});
    };

    return (std::function<void(compType&, compType, bool&)>{});
};

template <typename compType>
static inline compType ReduceOpZeroVal(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_AVG:
    case MIOPEN_REDUCE_TENSOR_NORM1:
    case MIOPEN_REDUCE_TENSOR_NORM2: return (convert_type<compType>(0.0f));

    case MIOPEN_REDUCE_TENSOR_MUL: return (convert_type<compType>(1.0f));

    case MIOPEN_REDUCE_TENSOR_MIN: return (std::numeric_limits<compType>::max());

    case MIOPEN_REDUCE_TENSOR_MAX: return (std::numeric_limits<compType>::min());
    case MIOPEN_REDUCE_TENSOR_AMAX: return (convert_type<compType>(0.0f));
    }

    return (convert_type<compType>(0.0f));
};

template <>
inline half_float::half ReduceOpZeroVal<half_float::half>(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_AVG:
    case MIOPEN_REDUCE_TENSOR_NORM1:
    case MIOPEN_REDUCE_TENSOR_NORM2:

    case MIOPEN_REDUCE_TENSOR_MUL: return (convert_type<half_float::half>(1.0f));

    case MIOPEN_REDUCE_TENSOR_MIN:
        return (convert_type<half_float::half>(std::numeric_limits<float>::max()));

    case MIOPEN_REDUCE_TENSOR_MAX:
        return (convert_type<half_float::half>(std::numeric_limits<float>::min()));
    case MIOPEN_REDUCE_TENSOR_AMAX: return (convert_type<half_float::half>(0.0f));
    }

    return (convert_type<half_float::half>(0.0f));
};

template <typename compType>
static inline void binop_with_nan_check(miopenNanPropagation_t nanOpt,
                                        std::function<void(compType&, compType)> opReduce,
                                        compType& accuVal,
                                        compType currVal)
{
    if(nanOpt == MIOPEN_NOT_PROPAGATE_NAN)
        opReduce(accuVal, currVal);
    else
    {
        if(reduce::IsNan(currVal))
            accuVal = currVal;
        else
            opReduce(accuVal, currVal);
    };
};

template <typename compType>
static inline void binop_with_nan_check2(miopenNanPropagation_t nanOpt,
                                         std::function<void(compType&, compType, bool&)> opReduce,
                                         compType& accuVal,
                                         compType currVal,
                                         int& accuIndex,
                                         int currIndex)
{
    if(nanOpt == MIOPEN_NOT_PROPAGATE_NAN)
    {
        bool changed;

        opReduce(accuVal, currVal, changed);

        if(changed)
            accuIndex = currIndex;
    }
    else
    {
        if(reduce::IsNan(currVal))
        {
            accuVal   = currVal;
            accuIndex = currIndex;
        }
        else
        {
            bool changed;

            opReduce(accuVal, currVal, changed);

            if(changed)
                accuIndex = currIndex;
        };
    };
};

}; // end of namespace reduce

#endif
