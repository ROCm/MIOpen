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
#include <miopen/miopen.h>
#include <miopen/reduce_common.hpp>

#include <miopen/bfloat16.hpp>

namespace reduce {

template <typename compType>
static inline std::function<void(compType&, compType)> ReduceOpFn(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD: return ([&](compType& a_, compType b_) { a_ = a_ + b_; });

    case MIOPEN_REDUCE_TENSOR_MUL: return ([&](compType& a_, compType b_) { a_ = a_ * b_; });

    case MIOPEN_REDUCE_TENSOR_MIN:
        return ([&](compType& a_, compType b_) {
            if(a_ > b_)
                a_ = b_;
        });

    case MIOPEN_REDUCE_TENSOR_MAX:
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
    case MIOPEN_REDUCE_TENSOR_MUL: return (std::function<void(compType&, compType, bool&)>{});
    };

    return (std::function<void(compType&, compType, bool&)>{});
};

template <typename compType>
static inline compType ReduceOpZeroVal(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD: return (convert_type<compType>(0.0f));

    case MIOPEN_REDUCE_TENSOR_MUL: return (convert_type<compType>(1.0f));

    case MIOPEN_REDUCE_TENSOR_MIN: return (std::numeric_limits<compType>::max());

    case MIOPEN_REDUCE_TENSOR_MAX: return (std::numeric_limits<compType>::min());
    }

    return (convert_type<compType>(0.0f));
};

template <>
inline half_float::half ReduceOpZeroVal<half_float::half>(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD: return (convert_type<half_float::half>(0.0f));

    case MIOPEN_REDUCE_TENSOR_MUL: return (convert_type<half_float::half>(1.0f));

    case MIOPEN_REDUCE_TENSOR_MIN:
        return (convert_type<half_float::half>(std::numeric_limits<float>::max()));

    case MIOPEN_REDUCE_TENSOR_MAX:
        return (convert_type<half_float::half>(std::numeric_limits<float>::min()));
    }

    return (convert_type<half_float::half>(0.0f));
};

template <typename T>
static inline bool IsNan(T x)
{
    // C++ isnan() is used for float and double
    return (std::isnan(x));
};

template <>
inline bool IsNan<half_float::half>(half_float::half x)
{
    return (half_float::isnan(x));
};

template <typename T>
static inline bool IsFinite(T x)
{
    // C++ isfinite() is used for float and double
    return (std::isfinite(x));
};

template <>
inline bool IsFinite<half_float::half>(half_float::half x)
{
    return (half_float::isfinite(x));
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
    return std::isfinite(x) and x <= 1.0f and x >= 1.0f;
};

template <>
inline bool float_equal_one<double>(double x)
{
    return std::isfinite(x) and x <= 1.0 and x >= 1.0;
};

template <>
inline bool float_equal_one<half_float::half>(half_float::half x)
{
    return half_float::isfinite(x) and x <= convert_type<half_float::half>(1.0f) and
           x >= convert_type<half_float::half>(1.0f);
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
    return std::isfinite(x) and x <= 0.0f and x >= 0.0f;
};

template <>
inline bool float_equal_zero<double>(double x)
{
    return std::isfinite(x) and x <= 0.0 and x >= 0.0;
};

template <>
inline bool float_equal_zero<half_float::half>(half_float::half x)
{
    return half_float::isfinite(x) and x <= convert_type<half_float::half>(0.0f) and
           x >= convert_type<half_float::half>(0.0f);
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
