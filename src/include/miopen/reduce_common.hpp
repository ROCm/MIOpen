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
#ifndef GUARD_MIOPEN_REDUCE_COMMON_HPP
#define GUARD_MIOPEN_REDUCE_COMMON_HPP

#include <half.hpp>
#include <limits>
#include <cmath>

namespace reduce {

typedef enum {
    Reduce_DirectThreadWise = 1,
    Reduce_DirectWarpWise   = 2,
    Reduce_BlockWise        = 3,
    Reduce_MultiBlock       = 4
} ReductionMethod_t;

// data type conversion
template <typename T>
struct type_convert
{
    template <typename X>
    T operator()(X x) const
    {
        return static_cast<T>(x);
    }
};

template <>
template <>
float type_convert<float>::operator()<half_float::half>(half_float::half x) const
{
    return half_float::half_cast<float>(x);
};

template <>
template <>
half_float::half type_convert<half_float::half>::operator()<float>(float x) const
{
    return half_float::half_cast<half_float::half>(x);
};

template <typename compType>
std::function<compType(compType, compType)> ReduceOpFn(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD: return ([&](compType a_, compType b_) { return a_ + b_; });

    case MIOPEN_REDUCE_TENSOR_MUL: return ([&](compType a_, compType b_) { return a_ * b_; });

    case MIOPEN_REDUCE_TENSOR_MIN:
        return ([&](compType a_, compType b_) {
            return (a_ > b_) ? b_ : a_;
        }); // a is selected when they are equal

    case MIOPEN_REDUCE_TENSOR_MAX:
        return ([&](compType a_, compType b_) {
            return (a_ < b_) ? b_ : a_;
        }); // a is selected when they are equal
    }
};

template <typename compType>
compType ReduceOpZeroVal(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD: return (type_convert<compType>{}(0.0));

    case MIOPEN_REDUCE_TENSOR_MUL: return (type_convert<compType>{}(1.0));

    case MIOPEN_REDUCE_TENSOR_MIN: return (std::numeric_limits<compType>::max());

    case MIOPEN_REDUCE_TENSOR_MAX: return (std::numeric_limits<compType>::min());
    }
};

template <>
half_float::half ReduceOpZeroVal<half_float::half>(miopenReduceTensorOp_t op_)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_ADD: return (type_convert<half_float::half>{}(0.0));

    case MIOPEN_REDUCE_TENSOR_MUL: return (type_convert<half_float::half>{}(1.0));

    case MIOPEN_REDUCE_TENSOR_MIN:
        return (type_convert<half_float::half>{}(std::numeric_limits<float>::max()));

    case MIOPEN_REDUCE_TENSOR_MAX:
        return (type_convert<half_float::half>{}(std::numeric_limits<float>::min()));
    }
};

template <typename T>
bool IsNan(T x)
{
    // C++ isnan() is used for float and double
    return (std::isnan(x));
};

template <>
bool IsNan<half_float::half>(half_float::half x)
{
    return (half_float::isnan(x));
};

#define binop_with_nan_check(nanOpt, opReduce, accuVal, currVal) \
    {                                                            \
        if(nanOpt == MIOPEN_NOT_PROPAGATE_NAN)                   \
            accuVal = opReduce(accuVal, currVal);                \
        else                                                     \
        {                                                        \
            if(IsNan(currVal))                                   \
                accuVal = currVal;                               \
            else                                                 \
                accuVal = opReduce(accuVal, currVal);            \
        };                                                       \
    }

#define binop_with_nan_check2(nanOpt, opReduce, accuVal, currVal, accuIndex, currIndex) \
    {                                                                                   \
        if(nanOpt == MIOPEN_NOT_PROPAGATE_NAN)                                          \
        {                                                                               \
            auto accuVal_new = opReduce(accuVal, currVal);                              \
            if(!miopen::float_equal(accuVal, accuVal_new))                              \
            {                                                                           \
                accuIndex = currIndex;                                                  \
                accuVal   = accuVal_new;                                                \
            };                                                                          \
        }                                                                               \
        else                                                                            \
        {                                                                               \
            decltype(accuVal) accuVal_new;                                              \
            if(IsNan(currVal))                                                          \
                accuVal_new = currVal;                                                  \
            else                                                                        \
                accuVal_new = opReduce(accuVal, currVal);                               \
                                                                                        \
            if(!miopen::float_equal(accuVal, accuVal_new))                              \
            {                                                                           \
                accuIndex = currIndex;                                                  \
                accuVal   = accuVal_new;                                                \
            };                                                                          \
        };                                                                              \
    }

}; // end of namespace reduce

#endif
