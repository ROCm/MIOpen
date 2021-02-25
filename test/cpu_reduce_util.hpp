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
#include <stdexcept>
#include <string>
#include <miopen/miopen.h>
#include <miopen/reduce_common.hpp>

namespace reduce {

template <typename T>
static inline bool float_equal_one(T);

static inline bool float_equal_one(float x) { return x == 1.0f; };

static inline bool float_equal_one(double x) { return x == 1.0; };

static inline bool float_equal_one(half_float::half x)
{
    return x == convert_type<half_float::half>(1.0f);
};

template <typename T>
static inline bool float_equal_zero(T x);

static inline bool float_equal_zero(float x) { return x == 0.0f; };

static inline bool float_equal_zero(double x) { return x == 0.0; };

static inline bool float_equal_zero(half_float::half x)
{
    return x == convert_type<half_float::half>(0.0f);
};

template <typename T>
static inline T Sqrt(T a);

static inline float Sqrt(float a) { return sqrtf(a); };

static inline double Sqrt(double a) { return sqrt(a); };

static inline half_float::half Sqrt(half_float::half a) { return half_float::sqrt(a); };

template <typename T>
static inline T Abs(T a);

static inline float Abs(float a) { return fabsf(a); };

static inline double Abs(double a) { return fabs(a); };

static inline half_float::half Abs(half_float::half a) { return half_float::abs(a); };

template <typename compType>
static inline std::function<void(compType&)> PreUnaryOpFn(miopenReduceTensorOp_t op_, int divider)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_NORM1:
        return ([&, divider](compType& a_) {
            a_ = Abs(a_) / convert_type<compType>(static_cast<float>(divider));
        });
    case MIOPEN_REDUCE_TENSOR_NORM2:
        return ([&, divider](compType& a_) {
            a_ = a_ * a_ / convert_type<compType>(static_cast<float>(divider));
        });
    case MIOPEN_REDUCE_TENSOR_AMAX: return ([&](compType& a_) { a_ = Abs(a_); });

    case MIOPEN_REDUCE_TENSOR_AVG:
    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_MUL:
    case MIOPEN_REDUCE_TENSOR_MIN:
    case MIOPEN_REDUCE_TENSOR_MAX: return ([&](compType& a_) { (void)a_; });
    }

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
};

template <typename compType>
static inline std::function<void(compType&)> PosUnaryOpFn(miopenReduceTensorOp_t op_, int divider)
{
    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_NORM2: return ([&](compType& a_) { a_ = Sqrt(a_); });

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

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
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

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
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

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
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

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
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

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
};

template <typename compType>
static inline void binop_with_nan_check(miopenNanPropagation_t nanOpt,
                                        std::function<void(compType&, compType)> opReduce,
                                        compType& accuVal,
                                        compType currVal)
{
    using std::isnan;

    if(nanOpt == MIOPEN_NOT_PROPAGATE_NAN)
        opReduce(accuVal, currVal);
    else
    {
        if(isnan(currVal))
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
    using std::isnan;

    if(nanOpt == MIOPEN_NOT_PROPAGATE_NAN)
    {
        bool changed;

        opReduce(accuVal, currVal, changed);

        if(changed)
            accuIndex = currIndex;
    }
    else
    {
        if(isnan(currVal))
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
