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

template <typename compType>
static inline std::function<void(compType&)> PreUnaryOpFn(miopenReduceTensorOp_t op_, std::size_t)
{
    using std::abs;

    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_NORM1: return ([&](compType& a_) { a_ = abs(a_); });
    case MIOPEN_REDUCE_TENSOR_NORM2: return ([&](compType& a_) { a_ = a_ * a_; });
    case MIOPEN_REDUCE_TENSOR_AMAX: return ([&](compType& a_) { a_ = abs(a_); });

    case MIOPEN_REDUCE_TENSOR_AVG:
    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_MUL:
    case MIOPEN_REDUCE_TENSOR_MIN:
    case MIOPEN_REDUCE_TENSOR_MAX: return ([&](compType&) {});
    }

    throw std::runtime_error(std::string(__FUNCTION__) +
                             ": using undefined Reduction operation is not permitted");
};

template <typename compType>
static inline std::function<void(compType&)> PosUnaryOpFn(miopenReduceTensorOp_t op_,
                                                          std::size_t divider)
{
    using std::sqrt;

    switch(op_)
    {
    case MIOPEN_REDUCE_TENSOR_NORM2: return ([&](compType& a_) { a_ = sqrt(a_); });

    case MIOPEN_REDUCE_TENSOR_AVG:
        return ([&, divider](compType& a_) {
            a_ = a_ / convert_type<compType>(static_cast<float>(divider));
        });

    case MIOPEN_REDUCE_TENSOR_ADD:
    case MIOPEN_REDUCE_TENSOR_NORM1:
    case MIOPEN_REDUCE_TENSOR_MUL:
    case MIOPEN_REDUCE_TENSOR_MIN:
    case MIOPEN_REDUCE_TENSOR_MAX:
    case MIOPEN_REDUCE_TENSOR_AMAX: return ([&](compType&) {});
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

template <typename T>
static void
get_all_indexes(const std::vector<T>& dimLengths, int dim, std::vector<std::vector<T>>& indexes)
{
    if(dim < dimLengths.size())
    {
        std::vector<std::vector<T>> updated_indexes;

        if(dim == 0)
        {
            assert(indexes.size() == 0);
            assert(dimLengths[dim] > 0);
            for(T i = 0; i < dimLengths[dim]; i++)
            {
                std::vector<T> index = {i};

                updated_indexes.push_back(index);
            };
        }
        else
        {
            // go through all the current indexes
            for(const auto& index : indexes)
                for(T i = 0; i < dimLengths[dim]; i++)
                {
                    auto index_new = index;
                    index_new.push_back(i);

                    updated_indexes.push_back(index_new);
                };
        };

        // update to the indexes (output)
        indexes = updated_indexes;

        // further to construct the indexes from the updated status
        get_all_indexes(dimLengths, dim + 1, indexes);
    };
};

template <typename T>
static T get_offset_from_index(const std::vector<T>& strides, const std::vector<T>& index)
{
    T offset = 0;

    assert(strides.size() == index.size());

    for(int i = 0; i < index.size(); i++)
        offset += strides[i] * index[i];

    return (offset);
};

template <typename T>
static T get_flatten_offset(const std::vector<T>& lengths, const std::vector<T>& index)
{
    T offset = 0;

    assert(lengths.size() == index.size() && lengths.size() > 0);

    int len  = lengths.size();
    T stride = 1;

    // for len==1, the loop is not executed
    for(int i = len - 1; i > 0; i--)
    {
        offset += stride * index[i];

        stride *= lengths[i];
    };

    offset += stride * index[0];

    return (offset);
};

#endif
