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
#ifndef GUARD_MIOPEN_REDUCTION_HOST_HPP_
#define GUARD_MIOPEN_REDUCTION_HOST_HPP_

#include <vector>
#include <functional>
#include <limits>
#include <type_traits>
#include <cassert>
#include <cmath>

#include "../test/cpu_reduce_util.hpp"

#include "tensor_driver.hpp"

using float16 = half_float::half;

static inline void
get_all_indexes(const std::vector<int>& dimLengths, int dim, std::vector<std::vector<int>>& indexes)
{
    if(dim < dimLengths.size())
    {
        std::vector<std::vector<int>> updated_indexes;

        if(dim == 0)
        {
            for(int i = 0; i < dimLengths[dim]; i++)
            {
                std::vector<int> index = {i};

                updated_indexes.push_back(index);
            };
        }
        else
        {
            // go through all the current indexes
            for(const auto& index : indexes)
                for(int i = 0; i < dimLengths[dim]; i++)
                {
                    auto index_new = index; // explicit copying

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

static inline int get_offset_from_index(const std::vector<int>& strides,
                                        const std::vector<int>& index)
{
    int offset = 0;

    assert(index.size() == strides.size());
    for(int i = 0; i < index.size(); i++)
        offset += strides[i] * index[i];

    return (offset);
};

static inline int get_flatten_offset(const std::vector<int>& lengths, const std::vector<int>& index)
{
    int offset = 0;

    assert(lengths.size() == index.size() && lengths.size() > 0);

    int len    = lengths.size();
    int stride = 1;

    // for len==1, the loop is not executed
    for(int i = len - 1; i > 0; i--)
    {
        offset += stride * index[i];

        stride *= lengths[i];
    };

    offset += stride * index[0];

    return (offset);
};

template <typename Tgpu, typename Tref>
class miopenReductionHost
{
    public:
    miopenReductionHost() = default;
    miopenReductionHost(const miopenReduceTensorDescriptor_t reduceDesc,
                        miopenTensorDescriptor_t inDesc,
                        miopenTensorDescriptor_t outDesc,
                        const std::vector<int>& invariantDims_,
                        const std::vector<int>& toReduceDims_)
    {
        miopenGetReduceTensorDescriptor(
            reduceDesc, &reduceOp, &compTypeVal, &nanOpt, &indicesOpt, &indicesType);

        this->inLengths  = GetTensorLengths(inDesc);
        this->outLengths = GetTensorLengths(outDesc);
        this->inStrides  = GetTensorStrides(inDesc);
        this->outStrides = GetTensorStrides(outDesc);

        this->invariantDims = invariantDims_;
        this->toReduceDims  = toReduceDims_;

        assert(this->inLengths.size() == this->outLengths.size());
        assert(!this->toReduceDims.empty());

        for(const auto dim : this->invariantDims)
            this->invariantLengths.push_back(this->inLengths[dim]);

        for(const auto dim : this->toReduceDims)
            toReduceLengths.push_back(this->inLengths[dim]);

        this->reduceAllDims = this->invariantDims.empty();
    };

    ~miopenReductionHost(){};

    void Run(Tgpu alpha, const Tgpu* in_data, Tgpu beta, Tref* out_data, int* indices)
    {
        if(compTypeVal == miopenFloat)
        {
            if(std::is_same<Tref, double>::value)
                RunImpl<double>(alpha, in_data, beta, out_data, indices);
            else
                RunImpl<float>(alpha, in_data, beta, out_data, indices);
        }
        else if(compTypeVal == miopenHalf)
        {
            if(std::is_same<Tref, double>::value || std::is_same<Tref, float>::value)
                RunImpl<Tref>(alpha, in_data, beta, out_data, indices);
            else
                RunImpl<float16>(alpha, in_data, beta, out_data, indices);
        };

        return;
    };

    private:
    miopenReduceTensorOp_t reduceOp;
    miopenDataType_t compTypeVal;
    miopenNanPropagation_t nanOpt;
    miopenReduceTensorIndices_t indicesOpt;
    miopenIndicesType_t indicesType;

    std::vector<int> inLengths;
    std::vector<int> outLengths;
    std::vector<int> inStrides;
    std::vector<int> outStrides;

    std::vector<int> invariantLengths;
    std::vector<int> toReduceLengths;

    std::vector<int> invariantDims;
    std::vector<int> toReduceDims;

    bool reduceAllDims;

    template <typename compType>
    void RunImpl(Tgpu alpha, const Tgpu* in_data, Tgpu beta, Tref* out_data, int* indices)
    {
        bool need_indices =
            (indicesOpt == MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES) &&
            (reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX);

        if(need_indices)
            RunImpl_with_indices<compType>(alpha, in_data, beta, out_data, indices);
        else
            RunImpl_no_indices<compType>(alpha, in_data, beta, out_data);
    };

    template <typename compType>
    void
    RunImpl_with_indices(Tgpu alpha, const Tgpu* in_data, Tgpu beta, Tref* out_data, int* indices)
    {
        using reduce::ReduceOpFn2;
        using reduce::ReduceOpZeroVal;
        using reduce::float_equal_one;
        using reduce::float_equal_zero;
        using reduce::convert_type;
        using reduce::binop_with_nan_check;
        using reduce::binop_with_nan_check2;

        auto opReduce = ReduceOpFn2<compType>(this->reduceOp);

        if(reduceAllDims)
        {
            std::vector<std::vector<int>> indexes_1;

            get_all_indexes(inLengths, 0, indexes_1); // generate the input indexes space

            auto accuVal  = ReduceOpZeroVal<compType>(this->reduceOp);
            int accuIndex = 0;

            // go through indexes of the invariant dimensions
            for(const auto& src_index : indexes_1)
            {
                auto src_offset = get_offset_from_index(this->inStrides, src_index);

                auto currVal = convert_type<compType>(in_data[src_offset]);

                auto currIndex = get_flatten_offset(inLengths, src_index);
                binop_with_nan_check2(nanOpt, opReduce, accuVal, currVal, accuIndex, currIndex);
            };

            // scale the accumulated value
            if(!float_equal_one(alpha))
                accuVal *= convert_type<compType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!float_equal_zero(beta))
                accuVal += convert_type<compType>(out_data[0] * convert_type<Tref>(beta));

            // store the reduced value to dst location
            out_data[0] = convert_type<Tref>(accuVal);
            indices[0]  = accuIndex;
        }
        else
        {
            std::vector<std::vector<int>> indexes_1, indexes_2;

            get_all_indexes(
                this->invariantLengths, 0, indexes_1); // generate the invariant indexes space
            get_all_indexes(
                this->toReduceLengths, 0, indexes_2); // generate the toReduce indexes space

            // go through indexes of the invariant dimensions
            for(const auto& index_1 : indexes_1)
            {
                std::vector<int> src_index;
                std::vector<int> dst_index;

                src_index.resize(this->inLengths.size());
                dst_index.resize(this->inLengths.size());

                // initialize the src index
                std::fill(dst_index.begin(), dst_index.end(), 0);

                for(int k                       = 0; k < invariantDims.size(); k++)
                    dst_index[invariantDims[k]] = index_1[k];

                int dst_offset = get_offset_from_index(this->outStrides, dst_index);

                // generate the part of src index belonging to invariant dims
                for(int k                       = 0; k < invariantDims.size(); k++)
                    src_index[invariantDims[k]] = index_1[k];

                compType accuVal = ReduceOpZeroVal<compType>(this->reduceOp);
                int accuIndex    = 0;

                // go through indexes of the toReduce dimensions
                for(const auto& index_2 : indexes_2)
                {
                    // generate the part of src index belonging to toReduce dims
                    for(int k                      = 0; k < toReduceDims.size(); k++)
                        src_index[toReduceDims[k]] = index_2[k];

                    auto src_offset = get_offset_from_index(this->inStrides, src_index);

                    auto currVal = convert_type<compType>(in_data[src_offset]);

                    auto currIndex = get_flatten_offset(toReduceLengths, index_2);
                    binop_with_nan_check2(nanOpt, opReduce, accuVal, currVal, accuIndex, currIndex);
                };

                // scale the accumulated value
                if(!float_equal_one(alpha))
                    accuVal *= convert_type<compType>(alpha);

                // scale the prior dst value and add it to the accumulated value
                if(!float_equal_zero(beta))
                    accuVal +=
                        convert_type<compType>(out_data[dst_offset] * convert_type<Tref>(beta));

                // store the reduced value to dst location
                out_data[dst_offset] = convert_type<Tref>(accuVal);
                indices[dst_offset]  = accuIndex;
            };
        };
    }; // end of RunImpl_with_indices()

    template <typename compType>
    void RunImpl_no_indices(Tgpu alpha, const Tgpu* in_data, Tgpu beta, Tref* out_data)
    {
        using reduce::ReduceOpFn;
        using reduce::ReduceOpZeroVal;
        using reduce::float_equal_one;
        using reduce::float_equal_zero;
        using reduce::convert_type;
        using reduce::binop_with_nan_check;
        using reduce::binop_with_nan_check2;

        auto opReduce = ReduceOpFn<compType>(this->reduceOp);

        if(reduceAllDims)
        {
            std::vector<std::vector<int>> indexes_1;

            get_all_indexes(inLengths, 0, indexes_1); // generate the input indexes space

            auto accuVal = ReduceOpZeroVal<compType>(this->reduceOp);

            // go through indexes of the invariant dimensions
            for(const auto& src_index : indexes_1)
            {
                auto src_offset = get_offset_from_index(this->inStrides, src_index);

                auto currVal = convert_type<compType>(in_data[src_offset]);

                binop_with_nan_check(nanOpt, opReduce, accuVal, currVal);
            };

            // scale the accumulated value
            if(!float_equal_one(alpha))
                accuVal *= convert_type<compType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!float_equal_zero(beta))
                accuVal += convert_type<compType>(out_data[0] * convert_type<Tref>(beta));

            // store the reduced value to dst location
            out_data[0] = convert_type<Tref>(accuVal);
        }
        else
        {
            std::vector<std::vector<int>> indexes_1, indexes_2;

            get_all_indexes(
                this->invariantLengths, 0, indexes_1); // generate the invariant indexes space
            get_all_indexes(
                this->toReduceLengths, 0, indexes_2); // generate the toReduce indexes space

            // go through indexes of the invariant dimensions
            for(const auto& index_1 : indexes_1)
            {
                std::vector<int> src_index;
                std::vector<int> dst_index;

                src_index.resize(this->inLengths.size());
                dst_index.resize(this->inLengths.size());

                // initialize the src index
                std::fill(dst_index.begin(), dst_index.end(), 0);

                for(int k                       = 0; k < invariantDims.size(); k++)
                    dst_index[invariantDims[k]] = index_1[k];

                int dst_offset = get_offset_from_index(this->outStrides, dst_index);

                // generate the part of src index belonging to invariant dims
                for(int k                       = 0; k < invariantDims.size(); k++)
                    src_index[invariantDims[k]] = index_1[k];

                compType accuVal = ReduceOpZeroVal<compType>(this->reduceOp);

                // go through indexes of the toReduce dimensions
                for(const auto& index_2 : indexes_2)
                {
                    // generate the part of src index belonging to toReduce dims
                    for(int k                      = 0; k < toReduceDims.size(); k++)
                        src_index[toReduceDims[k]] = index_2[k];

                    auto src_offset = get_offset_from_index(this->inStrides, src_index);

                    auto currVal = convert_type<compType>(in_data[src_offset]);

                    binop_with_nan_check(nanOpt, opReduce, accuVal, currVal);
                };

                // scale the accumulated value
                if(!float_equal_one(alpha))
                    accuVal *= convert_type<compType>(alpha);

                // scale the prior dst value and add it to the accumulated value
                if(!float_equal_zero(beta))
                    accuVal +=
                        convert_type<compType>(out_data[dst_offset] * convert_type<Tref>(beta));

                // store the reduced value to dst location
                out_data[dst_offset] = convert_type<Tref>(accuVal);
            };
        };
    }; // end of RunImpl_no_indices()
};

#endif
