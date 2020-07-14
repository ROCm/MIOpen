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
#include "driver.hpp"
#include "test.hpp"
#include "verify.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/reducetensor.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <limits>
#include <iostream>

#include "cpu_reduce_util.hpp"

static void get_all_indexes(const std::vector<std::size_t>& dimLengths,
                            int dim,
                            std::vector<std::vector<std::size_t>>& indexes)
{
    if(dim < dimLengths.size())
    {
        std::vector<std::vector<std::size_t>> updated_indexes;

        if(dim == 0)
        {
            assert(indexes.size() == 0);
            assert(dimLengths[dim] > 0);
            for(std::size_t i = 0; i < dimLengths[dim]; i++)
            {
                std::vector<std::size_t> index = {i};

                updated_indexes.push_back(index);
            };
        }
        else
        {
            // go through all the current indexes
            for(const auto& index : indexes)
                for(std::size_t i = 0; i < dimLengths[dim]; i++)
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

static std::size_t get_offset_from_index(const std::vector<std::size_t>& strides,
                                         const std::vector<std::size_t>& index)
{
    std::size_t offset = 0;

    assert(strides.size() == index.size());

    for(int i = 0; i < index.size(); i++)
        offset += strides[i] * index[i];

    return (offset);
};

static std::size_t get_flatten_offset(const std::vector<std::size_t>& lengths,
                                      const std::vector<std::size_t>& index)
{
    std::size_t offset = 0;

    assert(lengths.size() == index.size() && lengths.size() > 0);

    int len            = lengths.size();
    std::size_t stride = 1;

    // for len==1, the loop is not executed
    for(int i = len - 1; i > 0; i--)
    {
        offset += stride * index[i];

        stride *= lengths[i];
    };

    offset += stride * index[0];

    return (offset);
};

template <class T, bool toVerifyData>
struct verify_reduce_with_indices
{
    miopen::ReduceTensorDescriptor reduce;
    tensor<T> input;
    tensor<T> output;
    tensor<T> workspace;
    tensor<int> indices;
    T alpha;
    T beta;

    miopenReduceTensorOp_t reduceOp;
    miopenDataType_t compTypeVal;
    miopenNanPropagation_t nanOpt;
    miopenReduceTensorIndices_t indicesOpt;
    miopenIndicesType_t indicesType;

    verify_reduce_with_indices(const miopen::ReduceTensorDescriptor& reduce_,
                               const tensor<T>& input_,
                               const tensor<T>& output_,
                               const tensor<T>& workspace_,
                               const tensor<int>& indices_,
                               T alpha_,
                               T beta_)
    {
        reduce    = reduce_;
        input     = input_;
        output    = output_;
        workspace = workspace_;
        indices   = indices_;
        alpha     = alpha_;
        beta      = beta_;

        reduceOp    = reduce.reduceTensorOp_;
        compTypeVal = reduce.reduceTensorCompType_;
        nanOpt      = reduce.reduceTensorNanOpt_;
        indicesOpt  = reduce.reduceTensorIndices_;
        indicesType = reduce.reduceTensorIndicesType_;
    }

    tensor<float> cpu() const
    {
        using reduce::convert_type;

        std::tuple<tensor<T>, tensor<int>> results;

        if(compTypeVal == miopenFloat)
        {
            if(std::is_same<T, double>::value)
                results = cpuImpl<double>();
            else
                results = cpuImpl<float>();
        }
        else if(compTypeVal == miopenHalf)
        {
            if(std::is_same<T, double>::value)
                results = cpuImpl<double>();
            else if(std::is_same<T, float>::value)
                results = cpuImpl<float>();
            else
                results = cpuImpl<half_float::half>();
        };

        if(toVerifyData)
        {
            const auto dimLengths = output.desc.GetLengths();

            auto result_dataFloat = make_tensor<float>(dimLengths);

            auto& result_dataT = std::get<0>(results);

            for(size_t i                 = 0; i < result_dataT.data.size(); i++)
                result_dataFloat.data[i] = convert_type<float>(result_dataT.data[i]);

            return (result_dataFloat);
        }
        else
        {
            const auto dimLengths = indices.desc.GetLengths();

            auto result_indicesFloat = make_tensor<float>(dimLengths);

            auto& result_indices = std::get<1>(results);

            for(size_t i                    = 0; i < result_indices.data.size(); i++)
                result_indicesFloat.data[i] = static_cast<float>(result_indices.data[i]);

            return (result_indicesFloat);
        };
    };

    tensor<float> gpu() const
    {
        using reduce::convert_type;

        std::tuple<tensor<T>, tensor<int>> results;

        results = gpuImpl();

        if(toVerifyData)
        {
            const auto dimLengths = output.desc.GetLengths();

            auto result_dataFloat = make_tensor<float>(dimLengths);

            tensor<T>& result_dataT = std::get<0>(results);

            for(size_t i                 = 0; i < result_dataT.data.size(); i++)
                result_dataFloat.data[i] = convert_type<float>(result_dataT.data[i]);

            return (result_dataFloat);
        }
        else
        {
            const auto dimLengths = indices.desc.GetLengths();

            auto result_indicesFloat = make_tensor<float>(dimLengths);

            tensor<int>& result_indices = std::get<1>(results);

            for(size_t i                    = 0; i < result_indices.data.size(); i++)
                result_indicesFloat.data[i] = static_cast<float>(result_indices.data[i]);

            return (result_indicesFloat);
        };
    };

    template <typename compType>
    std::tuple<tensor<T>, tensor<int>> cpuImpl() const
    {
        using reduce::ReduceOpFn2;
        using reduce::ReduceOpZeroVal;
        using reduce::float_equal_one;
        using reduce::float_equal_zero;
        using reduce::convert_type;
        using reduce::binop_with_nan_check;
        using reduce::binop_with_nan_check2;

        auto inLengths  = input.desc.GetLengths();
        auto outLengths = output.desc.GetLengths();
        auto inStrides  = input.desc.GetStrides();
        auto outStrides = output.desc.GetStrides();

        // replicate
        auto res         = output;
        auto res_indices = indices;

        std::vector<std::size_t> invariantLengths;
        std::vector<std::size_t> toReduceLengths;

        std::vector<int> invariantDims;
        std::vector<int> toReduceDims;

        for(int i = 0; i < inLengths.size(); i++)
            if(inLengths[i] == outLengths[i])
                invariantDims.push_back(i);
            else
                toReduceDims.push_back(i);

        invariantLengths.resize(invariantDims.size());
        for(int i               = 0; i < invariantDims.size(); i++)
            invariantLengths[i] = inLengths[invariantDims[i]];

        toReduceLengths.resize(toReduceDims.size());
        for(int i              = 0; i < toReduceDims.size(); i++)
            toReduceLengths[i] = inLengths[toReduceDims[i]];

        bool reduceAllDims = invariantDims.empty();

        auto opReduce = ReduceOpFn2<compType>(reduceOp);

        if(reduceAllDims)
        {
            std::vector<std::vector<std::size_t>> indexes_1;

            get_all_indexes(inLengths, 0, indexes_1);

            compType accuVal = ReduceOpZeroVal<compType>(reduceOp);
            int accuIndex    = 0;

            // go through indexes of the invariant dimensions
            for(const auto& src_index : indexes_1)
            {
                auto src_offset = get_offset_from_index(inStrides, src_index);

                auto currVal = convert_type<compType>(input.data[src_offset]);

                int currIndex = get_flatten_offset(inLengths, src_index);
                binop_with_nan_check2(nanOpt, opReduce, accuVal, currVal, accuIndex, currIndex);
            };

            // scale the accumulated value
            if(!float_equal_one(alpha))
                accuVal *= convert_type<compType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!float_equal_zero(beta))
            {
                accuVal += convert_type<compType>(output.data[0] * beta);
            };

            // store the reduced value to dst location
            res.data[0]         = convert_type<T>(accuVal);
            res_indices.data[0] = accuIndex;
        }
        else
        {
            std::vector<std::vector<std::size_t>> indexes_1, indexes_2;

            get_all_indexes(invariantLengths, 0, indexes_1);
            get_all_indexes(toReduceLengths, 0, indexes_2);

            // go through indexes of the invariant dimensions
            for(const auto& index_1 : indexes_1)
            {
                std::vector<std::size_t> src_index;
                std::vector<std::size_t> dst_index;

                src_index.resize(inLengths.size());
                dst_index.resize(inLengths.size());

                std::fill(dst_index.begin(), dst_index.end(), 0);

                for(int k                       = 0; k < invariantDims.size(); k++)
                    dst_index[invariantDims[k]] = index_1[k];

                int dst_offset = get_offset_from_index(outStrides, dst_index);

                // generate the part of the index belonging to the invariant dims
                for(int k                       = 0; k < invariantDims.size(); k++)
                    src_index[invariantDims[k]] = index_1[k];

                compType accuVal = ReduceOpZeroVal<compType>(reduceOp);
                int accuIndex    = 0;

                // go through indexes of the toReduce dimensions
                for(const auto& index_2 : indexes_2)
                {
                    // generate the part of the index belonging to the toReduce dims
                    for(int k                      = 0; k < toReduceDims.size(); k++)
                        src_index[toReduceDims[k]] = index_2[k];

                    auto src_offset = get_offset_from_index(inStrides, src_index);

                    auto currVal = convert_type<compType>(input.data[src_offset]);

                    auto currIndex = get_flatten_offset(toReduceLengths, index_2);
                    binop_with_nan_check2(nanOpt, opReduce, accuVal, currVal, accuIndex, currIndex);
                };

                // scale the accumulated value
                if(!float_equal_one(alpha))
                    accuVal *= convert_type<compType>(alpha);

                // scale the prior dst value and add it to the accumulated value
                if(!float_equal_zero(beta))
                    accuVal += convert_type<compType>(output.data[dst_offset] * beta);

                // store the reduced value to dst location
                res.data[dst_offset]         = convert_type<T>(accuVal);
                res_indices.data[dst_offset] = accuIndex; // store the index
            };
        };

        return (std::make_tuple(res, res_indices));
    }

    std::tuple<tensor<T>, tensor<int>> gpuImpl() const
    {
        auto&& handle   = get_handle();
        auto input_dev  = handle.Write(input.data);
        auto output_dev = handle.Write(output.data);

        // replicate
        auto res         = output;
        auto res_indices = indices;

        auto indices_dev = handle.Write(indices.data);

        std::size_t ws_sizeInBytes      = workspace.desc.GetElementSize() * sizeof(T);
        std::size_t indices_sizeInBytes = indices.desc.GetElementSize() * sizeof(int);

        if(ws_sizeInBytes > 0)
        {
            auto workspace_dev = handle.Write(workspace.data);

            reduce.ReduceTensor(get_handle(),
                                indices_dev.get(),
                                indices_sizeInBytes,
                                workspace_dev.get(),
                                ws_sizeInBytes,
                                static_cast<const void*>(&alpha),
                                input.desc,
                                input_dev.get(),
                                static_cast<const void*>(&beta),
                                output.desc,
                                output_dev.get());
        }
        else
        {
            reduce.ReduceTensor(get_handle(),
                                indices_dev.get(),
                                indices_sizeInBytes,
                                nullptr,
                                0,
                                static_cast<const void*>(&alpha),
                                input.desc,
                                input_dev.get(),
                                static_cast<const void*>(&beta),
                                output.desc,
                                output_dev.get());
        };

        res.data         = handle.Read<T>(output_dev, res.data.size());
        res_indices.data = handle.Read<int>(indices_dev, res_indices.data.size());

        return (std::make_tuple(res, res_indices));
    }

    void fail(int) const
    {
        std::cout << "verify_reduce_with_indices failed" << std::endl;
        std::cout << "Input Tensor"
                  << " " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct verify_reduce_no_indices
{
    miopen::ReduceTensorDescriptor reduce;
    tensor<T> input;
    tensor<T> output;
    tensor<T> workspace;
    T alpha;
    T beta;

    miopenReduceTensorOp_t reduceOp;
    miopenDataType_t compTypeVal;
    miopenNanPropagation_t nanOpt;

    verify_reduce_no_indices(const miopen::ReduceTensorDescriptor& reduce_,
                             const tensor<T>& input_,
                             const tensor<T>& output_,
                             const tensor<T>& workspace_,
                             T alpha_,
                             T beta_)
    {
        reduce    = reduce_;
        input     = input_;
        output    = output_;
        workspace = workspace_;
        alpha     = alpha_;
        beta      = beta_;

        reduceOp    = reduce.reduceTensorOp_;
        compTypeVal = reduce.reduceTensorCompType_;
        nanOpt      = reduce.reduceTensorNanOpt_;
    }

    tensor<T> cpu()
    {
        if(compTypeVal == miopenFloat)
        {
            if(std::is_same<T, double>::value)
                return (cpuImpl<double>());
            else
                return (cpuImpl<float>());
        }
        else if(compTypeVal == miopenHalf)
        {
            if(std::is_same<T, double>::value)
                return (cpuImpl<double>());
            else if(std::is_same<T, float>::value)
                return (cpuImpl<float>());
            else
                return (cpuImpl<half_float::half>());
        };

        return (tensor<T>{});
    };

    template <typename compType>
    tensor<T> cpuImpl() const
    {
        using reduce::ReduceOpFn;
        using reduce::ReduceOpZeroVal;
        using reduce::float_equal_one;
        using reduce::float_equal_zero;
        using reduce::convert_type;
        using reduce::binop_with_nan_check;
        using reduce::binop_with_nan_check2;

        auto inLengths  = input.desc.GetLengths();
        auto outLengths = output.desc.GetLengths();
        auto inStrides  = input.desc.GetStrides();
        auto outStrides = output.desc.GetStrides();

        // replicate
        auto res = output;

        std::vector<std::size_t> invariantLengths;
        std::vector<std::size_t> toReduceLengths;

        std::vector<int> invariantDims;
        std::vector<int> toReduceDims;

        for(int i = 0; i < inLengths.size(); i++)
            if(inLengths[i] == outLengths[i])
                invariantDims.push_back(i);
            else
                toReduceDims.push_back(i);

        invariantLengths.resize(invariantDims.size());
        for(int i               = 0; i < invariantDims.size(); i++)
            invariantLengths[i] = inLengths[invariantDims[i]];

        toReduceLengths.resize(toReduceDims.size());
        for(int i              = 0; i < toReduceDims.size(); i++)
            toReduceLengths[i] = inLengths[toReduceDims[i]];

        bool reduceAllDims = invariantDims.empty();

        auto opReduce = ReduceOpFn<compType>(reduceOp);

        if(reduceAllDims)
        {
            std::vector<std::vector<std::size_t>> indexes_1;

            get_all_indexes(inLengths, 0, indexes_1);

            compType accuVal = ReduceOpZeroVal<compType>(reduceOp);

            // go through indexes of the invariant dimensions
            for(const auto& src_index : indexes_1)
            {
                auto src_offset = get_offset_from_index(inStrides, src_index);

                auto currVal = convert_type<compType>(input.data[src_offset]);

                binop_with_nan_check(nanOpt, opReduce, accuVal, currVal);
            };

            // scale the accumulated value
            if(!float_equal_one(alpha))
                accuVal *= convert_type<compType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!float_equal_one(beta))
                accuVal += convert_type<compType>(output.data[0] * beta);

            // store the reduced value to dst location
            res.data[0] = convert_type<T>(accuVal);
        }
        else
        {
            std::vector<std::vector<std::size_t>> indexes_1, indexes_2;

            get_all_indexes(invariantLengths, 0, indexes_1);
            get_all_indexes(toReduceLengths, 0, indexes_2);

            // go through indexes of the invariant dimensions
            for(const auto& index_1 : indexes_1)
            {
                std::vector<std::size_t> src_index;
                std::vector<std::size_t> dst_index;

                src_index.resize(inLengths.size());
                dst_index.resize(inLengths.size());

                std::fill(dst_index.begin(), dst_index.end(), 0);

                for(int k                       = 0; k < invariantDims.size(); k++)
                    dst_index[invariantDims[k]] = index_1[k];

                int dst_offset = get_offset_from_index(outStrides, dst_index);

                // generate the part of the index belonging to the invariant dims
                for(int k                       = 0; k < invariantDims.size(); k++)
                    src_index[invariantDims[k]] = index_1[k];

                compType accuVal = ReduceOpZeroVal<compType>(reduceOp);

                // go through indexes of the toReduce dimensions
                for(const auto& index_2 : indexes_2)
                {
                    // generate the part of the index belonging to the toReduce dims
                    for(int k                      = 0; k < toReduceDims.size(); k++)
                        src_index[toReduceDims[k]] = index_2[k];

                    auto src_offset = get_offset_from_index(inStrides, src_index);

                    auto currVal = convert_type<compType>(input.data[src_offset]);

                    binop_with_nan_check(nanOpt, opReduce, accuVal, currVal);
                };

                // scale the accumulated value
                if(!float_equal_one(alpha))
                    accuVal *= convert_type<compType>(alpha);

                // scale the prior dst value and add it to the accumulated value
                if(!float_equal_zero(beta))
                    accuVal += convert_type<compType>(output.data[dst_offset] * beta);

                // store the reduced value to dst location
                res.data[dst_offset] = convert_type<T>(accuVal);
            };
        };

        return (res);
    }

    tensor<T> gpu() const
    {
        auto&& handle   = get_handle();
        auto input_dev  = handle.Write(input.data);
        auto output_dev = handle.Write(output.data);

        // replicate
        auto res = output;

        std::size_t ws_sizeInBytes = workspace.desc.GetElementSize() * sizeof(T);

        if(ws_sizeInBytes > 0)
        {
            auto workspace_dev = handle.Write(workspace.data);

            reduce.ReduceTensor(get_handle(),
                                nullptr,
                                0,
                                workspace_dev.get(),
                                ws_sizeInBytes,
                                static_cast<const void*>(&alpha),
                                input.desc,
                                input_dev.get(),
                                static_cast<const void*>(&beta),
                                output.desc,
                                output_dev.get());
        }
        else
        {
            reduce.ReduceTensor(get_handle(),
                                nullptr,
                                0,
                                nullptr,
                                0,
                                static_cast<const void*>(&alpha),
                                input.desc,
                                input_dev.get(),
                                static_cast<const void*>(&beta),
                                output.desc,
                                output_dev.get());
        };

        res.data = handle.Read<T>(output_dev, res.data.size());

        return (res);
    }

    void fail(int) const
    {
        std::cout << "verify_reduce_no_indices failed" << std::endl;
        std::cout << "Input Tensor"
                  << " " << input.desc.ToString() << std::endl;
    }
};

template <class T>
struct reduce_driver : test_driver
{
    int reduceOp                    = 0; //  miopenReduceTensorOp_t reduceOp;
    int compTypeVal                 = 1; //  miopenDataType_t compTypeVal;
    int nanOpt                      = 0; //  miopenNanPropagation_t nanOpt;
    int indicesOpt                  = 0; //  miopenReduceTensorIndices_t indicesOpt;
    miopenIndicesType_t indicesType = MIOPEN_32BIT_INDICES;

    std::vector<std::size_t> inLengths; // the lengths of the input tensor's dimensions
    std::vector<int>
        toReduceDims; // the indexes of the dimensions to be reduced in the input tensor

    std::vector<float> scales;
    float alpha = 1.0f;
    float beta  = 0.0f;

    std::vector<std::vector<std::size_t>> get_tensor_lengths()
    {
        if(std::is_same<T, half_float::half>::value)
            return {
                {4, 3, 60, 50},
            };
        else
            return {
                {64, 3, 280, 81},
            };
    }

    std::vector<std::vector<int>> get_toreduce_dims()
    {
        return {
            {0},
            {1},
            {2},
            {3},
            {0, 1},
            {1, 2},
            {0, 3},
            {1, 3},
            {0, 2},
            {2, 3},
            {0, 1, 3},
            {1, 2, 3},
            {0, 1, 2, 3},
        };
    }

    reduce_driver()
    {
        add(inLengths, "D", generate_data(get_tensor_lengths()));
        add(toReduceDims, "R", generate_data(get_toreduce_dims()));
        add(reduceOp, "ReduceOp", generate_data({0, 2}));
        add(compTypeVal, "CompType", generate_data({1}));
        add(nanOpt, "N", generate_data({0}));
        add(indicesOpt, "I", generate_data({0, 1}));

        add(scales, "scales", generate_data({{1.0f, 0.0f}, {0.5f, 0.5f}}));

        auto&& handle = get_handle();
        handle.EnableProfiling();
    }

    void run()
    {
        using reduce::convert_type;

        if(std::is_same<T, half_float::half>::value)
        {
            if(reduceOp == static_cast<int>(MIOPEN_REDUCE_TENSOR_MIN) ||
               reduceOp == static_cast<int>(MIOPEN_REDUCE_TENSOR_MAX))
                compTypeVal = static_cast<int>(miopenHalf); // let compType be same as the data type
            else
                compTypeVal = static_cast<int>(miopenFloat);
        }

        miopen::ReduceTensorDescriptor reduceDesc(
            static_cast<miopenReduceTensorOp_t>(reduceOp),
            static_cast<miopenDataType_t>(compTypeVal),
            static_cast<miopenNanPropagation_t>(nanOpt),
            static_cast<miopenReduceTensorIndices_t>(indicesOpt),
            indicesType);

        alpha = scales[0];
        beta  = scales[1];

        auto outLengths = this->inLengths;

        assert(toReduceDims.size() <= outLengths.size());
        for(int i = 0; i < toReduceDims.size(); i++)
            assert(toReduceDims[i] < inLengths.size());

        // set the lengths of the dimensions to be reduced to 1 to represent the output Tensor
        for(int i                       = 0; i < toReduceDims.size(); i++)
            outLengths[toReduceDims[i]] = static_cast<std::size_t>(1);

        unsigned long max_value =
            miopen_type<T>{} == miopenHalf ? 13 : miopen_type<T>{} == miopenInt8 ? 127 : 17;

        auto gen_value = [&](auto... is) {
            return (tensor_elem_gen_integer{max_value}(is...) *
                    tensor_elem_gen_checkboard_sign{}(is...));
        };

        auto inputTensor  = tensor<T>{this->inLengths}.generate(gen_value);
        auto outputTensor = tensor<T>{outLengths};

        std::fill(outputTensor.begin(), outputTensor.end(), convert_type<T>(0.0f));

        auto indices_nelem =
            reduceDesc.GetIndicesSize(inputTensor.desc, outputTensor.desc) / sizeof(int);

        auto ws_sizeInBytes =
            reduceDesc.GetWorkspaceSize(get_handle(), inputTensor.desc, outputTensor.desc);
        auto workspace_nelem = (indices_nelem == 0) ? ws_sizeInBytes / sizeof(T)
                                                    : (ws_sizeInBytes + sizeof(T) - 1) / sizeof(T);

        std::vector<std::size_t> wsLengths = {static_cast<std::size_t>(workspace_nelem), 1};
        auto workspaceTensor               = tensor<T>{wsLengths};

        std::fill(workspaceTensor.begin(), workspaceTensor.end(), convert_type<T>(0.0f));

        if(indices_nelem > 0)
        {
            std::vector<std::size_t> indicesLengths = {static_cast<std::size_t>(indices_nelem), 1};
            auto indicesTensor                      = tensor<int>{indicesLengths};

            std::fill(indicesTensor.begin(), indicesTensor.end(), 1);

            verify(verify_reduce_with_indices<T, true>(reduceDesc,
                                                       inputTensor,
                                                       outputTensor,
                                                       workspaceTensor,
                                                       indicesTensor,
                                                       convert_type<T>(1.0),
                                                       convert_type<T>(0.0)));

            verify_equals(verify_reduce_with_indices<T, false>(reduceDesc,
                                                               inputTensor,
                                                               outputTensor,
                                                               workspaceTensor,
                                                               indicesTensor,
                                                               convert_type<T>(1.0),
                                                               convert_type<T>(0.0)));
        }
        else
        {
            verify(verify_reduce_no_indices<T>(reduceDesc,
                                               inputTensor,
                                               outputTensor,
                                               workspaceTensor,
                                               convert_type<T>(alpha),
                                               convert_type<T>(beta)));
        };
    };
};

int main(int argc, const char* argv[])
{
    std::vector<std::string> as(argv + 1, argv + argc);

    bool test_half = false;

    test_half = std::any_of(
        as.begin(), as.end(), [](const std::string& elem) { return (elem == "--half"); });

    if(test_half)
        test_drive<reduce_driver<half_float::half>>(argc, argv);
    else
        test_drive<reduce_driver<float>>(argc, argv);
};
