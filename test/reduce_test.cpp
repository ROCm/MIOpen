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
#include <miopen/config.h>
#include "driver.hpp"
#include "test.hpp"
#include "verify.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "random.hpp"
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/reducetensor.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <limits>
#include <iostream>
#include <type_traits>

#include "cpu_reduce_util.hpp"

/// Not reproducible with ROCm 4.1 and 4.2.
#define WORKAROUND_GPU_NUMERIC_ERROR \
    (HIP_PACKAGE_VERSION_MAJOR == 3 && HIP_PACKAGE_VERSION_MINOR == 7)

template <class T, bool toVerifyData>
struct verify_reduce_with_indices
{
    miopen::ReduceTensorDescriptor reduce;
    tensor<T> input;
    tensor<T> output;
    tensor<T> workspace;
    tensor<int> indices;
    float alpha;
    float beta;

    miopenReduceTensorOp_t reduceOp;
    miopenDataType_t compTypeVal;
    miopenNanPropagation_t nanOpt;
    miopenReduceTensorIndices_t indicesOpt;
    miopenIndicesType_t indicesType;

    verify_reduce_with_indices( // NOLINT (hicpp-member-init)
        const miopen::ReduceTensorDescriptor& reduce_,
        const tensor<T>& input_,
        const tensor<T>& output_,
        const tensor<T>& workspace_,
        const tensor<int>& indices_,
        float alpha_,
        float beta_)
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
        }
        else if(compTypeVal == miopenDouble)
            results = cpuImpl<double>();

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
        using reduce::PreUnaryOpFn;
        using reduce::PosUnaryOpFn;
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

        std::size_t divider = std::accumulate(
            toReduceLengths.begin(), toReduceLengths.end(), std::size_t{1}, std::multiplies<>{});

        auto PreUnaryOp = PreUnaryOpFn<compType>(reduceOp, divider);

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

                // unary operation before reducing, only needed by AMAX. For MIN/MAX, nothing is
                // actually done
                PreUnaryOp(currVal);

                int currIndex = get_flatten_offset(inLengths, src_index);
                binop_with_nan_check2(nanOpt, opReduce, accuVal, currVal, accuIndex, currIndex);
            };

            // scale the accumulated value
            if(!float_equal_one(alpha))
                accuVal *= convert_type<compType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!float_equal_zero(beta))
            {
                accuVal += convert_type<compType>(output.data[0]) * convert_type<compType>(beta);
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

                auto dst_offset = get_offset_from_index(outStrides, dst_index);

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

                    // unary operation before reducing, only needed by AMAX. For MIN/MAX, nothing is
                    // actually done
                    PreUnaryOp(currVal);

                    auto currIndex = get_flatten_offset(toReduceLengths, index_2);
                    binop_with_nan_check2(nanOpt, opReduce, accuVal, currVal, accuIndex, currIndex);
                };

                // scale the accumulated value
                if(!float_equal_one(alpha))
                    accuVal *= convert_type<compType>(alpha);

                // scale the prior dst value and add it to the accumulated value
                if(!float_equal_zero(beta))
                    accuVal += convert_type<compType>(output.data[dst_offset]) *
                               convert_type<compType>(beta);

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

        const double alpha64 = alpha;
        const double beta64  = beta;

        const void* const alphaPtr = (std::is_same<T, double>::value)
                                         ? static_cast<const void*>(&alpha64)
                                         : static_cast<const void*>(&alpha);
        const void* const betaPtr = (std::is_same<T, double>::value)
                                        ? static_cast<const void*>(&beta64)
                                        : static_cast<const void*>(&beta);

        if(ws_sizeInBytes > 0)
        {
            auto workspace_dev = handle.Write(workspace.data);

            reduce.ReduceTensor(get_handle(),
                                indices_dev.get(),
                                indices_sizeInBytes,
                                workspace_dev.get(),
                                ws_sizeInBytes,
                                alphaPtr,
                                input.desc,
                                input_dev.get(),
                                betaPtr,
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
                                alphaPtr,
                                input.desc,
                                input_dev.get(),
                                betaPtr,
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
    float alpha;
    float beta;

    miopenReduceTensorOp_t reduceOp;
    miopenDataType_t compTypeVal;
    miopenNanPropagation_t nanOpt;

    verify_reduce_no_indices( // NOLINT (hicpp-member-init)
        const miopen::ReduceTensorDescriptor& reduce_,
        const tensor<T>& input_,
        const tensor<T>& output_,
        const tensor<T>& workspace_,
        float alpha_,
        float beta_)
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

    tensor<float> cpu()
    {
        using reduce::convert_type;

        tensor<T> result;

        if(compTypeVal == miopenFloat)
        {
            if(std::is_same<T, double>::value)
                result = cpuImpl<double>();
            else
                result = cpuImpl<float>();
        }
        else if(compTypeVal == miopenHalf)
        {
            if(std::is_same<T, double>::value)
                result = cpuImpl<double>();
            else if(std::is_same<T, float>::value)
                result = cpuImpl<float>();
            else
                result = cpuImpl<half_float::half>();
        }
        else if(compTypeVal == miopenDouble)
            result = cpuImpl<double>();

        const auto dimLengths = output.desc.GetLengths();
        auto result_dataFloat = make_tensor<float>(dimLengths);

        for(size_t i                 = 0; i < result.data.size(); i++)
            result_dataFloat.data[i] = convert_type<float>(result.data[i]);

        return (result_dataFloat);
    };

    template <typename compType>
    tensor<T> cpuImpl() const
    {
        using reduce::ReduceOpFn;
        using reduce::PreUnaryOpFn;
        using reduce::PosUnaryOpFn;
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

        std::size_t divider = std::accumulate(
            toReduceLengths.begin(), toReduceLengths.end(), std::size_t{1}, std::multiplies<>{});

        auto PreUnaryOp = PreUnaryOpFn<compType>(reduceOp, divider);
        auto PosUnaryOp = PosUnaryOpFn<compType>(reduceOp, divider);

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

                PreUnaryOp(currVal);

                binop_with_nan_check(nanOpt, opReduce, accuVal, currVal);
            };

            PosUnaryOp(accuVal);

            // scale the accumulated value
            if(!float_equal_one(alpha))
                accuVal *= convert_type<compType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!float_equal_zero(beta))
                accuVal += convert_type<compType>(output.data[0]) * convert_type<compType>(beta);

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

                auto dst_offset = get_offset_from_index(outStrides, dst_index);

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

                    PreUnaryOp(currVal);

                    binop_with_nan_check(nanOpt, opReduce, accuVal, currVal);
                };

                PosUnaryOp(accuVal);

                // scale the accumulated value
                if(!float_equal_one(alpha))
                    accuVal *= convert_type<compType>(alpha);

                // scale the prior dst value and add it to the accumulated value
                if(!float_equal_zero(beta))
                    accuVal += convert_type<compType>(output.data[dst_offset]) *
                               convert_type<compType>(beta);

                // store the reduced value to dst location
                res.data[dst_offset] = convert_type<T>(accuVal);
            };
        };

        return (res);
    }

    tensor<float> gpu() const
    {
        using reduce::convert_type;

        auto result = gpuImpl();

        const auto dimLengths = output.desc.GetLengths();
        auto result_dataFloat = make_tensor<float>(dimLengths);

        for(size_t i                 = 0; i < result.data.size(); i++)
            result_dataFloat.data[i] = convert_type<float>(result.data[i]);

        return (result_dataFloat);
    };

    tensor<T> gpuImpl() const
    {
        auto&& handle   = get_handle();
        auto input_dev  = handle.Write(input.data);
        auto output_dev = handle.Write(output.data);

        // replicate
        auto res = output;

        std::size_t ws_sizeInBytes = workspace.desc.GetElementSize() * sizeof(T);

        const double alpha64 = alpha;
        const double beta64  = beta;

        const void* const alphaPtr = (std::is_same<T, double>::value)
                                         ? static_cast<const void*>(&alpha64)
                                         : static_cast<const void*>(&alpha);
        const void* const betaPtr = (std::is_same<T, double>::value)
                                        ? static_cast<const void*>(&beta64)
                                        : static_cast<const void*>(&beta);

        if(ws_sizeInBytes > 0)
        {
            auto workspace_dev = handle.Write(workspace.data);

            reduce.ReduceTensor(get_handle(),
                                nullptr,
                                0,
                                workspace_dev.get(),
                                ws_sizeInBytes,
                                alphaPtr,
                                input.desc,
                                input_dev.get(),
                                betaPtr,
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
                                alphaPtr,
                                input.desc,
                                input_dev.get(),
                                betaPtr,
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
        std::vector<std::vector<int>> tensor_dims = {
            {0}, {1}, {2}, {3}, {0, 1}, {0, 3}, {0, 2}, {2, 3}, {0, 1, 3}, {1, 2, 3}, {0, 1, 2, 3}};

        return tensor_dims;
    }

    reduce_driver()
    {
        add(inLengths, "D", generate_data(get_tensor_lengths()));
        add(toReduceDims, "R", generate_data(get_toreduce_dims()));
        add(reduceOp, "ReduceOp", generate_data({0, 1, 4, 5, 6, 7}));
        add(compTypeVal, "CompType", generate_data({1}));
        add(nanOpt, "N", generate_data({0, 1}));
        add(indicesOpt, "I", generate_data({0, 1}));

        add(scales, "scales", generate_data({{1.0f, 0.0f}, {0.5f, 0.5f}}));

        auto&& handle = get_handle();
        handle.EnableProfiling();
    }

    void run()
    {
        using reduce::convert_type;

        if(std::is_same<T, double>::value)
            compTypeVal = static_cast<int>(miopenDouble);

        if(std::is_same<T, half_float::half>::value)
        {
            if(reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX ||
               reduceOp == MIOPEN_REDUCE_TENSOR_AMAX)
                compTypeVal = static_cast<int>(miopenHalf); // let compType be same as the data type
            else
                compTypeVal = static_cast<int>(miopenFloat);
        }

#if WORKAROUND_GPU_NUMERIC_ERROR
        if(std::is_same<T, double>::value)
        {
            if(inLengths == std::vector<std::size_t>{64, 3, 280, 81} &&
               toReduceDims == std::vector<int>{0, 1, 2, 3} && (reduceOp == 3 || reduceOp == 4) &&
               indicesOpt == 1)
            {
                std::cout << "Workaround: Skipping the test." << std::endl;
                return;
            };
        }
#endif

        miopen::ReduceTensorDescriptor reduceDesc(
            static_cast<miopenReduceTensorOp_t>(reduceOp),
            static_cast<miopenDataType_t>(compTypeVal),
            static_cast<miopenNanPropagation_t>(nanOpt),
            static_cast<miopenReduceTensorIndices_t>(indicesOpt),
            indicesType);

        alpha = scales[0];
        beta  = scales[1];

        // The test is ignored if (alpha, beta) is not (1.0f, 0.0f) and reduceOp is not Add/MUL/AVG
        if(reduceOp != MIOPEN_REDUCE_TENSOR_ADD && reduceOp != MIOPEN_REDUCE_TENSOR_MUL &&
           reduceOp != MIOPEN_REDUCE_TENSOR_AVG && alpha != 1.0f && beta != 0.0f)
            return;

        // The test is ignored if indices are requested but the reduceOp is neither MIN nor MAX
        if(indicesOpt != MIOPEN_REDUCE_TENSOR_NO_INDICES && reduceOp != MIOPEN_REDUCE_TENSOR_MIN &&
           reduceOp != MIOPEN_REDUCE_TENSOR_MAX && reduceOp != MIOPEN_REDUCE_TENSOR_AMAX)
            return;

        auto outLengths = this->inLengths;

        assert(toReduceDims.size() <= outLengths.size());
        for(int i = 0; i < toReduceDims.size(); i++)
            assert(toReduceDims[i] < inLengths.size());

        // set the lengths of the dimensions to be reduced to 1 to represent the output Tensor
        for(const int& toReduceDim : toReduceDims)
            outLengths[toReduceDim] = static_cast<std::size_t>(1);

        unsigned long max_value;

        if(reduceOp == MIOPEN_REDUCE_TENSOR_MUL)
            max_value =
                miopen_type<T>{} == miopenHalf ? 41 : miopen_type<T>{} == miopenInt8 ? 127 : 111;
        else if(reduceOp == MIOPEN_REDUCE_TENSOR_NORM1 || reduceOp == MIOPEN_REDUCE_TENSOR_NORM2)
            max_value = 3;
        else
            max_value =
                miopen_type<T>{} == miopenHalf ? 13 : miopen_type<T>{} == miopenInt8 ? 127 : 17;

        // default data gneration (used by MIN/MAX)
        auto gen_value = [&](auto... is) {
            return (tensor_elem_gen_integer{max_value}(is...) *
                    tensor_elem_gen_checkboard_sign{}(is...));
        };

        // data generation used by ADD/AVG, data is distributed around 1.0 rather than 0.0, very low
        // probability to get a reduced result of zero-value
        auto gen_value_1 = [&](auto... is) {
            auto rand_value = tensor_elem_gen_integer{max_value}(is...);
            auto sign_value = tensor_elem_gen_checkboard_sign{}(is...);

            return (sign_value * rand_value + 1.0);
        };

        // Special data generation for MUL, to avoid all-zero and large accumulative error in the
        // reduced result
        auto gen_value_2 = [&](auto... is) {
            auto rand_value = tensor_elem_gen_integer{max_value}(is...);
            auto sign_value = tensor_elem_gen_checkboard_sign{}(is...);

            return sign_value > 0.0 ? (rand_value + max_value) / (rand_value + max_value + 1)
                                    : (rand_value + max_value + 1) / (rand_value + max_value);
        };

        // Special data generation for NORM1 and NORM2 using a space of limitless number of values.
        // This method is slower due to the use of GET_RAND(), it is usually used for manual testing
        auto gen_value_3 = [&](auto... is) {
            auto rand_upper   = tensor_elem_gen_integer{max_value}(is...);
            auto sign_value   = tensor_elem_gen_checkboard_sign{}(is...);
            double rand_ratio = static_cast<double>(GET_RAND() / (static_cast<double>(RAND_MAX)));

            return rand_upper * sign_value * rand_ratio;
        };

        // Special data generation for AMAX, no zero value used
        auto gen_value_4 = [&](auto... is) {
            auto rand_value = tensor_elem_gen_integer{max_value}(is...);
            auto sign_value = tensor_elem_gen_checkboard_sign{}(is...);

            return sign_value > 0.0 ? (rand_value + 0.5) : (-1.0 * rand_value - 0.5);
        };

        // default tolerance (refer to driver.hpp)
        this->tolerance = 80;

        if(reduceOp == MIOPEN_REDUCE_TENSOR_MUL)
            this->tolerance = 80 * 300;
        else if(reduceOp == MIOPEN_REDUCE_TENSOR_NORM1 || reduceOp == MIOPEN_REDUCE_TENSOR_NORM2)
        {
            if(toReduceDims.size() == 4)
                this->tolerance = 80 * 100;
            else
                this->tolerance = 80 * 10;
        };

        if(std::is_same<T, half_float::half>::value)
            this->tolerance *= this->tolerance * 10.0;

        tensor<T> inputTensor;

        switch(reduceOp)
        {
        case MIOPEN_REDUCE_TENSOR_ADD:
        case MIOPEN_REDUCE_TENSOR_AVG:
            inputTensor = tensor<T>{this->inLengths}.generate(gen_value_1);
            break;
        case MIOPEN_REDUCE_TENSOR_MUL:
            inputTensor = tensor<T>{this->inLengths}.generate(gen_value_2);
            break;
        case MIOPEN_REDUCE_TENSOR_NORM1:
        case MIOPEN_REDUCE_TENSOR_NORM2:
            inputTensor = tensor<T>{this->inLengths}.generate(gen_value_3);
            break;
        case MIOPEN_REDUCE_TENSOR_AMAX:
            inputTensor = tensor<T>{this->inLengths}.generate(gen_value_4);
            break;
        default: inputTensor = tensor<T>{this->inLengths}.generate(gen_value);
        };

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

            verify(verify_reduce_with_indices<T, true>(
                reduceDesc, inputTensor, outputTensor, workspaceTensor, indicesTensor, 1.0f, 0.0f));

            verify_equals(verify_reduce_with_indices<T, false>(
                reduceDesc, inputTensor, outputTensor, workspaceTensor, indicesTensor, 1.0f, 0.0f));
        }
        else
        {
            verify(verify_reduce_no_indices<T>(
                reduceDesc, inputTensor, outputTensor, workspaceTensor, alpha, beta));
        };
    };
};

int main(int argc, const char* argv[])
{
    std::vector<std::string> as(argv + 1, argv + argc);

    bool test_half   = false;
    bool test_double = false;

    test_half = std::any_of(
        as.begin(), as.end(), [](const std::string& elem) { return (elem == "--half"); });

    test_double = std::any_of(
        as.begin(), as.end(), [](const std::string& elem) { return (elem == "--double"); });

    if(test_half)
        test_drive<reduce_driver<half_float::half>>(argc, argv);
    else if(test_double)
        test_drive<reduce_driver<double>>(argc, argv);
    else
        test_drive<reduce_driver<float>>(argc, argv);
};
