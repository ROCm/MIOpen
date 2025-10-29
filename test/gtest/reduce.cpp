/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include "gtest_common.hpp"
#include <tensor_util.hpp>
#include <miopen/reducetensor.hpp>
#include "cpu_reduce_util.hpp"
#include "workspace.hpp"

// Recently added FP16, BFP16 and I8 reduce custom test cases fail compiler staging tests
#define WORKAROUND_ISSUE_895 1

#if WORKAROUND_ISSUE_895
#define CAN_BE_DISABLED_HalfTest_reduce_custom_fp16 DISABLED_HalfTest_reduce_custom_fp16
#define CAN_BE_DISABLED_BHalfTest_reduce_custom_fp16 DISABLED_BHalfTest_reduce_custom_bfp16
#define CAN_BE_DISABLED_IntTest_reduce_custom_fp16 DISABLED_IntTest_reduce_custom_i8
#else
#define CAN_BE_DISABLED_HalfTest_reduce_custom_fp16 HalfTest_reduce_custom_fp16
#define CAN_BE_DISABLED_BHalfTest_reduce_custom_fp16 BHalfTest_reduce_custom_bfp16
#define CAN_BE_DISABLED_IntTest_reduce_custom_fp16 IntTest_reduce_custom_i8
#endif

namespace {

using TestCase = std::tuple<std::vector<std::size_t>,
                            std::vector<int>,
                            miopenReduceTensorOp_t,
                            miopenNanPropagation_t,
                            miopenReduceTensorIndices_t,
                            std::vector<float>>;

template <typename T, class TVerifier>
auto GetCpuImplResult(miopenDataType_t compTypeVal, const TVerifier& verifier)
{
    if(compTypeVal == miopenFloat)
    {
        if(std::is_same<T, double>::value)
            return verifier.template cpuImpl<double>();
        else
            return verifier.template cpuImpl<float>();
    }
    else if(compTypeVal == miopenHalf)
    {
        if(std::is_same<T, double>::value)
            return verifier.template cpuImpl<double>();
        else if(std::is_same<T, float>::value)
            return verifier.template cpuImpl<float>();
        else
            return verifier.template cpuImpl<half_float::half>();
    }
    else if(compTypeVal == miopenDouble)
        return verifier.template cpuImpl<double>();

    throw std::runtime_error("Type is not supported!");
}

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

    verify_reduce_with_indices(const miopen::ReduceTensorDescriptor& reduce_,
                               const tensor<T>& input_,
                               const tensor<T>& output_,
                               const tensor<T>& workspace_,
                               const tensor<int>& indices_,
                               float alpha_,
                               float beta_)
        : reduce(reduce_),
          input(input_),
          output(output_),
          workspace(workspace_),
          indices(indices_),
          alpha(alpha_),
          beta(beta_),
          reduceOp(reduce.reduceTensorOp_),
          compTypeVal(reduce.reduceTensorCompType_),
          nanOpt(reduce.reduceTensorNanOpt_),
          indicesOpt(reduce.reduceTensorIndices_),
          indicesType(reduce.reduceTensorIndicesType_)
    {
    }

    tensor<float> cpu() const
    {
        using reduce::convert_type;

        const std::tuple<tensor<T>, tensor<int>> results =
            GetCpuImplResult<T, verify_reduce_with_indices>(compTypeVal, *this);

        if(toVerifyData)
        {
            const auto& dimLengths = output.desc.GetLengths();

            auto result_dataFloat = tensor<float>(dimLengths);

            auto& result_dataT = std::get<0>(results);

            for(size_t i = 0; i < result_dataT.data.size(); i++)
                result_dataFloat.data[i] = convert_type<float>(result_dataT.data[i]);

            return (result_dataFloat);
        }
        else
        {
            const auto& dimLengths = indices.desc.GetLengths();

            auto result_indicesFloat = tensor<float>(dimLengths);

            auto& result_indices = std::get<1>(results);

            for(size_t i = 0; i < result_indices.data.size(); i++)
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
            const auto& dimLengths = output.desc.GetLengths();

            auto result_dataFloat = tensor<float>(dimLengths);

            tensor<T>& result_dataT = std::get<0>(results);

            for(size_t i = 0; i < result_dataT.data.size(); i++)
                result_dataFloat.data[i] = convert_type<float>(result_dataT.data[i]);

            return (result_dataFloat);
        }
        else
        {
            const auto& dimLengths = indices.desc.GetLengths();

            auto result_indicesFloat = tensor<float>(dimLengths);

            tensor<int>& result_indices = std::get<1>(results);

            for(size_t i = 0; i < result_indices.data.size(); i++)
                result_indicesFloat.data[i] = static_cast<float>(result_indices.data[i]);

            return (result_indicesFloat);
        };
    };

    template <typename compType>
    std::tuple<tensor<T>, tensor<int>> cpuImpl() const
    {
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
        {
            if(inLengths[i] == outLengths[i])
                invariantDims.push_back(i);
            else
                toReduceDims.push_back(i);
        }

        invariantLengths.resize(invariantDims.size());
        std::transform(invariantDims.begin(),
                       invariantDims.end(),
                       invariantLengths.begin(),
                       [&](int i) { return inLengths[i]; });

        toReduceLengths.resize(toReduceDims.size());
        std::transform(toReduceDims.begin(),
                       toReduceDims.end(),
                       toReduceLengths.begin(),
                       [&](int i) { return inLengths[i]; });

        bool reduceAllDims = invariantDims.empty();

        auto opReduce = reduce::ReduceOpFn2<compType>(reduceOp);

        std::size_t divider = std::accumulate(
            toReduceLengths.begin(), toReduceLengths.end(), std::size_t{1}, std::multiplies<>{});

        auto PreUnaryOp = reduce::PreUnaryOpFn<compType>(reduceOp, divider);

        if(reduceAllDims)
        {
            cpuImplReduceAllDims<compType>(
                inLengths, inStrides, PreUnaryOp, opReduce, res, res_indices);
        }
        else
        {
            cpuImplReduceFewDims<compType>(invariantLengths,
                                           toReduceLengths,
                                           inLengths,
                                           invariantDims,
                                           outStrides,
                                           toReduceDims,
                                           inStrides,
                                           PreUnaryOp,
                                           opReduce,
                                           res,
                                           res_indices);
        };

        return (std::make_tuple(res, res_indices));
    }

    template <typename compType>
    void cpuImplReduceFewDims(const std::vector<std::size_t>& invariantLengths,
                              const std::vector<std::size_t>& toReduceLengths,
                              const std::vector<std::size_t>& inLengths,
                              const std::vector<int>& invariantDims,
                              const std::vector<std::size_t>& outStrides,
                              const std::vector<int>& toReduceDims,
                              const std::vector<std::size_t>& inStrides,
                              auto& PreUnaryOp,
                              auto& opReduce,
                              tensor<T>& res,
                              tensor<int>& res_indices) const
    {
        using reduce::convert_type;

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

            for(int k = 0; k < invariantDims.size(); k++)
                dst_index[invariantDims[k]] = index_1[k];

            auto dst_offset = get_offset_from_index(outStrides, dst_index);

            // generate the part of the index belonging to the invariant dims
            for(int k = 0; k < invariantDims.size(); k++)
                src_index[invariantDims[k]] = index_1[k];

            compType accuVal = reduce::ReduceOpZeroVal<compType>(reduceOp);
            int accuIndex    = 0;

            // go through indexes of the toReduce dimensions
            for(const auto& index_2 : indexes_2)
            {
                // generate the part of the index belonging to the toReduce dims
                for(int k = 0; k < toReduceDims.size(); k++)
                    src_index[toReduceDims[k]] = index_2[k];

                auto src_offset = get_offset_from_index(inStrides, src_index);

                auto currVal = convert_type<compType>(input.data[src_offset]);

                // unary operation before reducing, only needed by AMAX. For MIN/MAX, nothing is
                // actually done
                PreUnaryOp(currVal);

                auto currIndex = get_flatten_offset(toReduceLengths, index_2);
                reduce::binop_with_nan_check2(
                    nanOpt, opReduce, accuVal, currVal, accuIndex, currIndex);
            };

            // scale the accumulated value
            if(!reduce::float_equal_one(alpha))
                accuVal *= convert_type<compType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!reduce::float_equal_zero(beta))
            {
                accuVal +=
                    convert_type<compType>(output.data[dst_offset]) * convert_type<compType>(beta);
            }

            // store the reduced value to dst location
            res.data[dst_offset]         = convert_type<T>(accuVal);
            res_indices.data[dst_offset] = accuIndex; // store the index
        };
    }

    template <typename compType>
    void cpuImplReduceAllDims(const std::vector<std::size_t>& inLengths,
                              const std::vector<std::size_t>& inStrides,
                              auto& PreUnaryOp,
                              auto& opReduce,
                              tensor<T>& res,
                              tensor<int>& res_indices) const
    {
        using reduce::convert_type;

        std::vector<std::vector<std::size_t>> indexes_1;

        get_all_indexes(inLengths, 0, indexes_1);

        compType accuVal = reduce::ReduceOpZeroVal<compType>(reduceOp);
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
            reduce::binop_with_nan_check2(nanOpt, opReduce, accuVal, currVal, accuIndex, currIndex);
        }

        // scale the accumulated value
        if(!reduce::float_equal_one(alpha))
            accuVal *= convert_type<compType>(alpha);

        // scale the prior dst value and add it to the accumulated value
        if(!reduce::float_equal_zero(beta))
        {
            accuVal += convert_type<compType>(output.data[0]) * convert_type<compType>(beta);
        }

        // store the reduced value to dst location
        res.data[0]         = convert_type<T>(accuVal);
        res_indices.data[0] = accuIndex;
    }

    std::tuple<tensor<T>, tensor<int>> gpuImpl() const
    {
        auto&& handle   = get_handle();
        auto input_dev  = handle.Write(input.data);
        auto output_dev = handle.Write(output.data);

        // replicate
        auto res         = output;
        auto res_indices = indices;

        Workspace idxspace{};
        idxspace.Write(indices.data);

        Workspace wspace{};
        wspace.Write(workspace.data);

        const double alpha64 = alpha;
        const double beta64  = beta;

        const void* const alphaPtr = (std::is_same<T, double>::value)
                                         ? static_cast<const void*>(&alpha64)
                                         : static_cast<const void*>(&alpha);
        const void* const betaPtr  = (std::is_same<T, double>::value)
                                         ? static_cast<const void*>(&beta64)
                                         : static_cast<const void*>(&beta);

        if(wspace.size() > 0)
        {
            reduce.ReduceTensor(get_handle(),
                                idxspace.ptr(),
                                idxspace.size(),
                                wspace.ptr(),
                                wspace.size(),
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
                                idxspace.ptr(),
                                idxspace.size(),
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
        res_indices.data = idxspace.Read<decltype(res_indices.data)>();

        return (std::make_tuple(res, res_indices));
    }

    void fail() const
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
        : reduce(reduce_),
          input(input_),
          output(output_),
          workspace(workspace_),
          alpha(alpha_),
          beta(beta_),
          reduceOp(reduce_.reduceTensorOp_),
          compTypeVal(reduce_.reduceTensorCompType_),
          nanOpt(reduce_.reduceTensorNanOpt_)
    {
    }

    tensor<float> cpu() const
    {
        using reduce::convert_type;

        const tensor<T> result = GetCpuImplResult<T, verify_reduce_no_indices>(compTypeVal, *this);

        const auto& dimLengths = output.desc.GetLengths();
        auto result_dataFloat  = tensor<float>(dimLengths);

        for(size_t i = 0; i < result.data.size(); i++)
            result_dataFloat.data[i] = convert_type<float>(result.data[i]);

        return (result_dataFloat);
    };

    template <typename compType>
    tensor<T> cpuImpl() const
    {
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
        {
            if(inLengths[i] == outLengths[i])
                invariantDims.push_back(i);
            else
                toReduceDims.push_back(i);
        }

        invariantLengths.resize(invariantDims.size());
        for(int i = 0; i < invariantDims.size(); i++)
            invariantLengths[i] = inLengths[invariantDims[i]];

        toReduceLengths.resize(toReduceDims.size());
        for(int i = 0; i < toReduceDims.size(); i++)
            toReduceLengths[i] = inLengths[toReduceDims[i]];

        bool reduceAllDims = invariantDims.empty();

        auto opReduce = reduce::ReduceOpFn<compType>(reduceOp);

        std::size_t divider = std::accumulate(
            toReduceLengths.begin(), toReduceLengths.end(), std::size_t{1}, std::multiplies<>{});

        auto PreUnaryOp = reduce::PreUnaryOpFn<compType>(reduceOp, divider);
        auto PosUnaryOp = reduce::PosUnaryOpFn<compType>(reduceOp, divider);

        if(reduceAllDims)
        {
            cpuImplReduceAllDims<compType>(
                inLengths, inStrides, PreUnaryOp, opReduce, PosUnaryOp, res);
        }
        else
        {
            cpuImplReduceFewDims<compType>(invariantLengths,
                                           toReduceLengths,
                                           inLengths,
                                           invariantDims,
                                           outStrides,
                                           toReduceDims,
                                           inStrides,
                                           PreUnaryOp,
                                           opReduce,
                                           PosUnaryOp,
                                           res);
        };

        return (res);
    }

    template <typename compType>
    void cpuImplReduceFewDims(const std::vector<std::size_t>& invariantLengths,
                              const std::vector<std::size_t>& toReduceLengths,
                              const std::vector<std::size_t>& inLengths,
                              const std::vector<int>& invariantDims,
                              const std::vector<std::size_t>& outStrides,
                              const std::vector<int>& toReduceDims,
                              const std::vector<std::size_t>& inStrides,
                              auto& PreUnaryOp,
                              auto& opReduce,
                              auto& PosUnaryOp,
                              tensor<T>& res) const
    {
        using reduce::convert_type;

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

            for(int k = 0; k < invariantDims.size(); k++)
                dst_index[invariantDims[k]] = index_1[k];

            auto dst_offset = get_offset_from_index(outStrides, dst_index);

            // generate the part of the index belonging to the invariant dims
            for(int k = 0; k < invariantDims.size(); k++)
                src_index[invariantDims[k]] = index_1[k];

            compType accuVal = reduce::ReduceOpZeroVal<compType>(reduceOp);

            // go through indexes of the toReduce dimensions
            for(const auto& index_2 : indexes_2)
            {
                // generate the part of the index belonging to the toReduce dims
                for(int k = 0; k < toReduceDims.size(); k++)
                    src_index[toReduceDims[k]] = index_2[k];

                auto src_offset = get_offset_from_index(inStrides, src_index);

                auto currVal = convert_type<compType>(input.data[src_offset]);

                PreUnaryOp(currVal);

                reduce::binop_with_nan_check(nanOpt, opReduce, accuVal, currVal);
            };

            PosUnaryOp(accuVal);

            // scale the accumulated value
            if(!reduce::float_equal_one(alpha))
                accuVal *= convert_type<compType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!reduce::float_equal_zero(beta))
            {
                accuVal +=
                    convert_type<compType>(output.data[dst_offset]) * convert_type<compType>(beta);
            }

            // store the reduced value to dst location
            res.data[dst_offset] = convert_type<T>(accuVal);
        }
    }

    template <typename compType>
    void cpuImplReduceAllDims(const std::vector<std::size_t>& inLengths,
                              const std::vector<std::size_t>& inStrides,
                              auto& PreUnaryOp,
                              auto& opReduce,
                              auto& PosUnaryOp,
                              tensor<T>& res) const
    {
        using reduce::convert_type;

        std::vector<std::vector<std::size_t>> indexes_1;

        get_all_indexes(inLengths, 0, indexes_1);

        compType accuVal = reduce::ReduceOpZeroVal<compType>(reduceOp);

        // go through indexes of the invariant dimensions
        for(const auto& src_index : indexes_1)
        {
            auto src_offset = get_offset_from_index(inStrides, src_index);

            auto currVal = convert_type<compType>(input.data[src_offset]);

            PreUnaryOp(currVal);

            reduce::binop_with_nan_check(nanOpt, opReduce, accuVal, currVal);
        };

        PosUnaryOp(accuVal);

        // scale the accumulated value
        if(!reduce::float_equal_one(alpha))
            accuVal *= convert_type<compType>(alpha);

        // scale the prior dst value and add it to the accumulated value
        if(!reduce::float_equal_zero(beta))
            accuVal += convert_type<compType>(output.data[0]) * convert_type<compType>(beta);

        // store the reduced value to dst location
        res.data[0] = convert_type<T>(accuVal);
    }

    tensor<float> gpu() const
    {
        using reduce::convert_type;

        auto result = gpuImpl();

        const auto& dimLengths = output.desc.GetLengths();
        auto result_dataFloat  = tensor<float>(dimLengths);

        for(size_t i = 0; i < result.data.size(); i++)
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

        Workspace wspace{};
        wspace.Write(workspace.data);

        const double alpha64 = alpha;
        const double beta64  = beta;

        const void* const alphaPtr = (std::is_same<T, double>::value)
                                         ? static_cast<const void*>(&alpha64)
                                         : static_cast<const void*>(&alpha);
        const void* const betaPtr  = (std::is_same<T, double>::value)
                                         ? static_cast<const void*>(&beta64)
                                         : static_cast<const void*>(&beta);

        if(wspace.size() > 0)
        {
            reduce.ReduceTensor(get_handle(),
                                nullptr,
                                0,
                                wspace.ptr(),
                                wspace.size(),
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

    void fail() const
    {
        std::cout << "verify_reduce_no_indices failed" << std::endl;
        std::cout << "Input Tensor"
                  << " " << input.desc.ToString() << std::endl;
    }
};

template <typename T>
std::vector<std::vector<std::size_t>> get_tensor_lengths()
{
    if(std::is_same<T, half_float::half>::value)
    {
        return {
            {4, 3, 60, 50},
        };
    }
    else
    {
        return {
            {64, 3, 280, 81},
        };
    }
}

std::vector<std::vector<int>> get_toreduce_dims()
{
    std::vector<std::vector<int>> tensor_dims = {
        {0}, {1}, {2}, {3}, {0, 1}, {0, 3}, {0, 2}, {2, 3}, {0, 1, 3}, {1, 2, 3}, {0, 1, 2, 3}};

    return tensor_dims;
}

template <typename T>
inline auto GenCases()
{
    std::vector<std::vector<float>> alphabetas = {{1.0f, 0.0f}, {0.5f, 0.5f}};

    return testing::Combine(testing::ValuesIn(get_tensor_lengths<T>()),
                            testing::ValuesIn(get_toreduce_dims()),
                            testing::Values(MIOPEN_REDUCE_TENSOR_ADD,
                                            MIOPEN_REDUCE_TENSOR_MUL,
                                            MIOPEN_REDUCE_TENSOR_AMAX,
                                            MIOPEN_REDUCE_TENSOR_AVG,
                                            MIOPEN_REDUCE_TENSOR_NORM1,
                                            MIOPEN_REDUCE_TENSOR_NORM2),
                            testing::Values(0, 1),
                            testing::Values(0, 1),
                            testing::ValuesIn(alphabetas));
}

template <typename T>
inline auto GetCases()
{
    static const auto cases = GenCases<T>();
    return cases;
}

inline auto GetCasesReduceCustomTestSet1()
{
    static TestCase test_case = std::make_tuple(std::vector<size_t>{1024, 30528, 1},
                                                std::vector<int>{0, 1, 2},
                                                MIOPEN_REDUCE_TENSOR_ADD,
                                                MIOPEN_PROPAGATE_NAN,
                                                MIOPEN_REDUCE_TENSOR_NO_INDICES,
                                                std::vector<float>{1.0f, 0.0f});

    return testing::Values(test_case);
}

inline auto GetCasesReduceCustomTestSet2()
{
    static std::vector<float> alphabeta = {1.0f, 0.0f};

    constexpr miopenReduceTensorOp_t add = MIOPEN_REDUCE_TENSOR_ADD;
    constexpr miopenReduceTensorOp_t mul = MIOPEN_REDUCE_TENSOR_MUL;
    constexpr miopenReduceTensorOp_t min = MIOPEN_REDUCE_TENSOR_MIN;
    constexpr miopenReduceTensorOp_t max = MIOPEN_REDUCE_TENSOR_MAX;

    constexpr miopenNanPropagation_t nan   = MIOPEN_PROPAGATE_NAN;
    constexpr miopenNanPropagation_t nonan = MIOPEN_NOT_PROPAGATE_NAN;

    constexpr miopenReduceTensorIndices_t noind = MIOPEN_REDUCE_TENSOR_NO_INDICES;
    constexpr miopenReduceTensorIndices_t ind   = MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES;

    // clang-format off
    static std::vector<TestCase> test_cases = {
        std::make_tuple(std::vector<size_t>{8, 2, 1}, std::vector<int>{0, 1, 2}, add, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{160, 10, 1}, std::vector<int>{0, 1, 2}, add, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{7, 1024, 1}, std::vector<int>{0, 1, 2}, add, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{3, 1, 1}, std::vector<int>{0, 1, 2}, add, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{3, 1, 1}, std::vector<int>{0, 1, 2}, mul, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{3, 1, 1}, std::vector<int>{0, 1, 2}, max, nan, ind, alphabeta),
        std::make_tuple(std::vector<size_t>{3, 2, 1}, std::vector<int>{1, 2}, max, nan, ind, alphabeta),
        std::make_tuple(std::vector<size_t>{6, 2, 1}, std::vector<int>{1, 2}, max, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{6, 2, 1}, std::vector<int>{1, 2}, min, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{2, 2, 1}, std::vector<int>{1, 2}, add, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{4, 3, 1}, std::vector<int>{1, 2}, max, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{3, 4, 1}, std::vector<int>{1, 2}, max, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{3, 4, 1}, std::vector<int>{0, 2}, max, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{2048, 32, 1}, std::vector<int>{0, 2}, max, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{4, 3, 1}, std::vector<int>{1, 2}, min, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{3, 4, 1}, std::vector<int>{0, 2}, min, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{2048, 32, 1}, std::vector<int>{0, 2}, min, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{12, 11, 1}, std::vector<int>{0, 1, 2}, add, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{13, 4, 7, 7},std::vector<int>{0, 1, 2, 3}, add, nan, noind, alphabeta),
        std::make_tuple(std::vector<size_t>{64, 3, 280, 81}, std::vector<int>{0}, add, nonan, noind, alphabeta),
    };
    // clang-format on

    return testing::ValuesIn(test_cases);
}

} // anonymous namespace

template <class T>
struct ReduceCommon : public testing::TestWithParam<TestCase>
{
    void SetUp() override
    {
        auto&& handle           = get_handle();
        std::string device_name = handle.GetDeviceName();

        prng::reset_seed();

        handle.EnableProfiling();

        std::tie(inLengths, toReduceDims, reduceOp, nanOpt, indicesOpt, scales) = GetParam();
    }

    void Run()
    {
        using reduce::convert_type;

        if(std::is_same<T, double>::value)
            compTypeVal = miopenDouble;

        if(std::is_same<T, half_float::half>::value)
        {
            if(reduceOp == MIOPEN_REDUCE_TENSOR_MIN || reduceOp == MIOPEN_REDUCE_TENSOR_MAX ||
               reduceOp == MIOPEN_REDUCE_TENSOR_AMAX)
            {
                compTypeVal = miopenHalf; // let compType be same as the data type
            }
            else
            {
                compTypeVal = miopenFloat;
            }
        }

        miopen::ReduceTensorDescriptor reduceDesc(
            reduceOp, compTypeVal, nanOpt, indicesOpt, indicesType);

        float alpha = scales[0];
        float beta  = scales[1];

        // The test is ignored if (alpha, beta) is not (1.0f, 0.0f) and reduceOp is not Add/MUL/AVG
        if(reduceOp != MIOPEN_REDUCE_TENSOR_ADD && reduceOp != MIOPEN_REDUCE_TENSOR_MUL &&
           reduceOp != MIOPEN_REDUCE_TENSOR_AVG && alpha != 1.0f && beta != 0.0f)
        {
            GTEST_SKIP();
        }

        // The test is ignored if indices are requested but the reduceOp is neither MIN nor MAX
        if(indicesOpt != MIOPEN_REDUCE_TENSOR_NO_INDICES && reduceOp != MIOPEN_REDUCE_TENSOR_MIN &&
           reduceOp != MIOPEN_REDUCE_TENSOR_MAX && reduceOp != MIOPEN_REDUCE_TENSOR_AMAX)
        {
            GTEST_SKIP();
        }

        auto outLengths = this->inLengths;

        assert(toReduceDims.size() <= outLengths.size());

        // set the lengths of the dimensions to be reduced to 1 to represent the output Tensor
        for(const int& toReduceDim : toReduceDims)
        {
            assert(toReduceDim < inLengths.size());
            outLengths[toReduceDim] = static_cast<std::size_t>(1);
        }

        uint64_t max_value;

        if(reduceOp == MIOPEN_REDUCE_TENSOR_MUL)
        {
            max_value = miopen_type<T>{} == miopenHalf   ? 41
                        : miopen_type<T>{} == miopenInt8 ? 127
                                                         : 111;
        }
        else if(reduceOp == MIOPEN_REDUCE_TENSOR_NORM1 || reduceOp == MIOPEN_REDUCE_TENSOR_NORM2)
        {
            max_value = 3;
        }
        else
        {
            max_value = miopen_type<T>{} == miopenHalf   ? 13
                        : miopen_type<T>{} == miopenInt8 ? 127
                                                         : 999;
        }

        // default data gneration (used by MIN/MAX)
        auto gen_value_min_max = [&](auto... is) {
            return (tensor_elem_gen_integer{max_value}(is...) *
                    tensor_elem_gen_checkboard_sign{}(is...));
        };

        // data generation used by ADD/AVG, data is distributed around 1.0 rather than 0.0, very low
        // probability to get a reduced result of zero-value
        auto gen_value_add_avg = [&](auto... is) {
            auto rand_value = tensor_elem_gen_integer{max_value}(is...);
            auto sign_value = tensor_elem_gen_checkboard_sign{}(is...);

            return (sign_value * rand_value / max_value + 0.01);
        };

        // Special data generation for MUL, to avoid all-zero and large accumulative error in the
        // reduced result
        auto gen_value_mul = [&](auto... is) {
            auto rand_value = tensor_elem_gen_integer{max_value}(is...);
            auto sign_value = tensor_elem_gen_checkboard_sign{}(is...);

            return sign_value > 0.0 ? (rand_value + max_value) / (rand_value + max_value + 1)
                                    : (rand_value + max_value + 1) / (rand_value + max_value);
        };

        // Special data generation for NORM1 and NORM2 using a space of limitless number of values.
        auto gen_value_norm1_norm2 = [&](auto... is) {
            auto rand_upper = tensor_elem_gen_integer{max_value}(is...);
            auto sign_value = tensor_elem_gen_checkboard_sign{}(is...);
            auto rand_ratio = prng::gen_A_to_B(
                0.1, 1.); // limit range due to numeric errors, see WORKAROUND_GPU_NUMERIC_ERROR

            return rand_upper * sign_value * rand_ratio;
        };

        // Special data generation for AMAX, no zero value used
        auto gen_value_amax = [&](auto... is) {
            auto rand_value = tensor_elem_gen_integer{max_value}(is...);
            auto sign_value = tensor_elem_gen_checkboard_sign{}(is...);

            return sign_value > 0.0 ? (rand_value + 0.5) : (-1.0 * rand_value - 0.5);
        };

        // default tolerance (refer to driver.hpp)
        this->tolerance = 80;

        if(reduceOp == MIOPEN_REDUCE_TENSOR_ADD || reduceOp == MIOPEN_REDUCE_TENSOR_AVG)
            this->tolerance = 80 * 10;
        if(reduceOp == MIOPEN_REDUCE_TENSOR_MUL)
        {
            this->tolerance = 80 * 300;
        }
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
            inputTensor = tensor<T>{this->inLengths}.generate(gen_value_add_avg);
            break;
        case MIOPEN_REDUCE_TENSOR_MUL:
            inputTensor = tensor<T>{this->inLengths}.generate(gen_value_mul);
            break;
        case MIOPEN_REDUCE_TENSOR_NORM1:
        case MIOPEN_REDUCE_TENSOR_NORM2:
            inputTensor = tensor<T>{this->inLengths}.generate(gen_value_norm1_norm2);
            break;
        case MIOPEN_REDUCE_TENSOR_AMAX:
            inputTensor = tensor<T>{this->inLengths}.generate(gen_value_amax);
            break;
        default: inputTensor = tensor<T>{this->inLengths}.generate(gen_value_min_max);
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

            VerifyReduceWithIndices<true>(
                reduceDesc, inputTensor, outputTensor, workspaceTensor, indicesTensor, 1.0f, 0.0f);

            VerifyReduceWithIndices<false>(
                reduceDesc, inputTensor, outputTensor, workspaceTensor, indicesTensor, 1.0f, 0.0f);
        }
        else
        {
            VerifyReduceNoIndices(
                reduceDesc, inputTensor, outputTensor, workspaceTensor, alpha, beta);
        };
    };

private:
    template <bool toVerifyData>
    void VerifyReduceWithIndices(const miopen::ReduceTensorDescriptor& reduce,
                                 const tensor<T>& input,
                                 const tensor<T>& output,
                                 const tensor<T>& workspace,
                                 const tensor<int>& indices,
                                 float alpha,
                                 float beta) const
    {
        verify_reduce_with_indices<T, toVerifyData> reduce_with_indices(
            reduce, input, output, workspace, indices, alpha, beta);
        CompareResults(reduce_with_indices, toVerifyData);
    }

    void VerifyReduceNoIndices(const miopen::ReduceTensorDescriptor& reduce,
                               const tensor<T>& input,
                               const tensor<T>& output,
                               const tensor<T>& workspace,
                               float alpha,
                               float beta) const
    {
        verify_reduce_no_indices<T> reduce_no_indices(
            reduce, input, output, workspace, alpha, beta);
        CompareResults(reduce_no_indices, true);
    }

    std::string GetOutputValuesForError() const
    {
        std::ostringstream oss;
        oss << "reduceOp: " << reduceOp << std::endl
            << "compTypeVal: " << compTypeVal << std::endl
            << "nanOpt: " << nanOpt << std::endl
            << "indicesOpt: " << indicesOpt << std::endl;
        return oss.str();
    }

    template <class TVerifier>
    void CompareResults(const TVerifier& verifier, bool toVerifyData) const
    {
        const tensor<float> cpu = std::move(verifier.cpu());
        const tensor<float> gpu = std::move(verifier.gpu());

        if(toVerifyData)
        {
            double threshold = std::numeric_limits<float>::epsilon() * tolerance;
            double error     = miopen::rms_range(cpu, gpu);

            if(error > threshold)
            {
                verifier.fail();

                std::cout << "Tolerance: " << tolerance << std::endl;
            }

            EXPECT_LE(error, threshold) << GetOutputValuesForError();
        }
        else
        {
            auto idx   = miopen::mismatch_idx(cpu, gpu, miopen::float_equal);
            auto range = miopen::range_distance(cpu);
            if(idx < range)
            {
                verifier.fail();
            }

            EXPECT_GE(idx, range) << GetOutputValuesForError();
        }
    }

private:
    miopenReduceTensorOp_t reduceOp        = MIOPEN_REDUCE_TENSOR_ADD;
    miopenDataType_t compTypeVal           = miopenFloat;
    miopenNanPropagation_t nanOpt          = MIOPEN_NOT_PROPAGATE_NAN;
    miopenReduceTensorIndices_t indicesOpt = MIOPEN_REDUCE_TENSOR_NO_INDICES;

    miopenIndicesType_t indicesType = MIOPEN_32BIT_INDICES;

    std::vector<std::size_t> inLengths; // the lengths of the input tensor's dimensions
    std::vector<int>
        toReduceDims; // the indexes of the dimensions to be reduced in the input tensor

    std::vector<float> scales;

    double tolerance = 0.0; // will be calculated during test execution
};

using GPU_Reduce_FP32 = ReduceCommon<float>;
using GPU_Reduce_FP16 = ReduceCommon<half_float::half>;
using GPU_Reduce_FP64 = ReduceCommon<double>;

TEST_P(GPU_Reduce_FP32, TestFloat) { this->Run(); }
TEST_P(GPU_Reduce_FP16, TestFloat16) { this->Run(); }
TEST_P(GPU_Reduce_FP64, TestDouble) { this->Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Reduce_FP32, GetCases<float>());
INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Reduce_FP16, GetCases<half_float::half>());
INSTANTIATE_TEST_SUITE_P(Full, GPU_Reduce_FP64, GetCases<double>());

// Reduce Custom Tests
template <typename T>
class ReduceCustomCommon : public ReduceCommon<T>
{
    void SetUp() override
    {
        using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
        using d_mask = disabled<Gpu::Default>;
        if(!::IsTestSupportedForDevMask<d_mask, e_mask>())
        {
            GTEST_SKIP();
        }

        ReduceCommon<T>::SetUp();
    }
};

// original file reduce_custom_fp32.cpp
class GPU_reduce_custom_fp32_FP32 : public ReduceCustomCommon<float>
{
};

class GPU_reduce_custom_fp32_FP16 : public ReduceCustomCommon<half_float::half>
{
};

class GPU_reduce_custom_fp32_BFP16 : public ReduceCustomCommon<bfloat16>
{
};

class GPU_reduce_custom_fp32_I8 : public ReduceCustomCommon<int8_t>
{
};

TEST_P(GPU_reduce_custom_fp32_FP32, FloatTest_reduce_custom_fp32) { this->Run(); };
INSTANTIATE_TEST_SUITE_P(Full, GPU_reduce_custom_fp32_FP32, GetCasesReduceCustomTestSet1());

TEST_P(GPU_reduce_custom_fp32_FP16, CAN_BE_DISABLED_HalfTest_reduce_custom_fp16) { this->Run(); };
INSTANTIATE_TEST_SUITE_P(Full, GPU_reduce_custom_fp32_FP16, GetCasesReduceCustomTestSet1());

TEST_P(GPU_reduce_custom_fp32_BFP16, CAN_BE_DISABLED_BHalfTest_reduce_custom_fp16) { this->Run(); };
INSTANTIATE_TEST_SUITE_P(Full, GPU_reduce_custom_fp32_BFP16, GetCasesReduceCustomTestSet1());

TEST_P(GPU_reduce_custom_fp32_I8, CAN_BE_DISABLED_IntTest_reduce_custom_fp16) { this->Run(); };
INSTANTIATE_TEST_SUITE_P(Full, GPU_reduce_custom_fp32_I8, GetCasesReduceCustomTestSet1());

// original file reduce_custom_fp32_fp16.cpp
class GPU_reduce_custom_fp32_fp16_FP32 : public ReduceCustomCommon<float>
{
};

class GPU_reduce_custom_fp32_fp16_FP16 : public ReduceCustomCommon<half_float::half>
{
};

TEST_P(GPU_reduce_custom_fp32_fp16_FP32, FloatTest_reduce_custom_fp32_fp16) { this->Run(); };

TEST_P(GPU_reduce_custom_fp32_fp16_FP16, HalfTest_reduce_custom_fp32_fp16) { this->Run(); };

INSTANTIATE_TEST_SUITE_P(Full, GPU_reduce_custom_fp32_fp16_FP32, GetCasesReduceCustomTestSet2());

INSTANTIATE_TEST_SUITE_P(Full, GPU_reduce_custom_fp32_fp16_FP16, GetCasesReduceCustomTestSet2());
