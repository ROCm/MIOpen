/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "miopen/common.hpp"
#include "miopen/errors.hpp"
#include "miopen/miopen.h"
#include "miopen/tensor.hpp"
#include <cstddef>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/logcumsumexp/invoke_params.hpp>
#include <miopen/logcumsumexp/solvers.hpp>
#include <miopen/mlo_internal.hpp>

#define warpSizeCTX (context.GetStream().GetWavefrontWidth())
#define LOCAL_SIZE_MAX 1024
#define LOCAL_SIZE_MIN warpSizeCTX
#define LOCAL_SIZE_IMPROVEMENT_OVER_ROCM 256

namespace miopen {

namespace solver {

namespace logcumsumexp {

namespace {
bool IsImprovementOverROCm(const ExecutionContext& /*context*/,
                           const miopen::logcumsumexp::BackwardProblemDescription& problem)
{
    if(problem.GetInputDesc().GetLengths()[problem.GetDim()] > LOCAL_SIZE_IMPROVEMENT_OVER_ROCM)
        return false;
    return true;
}
} // namespace

bool BackwardContiguousSmallLastDim::IsApplicable(
    const ExecutionContext& context,
    const miopen::logcumsumexp::BackwardProblemDescription& problem) const
{
    if(!IsImprovementOverROCm(context, problem))
        return false;
    if(problem.GetInputDesc().GetLengths()[problem.GetDim()] > LOCAL_SIZE_MAX)
        return false;
    if(!problem.IsAllPacked())
        return false;
    if(!problem.IsAllDimStride1())
        return false;
    return true;
}

ConvSolution BackwardContiguousSmallLastDim::GetSolution(
    const ExecutionContext& context,
    const miopen::logcumsumexp::BackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetInputDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

    auto size       = problem.GetInputDesc().GetElementSize();
    auto inner_size = problem.GetInputDesc().GetLengths()[problem.GetDim()];
    auto outer_size = size / inner_size;

    // LOCAL_SIZE must be the smallest power of 2 that greater than inner_size and warpSize
    auto local_size = LOCAL_SIZE_MIN;
    while(local_size < inner_size)
        local_size *= 2;

    // Calculate log_grad_positive and log_grad_negative
    {
        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"REDUCE_SIZE", local_size},
        };
        result.construction_params.push_back(KernelInfo{
            build_params.GenerateFor(kbp::HIP{}),
            {LOCAL_SIZE_IMPROVEMENT_OVER_ROCM},
            {AlignUp(size, LOCAL_SIZE_IMPROVEMENT_OVER_ROCM)},
            "MIOpenLogCumSumExp.cpp",
            "InitLogGradContiguous",
        });
    }

    // Calculate pos_reverse_logcumsumexp and neg_reverse_logcumsumexp
    {
        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", 0},
            {"MIOPEN_USE_FP32", 1},
            {"MIOPEN_USE_FP64", 0},
            {"MIOPEN_USE_BFP16", 0},
            {"REDUCE_SIZE", local_size},
        };
        // pos_reverse_logcumsumexp
        result.construction_params.push_back(KernelInfo{
            build_params.GenerateFor(kbp::HIP{}),
            {1, local_size},
            {outer_size, AlignUp(inner_size, local_size)},
            "MIOpenLogCumSumExp.cpp",
            "LogCumSumExpForwardContiguousSmallLastDim",
        });

        // neg_reverse_logcumsumexp
        result.construction_params.push_back(KernelInfo{
            build_params.GenerateFor(kbp::HIP{}),
            {1, local_size},
            {outer_size, AlignUp(inner_size, local_size)},
            "MIOpenLogCumSumExp.cpp",
            "LogCumSumExpForwardContiguousSmallLastDim",
        });
    }

    // Calculate Input Grad
    {
        auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"REDUCE_SIZE", local_size},
        };
        result.construction_params.push_back(KernelInfo{
            build_params.GenerateFor(kbp::HIP{}),
            {LOCAL_SIZE_IMPROVEMENT_OVER_ROCM},
            {AlignUp(size, LOCAL_SIZE_IMPROVEMENT_OVER_ROCM)},
            "MIOpenLogCumSumExp.cpp",
            "LogCumSumExp1dBackwardStep2Contiguous",
        });
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            auto params = raw_params.CastTo<miopen::logcumsumexp::InvokeParamsBackward>();

            uint64_t size          = deref(params.inputDesc).GetElementSize();
            auto log_grad_positive = params.workspace;
            auto log_grad_negative =
                static_cast<Data_t>(static_cast<std::byte*>(log_grad_positive) +
                                    size * miopen::GetTypeSize(miopenFloat));
            auto pos_reverse_logcumsumexp =
                static_cast<Data_t>(static_cast<std::byte*>(log_grad_negative) +
                                    size * miopen::GetTypeSize(miopenFloat));
            auto neg_reverse_logcumsumexp =
                static_cast<Data_t>(static_cast<std::byte*>(pos_reverse_logcumsumexp) +
                                    size * miopen::GetTypeSize(miopenFloat));

            int kernelCnt = 0;

            // InitLogGrad
            {
                auto kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(params.doutput, params.output, log_grad_positive, log_grad_negative, size);
            }

            // LogCumSumExp1dForward
            {
                const int ndims             = deref(params.inputDesc).GetNumDims();
                const unsigned int true_dim = ((params.dim % ndims) + ndims) % ndims;
                auto kernel                 = handle_.Run(kernels[kernelCnt++]);
                kernel(log_grad_positive,
                       pos_reverse_logcumsumexp,
                       deref(params.inputDesc).GetLengths()[true_dim],
                       /*exclusive=*/false,
                       params.reverse);
            }
            {
                const int ndims             = deref(params.inputDesc).GetNumDims();
                const unsigned int true_dim = ((params.dim % ndims) + ndims) % ndims;
                auto kernel                 = handle_.Run(kernels[kernelCnt++]);
                kernel(log_grad_negative,
                       neg_reverse_logcumsumexp,
                       deref(params.inputDesc).GetLengths()[true_dim],
                       /*exclusive=*/false,
                       params.reverse);
            }

            // LogCumSumExp1dBackwardStep2
            {
                const int ndims             = deref(params.inputDesc).GetNumDims();
                const unsigned int true_dim = ((params.dim % ndims) + ndims) % ndims;
                uint64_t dim_size           = deref(params.inputDesc).GetLengths()[true_dim];
                auto kernel                 = handle_.Run(kernels[kernelCnt++]);
                kernel(pos_reverse_logcumsumexp,
                       neg_reverse_logcumsumexp,
                       params.input,
                       params.dinput,
                       size,
                       dim_size,
                       params.exclusive);
            }
        };
    };

    return result;
}

} // namespace logcumsumexp
} // namespace solver
} // namespace miopen
