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

#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/logsumexp/invoke_params.hpp>
#include <miopen/logsumexp/solvers.hpp>
#include <miopen/logsumexp.hpp>
#include "miopen/logsumexp/problem_description.hpp"

#define LOCAL_SIZE 1024
#define LOCAL_SIZE_LARGE_K_64 64
#define LIMIT_SMALL_K 16

#define VIEW_DIMS 5

namespace miopen {

namespace solver {

namespace logsumexp {

namespace {

std::size_t sizeof_local_memory(const miopen::logsumexp::ProblemDescriptionForward& problem)
{
    return LOCAL_SIZE_LARGE_K_64 * get_data_size(problem.GetInputDesc().GetType());
}

bool IsImprovementOverROCmForward(const miopen::logsumexp::ProblemDescriptionForward& problem)
{
    constexpr size_t max_input_numel = 1000000;
    constexpr size_t max_reduce_size = 1024;

    if(problem.GetInputDesc().GetElementSize() > max_input_numel)
        return false;

    size_t reduce_size =
        problem.GetInputDesc().GetElementSize() / problem.GetOutputDesc().GetElementSize();
    if(reduce_size > max_reduce_size)
        return false;

    if(!problem.IsAllPacked())
        return false;

    return true;
}
} // namespace

bool LogSumExpForward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::logsumexp::ProblemDescriptionForward& problem) const
{
    if(!(problem.GetInputDesc().GetType() == miopenFloat ||
         problem.GetInputDesc().GetType() == miopenHalf ||
         problem.GetInputDesc().GetType() == miopenBFloat16))
        return false;
    if(problem.GetInputDesc().GetNumDims() > VIEW_DIMS)
        return false;
    if(!(sizeof_local_memory(problem) <= TargetProperties::GetMaxLocalMemorySize()))
        return false;
    if(!IsImprovementOverROCmForward(problem))
        return false;
    return true;
}

ConvSolution
LogSumExpForward::GetSolution(const ExecutionContext& /*context*/,
                              const miopen::logsumexp::ProblemDescriptionForward& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto dtype = problem.GetInputDesc().GetType();

        auto input_numel  = problem.GetInputDesc().GetElementSize();
        auto output_numel = problem.GetOutputDesc().GetElementSize();
        auto reduce_size  = input_numel / output_numel;

        size_t xlocalsize;
        size_t xgridsize;
        size_t ylocalsize;
        size_t ygridsize;
        size_t zlocalsize;
        size_t zgridsize;

        if(reduce_size > LIMIT_SMALL_K)
        {
            xlocalsize = LOCAL_SIZE_LARGE_K_64;
            xgridsize  = output_numel * LOCAL_SIZE_LARGE_K_64;
            ylocalsize = 1;
            ygridsize  = 1;
            zlocalsize = 1;
            zgridsize  = 1;
        }
        else
        {
            xlocalsize = LOCAL_SIZE;
            xgridsize  = AlignUp(output_numel, xlocalsize);
            ylocalsize = 1;
            ygridsize  = 1;
            zlocalsize = 1;
            zgridsize  = 1;
        }

        auto kernel = KernelInfo{};

        if(reduce_size > LIMIT_SMALL_K)
        {
            kernel.kernel_file = "MIOpenLogSumExp.cpp";
            kernel.kernel_name = "LogSumExpLargeKForward";
        }
        else
        {
            kernel.kernel_file = "MIOpenLogSumExp.cpp";
            kernel.kernel_name = "LogSumExpSmallKForward";
        }

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"LOCAL_SIZE_LARGE_K_64", LOCAL_SIZE_LARGE_K_64},
            {"LIMIT_SMALL_K", LIMIT_SMALL_K},
            {"VIEW_DIMS", VIEW_DIMS},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::logsumexp::LogSumExpForwardInvokeParams>();

            uint64_t input_numel  = deref(params.inputDesc).GetElementSize();
            uint64_t output_numel = deref(params.outputDesc).GetElementSize();
            uint64_t reduce_size  = input_numel / output_numel;

            auto input_dims     = deref(params.inputDesc).GetLengths();
            auto output_dims    = deref(params.outputDesc).GetLengths();
            auto input_strides  = deref(params.inputDesc).GetStrides();
            auto output_strides = deref(params.outputDesc).GetStrides();

            std::vector<int> dims_vector(params.dims, params.dims + params.num_dims);
            for(int64_t d = input_dims.size() - 1; d >= 0; --d)
            {
                if(!(std::find(dims_vector.begin(), dims_vector.end(), d) != dims_vector.end()))
                    continue;
                for(int64_t dd = input_dims.size() - 1; dd > d; --dd)
                {
                    if(std::find(dims_vector.begin(), dims_vector.end(), dd) != dims_vector.end())
                        continue;
                    std::swap(input_dims[d], input_dims[dd]);
                    std::swap(input_strides[d], input_strides[dd]);
                    std::swap(output_dims[d], output_dims[dd]);
                    std::swap(output_strides[d], output_strides[dd]);
                }
            }

            auto new_inputDesc =
                TensorDescriptor(params.inputDesc->GetType(), input_dims, input_strides);
            auto new_outputDesc =
                TensorDescriptor(params.outputDesc->GetType(), output_dims, output_strides);

            auto input_tv  = get_inner_expanded_tv<VIEW_DIMS>(new_inputDesc);
            auto output_tv = get_inner_expanded_tv<VIEW_DIMS>(new_outputDesc);

            kernel(params.input,
                   params.output,
                   static_cast<uint64_t>(output_numel),
                   static_cast<uint64_t>(reduce_size),
                   input_tv,
                   output_tv);
        };
    };

    return result;
}

} // namespace logsumexp

} // namespace solver

} // namespace miopen
