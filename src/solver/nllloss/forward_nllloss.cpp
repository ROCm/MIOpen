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

#include "miopen/conv_solution.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/invoke_params.hpp"
#include "miopen/kernel_info.hpp"
#include <cstddef>
#include <miopen/nllloss/solvers.hpp>

#include <miopen/nllloss/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/nllloss.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view.hpp>

#define LOCAL_SIZE_CON_FWD 1024
#define LOCAL_SIZE_NON_CON_FWD 1024
#define LOCAL_SIZE_CON_BWD 1024
#define LOCAL_SIZE_NON_CON_BWD 1024
#define LOCAL_SIZE_REDUCE_FWD 256

namespace miopen {

namespace solver {

const auto make_hip_kernel = [](std::vector<size_t> localsize,
                                std::vector<size_t> gridsize,
                                std::string kernel_file,
                                std::string kernel_name,
                                KernelBuildParameters build_params) {
    while(localsize.size() < 3)
        localsize.push_back(1);
    while(gridsize.size() < 3)
        gridsize.push_back(1);
    for(int i = 0; i < localsize.size(); ++i)
        gridsize[i] = AlignUp(gridsize[i], localsize[i]);
    return KernelInfo{
        build_params.GenerateFor(kbp::HIP{}), localsize, gridsize, kernel_file, kernel_name};
};

namespace nllloss {

bool NLLLossUnreduceForwardSolver::IsApplicable(
    const ExecutionContext&, const miopen::nllloss::UnreduceProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightLength())
        return false;
    if(!problem.IsRightStride())
        return false;
    return true;
}

bool NLLLossUnreduceForwardContiguous::IsApplicable(
    const ExecutionContext& context,
    const miopen::nllloss::UnreduceProblemDescription& problem) const
{
    if(!problem.IsAllContiguous())
        return false;
    if(!NLLLossUnreduceForwardSolver::IsApplicable(context, problem))
        return false;
    return true;
}

ConvSolution NLLLossUnreduceForwardContiguous::GetSolution(
    const ExecutionContext& context,
    const miopen::nllloss::UnreduceProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

    {
        auto dtype     = problem.GetOutputDesc().GetType();
        size_t N_total = problem.GetNtotal();

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        };

        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_CON_FWD},
                                                             {N_total},
                                                             "MIOpenNLLLoss.cpp",
                                                             "NLLLossUnreducedForward4dContiguous",
                                                             build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::nllloss::InvokeParams>();

            size_t N_total = params.outputDesc->GetElementSize();
            auto dims      = params.inputDesc->GetLengths();
            size_t C       = dims[1];
            size_t D1      = dims[2];
            size_t D2      = dims[3];

            kernel(params.input,
                   params.target,
                   params.weight,
                   params.output,
                   params.ignore_index,
                   N_total,
                   C,
                   D1,
                   D2);
        };
    };

    return result;
}

bool NLLLossUnreduceForward4d::IsApplicable(
    const ExecutionContext& context,
    const miopen::nllloss::UnreduceProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetSize() > 4)
        return false;
    if(!NLLLossUnreduceForwardSolver::IsApplicable(context, problem))
        return false;
    return true;
}

ConvSolution NLLLossUnreduceForward4d::GetSolution(
    const ExecutionContext& context,
    const miopen::nllloss::UnreduceProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

    {
        auto dtype     = problem.GetOutputDesc().GetType();
        size_t N_total = problem.GetNtotal();

        auto kernel = KernelInfo{};

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        };

        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_NON_CON_FWD},
                                                             {N_total},
                                                             "MIOpenNLLLoss.cpp",
                                                             "NLLLossUnreducedForward4d",
                                                             build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::nllloss::InvokeParams>();

            auto input_tv  = get_inner_expanded_tv_4d(deref(params.inputDesc));
            auto target_tv = get_inner_expanded_tv_3d(deref(params.targetDesc));
            auto weight_tv = get_inner_expanded_tv_1d(deref(params.weightDesc));
            auto output_tv = get_inner_expanded_tv_3d(deref(params.outputDesc));

            kernel(params.input,
                   params.target,
                   params.weight,
                   params.output,
                   params.ignore_index,
                   input_tv,
                   target_tv,
                   weight_tv,
                   output_tv);
        };
    };

    return result;
}

bool NLLLossReduceForwardSolver::IsApplicable(
    const ExecutionContext&, const miopen::nllloss::ReduceProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightLength())
        return false;
    if(!problem.IsRightStride())
        return false;
    return true;
}

bool NLLLossReduceForward4d::IsApplicable(
    const ExecutionContext& context, const miopen::nllloss::ReduceProblemDescription& problem) const
{
    if(!NLLLossReduceForwardSolver::IsApplicable(context, problem))
        return false;
    if(problem.GetInputDesc().GetSize() > 4)
        return false;
    return true;
}

ConvSolution
NLLLossReduceForward4d::GetSolution(const ExecutionContext& context,
                                    const miopen::nllloss::ReduceProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto dtype        = problem.GetOutputDesc().GetType();
    size_t N_total    = problem.GetNtotal();

    auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
                              {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                              {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
                              {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
                              {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
                              {"REDUCE_SIZE", LOCAL_SIZE_REDUCE_FWD}};

    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_NON_CON_FWD},
                                                         {N_total},
                                                         "MIOpenNLLLoss.cpp",
                                                         "NLLLossReduceForward4d",
                                                         build_params));

    auto size = N_total;
    do
    {
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_REDUCE_FWD}, {size}, "MIOpenNLLLoss.cpp", "LossSum", build_params));
        size = (size + LOCAL_SIZE_REDUCE_FWD - 1) / LOCAL_SIZE_REDUCE_FWD;
    } while(size > 1);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::nllloss::InvokeParams>();
            auto elapsed          = 0.f;

            {
                decltype(auto) kernel = handle_.Run(kernels.front());

                auto input_tv  = get_inner_expanded_tv_4d(deref(params.inputDesc));
                auto target_tv = get_inner_expanded_tv_3d(deref(params.targetDesc));
                auto weight_tv = get_inner_expanded_tv_1d(deref(params.weightDesc));

                kernel(params.input,
                       params.target,
                       params.weight,
                       params.workspace,
                       params.ignore_index,
                       params.divisor,
                       input_tv,
                       target_tv,
                       weight_tv);
            }
            if(handle_.IsProfilingEnabled())
            {
                elapsed = handle_.GetKernelTime();
            }

            auto work_a = params.workspace;
            auto work_b =
                static_cast<Data_t>(static_cast<char*>(params.workspace) +
                                    deref(params.targetDesc).GetElementSize() *
                                        get_data_size(deref(params.outputDesc).GetType()));
            auto size = deref(params.targetDesc).GetElementSize();

            for(int i = 1; i < kernels.size(); ++i)
            {
                decltype(auto) kernel = handle_.Run(kernels[i]);
                if(i + 1 != kernels.size())
                {
                    kernel(work_a, work_b, size);
                    std::swap(work_a, work_b);
                }
                else
                    kernel(work_a, params.output, size);

                if(handle_.IsProfilingEnabled())
                    elapsed += handle_.GetKernelTime();
                size = (size + LOCAL_SIZE_REDUCE_FWD - 1) / LOCAL_SIZE_REDUCE_FWD;
            }
            if(handle_.IsProfilingEnabled())
            {
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            };
        };
    };

    return result;
}

std::size_t NLLLossReduceForward4d::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::nllloss::ReduceProblemDescription& problem) const
{
    return (problem.GetTargetDesc().GetElementSize() +
            AlignUp(problem.GetTargetDesc().GetElementSize(), LOCAL_SIZE_REDUCE_FWD) /
                LOCAL_SIZE_REDUCE_FWD) *
           get_data_size(problem.GetOutputDesc().GetType());
}

bool NLLLossUnreduceBackwardSolver::IsApplicable(
    const ExecutionContext&, const miopen::nllloss::UnreduceProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightLength())
        return false;
    if(!problem.IsRightStride())
        return false;
    if(!problem.IsAllContiguous())
        return false;
    return true;
}

bool NLLLossUnreduceBackwardContiguous::IsApplicable(
    const ExecutionContext& context,
    const miopen::nllloss::UnreduceProblemDescription& problem) const
{
    if(!problem.IsAllContiguous())
        return false;
    if(!NLLLossUnreduceBackwardSolver::IsApplicable(context, problem))
        return false;
    return true;
}

ConvSolution NLLLossUnreduceBackwardContiguous::GetSolution(
    const ExecutionContext& context,
    const miopen::nllloss::UnreduceProblemDescription& problem) const
{
    std::ignore = context;

    auto result            = ConvSolution{miopenStatusSuccess};
    auto input_grad_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_grad_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

    {
        auto dtype     = problem.GetInputDesc().GetType();
        size_t N_total = problem.GetNtotal();

        size_t xlocalsize = LOCAL_SIZE_CON_BWD;
        size_t xgridsize  = AlignUp(N_total, xlocalsize);

        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenNLLLoss.cpp";
        kernel.kernel_name = "NLLLossUnreducedBackward4dContiguous";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_grad_dtype == "bfloat16" ? "ushort" : input_grad_dtype},
            {"OUTPUT_TYPE", output_grad_dtype == "bfloat16" ? "ushort" : output_grad_dtype},
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
            decltype(auto) params = raw_params.CastTo<miopen::nllloss::BwdInvokeParams>();

            size_t N_total = params.outputGradDesc->GetElementSize();
            auto dims      = params.inputGradDesc->GetLengths();
            size_t C       = dims[1];
            size_t D1      = dims[2];
            size_t D2      = dims[3];

            kernel(params.input_grad,
                   params.target,
                   params.weight,
                   params.output_grad,
                   params.ignore_index,
                   N_total,
                   C,
                   D1,
                   D2);
        };
    };

    return result;
}

bool NLLLossUnreduceBackward4d::IsApplicable(
    const ExecutionContext& context,
    const miopen::nllloss::UnreduceProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetSize() > 4)
        return false;
    if(!NLLLossUnreduceBackwardSolver::IsApplicable(context, problem))
        return false;
    return true;
}

ConvSolution NLLLossUnreduceBackward4d::GetSolution(
    const ExecutionContext& context,
    const miopen::nllloss::UnreduceProblemDescription& problem) const
{
    std::ignore = context;

    auto result            = ConvSolution{miopenStatusSuccess};
    auto input_grad_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_grad_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

    {
        auto dtype     = problem.GetInputDesc().GetType();
        size_t N_total = problem.GetNtotal();

        size_t xlocalsize = LOCAL_SIZE_NON_CON_BWD;
        size_t xgridsize  = AlignUp(N_total, xlocalsize);

        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenNLLLoss.cpp";
        kernel.kernel_name = "NLLLossUnreducedBackward4d";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_grad_dtype == "bfloat16" ? "ushort" : input_grad_dtype},
            {"OUTPUT_TYPE", output_grad_dtype == "bfloat16" ? "ushort" : output_grad_dtype},
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
            decltype(auto) params = raw_params.CastTo<miopen::nllloss::BwdInvokeParams>();

            auto input_grad_tv  = get_inner_expanded_tv_4d(deref(params.inputGradDesc));
            auto target_tv      = get_inner_expanded_tv_3d(deref(params.targetDesc));
            auto weight_tv      = get_inner_expanded_tv_1d(deref(params.weightDesc));
            auto output_grad_tv = get_inner_expanded_tv_3d(deref(params.outputGradDesc));

            kernel(params.input_grad,
                   params.target,
                   params.weight,
                   params.output_grad,
                   params.ignore_index,
                   input_grad_tv,
                   target_tv,
                   weight_tv,
                   output_grad_tv);
        };
    };

    return result;
}

bool NLLLossReduceBackward4d::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::nllloss::ReduceProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetSize() > 4)
        return false;
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightLength())
        return false;
    if(!problem.IsRightStride())
        return false;
    return true;
}

ConvSolution
NLLLossReduceBackward4d::GetSolution(const ExecutionContext& context,
                                     const miopen::nllloss::ReduceProblemDescription& problem) const
{
    std::ignore = context;

    auto result            = ConvSolution{miopenStatusSuccess};
    auto input_grad_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_grad_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto dtype             = problem.GetOutputDesc().GetType();
    size_t N_total         = problem.GetNtotal();

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_grad_dtype == "bfloat16" ? "ushort" : input_grad_dtype},
        {"OUTPUT_TYPE", output_grad_dtype == "bfloat16" ? "ushort" : output_grad_dtype}};

    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_NON_CON_BWD},
                                                         {N_total},
                                                         "MIOpenNLLLoss.cpp",
                                                         "NLLLossReduceBackward4d",
                                                         build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::nllloss::BwdInvokeParams>();

            auto input_grad_tv  = get_inner_expanded_tv_4d(deref(params.inputGradDesc));
            auto target_grad_tv = get_inner_expanded_tv_3d(deref(params.targetDesc));
            auto weight_grad_tv = get_inner_expanded_tv_1d(deref(params.weightDesc));

            kernel(params.input_grad,
                   params.target,
                   params.weight,
                   params.output_grad,
                   params.ignore_index,
                   params.divisor,
                   input_grad_tv,
                   target_grad_tv,
                   weight_grad_tv);
        };
    };

    return result;
}

} // namespace nllloss

} // namespace solver

} // namespace miopen
