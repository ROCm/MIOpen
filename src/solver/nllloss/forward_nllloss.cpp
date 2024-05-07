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

namespace miopen {

namespace solver {

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
    if(!problem.IsAllPacked())
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

        size_t xlocalsize = LOCAL_SIZE_CON_FWD;
        size_t xgridsize  = AlignUp(N_total, xlocalsize);

        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenNLLLoss.cpp";
        kernel.kernel_name = "NLLLossUnreducedForward4dContiguous";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
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

        size_t xlocalsize = LOCAL_SIZE_NON_CON_FWD;
        size_t xgridsize  = AlignUp(N_total, xlocalsize);

        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenNLLLoss.cpp";
        kernel.kernel_name = "NLLLossUnreducedForward4d";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
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

bool NLLLossUnreduceBackwardSolver::IsApplicable(
    const ExecutionContext&, const miopen::nllloss::UnreduceProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightLength())
        return false;
    if(!problem.IsRightStride())
        return false;
    if(!problem.IsAllPacked())
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

} // namespace nllloss

} // namespace solver

} // namespace miopen
