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

#include <miopen/conv_solution.hpp>
#include <miopen/datatype.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/pad_reflection.hpp>
#include <miopen/pad_reflection/invoke_params.hpp>
#include <miopen/pad_reflection/problem_description.hpp>
#include <miopen/pad_reflection/solvers.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {
namespace solver {
namespace pad_reflection {

namespace {

bool IsImprovementOverROCm(
    const miopen::pad_reflection::PadReflectionFwdProblemDescription& problem)
{
    return !problem.IsContiguous();
}

bool IsImprovementOverROCm(
    const miopen::pad_reflection::PadReflectionBwdProblemDescription& problem)
{
    auto dtype             = problem.GetdXDesc().GetType();
    bool is_fp32           = dtype != miopenHalf && dtype != miopenBFloat16;
    bool is_small_last_dim = problem.GetdXDesc().GetLengths().back() <= 64;
    return is_fp32 && is_small_last_dim;
}

} // namespace

bool PadReflectionFwd::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::pad_reflection::PadReflectionFwdProblemDescription& problem) const
{
    if(!IsImprovementOverROCm(problem))
        return false;

    return true;
}

ConvSolution PadReflectionFwd::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::pad_reflection::PadReflectionFwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype    = problem.GetXDesc().GetType();
    auto io_dtype = miopen::GetDataType(dtype);

    auto output_numel = problem.GetYDesc().GetElementSize();

    {
        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenPadReflection.cpp";
        kernel.kernel_name = "PadReflection1dFwd";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;
        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::pad_reflection::FwdInvokeParams>();
            decltype(auto) kernel = handle_.Run(kernels[0]);

            auto input_tv  = get_inner_expanded_tv<3>(deref(params.xDesc));
            auto output_tv = get_inner_expanded_tv<3>(deref(params.yDesc));

            kernel(params.x, params.y, params.padding[0], output_numel, input_tv, output_tv);
        };
    };

    return result;
}

bool PadReflectionBwd::IsApplicable(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::pad_reflection::PadReflectionBwdProblemDescription& problem) const
{
    if(problem.GetdXDesc().GetType() == miopenBFloat16)
        return false;

    if(!IsImprovementOverROCm(problem))
        return false;

    return true;
}

ConvSolution PadReflectionBwd::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::pad_reflection::PadReflectionBwdProblemDescription& problem) const
{

    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype    = problem.GetdXDesc().GetType();
    auto io_dtype = miopen::GetDataType(dtype);

    auto output_numel = problem.GetdYDesc().GetElementSize();

    {
        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenPadReflection.cpp";
        kernel.kernel_name = "PadReflection1dBwd";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;
        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::pad_reflection::BwdInvokeParams>();

            /* Phase 1: Fill input grad with zeros */
            {
                auto input_grad_numel = params.dxDesc->GetElementSize();
                auto in_size_in_bytes = input_grad_numel * GetTypeSize(dtype);
                hipMemsetAsync(params.dx, 0, in_size_in_bytes, handle_.GetStream());
            }

            /* Phase 2: Calculate output pad reflection */
            {
                decltype(auto) kernel = handle_.Run(kernels[0]);

                auto input_grad_tv  = get_inner_expanded_tv<3>(deref(params.dxDesc));
                auto output_grad_tv = get_inner_expanded_tv<3>(deref(params.dyDesc));

                auto padding_l = params.padding[0];

                kernel(
                    params.dx, params.dy, padding_l, output_numel, input_grad_tv, output_grad_tv);
            }
        };
    };

    return result;
}

} // namespace pad_reflection
} // namespace solver
} // namespace miopen
