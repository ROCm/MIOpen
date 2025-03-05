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

#include <miopen/datatype.hpp>
#include <miopen/errors.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/maskedfill/invoke_params.hpp>
#include <miopen/maskedfill/problem_description.hpp>
#include <miopen/maskedfill/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

#include <cstddef>
#include <vector>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace maskedfill {

bool MaskedFillBackward::IsImprovementOverROCm(
    const ExecutionContext& /*context*/,
    const miopen::maskedfill::BwdProblemDescription& problem) const
{
    if(problem.IsAllContiguous())
    {
        return false;
    }

    auto output_grad_numel = problem.GetOutputGradDesc().GetElementSize();
    if(output_grad_numel > 2000)
    {
        return false;
    }

    return true;
}

bool MaskedFillBackward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::maskedfill::BwdProblemDescription& problem) const
{
    if(!(problem.GetInputGradDesc().GetType() == miopenFloat ||
         problem.GetInputGradDesc().GetType() == miopenHalf ||
         problem.GetInputGradDesc().GetType() == miopenBFloat16))
    {
        return false;
    }

    return true;
}

ConvSolution
MaskedFillBackward::GetSolution(const ExecutionContext& context,
                                const miopen::maskedfill::BwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype             = problem.GetInputGradDesc().GetType();
    auto io_dtype          = miopen::GetDataType(dtype);
    auto output_grad_numel = problem.GetOutputGradDesc().GetElementSize();

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(output_grad_numel, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenMaskedFill.cpp";
    kernel.kernel_name = "MaskedFillBackward";

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                              {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);
    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [output_grad_numel](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::maskedfill::BwdInvokeParams>();

            tensor_view_t<5> output_grad_tv =
                get_inner_expanded_tv<5>(miopen::deref(params.outputGradDesc));
            tensor_view_t<5> mask_tv = get_inner_expanded_tv<5>(miopen::deref(params.maskDesc));
            tensor_view_t<5> input_grad_tv =
                get_inner_expanded_tv<5>(miopen::deref(params.inputGradDesc));

            kernel(params.outputGrad,
                   params.mask,
                   params.inputGrad,
                   output_grad_tv,
                   mask_tv,
                   input_grad_tv,
                   output_grad_numel);
        };
    };

    return result;
}

} // namespace maskedfill

} // namespace solver

} // namespace miopen
