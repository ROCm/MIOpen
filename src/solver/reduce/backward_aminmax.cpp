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
#include <miopen/mlo_internal.hpp>
#include <miopen/reduce/invoke_params.hpp>
#include <miopen/reduce/solvers.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace reduce {

bool AMinMaxBackward::IsApplicable(
    const ExecutionContext&,
    const miopen::reduce::ProblemDescriptionExtremeAminmaxBackward& problem) const
{
    if(problem.GetXDesc().GetType() != miopenHalf && problem.GetXDesc().GetType() != miopenFloat &&
       problem.GetXDesc().GetType() != miopenBFloat16)
        return false;
    return true;
}

ConvSolution AMinMaxBackward::GetSolution(
    const ExecutionContext&,
    const miopen::reduce::ProblemDescriptionExtremeAminmaxBackward& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetXDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetXDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetYDesc().GetType());
    auto indice_dtype = miopen::GetDataType(problem.GetIndiceDesc().GetType());
    auto xdims        = problem.GetXDesc().GetLengths();
    auto ydims        = problem.GetYDesc().GetLengths();
    auto input_numel  = problem.GetXDesc().GetElementSize();

    {
        size_t xlocalsize;
        size_t xgridsize;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenReduceExtreme.cpp";
        kernel.kernel_name = "AminmaxBwd";
        xlocalsize         = LOCAL_SIZE;
        xgridsize          = AlignUp(input_numel, LOCAL_SIZE);

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"INDICE_TYPE", indice_dtype},
            {"OP_TYPE", "ReduceExtremeOp_t::Max"},
            {"MIOPEN_REDUCE_EXTREME_ARGMIN", MIOPEN_REDUCE_EXTREME_ARGMIN},
            {"MIOPEN_REDUCE_EXTREME_ARGMAX", MIOPEN_REDUCE_EXTREME_ARGMAX},
            {"MIOPEN_REDUCE_EXTREME_MIN", MIOPEN_REDUCE_EXTREME_MIN},
            {"MIOPEN_REDUCE_EXTREME_MAX", MIOPEN_REDUCE_EXTREME_MAX}};

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [input_numel](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::reduce::ExtremeAminmaxBackwardInvokeParams>();

            auto input_tv       = get_inner_expanded_tv<5>(deref(params.xDesc));
            auto input_grad_tv  = get_inner_expanded_tv<5>(deref(params.xGradDesc));
            auto output_tv      = get_inner_expanded_tv<5>(deref(params.yDesc));
            auto output_grad_tv = get_inner_expanded_tv<5>(deref(params.yGradDesc));
            auto count_tv       = get_inner_expanded_tv<5>(deref(params.indiceDesc));

            kernel(params.x,
                   params.x_grad,
                   params.y,
                   params.y_grad,
                   params.indice,
                   input_numel,
                   params.dim,
                   input_tv,
                   input_grad_tv,
                   output_tv,
                   output_grad_tv,
                   count_tv);
        };
    };

    return result;
}

} // namespace reduce

} // namespace solver

} // namespace miopen
