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
#include <miopen/errors.hpp>
#include <miopen/image_transform.hpp>
#include <miopen/image_transform/adjust_hue/invoke_params.hpp>
#include <miopen/image_transform/adjust_hue/problem_description.hpp>
#include <miopen/image_transform/solvers.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <vector>

namespace miopen {

namespace solver {

namespace image_transform {

namespace adjust_hue {

bool ImageAdjustHue::IsApplicable(
    const ExecutionContext& /* context */,
    const miopen::image_transform::adjust_hue::ProblemDescription& problem) const
{
    if(!(problem.GetInputTensorDesc().GetType() == miopenFloat ||
         problem.GetInputTensorDesc().GetType() == miopenHalf ||
         problem.GetInputTensorDesc().GetType() == miopenBFloat16))
    {
        return false;
    }

    return true;
}

ConvSolution ImageAdjustHue::GetSolution(
    const ExecutionContext& /* context */,
    const miopen::image_transform::adjust_hue::ProblemDescription& problem) const
{
    auto result = ConvSolution(miopenStatusSuccess);

    auto dtype    = problem.GetInputTensorDesc().GetType();
    auto io_dtype = miopen::GetDataType(dtype);
    auto numel    = problem.GetInputTensorDesc().GetElementSize();

    auto input_size = numel / 3; // RGB image

    size_t xlocalsize = 256;
    size_t xgridsize  = AlignUp(input_size, xlocalsize);

    size_t ylocalsize = 1;
    size_t ygridsize  = 1;

    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenImageAdjustHue.cpp";
    auto is_all_cont   = problem.IsAllContiguous();
    if(is_all_cont)
        kernel.kernel_name = "ImageAdjustHueContiguous";
    else
        kernel.kernel_name = "ImageAdjustHue";

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
                              {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
                              {"DTYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [is_all_cont](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::image_transform::adjust_hue::InvokeParams>();

            auto inputDesc  = miopen::deref(params.inputTensorDesc);
            auto outputDesc = miopen::deref(params.outputTensorDesc);

            auto input_tv  = get_inner_expanded_tv<4>(inputDesc);
            auto output_tv = get_inner_expanded_tv<4>(outputDesc);

            size_t N        = inputDesc.GetElementSize() / 3;
            size_t c_stride = inputDesc.GetLengths()[2] * inputDesc.GetLengths()[3];

            if(is_all_cont)
            {
                kernel(params.input, params.output, params.hue, N, c_stride);
            }
            else
                kernel(params.input, params.output, params.hue, N, input_tv, output_tv);
        };
    };

    return result;
}

} // namespace adjust_hue

} // namespace image_transform

} // namespace solver

} // namespace miopen
