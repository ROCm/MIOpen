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
#include <miopen/hipoc_kernel.hpp>
#include <miopen/image_transform/adjust_saturation/invoke_params.hpp>
#include <miopen/image_transform/adjust_saturation/problem_description.hpp>
#include <miopen/image_transform/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/tensor_view_utils.hpp>

namespace miopen {

namespace solver {

namespace image_transform {

namespace adjust_saturation {

bool ImageAdjustSaturation::IsImprovementOverROCm(
    const ExecutionContext& /*context*/,
    const miopen::image_transform::adjust_saturation::ProblemDescription& problem) const
{
    if(problem.GetInputTensorDesc().IsContiguous() && problem.GetOutputTensorDesc().IsContiguous())
        return true;

    return false;
}

bool ImageAdjustSaturation::IsApplicable(
    const ExecutionContext& context,
    const miopen::image_transform::adjust_saturation::ProblemDescription& problem) const
{
    if(!(problem.GetInputTensorDesc().GetType() == miopenFloat ||
         problem.GetInputTensorDesc().GetType() == miopenHalf ||
         problem.GetInputTensorDesc().GetType() == miopenBFloat16))
    {
        return false;
    }

    if(!IsImprovementOverROCm(context, problem))
        return false;

    return true;
}

size_t ImageAdjustSaturation::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::image_transform::adjust_saturation::ProblemDescription& problem) const
{
    auto dtype   = problem.GetInputTensorDesc().GetType();
    auto numel   = problem.GetInputTensorDesc().GetElementSize();
    auto el_size = get_data_size(dtype);

    auto ws_size = numel * el_size;
    return ws_size;
}

ConvSolution ImageAdjustSaturation::GetSolution(
    const ExecutionContext& /*context*/,
    const miopen::image_transform::adjust_saturation::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype    = problem.GetInputTensorDesc().GetType();
    auto io_dtype = GetDataType(dtype);
    auto numel    = problem.GetInputTensorDesc().GetElementSize();

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
                              {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
                              {"DTYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype}};

    {
        size_t xlocalsize = 256;
        size_t xgridsize  = AlignUp(numel / 3, xlocalsize);

        // first, RGB to Grayscale
        auto rgb_kernel        = KernelInfo{};
        rgb_kernel.kernel_file = "MIOpenImageBlend.cpp";
        rgb_kernel.kernel_name = "RGBToGrayscale";

        rgb_kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        rgb_kernel.l_wk.push_back(xlocalsize);
        rgb_kernel.l_wk.push_back(1);
        rgb_kernel.l_wk.push_back(1);
        rgb_kernel.g_wk.push_back(xgridsize);
        rgb_kernel.g_wk.push_back(1);
        rgb_kernel.g_wk.push_back(1);

        result.construction_params.push_back(rgb_kernel);
    }

    {
        auto blend_kernel        = KernelInfo{};
        blend_kernel.kernel_file = "MIOpenImageBlend.cpp";
        blend_kernel.kernel_name = "BlendContiguous";

        size_t xlocalsize = 256;
        size_t xgridsize  = AlignUp(numel, xlocalsize);

        blend_kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        blend_kernel.l_wk.push_back(xlocalsize);
        blend_kernel.l_wk.push_back(1);
        blend_kernel.l_wk.push_back(1);
        blend_kernel.g_wk.push_back(xgridsize);
        blend_kernel.g_wk.push_back(1);
        blend_kernel.g_wk.push_back(1);

        result.construction_params.push_back(blend_kernel);
    }

    result.invoker_factory = [numel](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& raw_params) {
            decltype(auto) params =
                raw_params.CastTo<miopen::image_transform::adjust_saturation::InvokeParams>();

            auto elapsed = 0.0f;
            HipEventPtr start, stop;

            const bool profiling = handle.IsProfilingEnabled();
            if(profiling)
            {
                handle.EnableProfiling(false);
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle.GetStream());
            }

            /* Phase 1: RGB to Grayscale */
            {
                auto input_desc = miopen::deref(params.inputTensorDesc);
                auto input_tv   = get_inner_expanded_tv<4>(input_desc);
                auto gray_tv    = get_inner_expanded_tv<4>(input_desc);

                auto kernel = handle.Run(kernels[0]);
                kernel(params.input, params.workspace, numel / 3, input_tv, gray_tv);
            }

            /* Phase 2: Blend */
            {
                auto input_desc = miopen::deref(params.inputTensorDesc);
                auto input_tv   = get_inner_expanded_tv<4>(input_desc);
                auto c_stride   = input_tv.size[2] * input_tv.size[3];
                auto n_stride   = c_stride * input_tv.size[1];
                float bound     = 1.0f;

                auto kernel = handle.Run(kernels[1]);
                kernel(params.input,
                       params.workspace,
                       params.output,
                       n_stride,
                       c_stride,
                       numel,
                       params.saturation_factor,
                       bound);
            }

            if(profiling)
            {
                hipEventRecord(stop.get(), handle.GetStream());
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());

                // Clean up
                hipEventDestroy(start.get());
                hipEventDestroy(stop.get());
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);

                handle.EnableProfiling(true);
            }
        };
    };

    return result;
}

} // namespace adjust_saturation

} // namespace image_transform

} // namespace solver

} // namespace miopen
