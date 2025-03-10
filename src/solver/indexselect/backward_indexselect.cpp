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
#include <miopen/indexselect.hpp>
#include <miopen/indexselect/invoke_params.hpp>
#include <miopen/indexselect/solvers.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace indexselect {

bool IsImprovementOverROCm(const miopen::indexselect::BwdProblemDescription& problem)
{
    auto output_numel = problem.GetOutputGradDesc().GetElementSize();
    return output_numel < 100000;
}

bool IndexSelectBackward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::indexselect::BwdProblemDescription& problem) const
{
    if(problem.GetInputGradDesc().GetType() != miopenFloat &&
       problem.GetInputGradDesc().GetType() != miopenHalf &&
       problem.GetInputGradDesc().GetType() != miopenBFloat16)
        return false;

    if(!IsImprovementOverROCm(problem))
    {
        return false;
    }

    return true;
}

ConvSolution
IndexSelectBackward::GetSolution(const ExecutionContext& /*context*/,
                                 const miopen::indexselect::BwdProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetInputGradDesc().GetType();
    auto io_dtype     = miopen::GetDataType(dtype);
    auto output_numel = problem.GetOutputGradDesc().GetElementSize();
    auto input_numel  = problem.GetInputGradDesc().GetElementSize();

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
                              {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
                              {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype}};

    /* Phase 1: Fill input grad with zeros */
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(input_numel, xlocalsize);

        auto kernel        = KernelInfo();
        kernel.kernel_file = "MIOpenFill.cpp";
        kernel.kernel_name = "FillZero";

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);
    }

    /* Phase 2: IndexSelect backward */
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_numel, xlocalsize);

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenIndexSelect.cpp";
        kernel.kernel_name = "IndexSelectBackward";

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [output_numel, input_numel](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::indexselect::BwdInvokeParams>();

            float elapsed = 0.f;
            HipEventPtr start, stop;

            const bool profiling = handle_.IsProfilingEnabled();
            if(profiling)
            {
                handle_.EnableProfiling(false);
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            /* Phase 1: Fill input grad with zeros. */
            {
                decltype(auto) kernel = handle_.Run(kernels.front());
                kernel(params.inputGrad, input_numel);
            }

            /* Phase 2: IndexSelect backward. */
            {
                auto output_grad_tv = get_inner_expanded_tv<5>(deref(params.outputGradDesc));
                auto indices_tv     = get_inner_expanded_tv<1>(deref(params.indicesDesc));
                auto inGrad_tv      = get_inner_expanded_tv<5>(deref(params.inputGradDesc));

                decltype(auto) kernel = handle_.Run(kernels.back());
                kernel(params.outputGrad,
                       params.indices,
                       params.inputGrad,
                       params.dim,
                       output_numel,
                       output_grad_tv,
                       indices_tv,
                       inGrad_tv);
            }

            if(profiling)
            {
                hipEventRecord(stop.get(), handle_.GetStream());
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());

                // Clean up
                hipEventDestroy(start.get());
                hipEventDestroy(stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);

                handle_.EnableProfiling(true);
            }
        };
    };

    return result;
}

} // namespace indexselect

} // namespace solver

} // namespace miopen
