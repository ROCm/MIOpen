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
#include <miopen/roialign.hpp>
#include <miopen/roialign/invoke_params.hpp>
#include <miopen/roialign/problem_description.hpp>
#include <miopen/roialign/solvers.hpp>
#include <miopen/tensor_view_utils.hpp>

#define ROIALIGN_LOCAL_SIZE 256

namespace miopen {
namespace solver {
namespace roialign {

bool IsImprovementOverROCm(const miopen::roialign::BwdProblemDescription& problem)
{
    auto input_grad_dtype = problem.GetInputGradDesc().GetType();
    auto input_numel      = problem.GetInputGradDesc().GetElementSize();

    // input.shape = [N, C, H, W]
    // input_numel smaller than 16^4 gives better performance
    auto is_small_size   = input_numel < (16ULL * 16 * 16 * 16);
    bool is_fp16_improve = (problem.IsAllContiguous() || is_small_size);

    return (input_grad_dtype == miopenBFloat16) ||
           (input_grad_dtype == miopenHalf && is_fp16_improve);
}

bool RoIAlignBackward::IsApplicable(const ExecutionContext& context,
                                    const miopen::roialign::BwdProblemDescription& problem) const
{
    if(!(problem.GetInputGradDesc().GetType() == miopenFloat ||
         problem.GetInputGradDesc().GetType() == miopenHalf ||
         problem.GetInputGradDesc().GetType() == miopenBFloat16))
        return false;

    if(!IsImprovementOverROCm(problem))
        return false;

    return true;
}

ConvSolution
RoIAlignBackward::GetSolution(const ExecutionContext& context,
                              const miopen::roialign::BwdProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype    = problem.GetInputGradDesc().GetType();
    auto io_dtype = miopen::GetDataType(dtype);

    auto output_grad_dims   = problem.GetOutputGradDesc().GetLengths();
    auto rois_lengths       = problem.GetRoisDesc().GetLengths();
    auto input_grad_lengths = problem.GetInputGradDesc().GetLengths();

    auto input_grad_numel = problem.GetInputGradDesc().GetElementSize();

    const auto N = input_grad_lengths[0];
    const auto C = input_grad_lengths[1];
    const auto H = input_grad_lengths[2];
    const auto W = input_grad_lengths[3];

    const auto K = rois_lengths[0];

    const auto OH = problem.GetAlignedHeight();
    const auto OW = problem.GetAlignedWidth();

    // Using atomic roialign enhance performance but lower precision accuracy
    const bool is_use_atomic_roialign_kernel = (io_dtype != "bfloat16");

    // Start building result.construction_params
    if(is_use_atomic_roialign_kernel)
    {
        /* Phrase 1: Fill input_grad with zeros */
        {
            const size_t xlocalsize = ROIALIGN_LOCAL_SIZE;
            size_t xgridsize        = AlignUp(input_grad_numel, xlocalsize);
            size_t ylocalsize       = 1;
            size_t ygridsize        = 1;
            size_t zlocalsize       = 1;
            size_t zgridsize        = 1;

            auto kernel        = KernelInfo{};
            kernel.kernel_file = "MIOpenFill.cpp";
            kernel.kernel_name = "FillZero";

            auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype},
                {"VIEW_DIMS", 4}};

            kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

            kernel.l_wk.push_back(xlocalsize);
            kernel.l_wk.push_back(ylocalsize);
            kernel.l_wk.push_back(zlocalsize);

            kernel.g_wk.push_back(xgridsize);
            kernel.g_wk.push_back(ygridsize);
            kernel.g_wk.push_back(zgridsize);

            result.construction_params.push_back(kernel);
        }

        /* Phrase 2: Run RoIAlign Backward Atomic */
        {
            size_t xlocalsize = ROIALIGN_LOCAL_SIZE;
            size_t xgridsize  = AlignUp(K * C * OH * OW, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel        = KernelInfo{};
            kernel.kernel_file = "MIOpenRoIAlign.cpp";
            kernel.kernel_name = "RoIAlignBackwardAtomic";

            auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
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
        }
    }
    else
    {
        size_t xlocalsize = ROIALIGN_LOCAL_SIZE;
        size_t xgridsize  = AlignUp(C * H * W, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = N;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenRoIAlign.cpp";
        kernel.kernel_name = "RoIAlignBackward";

        auto build_params =
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
    }
    // End building result.construction_params

    // Start building result.invoker_factory
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::roialign::BwdInvokeParams>();

            auto rois_tv        = miopen::get_inner_expanded_tv<2>(deref(params.roisDesc));
            auto output_grad_tv = miopen::get_inner_expanded_tv<4>(deref(params.outputGradDesc));
            auto input_grad_tv  = miopen::get_inner_expanded_tv<4>(deref(params.inputGradDesc));

            HipEventPtr start, stop;
            bool profiling = handle_.IsProfilingEnabled();
            if(profiling)
            {
                handle_.EnableProfiling(false);
                hipStreamSynchronize(handle_.GetStream());
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            /* Phase 1: Fill input grad with zeros */
            if(is_use_atomic_roialign_kernel)
            {

                decltype(auto) kernel = handle_.Run(kernels.front());
                kernel(params.inputGrad, input_grad_numel, input_grad_tv);
            }

            /* Phase 2: Run RoIAlign Backward */
            {
                decltype(auto) kernel = handle_.Run(kernels.back());
                kernel(params.outputGrad,
                       params.rois,
                       params.inputGrad,
                       N,
                       C,
                       H,
                       W,
                       K,
                       OH,
                       OW,
                       params.spatialScale,
                       params.samplingRatio,
                       params.aligned,
                       output_grad_tv,
                       rois_tv,
                       input_grad_tv);
            }

            if(profiling)
            {
                float elapsed = 0.0f;
                hipEventRecord(stop.get(), handle_.GetStream());
                handle_.EnableProfiling(true);
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());

                // Clean up
                hipEventDestroy(start.get());
                hipEventDestroy(stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            };
        };
    };

    return result;
}

} // namespace roialign
} // namespace solver
} // namespace miopen
