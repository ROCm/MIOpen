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
#include "miopen/miopen.h"
#include <miopen/interpolate/solvers.hpp>
#include <miopen/interpolate/utils.hpp>

#include <miopen/interpolate/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/interpolate.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE_BWD_BICUBIC 256

namespace miopen {

namespace solver {

namespace interpolate {

bool IsOverRocmBicubicBwd(const miopen::interpolate::BwdProblemDescription& problem)
{
    TensorDescriptor output_grad_desc = problem.GetOutputGradDesc();
    TensorDescriptor input_grad_desc  = problem.GetInputGradDesc();
    auto dtype                        = input_grad_desc.GetType();

    float scale_h =
        static_cast<float>(output_grad_desc.GetLengths()[2]) / input_grad_desc.GetLengths()[2];
    float scale_w =
        static_cast<float>(output_grad_desc.GetLengths()[3]) / input_grad_desc.GetLengths()[3];

    if(dtype == miopenHalf || dtype == miopenBFloat16)
    {
        if(scale_h * scale_w < 16 && scale_h * scale_w > 0.5)
            return true;
    }
    else
    {
        return true;
    }

    return true;
    // return false;
}

bool InterpolateBicubicBackward::IsApplicable(
    const ExecutionContext&, const miopen::interpolate::BwdProblemDescription& problem) const
{
    if(problem.GetMode() != miopenInterpolateMode_t::MIOPEN_INTERPOLATE_MODE_BICUBIC)
        return false;
    if(!IsOverRocmBicubicBwd(problem))
        return false;

    return true;
}

ConvSolution InterpolateBicubicBackward::GetSolution(
    const ExecutionContext& context,
    const miopen::interpolate::BwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetOutputGradDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetInputGradDesc().GetType());

    {
        auto dtype           = problem.GetInputGradDesc().GetType();
        size_t N_total       = problem.GetOutputGradDesc().GetElementSize();
        size_t N_total_paste = problem.GetInputGradDesc().GetElementSize();

        auto kernel = KernelInfo{};

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"DTYPE", "float"},
        };

        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_BWD_BICUBIC},
                                                             {N_total},
                                                             "MIOpenInterpolate.cpp",
                                                             "InterpolateBicubicBackward",
                                                             build_params));

        if(dtype != miopenFloat)
        {
            result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_BWD_BICUBIC},
                                                                 {N_total_paste},
                                                                 "MIOpenInterpolate.cpp",
                                                                 "InterpolateBicubicBackward_paste",
                                                                 build_params));
        }
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::interpolate::BwdInvokeParams>();

            auto input_grad_tv  = get_inner_expanded_tv<4>(deref(params.inputGradDesc));
            auto output_grad_tv = get_inner_expanded_tv<4>(deref(params.outputGradDesc));
            auto dtype          = deref(params.inputGradDesc).GetType();
            size_t nelems       = params.outputGradDesc->GetElementSize();

            int kernelCnt         = 0;
            decltype(auto) kernel = handle_.Run(kernels[kernelCnt++]);

            float elapsed = 0.0f;
            HipEventPtr start;
            HipEventPtr stop;

            bool reset_profiling_state = false;
            if(handle_.IsProfilingEnabled())
            {
                reset_profiling_state = true;
                handle_.EnableProfiling(false);
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            if(dtype == miopenFloat)
            {
                kernel(params.input_grad,
                       params.output_grad,
                       input_grad_tv,
                       output_grad_tv,
                       nelems,
                       params.scale_factors,
                       params.align_corners);
            }
            else
            {
                kernel(params.workspace,
                       params.output_grad,
                       input_grad_tv,
                       output_grad_tv,
                       nelems,
                       params.scale_factors,
                       params.align_corners);

                nelems = params.inputGradDesc->GetElementSize();
                kernel = handle_.Run(kernels[kernelCnt++]);
                kernel(params.input_grad, params.workspace, input_grad_tv, nelems);
            }

            if(reset_profiling_state)
            {
                handle_.EnableProfiling(true);
            }
            if(handle_.IsProfilingEnabled())
            {
                hipEventRecord(stop.get(), handle_.GetStream());
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());
                hipEventDestroy(start.get());
                hipEventDestroy(stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            };
        };
    };

    return result;
}

std::size_t InterpolateBicubicBackward::GetWorkspaceSize(
    const ExecutionContext&, const miopen::interpolate::BwdProblemDescription& problem) const
{
    return problem.GetInputGradDesc().GetElementSize() * sizeof(float);
}

} // namespace interpolate

} // namespace solver

} // namespace miopen
