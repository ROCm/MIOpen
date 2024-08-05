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

#include <miopen/sigmoidfocalloss/problem_description.hpp>
#include <miopen/miopen.h>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/sigmoidfocalloss/invoke_params.hpp>
#include <miopen/sigmoidfocalloss/solvers.hpp>
#include <miopen/sigmoid_focal_loss.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/sigmoidfocalloss/utils.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256
#define LOCAL_SIZE_REDUCE_FWD 256

namespace miopen {

namespace solver {

namespace sigmoidfocalloss {

bool SigmoidFocalLossFwd::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::sigmoidfocalloss::SigmoidFocalLossFwdProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetNumDims() > 5)
        return false;
    return true;
}

ConvSolution SigmoidFocalLossFwd::GetSolution(
    const ExecutionContext& context,
    const miopen::sigmoidfocalloss::SigmoidFocalLossFwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto in_dtype     = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto dtype        = problem.GetOutputDesc().GetType();
    auto target_dtype = miopen::GetDataType(problem.GetTargetDesc().GetType());
    auto size         = problem.GetInputDesc().GetElementSize();

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", in_dtype == "bfloat16" ? "ushort" : in_dtype},
        {"TARGET_TYPE", target_dtype == "bfloat16" ? "ushort" : in_dtype},
        {"LOCAL_SIZE", LOCAL_SIZE},
    };

    /* Prepare params for loss kernel */
    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE}, {size}, "MIOpenSigmoidFocalLoss.cpp", "SigmoidFocalLossFwd", build_params));

    /* Prepare params for reduce kernels */
    auto _size = size;
    do
    {
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_REDUCE_FWD}, {_size}, "MIOpenLossSum.cpp", "LossSum", build_params));
        _size = AlignUp(_size, LOCAL_SIZE_REDUCE_FWD) / LOCAL_SIZE_REDUCE_FWD;
    } while(_size > 1);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::sigmoidfocalloss::FwdInvokeParams>();
            auto size             = deref(params.inputDesc).GetElementSize();

            auto elapsed = 0.f;
            HipEventPtr start;
            HipEventPtr stop;

            bool resetProfilingState = false;
            if(handle_.IsProfilingEnabled())
            {
                resetProfilingState = true;
                handle_.EnableProfiling(false);
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            /* Execute loss kernel */
            {
                decltype(auto) kernel = handle_.Run(kernels.front());
                auto input_tv         = get_inner_expanded_tv<5>(deref(params.inputDesc));
                auto target_tv        = get_inner_expanded_tv<5>(deref(params.targetDesc));
                float divisor         = 1;
                if(params.reduction == MIOPEN_LOSS_REDUCTION_MEAN)
                {
                    divisor = size;
                }

                kernel(params.input,
                       params.target,
                       params.workspace,
                       params.alpha,
                       params.gamma,
                       divisor,
                       input_tv,
                       target_tv);
            }

            /* Execute reduce kernels */
            auto reduceIn = params.workspace;
            auto reduceOut =
                static_cast<Data_t>(static_cast<char*>(params.workspace) +
                                    deref(params.inputDesc).GetElementSize() *
                                        get_data_size(deref(params.outputDesc).GetType()));
            for(int i = 1; i < kernels.size(); ++i)
            {
                decltype(auto) kernel = handle_.Run(kernels[i]);
                if(i + 1 != kernels.size())
                {
                    kernel(reduceIn, reduceOut, size);
                    std::swap(reduceIn, reduceOut);
                }
                else
                {
                    kernel(reduceIn, params.output, size);
                }
                size = AlignUp(size, LOCAL_SIZE_REDUCE_FWD) / LOCAL_SIZE_REDUCE_FWD;
            }

            if(resetProfilingState)
            {
                handle_.EnableProfiling(true);
            }

            if(handle_.IsProfilingEnabled())
            {
                hipEventRecord(stop.get(), handle_.GetStream());
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            };
        };
    };

    return result;
}

std::size_t SigmoidFocalLossFwd::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::sigmoidfocalloss::SigmoidFocalLossFwdProblemDescription& problem) const
{
    size_t inputElements  = problem.GetInputDesc().GetElementSize();
    size_t reduceElements = (inputElements + LOCAL_SIZE_REDUCE_FWD - 1) / LOCAL_SIZE_REDUCE_FWD;
    size_t res =
        (inputElements + reduceElements) * get_data_size(problem.GetOutputDesc().GetType());

    return res;
}

} // namespace sigmoidfocalloss

} // namespace solver

} // namespace miopen
