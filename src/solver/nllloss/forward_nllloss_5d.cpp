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
#include <miopen/nllloss/solvers.hpp>

#include <miopen/nllloss/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/nllloss.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/nllloss/utils.hpp>

#define LOCAL_SIZE_NON_CON_FWD 1024
#define LOCAL_SIZE_REDUCE_FWD 256

namespace miopen {

namespace solver {

namespace nllloss {

bool NLLLossReduceForward5d::IsApplicable(const ExecutionContext& context,
                                          const miopen::nllloss::ProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetNumDims() > 5)
        return false;
    if(problem.GetReduction() == MIOPEN_LOSS_REDUCTION_NONE)
        return false;
    if(!NLLLossSolver::IsApplicable(context, problem))
        return false;
    return true;
}

ConvSolution
NLLLossReduceForward5d::GetSolution(const ExecutionContext& context,
                                    const miopen::nllloss::ProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto dtype        = problem.GetOutputDesc().GetType();
    size_t N_total    = problem.GetNtotal();

    auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
                              {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                              {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
                              {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
                              {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
                              {"REDUCE_SIZE", LOCAL_SIZE_REDUCE_FWD}};

    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_NON_CON_FWD},
                                                         {N_total},
                                                         "MIOpenNLLLoss.cpp",
                                                         "NLLLossForward5d",
                                                         build_params));

    auto size = N_total;
    do
    {
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_REDUCE_FWD}, {size}, "MIOpenNLLLoss.cpp", "LossSum", build_params));
        size = (size + LOCAL_SIZE_REDUCE_FWD - 1) / LOCAL_SIZE_REDUCE_FWD;
    } while(size > 1);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::nllloss::FwdInvokeParams>();

            decltype(auto) kernel = handle_.Run(kernels.front());

            auto input_tv  = get_inner_expanded_tv<5>(deref(params.inputDesc));
            auto target_tv = get_inner_expanded_tv<4>(deref(params.targetDesc));
            auto weight_tv = get_inner_expanded_tv<1>(deref(params.weightDesc));

            auto work_a = params.workspace;
            auto work_b =
                static_cast<Data_t>(static_cast<char*>(params.workspace) +
                                    deref(params.targetDesc).GetElementSize() *
                                        get_data_size(deref(params.outputDesc).GetType()));
            auto size = deref(params.targetDesc).GetElementSize();

            auto elapsed = 0.f;
            HipEventPtr start, stop;
            const bool profiling = handle_.IsProfilingEnabled();
            if(profiling)
            {
                start = make_hip_event();
                stop  = make_hip_event();
                handle_.EnableProfiling(false);
                hipEventRecord(start.get(), handle_.GetStream());
            }
            float divisor = 1;
            if(params.reduction == MIOPEN_LOSS_REDUCTION_MEAN)
            {
                divisor = static_cast<float>(deref(params.targetDesc).GetElementSize());
            }

            kernel(params.input,
                   params.target,
                   params.weight,
                   params.workspace,
                   params.ignore_index,
                   divisor,
                   input_tv,
                   target_tv,
                   weight_tv);

            for(int i = 1; i < kernels.size(); ++i)
            {
                kernel = handle_.Run(kernels[i]);
                if(i + 1 != kernels.size())
                {
                    kernel(work_a, work_b, size);
                    std::swap(work_a, work_b);
                }
                else
                    kernel(work_a, params.output, size);

                size = (size + LOCAL_SIZE_REDUCE_FWD - 1) / LOCAL_SIZE_REDUCE_FWD;
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

std::size_t
NLLLossReduceForward5d::GetWorkspaceSize(const ExecutionContext& /*context*/,
                                         const miopen::nllloss::ProblemDescription& problem) const
{
    return (problem.GetTargetDesc().GetElementSize() +
            AlignUp(problem.GetTargetDesc().GetElementSize(), LOCAL_SIZE_REDUCE_FWD) /
                LOCAL_SIZE_REDUCE_FWD) *
           get_data_size(problem.GetOutputDesc().GetType());
}

} // namespace nllloss

} // namespace solver

} // namespace miopen
