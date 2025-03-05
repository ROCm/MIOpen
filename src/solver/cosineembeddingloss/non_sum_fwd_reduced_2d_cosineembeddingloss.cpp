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
#include <miopen/kernel_build_params.hpp>
#include <miopen/cosineembeddingloss.hpp>
#include <miopen/cosineembeddingloss/invoke_params.hpp>
#include <miopen/cosineembeddingloss/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE_FWD_NON_SUM 1024
#define LOCAL_SIZE_REDUCED_NON_SUM 256

namespace miopen {

namespace solver {

namespace cosineembeddingloss {

namespace {

bool IsOverROCm(const miopen::cosineembeddingloss::FwdReducedProblemDescription& problem)
{
    if((problem.GetInput1Desc().GetLengths()[0] >= 768 &&
        problem.GetInput1Desc().GetLengths()[1] >= 128) ||
       problem.GetInput1Desc().GetLengths()[1] >= 2000)
        return false;

    return true;
}

} // namespace

bool CosineEmbeddingLossReducedForward2dNonSum::IsApplicable(
    const ExecutionContext&,
    const miopen::cosineembeddingloss::FwdReducedProblemDescription& problem) const
{
    if(!IsOverROCm(problem))
        return false;

    if(!(problem.GetOutputDesc().GetType() == miopenHalf ||
         problem.GetOutputDesc().GetType() == miopenFloat ||
         problem.GetOutputDesc().GetType() == miopenBFloat16))
    {
        return false;
    }

    return true;
}

ConvSolution CosineEmbeddingLossReducedForward2dNonSum::GetSolution(
    const ExecutionContext& context,
    const miopen::cosineembeddingloss::FwdReducedProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetInput1Desc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

    auto dtype     = problem.GetOutputDesc().GetType();
    size_t N_total = problem.GetNtotal();

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"REDUCE_SIZE", LOCAL_SIZE_REDUCED_NON_SUM},
    };

    result.construction_params.push_back(
        make_hip_kernel({LOCAL_SIZE_FWD_NON_SUM},
                        {N_total},
                        "MIOpenCosineEmbeddingLoss.cpp",
                        "CosineEmbeddingLossReducedForward2d_nonSum",
                        build_params));

    auto size = N_total;
    do
    {
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_REDUCED_NON_SUM}, {size}, "MIOpenLossSum.cpp", "LossSum", build_params));
        size = (size + LOCAL_SIZE_REDUCED_NON_SUM - 1) / LOCAL_SIZE_REDUCED_NON_SUM;
    } while(size > 1);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params =
                raw_params.CastTo<miopen::cosineembeddingloss::FwdInvokeParams>();
            auto elapsed = 0.f;

            HipEventPtr start;
            HipEventPtr stop;

            if(handle_.IsProfilingEnabled())
            {
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            {
                decltype(auto) kernel = handle_.Run(kernels.front());

                auto input1_tv = get_inner_expanded_tv<2>(deref(params.input1Desc));
                auto input2_tv = get_inner_expanded_tv<2>(deref(params.input2Desc));
                auto target_tv = get_inner_expanded_tv<1>(deref(params.targetDesc));
                float divisor  = 1;
                if(params.reduction == MIOPEN_LOSS_REDUCTION_MEAN)
                {
                    divisor *= deref(params.targetDesc).GetElementSize();
                }

                kernel(params.input1,
                       params.input2,
                       params.target,
                       params.workspace,
                       params.margin,
                       divisor,
                       input1_tv,
                       input2_tv,
                       target_tv);
            }

            auto work_a = params.workspace;
            auto work_b =
                static_cast<Data_t>(static_cast<char*>(params.workspace) +
                                    deref(params.targetDesc).GetElementSize() *
                                        get_data_size(deref(params.outputDesc).GetType()));
            auto size = deref(params.targetDesc).GetElementSize();

            for(int i = 1; i < kernels.size(); ++i)
            {
                decltype(auto) kernel = handle_.Run(kernels[i]);
                if(i + 1 != kernels.size())
                {
                    kernel(work_a, work_b, size);
                    std::swap(work_a, work_b);
                }
                else
                    kernel(work_a, params.output, size);

                size = (size + LOCAL_SIZE_REDUCED_NON_SUM - 1) / LOCAL_SIZE_REDUCED_NON_SUM;
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

std::size_t CosineEmbeddingLossReducedForward2dNonSum::GetWorkspaceSize(
    const ExecutionContext&,
    const miopen::cosineembeddingloss::FwdReducedProblemDescription& problem) const
{
    if(problem.GetTargetDesc().GetElementSize() <= LOCAL_SIZE_REDUCED_NON_SUM)
        return problem.GetTargetDesc().GetElementSize() *
               get_data_size(problem.GetOutputDesc().GetType());
    return (problem.GetTargetDesc().GetElementSize() +
            AlignUp(problem.GetTargetDesc().GetElementSize(), LOCAL_SIZE_REDUCED_NON_SUM) /
                LOCAL_SIZE_REDUCED_NON_SUM) *
           get_data_size(problem.GetOutputDesc().GetType());
}

} // namespace cosineembeddingloss

} // namespace solver

} // namespace miopen
