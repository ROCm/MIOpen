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

#define LOCAL_SIZE_NORM 256
#define LOCAL_SIZE_REDUCED_SUM 256
#define LOCAL_SIZE_UNREDUCED_BWD 1024

namespace miopen {

namespace solver {

namespace cosineembeddingloss {

namespace {

inline void ConstructNormParamsKernelsBwd(
    const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem,
    ConvSolution& result,
    const KernelBuildParameters& build_params)
{
    auto input_size = problem.GetInput1Desc().GetElementSize();
    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_NORM},
                                                         {input_size},
                                                         "MIOpenCosineEmbeddingLoss.cpp",
                                                         "CosineEmbeddingLossNorm2d",
                                                         build_params));

    auto output_numel = problem.GetInput1Desc().GetLengths()[0] * 3;
    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_REDUCED_SUM},
                                                         {output_numel * LOCAL_SIZE_REDUCED_SUM},
                                                         "MIOpenReduceSum.cpp",
                                                         "Reduce1dSumContiguous",
                                                         build_params));
}

inline void RunNormKernelsBwd(const std::vector<Kernel>& kernels,
                              const Handle& handle_,
                              const AnyInvokeParams& raw_params,
                              int& kernel_cnt,
                              Data_t& work_a,
                              Data_t& work_b)
{
    auto params = raw_params.CastTo<miopen::cosineembeddingloss::BwdInvokeParams>();

    {
        auto I1_tv  = get_inner_expanded_tv<2>(deref(params.input1Desc));
        auto I2_tv  = get_inner_expanded_tv<2>(deref(params.input2Desc));
        auto kernel = handle_.Run(kernels[kernel_cnt++]);

        kernel(params.input1, params.input2, work_a, I1_tv, I2_tv);
    }

    auto reduce_size  = params.input1Desc->GetLengths()[1];
    auto output_numel = params.input1Desc->GetLengths()[0] * 3;

    auto kernel = handle_.Run(kernels[kernel_cnt++]);
    kernel(work_a,
           work_b,
           static_cast<uint64_t>(output_numel),
           static_cast<uint64_t>(reduce_size),
           static_cast<uint64_t>(1));
    std::swap(work_a, work_b);
}

bool IsOverROCm(const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem)
{
    if(!((problem.GetInput1Desc().GetLengths()[0] >= 237 &&
          problem.GetInput1Desc().GetLengths()[1] >= 80) ||
         problem.GetInput1Desc().GetLengths()[1] >= 200))
        return false;
    return true;
}

} // namespace

bool CosineEmbeddingLossUnreducedBackward2d::IsApplicable(
    const ExecutionContext&,
    const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem) const
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

ConvSolution CosineEmbeddingLossUnreducedBackward2d::GetSolution(
    const ExecutionContext&,
    const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem) const
{
    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetInput1Desc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetInput1GradDesc().GetType());

    {
        auto dtype         = problem.GetInput1GradDesc().GetType();
        size_t Input_total = problem.GetInputTotal();

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_REDUCE_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"OUTPUT_REDUCE_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"REDUCE_SIZE", LOCAL_SIZE_REDUCED_SUM},
        };

        ConstructNormParamsKernelsBwd(problem, result, build_params);

        result.construction_params.push_back(
            make_hip_kernel({LOCAL_SIZE_UNREDUCED_BWD},
                            {Input_total},
                            "MIOpenCosineEmbeddingLoss.cpp",
                            "CosineEmbeddingLossUnreducedBackward2d",
                            build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params =
                raw_params.CastTo<miopen::cosineembeddingloss::BwdInvokeParams>();

            float elapsed = 0.0f;
            int kernelCnt = 0;

            auto work_a = params.workspace;
            auto work_b =
                reinterpret_cast<Data_t>(reinterpret_cast<char*>(params.workspace) +
                                         params.input1Desc->GetElementSize() *
                                             get_data_size(params.input1GradDesc->GetType()) * 3);

            HipEventPtr start;
            HipEventPtr stop;

            if(handle_.IsProfilingEnabled())
            {
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            RunNormKernelsBwd(kernels, handle_, raw_params, kernelCnt, work_a, work_b);

            auto input1_tv      = get_inner_expanded_tv<2>(deref(params.input1Desc));
            auto input2_tv      = get_inner_expanded_tv<2>(deref(params.input2Desc));
            auto target_tv      = get_inner_expanded_tv<1>(deref(params.targetDesc));
            auto output_grad_tv = get_inner_expanded_tv<1>(deref(params.outputGradDesc));
            auto input1_grad_tv = get_inner_expanded_tv<2>(deref(params.input1GradDesc));
            auto input2_grad_tv = get_inner_expanded_tv<2>(deref(params.input2GradDesc));

            auto kernel = handle_.Run(kernels[kernelCnt++]);
            kernel(work_a,
                   params.input1,
                   params.input2,
                   params.target,
                   params.output_grad,
                   params.input1_grad,
                   params.input2_grad,
                   params.margin,
                   input1_tv,
                   input2_tv,
                   target_tv,
                   output_grad_tv,
                   input1_grad_tv,
                   input2_grad_tv);

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

std::size_t CosineEmbeddingLossUnreducedBackward2d::GetWorkspaceSize(
    const ExecutionContext&,
    const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem) const
{
    std::size_t size = problem.GetInput1Desc().GetElementSize() *
                       get_data_size(problem.GetInput1GradDesc().GetType()) * 3;

    auto output_numel = problem.GetInput1Desc().GetLengths()[0] * 3;

    size += output_numel * get_data_size(problem.GetInput1GradDesc().GetType());
    return size;
}

} // namespace cosineembeddingloss

} // namespace solver

} // namespace miopen
