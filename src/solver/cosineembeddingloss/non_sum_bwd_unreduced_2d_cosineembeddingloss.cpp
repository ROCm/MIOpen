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

#define LOCAL_SIZE_UNREDUCED_BWD 1024

namespace miopen {

namespace solver {

namespace cosineembeddingloss {

namespace {

bool IsOverROCm(const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem)
{
    if((problem.GetInput1Desc().GetLengths()[0] >= 237 &&
        problem.GetInput1Desc().GetLengths()[1] >= 80) ||
       problem.GetInput1Desc().GetLengths()[1] >= 200)
        return false;

    return true;
}

} // namespace

bool CosineEmbeddingLossUnreducedBackward2dNonSum::IsApplicable(
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

ConvSolution CosineEmbeddingLossUnreducedBackward2dNonSum::GetSolution(
    const ExecutionContext&,
    const miopen::cosineembeddingloss::BwdUnreducedProblemDescription& problem) const
{
    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetInput1Desc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

    {
        auto dtype     = problem.GetOutputDesc().GetType();
        size_t N_total = problem.GetNtotal();

        auto kernel = KernelInfo{};

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        };

        result.construction_params.push_back(
            make_hip_kernel({LOCAL_SIZE_UNREDUCED_BWD},
                            {N_total},
                            "MIOpenCosineEmbeddingLoss.cpp",
                            "CosineEmbeddingLossUnreducedBackward2d_nonSum",
                            build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::cosineembeddingloss::BwdInvokeParams>();

            auto input1_tv      = get_inner_expanded_tv<2>(deref(params.input1Desc));
            auto input2_tv      = get_inner_expanded_tv<2>(deref(params.input2Desc));
            auto target_tv      = get_inner_expanded_tv<1>(deref(params.targetDesc));
            auto output_grad_tv = get_inner_expanded_tv<1>(deref(params.outputGradDesc));
            auto input1_grad_tv = get_inner_expanded_tv<2>(deref(params.input1GradDesc));
            auto input2_grad_tv = get_inner_expanded_tv<2>(deref(params.input2GradDesc));

            kernel(params.input1,
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
        };
    };

    return result;
}

} // namespace cosineembeddingloss

} // namespace solver

} // namespace miopen
