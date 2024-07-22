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

#include "miopen/activ.hpp"
#include "miopen/conv_solution.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/invoke_params.hpp"
#include <miopen/interpolate/solvers.hpp>
#include <miopen/interpolate/utils.hpp>

#include <miopen/interpolate/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/interpolate.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE_BWD_NEAREST 256

namespace miopen {

namespace solver {

namespace interpolate {

bool IsOverRocmNearestBwd(const miopen::interpolate::BwdProblemDescription& problem)
{
    TensorDescriptor input_grad_desc  = problem.GetInputGradDesc();
    TensorDescriptor output_grad_desc = problem.GetOutputGradDesc();
    if(input_grad_desc.GetLengths().size() == 3)
    {
        if(output_grad_desc.GetElementSize() < 8000 || input_grad_desc.GetLengths()[0] < 10)
            return false;
    }
    else if(input_grad_desc.GetLengths().size() == 4)
    {
        float scale_h =
            static_cast<float>(output_grad_desc.GetLengths()[2]) / input_grad_desc.GetLengths()[2];
        float scale_w =
            static_cast<float>(output_grad_desc.GetLengths()[3]) / input_grad_desc.GetLengths()[3];

        if(input_grad_desc.GetLengths()[0] < 10 || (scale_h + scale_w <= 4))
            return false;
    }
    else if(input_grad_desc.GetLengths().size() == 5)
    {
        float scale_h =
            static_cast<float>(output_grad_desc.GetLengths()[2]) / input_grad_desc.GetLengths()[2];
        float scale_w =
            static_cast<float>(output_grad_desc.GetLengths()[3]) / input_grad_desc.GetLengths()[3];
        float scale_d =
            static_cast<float>(output_grad_desc.GetLengths()[4]) / input_grad_desc.GetLengths()[4];

        if(scale_h + scale_w + scale_d < 6)
            return false;
    }

    return true;
}

bool InterpolateNearestBackward::IsApplicable(
    const ExecutionContext&, const miopen::interpolate::BwdProblemDescription& problem) const
{
    if(problem.GetMode() != miopenInterpolateMode_t::MIOPEN_INTERPOLATE_MODE_NEAREST)
        return false;
    if(!IsOverRocmNearestBwd(problem))
        return false;
    return true;
}

ConvSolution InterpolateNearestBackward::GetSolution(
    const ExecutionContext& context,
    const miopen::interpolate::BwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetOutputGradDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetInputGradDesc().GetType());

    {
        auto dtype     = problem.GetInputGradDesc().GetType();
        size_t N_total = problem.GetInputGradDesc().GetElementSize();

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

        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_BWD_NEAREST},
                                                             {N_total},
                                                             "MIOpenInterpolate.cpp",
                                                             "InterpolateNearestBackward",
                                                             build_params));
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::interpolate::BwdInvokeParams>();

            auto input_grad_tv  = get_inner_expanded_tv<5>(deref(params.inputGradDesc));
            auto output_grad_tv = get_inner_expanded_tv<5>(deref(params.outputGradDesc));
            size_t nelems       = params.inputGradDesc->GetElementSize();

            kernel(params.input_grad,
                   params.output_grad,
                   input_grad_tv,
                   output_grad_tv,
                   nelems,
                   params.scale_factors);
        };
    };

    return result;
}

} // namespace interpolate

} // namespace solver

} // namespace miopen
