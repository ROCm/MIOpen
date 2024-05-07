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

#include "miopen/kernel_info.hpp"
#include "miopen/mlo_internal.hpp"
#include <cstddef>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/glu/invoke_params.hpp>
#include <miopen/glu/solvers.hpp>
#include <miopen/glu.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace glu {

bool GLUBackward::IsApplicable(const ExecutionContext& context,
                               const miopen::glu::ProblemDescription& problem) const
{
    std::ignore    = context;
    auto inputDims = problem.GetInputDesc().GetLengths();
    auto input_numel =
        std::accumulate(inputDims.begin(), inputDims.end(), 1, std::multiplies<int32_t>());

    if(!problem.IsSameType())
        return false;
    if(!problem.IsAllPacked())
        return false;
    if(!problem.IsFirstDim())
        return false;
    if(!(input_numel < 800000))
        return false;
    return true;
}

ConvSolution GLUBackward::GetSolution(const ExecutionContext& context,
                                      const miopen::glu::ProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto dtype            = problem.GetInputDesc().GetType();
        auto input_dtype      = miopen::GetDataType(problem.GetInputDesc().GetType());
        auto input_grad_dtype = miopen::GetDataType(problem.GetInputGradDesc().GetType());
        auto inputDims        = problem.GetInputDesc().GetLengths();
        auto input_numel =
            std::accumulate(inputDims.begin(), inputDims.end(), 1, std::multiplies<int32_t>());

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(input_numel / 2, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenGLU.cpp";
        kernel.kernel_name = "GLUBwdContiguous";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", input_grad_dtype == "bfloat16" ? "ushort" : input_grad_dtype}};

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel  = handle_.Run(kernels.front());
            decltype(auto) params  = raw_params.CastTo<miopen::glu::BwdInvokeParams>();
            auto outputGradDims    = params.outputGradDesc->GetLengths();
            auto output_grad_numel = std::accumulate(
                outputGradDims.begin(), outputGradDims.end(), 1ULL, std::multiplies<size_t>());

            kernel(params.input, params.outputGrad, params.inputGrad, output_grad_numel);
        };
    };

    return result;
}

} // namespace glu

} // namespace solver

} // namespace miopen
