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

#include <miopen/rope.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/rope/invoke_params.hpp>
#include <miopen/rope/solvers.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace rope {

bool RoPEBackward::IsApplicable(const ExecutionContext& /*context*/,
                                const miopen::rope::ProblemDescriptionBwd& problem) const
{
    if(!problem.IsValidLength())
        return false;
    if(!problem.IsSameType())
        return false;
    if(!problem.IsAllContiguous())
        return false;
    return true;
}

ConvSolution RoPEBackward::GetSolution(const ExecutionContext&,
                                       const miopen::rope::ProblemDescriptionBwd& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetDYDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetDYDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetDXDesc().GetType());
    auto dxdims       = problem.GetDXDesc().GetLengths();
    auto output_numel =
        std::accumulate(dxdims.begin(), dxdims.end(), 1ULL, std::multiplies<size_t>());

    {
        size_t xlocalsize;
        size_t xgridsize;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenRoPE.cpp";
        kernel.kernel_name = "RoPEBwdContiguous";
        xlocalsize         = LOCAL_SIZE;
        xgridsize          = output_numel;

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype}};

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
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::rope::BwdInvokeParams>();

            auto dxdims  = params.dxDesc->GetLengths();
            auto cosdims = params.cosDesc->GetLengths();

            auto output_numel =
                std::accumulate(dxdims.begin(), dxdims.end(), 1ULL, std::multiplies<size_t>());
            auto rotary_numel =
                std::accumulate(cosdims.begin(), cosdims.end(), 1ULL, std::multiplies<size_t>());

            kernel(params.dy, params.cos, params.sin, params.dx, output_numel, rotary_numel);
        };
    };

    return result;
}

} // namespace rope

} // namespace solver

} // namespace miopen
