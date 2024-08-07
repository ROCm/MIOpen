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
#include "miopen/tensor_view_utils.hpp"
#include <cstdint>
#include <miopen/avgpool/solvers.hpp>

#include <miopen/avgpool/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/avgpool.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE_FWD_2D 1024

namespace miopen {

namespace solver {

namespace avgpool {

bool AvgPoolForward2d::IsApplicable(const ExecutionContext& context,
                                    const miopen::avgpool::FwdProblemDescription& problem) const
{
    return true;
}

ConvSolution
AvgPoolForward2d::GetSolution(const ExecutionContext& context,
                              const miopen::avgpool::FwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto dtype        = problem.GetOutputDesc().GetType();
    size_t N_total    = problem.GetNtotal();

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype}};

    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE_FWD_2D}, {N_total}, "MIOpenAvgPool.cpp", "AvgPoolForward2d", build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::avgpool::FwdInvokeParams>();

            decltype(auto) kernel = handle_.Run(kernels.front());

            auto input_tv  = get_inner_expanded_tv<4>(deref(params.inputDesc));
            auto output_tv = get_inner_expanded_tv<4>(deref(params.outputDesc));

            size_t N  = deref(params.inputDesc).GetLengths()[0];
            size_t C  = deref(params.inputDesc).GetLengths()[1];
            size_t H  = deref(params.inputDesc).GetLengths()[2];
            size_t W  = deref(params.inputDesc).GetLengths()[3];
            size_t OH = deref(params.outputDesc).GetLengths()[2];
            size_t OW = deref(params.outputDesc).GetLengths()[3];

            kernel(params.input,
                   params.output,
                   N,
                   C,
                   H,
                   W,
                   OH,
                   OW,
                   params.kinfor,
                   params.stride,
                   params.padding,
                   params.count_include_pad,
                   params.divisor_override,
                   input_tv,
                   output_tv);
        };
    };

    return result;
}

} // namespace avgpool

} // namespace solver

} // namespace miopen
