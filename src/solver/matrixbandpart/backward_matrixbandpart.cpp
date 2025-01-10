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

#include <miopen/datatype.hpp>
#include <miopen/matrixbandpart.hpp>
#include <miopen/matrixbandpart/invoke_params.hpp>
#include <miopen/matrixbandpart/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE_BWD 256

namespace miopen {

namespace solver {

namespace matrixbandpart {

bool IsOverRocmBwd(const miopen::matrixbandpart::ProblemDescription& problem)
{
    uint64_t mul_dims = 1;
    for(auto dim : problem.GetYDesc().GetLengths())
    {
        mul_dims *= dim;
    }
    if(mul_dims <= 524288)
    {
        return true;
    }
    return false;
}

bool MatrixBandPartBackward::IsApplicable(
    const ExecutionContext&, const miopen::matrixbandpart::ProblemDescription& problem) const
{
    if(!(problem.GetYDesc().GetType() == miopenHalf ||
         problem.GetYDesc().GetType() == miopenFloat ||
         problem.GetYDesc().GetType() == miopenBFloat16))
    {
        return false;
    }
    if(!IsOverRocmBwd(problem))
    {
        return false;
    }
    return true;
}

ConvSolution
MatrixBandPartBackward::GetSolution(const ExecutionContext& context,
                                    const miopen::matrixbandpart::ProblemDescription& problem) const
{
    std::ignore          = context;
    auto output_dtype    = miopen::GetDataType(problem.GetYDesc().GetType());
    auto num_lower_dtype = miopen::GetDataType(problem.GetNumLowerDesc().GetType());
    auto dtype           = problem.GetYDesc().GetType();
    auto numel           = problem.GetYDesc().GetElementSize();

    auto result       = ConvSolution{miopenStatusSuccess};
    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"LOCAL_SIZE", LOCAL_SIZE_BWD},
        {"N_TYPE", num_lower_dtype == "int64" ? "size_t" : num_lower_dtype},
    };

    if(!problem.IsAllContiguous())
    {
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_BWD}, {numel}, "MIOpenMatrixBandPart.cpp", "MatrixBandPart", build_params));

        result.invoker_factory = [numel](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params =
                    raw_params.CastTo<miopen::matrixbandpart::BwdInvokeParams>();
                auto output_grad_tv = get_inner_expanded_tv<5>(deref(params.outputGradDesc));
                auto input_grad_tv  = get_inner_expanded_tv<5>(deref(params.inputGradDesc));
                uint64_t num_dim    = deref(params.inputGradDesc).GetNumDims();

                kernel(params.output_grad,
                       params.input_grad,
                       params.num_lower,
                       params.num_upper,
                       numel,
                       num_dim,
                       output_grad_tv,
                       input_grad_tv);
            };
        };
    }
    else
    {
        result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_BWD},
                                                             {numel},
                                                             "MIOpenMatrixBandPart.cpp",
                                                             "MatrixBandPartContiguous",
                                                             build_params));

        result.invoker_factory = [numel](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params =
                    raw_params.CastTo<miopen::matrixbandpart::BwdInvokeParams>();
                auto num_dim = deref(params.inputGradDesc).GetNumDims();
                int64_t W    = deref(params.inputGradDesc).GetLengths()[num_dim - 1];
                int64_t H    = deref(params.inputGradDesc).GetLengths()[num_dim - 2];

                kernel(params.output_grad,
                       params.input_grad,
                       params.num_lower,
                       params.num_upper,
                       numel,
                       W,
                       H);
            };
        };
    }

    return result;
}

} // namespace matrixbandpart

} // namespace solver

} // namespace miopen
