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

#include <miopen/adaptivemaxpool/solvers.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <miopen/adaptivemaxpool/invoke_params.hpp>
#include <miopen/adaptivemaxpool.hpp>
#include <miopen/datatype.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE_BWD_3D 256

namespace miopen {

namespace solver {

namespace adaptivemaxpool {

namespace {

bool IsOverRocmBwd3d(const miopen::adaptivemaxpool::BwdProblemDescription& problem)
{
    if(!problem.IsAllContiguous())
    {
        return true;
    }
    return false;
}

} // namespace

bool AdaptiveMaxPoolBackward3d::IsApplicable(
    const ExecutionContext&, const miopen::adaptivemaxpool::BwdProblemDescription& problem) const
{
    if(problem.GetInputGradDesc().GetNumDims() != 5 ||
       problem.GetOutputGradDesc().GetNumDims() != 5)
    {
        return false;
    }
    if(!IsOverRocmBwd3d(problem))
    {
        return false;
    }
    if(!(problem.GetInputGradDesc().GetType() == miopenFloat ||
         problem.GetInputGradDesc().GetType() == miopenHalf ||
         problem.GetInputGradDesc().GetType() == miopenBFloat16))
        return false;
    return true;
}

ConvSolution AdaptiveMaxPoolBackward3d::GetSolution(
    const ExecutionContext& context,
    const miopen::adaptivemaxpool::BwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto output_dtype = miopen::GetDataType(problem.GetInputGradDesc().GetType());
    auto dtype        = problem.GetInputGradDesc().GetType();
    uint64_t N_total  = problem.GetNtotal();
    float infinity    = std::numeric_limits<float>::max();

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"INFINITY", infinity},
    };

    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_BWD_3D},
                                                         {N_total},
                                                         "MIOpenAdaptiveMaxPool.cpp",
                                                         "AdaptiveMaxPoolBackward3d",
                                                         build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::adaptivemaxpool::BwdInvokeParams>();

            decltype(auto) kernel = handle_.Run(kernels.front());

            auto indices_tv     = get_inner_expanded_tv<5>(deref(params.indicesDesc));
            auto input_grad_tv  = get_inner_expanded_tv<5>(deref(params.inputGradDesc));
            auto output_grad_tv = get_inner_expanded_tv<5>(deref(params.outputGradDesc));

            uint64_t N  = deref(params.inputGradDesc).GetLengths()[0];
            uint64_t C  = deref(params.inputGradDesc).GetLengths()[1];
            uint64_t D  = deref(params.inputGradDesc).GetLengths()[2];
            uint64_t H  = deref(params.inputGradDesc).GetLengths()[3];
            uint64_t W  = deref(params.inputGradDesc).GetLengths()[4];
            uint64_t OD = deref(params.outputGradDesc).GetLengths()[2];
            uint64_t OH = deref(params.outputGradDesc).GetLengths()[3];
            uint64_t OW = deref(params.outputGradDesc).GetLengths()[4];

            kernel(params.indices,
                   params.output_grad,
                   params.input_grad,
                   N,
                   C,
                   D,
                   H,
                   W,
                   OD,
                   OH,
                   OW,
                   indices_tv,
                   output_grad_tv,
                   input_grad_tv);
        };
    };

    return result;
}

} // namespace adaptivemaxpool

} // namespace solver

} // namespace miopen
