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

#define LOCAL_SIZE_FWD_2D 256

namespace miopen {

namespace solver {

namespace adaptivemaxpool {

bool AdaptiveMaxPoolForward2d::IsApplicable(
    const ExecutionContext&, const miopen::adaptivemaxpool::FwdProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetNumDims() != 4 || problem.GetOutputDesc().GetNumDims() != 4)
    {
        return false;
    }
    if(!(problem.GetInputDesc().GetType() == miopenFloat ||
         problem.GetInputDesc().GetType() == miopenHalf ||
         problem.GetInputDesc().GetType() == miopenBFloat16))
        return false;
    return true;
}

ConvSolution AdaptiveMaxPoolForward2d::GetSolution(
    const ExecutionContext& context,
    const miopen::adaptivemaxpool::FwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto dtype        = problem.GetOutputDesc().GetType();
    uint64_t N_total  = problem.GetNtotal();
    bool use_indices  = problem.IsUseIndices();
    float infinity    = std::numeric_limits<float>::max();

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"INFINITY", infinity},
    };

    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_FWD_2D},
                                                         {N_total},
                                                         "MIOpenAdaptiveMaxPool.cpp",
                                                         "AdaptiveMaxPoolForward2d",
                                                         build_params));

    result.invoker_factory = [use_indices](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::adaptivemaxpool::FwdInvokeParams>();

            decltype(auto) kernel = handle_.Run(kernels.front());

            auto input_tv  = get_inner_expanded_tv<4>(deref(params.inputDesc));
            auto output_tv = get_inner_expanded_tv<4>(deref(params.outputDesc));
            tensor_view_t<4> indices_tv;
            if(use_indices)
                indices_tv = get_inner_expanded_tv<4>(deref(params.indicesDesc));

            uint64_t N  = deref(params.inputDesc).GetLengths()[0];
            uint64_t C  = deref(params.inputDesc).GetLengths()[1];
            uint64_t H  = deref(params.inputDesc).GetLengths()[2];
            uint64_t W  = deref(params.inputDesc).GetLengths()[3];
            uint64_t OH = deref(params.outputDesc).GetLengths()[2];
            uint64_t OW = deref(params.outputDesc).GetLengths()[3];

            kernel(params.input,
                   params.output,
                   params.indices,
                   N,
                   C,
                   H,
                   W,
                   OH,
                   OW,
                   input_tv,
                   output_tv,
                   indices_tv);
        };
    };

    return result;
}

} // namespace adaptivemaxpool

} // namespace solver

} // namespace miopen
