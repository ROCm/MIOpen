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

#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/lppool/solvers.hpp>

#include <miopen/lppool/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/lppool.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE_FWD_1D 256

namespace miopen {

namespace solver {

namespace lppool {

bool LPPoolForward1d::IsApplicable(const ExecutionContext&,
                                   const miopen::lppool::FwdProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetNumDims() != 3)
    {
        return false;
    }
    if(!(problem.GetInputDesc().GetType() == miopenHalf ||
         problem.GetInputDesc().GetType() == miopenFloat ||
         problem.GetInputDesc().GetType() == miopenBFloat16))
    {
        return false;
    }
    return true;
}

ConvSolution
LPPoolForward1d::GetSolution(const ExecutionContext& context,
                             const miopen::lppool::FwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result       = ConvSolution{miopenStatusSuccess};
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto dtype        = problem.GetOutputDesc().GetType();
    uint64_t N_total  = problem.GetNtotal();

    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
    };

    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE_FWD_1D}, {N_total}, "MIOpenLPPool.cpp", "LPPoolForward1d", build_params));

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::lppool::FwdInvokeParams>();

            decltype(auto) kernel = handle_.Run(kernels.front());

            auto input_tv  = get_inner_expanded_tv<3>(deref(params.inputDesc));
            auto output_tv = get_inner_expanded_tv<3>(deref(params.outputDesc));

            int64_t N  = deref(params.inputDesc).GetLengths()[0];
            int64_t C  = deref(params.inputDesc).GetLengths()[1];
            int64_t D  = deref(params.inputDesc).GetLengths()[2];
            int64_t OD = deref(params.outputDesc).GetLengths()[2];

            kernel(params.input,
                   params.output,
                   N,
                   C,
                   D,
                   OD,
                   params.KD,
                   params.SD,
                   params.norm_type,
                   input_tv,
                   output_tv);
        };
    };

    return result;
}

} // namespace lppool

} // namespace solver

} // namespace miopen
