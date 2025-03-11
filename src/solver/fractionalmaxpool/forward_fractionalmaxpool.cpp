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

#include <miopen/fractionalmaxpool.hpp>
#include <miopen/fractionalmaxpool/invoke_params.hpp>
#include <miopen/fractionalmaxpool/solvers.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/datatype.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE_FWD 256

namespace miopen {

namespace solver {

namespace fractionalmaxpool {

bool FractionalMaxPoolForward::IsApplicable(
    const ExecutionContext&, const miopen::fractionalmaxpool::FwdProblemDescription& problem) const
{
    if(!(problem.GetOutputDesc().GetType() == miopenHalf ||
         problem.GetOutputDesc().GetType() == miopenFloat ||
         problem.GetOutputDesc().GetType() == miopenBFloat16))
    {
        return false;
    }
    return true;
}

ConvSolution FractionalMaxPoolForward::GetSolution(
    const ExecutionContext& context,
    const miopen::fractionalmaxpool::FwdProblemDescription& problem) const
{
    std::ignore        = context;
    auto output_dtype  = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto indices_dtype = miopen::GetDataType(problem.GetIndicesDesc().GetType());
    auto dtype         = problem.GetOutputDesc().GetType();

    auto result       = ConvSolution{miopenStatusSuccess};
    auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
        {"I_TYPE", indices_dtype == "int64" ? "size_t" : indices_dtype},
        {"MAX_FLOAT", std::numeric_limits<float>::max()},
    };

    if(problem.GetOutputDesc().GetNumDims() == 4)
    {
        result.construction_params.push_back(
            make_hip_kernel({LOCAL_SIZE_FWD},
                            {problem.GetOutputDesc().GetElementSize()},
                            "MIOpenFractionalMaxPool.cpp",
                            "FractionalMaxPool2dForward",
                            build_params));

        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params =
                    raw_params.CastTo<miopen::fractionalmaxpool::FwdInvokeParams>();
                auto input_tv         = get_inner_expanded_tv<4>(deref(params.inputDesc));
                auto output_tv        = get_inner_expanded_tv<4>(deref(params.outputDesc));
                auto indices_tv       = get_inner_expanded_tv<4>(deref(params.indicesDesc));
                auto random_sample_tv = get_inner_expanded_tv<3>(deref(params.randomSampleDesc));

                kernel(params.input,
                       params.output,
                       params.indices,
                       params.random_sample,
                       params.KD,
                       params.KH,
                       input_tv,
                       output_tv,
                       indices_tv,
                       random_sample_tv);
            };
        };
    }
    else
    {
        result.construction_params.push_back(
            make_hip_kernel({LOCAL_SIZE_FWD},
                            {problem.GetOutputDesc().GetElementSize()},
                            "MIOpenFractionalMaxPool.cpp",
                            "FractionalMaxPool3dForward",
                            build_params));

        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels[0]);
                decltype(auto) params =
                    raw_params.CastTo<miopen::fractionalmaxpool::FwdInvokeParams>();

                auto input_tv         = get_inner_expanded_tv<5>(deref(params.inputDesc));
                auto output_tv        = get_inner_expanded_tv<5>(deref(params.outputDesc));
                auto indices_tv       = get_inner_expanded_tv<5>(deref(params.indicesDesc));
                auto random_sample_tv = get_inner_expanded_tv<3>(deref(params.randomSampleDesc));

                kernel(params.input,
                       params.output,
                       params.indices,
                       params.random_sample,
                       params.KD,
                       params.KH,
                       params.KW,
                       input_tv,
                       output_tv,
                       indices_tv,
                       random_sample_tv);
            };
        };
    }

    return result;
};

} // namespace fractionalmaxpool

} // namespace solver

} // namespace miopen
