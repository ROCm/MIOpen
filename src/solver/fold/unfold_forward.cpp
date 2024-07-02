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

#include "miopen/fold/problem_description.hpp"
#include "miopen/miopen.h"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/fold/invoke_params.hpp>
#include <miopen/fold/solvers.hpp>
#include <miopen/fold.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace fold {

bool UnfoldFwd::IsApplicable(
    [[maybe_unused]] const ExecutionContext& /*context*/,
    const miopen::fold::UnfoldFwdProblemDescription& problem) const
{
    return true;
}

ConvSolution UnfoldFwd::GetSolution(
    [[maybe_unused]] const ExecutionContext& context,
    const miopen::fold::UnfoldFwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto in_dtype    = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto dtype       = problem.GetOutputDesc().GetType();
    auto input_dims  = problem.GetInputDesc().GetLengths();

    auto output_dims = problem.GetOutputDesc().GetLengths();
    const int32_t N = static_cast<int32_t>(input_dims[0]);
    const int32_t C = static_cast<int32_t>(input_dims[1]);
    int spatial_dim_size = input_dims.size() - 2;
    int32_t P = 1, L = 1;
    std::vector<int32_t> ls;
    for (int i = 0; i < spatial_dim_size; ++i) {
        P *= problem.kernel_size[i];
        int32_t l = (static_cast<int32_t>(input_dims[i + 2]) + 2 * problem.padding[i] -
                    problem.dilation[i] * (problem.kernel_size[i] - 1) - 1) /
                        problem.stride[i] +
                    1;
        L *= l;
        ls.push_back(l);
    }

    {
        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenUnfold.cpp";
        kernel.kernel_name = "UnfoldForward4D";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"IN_OUT_TYPE", in_dtype == "bfloat16" ? "ushort" : in_dtype},
        };
        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(N * C * P * L, LOCAL_SIZE);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;
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
            decltype(auto) params = raw_params.CastTo<miopen::fold::InvokeParams>();

            auto input_tv                = get_inner_expanded_tv<4>(deref(params.inputDesc));
            auto output_tv                = get_inner_expanded_tv<3>(deref(params.outputDesc));
            auto input_dims          = deref(params.inputDesc).GetLengths();
            auto output_dims          = deref(params.outputDesc).GetLengths();

            int spatial_dim_size = input_dims.size() - 2;
            const int32_t N = static_cast<int32_t>(input_dims[0]);
            const int32_t C = static_cast<int32_t>(input_dims[1]);
            int32_t P = 1, L = 1;
            std::vector<int32_t> ls;
            for (int i = 0; i < spatial_dim_size; ++i) {
                P *= params.kernel_size[i];
                int32_t l = (static_cast<int32_t>(input_dims[i + 2]) + 2 * params.padding[i] -
                            params.dilation[i] * (params.kernel_size[i] - 1) - 1) /
                                params.stride[i] +
                            1;
                L *= l;
                ls.push_back(l);
            }

            int32_t kernel_size_h = params.kernel_size[0];
            int32_t kernel_size_w = params.kernel_size[1];
            int32_t stride_h = params.stride[0];
            int32_t stride_w = params.stride[1];
            int32_t padding_h = params.padding[0];
            int32_t padding_w = params.padding[1];
            int32_t dilation_h = params.dilation[0];
            int32_t dilation_w = params.dilation[1];
            int32_t LH = ls[0];
            int32_t LW = ls[1];
            int32_t H = static_cast<int32_t>(input_dims[2]);
            int32_t W = static_cast<int32_t>(input_dims[3]);

            kernel(params.input,
                    params.output,
                    N,
                    C,
                    H,
                    W,
                    P,
                    L,
                    LH,
                    LW,
                    kernel_size_h,
                    kernel_size_w,
                    stride_h,
                    stride_w,
                    padding_h,
                    padding_w,
                    dilation_h,
                    dilation_w,
                    input_tv,
                    output_tv);
        };
    };

    return result;
}

} // namespace fold

} // namespace solver

} // namespace miopen
