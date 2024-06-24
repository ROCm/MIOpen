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

#include "miopen/errors.hpp"
#include "miopen/kthvalue/problem_description.hpp"
#include "miopen/miopen.h"
#include "miopen/tensor_view_utils.hpp"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kthvalue/invoke_params.hpp>
#include <miopen/kthvalue/solvers.hpp>
#include <miopen/kthvalue.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace kthvalue {

bool KthvalueFwd::IsApplicable(const ExecutionContext& /*context*/,
                               const miopen::kthvalue::KthvalueFwdProblemDescription& problem) const
{
    if(problem.GetInputDesc().GetSize() > 5)
        return false;
    return true;
}

ConvSolution
KthvalueFwd::GetSolution(const ExecutionContext& context,
                         const miopen::kthvalue::KthvalueFwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto input_desc = problem.GetInputDesc();
    auto in_dtype   = miopen::GetDataType(input_desc.GetType());
    auto dtype      = problem.GetOutputDesc().GetType();
    auto size       = input_desc.GetElementSize();
    auto dim_size   = input_desc.GetLengths()[problem.GetDim()];

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(size / dim_size, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenKthvalue.cpp";
    kernel.kernel_name = "KthvalueForward";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", in_dtype == "bfloat16" ? "ushort" : in_dtype},
        {"LOCAL_SIZE", LOCAL_SIZE},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::kthvalue::FwdInvokeParams>();
            size_t dimSize        = params.inputDesc->GetLengths()[params.dim];
            size_t dimStride      = params.inputDesc->GetStrides()[params.dim];

            auto input_tv                     = get_inner_expanded_tv<5>(deref(params.inputDesc));
            auto input_tv_without_reduced_dim = get_tv_without_dim<5, 4>(input_tv, params.dim);

            kernel(params.input,
                   params.output,
                   params.indices,
                   params.k,
                   dimSize,
                   dimStride,
                   input_tv_without_reduced_dim);
        };
    };

    return result;
}

std::size_t KthvalueFwd::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::kthvalue::KthvalueFwdProblemDescription& /*problem*/) const
{
    return 0;
}

} // namespace kthvalue

} // namespace solver

} // namespace miopen
