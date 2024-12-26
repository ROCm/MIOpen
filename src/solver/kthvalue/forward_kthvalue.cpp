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

#include <miopen/kthvalue/problem_description.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kthvalue/invoke_params.hpp>
#include <miopen/kthvalue/solvers.hpp>
#include <miopen/kthvalue.hpp>

namespace miopen {

namespace solver {

namespace kthvalue {

bool IsImprovementOverROCm(const miopen::kthvalue::FwdProblemDescription& problem)
{
    TensorDescriptor inputDesc = problem.GetInputDesc();
    size_t dimSize             = inputDesc.GetLengths()[problem.GetDim()];
    size_t dimStride           = inputDesc.GetStrides()[problem.GetDim()];
    size_t dimNum              = inputDesc.GetLengths().size();

    return dimNum >= 2 && dimStride == 1 && dimSize >= 300;
}

bool KthvalueFwd::IsApplicable(const ExecutionContext& /*context*/,
                               const miopen::kthvalue::FwdProblemDescription& problem) const
{
    if(!IsImprovementOverROCm(problem))
        return false;
    if(problem.GetInputDesc().GetNumDims() > 5)
        return false;
    return true;
}

ConvSolution KthvalueFwd::GetSolution(const ExecutionContext& context,
                                      const miopen::kthvalue::FwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto input_desc    = problem.GetInputDesc();
    auto in_dtype      = miopen::GetDataType(input_desc.GetType());
    auto dtype         = problem.GetOutputDesc().GetType();
    auto size          = input_desc.GetElementSize();
    auto dim_size      = input_desc.GetLengths()[problem.GetDim()];
    size_t output_size = size / dim_size;

    size_t xlocalsize = 256;
    if(dim_size >= 8192)
    {
        xlocalsize = 512;
    }
    size_t xgridsize  = output_size * xlocalsize;
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenKthvalue.cpp";
    kernel.kernel_name = "KthvalueFwd";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", in_dtype == "bfloat16" ? "ushort" : in_dtype},
        {"LOCAL_SIZE", xlocalsize},
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
            size_t dim_stride     = params.inputDesc->GetStrides()[params.dim];

            auto input_tv                      = get_inner_expanded_tv<5>(deref(params.inputDesc));
            auto input_tv_without_selected_dim = get_tv_without_dim<5>(input_tv, params.dim);

            auto output_tv  = get_inner_expanded_tv<5>(deref(params.outputDesc));
            auto indices_tv = get_inner_expanded_tv<5>(deref(params.indicesDesc));

            kernel(params.input,
                   params.output,
                   params.indices,
                   params.k,
                   dim_size,
                   dim_stride,
                   output_size,
                   input_tv_without_selected_dim,
                   output_tv,
                   indices_tv);
        };
    };

    return result;
}

} // namespace kthvalue

} // namespace solver

} // namespace miopen
