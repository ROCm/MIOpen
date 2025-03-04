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
#include <miopen/kernel_build_params.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/reduce/invoke_params.hpp>
#include <miopen/reduce/solvers.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace reduce {

size_t ArgminForward::XGridSize(std::vector<size_t> indicedims) const
{
    size_t indice_numel =
        std::accumulate(indicedims.begin(), indicedims.end(), 1ULL, std::multiplies<size_t>());
    return AlignUp(indice_numel, LOCAL_SIZE);
}

/// \todo https://github.com/ROCm/MIOpen/pull/2583#discussion_r1437054128
bool ArgminForward::OverMaxGridSize(const ExecutionContext& context,
                                    const miopen::reduce::ProblemDescriptionExtreme& problem) const
{
    auto indicedims = problem.GetIndiceDesc().GetLengths();
    if(XGridSize(indicedims) > context.GetStream().GetImage3dMaxWidth())
        return false;
    return true;
}

bool ArgminForward::IsApplicable(const ExecutionContext& context,
                                 const miopen::reduce::ProblemDescriptionExtreme& problem) const
{
    if(!problem.IsValidDim())
        return false;
    if(!problem.IsValidLengthIndice())
        return false;
    if(!problem.IsValidInputNumel())
        return false;
    if(!problem.IsAllContiguousIndice())
        return false;
    if(!problem.IsNotLastDim())
        return false;
    if(!OverMaxGridSize(context, problem))
        return false;
    return true;
}

ConvSolution
ArgminForward::GetSolution(const ExecutionContext&,
                           const miopen::reduce::ProblemDescriptionExtreme& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetXDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetXDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetYDesc().GetType());
    auto indice_dtype = miopen::GetDataType(problem.GetIndiceDesc().GetType());
    auto xdims        = problem.GetXDesc().GetLengths();
    auto indicedims   = problem.GetIndiceDesc().GetLengths();

    {
        size_t xlocalsize;
        size_t xgridsize;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenReduceExtreme.cpp";
        kernel.kernel_name = "ExtremeFwdContiguous";
        xlocalsize         = LOCAL_SIZE;
        xgridsize          = XGridSize(indicedims);

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"INDICE_TYPE", indice_dtype},
            {"OP_TYPE", "ReduceExtremeOp_t::Argmin"},
            {"MIOPEN_REDUCE_EXTREME_ARGMIN", MIOPEN_REDUCE_EXTREME_ARGMIN},
            {"MIOPEN_REDUCE_EXTREME_ARGMAX", MIOPEN_REDUCE_EXTREME_ARGMAX},
            {"MIOPEN_REDUCE_EXTREME_MIN", MIOPEN_REDUCE_EXTREME_MIN},
            {"MIOPEN_REDUCE_EXTREME_MAX", MIOPEN_REDUCE_EXTREME_MAX}};

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
            decltype(auto) params = raw_params.CastTo<miopen::reduce::ExtremeInvokeParams>();

            auto xdims      = params.xDesc->GetLengths();
            auto indicedims = params.indiceDesc->GetLengths();
            auto dim        = params.dim;

            int32_t reduce_size = static_cast<int32_t>(xdims[dim]);
            auto indice_numel   = std::accumulate(
                indicedims.begin(), indicedims.end(), 1ULL, std::multiplies<size_t>());

            auto inner_size = std::accumulate(
                xdims.begin() + dim + 1, xdims.end(), 1ULL, std::multiplies<size_t>());

            kernel(params.x, nullptr, params.indice, indice_numel, reduce_size, inner_size);
        };
    };

    return result;
}

} // namespace reduce

} // namespace solver

} // namespace miopen
