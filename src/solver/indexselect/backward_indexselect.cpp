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
#include <miopen/indexselect.hpp>
#include <miopen/indexselect/invoke_params.hpp>
#include <miopen/indexselect/solvers.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

namespace miopen {

namespace solver {

namespace indexselect {

static bool IsImprovementOverROCm(const miopen::indexselect::BwdProblemDescription& problem)
{
    return true;
}

bool IndexSelectBackward::IsApplicable(
    const ExecutionContext& context,
    const miopen::indexselect::BwdProblemDescription& problem) const
{
    if(!IsImprovementOverROCm(problem))
        return false;
    return true;
}

ConvSolution
IndexSelectBackward::GetSolution(const ExecutionContext& /*context*/,
                                 const miopen::indexselect::BwdProblemDescription& problem) const
{
    puts("st GetSolution in IndexSelectBackward");
    static const size_t LOCAL_SIZE = 256;
    auto result                    = ConvSolution{miopenStatusSuccess};

    auto dtype    = problem.GetInputGradDesc().GetType();
    auto io_dtype = miopen::GetDataType(dtype);
    auto dim      = problem.GetDim();

    size_t xlocalsize = LOCAL_SIZE;
    size_t ylocalsize = 1;
    size_t zlocalsize = 1;

    size_t xgridsize = 1ULL;
    for(size_t i = 0; i < ydims.size(); i++)
    {
        if(i != dim)
        {
            xgridsize *= ydims[i];
        }
    }
    if(xgridsize % LOCAL_SIZE != 0)
    {
        xgridsize = (xgridsize / LOCAL_SIZE + 1) * LOCAL_SIZE;
    }
    size_t ygridsize = 1;
    size_t zgridsize = 1;

    auto kernel        = KernelInfo();
    kernel.kernel_file = "MIOpenIndexSelect.cpp";
    kernel.kernel_name = "IndexSelectBackward";

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
                              {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
                              {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
                              {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);
    kernel.l_wk.push_back(zlocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);
    kernel.g_wk.push_back(zgridsize);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::indexselect::BwdInvokeParams>();

            auto xGradlens    = params.xGradDesc.GetLengths();
            auto yGradlens    = params.yGradDesc.GetLengths();
            auto xGradStrides = params.xGradDesc.GetStrides();
            auto yGradStrides = params.yGradDesc.GetStrides();
            auto dim          = params.dim;

            size_t N  = 1;
            size_t st = 1;

            for(size_t i = 0; i < xGradlens.size(); i++)
            {
                if(dim != i)
                {
                    N *= (int)xGradlens[i];
                }
                if(dim < i)
                {
                    st *= (int)xGradlens[i];
                }
            }

            tensor_view_t<5> outGrad_tv =
                get_inner_expanded_tv<5>(miopen::deref(params.outputGradDesc));
            tensor_view_t<5> inGrad_tv =
                get_inner_expanded_tv<5>(miopen::deref(params.inputGradDesc));

            kernel(params.outputGrad,
                   params.indices,
                   params.inputGrad,
                   dim,
                   outGrad_tv,
                   inGrad_tv,
                   N,
                   st,
                   iK,
                   oK);
        };
    };

    return result;
}

} // namespace indexselect

} // namespace solver

} // namespace miopen
