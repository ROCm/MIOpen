/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/layernorm/solvers.hpp>

#include <miopen/layernorm/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/layernorm.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace layernorm {

std::size_t sizeof_kernel_FLOAT(const miopen::layernorm::ProblemDescription& problem)
{
    const auto datatype = problem.GetXDesc().GetType();
    return get_data_size(datatype);
}

std::size_t sizeof_local_memory(const miopen::layernorm::ProblemDescription& problem)
{
    std::size_t rv = 0;
    rv += LOCAL_SIZE * sizeof_kernel_FLOAT(problem) * 2;
    return rv;
}

bool LayernormForward::IsApplicable(const ExecutionContext&,
                                    const miopen::layernorm::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsSameLength())
        return false;
    if(!problem.IsAllPacked())
        return false;
    if(!problem.IsRightNormDim())
        return false;
    if(!(sizeof_local_memory(problem) <= TargetProperties::GetMaxLocalMemorySize()))
        return false;
    return true;
}

ConvSolution
LayernormForward::GetSolution(const ExecutionContext& context,
                              const miopen::layernorm::ProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    {
        auto dtype = problem.GetXDesc().GetType();
        auto dims  = problem.GetXDesc().GetLengths();

        size_t outer_size = 1;
        for(size_t i = 0; i < problem.GetNormalizedDim(); i++)
        {
            outer_size *= dims[i];
        }

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = outer_size * xlocalsize;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenLayerNorm.cpp";
        kernel.kernel_name = "LayernormFwdContiguous";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
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
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::layernorm::InvokeParams>();

            auto dims         = params.xDesc->GetLengths();
            size_t inner_size = 1;

            for(size_t i = params.normalized_dim; i < dims.size(); i++)
            {
                inner_size *= dims[i];
            }

            kernel(params.x,
                   params.y,
                   params.weight,
                   params.bias,
                   params.mean,
                   params.rstd,
                   params.epsilon,
                   inner_size,
                   static_cast<bool>(params.mode));
        };
    };

    return result;
}

} // namespace layernorm

} // namespace solver

} // namespace miopen
