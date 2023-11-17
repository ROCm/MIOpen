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

#include <miopen/reduce/solvers.hpp>

#include <miopen/reduce/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/sum.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace reduce {

bool SumForward::IsApplicable(const ExecutionContext&,
                              const miopen::reduce::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightDim())
        return false;
    if(!problem.IsRightLength())
        return false;
    if(!problem.IsAllPacked())
        return false;
    if(!problem.IsNotLastDim())
        return false;
    return true;
}

ConvSolution SumForward::GetSolution(const ExecutionContext& context,
                                     const miopen::reduce::ProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetXDesc().GetType();
    auto xdims = problem.GetXDesc().GetLengths();
    auto ydims = problem.GetYDesc().GetLengths();
    auto dim   = problem.GetDim();

    auto reduce_size = xdims[dim];
    auto output_numel =
        std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

    auto reqd_work_item_cnt = static_cast<size_t>(256 * 120 * 4);
    // Now it is set for mi250.
    // TODO: parameterize this for different GPUs
    bool is_num_work_item_enough = (output_numel > reqd_work_item_cnt);
    bool is_parallelism_enough   = (output_numel * reduce_size > reqd_work_item_cnt);

    if(!is_num_work_item_enough && is_parallelism_enough)
    {
        size_t parallelism_size = 1;
        while(parallelism_size * output_numel < reqd_work_item_cnt &&
              parallelism_size < reduce_size)
        {
            parallelism_size *= 2;
        }

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(parallelism_size * output_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenSum.cpp";
        kernel.kernel_name = "SumParallelFwdContiguous";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int32_t>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int32_t>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int32_t>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int32_t>(dtype == miopenBFloat16)},
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

    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenSum.cpp";
        kernel.kernel_name = "SumFwdContiguous";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
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

    if(!is_num_work_item_enough && is_parallelism_enough)
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) parallel_kernel = handle_.Run(kernels[0]);
                decltype(auto) kernel          = handle_.Run(kernels[1]);
                decltype(auto) params          = raw_params.CastTo<miopen::reduce::InvokeParams>();

                auto xdims = params.xDesc->GetLengths();
                auto ydims = params.yDesc->GetLengths();
                auto dim   = params.dim;

                auto reduce_size = xdims[dim];
                auto output_numel =
                    std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

                auto inner_size = 1ULL;
                for(int32_t i = dim + 1; i < xdims.size(); i++)
                {
                    inner_size *= xdims[i];
                }

                auto reqd_work_item_cnt = static_cast<size_t>(256 * 120 * 4);

                auto parallelism_size = 1ULL;
                while(parallelism_size * output_numel < reqd_work_item_cnt &&
                      parallelism_size < reduce_size)
                {
                    parallelism_size *= 2ULL;
                }
                printf("parallelism_size: %d\n", parallelism_size);

                auto elapsed = 0.f;

                parallel_kernel(params.x,
                                params.workspace,
                                output_numel,
                                reduce_size,
                                parallelism_size,
                                inner_size,
                                static_cast<bool>(params.nanPropagation));

                if(handle_.IsProfilingEnabled())
                    elapsed = handle_.GetKernelTime();

                kernel(params.workspace,
                       params.y,
                       output_numel,
                       parallelism_size,
                       inner_size,
                       dim,
                       static_cast<bool>(params.nanPropagation));

                if(handle_.IsProfilingEnabled())
                {
                    elapsed += handle_.GetKernelTime();
                    handle_.ResetKernelTime();
                    handle_.AccumKernelTime(elapsed);
                };
            };
        };
    }
    else
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::reduce::InvokeParams>();

                auto xdims = params.xDesc->GetLengths();
                auto ydims = params.yDesc->GetLengths();
                auto dim   = params.dim;

                auto reduce_size = xdims[dim];
                auto output_numel =
                    std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

                size_t inner_size = 1;
                for(int32_t i = dim + 1; i < xdims.size(); i++)
                {
                    inner_size *= xdims[i];
                }

                kernel(params.x,
                       params.y,
                       output_numel,
                       reduce_size,
                       inner_size,
                       dim,
                       static_cast<bool>(params.nanPropagation));
            };
        };
    }
    return result;
}

std::size_t SumForward::GetWorkspaceSize(const ExecutionContext&,
                                         const miopen::reduce::ProblemDescription& problem) const
{
    auto xdims = problem.GetXDesc().GetLengths();
    auto ydims = problem.GetYDesc().GetLengths();

    auto reduce_size = xdims[problem.GetDim()];
    auto output_numel =
        std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

    auto reqd_work_item_cnt = static_cast<size_t>(256 * 120 * 4);
    // Now it is set for mi250.
    // TODO: parameterize this for different GPUs
    bool is_num_work_item_enough = (output_numel > reqd_work_item_cnt);
    bool is_parallelism_enough   = (output_numel * reduce_size > reqd_work_item_cnt);
    if(!is_num_work_item_enough && is_parallelism_enough)
    {
        size_t parallelism_size = 1;
        while(parallelism_size * output_numel < reqd_work_item_cnt &&
              parallelism_size < reduce_size)
        {
            parallelism_size *= 2;
        }

        return parallelism_size * output_numel * get_data_size(problem.GetXDesc().GetType());
    }

    return 0;
}

} // namespace reduce

} // namespace solver

} // namespace miopen
