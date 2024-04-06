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

#include <miopen/datatype.hpp>
#include <miopen/getitem.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/item/invoke_params.hpp>
#include <miopen/item/solvers.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace item {

bool GetitemBackward::IsApplicable(const ExecutionContext& context,
                                   const miopen::item::ProblemDescription& problem) const
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
    if(!IsImprovementOverROCm(context, problem))
        return false;
    return true;
}

ConvSolution GetitemBackward::GetSolution(const ExecutionContext& context,
                                          const miopen::item::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetDYDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetDYDesc().GetType());
    auto index_dtype  = miopen::GetDataType(problem.GetIndexDesc(0).GetType());
    auto output_dtype = miopen::GetDataType(problem.GetDXDesc().GetType());
    auto dy_dims      = problem.GetDYDesc().GetLengths();
    auto dy_strides   = problem.GetDYDesc().GetStrides();
    auto dx_dims      = problem.GetDXDesc().GetLengths();
    auto dx_strides   = problem.GetDXDesc().GetStrides();
    auto indexCount   = miopen::GetDataType(problem.GetIndexCount().GetType());
    auto dx_dims      = problem.GetDXDesc().GetLengths();
    auto dimCount     = problem.GetDimCount();
    auto dims         = problem.GetDims();
    auto sliceCount   = problem.GetSliceCount();
    auto slices       = problem.GetSlices();

    auto output_numel =
        std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

    std::vector<int32_t> output_dims;
    for(auto dim : dims)
    {
        output_dims.push_back(dx_dims[dim]);
    }

    int32_t dim_info_offset = indexCount * problem.GetIndexDesc(0).GetLengths();
    auto start_dim          = dims[0];

    for(i = 0; i < indexCount; i++)
    {
        auto dim_size         = output_dims[j];
        auto parallelism_size = get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(parallelism_size * output_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenGetitem.cpp";
        kernel.kernel_name = "GetitemBwd";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"INDEX_TYPE", index_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
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

    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenGetitem.cpp";
        kernel.kernel_name = "GetitemBwd";

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

    if(is_parallelism(reqd_work_item_cnt, output_numel, reduce_size))
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) parallel_kernel = handle_.Run(kernels[0]);
                decltype(auto) kernel          = handle_.Run(kernels[1]);
                decltype(auto) params          = raw_params.CastTo<miopen::item::InvokeParams>();

                auto xdims = params.xDesc->GetLengths();
                auto ydims = params.yDesc->GetLengths();
                auto dim   = params.dim;

                auto reduce_size = xdims[dim];
                auto output_numel =
                    std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

                auto inner_size = std::accumulate(
                    xdims.begin() + dim + 1, xdims.end(), 1ULL, std::multiplies<size_t>());

                auto reqd_work_item_cnt = get_reqd_work_item_cnt(handle_);
                auto parallelism_size =
                    get_parallelism_size(reqd_work_item_cnt, output_numel, reduce_size);

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
                decltype(auto) params = raw_params.CastTo<miopen::item::InvokeParams>();

                auto xdims = params.xDesc->GetLengths();
                auto ydims = params.yDesc->GetLengths();
                auto dim   = params.dim;

                auto reduce_size = xdims[dim];
                auto output_numel =
                    std::accumulate(ydims.begin(), ydims.end(), 1ULL, std::multiplies<size_t>());

                auto inner_size = std::accumulate(
                    xdims.begin() + dim + 1, xdims.end(), 1ULL, std::multiplies<size_t>());

                kernel(params.x,
                       params.y,
                       output_numel,
                       reduce_size,
                       inner_size,
                       static_cast<bool>(params.nanPropagation));
            };
        };
    }
    return result;
}

std::size_t GetitemBackward::GetWorkspaceSize(const ExecutionContext& context,
                                              const miopen::item::ProblemDescription& problem) const
{
    auto index_size = problem.GetIndexCount();
    if(index_size > 0)
    {
        auto index_dims = problem.GetIndexDesc(0).GetLength();
        auto index_numel =
            std::accumulate(index_dims.begin(), index_dims.end(), 1L, std::multiplies<int64_t>());
        return index_dims * index_numel * get_data_size(problem.GetIndexDesc(0).GetType()) +
               sizeof(int32_t);
    }

    return 0;
}

} // namespace item

} // namespace solver

} // namespace miopen
