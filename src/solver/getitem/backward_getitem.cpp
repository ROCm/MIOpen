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
#include <miopen/getitem.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/getitem/invoke_params.hpp>
#include <miopen/getitem/solvers.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace getitem {

bool IsLargeIndex(const miopen::getitem::ProblemDescription& problem)
{
    auto dy_dims = problem.GetDYDesc().GetLengths();
    auto dx_dims = problem.GetDXDesc().GetLengths();

    for(int32_t i = 0; i < problem.GetDimCount(); i++)
    {
        if(dy_dims[problem.GetDim(i)] / dx_dims[problem.GetDim(i)] > 400)
            return false;
    }

    return true;
}

bool GetitemBackward::IsApplicable(const ExecutionContext& /*context*/,
                                   const miopen::getitem::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!IsLargeIndex(problem))
        return false;
    return true;
}

ConvSolution GetitemBackward::GetSolution(const ExecutionContext& /*context*/,
                                          const miopen::getitem::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    const auto& dtype        = problem.GetDYDesc().GetType();
    const auto& input_dtype  = miopen::GetDataType(problem.GetDYDesc().GetType());
    const auto& index_dtype  = miopen::GetDataType(problem.GetIndexDesc(0).GetType());
    const auto& error_dtype  = miopen::GetDataType(problem.GetErrorDesc().GetType());
    const auto& output_dtype = miopen::GetDataType(problem.GetDXDesc().GetType());
    const auto& dy_dims      = problem.GetDYDesc().GetLengths();
    const auto& indexCount   = problem.GetIndexCount();

    auto dy_numel =
        std::accumulate(dy_dims.begin(), dy_dims.end(), 1ULL, std::multiplies<size_t>());

    for(int32_t i = 0; i < indexCount; i++)
    {
        const auto& index_dims = problem.GetIndexDesc(i).GetLengths();
        auto index_numel =
            std::accumulate(index_dims.begin(), index_dims.end(), 1L, std::multiplies<int64_t>());

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(index_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenGetitem.cpp";
        kernel.kernel_name = "GetItemBuildIndices";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"INDEX_TYPE", index_dtype},
            {"ERROR_TYPE", error_dtype},
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
        size_t xgridsize  = AlignUp(dy_numel, xlocalsize);
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
            {"ERROR_TYPE", error_dtype},
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

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::getitem::GetitemInvokeParams>();

            const auto& start_dim = params.dims[0];
            const auto& dx_dims   = params.dxDesc.GetLengths();

            const auto& dims     = params.dims;
            const auto& dimCount = params.dimCount;

            std::vector<int32_t> output_dims(dimCount);
            for(int32_t i = 0; i < dimCount; i++)
            {
                output_dims[i] = static_cast<int32_t>(dx_dims[dims[i]]);
            }

            const auto& indexCount = params.indexCount;
            const auto& index_dims = params.indexDescs[0]->GetLengths();
            const auto& sliceCount = params.sliceCount;
            const auto& slices     = params.slices;
            auto dim_info_offset =
                indexCount > 0 ? indexCount * static_cast<int32_t>(index_dims[0]) : 0;

            auto dy_tv = get_inner_expanded_tv<5>(params.dyDesc);
            auto dx_tv = get_inner_expanded_tv<5>(params.dxDesc);

            slice_tv<5>(dx_tv, sliceCount, slices);

            auto elapsed = 0.f;
            HipEventPtr start;
            HipEventPtr stop;
            bool reset_profiling_state = false;

            for(int32_t i = 0; i < indexCount; i++)
            {
                decltype(auto) build_index_kernel = handle_.Run(kernels[i]);

                const auto& index_dim  = dims[i];
                const auto& dim_size   = output_dims[i];
                auto index_tv          = get_inner_expanded_tv<5>(*params.indexDescs[i]);
                const auto& dim_offset = i;

                if((i == 0) && handle_.IsProfilingEnabled())
                {
                    handle_.EnableProfiling(false);
                    reset_profiling_state = true;
                    start                 = miopen::make_hip_event();
                    stop                  = miopen::make_hip_event();
                    hipEventRecord(start.get(), handle_.GetStream());
                }

                build_index_kernel(params.indexs[i],
                                   params.workspace,
                                   params.error,
                                   index_dim,
                                   indexCount,
                                   dim_size,
                                   index_tv,
                                   dim_offset,
                                   dim_info_offset);
            }

            if((indexCount == 0) && handle_.IsProfilingEnabled())
            {
                handle_.EnableProfiling(false);
                reset_profiling_state = true;
                start                 = miopen::make_hip_event();
                stop                  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            decltype(auto) kernel = handle_.Run(kernels[indexCount]);

            kernel(params.dy,
                   params.workspace,
                   params.dx,
                   start_dim,
                   indexCount,
                   dy_tv,
                   dx_tv,
                   dim_info_offset,
                   params.offset);

            if(reset_profiling_state)
            {
                hipEventRecord(stop.get(), handle_.GetStream());
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);

                hipEventDestroy(start.get());
                hipEventDestroy(stop.get());
                handle_.EnableProfiling(true);
            };
        };
    };

    return result;
}

std::size_t
GetitemBackward::GetWorkspaceSize(const ExecutionContext& /*context*/,
                                  const miopen::getitem::ProblemDescription& problem) const
{
    const auto& indexCount = problem.GetIndexCount();
    if(indexCount > 0)
    {
        const auto& index_dims = problem.GetIndexDesc(0).GetLengths();
        auto index_numel =
            std::accumulate(index_dims.begin(), index_dims.end(), 1L, std::multiplies<int64_t>());
        return (indexCount * index_numel + problem.GetIndexCount()) *
               get_data_size(problem.GetIndexDesc(0).GetType());
    }

    return 0;
}

} // namespace getitem

} // namespace solver

} // namespace miopen
