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
#include <miopen/layernorm/solvers.hpp>
#include <miopen/layernorm/invoke_params.hpp>
#include <miopen/layernorm/utils.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/t5layernorm.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace layernorm {

bool T5LayernormBackward::IsApplicable(const ExecutionContext&,
                                       const miopen::layernorm::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsSameLength())
        return false;
    if(!problem.IsAllPacked())
        return false;
    if(!(sizeof_local_memory_t5(problem) <= TargetProperties::GetMaxLocalMemorySize()))
        return false;
    return true;
}

ConvSolution
T5LayernormBackward::GetSolution(const ExecutionContext& context,
                                 const miopen::layernorm::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetDYDesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetDYDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetDXDesc().GetType());
    auto dims         = problem.GetDYDesc().GetLengths();

    auto outer_size =
        std::accumulate(dims.begin(), dims.end() - 1, 1ULL, std::multiplies<size_t>());
    auto inner_size = dims[dims.size() - 1];

    auto reqd_work_item_cnt = get_reqd_work_item_cnt(context);

    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = outer_size * xlocalsize;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenLayerNorm.cpp";
        kernel.kernel_name = "T5LayernormBwdContiguous";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"LOCAL_SIZE", LOCAL_SIZE},
            {"MIOPEN_ELEMENTWISE_AFFINE", 0},
            {"MIOPEN_WEIGHT_BIAS", 1},
            {"MIOPEN_ELEMENTWISE_AFFINE_FUSED_ADD", 2},
            {"MIOPEN_WEIGHT_BIAS_FUSED_ADD", 3},
            {"MIOPEN_ELEMENTWISE_AFFINE_T5", 4},
            {"MIOPEN_WEIGHT_BIAS_T5", 5},
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

    if(is_parallelism(reqd_work_item_cnt, inner_size, outer_size))
    {
        {
            auto parallelism_size =
                get_parallelism_size(reqd_work_item_cnt, inner_size, outer_size);

            size_t xlocalsize = LOCAL_SIZE;
            size_t xgridsize  = AlignUp(parallelism_size * inner_size, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel = KernelInfo{};

            kernel.kernel_file = "MIOpenLayerNorm.cpp";
            kernel.kernel_name = "T5LayernormBwdWeightContiguousParallel";

            const auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
                {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
                {"LOCAL_SIZE", LOCAL_SIZE},
                {"MIOPEN_ELEMENTWISE_AFFINE", 0},
                {"MIOPEN_WEIGHT_BIAS", 1},
                {"MIOPEN_ELEMENTWISE_AFFINE_FUSED_ADD", 2},
                {"MIOPEN_WEIGHT_BIAS_FUSED_ADD", 3},
                {"MIOPEN_ELEMENTWISE_AFFINE_T5", 4},
                {"MIOPEN_WEIGHT_BIAS_T5", 5},
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
            size_t xgridsize  = AlignUp(inner_size, LOCAL_SIZE);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel = KernelInfo{};

            kernel.kernel_file = "MIOpenLayerNorm.cpp";
            kernel.kernel_name = "T5LayernormBwdContiguousReduceSum";

            const auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
                {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
                {"LOCAL_SIZE", LOCAL_SIZE},
                {"MIOPEN_ELEMENTWISE_AFFINE", 0},
                {"MIOPEN_WEIGHT_BIAS", 1},
                {"MIOPEN_ELEMENTWISE_AFFINE_FUSED_ADD", 2},
                {"MIOPEN_WEIGHT_BIAS_FUSED_ADD", 3},
                {"MIOPEN_ELEMENTWISE_AFFINE_T5", 4},
                {"MIOPEN_WEIGHT_BIAS_T5", 5},
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
    }
    else
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = inner_size;
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        kernel.kernel_file = "MIOpenLayerNorm.cpp";
        kernel.kernel_name = "T5LayernormBwdWeightContiguous";

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
            {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
            {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
            {"LOCAL_SIZE", LOCAL_SIZE},
            {"MIOPEN_ELEMENTWISE_AFFINE", 0},
            {"MIOPEN_WEIGHT_BIAS", 1},
            {"MIOPEN_ELEMENTWISE_AFFINE_FUSED_ADD", 2},
            {"MIOPEN_WEIGHT_BIAS_FUSED_ADD", 3},
            {"MIOPEN_ELEMENTWISE_AFFINE_T5", 4},
            {"MIOPEN_WEIGHT_BIAS_T5", 5},
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

    if(is_parallelism(reqd_work_item_cnt, inner_size, outer_size))
    {
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel                 = handle_.Run(kernels[0]);
                decltype(auto) weight_parallel_kernel = handle_.Run(kernels[1]);
                decltype(auto) weight_kernel          = handle_.Run(kernels[2]);
                decltype(auto) params = raw_params.CastTo<miopen::layernorm::T5BwdInvokeParams>();

                auto dims = params.dyDesc->GetLengths();

                auto outer_size =
                    std::accumulate(dims.begin(), dims.end() - 1, 1ULL, std::multiplies<size_t>());

                auto inner_size = dims[dims.size() - 1];

                auto reqd_work_item_cnt = get_reqd_work_item_cnt(handle_);
                auto parallelism_size =
                    get_parallelism_size(reqd_work_item_cnt, inner_size, outer_size);

                auto elapsed = 0.f;
                HipEventPtr start;
                HipEventPtr stop;

                if(handle_.IsProfilingEnabled())
                {
                    start = miopen::make_hip_event();
                    stop  = miopen::make_hip_event();
                    hipEventRecord(start.get(), handle_.GetStream());
                }

                kernel(params.dy,
                       params.x,
                       params.weight,
                       params.rstd,
                       params.dx,
                       inner_size,
                       static_cast<int32_t>(params.mode));

                weight_parallel_kernel(params.dy,
                                       params.x,
                                       params.rstd,
                                       params.workspace,
                                       outer_size,
                                       inner_size,
                                       parallelism_size);

                weight_kernel(params.workspace, params.dw, inner_size, parallelism_size);

                if(handle_.IsProfilingEnabled())
                {
                    hipEventRecord(stop.get(), handle_.GetStream());
                    hipEventSynchronize(stop.get());
                    hipEventElapsedTime(&elapsed, start.get(), stop.get());
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
                decltype(auto) kernel        = handle_.Run(kernels[0]);
                decltype(auto) weight_kernel = handle_.Run(kernels[1]);
                decltype(auto) params = raw_params.CastTo<miopen::layernorm::T5BwdInvokeParams>();

                auto dims = params.dyDesc->GetLengths();

                auto outer_size =
                    std::accumulate(dims.begin(), dims.end() - 1, 1ULL, std::multiplies<size_t>());

                auto inner_size = dims[dims.size() - 1];

                auto elapsed = 0.f;
                HipEventPtr start;
                HipEventPtr stop;

                if(handle_.IsProfilingEnabled())
                {
                    start = miopen::make_hip_event();
                    stop  = miopen::make_hip_event();
                    hipEventRecord(start.get(), handle_.GetStream());
                }

                kernel(params.dy,
                       params.x,
                       params.weight,
                       params.rstd,
                       params.dx,
                       inner_size,
                       static_cast<int32_t>(params.mode));

                weight_kernel(params.dy, params.x, params.rstd, params.dw, outer_size, inner_size);

                if(handle_.IsProfilingEnabled())
                {
                    hipEventRecord(stop.get(), handle_.GetStream());
                    hipEventSynchronize(stop.get());
                    hipEventElapsedTime(&elapsed, start.get(), stop.get());
                    handle_.ResetKernelTime();
                    handle_.AccumKernelTime(elapsed);
                };
            };
        };
    }

    return result;
}

std::size_t
T5LayernormBackward::GetWorkspaceSize(const ExecutionContext& context,
                                      const miopen::layernorm::ProblemDescription& problem) const
{
    auto dims = problem.GetDYDesc().GetLengths();

    auto outer_size =
        std::accumulate(dims.begin(), dims.end() - 1, 1ULL, std::multiplies<size_t>());

    auto inner_size = dims[dims.size() - 1];

    auto reqd_work_item_cnt = get_reqd_work_item_cnt(context);

    if(is_parallelism(reqd_work_item_cnt, inner_size, outer_size))
    {
        auto parallelism_size = get_parallelism_size(reqd_work_item_cnt, inner_size, outer_size);

        return parallelism_size * inner_size * get_data_size(problem.GetXDesc().GetType());
    }

    return 0;
}

} // namespace layernorm

} // namespace solver

} // namespace miopen
