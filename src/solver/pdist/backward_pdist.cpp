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
#include "miopen/tensor_view_utils.hpp"
#include <miopen/buffer_info.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/datatype.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/miopen.h>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/pdist.hpp>
#include <miopen/pdist/solvers.hpp>
#include <miopen/pdist/invoke_params.hpp>
#include <miopen/pdist/problem_description.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/reduce/utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace pdist {

MultiBufferWorkspaceTraits GetMultiBufferWorkspaceTraits(const TensorDescriptor& inputDesc)
{
    auto N = inputDesc.GetLengths()[0];
    auto M = inputDesc.GetLengths()[1];

    auto input_dtype = inputDesc.GetType();

    return MultiBufferWorkspaceTraits{(N - 1) * N * M * get_data_size(input_dtype)};
}

bool PdistBackward::IsApplicable(const ExecutionContext& context,
                                 const miopen::pdist::BackwardProblemDescription& problem) const
{
    std::ignore = context;

    if(!problem.IsAllContiguous())
    {
        return false;
    }

    if(!(problem.GetInputDesc().GetType() == miopenFloat ||
         problem.GetInputDesc().GetType() == miopenHalf ||
         problem.GetInputDesc().GetType() == miopenBFloat16))
    {
        return false;
    }

    return true;
}

ConvSolution
PdistBackward::GetSolution(const ExecutionContext& /* context */,
                           const miopen::pdist::BackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    // auto is_contiguous = problem.GetInputDesc().IsContiguous();

    auto dtype = problem.GetInputDesc().GetType();

    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto dinput_dtype = miopen::GetDataType(problem.GetdInputDesc().GetType());

    auto dinput_numel = problem.GetdInputDesc().GetElementSize();

    auto input_lengths  = problem.GetInputDesc().GetLengths();
    auto output_lengths = problem.GetOutputDesc().GetLengths();
    auto dinput_dims    = problem.GetdInputDesc().GetLengths();

    auto N  = input_lengths[0];
    auto NO = N * (N - 1) / 2;
    auto M  = input_lengths[1];

    if(problem.IsAllContiguous())
    {
        // Start building result.construction_params

        /* Phrase 1: Calculate gradients for each pair of elements in the input tensor */
        {
            size_t xlocalsize = LOCAL_SIZE;
            size_t xgridsize  = AlignUp(NO * M, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel        = KernelInfo{};
            kernel.kernel_file = "MIOpenPdist.cpp";
            kernel.kernel_name = "PdistBackwardContiguous";

            auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
                {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
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

        /* Phrase 2: Accumulate gradients for each element in the input tensor */
        // operation: sum
        // input: ws_dinput (shape=[N-1,N,M])
        // reduce_dim: 0
        // output: dinput (shape=[N,M])
        {
            size_t xlocalsize = LOCAL_SIZE;
            size_t xgridsize  = AlignUp(dinput_numel, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel = KernelInfo{};

            kernel.kernel_file = "MIOpenReduceCalculation.cpp";
            kernel.kernel_name = "CalculationFwdContiguous";

            const auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
                {"OUTPUT_TYPE", dinput_dtype == "bfloat16" ? "ushort" : dinput_dtype},
                {"OP_TYPE", "ReduceCalculationOp_t::Sum"},
                {"MIOPEN_REDUCE_CALCULATION_PROD", MIOPEN_REDUCE_CALCULATION_PROD},
                {"MIOPEN_REDUCE_CALCULATION_SUM", MIOPEN_REDUCE_CALCULATION_SUM}};

            kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

            kernel.l_wk.push_back(xlocalsize);
            kernel.l_wk.push_back(ylocalsize);
            kernel.l_wk.push_back(zlocalsize);

            kernel.g_wk.push_back(xgridsize);
            kernel.g_wk.push_back(ygridsize);
            kernel.g_wk.push_back(zgridsize);

            result.construction_params.push_back(kernel);
        }
        // End building result.construction_params

        auto getBuffPart = [ws = GetMultiBufferWorkspaceTraits(problem.GetdInputDesc())](
                               void* buffer, size_t part_idx) {
            return static_cast<void*>(static_cast<std::byte*>(buffer) + ws.GetOffset(part_idx));
        };

        // Start building result.invoker_factory
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) params = raw_params.CastTo<miopen::pdist::BackwardInvokeParams>();

                double p       = params.p;
                auto ws_dinput = getBuffPart(params.GetWorkspace(), 0);

                // Prepare some constants
                double n2                 = N - 0.5;
                double n2_squared_minus_1 = n2 * n2 - 1;

                HipEventPtr start, stop;
                bool profiling = handle_.IsProfilingEnabled();
                if(profiling)
                {
                    handle_.EnableProfiling(false);
                    hipStreamSynchronize(handle_.GetStream());
                    start = miopen::make_hip_event();
                    stop  = miopen::make_hip_event();
                    hipEventRecord(start.get(), handle_.GetStream());
                }

                /* Phrase 1: Calculate gradients for each pair of elements in the input tensor */
                {
                    decltype(auto) kernel = handle_.Run(kernels[0]);
                    kernel(params.input,
                           params.output,
                           params.doutput,
                           ws_dinput,
                           p,
                           n2,
                           n2_squared_minus_1,
                           N,
                           NO,
                           M);
                }

                /* Phrase 2: Accumulate gradients for each element in the input tensor */
                // operation: sum
                // input: ws_dinput
                // reduce_dim: 0
                // output: dinput
                {
                    decltype(auto) kernel = handle_.Run(kernels[1]);

                    kernel(ws_dinput,
                           params.dinput,
                           dinput_numel, // output numel
                           N - 1,        // reduce_size
                           N * M,        // inner_size
                           true          // Default nanPropagation=True
                    );
                }

                if(profiling)
                {
                    float elapsed = 0.0f;
                    hipEventRecord(stop.get(), handle_.GetStream());
                    handle_.EnableProfiling(true);
                    hipEventSynchronize(stop.get());
                    hipEventElapsedTime(&elapsed, start.get(), stop.get());

                    // Clean up
                    hipEventDestroy(start.get());
                    hipEventDestroy(stop.get());
                    handle_.ResetKernelTime();
                    handle_.AccumKernelTime(elapsed);
                };
            };
        };
    }
    else
    {
        // Start building result.construction_params

        /* Phrase 1: Calculate gradients for each pair of elements in the input tensor */
        {
            size_t xlocalsize = LOCAL_SIZE;
            size_t xgridsize  = AlignUp(NO * M, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel        = KernelInfo{};
            kernel.kernel_file = "MIOpenPdist.cpp";
            kernel.kernel_name = "PdistBackward";

            auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
                {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
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

        /* Phrase 2: Accumulate gradients for each element in the input tensor */
        // operation: sum
        // input: ws_dinput (shape=[N-1,N,M])
        // reduce_dim: 0
        // output: dinput (shape=[N,M])
        {
            size_t xlocalsize = LOCAL_SIZE;
            size_t xgridsize  = AlignUp(dinput_numel, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel = KernelInfo{};

            kernel.kernel_file = "MIOpenSumForward.cpp";
            kernel.kernel_name = "Sum1dForward";

            const auto build_params = KernelBuildParameters{
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
                {"OUTPUT_TYPE", dinput_dtype == "bfloat16" ? "ushort" : dinput_dtype},
                {"IN_VIEW_DIMS", 3},
                {"OUT_VIEW_DIMS", 2},
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
        // End building result.construction_params

        auto getBuffPart = [ws = GetMultiBufferWorkspaceTraits(problem.GetdInputDesc())](
                               void* buffer, size_t part_idx) {
            return static_cast<void*>(static_cast<std::byte*>(buffer) + ws.GetOffset(part_idx));
        };

        // Start building result.invoker_factory
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) params = raw_params.CastTo<miopen::pdist::BackwardInvokeParams>();

                const auto input_tv   = get_inner_expanded_tv<2>(deref(params.inputDesc));
                const auto output_tv  = get_inner_expanded_tv<1>(deref(params.outputDesc));
                const auto doutput_tv = get_inner_expanded_tv<1>(deref(params.doutputDesc));
                const auto dinput_tv  = get_inner_expanded_tv<2>(deref(params.dinputDesc));
                tensor_view_t<3> ws_dinput_tv{{N * M, M, 1}, {N - 1, N, M}};

                double p       = params.p;
                auto ws_dinput = getBuffPart(params.GetWorkspace(), 0);

                // Prepare some constants
                double n2                 = N - 0.5;
                double n2_squared_minus_1 = n2 * n2 - 1;

                HipEventPtr start, stop;
                bool profiling = handle_.IsProfilingEnabled();
                if(profiling)
                {
                    handle_.EnableProfiling(false);
                    hipStreamSynchronize(handle_.GetStream());
                    start = miopen::make_hip_event();
                    stop  = miopen::make_hip_event();
                    hipEventRecord(start.get(), handle_.GetStream());
                }

                /* Phrase 1: Calculate gradients for each pair of elements in the input tensor */
                {
                    decltype(auto) kernel = handle_.Run(kernels[0]);
                    kernel(params.input,
                           params.output,
                           params.doutput,
                           ws_dinput,
                           p,
                           n2,
                           n2_squared_minus_1,
                           input_tv,
                           output_tv,
                           doutput_tv);
                }

                /* Phrase 2: Accumulate gradients for each element in the input tensor */
                // operation: sum
                // input: ws_dinput
                // reduce_dim: 0
                // output: dinput
                {
                    decltype(auto) kernel = handle_.Run(kernels[1]);

                    kernel(ws_dinput,
                           params.dinput,
                           dinput_numel,
                           N - 1,                    // reduce_size
                           N * M,                    // inner_size
                           static_cast<uint64_t>(0), // reduce_dim
                           true,                     // Default nanPropagation=True,
                           ws_dinput_tv,
                           dinput_tv);
                }

                if(profiling)
                {
                    float elapsed = 0.0f;
                    hipEventRecord(stop.get(), handle_.GetStream());
                    handle_.EnableProfiling(true);
                    hipEventSynchronize(stop.get());
                    hipEventElapsedTime(&elapsed, start.get(), stop.get());

                    // Clean up
                    hipEventDestroy(start.get());
                    hipEventDestroy(stop.get());
                    handle_.ResetKernelTime();
                    handle_.AccumKernelTime(elapsed);
                };
            };
        };
    }

    return result;
}

std::size_t
PdistBackward::GetWorkspaceSize(const ExecutionContext& /*context*/,
                                const miopen::pdist::BackwardProblemDescription& problem) const
{
    return GetMultiBufferWorkspaceTraits(problem.GetdInputDesc()).GetSize();
}

} // namespace pdist

} // namespace solver

} // namespace miopen
