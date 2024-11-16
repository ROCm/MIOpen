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
// #include <cstddef>
// #include <cstdint>

#include <numeric>
#include <vector>

#include <miopen/pdist.hpp>
#include <miopen/pdist/solvers.hpp>
#include <miopen/pdist/invoke_params.hpp>
// #include "miopen/common.hpp"
#include "miopen/conv_solution.hpp"
#include "miopen/pdist/problem_description.hpp"

#include "miopen/miopen.h"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/target_properties.hpp>
#include "miopen/buffer_info.hpp"
#include "miopen/errors.hpp"
#include "miopen/execution_context.hpp"
#include "miopen/invoke_params.hpp"
#include "miopen/tensor.hpp"
#include "miopen/tensor_view_utils.hpp"

#include <miopen/reduce/utils.hpp>

#define LOCAL_SIZE 256

using namespace miopen::solver::reduce;

namespace miopen {

namespace solver {

namespace pdist {

MultiBufferWorkspaceTraits GetMultiBufferWorkspaceTraits(const TensorDescriptor& inputDesc)
{
    auto N = inputDesc.GetLengths()[0];
    auto M = inputDesc.GetLengths()[1];

    auto input_dtype = inputDesc.GetType();
    // std::size_t ws_dinput_size = (N - 1) * N * M * get_data_size(input_dtype);

    // std::size_t ws_

    // auto input_numel = inputDesc.GetElementSize();

    // auto dtype            = inputDesc.GetType();
    // size_t data_size      = get_data_size(dtype);
    // size_t workspace_size = AlignUp(input_numel, LOCAL_SIZE) / LOCAL_SIZE;
    // size_t ws_scratch_mem = 2 * workspace_size * data_size;
    // size_t ws_local_mem   = LOCAL_SIZE * data_size;

    return MultiBufferWorkspaceTraits{(N - 1) * N * M * get_data_size(input_dtype)};
}

bool PdistBackward::IsApplicable(const ExecutionContext& context,
                                 const miopen::pdist::BackwardProblemDescription& problem) const
{
    std::ignore = context;

    // problem.IsSameType();
    // problem.IsRightLength();
    // problem.IsAllContiguous();

    if(!problem.IsAllContiguous())
    {
        return false;
    }
    return true;
}

ConvSolution
PdistBackward::GetSolution(const ExecutionContext& context,
                           const miopen::pdist::BackwardProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype = problem.GetInputDesc().GetType();

    auto input_dtype  = miopen::GetDataType(problem.GetdInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetdOutputDesc().GetType());
    auto dinput_dtype = miopen::GetDataType(problem.GetdInputDesc().GetType());

    // auto dinput_numel = problem.GetdInputDesc().GetElementSize();
    auto dinput_numel = problem.GetdInputDesc().GetElementSize();

    auto input_lengths  = problem.GetInputDesc().GetLengths();
    auto output_lengths = problem.GetOutputDesc().GetLengths();
    auto dinput_dims    = problem.GetdInputDesc().GetLengths();

    auto N = input_lengths[0];
    auto M = input_lengths[1];

    // Start building result.construction_params

    /* Phrase 1: Calculate gradients for each pair of elements in the input tensor */
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_lengths[0] * input_lengths[1], xlocalsize);
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

        // TODO: Add paralellism for efficiency if needed
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
            {"OUTPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : output_dtype},
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

            auto ws_dinput = getBuffPart(params.GetWorkspace(), 0);

            /* Phrase 1: Calculate gradients for each pair of elements in the input tensor */
            {
                double n2                 = input_lengths[0] - 0.5;
                double n2_squared_minus_1 = n2 * n2 - 1;

                auto input_tv   = get_inner_expanded_tv<2>(deref(params.inputDesc));
                auto output_tv  = get_inner_expanded_tv<1>(deref(params.outputDesc));
                auto doutput_tv = get_inner_expanded_tv<1>(deref(params.doutputDesc));
                // auto dinput_tv  = get_inner_expanded_tv<2>(deref(params.dinputDesc));
                // auto ws_dinput_tv = get_inner_expanded_tv<3>()

                decltype(auto) kernel = handle_.Run(kernels[0]);
                double p              = params.p;
                // std::cout << "[Out kernel] p = " << p << std::endl;
                kernel(params.input,
                       params.output,
                       params.doutput,
                       ws_dinput,
                       p,
                       n2,
                       n2_squared_minus_1,
                       input_tv,
                       output_tv,
                       doutput_tv
                       //    dinput_tv
                );
            }

            /* Phrase 2: Accumulate gradients for each element in the input tensor */
            // operation: sum
            // input: ws_dinput
            // reduce_dim: 0
            // output: dinput
            {
                // TODO: Add paralellism for efficiency if needed
                decltype(auto) kernel = handle_.Run(kernels[1]);
                // uint64_t dim = 0;
                auto reduce_size = N - 1;

                // auto inner_size = N * M; // ws_dinput numel
                auto inner_size = N * M;

                // print ws_dinput
                // for()

                kernel(ws_dinput,
                       params.dinput,
                       dinput_numel, // output numel
                       reduce_size,
                       inner_size,
                       true // Set default nanPropagation=True
                );

                // print params.dinput
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
