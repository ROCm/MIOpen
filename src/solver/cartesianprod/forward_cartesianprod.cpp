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

#include "miopen/miopen.h"
#include <cstdio>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/cartesianprod/solvers.hpp>

#include <miopen/cartesianprod/invoke_params.hpp>
#include <miopen/datatype.hpp>
#include <miopen/cartesianprod.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/par_for.hpp>

#define LOCAL_SIZE_FWD 128
#define TILE_SIZE 16

namespace miopen {

namespace solver {

namespace cartesianprod {

bool CartesianProdForward::IsApplicable(
    const ExecutionContext&, const miopen::cartesianprod::FwdProblemDescription& problem) const
{
    if(!(problem.GetOutputDesc().GetType() == miopenHalf ||
         problem.GetOutputDesc().GetType() == miopenFloat))
    {
        return false;
    }
    if(!problem.IsAllPacked())
    {
        return false;
    }
    return true;
}

ConvSolution
CartesianProdForward::GetSolution(const ExecutionContext& context,
                                  const miopen::cartesianprod::FwdProblemDescription& problem) const
{
    std::ignore       = context;
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());
    auto dtype        = problem.GetOutputDesc().GetType();
    auto inputCount   = problem.GetInputCount();

    auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                              {"TILE_SIZE", TILE_SIZE},
                              {"D_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype}};

    if(inputCount == 1)
    {
        return ConvSolution{miopenStatusNotImplemented};
    }
    auto output_dims = problem.GetOutputDesc().GetLengths();
    auto result      = ConvSolution{miopenStatusSuccess};
    result.construction_params.push_back(make_hip_kernel({LOCAL_SIZE_FWD},
                                                         {output_dims[0]},
                                                         "MIOpenCartesianProd.cpp",
                                                         "CartesianProdForward",
                                                         build_params));
    result.construction_params.push_back(make_hip_kernel({TILE_SIZE, TILE_SIZE},
                                                         {output_dims[0], output_dims[1]},
                                                         "MIOpenCartesianProd.cpp",
                                                         "CartesianProdTranspose",
                                                         build_params));

    result.invoker_factory = [inputCount](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            size_t stride = 1;
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

            decltype(auto) kernel = handle_.Run(kernels[0]);
            decltype(auto) params = raw_params.CastTo<miopen::cartesianprod::FwdInvokeParams>();
            auto output_tv        = get_inner_expanded_tv<2>(deref(params.outputDesc));

            par_for(inputCount, [&](auto i) {
                i = inputCount - 1 - i;
                auto input_tv =
                    get_inner_expanded_tv<1>(*params.GetInputDesc(static_cast<size_t>(i)));
                kernel(params.GetInput(static_cast<size_t>(i)),
                       params.workspace,
                       input_tv,
                       output_tv,
                       stride,
                       static_cast<size_t>(i));
                stride *= params.GetInputDesc(static_cast<size_t>(i))->GetElementSize();
            });
            kernel = handle_.Run(kernels[1]);
            kernel(params.workspace, params.output, output_tv);

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
};

std::size_t CartesianProdForward::GetWorkspaceSize(
    const ExecutionContext& /*context*/,
    const miopen::cartesianprod::FwdProblemDescription& problem) const
{
    return problem.GetOutputDesc().GetElementSize() *
           get_data_size(problem.GetOutputDesc().GetType());
}

} // namespace cartesianprod

} // namespace solver

} // namespace miopen
