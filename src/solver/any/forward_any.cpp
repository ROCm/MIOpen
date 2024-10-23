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
#include <cstddef>
#include <cstdint>

#include <vector>

#include <miopen/any.hpp>
#include <miopen/any/solvers.hpp>
#include <miopen/any/invoke_params.hpp>
#include "miopen/any/problem_description.hpp"

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
#include "miopen/tensor_view_utils.hpp"

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace any {

constexpr uint64_t DivCeil(uint64_t numer, uint64_t denom) { return (numer + denom - 1) / denom; }

MultiBufferWorkspaceTraits GetMultiBufferWorkspaceTraits(const TensorDescriptor& inputDesc)
{
    auto input_numel = inputDesc.GetElementSize();
    auto size        = ((input_numel + LOCAL_SIZE - 1) / LOCAL_SIZE);

    auto dtype = inputDesc.GetType();
    size *= get_data_size(dtype);
    size_t data_size      = get_data_size(dtype);
    size_t workspace_size = AlignUp(size, LOCAL_SIZE) / LOCAL_SIZE;
    size_t ws_scratch_mem = 2 * workspace_size * data_size;
    size_t ws_local_mem   = LOCAL_SIZE * data_size;

    return MultiBufferWorkspaceTraits{ws_scratch_mem, ws_local_mem};
}

bool AnyForward::IsApplicable(const ExecutionContext& context,
                              const miopen::any::ProblemDescription& problem) const
{
    // std::ignore = context;

    // if(!problem.IsAllPacked())
    // {
    //     return false;
    // }

    return true;
}

ConvSolution AnyForward::GetSolution(const ExecutionContext& context,
                                     const miopen::any::ProblemDescription& problem) const
{
    // NOTE(anhduong): What's this for?
    // std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto input_dtype  = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetOutputDesc().GetType());

    auto input_dims  = problem.GetInputDesc().GetLengths();
    auto output_dims = problem.GetOutputDesc().GetLengths();
    auto dim         = problem.GetDim();
    auto keep_dim    = problem.GetKeepDim();

    auto input_numel  = problem.GetInputDesc().GetElementSize();
    auto output_numel = problem.GetOutputDesc().GetElementSize();

    std::string i_dtype = input_dtype;
    std::string o_dtype = output_dtype;

    if(input_dtype == "int8_t")
    {
        i_dtype = "char";
    }
    else if(input_dtype == "bfloat16")
    {
        i_dtype = "ushort";
    }

    if(dim != -1)
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenAny.cpp";
        kernel.kernel_name = "AnyForward";

        // MIOpen doesn't support for bool so I have to use char instead
        auto build_params = KernelBuildParameters{
            {"INPUT_TYPE", i_dtype},
            {"OUTPUT_TYPE", "char"},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);

        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                auto params           = raw_params.CastTo<miopen::any::InvokeParams>();

                auto input_tv  = get_inner_expanded_tv<5>(deref(params.inputDesc));
                auto output_tv = get_inner_expanded_tv<5>(deref(params.outputDesc));

                auto N    = output_numel;
                auto K    = problem.GetOutputDesc().GetLengths()[dim];
                size_t st = 1;
                for(int i = dim + 1; i < output_dims.size(); i++)
                {
                    st *= output_dims[i];
                }

                kernel(params.input,
                       params.output,
                       //    params.workspace,
                       N,
                       K,
                       st,
                       dim,
                       input_tv,
                       output_tv);
            };
        };
    }
    else
    {
        // Start building result.construction_params
        auto N = input_numel;
        {
            /* Any Reduction */
            while(N > LOCAL_SIZE)
            {
                size_t xlocalsize = LOCAL_SIZE;
                size_t xgridsize  = AlignUp(N, xlocalsize);
                size_t ylocalsize = 1;
                size_t ygridsize  = 1;
                size_t zlocalsize = 1;
                size_t zgridsize  = 1;

                auto kernel        = KernelInfo{};
                kernel.kernel_file = "MIOpenAny.cpp";
                kernel.kernel_name = "ReduceAny";

                auto build_params = KernelBuildParameters{
                    {"INPUT_TYPE", i_dtype},
                    {"OUTPUT_TYPE", "char"},
                };

                kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

                kernel.l_wk.push_back(xlocalsize);
                kernel.l_wk.push_back(ylocalsize);
                kernel.l_wk.push_back(zlocalsize);

                kernel.g_wk.push_back(xgridsize);
                kernel.g_wk.push_back(ygridsize);
                kernel.g_wk.push_back(zgridsize);

                result.construction_params.push_back(kernel);
                N = DivCeil(N, LOCAL_SIZE);
            }
        }

        // Last Reduction
        {
            size_t xlocalsize = LOCAL_SIZE;
            size_t xgridsize  = AlignUp(N, xlocalsize);
            size_t ylocalsize = 1;
            size_t ygridsize  = 1;
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            auto kernel        = KernelInfo{};
            kernel.kernel_file = "MIOpenAny.cpp";
            kernel.kernel_name = "ReduceAny";

            auto build_params = KernelBuildParameters{
                {"INPUT_TYPE", i_dtype},
                {"OUTPUT_TYPE", "char"},
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

        auto getBuffPart = [ws = GetMultiBufferWorkspaceTraits(problem.GetInputDesc())](
                               void* buffer, size_t part_idx) {
            return static_cast<void*>(static_cast<std::byte*>(buffer) + ws.GetOffset(part_idx));
        };

        // Start building result.invoker_factory
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                // decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::any::InvokeParams>();

                /* Any Reduction */
                auto size      = deref(params.inputDesc).GetElementSize();
                auto data_size = get_data_size(miopenFloat);
                // auto wt        = MultiBufferWorkspaceTraits{
                //     size * data_size, (size + LOCAL_SIZE - 1) / LOCAL_SIZE * data_size};
                // auto reduce_in = params.workspace;
                // auto reduce_out =
                //     static_cast<Data_t>(static_cast<std::byte*>(params.workspace) +
                //     wt.GetOffset(1));

                // auto wt =
                // auto getBuffPart = [ws = GetMultiBufferWorkspaceTraits(problem.GetInputDesc())](
                //                        void* buffer, size_t part_idx) {
                //     return static_cast<void*>(static_cast<std::byte*>(buffer) +
                //                               ws.GetOffset(part_idx));
                // };

                auto scratch_mem = getBuffPart(params.GetWorkspace(), 0);
                auto local_mem   = getBuffPart(params.GetWorkspace(), 1);

                auto reduce_in = params.input;
                // auto reduce_out = static_cast<Data_t>(static_cast<std::byte*>(params.workspace) +
                //                                       wt.GetOffset(0));

                auto input_tv = get_inner_expanded_tv<5>(deref(params.inputDesc));
                // auto output_tv = get_inner_expanded_tv<5>(deref(params.outputDesc));

                tensor_view_t<5> output_tv;
                for(int i = 0; i < 5; i++)
                {
                    output_tv.size[i]   = 1;
                    output_tv.stride[i] = 1;
                }
                // output_tv.off

                auto N = input_numel;

                // int kernelCnt = 0;
                // for(int i = 0; i < kernelCnt; i++) {

                // }
                int kernelCnt = 0;
                for(kernelCnt; kernelCnt < kernels.size() - 1; ++kernelCnt)
                {
                    decltype(auto) kernel = handle_.Run(kernels[kernelCnt]);
                    kernel(reduce_in,
                           scratch_mem, // scratch_mem
                           local_mem,
                           //    params.workspace,
                           N,
                           //    K,
                           //    st,
                           //    dim,
                           input_tv,
                           output_tv);
                    // kernel()

                    reduce_in = scratch_mem;
                    output_tv = input_tv;
                }

                /* Last Reduction */
                {
                    output_tv             = get_inner_expanded_tv<5>(deref(params.outputDesc));
                    decltype(auto) kernel = handle_.Run(kernels[kernelCnt]);
                    kernel(scratch_mem, params.output, local_mem, N, input_tv, output_tv);
                }
            };
        };
        // End building result.invoker_factory
    }

    return result;
}

std::size_t AnyForward::GetWorkspaceSize(const ExecutionContext& /*context*/,
                                         const miopen::any::ProblemDescription& problem) const
{
    if(problem.GetDim() != -1)
    {
        return 0;
    }

    return GetMultiBufferWorkspaceTraits(problem.GetInputDesc()).GetSize();

    // auto input_numel = problem.GetInputDesc().GetElementSize();
    // auto size        = ((input_numel + LOCAL_SIZE - 1) / LOCAL_SIZE);

    // auto dtype = problem.GetInputDesc().GetType();
    // size *= get_data_size(dtype);
    // size_t data_size      = get_data_size(dtype);
    // size_t workspace_size = AlignUp(size, LOCAL_SIZE) / LOCAL_SIZE;
    // size_t ws_scratch_mem = 2 * workspace_size * data_size;
    // size_t ws_local_mem = LOCAL_SIZE * data_size;

    // // Input and output memories are swapped in each iteration,
    // // require 2 * workspace_size. Read AnyReduce implementation for more
    // // information.
    // // return 2 * workspace_size * data_size;
    // return MultiBufferWorkspaceTraits
}

} // namespace any

} // namespace solver

} // namespace miopen
