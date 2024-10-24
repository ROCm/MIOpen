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
    // std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetInputDesc().GetType();
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

    std::cout << "input_dtype: " << input_dtype << ", i_dtype: " << i_dtype << std::endl;

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
                    {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                    {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                    {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                    {"MIOPEN_USE_INT8", static_cast<int>(dtype == miopenInt8)},
                    {"MIOPEN_USE_INT32", static_cast<int>(dtype == miopenInt32)},
                    // {}
                };

                kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

                kernel.l_wk.push_back(xlocalsize);
                kernel.l_wk.push_back(ylocalsize);
                kernel.l_wk.push_back(zlocalsize);

                kernel.g_wk.push_back(xgridsize);
                kernel.g_wk.push_back(ygridsize);
                kernel.g_wk.push_back(zgridsize);

                result.construction_params.push_back(kernel);
                // N = DivCeil(N, LOCAL_SIZE);
                N = AlignUp(N, LOCAL_SIZE) / LOCAL_SIZE;
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
                {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                {"MIOPEN_USE_INT8", static_cast<int>(dtype == miopenInt8)},
                {"MIOPEN_USE_INT32", static_cast<int>(dtype == miopenInt32)},
            };

            kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

            kernel.l_wk.push_back(xlocalsize);
            kernel.l_wk.push_back(ylocalsize);
            kernel.l_wk.push_back(zlocalsize);

            kernel.g_wk.push_back(xgridsize);
            kernel.g_wk.push_back(ygridsize);
            kernel.g_wk.push_back(zgridsize);

            // Print local work size
            printf("local_size(x, y, z) = (%d, %d, %d)\n", xlocalsize, ylocalsize, zlocalsize);
            // Print global work size
            printf("grid_size(x, y, z) = (%d, %d, %d)\n", xgridsize, ygridsize, zgridsize);

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
                auto scratch_mem = getBuffPart(params.GetWorkspace(), 0);
                auto local_mem   = getBuffPart(params.GetWorkspace(), 1);

                auto input_mem = params.input;

                auto input_tv = get_inner_expanded_tv<5>(deref(params.inputDesc));

                tensor_view_t<5> output_tv;
                for(int i = 0; i < 5; i++)
                {
                    output_tv.size[i]   = 1;
                    output_tv.stride[i] = 1;
                }

                auto N = input_numel;

                int kernelCnt = 0;
                for(kernelCnt; kernelCnt < kernels.size() - 1; ++kernelCnt)
                {
                    std::cout << "warp reduce" << std::endl;
                    decltype(auto) kernel = handle_.Run(kernels[kernelCnt]);
                    kernel(input_mem, scratch_mem, local_mem, N, input_tv, output_tv);
                    input_mem = scratch_mem;
                    output_tv = input_tv;
                }

                /* Last Reduction */
                {
                    output_tv             = get_inner_expanded_tv<5>(deref(params.outputDesc));
                    decltype(auto) kernel = handle_.Run(kernels[kernelCnt]);
                    kernel(input_mem, params.output, local_mem, N, input_tv, output_tv);
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
}

} // namespace any

} // namespace solver

} // namespace miopen
