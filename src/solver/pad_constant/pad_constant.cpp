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

#include <miopen/conv_solution.hpp>
#include <miopen/datatype.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/pad_constant.hpp>
#include <miopen/pad_constant/invoke_params.hpp>
#include <miopen/pad_constant/problem_description.hpp>
#include <miopen/pad_constant/solvers.hpp>
#include <miopen/tensor_view_utils.hpp>

#include "../src/kernels/MIOpenPadConstant.hpp"

#define PAD_CONSTANT_LOCAL_SIZE 256

namespace miopen {
namespace solver {
namespace pad_constant_fwd {
bool PadConstantFwd::IsApplicable(const ExecutionContext& /*context*/,
                                  const miopen::pad_constant_fwd::ProblemDescription& problem) const
{
    if(!problem.IsImprovementOverROCm())
        return false;

    return true;
}

ConvSolution
PadConstantFwd::GetSolution(const ExecutionContext& /*context*/,
                            const miopen::pad_constant_fwd::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto input_dtype  = miopen::GetDataType(problem.GetXDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetYDesc().GetType());

    auto dtype    = problem.GetXDesc().GetType();
    auto io_dtype = miopen::GetDataType(dtype);

    size_t output_size = problem.GetYDesc().GetElementSize();

    size_t xlocalsize = PAD_CONSTANT_LOCAL_SIZE;
    size_t xgridsize  = AlignUpUL(output_size, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenPadConstant.cpp";
    kernel.kernel_name = "PadConstantFwd";

    // auto test = problem.GetPadding();
    //  // print padding
    //  std::cout << "\n\ntest: ";
    //  for(auto i : test)
    //  {
    //      std::cout << i << " ";
    //  }
    //  std::cout << "\n\n";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);

    result.construction_params.push_back(kernel);

    // Prepare padding
    padding_5d_t padding_args = {};
    int io_dim_size           = problem.GetXDesc().GetNumDims();

    auto padding_size = problem.GetPaddingSize();
    auto padding      = problem.GetPadding();

    // print padding
    // std::cout << "padding: ";
    // for(auto i : padding)
    // {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;

    for(uint64_t i = 0; i < padding_size / 2; i++)
    {
        size_t idx                    = io_dim_size - i - 1;
        padding_args.val[idx * 2]     = padding[i * 2];
        padding_args.val[idx * 2 + 1] = padding[i * 2 + 1];
    }

    // print padding_args
    // std::cout << "padding_args: " << std::endl;
    // for(auto i : padding_args.val)
    // {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;

    // TODO: Handle cases where all paddings are zeros
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& invoke_params) {
            decltype(auto) params = invoke_params.CastTo<miopen::pad_constant_fwd::InvokeParams>();
            decltype(auto) kernel = handle_.Run(kernels[0]);

            auto input_tv  = get_inner_expanded_tv<5>(deref(params.xDesc));
            auto output_tv = get_inner_expanded_tv<5>(deref(params.yDesc));

            // // Prepare padding
            // padding_5d_t padding_args = {};
            // int io_dim_size           = problem.GetXDesc().GetNumDims();

            // auto padding_size = problem.GetPaddingSize();
            // auto padding      = problem.GetPadding();

            // // print padding
            // std::cout << "padding: ";
            // for(auto i : padding)
            // {
            //     std::cout << i << " ";
            // }
            // std::cout << std::endl;

            // for(uint64_t i = 0; i < padding_size / 2; i++)
            // {
            //     size_t idx                    = io_dim_size - i - 1;
            //     padding_args.val[idx * 2]     = padding[i * 2];
            //     padding_args.val[idx * 2 + 1] = padding[i * 2 + 1];
            // }

            // // print padding_args
            // std::cout << "padding_args: " << std::endl;
            // for(auto i : padding_args.val)
            // {
            //     std::cout << i << " ";
            // }
            // std::cout << std::endl;

            kernel(params.x,
                   params.y,
                   padding_args,
                   output_size,
                   params.padding_value,
                   input_tv,
                   output_tv);
        };
    };

    return result;
}
} // namespace pad_constant_fwd

namespace pad_constant_bwd {

bool PadConstantBwd::IsApplicable(const ExecutionContext& /*context*/,
                                  const miopen::pad_constant_bwd::ProblemDescription& problem) const
{
    if(!problem.IsImprovementOverROCm())
        return false;

    return true;
}

ConvSolution
PadConstantBwd::GetSolution(const ExecutionContext& /*context*/,
                            const miopen::pad_constant_bwd::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype    = problem.GetdXDesc().GetType();
    auto io_dtype = miopen::GetDataType(dtype);

    size_t input_grad_size = problem.GetdXDesc().GetElementSize();

    size_t xlocalsize = PAD_CONSTANT_LOCAL_SIZE;
    size_t xgridsize  = AlignUpUL(input_grad_size, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenPadConstant.cpp";
    kernel.kernel_name = "PadConstantBwd";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(ylocalsize);

    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(ygridsize);

    result.construction_params.push_back(kernel);

    // Prepare padding
    padding_5d_t padding_args = {};
    int io_dim_size           = problem.GetdXDesc().GetNumDims();

    auto padding_size = problem.GetPaddingSize();
    auto padding      = problem.GetPadding();

    for(uint64_t i = 0; i < padding_size / 2; i++)
    {
        size_t idx                    = io_dim_size - i - 1;
        padding_args.val[idx * 2]     = padding[i * 2];
        padding_args.val[idx * 2 + 1] = padding[i * 2 + 1];
    }

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& invoke_params) {
            decltype(auto) params = invoke_params.CastTo<miopen::pad_constant_bwd::InvokeParams>();
            decltype(auto) kernel = handle_.Run(kernels[0]);

            auto input_grad_tv  = get_inner_expanded_tv<5>(deref(params.dxDesc));
            auto output_grad_tv = get_inner_expanded_tv<5>(deref(params.dyDesc));

            kernel(
                params.dx, params.dy, padding_args, input_grad_size, input_grad_tv, output_grad_tv);
        };
    };

    return result;
}

} // namespace pad_constant_bwd
} // namespace solver
} // namespace miopen
