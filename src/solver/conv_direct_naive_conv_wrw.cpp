/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <miopen/solver.hpp>
#include <miopen/env.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/conv/invokers/naive_conv.hpp>
#include <miopen/conv/conv_direct_naive_conv.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_WRW)

namespace miopen {
namespace solver {

bool ConvDirectNaiveConvWrw::IsApplicable(const ConvolutionContext& ctx) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_WRW{}))
        return false;

    if(!ctx.IsLayoutDefault())
    {
        return false;
    }

    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;

    if(!ctx.direction.IsBackwardWrW())
    {
        return false;
    }

    return true;
}

ConvSolution ConvDirectNaiveConvWrw::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution result;

    std::string kernel_name = ConvDirectNaiveConvKernelName(ctx);

    int block_size = 256;
    int grid_size  = ctx.n_inputs; // k

    KernelInfo kernel;

    kernel.kernel_file = "naive_conv.cpp";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    /* Note here, for API like hipHccModuleLaunchKernel(), hipExtModuleLaunchKernel()
    * grid dims is in unit of work item.
    * But for api like hipModuleLaunchKernel(), grid dim is in unit of block.
    */
    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.comp_options = ctx.general_compile_options;

    MIOPEN_LOG_I2(kernel.kernel_file + ":" + kernel.kernel_name);

    result.invoker_factory = conv::MakeNaiveConvWrwInvokerFactory(ctx);
    result.construction_params.push_back(kernel);
    return result;
}

} // namespace solver
} // namespace miopen
