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

#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/solver.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD)

namespace miopen {
namespace solver {

bool ConvDirectNaiveConvFwd::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!miopen::debug::AlwaysEnableConvDirectNaive &&
       miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD{}))
        return false;

    if(!ConvDirectNaiveConvIsApplicableByKernelType(ctx))
        return false;

    if(!ctx.problem.IsLayoutDefault() && !ctx.problem.IsLayoutNHWC())
        return false;

    if(!(ctx.problem.IsFp32() || ctx.problem.IsFp16() || ctx.problem.IsBfp16() || ctx.problem.IsInt8()))
        return false;

    if(!ctx.problem.direction.IsForward())
        return false;

    return true;
}

ConvSolution ConvDirectNaiveConvFwd::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution result;

    int di          = ctx.problem.in_depth;
    int hi          = ctx.problem.in_height;
    int wi          = ctx.problem.in_width;
    int n           = ctx.problem.batch_sz;
    int k           = ctx.problem.n_outputs;
    int c           = ctx.problem.n_inputs;
    int do_         = ctx.problem.out_depth;
    int ho          = ctx.problem.out_height;
    int wo          = ctx.problem.out_width;
    int sz          = ctx.problem.kernel_stride_d;
    int sy          = ctx.problem.kernel_stride_h;
    int sx          = ctx.problem.kernel_stride_w;
    int dz          = ctx.problem.kernel_dilation_d;
    int dy          = ctx.problem.kernel_dilation_h;
    int dx          = ctx.problem.kernel_dilation_w;
    int pz          = ctx.problem.pad_d;
    int py          = ctx.problem.pad_h;
    int px          = ctx.problem.pad_w;
    int fz          = ctx.problem.kernel_size_d;
    int fy          = ctx.problem.kernel_size_h;
    int fx          = ctx.problem.kernel_size_w;
    int group       = ctx.problem.group_counts;
    int c_per_group = c / group;
    int k_per_group = k / group;

    size_t block_size = 256;
    size_t grid_size  = 1;
    if(ctx.problem.IsLayoutDefault())
    {
        grid_size = static_cast<size_t>(n) * k;
    }
    else if(ctx.problem.IsLayoutNHWC())
    {
        if(ctx.problem.Is2d())
            grid_size = static_cast<size_t>(group) * n * ho;
        else
            grid_size = static_cast<size_t>(group) * n * do_;
    }
    else
        MIOPEN_THROW("Unsupported layout");

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile();
    kernel.kernel_name = ConvDirectNaiveConvKernelName(ctx);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx);

    if(ctx.problem.Is2d())
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            const auto kern = kernels[0];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
                const auto& tensors     = data_ctx.tensors;
                float elapsed           = 0;

                handle.Run(kern)(tensors.in,
                                 tensors.w,
                                 tensors.out,
                                 hi,
                                 wi,
                                 n,
                                 k_per_group,
                                 c_per_group,
                                 ho,
                                 wo,
                                 sy,
                                 sx,
                                 dy,
                                 dx,
                                 py,
                                 px,
                                 fy,
                                 fx,
                                 group);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
    else
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            const auto kern = kernels[0];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
                const auto& tensors     = data_ctx.tensors;
                float elapsed           = 0;

                handle.Run(kern)(tensors.in,
                                 tensors.w,
                                 tensors.out,
                                 di,
                                 hi,
                                 wi,
                                 n,
                                 k_per_group,
                                 c_per_group,
                                 do_,
                                 ho,
                                 wo,
                                 sz,
                                 sy,
                                 sx,
                                 dz,
                                 dy,
                                 dx,
                                 pz,
                                 py,
                                 px,
                                 fz,
                                 fy,
                                 fx,
                                 group);
                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
    result.construction_params.push_back(kernel);
    return result;
}

} // namespace solver
} // namespace miopen
