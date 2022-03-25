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

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD)

namespace miopen {
namespace solver {

bool ConvDirectNaiveConvBwd::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!miopen::debug::AlwaysEnableConvDirectNaive &&
       miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD{}))
        return false;

    if(!ConvDirectNaiveConvIsApplicableByKernelType(ctx))
        return false;

    if(!ctx.IsLayoutDefault() && !ctx.IsLayoutNHWC())
        return false;

    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;

    if(!ctx.direction.IsBackwardData())
        return false;

    return true;
}

ConvSolution ConvDirectNaiveConvBwd::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution result;

    int di          = ctx.out_depth;
    int hi          = ctx.out_height;
    int wi          = ctx.out_width;
    int n           = ctx.batch_sz;
    int k           = ctx.n_inputs;
    int c           = ctx.n_outputs;
    int do_         = ctx.in_depth;
    int ho          = ctx.in_height;
    int wo          = ctx.in_width;
    int sz          = ctx.in_depth > 1 ? ctx.kernel_stride_d : 1;
    int sy          = ctx.in_height > 1 ? ctx.kernel_stride_h : 1;
    int sx          = ctx.in_width > 1 ? ctx.kernel_stride_w : 1;
    int dz          = ctx.kernel_size_d > 1 ? ctx.kernel_dilation_d : 1;
    int dy          = ctx.kernel_size_h > 1 ? ctx.kernel_dilation_h : 1;
    int dx          = ctx.kernel_size_w > 1 ? ctx.kernel_dilation_w : 1;
    int pz          = ctx.pad_d;
    int py          = ctx.pad_h;
    int px          = ctx.pad_w;
    int fz          = ctx.kernel_size_d;
    int fy          = ctx.kernel_size_h;
    int fx          = ctx.kernel_size_w;
    int group       = ctx.group_counts;
    int c_per_group = c / group;
    int k_per_group = k / group;

    size_t block_size = 256;
    size_t grid_size  = 1;
    if(ctx.IsLayoutDefault())
    {
        grid_size = static_cast<size_t>(n) * c;
    }
    else if(ctx.IsLayoutNHWC())
    {
        if(ctx.Is2d())
            grid_size = static_cast<size_t>(group) * n * hi;
        else
            grid_size = static_cast<size_t>(group) * n * di;
    }
    else
        MIOPEN_THROW("Unsupported layout");

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx);
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

    if(ctx.Is2d())
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            const auto kern = kernels[0];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
                const auto& tensors     = data_ctx.tensors;
                float elapsed           = 0;

                handle.Run(kern)(tensors.out,
                                 tensors.w,
                                 tensors.in,
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

                handle.Run(kern)(tensors.out,
                                 tensors.w,
                                 tensors.in,
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
