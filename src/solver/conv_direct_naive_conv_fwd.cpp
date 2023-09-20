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

bool ConvDirectNaiveConvFwd::IsApplicable(const ConvolutionContext& ctx,
                                          const ProblemDescription& problem) const
{
    if(!miopen::debug::AlwaysEnableConvDirectNaive &&
       miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD{}))
        return false;

    if(!ConvDirectNaiveConvIsApplicableByKernelType(ctx, problem))
        return false;

    if(!problem.IsLayoutDefault() && !problem.IsLayoutNHWC())
        return false;

    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16() || problem.IsInt8() ||
         problem.IsFp8() || problem.IsBfp8()))
        return false;

    if(!problem.direction.IsForward())
        return false;

    if(problem.IsTensorsCasted())
    {
        auto test_cast = [&](const TensorDescriptor& desc) {
            if(desc.GetCastType())
            {
                const auto cast_type = *desc.GetCastType();
                if(cast_type == miopenFloat8 || cast_type == miopenBFloat8)
                    return false;
            }
            // all tested tensors must have cast type set
            return true;
        };
        if(test_cast(problem.GetIn()))
            return false;
        if(test_cast(problem.GetWeights()))
            return false;
    }
    return true;
}

ConvSolution ConvDirectNaiveConvFwd::GetSolution(const ConvolutionContext& ctx,
                                                 const ProblemDescription& problem) const
{
    ConvSolution result;

    int di          = problem.GetInDepth_();
    int hi          = problem.GetInHeight_();
    int wi          = problem.GetInWidth_();
    int n           = problem.GetBatchSize_();
    int k           = problem.GetOutChannels_();
    int c           = problem.GetInChannels_();
    int do_         = problem.GetOutDepth_();
    int ho          = problem.GetOutHeight_();
    int wo          = problem.GetOutWidth_();
    int sz          = problem.GetKernelStrideD();
    int sy          = problem.GetKernelStrideH();
    int sx          = problem.GetKernelStrideW();
    int dz          = problem.GetDilationD();
    int dy          = problem.GetDilationH();
    int dx          = problem.GetDilationW();
    int pz          = problem.GetPadD();
    int py          = problem.GetPadH();
    int px          = problem.GetPadW();
    int fz          = problem.GetWeightsDepth_();
    int fy          = problem.GetWeightsHeight_();
    int fx          = problem.GetWeightsWidth_();
    int group       = problem.GetGroupCount();
    int c_per_group = c / group;
    int k_per_group = k / group;

    size_t block_size = 256;
    size_t grid_size  = 1;
    if(problem.IsLayoutDefault())
    {
        grid_size = static_cast<size_t>(n) * k;
    }
    else if(problem.IsLayoutNHWC())
    {
        if(problem.Is2d())
            grid_size = static_cast<size_t>(group) * n * ho;
        else
            grid_size = static_cast<size_t>(group) * n * do_;
    }
    else
        MIOPEN_THROW("Unsupported layout");

    KernelInfo kernel;

    kernel.kernel_file = ConvDirectNaiveConvKernelFile(ctx, problem);
    const auto is_f8   = [&]() {
        if(kernel.kernel_file == "fp8_naive_conv.cpp")
            return true;
        else
            return false;
    }();
    kernel.kernel_name = ConvDirectNaiveConvKernelName(problem);
    kernel.g_wk.clear();

    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.comp_options = ConvDirectNaiveConvCompileOption(ctx, problem);

    if(problem.Is2d())
        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            const auto kern = kernels[0];
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
                const auto& tensors     = data_ctx.tensors;
                float elapsed           = 0;
                if(is_f8)
                {
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
                                     group,
                                     problem.GetConv().attribute.fp8rounding_mode.Get() ==
                                         miopenF8RoundingModeStochastic,
                                     problem.GetConv().attribute.fp8rounding_mode.GetSeed());
                }
                else
                {
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
                }
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
