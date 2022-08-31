/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#ifndef GUARD_GPU_CONV_HPP
#define GUARD_GPU_CONV_HPP

#include <miopen/convolution.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>

#include "get_handle.hpp"
#include "tensor_holder.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF)

template <typename Tin, typename Twei, typename Tout>
bool gpu_ref_convolution_fwd(const tensor<Tin>& input,
                             const tensor<Twei>& weights,
                             tensor<Tout>& rout,
                             miopen::ConvolutionDescriptor filter)
{
    bool gpu_ref_used = false;
    if(!miopen::IsEnabled(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF{}))
    {
        auto&& handle            = get_handle();
        auto in_dev              = handle.Write(input.data);
        auto wei_dev             = handle.Write(weights.data);
        auto out_dev             = handle.Write(rout.data);
        const auto naive_conv_id = miopen::solver::Id{"ConvDirectNaiveConvFwd"};
        const auto naive_solver  = naive_conv_id.GetSolver();

        const auto tensors = miopen::ConvFwdTensors{
            input.desc, in_dev.get(), weights.desc, wei_dev.get(), rout.desc, out_dev.get()};
        auto ctx = miopen::ConvolutionContext{
            input.desc, weights.desc, rout.desc, filter, miopen::conv::Direction::Forward};
        ctx.SetStream(&handle);
        ctx.DetectRocm();
        if(naive_solver.IsApplicable(ctx))
        {
            gpu_ref_used          = true;
            const auto invoke_ctx = miopen::conv::DataInvokeParams{
                tensors, nullptr, 0, filter.attribute.gfx90aFp16alt.GetFwd()};
            const auto invoker = miopen::LoadOrPrepareInvoker(
                handle, ctx, naive_conv_id.Value(), miopen::conv::Direction::Forward);
            invoker(handle, invoke_ctx);
            rout.data = handle.Read<Tout>(out_dev, rout.data.size());
        }
    }
    return gpu_ref_used;
}

template <typename Tin, typename Twei, typename Tout>
bool gpu_ref_convolution_bwd(tensor<Tin>& input,
                             const tensor<Twei>& weights,
                             const tensor<Tout> output,
                             miopen::ConvolutionDescriptor filter)
{
    bool gpu_ref_used = false;
    if(!miopen::IsEnabled(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF{}))
    {
        auto&& handle            = get_handle();
        auto in_dev              = handle.Write(input.data);
        auto wei_dev             = handle.Write(weights.data);
        auto out_dev             = handle.Write(output.data);
        const auto naive_conv_id = miopen::solver::Id{"ConvDirectNaiveConvBwd"};
        const auto naive_solver  = naive_conv_id.GetSolver();

        const auto tensors = miopen::ConvBwdTensors{
            output.desc, out_dev.get(), weights.desc, wei_dev.get(), input.desc, in_dev.get()};
        auto ctx = miopen::ConvolutionContext{
            input.desc, weights.desc, output.desc, filter, miopen::conv::Direction::BackwardData};
        ctx.SetStream(&handle);
        ctx.DetectRocm();
        if(naive_solver.IsApplicable(ctx))
        {
            gpu_ref_used          = true;
            const auto invoke_ctx = miopen::conv::DataInvokeParams{
                tensors, nullptr, 0, filter.attribute.gfx90aFp16alt.GetBwd()};
            const auto invoker = miopen::LoadOrPrepareInvoker(
                handle, ctx, naive_conv_id.Value(), miopen::conv::Direction::BackwardData);
            invoker(handle, invoke_ctx);
            input.data = handle.Read<Tin>(in_dev, input.data.size());
        }
    }
    return gpu_ref_used;
}

template <typename Tin, typename Twei, typename Tout>
bool gpu_ref_convolution_wrw(const tensor<Tin>& input,
                             tensor<Twei>& weights,
                             const tensor<Tout> output,
                             miopen::ConvolutionDescriptor filter)
{
    bool gpu_ref_used = false;
    if(!miopen::IsEnabled(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF{}))
    {
        auto&& handle            = get_handle();
        auto in_dev              = handle.Write(input.data);
        auto wei_dev             = handle.Write(weights.data);
        auto out_dev             = handle.Write(output.data);
        const auto naive_conv_id = miopen::solver::Id{"ConvDirectNaiveConvWrw"};
        const auto naive_solver  = naive_conv_id.GetSolver();

        const auto tensors = miopen::ConvWrwTensors{
            output.desc, out_dev.get(), input.desc, in_dev.get(), weights.desc, wei_dev.get()};
        auto ctx = miopen::ConvolutionContext{input.desc,
                                              weights.desc,
                                              output.desc,
                                              filter,
                                              miopen::conv::Direction::BackwardWeights};
        ctx.SetStream(&handle);
        ctx.DetectRocm();
        if(naive_solver.IsApplicable(ctx))
        {
            gpu_ref_used          = true;
            const auto invoke_ctx = miopen::conv::WrWInvokeParams{
                tensors, nullptr, 0, filter.attribute.gfx90aFp16alt.GetWrW()};
            const auto invoker = miopen::LoadOrPrepareInvoker(
                handle, ctx, naive_conv_id.Value(), miopen::conv::Direction::BackwardWeights);
            invoker(handle, invoke_ctx);
            weights.data = handle.Read<Twei>(wei_dev, weights.data.size());
        }
    }
    return gpu_ref_used;
}

#endif // GUARD_GPU_CONV_HPP
