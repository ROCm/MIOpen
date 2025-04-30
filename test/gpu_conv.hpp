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

#include <miopen/env.hpp>
#include <miopen/convolution.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>

#include "get_handle.hpp"
#include "tensor_holder.hpp"

namespace env = miopen::env;

namespace miopen {
namespace debug {

MIOPEN_EXPORT extern bool
    AlwaysEnableConvDirectNaive; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)
MIOPEN_EXPORT extern bool
    LoggingQuiet; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)

} // namespace debug
} // namespace miopen

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF)

struct AutoPrepareForGpuReference
{
    AutoPrepareForGpuReference()
    {
        quiet_prev                                 = miopen::debug::LoggingQuiet;
        naive_prev                                 = miopen::debug::AlwaysEnableConvDirectNaive;
        miopen::debug::AlwaysEnableConvDirectNaive = true;
        miopen::debug::LoggingQuiet                = true;
    }
    AutoPrepareForGpuReference(const AutoPrepareForGpuReference&) = delete;
    AutoPrepareForGpuReference(AutoPrepareForGpuReference&&)      = delete;
    AutoPrepareForGpuReference& operator=(const AutoPrepareForGpuReference&) = delete;
    AutoPrepareForGpuReference& operator=(AutoPrepareForGpuReference&&) = delete;
    ~AutoPrepareForGpuReference()
    {
        miopen::debug::LoggingQuiet                = quiet_prev;
        miopen::debug::AlwaysEnableConvDirectNaive = naive_prev;
    }

private:
    bool naive_prev;
    bool quiet_prev;
};

template <typename Tin, typename Twei, typename Tout>
bool gpu_ref_convolution_fwd(const tensor<Tin>& input,
                             const tensor<Twei>& weights,
                             tensor<Tout>& rout,
                             miopen::ConvolutionDescriptor filter,
                             const miopen::Scalar& alpha = miopen::Scalar(1.0),
                             const miopen::Scalar& beta  = miopen::Scalar(0.0))
{
    bool gpu_ref_used = false;
    if(!env::enabled(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF))
    {
        const AutoPrepareForGpuReference guard;
        auto&& handle            = get_handle();
        auto in_dev              = handle.Write(input.data);
        auto wei_dev             = handle.Write(weights.data);
        auto out_dev             = handle.Write(rout.data);
        const auto naive_conv_id = miopen::solver::Id{"ConvDirectNaiveConvFwd"};
        const auto naive_solver  = naive_conv_id.GetSolver();

        const auto tensors = miopen::ConvFwdTensors{
            input.desc, in_dev.get(), weights.desc, wei_dev.get(), rout.desc, out_dev.get()};
        const auto problem = miopen::conv::ProblemDescription{input.desc,
                                                              weights.desc,
                                                              rout.desc,
                                                              filter,
                                                              miopen::conv::Direction::Forward,
                                                              0,
                                                              alpha,
                                                              beta};
        auto ctx           = miopen::ExecutionContext{};
        ctx.SetStream(&handle);
        if(naive_solver.IsApplicable(ctx, problem))
        {
            gpu_ref_used          = true;
            const auto invoke_ctx = miopen::conv::DataInvokeParams{
                tensors, nullptr, 0, filter.attribute.gfx90aFp16alt.GetFwd(), alpha, beta};
            const auto invoker = miopen::LoadOrPrepareInvoker(ctx, problem, naive_conv_id.Value());
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
                             miopen::ConvolutionDescriptor filter,
                             const miopen::Scalar& alpha = miopen::Scalar(1.0),
                             const miopen::Scalar& beta  = miopen::Scalar(0.0))
{
    bool gpu_ref_used = false;
    if(!env::enabled(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF))
    {
        const AutoPrepareForGpuReference guard;
        auto&& handle            = get_handle();
        auto in_dev              = handle.Write(input.data);
        auto wei_dev             = handle.Write(weights.data);
        auto out_dev             = handle.Write(output.data);
        const auto naive_conv_id = miopen::solver::Id{"ConvDirectNaiveConvBwd"};
        const auto naive_solver  = naive_conv_id.GetSolver();

        const auto tensors = miopen::ConvBwdTensors{
            output.desc, out_dev.get(), weights.desc, wei_dev.get(), input.desc, in_dev.get()};
        const auto problem = miopen::conv::ProblemDescription{output.desc,
                                                              weights.desc,
                                                              input.desc,
                                                              filter,
                                                              miopen::conv::Direction::BackwardData,
                                                              0,
                                                              alpha,
                                                              beta};
        auto ctx           = miopen::ExecutionContext{};
        ctx.SetStream(&handle);

        if(naive_solver.IsApplicable(ctx, problem))
        {
            gpu_ref_used          = true;
            const auto invoke_ctx = miopen::conv::DataInvokeParams{
                tensors, nullptr, 0, filter.attribute.gfx90aFp16alt.GetBwd(), alpha, beta};
            const auto invoker = miopen::LoadOrPrepareInvoker(ctx, problem, naive_conv_id.Value());
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
                             miopen::ConvolutionDescriptor filter,
                             const miopen::Scalar& alpha = miopen::Scalar(1.0),
                             const miopen::Scalar& beta  = miopen::Scalar(0.0))
{
    bool gpu_ref_used = false;
    if(!env::enabled(MIOPEN_DEBUG_TEST_DISABLE_GPU_REF))
    {
        const AutoPrepareForGpuReference guard;
        auto&& handle            = get_handle();
        auto in_dev              = handle.Write(input.data);
        auto wei_dev             = handle.Write(weights.data);
        auto out_dev             = handle.Write(output.data);
        const auto naive_conv_id = miopen::solver::Id{"ConvDirectNaiveConvWrw"};
        const auto naive_solver  = naive_conv_id.GetSolver();

        const auto tensors = miopen::ConvWrwTensors{
            output.desc, out_dev.get(), input.desc, in_dev.get(), weights.desc, wei_dev.get()};
        const auto problem =
            miopen::conv::ProblemDescription{output.desc,
                                             weights.desc,
                                             input.desc,
                                             filter,
                                             miopen::conv::Direction::BackwardWeights,
                                             0,
                                             alpha,
                                             beta};
        auto ctx = miopen::ExecutionContext{};
        ctx.SetStream(&handle);
        if(naive_solver.IsApplicable(ctx, problem))
        {
            gpu_ref_used          = true;
            const auto invoke_ctx = miopen::conv::WrWInvokeParams{
                tensors, nullptr, 0, filter.attribute.gfx90aFp16alt.GetWrW(), alpha, beta};
            const auto invoker = miopen::LoadOrPrepareInvoker(ctx, problem, naive_conv_id.Value());
            invoker(handle, invoke_ctx);
            weights.data = handle.Read<Twei>(wei_dev, weights.data.size());
        }
    }
    return gpu_ref_used;
}

#endif // GUARD_GPU_CONV_HPP
