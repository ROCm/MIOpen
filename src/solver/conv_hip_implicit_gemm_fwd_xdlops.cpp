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

#include <vector>
#include <cstdint>

#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/library/tensor_operation_instance/gpu/convolution_forward.hpp>
#endif

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)

namespace miopen {
namespace solver {

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
using DeviceConvFwdPtr_t = std::unique_ptr<
    ck::tensor_operation::device::DeviceConvFwd<2,
                                                ck::tensor_layout::convolution::NHWC,
                                                ck::tensor_layout::convolution::KYXC,
                                                ck::tensor_layout::convolution::NHWK,
                                                int8_t,
                                                int8_t,
                                                int8_t,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>>;

struct CKArgs
{
    CKArgs(const ConvolutionContext& ctx)
    {
        N        = ProblemInterpreter::GetBatchN(ctx.problem);
        K        = ProblemInterpreter::GetOutputChannelK(ctx.problem);
        C        = ProblemInterpreter::GetInputChannelC(ctx.problem);
        input    = {ProblemInterpreter::GetInputHeightHi(ctx.problem),
                 ProblemInterpreter::GetInputWidthWi(ctx.problem)};
        output   = {ProblemInterpreter::GetOutputHeightHo(ctx.problem),
                  ProblemInterpreter::GetOutputWidthWo(ctx.problem)};
        filter   = {ProblemInterpreter::GetFilterHeightY(ctx.problem),
                  ProblemInterpreter::GetFilterWidthX(ctx.problem)};
        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(ctx.problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(ctx.problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(ctx.problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(ctx.problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(ctx.problem),
                    ProblemInterpreter::GetInputLeftPadW(ctx.problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(ctx.problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(ctx.problem)};
    }
    int N;
    int K;
    int C;
    std::vector<int> input;
    std::vector<int> output;
    std::vector<int> filter;
    std::vector<int> strides;
    std::vector<int> dilation;
    std::vector<int> lPadding;
    std::vector<int> rPadding;
};
#endif

void PerformanceConfigHipImplicitGemmFwdXdlops::HeuristicInit(const ConvolutionContext& ctx)
{
    this->index = 0;
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
#else
    this->index = 0;
    std::vector<DeviceConvFwdPtr_t> conv_ptrs;
    ck::tensor_operation::device::instance::add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(
        conv_ptrs);
    assert(!conv_ptrs.empty());
    this->total_size = conv_ptrs.size();
    const auto args  = CKArgs{ctx};
    for(auto& conv_ptr : conv_ptrs)
    {
        auto argument_ptr = conv_ptr->MakeArgumentPointer(nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          args.N,
                                                          args.K,
                                                          args.C,
                                                          args.input,
                                                          args.filter,
                                                          args.output,
                                                          args.strides,
                                                          args.dilation,
                                                          args.lPadding,
                                                          args.rPadding,
                                                          {},
                                                          {},
                                                          {});
        if(conv_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            this->kernel_id = conv_ptr->GetTypeString();
            break;
        }
        ++this->index;
    }
#endif
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::SetNextValue(const ConvolutionContext& ctx)
{
    if(total_size == -1)
        this->HeuristicInit(ctx);
    assert(total_size != -1);
    if((index + 1) < total_size)
    {
        ++index;
        return true;
    }
    else
        return false;
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::IsValidValue() const { return index < total_size; }

bool PerformanceConfigHipImplicitGemmFwdXdlops::IsValid(const ConvolutionContext& ctx) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    return false;
#else
    std::vector<DeviceConvFwdPtr_t> conv_ptrs;
    ck::tensor_operation::device::instance::add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(
        conv_ptrs);
    const auto args   = CKArgs{ctx};
    auto argument_ptr = conv_ptrs[this->index]->MakeArgumentPointer(nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    args.N,
                                                                    args.K,
                                                                    args.C,
                                                                    args.input,
                                                                    args.filter,
                                                                    args.input,
                                                                    args.strides,
                                                                    args.dilation,
                                                                    args.lPadding,
                                                                    args.rPadding,
                                                                    {},
                                                                    {},
                                                                    {});
    return conv_ptrs[this->index]->IsSupportedArgument(argument_ptr.get());
#endif
}

bool PerformanceConfigHipImplicitGemmFwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemmFwdXdlops& other) const
{
    return this->index == other.index;
}

PerformanceConfigHipImplicitGemmFwdXdlops
ConvHipImplicitGemmFwdXdlops::GetDefaultPerformanceConfig(const ConvolutionContext& ctx) const
{
    PerformanceConfigHipImplicitGemmFwdXdlops pp;
    pp.HeuristicInit(ctx);
    return pp;
}

bool ConvHipImplicitGemmFwdXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceConfigHipImplicitGemmFwdXdlops& config) const
{
    return config.IsValid(ctx);
}

PerformanceConfigHipImplicitGemmFwdXdlops
ConvHipImplicitGemmFwdXdlops::Search(const ConvolutionContext& ctx,
                                     const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, invoke_ctx);
}

size_t ConvHipImplicitGemmFwdXdlops::GetWorkspaceSize(const ConvolutionContext& ctx) const
{
    std::ignore = ctx;
    return 0;
}

bool ConvHipImplicitGemmFwdXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    return false;
#else
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS{}))
        return false;
    if(miopen::IsEnabled(MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC{}))
        return false;
    if(!(ctx.problem.conv_problem.GetInDataType() == miopenInt8 &&
         ctx.problem.conv_problem.GetWeightsDataType() == miopenInt8 &&
         ctx.problem.conv_problem.GetOutDataType() == miopenInt8))
        return false;
    if(!ctx.problem.direction.IsForward())
        return false;
    if(!ctx.problem.Is2d())
        return false;
    if(ctx.GetStream().GetDeviceName() != "gfx908")
        return false;
    if(!ctx.problem.IsLayoutNHWC())
        return false;

    const auto args = CKArgs{ctx};
    if(!std::all_of(args.strides.begin(), args.strides.end(), [&](auto x) { return x == 1; }))
        return false;

    std::vector<DeviceConvFwdPtr_t> conv_ptrs;
    ck::tensor_operation::device::instance::add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(
        conv_ptrs);
    assert(!conv_ptrs.empty());

    for(auto& conv_ptr : conv_ptrs)
    {
        auto argument_ptr = conv_ptr->MakeArgumentPointer(nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          args.N,
                                                          args.K,
                                                          args.C,
                                                          args.input,
                                                          args.filter,
                                                          args.input,
                                                          args.strides,
                                                          args.dilation,
                                                          args.lPadding,
                                                          args.rPadding,
                                                          {},
                                                          {},
                                                          {});
        if(conv_ptr->IsSupportedArgument(argument_ptr.get()))
            return true;
    }
    return false;
#endif
}

ConvSolution ConvHipImplicitGemmFwdXdlops::GetSolution(
    const ConvolutionContext& ctx, const PerformanceConfigHipImplicitGemmFwdXdlops& config) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    std::ignore = config;
    return {};
#else
    ConvSolution result;
    const auto args        = CKArgs{ctx};
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            std::vector<DeviceConvFwdPtr_t> conv_ptrs;
            ck::tensor_operation::device::instance::
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(conv_ptrs);
            auto& conv_ptr       = conv_ptrs.at(config.index);
            const auto& data_ctx = primitive_parameters.CastTo<conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;
            auto argument_ptr    = conv_ptr->MakeArgumentPointer(
                const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
                    static_cast<const void*>(tensors.in)),
                const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
                    static_cast<const void*>(tensors.w)),
                static_cast<void*>(tensors.out),
                args.N,
                args.K,
                args.C,
                args.input,
                args.filter,
                args.input,
                args.strides,
                args.dilation,
                args.lPadding,
                args.rPadding,
                {},
                {},
                {});
            auto invoker_ptr            = conv_ptr->MakeInvokerPointer();
            const auto enable_profiling = handle.IsProfilingEnabled();

            float elapsed_time =
                invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), enable_profiling});
            if(enable_profiling)
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed_time);
            }
        };
    };
    return result;
#endif
}

} // namespace solver
} // namespace miopen
