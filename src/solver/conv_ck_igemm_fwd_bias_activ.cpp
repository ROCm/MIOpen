
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

#include <miopen/check_numerics.hpp>
#include <miopen/solver.hpp>
#include <miopen/conv/fused_data_invoke_params.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/tensor_operation/gpu/device/device_conv_fwd_bias_activation.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
// Forward declare CK's function.
namespace ck {
namespace tensor_operation {
namespace device {
using DeviceConvFwdBiasReluPtr = ck::tensor_operation::device::DeviceConvFwdBiasActivationPtr<
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::AddRelu>;

namespace instance {

void add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(
    std::vector<DeviceConvFwdBiasReluPtr>&);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif

namespace miopen {
namespace solver {
namespace fusion {
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        N        = ProblemInterpreter::GetBatchN(problem);
        K        = ProblemInterpreter::GetOutputChannelK(problem);
        C        = ProblemInterpreter::GetInputChannelC(problem);
        input    = {ProblemInterpreter::GetInputHeightHi(problem),
                 ProblemInterpreter::GetInputWidthWi(problem)};
        output   = {ProblemInterpreter::GetOutputHeightHo(problem),
                  ProblemInterpreter::GetOutputWidthWo(problem)};
        filter   = {ProblemInterpreter::GetFilterHeightY(problem),
                  ProblemInterpreter::GetFilterWidthX(problem)};
        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
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

template <typename DataType>
void PerformanceConfigConvCKIgemmFwdBiasActiv::Init(const ProblemDescription& problem)
{
    const auto& args = CKArgs{problem};
    std::vector<ck::tensor_operation::device::DeviceConvFwdBiasReluPtr> conv;
    ck::tensor_operation::device::instance::
        add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(conv);
    assert(!conv.empty());
    this->total_size = conv.size();
    for(const auto& it : conv)
    {
        auto argument_ptr = it->MakeArgumentPointer(nullptr,
                                                    nullptr,
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
        if(it->IsSupportedArgument(argument_ptr.get()))
        {
            this->kernel_id = it->GetTypeString();
            break;
        }
    }
}

template <typename DataType>
bool PerformanceConfigConvCKIgemmFwdBiasActiv::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    const auto& args = CKArgs{problem};
    std::vector<ck::tensor_operation::device::DeviceConvFwdBiasReluPtr> conv;
    ck::tensor_operation::device::instance::
        add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(conv);
    auto argument_ptr = conv[this->index]->MakeArgumentPointer(nullptr,
                                                               nullptr,
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
    return conv[this->index]->IsSupportedArgument(argument_ptr.get());
}

template <typename DataType>
bool ConvCKIgemmFwdBiasActiv::CheckCKApplicability(const ProblemDescription& problem) const
{
    std::vector<ck::tensor_operation::device::DeviceConvFwdBiasReluPtr> conv;
    ck::tensor_operation::device::instance::
        add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(conv);
    assert(!conv.empty());
    const auto& args = CKArgs{problem};
    for(const auto& it : conv)
    {
        auto argument_ptr = it->MakeArgumentPointer(nullptr,
                                                    nullptr,
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
        if(it->IsSupportedArgument(argument_ptr.get()))
            return true;
    }
    return false;
}

template <typename DataType>
void ConvCKIgemmFwdBiasActiv::RunCKSolution(
    const Handle& handle,
    const AnyInvokeParams& primitive_parameters,
    const ProblemDescription& problem,
    const PerformanceConfigConvCKIgemmFwdBiasActiv& config) const
{
    const auto& args = CKArgs{problem};
    std::vector<ck::tensor_operation::device::DeviceConvFwdBiasReluPtr> conv;
    ck::tensor_operation::device::instance::
        add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(conv);
    // From the list of kernels provided by CK, config.index is the one that was
    // tuned by PerformanceConfigConvCKIgemmFwdBiasActiv.
    auto& conv_ptr = conv.at(config.index);

    const auto& invoke_ctx = primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
    const auto& wei_buf    = std::dynamic_pointer_cast<miopen::fusion::ConvolutionOpInvokeParam>(
                              invoke_ctx.op_invokers[0])
                              ->weights;
    const auto& bias_buf =
        std::dynamic_pointer_cast<miopen::fusion::BiasOpInvokeParam>(invoke_ctx.op_invokers[1])
            ->bdata;

    auto argument_ptr = conv_ptr->MakeArgumentPointer(
        const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
            static_cast<const void*>(invoke_ctx.in)),
        const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
            static_cast<const void*>(wei_buf)),
        invoke_ctx.out,
        const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
            static_cast<const void*>(bias_buf)),
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
    auto invoker_ptr            = conv_ptr->MakeInvokerPointer();
    const auto enable_profiling = handle.IsProfilingEnabled();

    float elapsed_time =
        invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), enable_profiling});
    if(enable_profiling)
    {
        handle.ResetKernelTime();
        handle.AccumKernelTime(elapsed_time);
    }
}
#endif

void PerformanceConfigConvCKIgemmFwdBiasActiv::HeuristicInit(const FusionContext& ctx)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
#else
    const auto& conv_prob = ctx.problem.GetConvProblem(0, conv::Direction::Forward).conv_problem;
    switch(conv_prob.GetInDataType())
    {
    case miopenInt8:
    case miopenHalf: Init<ck::half_t>(conv_prob); break;
    case miopenFloat:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
#endif
}

bool PerformanceConfigConvCKIgemmFwdBiasActiv::SetNextValue(const FusionContext& ctx)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    return false;
#else
    if(this->total_size == -1)
    {
        this->HeuristicInit(ctx);
        assert(this->total_size != -1);
        return true;
    }
    if((this->index + 1) < this->total_size)
    {
        ++this->index;
        std::vector<ck::tensor_operation::device::DeviceConvFwdBiasReluPtr> conv;
        ck::tensor_operation::device::instance::
            add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(conv);
        this->kernel_id = conv[this->index]->GetTypeString();
        return true;
    }
    return false;
#endif
}

bool PerformanceConfigConvCKIgemmFwdBiasActiv::IsValidValue() const
{
    return this->index < this->total_size;
}

bool PerformanceConfigConvCKIgemmFwdBiasActiv::IsValid(const FusionContext& ctx) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    return false;
#else
    // Extract convolution problem from the fusion context.
    const auto& problem = ctx.problem.GetConvProblem(0, conv::Direction::Forward);
    switch(problem.conv_problem.GetInDataType())
    {
    case miopenInt8:
    case miopenHalf: return CheckIsSupportCKArgs<ck::half_t>(problem);
    case miopenFloat:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
    return false;
#endif
}

bool PerformanceConfigConvCKIgemmFwdBiasActiv::operator==(
    const PerformanceConfigConvCKIgemmFwdBiasActiv& other) const
{
    return this->index == other.index && this->total_size == other.total_size;
}

PerformanceConfigConvCKIgemmFwdBiasActiv
ConvCKIgemmFwdBiasActiv::GetDefaultPerformanceConfig(const FusionContext& ctx) const
{
    PerformanceConfigConvCKIgemmFwdBiasActiv pp;
    pp.HeuristicInit(ctx);
    return pp;
}

bool ConvCKIgemmFwdBiasActiv::IsValidPerformanceConfig(
    const FusionContext& ctx, const PerformanceConfigConvCKIgemmFwdBiasActiv& config) const
{
    return config.IsValid(ctx);
}

PerformanceConfigConvCKIgemmFwdBiasActiv
ConvCKIgemmFwdBiasActiv::Search(const FusionContext& ctx, const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, invoke_ctx);
}

bool ConvCKIgemmFwdBiasActiv::IsApplicable(const FusionContext& ctx) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    return false;
#else
    const auto& problem = ctx.problem.GetConvProblem(0, conv::Direction::Forward);

    if(problem.conv_problem.GetConv().attribute.deterministic)
        return false;

    if(problem.conv_problem.GetInDataType() != problem.conv_problem.GetWeightsDataType() ||
       problem.conv_problem.GetWeightsDataType() != problem.conv_problem.GetOutDataType() ||
       problem.conv_problem.GetInDataType() != problem.conv_problem.GetOutDataType())
        return false;

    if(!problem.direction.IsForward())
        return false;

    if(!problem.Is2d())
        return false;

    const std::string name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(name, "gfx8") || StartsWith(name, "gfx9")))
        return false;

    if(!problem.IsLayoutNHWC())
        return false;

    switch(problem.conv_problem.GetInDataType())
    {
    case miopenInt8:
    case miopenHalf: return CheckCKApplicability<ck::half_t>(problem);
    case miopenFloat:
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
    return false;
#endif
}

ConvSolution
ConvCKIgemmFwdBiasActiv::GetSolution(const FusionContext& ctx,
                                     const PerformanceConfigConvCKIgemmFwdBiasActiv& config) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    std::ignore = config;
    return {};
#else
    const auto& problem = ctx.problem.GetConvProblem(0, conv::Direction::Forward);
    ConvSolution result;
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            switch(problem.conv_problem.GetInDataType())
            {
            case miopenHalf:
                RunCKSolution<ck::half_t>(handle, primitive_parameters, problem, config);
                break;
            case miopenInt8:
            case miopenFloat:
            case miopenInt32:
            case miopenInt8x4:
            case miopenBFloat16:
            case miopenDouble: break;
            }
        };
    };
    return result;
#endif
}

} // namespace fusion
} // namespace solver
} // namespace miopen
