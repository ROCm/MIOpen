
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/fusion/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/tensor_operation/gpu/device/device_conv_fwd_bias_activation.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_ACTIV)

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
namespace {

struct CKArgs
{
    CKArgs(const miopen::conv::ProblemDescription& problem)
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

} // namespace

template <typename DataType>
void PerformanceConfigConvCKIgemmFwdBiasActivFused::Init(
    const miopen::conv::ProblemDescription& problem)
{
    const auto& args = CKArgs{problem};
    std::vector<ck::tensor_operation::device::DeviceConvFwdBiasReluPtr> conv_ptrs;
    ck::tensor_operation::device::instance::
        add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
    assert(!conv_ptrs.empty());
    for(const auto& it : conv_ptrs)
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
            valid_kernels.push_back(it->GetTypeString());
        }
    }

    assert(!valid_kernels.empty());
    this->index     = 0;
    this->kernel_id = valid_kernels[0];
}

template <typename DataType>
bool PerformanceConfigConvCKIgemmFwdBiasActivFused::CheckIsSupportCKArgs(
    const miopen::conv::ProblemDescription& problem) const
{
    const auto& args = CKArgs{problem};
    std::vector<ck::tensor_operation::device::DeviceConvFwdBiasReluPtr> conv_ptrs;
    ck::tensor_operation::device::instance::
        add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);

    int i = 0;
    for(; i < conv_ptrs.size(); i++)
    {
        if(conv_ptrs[i]->GetTypeString() == this->kernel_id)
        {
            break;
        }
    }
    if(i == valid_kernels.size())
    {
        return false;
    }
    auto argument_ptr = conv_ptrs[i]->MakeArgumentPointer(nullptr,
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
    return conv_ptrs[i]->IsSupportedArgument(argument_ptr.get());
}

template <typename DataType>
bool ConvCKIgemmFwdBiasActivFused::CheckCKApplicability(
    const miopen::conv::ProblemDescription& problem) const
{
    std::vector<ck::tensor_operation::device::DeviceConvFwdBiasReluPtr> conv_ptrs;
    ck::tensor_operation::device::instance::
        add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
    assert(!conv_ptrs.empty());
    const auto& args = CKArgs{problem};
    for(const auto& it : conv_ptrs)
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

namespace {

template <typename DataType>
void RunCKSolution(const Handle& handle,
                   const AnyInvokeParams& primitive_parameters,
                   const miopen::conv::ProblemDescription& problem,
                   const PerformanceConfigConvCKIgemmFwdBiasActivFused& config)
{
    const auto& args = CKArgs{problem};
    std::vector<ck::tensor_operation::device::DeviceConvFwdBiasReluPtr> conv_ptrs;
    ck::tensor_operation::device::instance::
        add_device_conv2d_fwd_xdl_c_shuffle_bias_relu_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);

    int id = 0;
    for(; id < conv_ptrs.size(); id++)
    {
        if(conv_ptrs[id]->GetTypeString() == config.kernel_id)
        {
            break;
        }
    }
    assert(id < conv_ptrs.size());
    auto& conv_ck          = conv_ptrs.at(id);
    const auto& invoke_ctx = primitive_parameters.CastTo<miopen::fusion::FusionInvokeParams>();
    const auto& wei_buf =
        dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(*invoke_ctx.op_args.params[0])
            .weights;
    const auto& bias_buf =
        dynamic_cast<miopen::fusion::BiasOpInvokeParam&>(*invoke_ctx.op_args.params[1]).bdata;

    auto argument_ptr = conv_ck->MakeArgumentPointer(
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
    auto invoker_ptr            = conv_ck->MakeInvokerPointer();
    const auto enable_profiling = handle.IsProfilingEnabled();

    float elapsed_time =
        invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), enable_profiling});
    if(enable_profiling)
    {
        handle.ResetKernelTime();
        handle.AccumKernelTime(elapsed_time);
    }
}

} // namespace
#endif

void PerformanceConfigConvCKIgemmFwdBiasActivFused::HeuristicInit(
    const FusionDescription& fdesc_problem)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
#else
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    switch(conv_problem.GetInDataType())
    {
    case miopenHalf: Init<ck::half_t>(conv_problem); break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenFloat:
    case miopenInt32:
    case miopenInt64:
    case miopenBFloat16:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }

#endif
}

bool PerformanceConfigConvCKIgemmFwdBiasActivFused::SetNextValue(
    const FusionDescription& fdesc_problem)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
    return false;
#else
    if(this->valid_kernels.empty())
    {
        this->HeuristicInit(fdesc_problem);
        assert(!valid_kernels.empty());
        return true;
    }
    if((this->index + 1) < valid_kernels.size())
    {
        ++this->index;
        this->kernel_id = this->valid_kernels[index];
        return true;
    }
    else
        return false;
#endif
}

bool PerformanceConfigConvCKIgemmFwdBiasActivFused::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool PerformanceConfigConvCKIgemmFwdBiasActivFused::IsValid(
    const FusionContext&, const FusionDescription& fdesc_problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
    return false;
#else
    // Extract convolution problem from the fusion context.
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    switch(conv_problem.GetInDataType())
    {
    case miopenHalf: return CheckIsSupportCKArgs<ck::half_t>(conv_problem);
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenFloat:
    case miopenInt32:
    case miopenInt64:
    case miopenBFloat16:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

bool PerformanceConfigConvCKIgemmFwdBiasActivFused::operator==(
    const PerformanceConfigConvCKIgemmFwdBiasActivFused& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigConvCKIgemmFwdBiasActivFused
ConvCKIgemmFwdBiasActivFused::GetDefaultPerformanceConfig(
    const FusionContext&, const FusionDescription& fdesc_problem) const
{
    PerformanceConfigConvCKIgemmFwdBiasActivFused pp;
    pp.HeuristicInit(fdesc_problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvCKIgemmFwdBiasActivFused::IsValidPerformanceConfig(
    const FusionContext& ctx,
    const FusionDescription& fdesc_problem,
    const PerformanceConfigConvCKIgemmFwdBiasActivFused& config) const
{
    return config.IsValid(ctx, fdesc_problem);
}

PerformanceConfigConvCKIgemmFwdBiasActivFused
ConvCKIgemmFwdBiasActivFused::Search(const FusionContext& ctx,
                                     const FusionDescription& fdesc_problem,
                                     const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, fdesc_problem, invoke_ctx);
}

bool ConvCKIgemmFwdBiasActivFused::IsApplicable(const FusionContext& ctx,
                                                const FusionDescription& fdesc_problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    std::ignore = fdesc_problem;
    return false;
#else
    const auto& desc = *fdesc_problem.fusion_plan_desc;
    if(desc.op_map.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError, "desc.op_map.empty()");
    }
    if(env::disabled(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_ACTIV))
        return false;
    // check the sequence of prims
    if(desc.op_map.size() != 3)
        return false;
    if(desc.op_map[0]->kind() != miopenFusionOpConvForward)
        return false;
    if(desc.op_map[1]->kind() != miopenFusionOpBiasForward)
        return false;
    if(desc.op_map[2]->kind() != miopenFusionOpActivForward)
        return false;
    const auto& activ_op = dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[2]);
    if(activ_op.activMode != miopenActivationRELU)
        return false;
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);

    if(conv_problem.IsTensorsCasted())
        return false;
    if(conv_problem.GetConv().attribute.deterministic)
        return false;
    if(conv_problem.HasNonPackedTensors())
        return false;
    if(!conv_problem.AllTensorsDimsFitIntoInt())
        return false;
    if(conv_problem.HasMixedDataTypes())
        return false;
    if(!conv_problem.Is2d())
        return false;
    const std::string arch = ctx.GetStream().GetDeviceName();
    if(arch != "gfx908" && arch != "gfx90a" && arch != "gfx940" && arch != "gfx941" &&
       arch != "gfx942")
        return false;
    if(!conv_problem.IsLayoutNHWC())
        return false;

    switch(conv_problem.GetInDataType())
    {
    case miopenHalf: return CheckCKApplicability<ck::half_t>(conv_problem);
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenFloat:
    case miopenInt32:
    case miopenInt64:
    case miopenBFloat16:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

ConvSolution ConvCKIgemmFwdBiasActivFused::GetSolution(
    const FusionContext&,
    const FusionDescription& fdesc_problem,
    const PerformanceConfigConvCKIgemmFwdBiasActivFused& config) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
    std::ignore = config;
    return {};
#else
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    ConvSolution result;
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            switch(conv_problem.GetInDataType())
            {
            case miopenHalf:
                RunCKSolution<ck::half_t>(handle, primitive_parameters, conv_problem, config);
                break;
            case miopenFloat8:
            case miopenBFloat8:
            case miopenInt8:
            case miopenFloat:
            case miopenInt32:
            case miopenInt64:
            case miopenBFloat16:
            case miopenDouble:
            default: MIOPEN_THROW("Unsupported datatype");
            }
        };
    };
    return result;
#endif
}

} // namespace fusion
} // namespace solver
} // namespace miopen
