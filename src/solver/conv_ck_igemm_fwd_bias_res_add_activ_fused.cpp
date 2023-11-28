
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
#include <miopen/solver.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/tensor_operation/gpu/device/device_conv_fwd_bias_activation.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_ACTIV)

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


using CK_OutLayout = ck::tensor_layout::convolution::NDHWGK;

// InDataType also applies to weights
// OutDataType also applies to added z & bias tensors
template <typename InDataType, typename OutDataType = InDataType>
using DeviceOp = 
  ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD<
    3,
    ck::tensor_layout::convolution::NDHWGC,
    ck::tensor_layout::convolution::GKZYXC,
    ck::Tuple<CK_OutLayout, CK_OutLayout>,
    CK_OutLayout,
    InDataType, // in data type
    InDataType, // wei data type
    ck::Tuple<OutDataType, OutDataType>, // z & bias tensors data type
    InDataType, // out data type
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::ScaleAddScaleAddRelu>;


#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
namespace {

struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input Channel per group
        K  = K1 / G; // Number of output Channel per group
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);
        Di = ProblemInterpreter::GetInputDepthDi(problem);
        Do = ProblemInterpreter::GetOutputDepthDo(problem);
        Z  = ProblemInterpreter::GetFilterDepthZ(problem);

        input  = {G, N, C, Di, Hi, Wi};
        output = {G, N, K, Do, Ho, Wo};
        weight = {G, K, C, Z, Y, X};

        // miopen filter_stride to CK filter_stride
        auto miopen_in_strides  = problem.GetIn().GetStrides();
        auto miopen_out_strides = problem.GetOut().GetStrides();
        auto miopen_wei_strides = problem.GetWeights().GetStrides();
        miopen_in_strides.insert(miopen_in_strides.begin(), C);
        miopen_out_strides.insert(miopen_out_strides.begin(), K);
        miopen_wei_strides.insert(miopen_wei_strides.begin(), K * miopen_wei_strides[0]);
        std::copy(miopen_in_strides.begin(), miopen_in_strides.end(), in_strides.begin());
        std::copy(miopen_out_strides.begin(), miopen_out_strides.end(), out_strides.begin());
        std::copy(miopen_wei_strides.begin(), miopen_wei_strides.end(), wei_strides.begin());

        filter_stride  = {ProblemInterpreter::GetAdjustedConvolutionStrideD(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationD(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadD(problem),
                    ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadD(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename OpPtr>
    auto MakeArgPtr(const OpPtr& op_ptr, ConstData_t in, ConstData_t w, Data_t out,
        ConstData_t z, ConstData_t bias) const
    {

        using ScaleAddScaleAddRelu = ck::tensor_operation::element_wise::ScaleAddScaleAddRelu;
        return op_ptr->MakeArgumentPointer(in,
                                             w,
                                             {z, bias},
                                             out,
                                             in_lens,
                                             in_strides,
                                             wei_lens,
                                             wei_strides,
                                             {out_lens, out_lens},
                                             {out_strides, out_strides},
                                             out_lens,
                                             out_strides,
                                             filter_stride,
                                             filter_dilation,
                                             lPadding,
                                             rPadding,
                                             {}, // PassThrough
                                             {}, // PassThrough
                                             ScaleAddScaleAddRelu{alpha1, alpha2});
    }

#if 0
    template <typename OpPtr>
    auto MakeArgPtr(const OpPtr& op_ptr, const ConvDataTensors& tensors) const
    {
        return MakeArgPtr(op_ptr, tensors.in, tensors.w, tensors.out);
    }
#endif

    template <typename OpPtr>
    bool IsSupportedBy(const OpPtr& op_ptr) const
    {
        auto arg_ptr = MakeArgPtr(op_ptr, nullptr, nullptr, nullptr);
        return op_ptr->IsSupportedArgument(arg_ptr.get());
    }

    int G;
    int N;
    int K;
    int C;
    int C1;
    int K1;
    int Hi;
    int Wi;
    int Di;
    int Ho;
    int Wo;
    int Do;
    int Y;
    int X;
    int Z;
    float alpha1 = 1.0f;
    float alpha2 = 1.0f;
    std::array<ck::index_t, 6> input;
    std::array<ck::index_t, 6> in_strides;
    std::array<ck::index_t, 6> output;
    std::array<ck::index_t, 6> out_strides;
    std::array<ck::index_t, 6> weight;
    std::array<ck::index_t, 6> wei_strides;
    std::array<ck::index_t, 3> filter_stride;
    std::array<ck::index_t, 3> filter_dilation;
    std::array<ck::index_t, 3> lPadding;
    std::array<ck::index_t, 3> rPadding;
};

} // namespace

template <typename DataType>
void PerfConfigConvCKIgemmFwdBiasResAddActivFused::Init(
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
                                                    args.filter_stride,
                                                    args.filter_dilation,
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
bool PerfConfigConvCKIgemmFwdBiasResAddActivFused::CheckIsSupportCKArgs(
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
                                                          args.filter_stride,
                                                          args.filter_dilation,
                                                          args.lPadding,
                                                          args.rPadding,
                                                          {},
                                                          {},
                                                          {});
    return conv_ptrs[i]->IsSupportedArgument(argument_ptr.get());
}

template <typename DataType>
bool ConvCKIgemmFwdBiasResAddActivFused::CheckCKApplicability(
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
                                                    args.filter_stride,
                                                    args.filter_dilation,
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
                   const PerfConfigConvCKIgemmFwdBiasResAddActivFused& config)
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
        args.filter_stride,
        args.filter_dilation,
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

void PerfConfigConvCKIgemmFwdBiasResAddActivFused::HeuristicInit(
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
    case miopenBFloat16:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }

#endif
}

bool PerfConfigConvCKIgemmFwdBiasResAddActivFused::SetNextValue(
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

bool PerfConfigConvCKIgemmFwdBiasResAddActivFused::IsValidValue() const
{
    return this->index >= 0 && this->index < valid_kernels.size();
}

bool PerfConfigConvCKIgemmFwdBiasResAddActivFused::IsValid(
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
    case miopenBFloat16:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

bool PerfConfigConvCKIgemmFwdBiasResAddActivFused::operator==(
    const PerfConfigConvCKIgemmFwdBiasResAddActivFused& other) const
{
    return this->kernel_id == other.kernel_id;
}
PerfConfigConvCKIgemmFwdBiasResAddActivFused
ConvCKIgemmFwdBiasResAddActivFused::GetDefaultPerformanceConfig(
    const FusionContext&, const FusionDescription& fdesc_problem) const
{
    PerfConfigConvCKIgemmFwdBiasResAddActivFused pp;
    pp.HeuristicInit(fdesc_problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvCKIgemmFwdBiasResAddActivFused::IsValidPerformanceConfig(
    const FusionContext& ctx,
    const FusionDescription& fdesc_problem,
    const PerfConfigConvCKIgemmFwdBiasResAddActivFused& config) const
{
    return config.IsValid(ctx, fdesc_problem);
}

PerfConfigConvCKIgemmFwdBiasResAddActivFused
ConvCKIgemmFwdBiasResAddActivFused::Search(const FusionContext& ctx,
                                     const FusionDescription& fdesc_problem,
                                     const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, fdesc_problem, invoke_ctx);
}

bool ConvCKIgemmFwdBiasResAddActivFused::IsApplicable(const FusionContext& ctx,
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
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_ACTIV{}))
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
    case miopenBFloat16:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

ConvSolution ConvCKIgemmFwdBiasResAddActivFused::GetSolution(
    const FusionContext&,
    const FusionDescription& fdesc_problem,
    const PerfConfigConvCKIgemmFwdBiasResAddActivFused& config) const
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
