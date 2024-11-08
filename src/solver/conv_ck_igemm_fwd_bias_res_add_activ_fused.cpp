
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
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <ck/tensor_operation/gpu/device/device_conv_fwd_bias_activation.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scaleadd_scaleadd_relu.hpp>
#endif
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_RES_ADD_ACTIV)

namespace miopen {
namespace solver {
namespace fusion {

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using CK_OutLayout = ck::tensor_layout::convolution::NDHWGK;

// DataType also applies to weights
// AccumDataType also applies to added z & bias tensors
template <typename DataType, typename AccumDataType = DataType>
using DeviceOp = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
        3,
        ck::tensor_layout::convolution::NDHWGC,
        ck::tensor_layout::convolution::GKZYXC,
        ck::Tuple<CK_OutLayout, ck::tensor_layout::convolution::G_K>,
        CK_OutLayout,
        DataType,                                // in data type
        DataType,                                // wei data type
        ck::Tuple<AccumDataType, AccumDataType>, // z & bias tensors data type
        DataType,                                // out data type
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::
            ScaleAddScaleAddRelu>>; // end DeviceOperationInstanceFactory

namespace {

struct CKArgs
{
    CKArgs(const miopen::conv::ProblemDescription& problem)
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

        in_lens      = {G, N, C, Di, Hi, Wi};
        out_lens     = {G, N, K, Do, Ho, Wo};
        wei_lens     = {G, K, C, Z, Y, X};
        bias_lens    = {G, 1, K, 1, 1, 1};
        bias_strides = {K, 0, 1, 0, 0, 0};

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

        filter_stride   = {ProblemInterpreter::GetAdjustedConvolutionStrideD(problem),
                         ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                         ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationD(problem),
                           ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                           ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding        = {ProblemInterpreter::GetInputLeftPadD(problem),
                    ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding        = {ProblemInterpreter::GetAdjustedInputRightPadD(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename DevOpPtr>
    auto MakeArgPtr(const DevOpPtr& op_ptr,
                    ConstData_t in_buf,
                    ConstData_t wei_buf,
                    Data_t out_buf,
                    ConstData_t z_buf,
                    ConstData_t bias_buf,
                    float alpha1,
                    float alpha2) const
    {
        using ScaleAddScaleAddRelu = ck::tensor_operation::element_wise::ScaleAddScaleAddRelu;
        return op_ptr->MakeArgumentPointer(in_buf,
                                           wei_buf,
                                           {z_buf, bias_buf},
                                           out_buf,
                                           in_lens,
                                           in_strides,
                                           wei_lens,
                                           wei_strides,
                                           {out_lens, bias_lens},
                                           {out_strides, bias_strides},
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

    template <typename DevOpPtr>
    auto MakeArgPtr(const DevOpPtr& op_ptr,
                    const miopen::fusion::FusionInvokeParams& data_ctx) const
    {
        const auto& conv_param =
            dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(*data_ctx.op_args.params[0]);
        assert(&conv_param);

        const auto& z_param =
            dynamic_cast<miopen::fusion::TensorScaleAddOpInvokeParam&>(*data_ctx.op_args.params[1]);
        assert(&z_param);

        const auto& bias_param =
            dynamic_cast<miopen::fusion::BiasOpInvokeParam&>(*data_ctx.op_args.params[2]);
        assert(&bias_param);

        /// \todo: Support general activation functions.
        /// only relu activation supported and hardcoded for now
        [[maybe_unused]] const auto& activ_param =
            dynamic_cast<miopen::fusion::ActivationOpInvokeParam&>(*data_ctx.op_args.params[3]);
        assert(&activ_param);

        return MakeArgPtr(op_ptr,
                          data_ctx.in,
                          conv_param.weights,
                          data_ctx.out,
                          z_param.tensor_ptr,
                          bias_param.bdata,
                          conv_param.alpha,
                          z_param.alpha);
    }

#if 0
    template <typename OpPtr>
    auto MakeArgPtr(const OpPtr& op_ptr, const ConvDataTensors& tensors) const
    {
        return MakeArgPtr(op_ptr, tensors.in, tensors.w, tensors.out);
    }
#endif

    template <typename DevOpPtr>
    bool IsSupportedBy(const DevOpPtr& op_ptr) const
    {
        auto arg_ptr = MakeArgPtr(op_ptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1.0, 1.0);
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
    std::array<ck::index_t, 6> in_lens;
    std::array<ck::index_t, 6> in_strides;
    std::array<ck::index_t, 6> out_lens;
    std::array<ck::index_t, 6> out_strides;
    std::array<ck::index_t, 6> wei_lens;
    std::array<ck::index_t, 6> wei_strides;
    std::array<ck::index_t, 6> bias_lens;
    std::array<ck::index_t, 6> bias_strides;
    std::array<ck::index_t, 3> filter_stride;
    std::array<ck::index_t, 3> filter_dilation;
    std::array<ck::index_t, 3> lPadding;
    std::array<ck::index_t, 3> rPadding;
};

} // namespace

// TODO: deal with separate input/output data types
template <typename DataType, typename AccumDataType>
void PerfConfigConvCKIgemmFwdBiasResAddActivFused::Init(
    const miopen::conv::ProblemDescription& problem)
{

    valid_kernels = FillValidKernelsIDs<DeviceOp<DataType, AccumDataType>, CKArgs>(problem);
    index         = 0;
    assert(!valid_kernels.empty());
    kernel_id = valid_kernels[0];
}

template <typename DataType, typename AccumDataType>
bool PerfConfigConvCKIgemmFwdBiasResAddActivFused::CheckIsSupportCKArgs(
    const miopen::conv::ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOp<DataType, AccumDataType>, CKArgs>(problem, kernel_id);
}

template <typename DataType, typename AccumDataType>
bool ConvCKIgemmFwdBiasResAddActivFused::CheckCKApplicability(
    const miopen::conv::ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOp<DataType, AccumDataType>, CKArgs>(problem);
}

#endif

void PerfConfigConvCKIgemmFwdBiasResAddActivFused::HeuristicInit(
    const FusionDescription& fdesc_problem)
{
    index     = 0;
    kernel_id = "";

#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
#else
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    switch(conv_problem.GetInDataType())
    {
    case miopenHalf: Init<ck::half_t>(conv_problem); break;
    case miopenFloat: Init<float>(conv_problem); break;
    case miopenBFloat16: Init<ck::bhalf_t>(conv_problem); break;
    case miopenInt8: Init<int8_t, float>(conv_problem); break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt32:
    case miopenInt64:
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
    case miopenFloat: return CheckIsSupportCKArgs<float>(conv_problem);
    case miopenBFloat16: return CheckIsSupportCKArgs<ck::bhalf_t>(conv_problem);
    case miopenInt8: return CheckIsSupportCKArgs<int8_t, float>(conv_problem);
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt32:
    case miopenInt64:
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
    if(env::disabled(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_BIAS_RES_ADD_ACTIV))
        return false;
    // check the sequence of prims
    if(desc.op_map.size() != 4)
        return false;
    if(desc.op_map[0]->kind() != miopenFusionOpConvForward)
        return false;
    if(desc.op_map[1]->kind() != miopenFusionOpTensorScaleAdd)
        return false;
    if(desc.op_map[2]->kind() != miopenFusionOpBiasForward)
        return false;
    if(desc.op_map[3]->kind() != miopenFusionOpActivForward)
        return false;
    const auto& activ_op = dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[3]);
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
    if(!(conv_problem.Is2d() || conv_problem.Is3d()))
        return false;
    if(!conv_problem.IsLayoutNHWC())
        return false;
    if(!ck_utility::is_ck_whitelist(ctx.GetStream().GetDeviceName()))
        return false;

    switch(conv_problem.GetInDataType())
    {
    case miopenHalf: return CheckCKApplicability<ck::half_t>(conv_problem);
    case miopenFloat: return CheckCKApplicability<float>(conv_problem);
    case miopenBFloat16: return CheckCKApplicability<ck::bhalf_t>(conv_problem);
    case miopenInt8: return CheckCKApplicability<int8_t, float>(conv_problem);
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt32:
    case miopenInt64:
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

    using ParamType = miopen::fusion::FusionInvokeParams;
    switch(conv_problem.GetInDataType())
    {
    case miopenInt8:
        return InitAnyInvokerFactory<DeviceOp<int8_t, float>, CKArgs, ParamType>(conv_problem,
                                                                                 config.kernel_id);
    case miopenHalf:
        return InitAnyInvokerFactory<DeviceOp<ck::half_t, ck::half_t>, CKArgs, ParamType>(
            conv_problem, config.kernel_id);
    case miopenFloat:
        return InitAnyInvokerFactory<DeviceOp<float, float>, CKArgs, ParamType>(conv_problem,
                                                                                config.kernel_id);
    case miopenBFloat16:
        return InitAnyInvokerFactory<DeviceOp<ck::bhalf_t, ck::bhalf_t>, CKArgs, ParamType>(
            conv_problem, config.kernel_id);

    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
    case miopenFloat8:
    case miopenBFloat8:
    default:
        MIOPEN_THROW(miopenStatusInternalError,
                     "ConvHipImplicitGemmBwdXdlops operation not implemented for this data type");
    }

#endif
}

} // namespace fusion
} // namespace solver
} // namespace miopen
