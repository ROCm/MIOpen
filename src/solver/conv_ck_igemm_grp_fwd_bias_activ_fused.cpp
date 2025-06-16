
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
#include <miopen/env.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bias_clamp.hpp"
#endif

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_CK_IGEMM_GRP_FWD_BIAS_ACTIV)
namespace miopen {
namespace solver {
namespace fusion {

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
static constexpr ck::index_t NDimSpatial = 2;
// todo : support 3D
// static constexpr ck::index_t NDimSpatial = 3;
// if relu passed by user then
// const float floor = 0;
// const float ceil  = std::numeric_limits<ck::bhalf_t>::max();

using InLayout  = ck::tensor_layout::convolution::NHWGC;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using OutLayout = ck::tensor_layout::convolution::NHWGK;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddClamp;

const auto in_element_op  = InElementOp{};
const auto wei_element_op = WeiElementOp{};

OutElementOp GetOutElementOp(const miopen::fusion::ActivationOpInvokeParam& activationOp)
{
    auto activationMode = activationOp.activMode;
    switch(activationMode)
    {
    case miopenActivationRELU: return OutElementOp{0, std::numeric_limits<ck::bhalf_t>::max()};
    case miopenActivationCLIPPEDRELU: return OutElementOp{0, activationOp.activAlpha};
    case miopenActivationCLAMP:
        return OutElementOp{activationOp.activAlpha, activationOp.activBeta};
    default:
        MIOPEN_THROW(miopenStatusInternalError,
                     "Unsupported activation type: " + std::to_string(activationMode));
    }
}

template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AComputeType = InDataType,
          typename BComputeType = AComputeType>
using DeviceOpGFwdBiasRelu =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<OutLayout>,
                                                                  OutLayout,
                                                                  InDataType,
                                                                  WeiDataType,
                                                                  ck::Tuple<OutDataType>,
                                                                  OutDataType,
                                                                  InElementOp,
                                                                  WeiElementOp,
                                                                  OutElementOp,
                                                                  AComputeType,
                                                                  BComputeType>;

template <typename DataType>
using DeviceOpGFwdBiasReluPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdBiasRelu<DataType, DataType, DataType>>;
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

        input_len  = {G, N, C, Hi, Wi};
        output_len = {G, N, K, Ho, Wo};
        weight_len = {G, K, C, Y, X};
        bias_len   = {G, 1, K, 1, 1};

        // NHWGC
        bias_strides = {K, 0, 1, 0, 0};

        // in_strides  = {C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
        // out_strides = {K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
        // wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};
        auto miopen_in_strides  = problem.GetIn().GetStrides();
        auto miopen_out_strides = problem.GetOut().GetStrides();
        auto miopen_wei_strides = problem.GetWeights().GetStrides();
        miopen_in_strides.insert(miopen_in_strides.begin(), C);
        miopen_out_strides.insert(miopen_out_strides.begin(), K);
        miopen_wei_strides.insert(miopen_wei_strides.begin(), K * miopen_wei_strides[0]);
        std::copy(miopen_in_strides.begin(), miopen_in_strides.end(), in_strides.begin());
        std::copy(miopen_out_strides.begin(), miopen_out_strides.end(), out_strides.begin());
        std::copy(miopen_wei_strides.begin(), miopen_wei_strides.end(), wei_strides.begin());

        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    ConstData_t in_buf,
                    ConstData_t w_buf,
                    ConstData_t bias_buf,
                    Data_t out_buf,
                    float alpha,
                    float beta,
                    OutElementOp clampOp) const
    {
        (void)alpha;
        (void)beta;

        return conv_ptr->MakeArgumentPointer(
            in_buf,
            w_buf,
            {bias_buf},
            out_buf,
            input_len,
            in_strides,
            weight_len,
            wei_strides,
            {output_len}, // hack CK's is applicable check. instead of bias_len we use output_len
            {bias_strides},
            output_len,
            out_strides,
            strides,
            dilation,
            lPadding,
            rPadding,
            in_element_op,
            wei_element_op,
            clampOp);
    }

    template <typename DevOpPtr>
    auto MakeArgPtr(const DevOpPtr& op_ptr,
                    const miopen::fusion::FusionInvokeParams& data_ctx) const
    {
        const auto& conv_param =
            dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(*data_ctx.op_args.params[0]);
        assert(&conv_param);

        const auto& bias_param =
            dynamic_cast<miopen::fusion::BiasOpInvokeParam&>(*data_ctx.op_args.params[1]);
        assert(&bias_param);

        const auto& activ_param =
            dynamic_cast<miopen::fusion::ActivationOpInvokeParam&>(*data_ctx.op_args.params[2]);

        return MakeArgPtr(op_ptr,
                          data_ctx.in,
                          conv_param.weights,
                          bias_param.bdata,
                          data_ctx.out,
                          conv_param.alpha,
                          conv_param.beta,
                          GetOutElementOp(activ_param));
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  1.0f,
                                  0.0f,
                                  OutElementOp{0, std::numeric_limits<ck::bhalf_t>::max()});
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    int G;
    int N;
    int K1;
    int C1;
    int K;
    int C;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int Y;
    int X;
    std::array<ck::index_t, 5> input_len;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> output_len;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> bias_len;
    std::array<ck::index_t, 5> bias_strides;
    std::array<ck::index_t, 5> weight_len;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 2> strides;
    std::array<ck::index_t, 2> dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};

} // namespace

template <typename DataType>
void PerformanceConfigConvCKIgemmGrpFwdBiasActivFused::Init(
    const miopen::conv::ProblemDescription& problem)
{
    if(valid_kernels.empty())
        valid_kernels = FillValidKernelsIDs<DeviceOpGFwdBiasReluPtrs<DataType>, CKArgs>(problem);
    index     = 0;
    kernel_id = valid_kernels[index];
}

template <typename DataType>
bool PerformanceConfigConvCKIgemmGrpFwdBiasActivFused::CheckIsSupportCKArgs(
    const miopen::conv::ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpGFwdBiasReluPtrs<DataType>, CKArgs>(problem, kernel_id);
}

template <typename DataType>
bool ConvCKIgemmGrpFwdBiasActivFused::CheckCKApplicability(
    const miopen::conv::ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpGFwdBiasReluPtrs<DataType>, CKArgs>(problem);
}

#endif

void PerformanceConfigConvCKIgemmGrpFwdBiasActivFused::HeuristicInit(
    const FusionDescription& fdesc_problem)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
#else
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    switch(conv_problem.GetInDataType())
    {
    case miopenBFloat16: Init<ck::bhalf_t>(conv_problem); break;
    case miopenHalf:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenFloat:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }

#endif
}

bool PerformanceConfigConvCKIgemmGrpFwdBiasActivFused::SetNextValue(
    const FusionDescription& fdesc_problem)
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(valid_kernels.empty())
    {
        const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
        switch(conv_problem.GetInDataType())
        {
        case miopenBFloat16: Init<ck::bhalf_t>(conv_problem); break;
        case miopenHalf:
        case miopenFloat:
        case miopenInt8:
        case miopenInt64:
        case miopenInt32:
        case miopenFloat8_fnuz:
        case miopenBFloat8_fnuz:
        case miopenDouble: break;
        }
        assert(!valid_kernels.empty());
        return true;
    }
    if((index + 1) < valid_kernels.size())
    {
        ++index;
        kernel_id = valid_kernels[index];
        return true;
    }
    else
#endif
        return false;
}

bool PerformanceConfigConvCKIgemmGrpFwdBiasActivFused::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigConvCKIgemmGrpFwdBiasActivFused::IsValid(
    const FusionContext&, const FusionDescription& fdesc_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    switch(conv_problem.GetInDataType())
    {
    case miopenBFloat16: return CheckIsSupportCKArgs<ck::bhalf_t>(conv_problem);
    case miopenHalf:
    case miopenFloat:
    case miopenInt8:
    case miopenInt64:
    case miopenInt32:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenDouble: break;
    }
#endif
    return false;
}

bool PerformanceConfigConvCKIgemmGrpFwdBiasActivFused::operator==(
    const PerformanceConfigConvCKIgemmGrpFwdBiasActivFused& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigConvCKIgemmGrpFwdBiasActivFused
ConvCKIgemmGrpFwdBiasActivFused::GetDefaultPerformanceConfig(
    const FusionContext&, const FusionDescription& fdesc_problem) const
{
    PerformanceConfigConvCKIgemmGrpFwdBiasActivFused pp;
    pp.HeuristicInit(fdesc_problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvCKIgemmGrpFwdBiasActivFused::IsValidPerformanceConfig(
    const FusionContext& ctx,
    const FusionDescription& fdesc_problem,
    const PerformanceConfigConvCKIgemmGrpFwdBiasActivFused& config) const
{
    return config.IsValid(ctx, fdesc_problem);
}

PerformanceConfigConvCKIgemmGrpFwdBiasActivFused
ConvCKIgemmGrpFwdBiasActivFused::Search(const FusionContext& ctx,
                                        const FusionDescription& fdesc_problem,
                                        const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, fdesc_problem, invoke_ctx);
}

bool ConvCKIgemmGrpFwdBiasActivFused::IsApplicable(const FusionContext& ctx,
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
    if(desc.op_map.size() != 3)
        return false;
    if(desc.op_map[0]->kind() != miopenFusionOpConvForward)
        return false;
    if(desc.op_map[1]->kind() != miopenFusionOpBiasForward)
        return false;
    if(desc.op_map[2]->kind() != miopenFusionOpActivForward)
        return false;
    const auto& activationType =
        dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[2]).activMode;
    if(activationType != miopenActivationRELU && activationType != miopenActivationCLIPPEDRELU &&
       activationType != miopenActivationCLAMP)
        return false;
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    if(env::disabled(MIOPEN_DEBUG_CONV_CK_IGEMM_GRP_FWD_BIAS_ACTIV))
        return false;
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
    if(arch != "gfx908" && arch != "gfx90a" && arch != "gfx942")
        return false;
    if(!conv_problem.IsLayoutNHWC())
        return false;

    switch(conv_problem.GetInDataType())
    {
    case miopenBFloat16: return CheckCKApplicability<ck::bhalf_t>(conv_problem);
    case miopenHalf:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenFloat:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

ConvSolution ConvCKIgemmGrpFwdBiasActivFused::GetSolution(
    const FusionContext&,
    const FusionDescription& fdesc_problem,
    const PerformanceConfigConvCKIgemmGrpFwdBiasActivFused& config) const
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
    case miopenBFloat16:
        return InitAnyInvokerFactory<DeviceOpGFwdBiasReluPtrs<ck::bhalf_t>, CKArgs, ParamType>(
            conv_problem, config.kernel_id);
    case miopenInt8:
    case miopenHalf:
    case miopenFloat:

    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    default:
        MIOPEN_THROW(miopenStatusInternalError,
                     "ConvHipImplicitGemmBwdXdlops operation not implemented for this data type");
    }

#endif
}

} // namespace fusion
} // namespace solver
} // namespace miopen
