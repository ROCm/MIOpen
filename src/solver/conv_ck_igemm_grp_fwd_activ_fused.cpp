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

#include <miopen/fusion/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/solver/ck_utility_common.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp.hpp"
#endif
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_CK_IGEMM_GRP_FWD_ACTIV)

namespace miopen {
namespace solver {
namespace fusion {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

template <ck::index_t NDimSpatial>
struct LayoutsSelector;

template <>
struct LayoutsSelector<2>
{
    using InLayout  = ck::tensor_layout::convolution::NHWGC;
    using WeiLayout = ck::tensor_layout::convolution::GKYXC;
    using OutLayout = ck::tensor_layout::convolution::NHWGK;
};

template <>
struct LayoutsSelector<3>
{
    using InLayout  = ck::tensor_layout::convolution::NDHWGC;
    using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
    using OutLayout = ck::tensor_layout::convolution::NDHWGK;
};

inline auto Get2DLayouts()
{
    struct Layouts
    {
        using InLayout  = ck::tensor_layout::convolution::NHWGC;
        using WeiLayout = ck::tensor_layout::convolution::GKYXC;
        using OutLayout = ck::tensor_layout::convolution::NHWGK;
    };
    return Layouts{};
}

inline auto Get3DLayouts()
{
    struct Layouts
    {
        using InLayout  = ck::tensor_layout::convolution::NDHWGC;
        using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
        using OutLayout = ck::tensor_layout::convolution::NDHWGK;
    };
    return Layouts{};
}

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::Clamp;

const auto in_element_op  = InElementOp{};
const auto wei_element_op = WeiElementOp{};

template <ck::index_t NumDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AComputeType = InDataType,
          typename BComputeType = AComputeType,
          typename InLayout     = ck::tensor_layout::convolution::NHWGC,
          typename WeiLayout    = ck::tensor_layout::convolution::GKYXC,
          typename OutLayout    = ck::tensor_layout::convolution::NHWGK>
using DeviceOpGFwdAct =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<>, // diff
                                                                  OutLayout,
                                                                  InDataType,
                                                                  WeiDataType,
                                                                  ck::Tuple<>, // diff
                                                                  OutDataType,
                                                                  InElementOp,
                                                                  WeiElementOp,
                                                                  OutElementOp,
                                                                  AComputeType,
                                                                  BComputeType>;

template <ck::index_t NumDimSpatial,
          typename DataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
using DeviceOpGFwdActPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    DeviceOpGFwdAct<NumDimSpatial,
                    DataType,
                    DataType,
                    DataType,
                    DataType,
                    DataType,
                    InLayout,
                    WeiLayout,
                    OutLayout>>;

namespace {

template <int NDimSpatial, typename DataType>
struct CKArgs
{
    using OutputElementOpType = OutElementOp;
    using OutputDataType      = DataType;

    CKArgs(const ProblemDescription& problem)
    {
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input Channel per group
        K  = K1 / G; // Number of output Channel per group

        if(problem.Is3d())
        {
            Di = ProblemInterpreter::GetInputDepthDi(problem);
            Do = ProblemInterpreter::GetOutputDepthDo(problem);
            Z  = ProblemInterpreter::GetFilterDepthZ(problem);
            Hi = ProblemInterpreter::GetInputHeightHi(problem);
            Wi = ProblemInterpreter::GetInputWidthWi(problem);
            Ho = ProblemInterpreter::GetOutputHeightHo(problem);
            Wo = ProblemInterpreter::GetOutputWidthWo(problem);
            Y  = ProblemInterpreter::GetFilterHeightY(problem);
            X  = ProblemInterpreter::GetFilterWidthX(problem);

            in_lens  = {G, N, C, Di, Hi, Wi};
            out_lens = {G, N, K, Do, Ho, Wo};
            wei_lens = {G, K, C, Z, Y, X};

            in_strides  = {C, Di * Hi * Wi * G * C, 1, Hi * Wi * G * C, Wi * G * C, G * C};
            out_strides = {K, Do * Ho * Wo * G * K, 1, Ho * Wo * G * K, Wo * G * K, G * K};
            wei_strides = {K * Z * Y * X * C, Z * Y * X * C, 1, Y * X * C, X * C, C};

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
        else
        {

            Hi = ProblemInterpreter::GetInputHeightHi(problem);
            Wi = ProblemInterpreter::GetInputWidthWi(problem);
            Ho = ProblemInterpreter::GetOutputHeightHo(problem);
            Wo = ProblemInterpreter::GetOutputWidthWo(problem);
            Y  = ProblemInterpreter::GetFilterHeightY(problem);
            X  = ProblemInterpreter::GetFilterWidthX(problem);

            in_lens  = {G, N, C, Hi, Wi};
            out_lens = {G, N, K, Ho, Wo};
            wei_lens = {G, K, C, Y, X};

            in_strides  = {C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
            out_strides = {K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
            wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};

            filter_stride   = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                             ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
            filter_dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                               ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
            lPadding        = {ProblemInterpreter::GetInputLeftPadH(problem),
                        ProblemInterpreter::GetInputLeftPadW(problem)};
            rPadding        = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                        ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
        }
    }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    ConstData_t in,
                    ConstData_t w,
                    ConstData_t bias,
                    Data_t out,
                    float alpha,
                    float beta,
                    OutElementOp clampOp) const
    {
        (void)alpha;
        (void)beta;
        (void)bias;
        constexpr bool is3DConv = (NDimSpatial == 3);

        if constexpr(is3DConv)
        {
            return conv_ptr->MakeArgumentPointer(in,
                                                 w,
                                                 {},
                                                 out,
                                                 in_lens,
                                                 in_strides,
                                                 wei_lens,
                                                 wei_strides,
                                                 {},
                                                 {},
                                                 out_lens,
                                                 out_strides,
                                                 filter_stride,
                                                 filter_dilation,
                                                 lPadding,
                                                 rPadding,
                                                 in_element_op,
                                                 wei_element_op,
                                                 clampOp);
        }
        else
        {
            std::array<ck::index_t, 5> adjusted_in_lens{};
            std::array<ck::index_t, 5> adjusted_out_lens{};
            std::array<ck::index_t, 5> adjusted_wei_lens{};

            std::copy(in_lens.begin(), in_lens.begin() + 5, adjusted_in_lens.begin());
            std::copy(out_lens.begin(), out_lens.begin() + 5, adjusted_out_lens.begin());
            std::copy(wei_lens.begin(), wei_lens.begin() + 5, adjusted_wei_lens.begin());

            std::array<ck::index_t, 5> adjusted_in_strides{};
            std::array<ck::index_t, 5> adjusted_out_strides{};
            std::array<ck::index_t, 5> adjusted_wei_strides{};
            std::copy(in_strides.begin(), in_strides.begin() + 5, adjusted_in_strides.begin());
            std::copy(out_strides.begin(), out_strides.begin() + 5, adjusted_out_strides.begin());
            std::copy(wei_strides.begin(), wei_strides.begin() + 5, adjusted_wei_strides.begin());

            std::array<ck::index_t, 2> adjusted_filter_stride{};
            std::array<ck::index_t, 2> adjusted_filter_dilation{};
            std::array<ck::index_t, 2> adjusted_lPadding{};
            std::array<ck::index_t, 2> adjusted_rPadding{};

            std::copy(
                filter_stride.begin(), filter_stride.begin() + 2, adjusted_filter_stride.begin());
            std::copy(filter_dilation.begin(),
                      filter_dilation.begin() + 2,
                      adjusted_filter_dilation.begin());
            std::copy(lPadding.begin(), lPadding.begin() + 2, adjusted_lPadding.begin());
            std::copy(rPadding.begin(), rPadding.begin() + 2, adjusted_rPadding.begin());

            return conv_ptr->MakeArgumentPointer(in,
                                                 w,
                                                 {},
                                                 out,
                                                 adjusted_in_lens,
                                                 adjusted_in_strides,
                                                 adjusted_wei_lens,
                                                 adjusted_wei_strides,
                                                 {},
                                                 {},
                                                 adjusted_out_lens,
                                                 adjusted_out_strides,
                                                 adjusted_filter_stride,
                                                 adjusted_filter_dilation,
                                                 adjusted_lPadding,
                                                 adjusted_rPadding,
                                                 in_element_op,
                                                 wei_element_op,
                                                 clampOp);
        }
    }

    template <typename DevOpPtr>
    auto MakeArgPtr(const DevOpPtr& op_ptr,
                    const miopen::fusion::FusionInvokeParams& data_ctx) const
    {
        const auto& conv_param =
            dynamic_cast<miopen::fusion::ConvolutionOpInvokeParam&>(*data_ctx.op_args.params[0]);
        assert(&conv_param);

        const auto& activ_param =
            dynamic_cast<miopen::fusion::ActivationOpInvokeParam&>(*data_ctx.op_args.params[1]);

        return MakeArgPtr(op_ptr,
                          data_ctx.in,
                          conv_param.weights,
                          nullptr,
                          data_ctx.out,
                          conv_param.alpha,
                          conv_param.beta,
                          GetOutElementOp<DataType, OutElementOp>(activ_param));
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
                                  OutElementOp{0, std::numeric_limits<DataType>::max()});
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
    int Di = 0; // Depth for 3D
    int Do = 0; // Depth for 3D
    int Z  = 0; // Filter depth for 3D
    std::array<ck::index_t, 6> in_lens;
    std::array<ck::index_t, 6> in_strides;
    std::array<ck::index_t, 6> out_lens;
    std::array<ck::index_t, 6> out_strides;
    std::array<ck::index_t, 6> wei_lens;
    std::array<ck::index_t, 6> wei_strides;
    std::array<ck::index_t, 3> filter_stride;
    std::array<ck::index_t, 3> filter_dilation;
    std::array<ck::index_t, 3> lPadding;
    std::array<ck::index_t, 3> rPadding;
};

} // namespace

template <typename DataType>
void PerformanceConfigConvCKIgemmGrpFwdActivFused::Init(
    const miopen::conv::ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        if(problem.Is3d())
        {
            using Layouts = decltype(Get3DLayouts());
            valid_kernels = FillValidKernelsIDs<DeviceOpGFwdActPtrs<3,
                                                                    DataType,
                                                                    Layouts::InLayout,
                                                                    Layouts::WeiLayout,
                                                                    Layouts::OutLayout>,
                                                CKArgs<3, DataType>>(problem);
        }
        else
        {
            using Layouts = decltype(Get2DLayouts());
            valid_kernels = FillValidKernelsIDs<DeviceOpGFwdActPtrs<2,
                                                                    DataType,
                                                                    Layouts::InLayout,
                                                                    Layouts::WeiLayout,
                                                                    Layouts::OutLayout>,
                                                CKArgs<2, DataType>>(problem);
        }
    }
    index     = 0;
    kernel_id = valid_kernels[index];
}

template <typename DataType>
bool PerformanceConfigConvCKIgemmGrpFwdActivFused::CheckIsSupportCKArgs(
    const miopen::conv::ProblemDescription& problem) const
{
    bool supported = false;
    if(problem.Is3d())
    {
        using Layouts = decltype(Get3DLayouts());
        supported     = IsCKArgsSupported<DeviceOpGFwdActPtrs<3,
                                                          DataType,
                                                          Layouts::InLayout,
                                                          Layouts::WeiLayout,
                                                          Layouts::OutLayout>,
                                      CKArgs<3, DataType>>(problem, kernel_id);
    }
    else
    {
        using Layouts = decltype(Get2DLayouts());
        supported     = IsCKArgsSupported<DeviceOpGFwdActPtrs<2,
                                                          DataType,
                                                          Layouts::InLayout,
                                                          Layouts::WeiLayout,
                                                          Layouts::OutLayout>,
                                      CKArgs<2, DataType>>(problem, kernel_id);
    }
    return supported;
}

template <typename DataType>
bool ConvCKIgemmGrpFwdActivFused::CheckCKApplicability(
    const miopen::conv::ProblemDescription& problem) const
{
    bool applicable = false;
    if(problem.Is3d())
    {
        using Layouts = decltype(Get3DLayouts());
        applicable    = IsCKApplicable<DeviceOpGFwdActPtrs<3,
                                                        DataType,
                                                        Layouts::InLayout,
                                                        Layouts::WeiLayout,
                                                        Layouts::OutLayout>,
                                    CKArgs<3, DataType>>(problem);
    }
    else
    {
        using Layouts = decltype(Get2DLayouts());
        applicable    = IsCKApplicable<DeviceOpGFwdActPtrs<2,
                                                        DataType,
                                                        Layouts::InLayout,
                                                        Layouts::WeiLayout,
                                                        Layouts::OutLayout>,
                                    CKArgs<2, DataType>>(problem);
    }
    return applicable;
}

#endif

void PerformanceConfigConvCKIgemmGrpFwdActivFused::HeuristicInit(
    const FusionDescription& fdesc_problem)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = fdesc_problem;
#else
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    switch(conv_problem.GetInDataType())
    {
    case miopenBFloat16: Init<ck::bhalf_t>(conv_problem); break;
    case miopenHalf: Init<ck::half_t>(conv_problem); break;
    case miopenFloat: Init<float>(conv_problem); break;
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }

#endif
}

bool PerformanceConfigConvCKIgemmGrpFwdActivFused::SetNextValue(
    const FusionDescription& fdesc_problem)
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(valid_kernels.empty())
    {
        const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
        switch(conv_problem.GetInDataType())
        {
        case miopenBFloat16: Init<ck::bhalf_t>(conv_problem); break;
        case miopenHalf: Init<ck::half_t>(conv_problem); break;
        case miopenFloat: Init<float>(conv_problem); break;
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

bool PerformanceConfigConvCKIgemmGrpFwdActivFused::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigConvCKIgemmGrpFwdActivFused::IsValid(
    const FusionContext&, const FusionDescription& fdesc_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    switch(conv_problem.GetInDataType())
    {
    case miopenBFloat16: return CheckIsSupportCKArgs<ck::bhalf_t>(conv_problem);
    case miopenHalf: return CheckIsSupportCKArgs<ck::half_t>(conv_problem);
    case miopenFloat: return CheckIsSupportCKArgs<float>(conv_problem);
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

bool PerformanceConfigConvCKIgemmGrpFwdActivFused::operator==(
    const PerformanceConfigConvCKIgemmGrpFwdActivFused& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigConvCKIgemmGrpFwdActivFused
ConvCKIgemmGrpFwdActivFused::GetDefaultPerformanceConfig(
    const FusionContext&, const FusionDescription& fdesc_problem) const
{
    PerformanceConfigConvCKIgemmGrpFwdActivFused pp;
    pp.HeuristicInit(fdesc_problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvCKIgemmGrpFwdActivFused::IsValidPerformanceConfig(
    const FusionContext& ctx,
    const FusionDescription& fdesc_problem,
    const PerformanceConfigConvCKIgemmGrpFwdActivFused& config) const
{
    return config.IsValid(ctx, fdesc_problem);
}

size_t ConvCKIgemmGrpFwdActivFused::GetWorkspaceSize(const FusionContext&,
                                                     const FusionDescription& fdesc_problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    return GetWorkspaceSizeLayoutTransformConv(conv_problem);
#else
    std::ignore = fdesc_problem;
    return 0;
#endif
}

PerformanceConfigConvCKIgemmGrpFwdActivFused
ConvCKIgemmGrpFwdActivFused::Search(const FusionContext& ctx,
                                    const FusionDescription& fdesc_problem,
                                    const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, fdesc_problem, invoke_ctx);
}

bool ConvCKIgemmGrpFwdActivFused::IsApplicable(const FusionContext& ctx,
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
    if(desc.op_map.size() != 2)
        return false;
    if(desc.op_map[0]->kind() != miopenFusionOpConvForward)
        return false;
    if(desc.op_map[1]->kind() != miopenFusionOpActivForward)
        return false;
    const auto& activationType =
        dynamic_cast<ActivFwdFusionOpDescriptor&>(*desc.op_map[1]).activMode;
    if(activationType != miopenActivationRELU && activationType != miopenActivationCLIPPEDRELU &&
       activationType != miopenActivationCLAMP)
        return false;
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);
    if(env::disabled(MIOPEN_DEBUG_CONV_CK_IGEMM_GRP_FWD_ACTIV))
        return false;
    if(!conv_problem.IsBfp16() && !conv_problem.IsFp16() && !conv_problem.IsFp32())
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
    if(!(conv_problem.Is2d() || conv_problem.Is3d()))
        return false;
    if(!ck_utility::is_ck_whitelist(ctx.GetStream().GetDeviceName()))
        return false;
    if(!conv_problem.IsLayoutNHWC() && !conv_problem.IsLayoutDefault())
        return false;

    switch(conv_problem.GetInDataType())
    {
    case miopenBFloat16: return CheckCKApplicability<ck::bhalf_t>(conv_problem);
    case miopenHalf: return CheckCKApplicability<ck::half_t>(conv_problem);
    case miopenFloat: return CheckCKApplicability<float>(conv_problem);
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return false;
#endif
}

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <ck::index_t NDimSpatial, typename DataType>
ConvSolution
GetSolutionForDimensionality(const FusionContext& ctx,
                             const miopen::conv::ProblemDescription& conv_problem,
                             const PerformanceConfigConvCKIgemmGrpFwdActivFused& config)
{
    using Layouts = LayoutsSelector<NDimSpatial>;
    return MakeSolutionGroupConvImplicitGemmXdlops(
        conv_problem,
        [&](auto data_type_val) {
            (void)data_type_val;
            return InitInvokerFactoryFwdNCHW<NDimSpatial,
                                             false,
                                             DeviceOpGFwdActPtrs<NDimSpatial,
                                                                 DataType,
                                                                 typename Layouts::InLayout,
                                                                 typename Layouts::WeiLayout,
                                                                 typename Layouts::OutLayout>,
                                             CKArgs<NDimSpatial, DataType>,
                                             miopen::fusion::FusionInvokeParams>(
                ctx, conv_problem, config.kernel_id);
        },
        [&](auto data_type_val) {
            (void)data_type_val;
            return InitInvokerFactoryNHWC<false,
                                          DeviceOpGFwdActPtrs<NDimSpatial,
                                                              DataType,
                                                              typename Layouts::InLayout,
                                                              typename Layouts::WeiLayout,
                                                              typename Layouts::OutLayout>,
                                          CKArgs<NDimSpatial, DataType>,
                                          miopen::fusion::FusionInvokeParams>(
                ctx, conv_problem, config.kernel_id);
        });
}

template <ck::index_t NDim>
ConvSolution GetSolutionWithDim(const FusionContext& ctx,
                                const FusionDescription& fdesc_problem,
                                const PerformanceConfigConvCKIgemmGrpFwdActivFused& config)
{
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);

    switch(conv_problem.GetInDataType())
    {
    case miopenBFloat16:
        return GetSolutionForDimensionality<NDim, ck::bhalf_t>(ctx, conv_problem, config);
    case miopenHalf:
        return GetSolutionForDimensionality<NDim, ck::half_t>(ctx, conv_problem, config);
    case miopenFloat: return GetSolutionForDimensionality<NDim, float>(ctx, conv_problem, config);
    case miopenInt8:
    case miopenInt64:
    case miopenInt32:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }
}
#endif

ConvSolution ConvCKIgemmGrpFwdActivFused::GetSolution(
    const FusionContext& ctx,
    const FusionDescription& fdesc_problem,
    const PerformanceConfigConvCKIgemmGrpFwdActivFused& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    const auto conv_problem = fdesc_problem.GetConvProblem(0, miopen::conv::Direction::Forward);

    if(conv_problem.Is3d())
        return GetSolutionWithDim<3>(ctx, fdesc_problem, config);
    else
        return GetSolutionWithDim<2>(ctx, fdesc_problem, config);
#else
    return {};
#endif
}

} // namespace fusion
} // namespace solver
} // namespace miopen
