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

#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR)

#include <miopen/conv/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data_scale.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data.hpp>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_3d_kernel_tuning_utils.hpp>
#endif
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>

namespace miopen {
namespace solver {
namespace conv {

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

using InLayout                             = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout                            = ck::tensor_layout::convolution::GKZYXC;
using OutLayout                            = ck::tensor_layout::convolution::NDHWGK;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
using Bilinear                             = ck::tensor_operation::element_wise::Bilinear;
using Scale                                = ck::tensor_operation::element_wise::Scale;
static constexpr ck::index_t NumDimSpatial = 3;

template <typename DataType>
using DeviceOpGBwdBilinear =
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NumDimSpatial,
                                                                    OutLayout,
                                                                    WeiLayout,
                                                                    ck::Tuple<InLayout>,
                                                                    InLayout,
                                                                    DataType,
                                                                    DataType,
                                                                    ck::Tuple<DataType>,
                                                                    DataType,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    Bilinear>;

template <typename DataType>
using DeviceOpGBwdScale =
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NumDimSpatial,
                                                                    OutLayout,
                                                                    WeiLayout,
                                                                    ck::Tuple<>,
                                                                    InLayout,
                                                                    DataType,
                                                                    DataType,
                                                                    ck::Tuple<>,
                                                                    DataType,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    Scale>;

template <typename DataType>
using DeviceOpGBwdDefault =
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NumDimSpatial,
                                                                    OutLayout,
                                                                    WeiLayout,
                                                                    ck::Tuple<>,
                                                                    InLayout,
                                                                    DataType,
                                                                    DataType,
                                                                    ck::Tuple<>,
                                                                    DataType,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    DataType,
                                                                    DataType>;

template <typename DataType>
using DeviceOpGBwdBilinearPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdBilinear<DataType>>;

template <typename DataType>
using DeviceOpGBwdScalePtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdScale<DataType>>;

template <typename DataType>
using DeviceOpGBwdDefaultPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdDefault<DataType>>;

namespace {

template <typename DataType>
struct CKArgs
{
    CKArgs(const ::miopen::conv::ProblemDescription& problem)
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

        in_lengths  = {G, N, C, Di, Hi, Wi};
        out_lengths = {G, N, K, Do, Ho, Wo};
        wei_lengths = {G, K, C, Z, Y, X};

        // CK strides are in GNCDHW order
        if(problem.IsLayoutNHWC())
        {
            // first entry reserved for G's stride
            auto copy_strides = [](const auto& src, auto& dst) {
                assert(dst.size() == (src.size() + 1));
                std::copy(src.begin(), src.end(), dst.begin() + 1);
            };
            copy_strides(problem.GetIn().GetStrides(), in_strides);
            copy_strides(problem.GetOut().GetStrides(), out_strides);
            copy_strides(problem.GetWeights().GetStrides(), wei_strides);

            // On a backward pass, problem.GetIn() means y(or out),
            // and problem.GetOut means x(or in)
            /// \todo remove this when we stop swapping in and out tensors/descriptors
            std::swap(in_strides, out_strides);

            // Now compute G's stride
            in_strides[0]  = C;
            out_strides[0] = K;
            wei_strides[0] = K * wei_strides[1];
        }
        else
        {
            assert(problem.IsLayoutDefault()); // already checked in IsApplicable
            // for default layout, we produce packed strides for NHWC layout
            // because we transpose to NHWC layout before calling CK kernel
            in_strides  = {C, Di * Hi * Wi * G * C, 1, Hi * Wi * G * C, Wi * G * C, G * C};
            out_strides = {K, Do * Ho * Wo * G * K, 1, Ho * Wo * G * K, Wo * G * K, G * K};
            wei_strides = {K * Z * Y * X * C, Z * Y * X * C, 1, Y * X * C, X * C, C};
        }

        filter_strides   = {ProblemInterpreter::GetAdjustedConvolutionStrideD(problem),
                          ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                          ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilations = {ProblemInterpreter::GetAdjustedConvolutionDilationD(problem),
                            ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                            ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding         = {ProblemInterpreter::GetInputLeftPadD(problem),
                    ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding         = {ProblemInterpreter::GetAdjustedInputRightPadD(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    Data_t in,
                    ConstData_t w,
                    ConstData_t out,
                    float alpha,
                    float beta) const
    {
        using DeviceP = std::remove_pointer_t<decltype(conv_ptr.get())>;
        if constexpr(std::is_same_v<DeviceP, DeviceOpGBwdBilinear<DataType>>)
        {
            return MakeBilinearArgPtr(conv_ptr, in, w, out, alpha, beta);
        }
        else if constexpr(std::is_same_v<DeviceP, DeviceOpGBwdScale<DataType>>)
        {
            (void)beta;
            return MakeScaleArgPtr(conv_ptr, in, w, out, alpha);
        }
        else
        {
            (void)alpha;
            (void)beta;
            static_assert(std::is_same_v<DeviceP, DeviceOpGBwdDefault<DataType>>,
                          "Default should be bwd pass through");
            return MakeDefaultArgPtr(conv_ptr, in, w, out);
        }
    }

    template <typename ConvPtr>
    auto MakeBilinearArgPtr(const ConvPtr& conv_ptr,
                            Data_t in,
                            ConstData_t w,
                            ConstData_t out,
                            float alpha,
                            float beta) const
    {
        return conv_ptr->MakeArgumentPointer(out,
                                             w,
                                             {in},
                                             in,
                                             out_lengths,
                                             out_strides,
                                             wei_lengths,
                                             wei_strides,
                                             {in_lengths},
                                             {in_strides},
                                             in_lengths,
                                             in_strides,
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             PassThrough{},
                                             Bilinear{alpha, beta});
    }

    template <typename ConvPtr>
    auto MakeScaleArgPtr(
        const ConvPtr& conv_ptr, Data_t in, ConstData_t w, ConstData_t out, float alpha) const
    {
        return conv_ptr->MakeArgumentPointer(out,
                                             w,
                                             {},
                                             in,
                                             out_lengths,
                                             out_strides,
                                             wei_lengths,
                                             wei_strides,
                                             {},
                                             {},
                                             in_lengths,
                                             in_strides,
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             PassThrough{},
                                             Scale{alpha});
    }

    template <typename ConvPtr>
    auto MakeDefaultArgPtr(const ConvPtr& conv_ptr, Data_t in, ConstData_t w, ConstData_t out) const
    {
        return conv_ptr->MakeArgumentPointer(out,
                                             w,
                                             {},
                                             in,
                                             out_lengths,
                                             out_strides,
                                             wei_lengths,
                                             wei_strides,
                                             {},
                                             {},
                                             in_lengths,
                                             in_strides,
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             PassThrough{},
                                             PassThrough{});
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    const ConvDataTensors& tensors,
                    float alpha,
                    float beta) const
    {
        return MakeArgPtr(conv_ptr, tensors.out, tensors.w, tensors.in, alpha, beta);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f);
        return conv_ptr->IsSupportedArgument(arg_ptr.get());
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
    std::array<ck::index_t, 6> in_lengths;
    std::array<ck::index_t, 6> in_strides;
    std::array<ck::index_t, 6> out_lengths;
    std::array<ck::index_t, 6> out_strides;
    std::array<ck::index_t, 6> wei_lengths;
    std::array<ck::index_t, 6> wei_strides;
    std::array<ck::index_t, 3> filter_strides;
    std::array<ck::index_t, 3> filter_dilations;
    std::array<ck::index_t, 3> lPadding;
    std::array<ck::index_t, 3> rPadding;
};

template <typename DataType>
std::vector<std::string>
FillValidKernelsByAlphaBeta(const ::miopen::conv::ProblemDescription& problem)
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return FillValidKernelsIDs<DeviceOpGBwdBilinearPtrs<DataType>, CKArgs<DataType>>(problem);
    case SCALE:
        return FillValidKernelsIDs<DeviceOpGBwdScalePtrs<DataType>, CKArgs<DataType>>(problem);
    default:
        return FillValidKernelsIDs<DeviceOpGBwdDefaultPtrs<DataType>, CKArgs<DataType>>(problem);
    }
}
} // namespace

template <typename DataType>
void PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::Init(
    const ::miopen::conv::ProblemDescription& problem)
{
    valid_kernels = FillValidKernelsByAlphaBeta<DataType>(problem);
    index         = 0;
    kernel_id     = valid_kernels[index];
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::CheckIsSupportCKArgs(
    const ::miopen::conv::ProblemDescription& problem) const
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return IsCKArgsSupported<DeviceOpGBwdBilinearPtrs<DataType>, CKArgs<DataType>>(problem,
                                                                                       kernel_id);
    case SCALE:
        return IsCKArgsSupported<DeviceOpGBwdScalePtrs<DataType>, CKArgs<DataType>>(problem,
                                                                                    kernel_id);
    default:
        return IsCKArgsSupported<DeviceOpGBwdDefaultPtrs<DataType>, CKArgs<DataType>>(problem,
                                                                                      kernel_id);
    }
}

template <typename DataType>
bool ConvHipImplicitGemm3DGroupBwdXdlops::CheckCKApplicability(
    const ::miopen::conv::ProblemDescription& problem) const
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return IsCKApplicable<DeviceOpGBwdBilinearPtrs<DataType>, CKArgs<DataType>>(problem);
    case SCALE: return IsCKApplicable<DeviceOpGBwdScalePtrs<DataType>, CKArgs<DataType>>(problem);
    default: return IsCKApplicable<DeviceOpGBwdDefaultPtrs<DataType>, CKArgs<DataType>>(problem);
    }
}

void PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::InitValidKernels(
    const ::miopen::conv::ProblemDescription& problem)
{
    switch(problem.GetInDataType())
    {
    case miopenHalf: Init<ck::half_t>(problem); break;
    case miopenFloat: Init<float>(problem); break;
    case miopenInt8: Init<int8_t>(problem); break;
    case miopenBFloat16: Init<ck::bhalf_t>(problem); break;
    default: break; // Unsupported data types - valid_kernels remains empty
    }
}
#endif

void PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::HeuristicInit(
    const miopen::ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "None";
    split_k   = 0; // split_k is not used in this solver, but it is required by the interface

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    // 1. AI heuristics (if enabled)
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    if(&ctx != &GetDummyCtx() &&
       !env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR))
    {
        MIOPEN_LOG_I2(
            "Step 1: Attempting AI heuristics for data type: " << problem.GetInDataType());

        std::string solver_name = "ConvHipImplicitGemm3DGroupBwdXdlops";

        bool ai_success = false;
        miopen::ai::tuning::candidate_selection::CandidateSelectionResult result;

        auto run_ai_heuristics = [&](auto CKDataType) {
            using T = decltype(CKDataType);
            auto fill_valid_kernels =
                [=](const ::miopen::conv::ProblemDescription& problem) -> std::vector<std::string> {
                return FillValidKernelsByAlphaBeta<T>(problem);
            };
            return miopen::solver::conv::RunParameterPredictionModel<T>(ctx,
                                                                        problem,
                                                                        valid_kernels,
                                                                        index,
                                                                        split_k,
                                                                        kernel_id,
                                                                        fill_valid_kernels,
                                                                        solver_name);
        };
        switch(problem.GetInDataType())
        {
        case miopenHalf: std::tie(ai_success, result) = run_ai_heuristics(ck::half_t{}); break;
        case miopenFloat: std::tie(ai_success, result) = run_ai_heuristics(float{}); break;
        case miopenBFloat16: std::tie(ai_success, result) = run_ai_heuristics(ck::bhalf_t{}); break;
        default: break;
        }

        if(ai_success && !result.IsEmpty())
        {
            MIOPEN_LOG_I("Step 1: AI heuristics selected kernel: " << kernel_id);
            return;
        }
        else
        {
            MIOPEN_LOG_I2("Step 1: AI heuristics failed, proceeding to default initialization");
            // Continue to default initialization
        }
    }
    else
    {
        MIOPEN_LOG_I2("Step 1: AI heuristics skipped (disabled or dummy context)");
    }
#else
    MIOPEN_LOG_I2("Step 1: AI heuristics not available (MIOPEN_ENABLE_AI_KERNEL_TUNING disabled)");
#endif

    // 2. Default: index remains 0, first valid_kernel will be used
    MIOPEN_LOG_I2("Step 2: Using default initialization (index=0)");
    InitValidKernels(problem);
    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[index];
        MIOPEN_LOG_I("Step 2: Default initialization selected kernel: " << kernel_id
                                                                        << " at index: 0");
    }
    else
    {
        MIOPEN_LOG_W("Step 2: Default initialization failed - no valid kernels found");
    }
#endif
}

bool PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::SetNextValue(
    const ::miopen::conv::ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        // For generic search, we want all available kernels, not heuristic selection
        InitValidKernels(problem);
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
        return false;
}

bool PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::IsValid(
    [[maybe_unused]] const ::miopen::conv::ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetInDataType())
    {
    case miopenHalf: return CheckIsSupportCKArgs<ck::half_t>(problem);
    case miopenFloat: return CheckIsSupportCKArgs<float>(problem);
    case miopenInt8: return CheckIsSupportCKArgs<int8_t>(problem);
    case miopenBFloat16: return CheckIsSupportCKArgs<ck::bhalf_t>(problem);
    case miopenInt64:
    case miopenInt32:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenDouble: break;
    }
#endif
    return false;
}

bool PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemm3DGroupBwdXdlops
ConvHipImplicitGemm3DGroupBwdXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemm3DGroupBwdXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemm3DGroupBwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ::miopen::conv::ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops& config) const
{
    return config.IsValid(problem);
}

size_t ConvHipImplicitGemm3DGroupBwdXdlops::GetWorkspaceSize(
    const ExecutionContext&, const ::miopen::conv::ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemm3DGroupBwdXdlops
ConvHipImplicitGemm3DGroupBwdXdlops::Search(const ExecutionContext& ctx,
                                            const ::miopen::conv::ProblemDescription& problem,
                                            const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemm3DGroupBwdXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ::miopen::conv::ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS))
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.IsDirectionBackwardData())
        return false;
    if(!problem.Is3d())
        return false;
    if(!(problem.IsLayoutNHWC() || problem.IsLayoutDefault()))
        return false;
    // needed because layout transpose kernel does not support non-packed tensors
    if(problem.IsLayoutDefault() && problem.HasNonPackedTensors())
        return false;
    if(!ck_utility::is_ck_whitelist(ctx.GetStream().GetDeviceName()))
        return false;
    switch(problem.GetInDataType())
    {
    case miopenHalf: return CheckCKApplicability<ck::half_t>(problem);
    case miopenFloat: return CheckCKApplicability<float>(problem);
    case miopenInt8: return CheckCKApplicability<int8_t>(problem);
    case miopenBFloat16: return CheckCKApplicability<ck::bhalf_t>(problem);
    case miopenInt64:
    case miopenInt32:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenDouble: break;
    }
#endif
    return false;
}

ConvSolution ConvHipImplicitGemm3DGroupBwdXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ::miopen::conv::ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemm3DGroupBwdXdlops& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeSolutionGroupConvImplicitGemmXdlops(
        problem,
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            switch(problem.GetAlphaBetaCase())
            {
            case BILINEAR:
                return InitInvokerFactoryBwdNCHW<3,
                                                 false,
                                                 DeviceOpGBwdBilinearPtrs<T>,
                                                 CKArgs<T>,
                                                 miopen::conv::DataInvokeParams>(
                    ctx, problem, config.kernel_id);
            case SCALE:
                return InitInvokerFactoryBwdNCHW<3,
                                                 false,
                                                 DeviceOpGBwdScalePtrs<T>,
                                                 CKArgs<T>,
                                                 miopen::conv::DataInvokeParams>(
                    ctx, problem, config.kernel_id);
            default:
                return InitInvokerFactoryBwdNCHW<3,
                                                 false,
                                                 DeviceOpGBwdDefaultPtrs<T>,
                                                 CKArgs<T>,
                                                 miopen::conv::DataInvokeParams>(
                    ctx, problem, config.kernel_id);
            }
        },
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            switch(problem.GetAlphaBetaCase())
            {
            case BILINEAR:
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGBwdBilinearPtrs<T>,
                                              CKArgs<T>,
                                              miopen::conv::DataInvokeParams>(
                    ctx, problem, config.kernel_id);
            case SCALE:
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGBwdScalePtrs<T>,
                                              CKArgs<T>,
                                              miopen::conv::DataInvokeParams>(
                    ctx, problem, config.kernel_id);
            default:
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGBwdDefaultPtrs<T>,
                                              CKArgs<T>,
                                              miopen::conv::DataInvokeParams>(
                    ctx, problem, config.kernel_id);
            }
        });

#else
    return {};
#endif
}

#if !(MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL)
void miopen::solver::conv::PerformanceConfigHipImplicitGemm3DGroupBwdXdlops::InitValidKernels(
    const ::miopen::conv::ProblemDescription&)
{
    // No-op stub for non-CK builds
}
#endif

} // namespace conv
} // namespace solver
} // namespace miopen
