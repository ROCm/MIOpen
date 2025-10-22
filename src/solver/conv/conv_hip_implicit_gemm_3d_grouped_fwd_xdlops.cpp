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
#include <optional>
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_IDX_OVERRIDE);
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_AI_HEUR)

#include <miopen/conv/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp>
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"
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
using DeviceOpGFwdBilinear =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<OutLayout>,
                                                                  OutLayout,
                                                                  DataType,
                                                                  DataType,
                                                                  ck::Tuple<DataType>,
                                                                  DataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  Bilinear>;

template <typename DataType>
using DeviceOpGFwdScale =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<>,
                                                                  OutLayout,
                                                                  DataType,
                                                                  DataType,
                                                                  ck::Tuple<>,
                                                                  DataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  Scale>;

template <typename DataType>
using DeviceOpGFwdDefault =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<>,
                                                                  OutLayout,
                                                                  DataType,
                                                                  DataType,
                                                                  ck::Tuple<>,
                                                                  DataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>;

template <typename DataType>
using DeviceOpGFwdBilinearPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdBilinear<DataType>>;

template <typename DataType>
using DeviceOpGFwdScalePtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdScale<DataType>>;

template <typename DataType>
using DeviceOpGFwdDefaultPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGFwdDefault<DataType>>;

namespace {

template <typename DataType>
struct CKArgs
{
    CKArgs(const ::miopen::conv::ProblemDescription& problem)
    {
        G               = ProblemInterpreter::GetGroupCountG(problem);
        N               = ProblemInterpreter::GetBatchN(problem);
        K1              = ProblemInterpreter::GetOutputChannelK(problem);
        C1              = ProblemInterpreter::GetInputChannelC(problem);
        C               = C1 / G; // Number of input Channel per group
        K               = K1 / G; // Number of output Channel per group
        Hi              = ProblemInterpreter::GetInputHeightHi(problem);
        Wi              = ProblemInterpreter::GetInputWidthWi(problem);
        Ho              = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo              = ProblemInterpreter::GetOutputWidthWo(problem);
        Y               = ProblemInterpreter::GetFilterHeightY(problem);
        X               = ProblemInterpreter::GetFilterWidthX(problem);
        Di              = ProblemInterpreter::GetInputDepthDi(problem);
        Do              = ProblemInterpreter::GetOutputDepthDo(problem);
        Z               = ProblemInterpreter::GetFilterDepthZ(problem);
        alpha_beta_case = ProblemInterpreter::GetAlphaBetaCase(problem);

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

    CKArgs(const CKArgs&)     = default;
    CKArgs(CKArgs&&) noexcept = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    ConstData_t in,
                    ConstData_t w,
                    Data_t out,
                    float alpha,
                    float beta) const
    {
        using DeviceP = std::remove_pointer_t<decltype(conv_ptr.get())>;
        if constexpr(std::is_same_v<DeviceP, DeviceOpGFwdBilinear<DataType>>)
        {
            return MakeBilinearArgPtr(conv_ptr, in, w, out, alpha, beta);
        }
        else if constexpr(std::is_same_v<DeviceP, DeviceOpGFwdScale<DataType>>)
        {
            (void)beta;
            return MakeScaleArgPtr(conv_ptr, in, w, out, alpha);
        }
        else
        {
            (void)alpha;
            (void)beta;
            static_assert(std::is_same_v<DeviceP, DeviceOpGFwdDefault<DataType>>,
                          "Default should be fwd pass through");
            return MakeDefaultArgPtr(conv_ptr, in, w, out);
        }
    }

    template <typename ConvPtr>
    auto MakeBilinearArgPtr(const ConvPtr& conv_ptr,
                            ConstData_t in,
                            ConstData_t w,
                            Data_t out,
                            float alpha,
                            float beta) const
    {
        return conv_ptr->MakeArgumentPointer(in,
                                             w,
                                             {out},
                                             out,
                                             in_lengths,
                                             in_strides,
                                             wei_lengths,
                                             wei_strides,
                                             {out_lengths},
                                             {out_strides},
                                             out_lengths,
                                             out_strides,
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
        const ConvPtr& conv_ptr, ConstData_t in, ConstData_t w, Data_t out, float alpha) const
    {
        return conv_ptr->MakeArgumentPointer(in,
                                             w,
                                             {},
                                             out,
                                             in_lengths,
                                             in_strides,
                                             wei_lengths,
                                             wei_strides,
                                             {},
                                             {},
                                             out_lengths,
                                             out_strides,
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             PassThrough{},
                                             Scale{alpha});
    }

    template <typename ConvPtr>
    auto MakeDefaultArgPtr(const ConvPtr& conv_ptr, ConstData_t in, ConstData_t w, Data_t out) const
    {
        return conv_ptr->MakeArgumentPointer(in,
                                             w,
                                             {},
                                             out,
                                             in_lengths,
                                             in_strides,
                                             wei_lengths,
                                             wei_strides,
                                             {},
                                             {},
                                             out_lengths,
                                             out_strides,
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
        return MakeArgPtr(conv_ptr, tensors.in, tensors.w, tensors.out, alpha, beta);
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
    miopenAlphaBetaCase_t alpha_beta_case;
};

template <typename DataType>
std::vector<std::string>
FillValidKernelsByAlphaBeta(const ::miopen::conv::ProblemDescription& problem)
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return miopen::solver::FillValidKernelsIDs<DeviceOpGFwdBilinearPtrs<DataType>,
                                                   CKArgs<DataType>>(problem);
    case SCALE:
        return miopen::solver::FillValidKernelsIDs<DeviceOpGFwdScalePtrs<DataType>,
                                                   CKArgs<DataType>>(problem);
    default:
        return miopen::solver::FillValidKernelsIDs<DeviceOpGFwdDefaultPtrs<DataType>,
                                                   CKArgs<DataType>>(problem);
    }
}
} // namespace

template <typename DataType>
void PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::Init(
    const ::miopen::conv::ProblemDescription& problem)
{
    valid_kernels = FillValidKernelsByAlphaBeta<DataType>(problem);
    index         = 0;
    kernel_id     = valid_kernels[index];
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::CheckIsSupportCKArgs(
    const ::miopen::conv::ProblemDescription& problem) const
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return IsCKArgsSupported<DeviceOpGFwdBilinearPtrs<DataType>, CKArgs<DataType>>(problem,
                                                                                       kernel_id);
    case SCALE:
        return IsCKArgsSupported<DeviceOpGFwdScalePtrs<DataType>, CKArgs<DataType>>(problem,
                                                                                    kernel_id);
    default:
        return IsCKArgsSupported<DeviceOpGFwdDefaultPtrs<DataType>, CKArgs<DataType>>(problem,
                                                                                      kernel_id);
    }
}

template <typename DataType>
bool ConvHipImplicitGemm3DGroupFwdXdlops::CheckCKApplicability(
    const ::miopen::conv::ProblemDescription& problem) const
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return IsCKApplicable<DeviceOpGFwdBilinearPtrs<DataType>, CKArgs<DataType>>(problem);
    case SCALE: return IsCKApplicable<DeviceOpGFwdScalePtrs<DataType>, CKArgs<DataType>>(problem);
    default: return IsCKApplicable<DeviceOpGFwdDefaultPtrs<DataType>, CKArgs<DataType>>(problem);
    }
}

void PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::InitValidKernels(
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

void PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::HeuristicInit(
    const miopen::ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "None";
    split_k   = 0; // split_k is not used in this solver, but it is required by the interface

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    // 1. IDX_OVERRIDE is preferred
    auto idx_override = env::value(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_IDX_OVERRIDE);
    if(idx_override != 0)
    {
        MIOPEN_LOG_I2("Step 1: Attempting index override with value: " << idx_override);
        switch(problem.GetInDataType())
        {
        case miopenHalf: Init<ck::half_t>(problem); break;
        case miopenBFloat16: Init<ck::bhalf_t>(problem); break;
        default: break;
        }

        if(idx_override < valid_kernels.size())
        {
            index     = idx_override;
            kernel_id = valid_kernels[index];
            MIOPEN_LOG_I("Step 1: Index override selected kernel: " << kernel_id
                                                                    << " at index: " << index);
            return;
        }
        else
        {
            MIOPEN_LOG_W("Step 1: Index override failed, index "
                         << idx_override << " out of range, proceeding to next step");
            // Continue to hard-coded heuristics
        }
    }
    else
    {
        MIOPEN_LOG_I2("Step 1: Index override not set, proceeding to next step");
    }

    // 2. Hard-coded heuristics for BF16/FP16 on gfx942 only
    if((problem.GetInDataType() == miopenBFloat16 || problem.GetInDataType() == miopenHalf) &&
       ctx.GetStream().GetDeviceName() == "gfx942")
    {
        MIOPEN_LOG_I2("Step 2: Attempting hard-coded heuristics for "
                      << (problem.GetInDataType() == miopenBFloat16 ? "BF16" : "FP16")
                      << " on gfx942");

        switch(problem.GetInDataType())
        {
        case miopenHalf: Init<ck::half_t>(problem); break;
        case miopenBFloat16: Init<ck::bhalf_t>(problem); break;
        default: break;
        }

        auto find_kernel = [&valid_kernels = std::as_const(valid_kernels)](
                               const std::size_t& expected_index,
                               const std::string& kernel_id) -> std::optional<std::size_t> {
            if(expected_index < valid_kernels.size() && valid_kernels[expected_index] == kernel_id)
                return expected_index;
            auto it = std::find(valid_kernels.begin(), valid_kernels.end(), kernel_id);
            if(it != valid_kernels.end())
                return static_cast<std::size_t>(it - valid_kernels.begin());
            MIOPEN_LOG_I2("Hard-coded heuristics did not find kernel: " << kernel_id);
            return std::nullopt;
        };

        if(problem.GetInChannels() > 8 && problem.GetGroupCount() == 1 &&
           problem.GetAlphaBetaCase() == DEFAULT)
        {
            int K = problem.GetOutChannels();
            std::optional<std::size_t> found_index;

            if(problem.GetInDataType() == miopenBFloat16)
            {
                if(K < 64)
                {
                    found_index = find_kernel(
                        38,
                        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"
                        "<256, 64, 64, 64, Default, 32, 32, 1, 1, 8, 8, 8, 1, 1, "
                        "BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>");
                }
                else
                {
                    found_index = find_kernel(
                        30,
                        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"
                        "<256, 128, 128, 64, Default, 32, 32, 2, 2, 8, 8, 8, 1, 1, "
                        "BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>");
                }
            }
            else if(problem.GetInDataType() == miopenHalf)
            {
                if(K < 64)
                {
                    found_index = find_kernel(
                        57,
                        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"
                        "<64, 16, 16, 128, Default, 16, 16, 1, 1, 8, 8, 4, 1, 1, "
                        "BlkGemmPipelineScheduler: Interwave, BlkGemmPipelineVersion: v1>");
                }
                else
                {
                    found_index = find_kernel(
                        31,
                        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"
                        "<256, 128, 128, 64, Default, 32, 32, 2, 2, 8, 8, 8, 1, 1, "
                        "BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v3>");
                }
            }

            if(found_index.has_value())
            {
                index     = found_index.value();
                kernel_id = valid_kernels[index];
                MIOPEN_LOG_I("Step 2: Hard-coded heuristics selected kernel: "
                             << kernel_id << " at index: " << index);
                return;
            }
        }

        MIOPEN_LOG_I2(
            "Step 2: Hard-coded heuristics did not select a kernel, proceeding to next step");
        // Continue to AI heuristics
    }
    else
    {
        MIOPEN_LOG_I2("Step 2: Hard-coded heuristics skipped (data type: "
                      << problem.GetInDataType() << ", device: " << ctx.GetStream().GetDeviceName()
                      << ")");
    }

    // 3. AI heuristics (if enabled)
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    if(&ctx != &GetDummyCtx() &&
       !env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_AI_HEUR))
    {
        MIOPEN_LOG_I2(
            "Step 3: Attempting AI heuristics for data type: " << problem.GetInDataType());

        std::string solver_name = "ConvHipImplicitGemm3DGroupFwdXdlops";

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
            MIOPEN_LOG_I("Step 3: AI heuristics selected kernel: " << kernel_id);
            return;
        }
        else
        {
            MIOPEN_LOG_I2("Step 3: AI heuristics failed, proceeding to default initialization");
            // Continue to default initialization
        }
    }
    else
    {
        MIOPEN_LOG_I2("Step 3: AI heuristics skipped (disabled or dummy context)");
    }
#else
    MIOPEN_LOG_I2("Step 3: AI heuristics not available (MIOPEN_ENABLE_AI_KERNEL_TUNING disabled)");
#endif

    // 4. Default: index remains 0, first valid_kernel will be used
    MIOPEN_LOG_I2("Step 4: Using default initialization (index=0)");
    InitValidKernels(problem);
    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[index];
        MIOPEN_LOG_I("Step 4: Default initialization selected kernel: " << kernel_id
                                                                        << " at index: 0");
    }
    else
    {
        MIOPEN_LOG_W("Step 4: Default initialization failed - no valid kernels found");
    }
#endif
}

bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::SetNextValue(
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

bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::IsValid(
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

bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
ConvHipImplicitGemm3DGroupFwdXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ::miopen::conv::ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemm3DGroupFwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ::miopen::conv::ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops& config) const
{
    return config.IsValid(problem);
}

size_t ConvHipImplicitGemm3DGroupFwdXdlops::GetWorkspaceSize(
    const ExecutionContext&, const ::miopen::conv::ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
ConvHipImplicitGemm3DGroupFwdXdlops::Search(const ExecutionContext& ctx,
                                            const ::miopen::conv::ProblemDescription& problem,
                                            const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemm3DGroupFwdXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ::miopen::conv::ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.IsDirectionForward())
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

float ConvHipImplicitGemm3DGroupFwdXdlops::GetWti(
    const ExecutionContext&, const ::miopen::conv::ProblemDescription& problem) const
{
    decltype(auto) xDesc = problem.GetIn();
    decltype(auto) wDesc = problem.GetWeights();

    if(xDesc.GetType() == miopenHalf || xDesc.GetType() == miopenBFloat16)
    {
        std::size_t in_n, in_c, w_x, w_y, w_d;
        std::tie(in_n, in_c)    = tie_pick<0, 1>()(xDesc.GetLengths());
        std::tie(w_x, w_y, w_d) = tie_pick<2, 3, 4>()(wDesc.GetLengths());
        // For cases where the filter shape is not 1x1x1 and the input channel (in_c) is greater
        // than 8, CK's implementation offers better performance.
        if((w_x == 1 && w_y == 1 && w_d == 1) == false)
        {
            if(in_c < 8 && in_n < 4)
            {
                return 0.00002; // force disable
            }
            else
            {
                return 1.0; // force enable
            }
        }
    }
    return 0.02f;
}

ConvSolution ConvHipImplicitGemm3DGroupFwdXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ::miopen::conv::ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeSolutionGroupConvImplicitGemmXdlops(
        problem,
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            switch(problem.GetAlphaBetaCase())
            {
            case BILINEAR:
                return InitInvokerFactoryFwdNCHW<3,
                                                 false,
                                                 DeviceOpGFwdBilinearPtrs<T>,
                                                 CKArgs<T>,
                                                 miopen::conv::DataInvokeParams>(
                    ctx, problem, config.kernel_id);
            case SCALE:
                return InitInvokerFactoryFwdNCHW<3,
                                                 false,
                                                 DeviceOpGFwdScalePtrs<T>,
                                                 CKArgs<T>,
                                                 miopen::conv::DataInvokeParams>(
                    ctx, problem, config.kernel_id);
            default:
                return InitInvokerFactoryFwdNCHW<3,
                                                 false,
                                                 DeviceOpGFwdDefaultPtrs<T>,
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
                                              DeviceOpGFwdBilinearPtrs<T>,
                                              CKArgs<T>,
                                              miopen::conv::DataInvokeParams>(
                    ctx, problem, config.kernel_id);
            case SCALE:
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGFwdScalePtrs<T>,
                                              CKArgs<T>,
                                              miopen::conv::DataInvokeParams>(
                    ctx, problem, config.kernel_id);
            default:
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGFwdDefaultPtrs<T>,
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
void miopen::solver::conv::PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::InitValidKernels(
    const ::miopen::conv::ProblemDescription&)
{
    // No-op stub for non-CK builds
}
#endif

} // namespace conv
} // namespace solver
} // namespace miopen
