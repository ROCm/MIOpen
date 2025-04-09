/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/conv/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_NCHW_XDLOPS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

namespace {

struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
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
        data_type       = ProblemInterpreter::GetOutputDataType(problem);
        alpha_beta_case = ProblemInterpreter::GetAlphaBetaCase(problem);
        input           = {G, N, C, Hi, Wi};
        output          = {G, N, K, Ho, Wo};
        weight          = {G, K, C, Y, X};

        in_strides  = {Hi * Wi * C, Hi * Wi * G * C, Hi * Wi, Wi, 1};
        out_strides = {Ho * Wo * K, Ho * Wo * G * K, Ho * Wo, Wo, 1};
        wei_strides = {K * C * Y * X, C * Y * X, Y * X, X, 1};

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
                    ConstData_t x,
                    Data_t dw,
                    ConstData_t dy,
                    float alpha,
                    float beta,
                    int split_k) const
    {
        (void)alpha;
        (void)beta;
        return conv_ptr->MakeArgumentPointer(x,
                                             dw,
                                             dy,
                                             input,
                                             in_strides,
                                             weight,
                                             wei_strides,
                                             output,
                                             out_strides,
                                             strides,
                                             dilation,
                                             lPadding,
                                             rPadding,
                                             {},
                                             {},
                                             {},
                                             split_k);
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    const ConvWrwTensors& tensors,
                    float alpha,
                    float beta,
                    int split_k) const

    {
        return MakeArgPtr(conv_ptr, tensors.x, tensors.dw, tensors.dy, alpha, beta, split_k);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f, 1);
        // Creat dummy workspace to pass the ck IsSupportedArgument check.

        int dummy_var = 1;
        conv_ptr->SetWorkSpacePointer(arg_ptr.get(), &dummy_var);

        return conv_ptr->IsSupportedArgument(arg_ptr.get());
    }

    template <typename ConvPtr>
    bool IsSupportedBySplitK(const ConvPtr& conv_ptr, int split_k) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr, 1.0f, 0.0f, split_k);

        // wrw solver with CK NCHW instances always require set workspace
        int dummy_var = 1;
        conv_ptr->SetWorkSpacePointer(arg_ptr.get(), &dummy_var);
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
    int Ho;
    int Wo;
    int Y;
    int X;
    // ck::index_t split_k = 1;
    miopenDataType_t data_type;
    miopenAlphaBetaCase_t alpha_beta_case;
    std::array<ck::index_t, 5> input;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> output;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> weight;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 2> strides;
    std::array<ck::index_t, 2> dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};
} // namespace

template <typename DataType>
void PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops::Init(const ProblemDescription& problem)
{
    valid_kernels = FillValidKernelsIDs<DeviceOpGWrwNCHWPtrs<DataType>, CKArgs>(problem);
    index         = 0;
    split_k       = 1;
    kernel_id     = valid_kernels[index] + "+" + std::to_string(split_k);
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpGWrwNCHWPtrs<DataType>, CKArgs>(problem, kernel_id);
}

template <typename DataType>
bool ConvHipImplicitGemmGroupWrwCKNCHWXdlops::CheckCKApplicability(
    const ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpGWrwNCHWPtrs<DataType>, CKArgs>(problem);
}
#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

void PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    // these seem redundant
    split_k   = 1;
    index     = 0;
    kernel_id = "";
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetInDataType())
    {
    case miopenHalf: Init<ck::half_t>(problem); break;
    case miopenFloat: Init<float>(problem); break;
    case miopenInt8: Init<int8_t>(problem); break;
    case miopenBFloat16: Init<ck::bhalf_t>(problem); break;
    case miopenInt64:
    case miopenInt32:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenDouble: break;
    }
#endif
}

bool PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops::SetNextValue(
    const ProblemDescription& problem)
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(valid_kernels.empty())
    {
        switch(problem.GetInDataType())
        {
        case miopenHalf: Init<ck::half_t>(problem); break;
        case miopenFloat: Init<float>(problem); break;
        case miopenInt8: Init<int8_t>(problem); break;
        case miopenBFloat16: Init<ck::bhalf_t>(problem); break;
        case miopenInt64:
        case miopenInt32:
        case miopenFloat8_fnuz:
        case miopenBFloat8_fnuz:
        case miopenDouble: break;
        }
        assert(!valid_kernels.empty());
        return true;
    }
    do
    {
        bool flag = NextTwoPower<1, 128>(split_k);
        if(!flag)
        {
            kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
            break;
        }

        if(!NextLinear(0, valid_kernels.size() - 1, index))
        {
            kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
            break;
        }
        // All split_k and index values were iterated
        return false;
    } while(false);

#endif
    return true;
}

bool PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
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

bool PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops::operator==(
    const PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops
ConvHipImplicitGemmGroupWrwCKNCHWXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemmGroupWrwCKNCHWXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops& config) const
{
    return config.IsValid(problem);
}

size_t
ConvHipImplicitGemmGroupWrwCKNCHWXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                          const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops
ConvHipImplicitGemmGroupWrwCKNCHWXdlops::Search(const ExecutionContext& ctx,
                                                const ProblemDescription& problem,
                                                const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmGroupWrwCKNCHWXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_NCHW_XDLOPS))
        return false;
    if(env::enabled(MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC))
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(!problem.IsDirectionBackwardWrW())
        return false;
    if(!problem.Is2d())
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    // needed because layout transpose kernel does not support non-packed tensors
    if(problem.HasNonPackedTensors())
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

ConvSolution ConvHipImplicitGemmGroupWrwCKNCHWXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemmGroupWrwCKNCHWXdlops& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeSolutionGroupConvImplicitGemmNCHWXdlops(problem, [&](auto data_type_val) {
        using T = decltype(data_type_val);
        return InitInvokerFactoryNHWC<DeviceOpGWrwNCHWPtrs<T>,
                                      CKArgs,
                                      miopen::conv::WrWInvokeParams>(
            ctx, problem, config.kernel_id);
    });

#else
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
