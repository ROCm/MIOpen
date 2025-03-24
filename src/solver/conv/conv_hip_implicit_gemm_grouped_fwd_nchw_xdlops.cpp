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
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_NCHW_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_NCHW_XDLOPS_AI_HEUR)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <typename DataType>
using DeviceOpGFwd = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    2,
    ck::tensor_layout::convolution::NGCHW,
    ck::tensor_layout::convolution::GKYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NGKHW,
    DataType,
    DataType,
    ck::Tuple<>,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

template <typename DataType>
using DeviceOpGFwdPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOpGFwd<DataType>>;

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

        input  = {G, N, C, Hi, Wi};
        output = {G, N, K, Ho, Wo};
        weight = {G, K, C, Y, X};

        // strides from NGCHW to GNCHW layout
        in_strides  = {Hi * Wi * C, Hi * Wi * G * C, Hi * Wi, Wi, 1};
        out_strides = {Ho * Wo * K, Ho * Wo * G * K, Ho * Wo, Wo, 1};
        wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};
        strides     = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation    = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding    = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding    = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr,
                    ConstData_t in,
                    ConstData_t w,
                    Data_t out,
                    float alpha,
                    float beta) const
    {
        (void)alpha;
        (void)beta;
        return conv_ptr->MakeArgumentPointer(in,
                                             w,
                                             {},
                                             out,
                                             input,
                                             in_strides,
                                             weight,
                                             wei_strides,
                                             {},
                                             {},
                                             output,
                                             out_strides,
                                             strides,
                                             dilation,
                                             lPadding,
                                             rPadding,
                                             {},
                                             {},
                                             {});
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

        int dummy_var = 1;
        conv_ptr->SetWorkSpacePointer(arg_ptr.get(), &dummy_var);
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
void PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops::Init(
    const ProblemDescription& problem) // should be parameterized with execution context
{
    if(valid_kernels.empty())
        valid_kernels = FillValidKernelsIDs<DeviceOpGFwdPtrs<DataType>, CKArgs>(problem);
    index     = 0;
    kernel_id = valid_kernels[index];
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpGFwdPtrs<DataType>, CKArgs>(problem, kernel_id);
}

template <typename DataType>
bool ConvHipImplicitGemmGroupFwdCKNCHWXdlops::CheckCKApplicability(
    const ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpGFwdPtrs<DataType>, CKArgs>(problem);
}
#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

bool PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops::IsModelApplicable(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(ctx.GetStream().GetDeviceName() != "gfx90a" && ctx.GetStream().GetDeviceName() != "gfx942")
        return false;
    if(problem.GetInDataType() != miopenFloat && problem.GetInDataType() != miopenHalf &&
       problem.GetInDataType() != miopenBFloat16)
        return false;
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_NCHW_XDLOPS_AI_HEUR))
        return false;
    return true;
}

void PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
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

bool PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops::SetNextValue(
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

bool PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops::IsValid(
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

bool PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops::operator==(
    const PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops
ConvHipImplicitGemmGroupFwdCKNCHWXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemmGroupFwdCKNCHWXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops& config) const
{
    return config.IsValid(problem);
}

size_t
ConvHipImplicitGemmGroupFwdCKNCHWXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                          const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops
ConvHipImplicitGemmGroupFwdCKNCHWXdlops::Search(const ExecutionContext& ctx,
                                                const ProblemDescription& problem,
                                                const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmGroupFwdCKNCHWXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_NCHW_XDLOPS))
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    if(!problem.IsDirectionForward())
        return false;
    if(!problem.Is2d())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(problem.HasMixedDataTypes())
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

ConvSolution ConvHipImplicitGemmGroupFwdCKNCHWXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemmGroupFwdCKNCHWXdlops& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeSolutionGroupConvImplicitGemmCKNCHWXdlops(problem, [&](auto data_type_val) {
        using T = decltype(data_type_val);
        return InitInvokerFactoryNHWC<DeviceOpGFwdPtrs<T>, CKArgs, miopen::conv::DataInvokeParams>(
            ctx, problem, config.kernel_id);
    });
#else
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
