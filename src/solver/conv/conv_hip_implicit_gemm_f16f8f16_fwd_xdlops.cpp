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

#include <miopen/config.h>
#include <miopen/conv/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/env.hpp>
#include <vector>
#include <cstdint>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_F16F8F16_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_USE_COMPOSABLEKERNEL
using d_type = ck::half_t;
using c_type = ck::f8_t;

template <typename DataType, typename ComputeType>
using DeviceOpF8Fwd = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    3,
    ck::tensor_layout::convolution::NDHWGC,
    ck::tensor_layout::convolution::GKZYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NDHWGK,
    DataType,
    DataType,
    ck::Tuple<>,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ComputeType>;

template <typename DataType, typename ComputeType>
using DeviceOpF8FwdPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    DeviceOpF8Fwd<DataType, ComputeType>>;

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

        // miopen strides to CK strides
        auto miopen_in_strides  = problem.GetIn().GetStrides();
        auto miopen_out_strides = problem.GetOut().GetStrides();
        auto miopen_wei_strides = problem.GetWeights().GetStrides();
        miopen_in_strides.insert(miopen_in_strides.begin(), C);
        miopen_out_strides.insert(miopen_out_strides.begin(), K);
        miopen_wei_strides.insert(miopen_wei_strides.begin(), K * miopen_wei_strides[0]);
        std::copy(miopen_in_strides.begin(), miopen_in_strides.end(), in_strides.begin());
        std::copy(miopen_out_strides.begin(), miopen_out_strides.end(), out_strides.begin());
        std::copy(miopen_wei_strides.begin(), miopen_wei_strides.end(), wei_strides.begin());

        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideD(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationD(problem),
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
    std::array<ck::index_t, 6> input;
    std::array<ck::index_t, 6> in_strides;
    std::array<ck::index_t, 6> output;
    std::array<ck::index_t, 6> out_strides;
    std::array<ck::index_t, 6> weight;
    std::array<ck::index_t, 6> wei_strides;
    std::array<ck::index_t, 3> strides;
    std::array<ck::index_t, 3> dilation;
    std::array<ck::index_t, 3> lPadding;
    std::array<ck::index_t, 3> rPadding;
};
} // namespace

template <typename DataType, typename ComputeType>
void PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops::Init(const ProblemDescription& problem)
{
    valid_kernels = FillValidKernelsIDs<DeviceOpF8FwdPtrs<DataType, ComputeType>, CKArgs>(problem);
    index         = 0;
    kernel_id     = valid_kernels[index];
}

template <typename DataType, typename ComputeType>
bool PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpF8FwdPtrs<DataType, ComputeType>, CKArgs>(problem, kernel_id);
}

template <typename DataType, typename ComputeType>
bool ConvHipImplicitGemmF16F8F16FwdXdlops::CheckCKApplicability(
    const ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpF8FwdPtrs<DataType, ComputeType>, CKArgs>(problem);
}
#endif

void PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops::HeuristicInit(
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";

#if MIOPEN_USE_COMPOSABLEKERNEL
    if(problem.GetIn().GetCastType() == miopenFloat8 &&
       problem.GetWeights().GetCastType() == miopenFloat8)
        Init<ck::half_t, c_type>(problem);
#endif
}

bool PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops::SetNextValue(
    const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        HeuristicInit(problem);
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

bool PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(problem.GetIn().GetCastType() == miopenFloat8 &&
       problem.GetWeights().GetCastType() == miopenFloat8)
        return CheckIsSupportCKArgs<ck::half_t, c_type>(problem);
#endif
    return false;
}

bool PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops
ConvHipImplicitGemmF16F8F16FwdXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext&, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops pp;
    pp.HeuristicInit(problem);
    return pp;
}

bool ConvHipImplicitGemmF16F8F16FwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops& config) const
{
    return config.IsValid(problem);
}

PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops
ConvHipImplicitGemmF16F8F16FwdXdlops::Search(const ExecutionContext& ctx,
                                             const ProblemDescription& problem,
                                             const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmF16F8F16FwdXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_F16F8F16_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS))
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(!problem.IsTensorsCasted())
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(!problem.IsDirectionForward())
        return false;
    if(!problem.IsLayoutNHWC())
        return false;
    if(!problem.IsFp16())
        return false;
    if(!ck_utility::is_ck_whitelist(ctx.GetStream().GetDeviceName()))
        return false;
    if(problem.GetIn().GetCastType() == miopenFloat8 &&
       problem.GetWeights().GetCastType() == miopenFloat8)
        return CheckCKApplicability<ck::half_t, c_type>(problem);
#endif
    return false;
}

ConvSolution ConvHipImplicitGemmF16F8F16FwdXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemmF16F8F16FwdXdlops& config) const
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    return InitInvokerFactoryNHWC<DeviceOpF8FwdPtrs<ck::half_t, c_type>,
                                  CKArgs,
                                  miopen::conv::DataInvokeParams>(ctx, problem, config.kernel_id);
#else
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
