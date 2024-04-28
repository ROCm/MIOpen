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

#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <typename DataType>
using DeviceOpGFwd = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
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
    ck::tensor_operation::element_wise::PassThrough>;

template <typename DataType>
using DeviceOpGFwdPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOpGFwd<DataType>>;

namespace {

template <typename DataType>
struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        G     = ProblemInterpreter::GetGroupCountG(problem);
        N     = ProblemInterpreter::GetBatchN(problem);
        K1    = ProblemInterpreter::GetOutputChannelK(problem);
        C1    = ProblemInterpreter::GetInputChannelC(problem);
        C     = C1 / G; // Number of input Channel per group
        K     = K1 / G; // Number of output Channel per group
        Hi    = ProblemInterpreter::GetInputHeightHi(problem);
        Wi    = ProblemInterpreter::GetInputWidthWi(problem);
        Ho    = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo    = ProblemInterpreter::GetOutputWidthWo(problem);
        Y     = ProblemInterpreter::GetFilterHeightY(problem);
        X     = ProblemInterpreter::GetFilterWidthX(problem);
        Di    = ProblemInterpreter::GetInputDepthDi(problem);
        Do    = ProblemInterpreter::GetOutputDepthDo(problem);
        Z     = ProblemInterpreter::GetFilterDepthZ(problem);
        alpha = ProblemInterpreter::GetAlpha(problem);
        beta  = ProblemInterpreter::GetBeta(problem);

        input  = {G, N, C, Di, Hi, Wi};
        output = {G, N, K, Do, Ho, Wo};
        weight = {G, K, C, Z, Y, X};

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
    auto MakeArgPtr(const ConvPtr& conv_ptr, ConstData_t in, ConstData_t w, Data_t out) const
    {
        // using T = typename ::internel::AlphaBetaTypeMapper<DataType>::type;
        // T alpha_val =  alpha.GetVal<T>();
        // T beta_val  =  beta.GetVal<T>();
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
                                             {} /*Bilinear{alpha_val, beta_val}*/);
    }

    template <typename ConvPtr>
    auto MakeArgPtr(const ConvPtr& conv_ptr, const ConvDataTensors& tensors) const
    {
        return MakeArgPtr(conv_ptr, tensors.in, tensors.w, tensors.out);
    }

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr& conv_ptr) const
    {
        auto arg_ptr = MakeArgPtr(conv_ptr, nullptr, nullptr, nullptr);
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
    Scalar alpha;
    Scalar beta;
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

template <typename DataType>
void PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::Init(const ProblemDescription& problem)
{

    // AlphaBetaType alpha_beta_operation_type = problem.GetEncodedAlphaBeta();

    // Will use alpha_beta_operation_type to select CK solver
    valid_kernels = FillValidKernelsIDs<DeviceOpGFwdPtrs<DataType>, CKArgs<DataType>>(problem);
    index         = 0;
    kernel_id     = valid_kernels[index];
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpGFwdPtrs<DataType>, CKArgs<DataType>>(problem, kernel_id);
}

template <typename DataType>
bool ConvHipImplicitGemm3DGroupFwdXdlops::CheckCKApplicability(
    const ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpGFwdPtrs<DataType>, CKArgs<DataType>>(problem);
}
#endif

void PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::HeuristicInit(
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
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenBFloat16:
    case miopenDouble: break;
    }
#endif
}

bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::SetNextValue(
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

bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemm3DGroupFwdXdlops::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    switch(problem.GetInDataType())
    {
    case miopenHalf: return CheckIsSupportCKArgs<ck::half_t>(problem);
    case miopenFloat: return CheckIsSupportCKArgs<float>(problem);
    case miopenInt8: return CheckIsSupportCKArgs<int8_t>(problem);
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenBFloat16:
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
    const ExecutionContext&, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemm3DGroupFwdXdlops pp;
    pp.HeuristicInit(problem);
    return pp;
}

bool ConvHipImplicitGemm3DGroupFwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops& config) const
{
    return config.IsValid(problem);
}

size_t
ConvHipImplicitGemm3DGroupFwdXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                      const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemm3DGroupFwdXdlops
ConvHipImplicitGemm3DGroupFwdXdlops::Search(const ExecutionContext& ctx,
                                            const ProblemDescription& problem,
                                            const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemm3DGroupFwdXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(miopen::IsDisabled(ENV(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)))
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
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenBFloat16:
    case miopenDouble: break;
    }
#endif
    return false;
}

ConvSolution ConvHipImplicitGemm3DGroupFwdXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemm3DGroupFwdXdlops& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeSolutionGroupConvImplicitGemmXdlops(
        problem,
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            return InitInvokerFactoryFwdNCHW<3,
                                             DeviceOpGFwdPtrs<T>,
                                             CKArgs<T>,
                                             miopen::conv::DataInvokeParams>(
                ctx, problem, config.kernel_id);
        },
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            return InitInvokerFactoryNHWC<DeviceOpGFwdPtrs<T>,
                                          CKArgs<T>,
                                          miopen::conv::DataInvokeParams>(
                ctx, problem, config.kernel_id);
        });

#else
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
