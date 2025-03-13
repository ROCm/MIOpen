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

#include <miopen/conv/solvers.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_AI_HEUR)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <typename DataType>
using DeviceOpGFwd = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
    2,
    ck::tensor_layout::convolution::NHWGC,
    ck::tensor_layout::convolution::GKYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::NHWGK,
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

        // strides from NHWGC to GNCHW laout
        in_strides  = {C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
        out_strides = {K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
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
void PerformanceConfigHipImplicitGemmGroupFwdXdlops::Init(
    const ProblemDescription& problem) // should be parameterized with execution context
{
    if(valid_kernels.empty())
        valid_kernels = FillValidKernelsIDs<DeviceOpGFwdPtrs<DataType>, CKArgs>(problem);
    index     = 0;
    kernel_id = valid_kernels[index];
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpGFwdPtrs<DataType>, CKArgs>(problem, kernel_id);
}

template <typename DataType>
bool ConvHipImplicitGemmGroupFwdXdlops::CheckCKApplicability(
    const ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpGFwdPtrs<DataType>, CKArgs>(problem);
}

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
static std::vector<std::string> GetKernelAsTokens(const std::string& kernel)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(
        kernel.substr(kernel.find('<') + 1, kernel.find('>') - kernel.find('<') - 1));
    while(std::getline(tokenStream, token, ','))
    {
        token.erase(remove_if(token.begin(), token.end(), isspace),
                    token.end()); // strip whitespace
        tokens.push_back(token);
    }
    return tokens;
}

void PerformanceConfigHipImplicitGemmGroupFwdXdlops::InitHeuristicKernelIDs(const std::string& type)
{
    for(int i = 0; i < valid_kernels.size(); i++)
    {
        if(valid_kernels[i].find(type) != std::string::npos)
        {
            heuristic_indexes.push_back(i);
            heuristic_kernels[i] = GetKernelAsTokens(valid_kernels[i]);
        }
    }
}

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::ModelApplyToken(int idx,
                                                                     std::string value,
                                                                     const std::string& arch)
{
    if(arch == "gfx90a")
    {
        if(idx >= 5)
        {
            idx += 2; // skip MPerXDL and NPerXDL as they are constant
        }
    }
    if(idx == 0 && arch == "gfx942")
    {
        InitHeuristicKernelIDs(value);
        if(!heuristic_indexes.empty())
            return true;
        return false;
    }
    if(idx >= 1 && arch == "gfx942")
        idx--;
    auto eraseBegin = std::remove_if(
        heuristic_indexes.begin(), heuristic_indexes.end(), [&](int heuristic_index) {
            return heuristic_kernels[heuristic_index][idx] != value;
        });

    if(eraseBegin != heuristic_indexes.begin())
    {
        heuristic_indexes.erase(eraseBegin, heuristic_indexes.end());
        return true;
    }
    return false;
}

static std::vector<float>
GetFeatures(const ProblemDescription& problem, std::size_t num_cu, const std::string& arch)
{
    if(arch == "gfx90a")
    {
        std::size_t n = 18;
        std::vector<float> features(n, 0.0f);
        features[0]  = problem.GetInDataType() == miopenFloat ? 2 : 1;
        features[1]  = problem.GetInChannels();
        features[2]  = problem.GetInHeight();
        features[3]  = problem.GetInWidth();
        features[4]  = problem.GetOutChannels();
        features[5]  = problem.GetOutHeight();
        features[6]  = problem.GetOutWidth();
        features[7]  = problem.GetWeightsHeight();
        features[8]  = problem.GetWeightsWidth();
        features[9]  = problem.GetPadH();
        features[10] = problem.GetPadW();
        features[11] = problem.GetKernelStrideH();
        features[12] = problem.GetKernelStrideW();
        features[13] = problem.GetDilationH();
        features[14] = problem.GetDilationW();
        features[15] = problem.GetBatchSize();
        features[16] = problem.GetGroupCount();
        features[17] = num_cu;
        return features;
    }
    std::size_t n = 17;
    std::vector<float> features(n * n, 0.0f);
    features[0]           = problem.GetInChannels();
    features[n + 1]       = problem.GetInHeight();
    features[2 * n + 2]   = problem.GetInWidth();
    features[3 * n + 3]   = problem.GetOutChannels();
    features[4 * n + 4]   = problem.GetOutHeight();
    features[5 * n + 5]   = problem.GetOutWidth();
    features[6 * n + 6]   = problem.GetWeightsHeight();
    features[7 * n + 7]   = problem.GetWeightsWidth();
    features[8 * n + 8]   = problem.GetPadH();
    features[9 * n + 9]   = problem.GetPadW();
    features[10 * n + 10] = problem.GetKernelStrideH();
    features[11 * n + 11] = problem.GetKernelStrideW();
    features[12 * n + 12] = problem.GetDilationH();
    features[13 * n + 13] = problem.GetDilationW();
    features[14 * n + 14] = problem.GetBatchSize();
    features[15 * n + 15] = problem.GetInDataType() == miopenFloat ? 2.0 : 1.0;
    features[16 * n + 16] = problem.GetGroupCount();
    return features;
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::RunParameterPredictionModel(
    const ExecutionContext& ctx, const ProblemDescription& problem)
{
    valid_kernels = FillValidKernelsIDs<DeviceOpGFwdPtrs<DataType>, CKArgs>(
        problem); // filter valid_kernel ID's
    static const std::string& arch = ctx.GetStream().GetDeviceName();
    if(arch == "gfx90a")
        InitHeuristicKernelIDs("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle");
    static const std::string solver = "ConvHipIgemmGroupFwdXdlops";
    std::vector<float> features = GetFeatures(problem, ctx.GetStream().GetMaxComputeUnits(), arch);
    bool transform              = (arch == "gfx90a") ? false : true;
    if(ai::tuning::ModelSetParams(arch,
                                  solver,
                                  problem.GetDirection(),
                                  features,
                                  transform,
                                  [&](int idx, const std::string& value) {
                                      return this->ModelApplyToken(idx, value, arch);
                                  }))
    {
        index     = heuristic_indexes[0];
        kernel_id = valid_kernels[index];
        MIOPEN_LOG_I("Params set by AI: " << ToString());
        return true;
    }
    return false;
}
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::IsModelApplicable(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(ctx.GetStream().GetDeviceName() != "gfx90a" && ctx.GetStream().GetDeviceName() != "gfx942")
        return false;
    if(problem.GetInDataType() != miopenFloat && problem.GetInDataType() != miopenHalf)
        return false;
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_AI_HEUR))
        return false;
    return true;
}

void PerformanceConfigHipImplicitGemmGroupFwdXdlops::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    if(IsModelApplicable(ctx, problem))
    {
        if(problem.GetInDataType() == miopenFloat)
        {
            if(RunParameterPredictionModel<float>(ctx, problem))
                return;
        }
        else
        {
            if(RunParameterPredictionModel<ck::half_t>(ctx, problem))
                return;
        }
    }
#endif
    switch(problem.GetInDataType())
    {
    case miopenHalf: Init<ck::half_t>(problem); break;
    case miopenFloat: Init<float>(problem); break;
    case miopenInt8: Init<int8_t>(problem); break;
    case miopenBFloat16: Init<ck::bhalf_t>(problem); break;
    case miopenInt64:
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble: break;
    }
#endif
}

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::SetNextValue(const ProblemDescription& problem)
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
        case miopenFloat8:
        case miopenBFloat8:
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

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::IsValid(
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
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble: break;
    }
#endif
    return false;
}

bool PerformanceConfigHipImplicitGemmGroupFwdXdlops::operator==(
    const PerformanceConfigHipImplicitGemmGroupFwdXdlops& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmGroupFwdXdlops
ConvHipImplicitGemmGroupFwdXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmGroupFwdXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemmGroupFwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmGroupFwdXdlops& config) const
{
    return config.IsValid(problem);
}

size_t ConvHipImplicitGemmGroupFwdXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                           const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemmGroupFwdXdlops
ConvHipImplicitGemmGroupFwdXdlops::Search(const ExecutionContext& ctx,
                                          const ProblemDescription& problem,
                                          const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmGroupFwdXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS))
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
    if(!problem.IsDirectionForward())
        return false;
    if(!problem.Is2d())
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
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble: break;
    }
#endif
    return false;
}

ConvSolution ConvHipImplicitGemmGroupFwdXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemmGroupFwdXdlops& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeSolutionGroupConvImplicitGemmXdlops(
        problem,
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            return InitInvokerFactoryFwdNCHW<2,
                                             DeviceOpGFwdPtrs<T>,
                                             CKArgs,
                                             miopen::conv::DataInvokeParams>(
                ctx, problem, config.kernel_id);
        },
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            return InitInvokerFactoryNHWC<DeviceOpGFwdPtrs<T>,
                                          CKArgs,
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
