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
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS_AI_HEUR)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

template <typename DataType>
using DeviceOpGWrwPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOpGWrw<DataType>>;

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
            in_strides  = {C, Hi * Wi * G * C, 1, Wi * G * C, G * C};
            out_strides = {K, Ho * Wo * G * K, 1, Wo * G * K, G * K};
            wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};
        }

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

        if(CKWrwRequireWorkspace(G, C1, K1, data_type, alpha_beta_case))
        {
            // Creat dummy workspace to pass the ck IsSupportedArgument check.
            int dummy_var = 1;
            conv_ptr->SetWorkSpacePointer(arg_ptr.get(), &dummy_var);
        }
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
void PerformanceConfigHipImplicitGemmGroupWrwXdlops::Init(const ProblemDescription& problem)
{
    valid_kernels = FillValidKernelsIDs<DeviceOpGWrwPtrs<DataType>, CKArgs>(problem);
    index         = 0;
    split_k       = 1;
    kernel_id     = valid_kernels[index] + "+" + std::to_string(split_k);
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    return IsCKArgsSupported<DeviceOpGWrwPtrs<DataType>, CKArgs>(problem, kernel_id);
}

template <typename DataType>
bool ConvHipImplicitGemmGroupWrwXdlops::CheckCKApplicability(
    const ProblemDescription& problem) const
{
    return IsCKApplicable<DeviceOpGWrwPtrs<DataType>, CKArgs>(problem);
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

/**
 * @param type is the kernel type predicted by the parameter prediction model
 */
void PerformanceConfigHipImplicitGemmGroupWrwXdlops::InitHeuristicKernelIDs(const std::string& type)
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

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::ModelApplyToken(
    int idx, std::string value, const std::string& arch, const ProblemDescription& problem)
{
    if(idx == 13 && arch == "gfx90a")
        idx += 1; // skip
    if(arch == "gfx942")
    {
        if(idx == 0)
        {
            InitHeuristicKernelIDs(value);
            if(!valid_kernels.empty())
                return true;
            return false;
        }
        if(idx > 0)
            idx--;
        if(idx >= 12)
            idx += 2;
        if(((idx == 15 && (heuristic_kernels[heuristic_indexes[0]].size() == 15)) || idx == 18))
        {
            kernel_id =
                valid_kernels[heuristic_indexes[0]] + "+" + value + "+" + std::to_string(split_k);
            index              = heuristic_indexes[0];
            bool valid_split_k = false;
            switch(problem.GetInDataType())
            {
            case miopenHalf: valid_split_k = CheckIsSupportCKArgs<ck::half_t>(problem); break;
            case miopenFloat: valid_split_k = CheckIsSupportCKArgs<float>(problem); break;
            case miopenInt8:
            case miopenBFloat16:
            case miopenInt64:
            case miopenInt32:
            case miopenFloat8:
            case miopenBFloat8:
            case miopenDouble: break;
            }
            if(valid_split_k)
                return true;
            else
            {
                kernel_id = "";
                index     = 0;
                return false;
            }
        }
    }

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

static std::vector<float> GetFeatures(const ProblemDescription& problem, const std::string& arch)
{
    if(arch == "gfx90a")
    {
        std::size_t n = 18; // takes 18 convolution parameters as inputs
        std::vector<float> features(n * n, 0.0f);
        features[0]           = 1.0;
        features[n + 1]       = problem.GetOutChannels();
        features[2 * n + 2]   = problem.GetOutHeight();
        features[3 * n + 3]   = problem.GetOutWidth();
        features[4 * n + 4]   = problem.GetInChannels();
        features[5 * n + 5]   = problem.GetInHeight();
        features[6 * n + 6]   = problem.GetInWidth();
        features[7 * n + 7]   = problem.GetWeightsHeight();
        features[8 * n + 8]   = problem.GetWeightsWidth();
        features[9 * n + 9]   = problem.GetPadH();
        features[10 * n + 10] = problem.GetPadW();
        features[11 * n + 11] = problem.GetKernelStrideH();
        features[12 * n + 12] = problem.GetKernelStrideW();
        features[13 * n + 13] = problem.GetDilationH();
        features[14 * n + 14] = problem.GetDilationW();
        features[15 * n + 15] = problem.GetBatchSize();
        features[16 * n + 16] = problem.GetInDataType() == miopenFloat ? 2.0 : 1.0;
        features[17 * n + 17] = problem.GetGroupCount();
        return features;
    }
    std::size_t n = 17; // takes 17 convolution parameters as inputs
    std::vector<float> features(n * n, 0.0f);
    features[0]           = problem.GetOutChannels();
    features[n + 1]       = problem.GetOutHeight();
    features[2 * n + 2]   = problem.GetOutWidth();
    features[3 * n + 3]   = problem.GetInChannels();
    features[4 * n + 4]   = problem.GetInHeight();
    features[5 * n + 5]   = problem.GetInWidth();
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
bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::RunParameterPredictionModel(
    const ExecutionContext& ctx, const ProblemDescription& problem)
{
    valid_kernels = FillValidKernelsIDs<DeviceOpGWrwPtrs<DataType>, CKArgs>(
        problem); // filter valid_kernel ID's
    static const std::string& arch = ctx.GetStream().GetDeviceName();
    if(arch == "gfx90a")
        InitHeuristicKernelIDs("DeviceGroupedConvBwdWeight_Xdl_CShuffle");
    static const std::string solver =
        (arch == "gfx90a") ? "ConvHipIgemmGroupXdlops" : "ConvHipIgemmGroupWrwXdlops";
    std::vector<float> features = GetFeatures(problem, arch);
    if(ai::tuning::ModelSetParams(
           arch, solver, problem.GetDirection(), features, true, [&](int idx, std::string value) {
               return this->ModelApplyToken(idx, value, arch, problem);
           }))
    {
        if(arch == "gfx90a") // if gfx942 this is already set in ModelApplyToken
        {
            index     = heuristic_indexes[0];
            kernel_id = valid_kernels[index] + "+1";
        }
        MIOPEN_LOG_I("Params set by AI: " << ToString());
        return true;
    }
    return false;
}
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::IsModelApplicable(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(ctx.GetStream().GetDeviceName() != "gfx90a" && ctx.GetStream().GetDeviceName() != "gfx942")
        return false;
    if(problem.GetInDataType() != miopenFloat && problem.GetInDataType() != miopenHalf)
        return false;
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS_AI_HEUR))
        return false;
    return true;
}

void PerformanceConfigHipImplicitGemmGroupWrwXdlops::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    // these seem redundant
    split_k   = 1;
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

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::SetNextValue(const ProblemDescription& problem)
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

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::IsValid(
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

bool PerformanceConfigHipImplicitGemmGroupWrwXdlops::operator==(
    const PerformanceConfigHipImplicitGemmGroupWrwXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmGroupWrwXdlops
ConvHipImplicitGemmGroupWrwXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmGroupWrwXdlops pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvHipImplicitGemmGroupWrwXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmGroupWrwXdlops& config) const
{
    return config.IsValid(problem);
}

size_t ConvHipImplicitGemmGroupWrwXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                           const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemmGroupWrwXdlops
ConvHipImplicitGemmGroupWrwXdlops::Search(const ExecutionContext& ctx,
                                          const ProblemDescription& problem,
                                          const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmGroupWrwXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS))
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

ConvSolution ConvHipImplicitGemmGroupWrwXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemmGroupWrwXdlops& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeSolutionGroupConvImplicitGemmXdlops(
        problem,
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            return InitInvokerFactoryWrwNCHW<2,
                                             DeviceOpGWrwPtrs<T>,
                                             CKArgs,
                                             miopen::conv::WrWInvokeParams>(
                ctx, problem, config.kernel_id);
        },
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            return InitInvokerFactoryNHWC<DeviceOpGWrwPtrs<T>,
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
