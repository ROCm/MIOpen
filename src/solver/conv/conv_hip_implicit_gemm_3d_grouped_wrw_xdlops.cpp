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
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#endif
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS_AI_HEUR)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

template <typename DataType>
using DeviceOpGWrwPtrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    DeviceOpGBwdWeightDefault<DataType>>;

// Add these new template specializations for different alpha/beta cases
template <typename DataType>
using DeviceOpGWrwBilinearPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdWeightBilinear<DataType>>;

template <typename DataType>
using DeviceOpGWrwScalePtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOpGBwdWeightScale<DataType>>;

namespace {

template <typename DataType>
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
        Di              = ProblemInterpreter::GetInputDepthDi(problem);
        Do              = ProblemInterpreter::GetOutputDepthDo(problem);
        Z               = ProblemInterpreter::GetFilterDepthZ(problem);
        data_type       = ProblemInterpreter::GetOutputDataType(problem);
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
    CKArgs(const CKArgs&)            = default;
    CKArgs(CKArgs&&)                 = default;
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
        using DeviceP = std::remove_pointer_t<decltype(conv_ptr.get())>;
        if constexpr(std::is_same_v<DeviceP, DeviceOpGBwdWeightBilinear<DataType>>)
        {
            return MakeBilinearArgPtr(conv_ptr, x, dw, dy, alpha, beta, split_k);
        }
        else if constexpr(std::is_same_v<DeviceP, DeviceOpGBwdWeightScale<DataType>>)
        {
            (void)beta;
            return MakeScaleArgPtr(conv_ptr, x, dw, dy, alpha, split_k);
        }
        else
        {
            (void)alpha;
            (void)beta;
            static_assert(std::is_same_v<DeviceP, DeviceOpGBwdWeightDefault<DataType>>,
                          "Default should be wrw pass through");
            return MakeDefaultArgPtr(conv_ptr, x, dw, dy, split_k);
        }
    }
    template <typename ConvPtr>
    auto MakeBilinearArgPtr(const ConvPtr& conv_ptr,
                            ConstData_t x,
                            Data_t dw,
                            ConstData_t dy,
                            float alpha,
                            float beta,
                            int split_k) const
    {
        return conv_ptr->MakeArgumentPointer(x,
                                             dw,
                                             dy,
                                             {dw},
                                             in_lengths,
                                             in_strides,
                                             wei_lengths,
                                             wei_strides,
                                             out_lengths,
                                             out_strides,
                                             {wei_lengths},
                                             {wei_strides},
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             Bilinear{alpha, beta},
                                             PassThrough{},
                                             split_k);
    }

    template <typename ConvPtr>
    auto MakeScaleArgPtr(const ConvPtr& conv_ptr,
                         ConstData_t x,
                         Data_t dw,
                         ConstData_t dy,
                         float alpha,
                         int split_k) const
    {
        return conv_ptr->MakeArgumentPointer(x,
                                             dw,
                                             dy,
                                             {},
                                             in_lengths,
                                             in_strides,
                                             wei_lengths,
                                             wei_strides,
                                             out_lengths,
                                             out_strides,
                                             {},
                                             {},
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             Scale{alpha},
                                             PassThrough{},
                                             split_k);
    }

    template <typename ConvPtr>
    auto MakeDefaultArgPtr(
        const ConvPtr& conv_ptr, ConstData_t x, Data_t dw, ConstData_t dy, int split_k) const
    {
        return conv_ptr->MakeArgumentPointer(x,
                                             dw,
                                             dy,
                                             in_lengths,
                                             in_strides,
                                             wei_lengths,
                                             wei_strides,
                                             out_lengths,
                                             out_strides,
                                             filter_strides,
                                             filter_dilations,
                                             lPadding,
                                             rPadding,
                                             PassThrough{},
                                             PassThrough{},
                                             PassThrough{},
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
    int Di;
    int Ho;
    int Wo;
    int Do;
    int Y;
    int X;
    int Z;
    miopenAlphaBetaCase_t alpha_beta_case;
    miopenDataType_t data_type;
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
} // namespace

template <typename DataType>
void PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::Init(const ProblemDescription& problem)
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        valid_kernels =
            FillValidKernelsIDs<DeviceOpGBwdWeightBilinearPtrs<DataType>, CKArgs<DataType>>(
                problem);
        break;
    case SCALE:
        valid_kernels =
            FillValidKernelsIDs<DeviceOpGBwdWeightScalePtrs<DataType>, CKArgs<DataType>>(problem);
        break;
    default:
        valid_kernels =
            FillValidKernelsIDs<DeviceOpGBwdWeightDefaultPtrs<DataType>, CKArgs<DataType>>(problem);
        break;
    }
    index     = 0;
    split_k   = 1;
    kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return IsCKArgsSupported<DeviceOpGBwdWeightBilinearPtrs<DataType>, CKArgs<DataType>>(
            problem, kernel_id);
    case SCALE:
        return IsCKArgsSupported<DeviceOpGBwdWeightScalePtrs<DataType>, CKArgs<DataType>>(
            problem, kernel_id);
    default:
        return IsCKArgsSupported<DeviceOpGBwdWeightDefaultPtrs<DataType>, CKArgs<DataType>>(
            problem, kernel_id);
    }
}

template <typename DataType>
bool ConvHipImplicitGemm3DGroupWrwXdlops::CheckCKApplicability(
    const ProblemDescription& problem) const
{
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        return IsCKApplicable<DeviceOpGBwdWeightBilinearPtrs<DataType>, CKArgs<DataType>>(problem);
    case SCALE:
        return IsCKApplicable<DeviceOpGBwdWeightScalePtrs<DataType>, CKArgs<DataType>>(problem);
    default:
        return IsCKApplicable<DeviceOpGBwdWeightDefaultPtrs<DataType>, CKArgs<DataType>>(problem);
    }
}
#endif

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace {
static std::vector<std::string> GetKernelAsTokens(const std::string& kernel)
{
    std::vector<std::string> tokens;
    std::stringstream ss(kernel);
    std::string token;

    while(std::getline(ss, token, '_'))
    {
        if(!token.empty())
        {
            tokens.push_back(token);
        }
    }
    return tokens;
}

/**
 * @param type is the kernel type predicted by the parameter prediction model
 */
static void InitHeuristicKernelIDs(const std::string& type,
                                   const std::vector<std::string>& valid_kernels,
                                   std::vector<int>& heuristic_indexes,
                                   std::vector<std::vector<std::string>>& heuristic_kernels)
{
    heuristic_indexes.clear();
    heuristic_kernels.clear();

    for(std::size_t i = 0; i < valid_kernels.size(); i++)
    {
        const auto tokens = GetKernelAsTokens(valid_kernels[i]);
        if(!tokens.empty() && tokens[0] == type)
        {
            heuristic_indexes.push_back(i);
            heuristic_kernels.push_back(tokens);
        }
    }
}

// Helper function to get 3D convolution features (adapt from existing GetFeatures if available)
static std::vector<float>
GetFeatures3D(const ProblemDescription& problem, int max_cu, const std::string& arch)
{
    // Extract 3D-specific features
    std::vector<float> features;

    // Basic problem dimensions
    features.push_back(static_cast<float>(ProblemInterpreter::GetBatchN(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputChannelC(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputChannelK(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetGroupCountG(problem)));

    // 3D spatial dimensions
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputDepthDi(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputHeightHi(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputWidthWi(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputDepthDo(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputHeightHo(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputWidthWo(problem)));

    // Filter dimensions
    features.push_back(static_cast<float>(ProblemInterpreter::GetFilterDepthZ(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetFilterHeightY(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetFilterWidthX(problem)));

    // Strides and dilations
    features.push_back(
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideD(problem)));
    features.push_back(
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideH(problem)));
    features.push_back(
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)));
    features.push_back(
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionDilationD(problem)));
    features.push_back(
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionDilationH(problem)));
    features.push_back(
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)));

    // Padding
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputLeftPadD(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputLeftPadH(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputLeftPadW(problem)));

    // Device features
    features.push_back(static_cast<float>(max_cu));

    // Data type encoding
    features.push_back(static_cast<float>(problem.GetInDataType()));

    // Layout encoding
    features.push_back(problem.IsLayoutNHWC() ? 1.0f : 0.0f);

    return features;
}

// Helper: Tokenize kernel string
static std::vector<std::string> TokenizeKernel(const std::string& kernel)
{
    std::vector<std::string> tokens;
    std::stringstream ss(kernel);
    std::string token;
    while(std::getline(ss, token, '_'))
    {
        if(!token.empty())
            tokens.push_back(token);
    }
    return tokens;
}

// Helper: Filter kernels by type and collect indexes/tokens
static void FilterHeuristicKernels(const std::string& type,
                                   const std::vector<std::string>& valid_kernels,
                                   std::vector<int>& indexes,
                                   std::vector<std::vector<std::string>>& kernels)
{
    indexes.clear();
    kernels.clear();
    for(std::size_t i = 0; i < valid_kernels.size(); ++i)
    {
        auto tokens = TokenizeKernel(valid_kernels[i]);
        if(!tokens.empty() && tokens[0] == type)
        {
            indexes.push_back(i);
            kernels.push_back(tokens);
        }
    }
}

// Helper: Generate split_k values (powers of two)
static std::vector<int> GenerateSplitK(int max_split_k)
{
    std::vector<int> split_ks;
    for(int k = 1; k <= max_split_k; k *= 2)
        split_ks.push_back(k);
    return split_ks;
}

// Helper: Expand kernel params with split_k and keep mapping
static std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<int, int>>>
ExpandKernelParamsWithSplitK(const std::vector<std::vector<std::string>>& kernels,
                             const std::vector<int>& indexes,
                             const std::vector<int>& split_ks)
{
    std::vector<std::vector<std::string>> expanded;
    std::vector<std::pair<int, int>> mapping;
    for(size_t i = 0; i < kernels.size(); ++i)
    {
        for(int split_k : split_ks)
        {
            auto candidate = kernels[i];
            candidate.push_back(std::to_string(split_k));
            expanded.push_back(candidate);
            mapping.emplace_back(indexes[i], split_k);
        }
    }
    return {expanded, mapping};
}

// Main: Run AI parameter prediction model
template <typename DataType>
static bool RunParameterPredictionModel(const ExecutionContext& ctx,
                                        const ProblemDescription& problem,
                                        std::vector<std::string>& valid_kernels,
                                        int& index,
                                        int& split_k,
                                        std::string& kernel_id)
{
    // Select valid kernels based on alpha/beta case
    switch(problem.GetAlphaBetaCase())
    {
    case BILINEAR:
        valid_kernels =
            FillValidKernelsIDs<DeviceOpGWrwBilinearPtrs<DataType>, CKArgs<DataType>>(problem);
        break;
    case SCALE:
        valid_kernels =
            FillValidKernelsIDs<DeviceOpGWrwScalePtrs<DataType>, CKArgs<DataType>>(problem);
        break;
    default:
        valid_kernels = FillValidKernelsIDs<DeviceOpGWrwPtrs<DataType>, CKArgs<DataType>>(problem);
        break;
    }

    // Filter kernels by type
    std::vector<int> heuristic_indexes;
    std::vector<std::vector<std::string>> heuristic_kernels;
    FilterHeuristicKernels(
        "DeviceGroupedConvBwdWeight", valid_kernels, heuristic_indexes, heuristic_kernels);

    // Prepare features and split_k values
    const std::string& arch = ctx.GetStream().GetDeviceName();
    std::string solver =
        (arch == "gfx90a") ? "ConvHipIgemm3DGroupXdlops" : "ConvHipIgemmGroup3DWrwXdlops";
    std::vector<float> features =
        GetFeatures3D(problem, ctx.GetStream().GetMaxComputeUnits(), arch);
    std::vector<int> split_ks = GenerateSplitK(128); // TODO: make configurable

    // Expand kernel params with split_k and keep mapping
    auto [expanded_params, mapping_pairs] =
        ExpandKernelParamsWithSplitK(heuristic_kernels, heuristic_indexes, split_ks);

    // Use AI model to select best candidate
    try
    {
        int best_idx = ai::tuning::ModelSelectBestCandidate(
            arch, solver, problem.GetDirection(), features, expanded_params);

        if(best_idx >= 0 && best_idx < static_cast<int>(mapping_pairs.size()))
        {
            index     = mapping_pairs[best_idx].first;
            split_k   = mapping_pairs[best_idx].second;
            kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
            return true;
        }
        MIOPEN_LOG_I("AI prediction returned invalid kernel index, falling back");
        return false;
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_I2("[Warning] AI model failed: " << ex.what());
        return false;
    }
}
} // namespace
#endif

void PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::HeuristicInit(
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    split_k   = 1;
    kernel_id = "";

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    // Try AI heuristics first if enabled
    if(!env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS_AI_HEUR))
    {
        bool ai_success = false;
        switch(problem.GetInDataType())
        {
        case miopenHalf:
            ai_success = RunParameterPredictionModel<ck::half_t>(
                ExecutionContext{}, problem, valid_kernels, index, split_k, kernel_id);
            break;
        case miopenFloat:
            ai_success = RunParameterPredictionModel<float>(
                ExecutionContext{}, problem, valid_kernels, index, split_k, kernel_id);
            break;
        case miopenInt8:
            ai_success = RunParameterPredictionModel<int8_t>(
                ExecutionContext{}, problem, valid_kernels, index, split_k, kernel_id);
            break;
        case miopenBFloat16:
            ai_success = RunParameterPredictionModel<ck::bhalf_t>(
                ExecutionContext{}, problem, valid_kernels, index, split_k, kernel_id);
            break;
        case miopenInt64:
        case miopenInt32:
        case miopenFloat8_fnuz:
        case miopenBFloat8_fnuz:
        case miopenDouble: break;
        }

        if(ai_success)
        {
            MIOPEN_LOG_I("AI heuristics successfully selected kernel: " << kernel_id);
            return;
        }
        else
        {
            MIOPEN_LOG_I("AI heuristics failed, falling back to default initialization");
        }
    }
#endif

    // Fallback to original initialization
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

bool PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::SetNextValue(
    const ProblemDescription& problem)
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    if(valid_kernels.empty())
    {
        HeuristicInit(problem);
        if(valid_kernels.empty())
        {
            return false;
        }
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

bool PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::IsValid(
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

bool PerformanceConfigHipImplicitGemm3DGroupWrwXdlops::operator==(
    const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops& other) const
{
    return kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemm3DGroupWrwXdlops
ConvHipImplicitGemm3DGroupWrwXdlops::GetDefaultPerformanceConfig(
    const ExecutionContext&, const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemm3DGroupWrwXdlops pp;
    pp.HeuristicInit(problem);
    return pp;
}

bool ConvHipImplicitGemm3DGroupWrwXdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops& config) const
{
    return config.IsValid(problem);
}

size_t
ConvHipImplicitGemm3DGroupWrwXdlops::GetWorkspaceSize(const ExecutionContext&,
                                                      const ProblemDescription& problem) const
{
    return GetWorkspaceSizeLayoutTransformConv(problem);
}

PerformanceConfigHipImplicitGemm3DGroupWrwXdlops
ConvHipImplicitGemm3DGroupWrwXdlops::Search(const ExecutionContext& ctx,
                                            const ProblemDescription& problem,
                                            const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemm3DGroupWrwXdlops::IsApplicable(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    if(env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.HasMixedDataTypes())
        return false;
    if(!problem.IsDirectionBackwardWrW())
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
    case miopenBFloat16:
        return (ctx.GetStream().GetDeviceName() == "gfx942" ||
                StartsWith(ctx.GetStream().GetDeviceName(), "gfx95")) &&
               CheckCKApplicability<ck::bhalf_t>(problem);
    case miopenInt64:
    case miopenInt32:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
    case miopenDouble: break;
    }
#endif
    return false;
}

ConvSolution ConvHipImplicitGemm3DGroupWrwXdlops::GetSolution(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem,
    [[maybe_unused]] const PerformanceConfigHipImplicitGemm3DGroupWrwXdlops& config) const
{
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
    return MakeSolutionGroupConvImplicitGemmXdlops(
        problem,
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            switch(problem.GetAlphaBetaCase())
            {
            case BILINEAR:
                return InitInvokerFactoryWrwNCHW<3,
                                                 false,
                                                 DeviceOpGBwdWeightBilinearPtrs<T>,
                                                 CKArgs<T>,
                                                 miopen::conv::WrWInvokeParams>(
                    ctx, problem, config.kernel_id);
            case SCALE:
                return InitInvokerFactoryWrwNCHW<3,
                                                 false,
                                                 DeviceOpGBwdWeightScalePtrs<T>,
                                                 CKArgs<T>,
                                                 miopen::conv::WrWInvokeParams>(
                    ctx, problem, config.kernel_id);
            default:
                return InitInvokerFactoryWrwNCHW<3,
                                                 false,
                                                 DeviceOpGBwdWeightDefaultPtrs<T>,
                                                 CKArgs<T>,
                                                 miopen::conv::WrWInvokeParams>(
                    ctx, problem, config.kernel_id);
            }
        },
        [&](auto data_type_val) {
            using T = decltype(data_type_val);
            switch(problem.GetAlphaBetaCase())
            {
            case BILINEAR:
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGBwdWeightBilinearPtrs<T>,
                                              CKArgs<T>,
                                              miopen::conv::WrWInvokeParams>(
                    ctx, problem, config.kernel_id);
            case SCALE:
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGBwdWeightScalePtrs<T>,
                                              CKArgs<T>,
                                              miopen::conv::WrWInvokeParams>(
                    ctx, problem, config.kernel_id);
            default:
                return InitInvokerFactoryNHWC<false,
                                              DeviceOpGBwdWeightDefaultPtrs<T>,
                                              CKArgs<T>,
                                              miopen::conv::WrWInvokeParams>(
                    ctx, problem, config.kernel_id);
            }
        });

#else
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
