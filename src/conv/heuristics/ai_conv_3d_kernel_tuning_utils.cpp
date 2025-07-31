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
 *******************************************************************************
 * general AI-related code for kernel tuning and heuristics. To be called in the
 * solver-specific code.
 *******************************************************************************/
#include <miopen/conv/heuristics/ai_conv_3d_kernel_tuning_utils.hpp>
#include <sstream>
#include <algorithm>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/logger.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/conv/problem_description.hpp>

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace miopen {
namespace solver {
namespace conv {
using ProblemInterpreter = miopen::solver::ProblemInterpreter;
using ProblemDescription = miopen::conv::ProblemDescription;

int LayoutStringToCode(const std::string& layout)
{
    if(layout == "NCDHW")
        return 0.0;
    if(layout == "NDHWC")
        return 1.0;
    // Add more as needed
    return -1.0; // Unknown
}

// Helper: Extract 3D convolution features
std::vector<float>
GetFeatures3D(const ProblemDescription& problem, int /*max_cu*/, const std::string& /*arch*/)
{
    // TODO: take metadata as input to look up encoding of string features (e.g., layout)
    // TODO: consider dynamically generating the features vector based on the values required by
    // metadata.
    std::vector<float> features;
    // 1–4: in_channels, in_d, in_h, in_w
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputChannelC(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputDepthDi(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputHeightHi(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputWidthWi(problem)));
    // 5–8: out_channels, out_d, out_h, out_w
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputChannelK(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputDepthDo(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputHeightHo(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputWidthWo(problem)));
    // 9–11: fil_d, fil_h, fil_w
    features.push_back(static_cast<float>(ProblemInterpreter::GetFilterDepthZ(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetFilterHeightY(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetFilterWidthX(problem)));
    // 12–14: pad_d, pad_h, pad_w
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputLeftPadD(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputLeftPadH(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputLeftPadW(problem)));
    // 15–17: conv_stride_d, conv_stride_h, conv_stride_w
    features.push_back(
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideD(problem)));
    features.push_back(
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideH(problem)));
    features.push_back(
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)));
    // 18: batchsize
    features.push_back(static_cast<float>(ProblemInterpreter::GetBatchN(problem)));
    // 19–21: in_layout, fil_layout, out_layout
    features.push_back(
        static_cast<float>(LayoutStringToCode(ProblemInterpreter::GetInputLayout(problem))));
    features.push_back(
        static_cast<float>(LayoutStringToCode(ProblemInterpreter::GetFilterLayout(problem))));
    features.push_back(
        static_cast<float>(LayoutStringToCode(ProblemInterpreter::GetOutputLayout(problem))));
    // 22: precision
    features.push_back(static_cast<float>(problem.GetInDataType()));
    return features;
}

// Helper: Tokenize kernel string
std::vector<std::string> TokenizeKernel(const std::string& kernel)
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
void FilterHeuristicKernels(const std::string& type,
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
// TODO: new CK functionality will use -1 for autodeduction, so we could add -1 to the list.
// Note that the current models have not been trained with -1 in mind, so it may not work as
// expected.
std::vector<int> GenerateSplitK(int max_split_k)
{
    std::vector<int> split_ks;
    for(int k = 1; k <= max_split_k; k *= 2)
        split_ks.push_back(k);
    return split_ks;
}

// Helper: Expand kernel params with split_k and keep mapping
std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<int, int>>>
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

// Main template implementation
template <typename DataType>
bool RunParameterPredictionModel(
    const miopen::ExecutionContext& ctx,
    const miopen::conv::ProblemDescription& problem,
    std::vector<std::string>& valid_kernels,
    int& index,
    int& split_k,
    std::string& kernel_id,
    std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)>
        fill_valid_kernels,
    std::string solver_name)
{
    valid_kernels = fill_valid_kernels(problem);

    // Filter kernels by type
    std::vector<int> heuristic_indexes;
    std::vector<std::vector<std::string>> heuristic_kernels;
    FilterHeuristicKernels(
        "DeviceGroupedConvBwdWeight", valid_kernels, heuristic_indexes, heuristic_kernels);

    // Prepare features and split_k values
    const std::string& arch = ctx.GetStream().GetDeviceName();
    std::vector<float> features =
        GetFeatures3D(problem, ctx.GetStream().GetMaxComputeUnits(), arch);
    std::vector<int> split_ks = GenerateSplitK(128); // TODO: make configurable

    // Expand kernel params with split_k and keep mapping
    auto [expanded_params, mapping_pairs] =
        ExpandKernelParamsWithSplitK(heuristic_kernels, heuristic_indexes, split_ks);

    // Use AI model to select best candidate
    try
    {
        int best_idx = ai::tuning::candidate_selection::ModelSelectBestCandidate(
            arch, solver_name, features, expanded_params);

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

// Explicit template instantiations for common types
template bool RunParameterPredictionModel<float>(
    const ExecutionContext&,
    const ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const ProblemDescription&)>,
    std::string);
template bool RunParameterPredictionModel<ck::half_t>(
    const ExecutionContext&,
    const ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const ProblemDescription&)>,
    std::string);
template bool RunParameterPredictionModel<int8_t>(
    const ExecutionContext&,
    const ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const ProblemDescription&)>,
    std::string);
template bool RunParameterPredictionModel<ck::bhalf_t>(
    const ExecutionContext&,
    const ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const ProblemDescription&)>,
    std::string);

} // namespace conv
} // namespace solver
} // namespace miopen
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
