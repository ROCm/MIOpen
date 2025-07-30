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

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace miopen {
namespace solver {
namespace conv {

// Helper: Extract 3D convolution features
std::vector<float>
GetFeatures3D(const ProblemDescription& problem, int max_cu, const std::string& arch)
{
    std::vector<float> features;
    features.push_back(static_cast<float>(ProblemInterpreter::GetBatchN(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputChannelC(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputChannelK(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetGroupCountG(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputDepthDi(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputHeightHi(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputWidthWi(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputDepthDo(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputHeightHo(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetOutputWidthWo(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetFilterDepthZ(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetFilterHeightY(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetFilterWidthX(problem)));
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
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputLeftPadD(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputLeftPadH(problem)));
    features.push_back(static_cast<float>(ProblemInterpreter::GetInputLeftPadW(problem)));
    features.push_back(static_cast<float>(max_cu));
    features.push_back(static_cast<float>(problem.GetInDataType()));
    features.push_back(problem.IsLayoutNHWC() ? 1.0f : 0.0f);
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

// Main: Run AI parameter prediction model

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