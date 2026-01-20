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
 *******************************************************************************
 * general AI-related code for kernel tuning and heuristics. To be called in the
 * solver-specific code.
 *******************************************************************************/
#include <algorithm>
#include <iterator>
#include <map>
#include <sstream>
#include <string>

#include <miopen/conv/heuristics/ai_conv_3d_kernel_tuning_utils.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/logger.hpp>

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace miopen {
namespace solver {
namespace conv {
using ProblemInterpreter = miopen::solver::ProblemInterpreter;
using ProblemDescription = miopen::conv::ProblemDescription;

int LayoutStringToCode(const std::string& layout)
{
    if(layout == "NCDHW")
        return 0;
    if(layout == "NDHWC")
        return 1;
    // Add more as needed
    return -1; // Unknown
}

// Helper: Extract 3D convolution features
MIOPEN_INTERNALS_EXPORT
std::map<std::string, float>
GetFeatures3D(const ProblemDescription& problem, int /*max_cu*/, const std::string& /*arch*/)
{
    std::map<std::string, float> features;

    // 1: spatial_dim
    features["spatial_dim"] = 3.0f;

    // 2–5: in_channels, in_d, in_h, in_w
    features["in_channels"] = static_cast<float>(ProblemInterpreter::GetInputChannelC(problem));
    features["in_d"]        = static_cast<float>(ProblemInterpreter::GetInputDepthDi(problem));
    features["in_h"]        = static_cast<float>(ProblemInterpreter::GetInputHeightHi(problem));
    features["in_w"]        = static_cast<float>(ProblemInterpreter::GetInputWidthWi(problem));

    // 6–9: out_channels, out_d, out_h, out_w
    features["out_channels"] = static_cast<float>(ProblemInterpreter::GetOutputChannelK(problem));
    features["out_d"]        = static_cast<float>(ProblemInterpreter::GetOutputDepthDo(problem));
    features["out_h"]        = static_cast<float>(ProblemInterpreter::GetOutputHeightHo(problem));
    features["out_w"]        = static_cast<float>(ProblemInterpreter::GetOutputWidthWo(problem));

    // 10–12: fil_d, fil_h, fil_w
    features["fil_d"] = static_cast<float>(ProblemInterpreter::GetFilterDepthZ(problem));
    features["fil_h"] = static_cast<float>(ProblemInterpreter::GetFilterHeightY(problem));
    features["fil_w"] = static_cast<float>(ProblemInterpreter::GetFilterWidthX(problem));

    // 13–15: pad_d, pad_h, pad_w
    features["pad_d"] = static_cast<float>(ProblemInterpreter::GetInputLeftPadD(problem));
    features["pad_h"] = static_cast<float>(ProblemInterpreter::GetInputLeftPadH(problem));
    features["pad_w"] = static_cast<float>(ProblemInterpreter::GetInputLeftPadW(problem));

    // 16–18: conv_stride_d, conv_stride_h, conv_stride_w
    features["conv_stride_d"] =
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideD(problem));
    features["conv_stride_h"] =
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideH(problem));
    features["conv_stride_w"] =
        static_cast<float>(ProblemInterpreter::GetAdjustedConvolutionStrideW(problem));

    // 19–21: dilation_d, dilation_h, dilation_w
    features["dilation_d"] = static_cast<float>(problem.GetDilationD());
    features["dilation_h"] = static_cast<float>(problem.GetDilationH());
    features["dilation_w"] = static_cast<float>(problem.GetDilationW());

    // 22: batchsize
    features["batchsize"] = static_cast<float>(ProblemInterpreter::GetBatchN(problem));

    // 23: bias
    features["bias"] = static_cast<float>(problem.GetBias());

    // 24–26: in_layout, fil_layout, out_layout (as codes)
    features["in_layout"] =
        static_cast<float>(LayoutStringToCode(ProblemInterpreter::GetInputLayout(problem)));
    features["fil_layout"] =
        static_cast<float>(LayoutStringToCode(ProblemInterpreter::GetFilterLayout(problem)));
    features["out_layout"] =
        static_cast<float>(LayoutStringToCode(ProblemInterpreter::GetOutputLayout(problem)));

    // 27: precision
    features["precision"] = static_cast<float>(problem.GetInDataType());

    // 28: direction
    features["direction"] = static_cast<float>(
        problem.GetDirection() == miopen::conv::Direction::Forward           ? 0.0f
        : problem.GetDirection() == miopen::conv::Direction::BackwardData    ? 1.0f
        : problem.GetDirection() == miopen::conv::Direction::BackwardWeights ? 2.0f
                                                                             : -1.0f);

    // 29: group_count
    features["group_count"] = static_cast<float>(problem.GetGroupCount());

    return features;
}

// Helper: Process Explicit_Xdl kernel parameters to match implicit GEMM format
MIOPEN_INTERNALS_EXPORT
std::vector<std::string> ProcessExplicitXdlParams(const std::vector<std::string>& params)
{
    if(params.empty())
        return {};

    std::vector<std::string> processed;
    processed.push_back(params[0]); // Keep first element unchanged

    for(size_t i = 1; i < params.size(); ++i)
    {
        const auto& param = params[i];

        // Special handling for BlkGemmPipelineScheduler and BlkGemmPipelineVersion
        // Keep the full "Label: Value" format for these
        if(param.find("BlkGemmPipelineScheduler:") != std::string::npos ||
           param.find("BlkGemmPipelineVersion:") != std::string::npos)
        {
            processed.push_back(param);
            continue;
        }

        // For all other parameters: remove label (everything before and including ':')
        std::string value = param;
        auto colon_pos    = param.find(':');
        if(colon_pos != std::string::npos)
        {
            value = param.substr(colon_pos + 1);
            // Trim leading whitespace
            value.erase(0, value.find_first_not_of(" \t"));
        }

        // Split on 'x' if present
        auto x_pos = value.find('x');
        if(x_pos != std::string::npos)
        {
            // Split and add each component
            size_t start = 0;
            while(start < value.length())
            {
                auto next_x = value.find('x', start);
                if(next_x == std::string::npos)
                {
                    processed.push_back(value.substr(start));
                    break;
                }
                processed.push_back(value.substr(start, next_x - start));
                start = next_x + 1;
            }
        }
        else
        {
            processed.push_back(value);
        }
    }

    return processed;
}

// Helper: Tokenize kernel string
MIOPEN_INTERNALS_EXPORT
std::vector<std::string> GetKernelAsTokens(const std::string& kernel)
{
    std::vector<std::string> tokens;

    // Split on '<' to separate prefix from parameters
    auto lt_pos = kernel.find('<');
    if(lt_pos != std::string::npos)
    {
        // Add the entire prefix (before '<') as a single token, removing whitespace
        std::string prefix;
        std::remove_copy_if(kernel.begin(),
                            kernel.begin() + lt_pos,
                            std::back_inserter(prefix),
                            [](char c) { return std::isspace(c); });
        if(!prefix.empty())
            tokens.push_back(prefix);

        // Find the matching '>' for the opening '<' by counting brackets
        size_t gt_pos     = std::string::npos;
        int bracket_count = 1; // We've already seen one '<'
        for(size_t i = lt_pos + 1; i < kernel.size(); ++i)
        {
            if(kernel[i] == '<')
                bracket_count++;
            else if(kernel[i] == '>')
            {
                bracket_count--;
                if(bracket_count == 0)
                {
                    gt_pos = i;
                    break;
                }
            }
        }

        if(gt_pos != std::string::npos && gt_pos > lt_pos + 1)
        {
            std::string params = kernel.substr(lt_pos + 1, gt_pos - lt_pos - 1);

            // Smart splitting: split on commas at depth 0, and after closing brackets at depth 0
            std::string current_token;
            int depth = 0;

            for(char c : params)
            {
                if(c == '<')
                {
                    depth++;
                    current_token += c;
                }
                else if(c == '>')
                {
                    depth--;
                    current_token += c;

                    // If we just closed a bracket at depth 0, finalize the token
                    if(depth == 0)
                    {
                        std::string clean_token;
                        std::remove_copy_if(current_token.begin(),
                                            current_token.end(),
                                            std::back_inserter(clean_token),
                                            [](char ch) { return std::isspace(ch); });
                        if(!clean_token.empty())
                            tokens.push_back(clean_token);
                        current_token.clear();
                    }
                }
                else if(c == ',' && depth == 0)
                {
                    // This is a top-level comma, finalize the current token
                    std::string clean_token;
                    std::remove_copy_if(current_token.begin(),
                                        current_token.end(),
                                        std::back_inserter(clean_token),
                                        [](char ch) { return std::isspace(ch); });
                    if(!clean_token.empty())
                        tokens.push_back(clean_token);
                    current_token.clear();
                }
                else
                {
                    current_token += c;
                }
            }

            // Don't forget the last token (if any)
            if(!current_token.empty())
            {
                std::string clean_token;
                std::remove_copy_if(current_token.begin(),
                                    current_token.end(),
                                    std::back_inserter(clean_token),
                                    [](char ch) { return std::isspace(ch); });
                if(!clean_token.empty())
                    tokens.push_back(clean_token);
            }
        }
    }
    else
    {
        // No '<', just add the whole string as a single token
        std::string trimmed = kernel;
        trimmed.erase(remove_if(trimmed.begin(), trimmed.end(), isspace), trimmed.end());
        if(!trimmed.empty())
            tokens.push_back(trimmed);
    }

    return tokens;
}

// Helper: Fill indexes and kernels from valid_kernels
MIOPEN_INTERNALS_EXPORT
void FillHeuristicKernels(const std::vector<std::string>& valid_kernels,
                          std::vector<int>& indexes,
                          std::vector<std::vector<std::string>>& kernels)
{
    indexes.clear();
    kernels.clear();
    for(std::size_t i = 0; i < valid_kernels.size(); ++i)
    {
        auto tokens = GetKernelAsTokens(valid_kernels[i]);

        // Special processing for Explicit_Xdl kernels to match format of implicit GEMM kernels (and
        // thus the format expected by the heuristics)
        if(!tokens.empty() &&
           tokens[0].find("DeviceGroupedConvBwdWeight_Explicit_Xdl") != std::string::npos)
        {
            tokens = ProcessExplicitXdlParams(tokens);
        }

        indexes.push_back(i);
        kernels.push_back(tokens);
    }
}

// Helper: Generate split_k values (powers of two)
// TODO: new CK functionality will use -1 for autodeduction, so we could add -1 to the list.
// Note that the current models have not been trained with -1 in mind, so it may not work as
// expected.
MIOPEN_INTERNALS_EXPORT
std::vector<int> GenerateSplitK(int max_split_k)
{
    std::vector<int> split_ks;
    for(int k = 1; k <= max_split_k; k *= 2)
        split_ks.push_back(k);
    return split_ks;
}

// Explicit template instantiations for common types
// Note: We instantiate with a generic lambda type for the validation function
template std::pair<bool, miopen::ai::tuning::candidate_selection::CandidateSelectionResult>
RunParameterPredictionModel<float, bool (*)(int, int)>(
    const miopen::ExecutionContext&,
    const ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const ProblemDescription&)>,
    std::string,
    bool (*&&)(int, int));

template std::pair<bool, miopen::ai::tuning::candidate_selection::CandidateSelectionResult>
RunParameterPredictionModel<int8_t, bool (*)(int, int)>(
    const miopen::ExecutionContext&,
    const ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const ProblemDescription&)>,
    std::string,
    bool (*&&)(int, int));
#if MIOPEN_USE_COMPOSABLEKERNEL

template std::pair<bool, miopen::ai::tuning::candidate_selection::CandidateSelectionResult>
RunParameterPredictionModel<ck::half_t, bool (*)(int, int)>(
    const miopen::ExecutionContext&,
    const ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const ProblemDescription&)>,
    std::string,
    bool (*&&)(int, int));

template std::pair<bool, miopen::ai::tuning::candidate_selection::CandidateSelectionResult>
RunParameterPredictionModel<ck::bhalf_t, bool (*)(int, int)>(
    const miopen::ExecutionContext&,
    const ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const ProblemDescription&)>,
    std::string,
    bool (*&&)(int, int));
#endif // MIOPEN_USE_COMPOSABLEKERNEL

// helper function to get a dummy execution context for when we do not have a real context
const miopen::ExecutionContext& GetDummyCtx()
{
    static miopen::ExecutionContext ctx;
    return ctx;
}

} // namespace conv
} // namespace solver
} // namespace miopen
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
