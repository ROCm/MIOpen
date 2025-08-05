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
#include <map>
#include <string>

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
// Helper: Tokenize kernel string
std::vector<std::string> GetKernelAsTokens(const std::string& kernel)
{
    std::vector<std::string> tokens;

    // Split on '<' to separate prefix from parameters
    auto lt_pos = kernel.find('<');
    if(lt_pos != std::string::npos)
    {
        // Add the entire prefix (before '<') as a single token
        std::string prefix = kernel.substr(0, lt_pos);
        prefix.erase(remove_if(prefix.begin(), prefix.end(), isspace), prefix.end());
        if(!prefix.empty())
            tokens.push_back(prefix);

        // Split parameters (inside '<...>') by commas
        auto gt_pos = kernel.find('>', lt_pos);
        if(gt_pos != std::string::npos && gt_pos > lt_pos + 1)
        {
            std::string params = kernel.substr(lt_pos + 1, gt_pos - lt_pos - 1);
            std::stringstream ps(params);
            std::string token;
            while(std::getline(ps, token, ','))
            {
                token.erase(remove_if(token.begin(), token.end(), isspace), token.end());
                if(!token.empty())
                    tokens.push_back(token);
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
        auto tokens = GetKernelAsTokens(valid_kernels[i]);
        if(!tokens.empty() && tokens[0].starts_with(type)) // Check if tokens[0] starts with type
        {
            indexes.push_back(i);
            kernels.push_back(tokens);
        }
        else
        {
            MIOPEN_LOG_I2("Skipping kernel: " << valid_kernels[i] << " as " << tokens[0]
                                              << " does not match type: " << type);
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
    std::cerr << "RunParameterPredictionModel: entered" << std::endl;
    valid_kernels = fill_valid_kernels(problem);
    std::cerr << "RunParameterPredictionModel: valid_kernels.size() = " << valid_kernels.size()
              << std::endl;

    // Filter kernels by type
    std::vector<int> heuristic_indexes;
    std::vector<std::vector<std::string>> heuristic_kernels;
    // TODO: why "DeviceGroupedConvBwdWeight" hardcoded here?
    FilterHeuristicKernels(
        "DeviceGroupedConvBwdWeight", valid_kernels, heuristic_indexes, heuristic_kernels);
    std::cerr << "RunParameterPredictionModel: heuristic_kernels.size() = "
              << heuristic_kernels.size() << std::endl;

    // print out valid kernels
    std::cerr << "RunParameterPredictionModel: valid_kernels = " << std::endl;
    for(const auto& kernel : valid_kernels)
    {
        std::cerr << kernel << std::endl;
    }
    // print out heuristic kernels
    std::cerr << "RunParameterPredictionModel: heuristic_kernels = " << std::endl;
    for(const auto& kernel : heuristic_kernels)
    {
        std::cerr << "  ";
        for(const auto& token : kernel)
        {
            std::cerr << token << "; ";
        }
        std::cerr << std::endl;
    }
    // Prepare features and split_k values
    const std::string& arch = ctx.GetStream().GetDeviceName();
    std::cerr << "RunParameterPredictionModel: arch = " << arch << std::endl;
    std::map<std::string, float> features =
        GetFeatures3D(problem, ctx.GetStream().GetMaxComputeUnits(), arch);
    std::vector<int> split_ks = GenerateSplitK(128); // TODO: make configurable
    std::cerr << "RunParameterPredictionModel: split_ks.size() = " << split_ks.size() << std::endl;

    // Expand kernel params with split_k and keep mapping
    auto [expanded_params, mapping_pairs] =
        ExpandKernelParamsWithSplitK(heuristic_kernels, heuristic_indexes, split_ks);
    std::cerr << "RunParameterPredictionModel: expanded_params.size() = " << expanded_params.size()
              << std::endl;

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
