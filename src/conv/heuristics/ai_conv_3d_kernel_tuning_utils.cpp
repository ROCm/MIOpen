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
// Helper: Tokenize kernel string
MIOPEN_INTERNALS_EXPORT
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
    FillHeuristicKernels(valid_kernels, heuristic_indexes, heuristic_kernels);
    // Prepare features and split_k values
    const std::string& arch = ctx.GetStream().GetDeviceName();

    // Use AI model to select best candidate
    try
    {
        std::map<std::string, float> features =
            GetFeatures3D(problem, ctx.GetStream().GetMaxComputeUnits(), arch);

        bool use_split_k = split_k == 1;
        if(split_k > 1)
        {
            MIOPEN_THROW("Invalid initial split_k value for performing AI Heuristics: " +
                         std::to_string(split_k) + ". Expected 0 (no split) or 1 (default split).");
        }

        auto result = ai::tuning::candidate_selection::ModelSelectBestCandidate(
            arch, solver_name, features, heuristic_kernels, use_split_k);

        if(result.kernel_index >= 0 && result.kernel_index < static_cast<int>(valid_kernels.size()))
        {
            index   = result.kernel_index;
            split_k = result.split_k;
            if(use_split_k)
            {
                kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
            }
            else
            {
                kernel_id = valid_kernels[index];
            }
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

template bool RunParameterPredictionModel<int8_t>(
    const ExecutionContext&,
    const ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const ProblemDescription&)>,
    std::string);
#if MIOPEN_USE_COMPOSABLEKERNEL

template bool RunParameterPredictionModel<ck::half_t>(
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
#endif

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
