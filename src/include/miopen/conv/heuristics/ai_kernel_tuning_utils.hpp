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

#pragma once

#include <vector>
#include <string>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

namespace miopen {
namespace solver {
namespace conv {

std::vector<float>
GetFeatures3D(const miopen::conv::ProblemDescription&, int max_cu, const std::string& arch);
std::vector<std::string> TokenizeKernel(const std::string& kernel);
void FilterHeuristicKernels(const std::string& type,
                            const std::vector<std::string>& valid_kernels,
                            std::vector<int>& indexes,
                            std::vector<std::vector<std::string>>& kernels);
std::vector<int> GenerateSplitK(int max_split_k);
std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<int, int>>>
ExpandKernelParamsWithSplitK(const std::vector<std::vector<std::string>>& kernels,
                             const std::vector<int>& indexes,
                             const std::vector<int>& split_ks);

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
} // namespace conv
} // namespace solver
} // namespace miopen
