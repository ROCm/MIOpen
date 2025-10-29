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
 *******************************************************************************/

#pragma once

#include <vector>
#include <string>
#include <miopen/config.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

namespace miopen {
namespace solver {
namespace conv {
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
const miopen::ExecutionContext& GetDummyCtx();

MIOPEN_INTERNALS_EXPORT std::map<std::string, float>
GetFeatures3D(const miopen::conv::ProblemDescription&, int max_cu, const std::string& arch);

MIOPEN_INTERNALS_EXPORT std::vector<std::string> GetKernelAsTokens(const std::string& kernel);
MIOPEN_INTERNALS_EXPORT void FillHeuristicKernels(const std::vector<std::string>& valid_kernels,
                                                  std::vector<int>& indexes,
                                                  std::vector<std::vector<std::string>>& kernels);

MIOPEN_INTERNALS_EXPORT std::vector<int> GenerateSplitK(int max_split_k);
// Main template implementation
template <typename DataType>
std::pair<bool, miopen::ai::tuning::candidate_selection::CandidateSelectionResult>
RunParameterPredictionModel(
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

        bool use_split_k = split_k != 0;
        if(split_k > 1)
        {
            MIOPEN_THROW("Invalid initial split_k value for performing AI Heuristics: " +
                         std::to_string(split_k) + ". Expected 0 (no split) or 1 (default split).");
        }

        auto result = ai::tuning::candidate_selection::ModelSelectBestCandidate(
            arch, solver_name, features, heuristic_kernels, use_split_k);

        // Check if we have any candidates
        if(!result.IsEmpty())
        {
            // Get the best candidate (first in the sorted list)
            int best_index   = result.GetBestKernelIndex();
            int best_split_k = result.GetBestSplitK();

            if(best_index >= 0 && best_index < static_cast<int>(valid_kernels.size()))
            {
                index   = best_index;
                split_k = best_split_k;
                if(use_split_k)
                {
                    kernel_id = valid_kernels[index] + "+" + std::to_string(split_k);
                }
                else
                {
                    kernel_id = valid_kernels[index];
                }
                return {true, result};
            }
        }
        MIOPEN_LOG_I("AI prediction returned invalid kernel index, falling back");
        return {false, result};
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_I2("[Warning] AI model failed: " << ex.what());
        return {false, miopen::ai::tuning::candidate_selection::CandidateSelectionResult{}};
    }
}
} // namespace conv
} // namespace solver
} // namespace miopen
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
