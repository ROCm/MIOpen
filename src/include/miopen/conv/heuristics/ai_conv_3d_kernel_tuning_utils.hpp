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

const miopen::ExecutionContext& GetDummyCtx();

MIOPEN_INTERNALS_EXPORT std::map<std::string, float>
GetFeatures3D(const miopen::conv::ProblemDescription&, int max_cu, const std::string& arch);

MIOPEN_INTERNALS_EXPORT std::vector<std::string> GetKernelAsTokens(const std::string& kernel);
MIOPEN_INTERNALS_EXPORT void FillHeuristicKernels(const std::vector<std::string>& valid_kernels,
                                                  std::vector<int>& indexes,
                                                  std::vector<std::vector<std::string>>& kernels);

MIOPEN_INTERNALS_EXPORT std::vector<int> GenerateSplitK(int max_split_k);
template <typename DataType>
MIOPEN_INTERNALS_EXPORT bool RunParameterPredictionModel(
    const miopen::ExecutionContext& ctx,
    const miopen::conv::ProblemDescription& problem,
    std::vector<std::string>& valid_kernels,
    int& index,
    int& split_k,
    std::string& kernel_id,
    std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)>
        fill_valid_kernels,
    std::string solver_name);

extern template bool RunParameterPredictionModel<float>(
    const miopen::ExecutionContext&,
    const miopen::conv::ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)>,
    std::string);
#if MIOPEN_USE_COMPOSABLEKERNEL
extern template bool RunParameterPredictionModel<ck::half_t>(
    const miopen::ExecutionContext&,
    const miopen::conv::ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)>,
    std::string);

extern template bool RunParameterPredictionModel<ck::bhalf_t>(
    const miopen::ExecutionContext&,
    const miopen::conv::ProblemDescription&,
    std::vector<std::string>&,
    int&,
    int&,
    std::string&,
    std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)>,
    std::string);
#endif
} // namespace conv
} // namespace solver
} // namespace miopen
