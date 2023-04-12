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

#ifndef GAURD_MIOPEN_AI_HEURISTICS_HPP_
#define GAURD_MIOPEN_AI_HEURISTICS_HPP_

#include <miopen/miopen.h>
#include <miopen/conv/context.hpp>
#include <miopen/solver.hpp>
#include <unordered_map>
#include <typeinfo>
#include <string>
#include <vector>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <miopen/db_path.hpp>
#include <fstream>
#include <miopen/config.h>
#include <miopen/any_solver.hpp>
#include <set>
#include <queue>
#include <boost/filesystem.hpp>
#include <miopen/anyramdb.hpp>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING 
namespace fdeep {
    class model;
}

namespace miopen {
namespace ai {
#endif
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
namespace immed_mode {
bool IsDeviceSupported(const std::string& device);
nlohmann::json GetMetadata(const std::string& device);
fdeep::model GetModel(const std::string& device);
std::vector<std::string> GetFeatureNames(const nlohmann::json& metadata);
std::unordered_map<size_t, std::string> GetSolverMap(const nlohmann::json& metadata);
size_t GetNumInputs(const nlohmann::json& metadata);
size_t GetNumOutputs(const nlohmann::json& metadata);
size_t GetNumSolvers(const nlohmann::json& metadata);
size_t GetDirectionCode(const miopen::conv::Direction& dir, const nlohmann::json& metadata);
size_t GetPrecisionCode(const miopenDataType_t& data_type, const nlohmann::json& metadata);
size_t GetLayoutCode(const std::string& layout, const nlohmann::json& metadata);
std::vector<float> GetFeaturesMean(const nlohmann::json& metadata);
std::vector<float> GetFeaturesStd(const nlohmann::json& metadata);
std::vector<float> ToFeatures(const conv::ProblemDescription& problem,
                              const nlohmann::json& metadata,
                              const bool normalize);
bool AreFeaturesInDistributionL1(const std::vector<float>& features,
                                 const float threshold);
bool AreFeaturesInDistributionL2(const std::vector<float>& features,
                                 const float threshold,
                                 const nlohmann::json& metadata);
bool IsProblemSupported(const ProblemDescription& problem,
                        const ConvolutionContext& ctx,
                        const nlohmann::json& metadata);
std::vector<float> CallModel(const fdeep::model& model,
                             const std::vector<float>& normalized_features,
                             const nlohmann::json& metadata);
std::vector<uint64_t> PredictSolver(const conv::ProblemDescription& problem,
                                    const std::vector<float>& normalized_features,
                                    bool& cached,
                                    const std::string& device,
                                    const nlohmann::json& metadata);
} // namespace immed_mode

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace kernel_tuning {
struct Metadata {
    std::size_t num_conv_params;
    std::size_t num_tuning_params;
    std::size_t sos_token;
    std::unordered_map<std::string, int> tuning_decodings;
    Metadata (const std::string& arch, const std::string& solver);
};
class Model;
std::unordered_map<std::string, Model*> GetModels(const std::string& arch);
bool ModelSetParams(const std::string& arch,
                    const std::string& solver,
                    solver::PerformanceConfigConvAsm1x1U& config,
                    const ProblemDescription& problem);
} // namespace kernel_tuning
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING
} // namespace ai
} // namespace miopen
#endif
#endif // GAURD_MIOPEN_AI_HEURISTICS_HPP_
