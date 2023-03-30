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

#ifndef GAURD_MIOPEN_HEURISTIC_HPP_
#define GAURD_MIOPEN_HEURISTIC_HPP_

#include <miopen/conv/context.hpp>
#include <miopen/solver.hpp>
#include <unordered_map>
#include <set>
#include <queue>
#include <typeinfo>
#include <string>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>

namespace miopen {
namespace ai {
namespace tn {
std::set<std::string> GetSupportedArchs();
nlohmann::json GetMetadata(const std::string& arch);
std::vector<std::string> GetFeatureNames(const nlohmann::json& metadata);
std::unordered_map<size_t, std::string> GetSolverMap(const nlohmann::json& metadata);
size_t GetNumSolvers(const nlohmann::json& metadata);
size_t GetDirectionCode(const miopen::conv::Direction& dir, const nlohmann::json& metadata);
size_t GetPrecisionCode(const miopenDataType_t& data_type, const nlohmann::json& metadata);
size_t GetLayoutCode(const std::string& layout, const nlohmann::json& metadata);
std::vector<float> GetFeaturesMean(const nlohmann::json& metadata);
std::vector<float> GetFeaturesStd(const nlohmann::json& metadata);
std::vector<float> ToFeatures(const conv::ProblemDescription& conv_problem,
                              const nlohmann::json& metadata,
                              const bool normalize);
bool IsProblemInDistributionL1(const std::vector<float>& features,
                               const float threshold);
bool IsProblemInDistributionL2(const std::vector<float>& features,
                               const float threshold,
                               const nlohmann::json& metadata);
} // namespace tn

namespace ktn {
nlohmann::json get_metadata(const std::string& arch, const std::string& solver);
bool model_set_params(const std::string& encoder_path,
                      const std::string& decoder_path,
                      const nlohmann::json& metadata,
                      solver::PerformanceConfigConvAsm1x1U& config,
                      const ProblemDescription& problem,
                      std::vector<float>& features);
} // namespace ktn
} // namespace ai
} // namespace miopen
#endif
