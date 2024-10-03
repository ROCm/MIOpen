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

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <unordered_map>
#include <typeinfo>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <fstream>
#include <miopen/miopen.h>
#include <nlohmann/json.hpp>
#include <miopen/db_path.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/anyramdb.hpp>

namespace miopen {
namespace ai {
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
namespace immed_mode {
struct Metadata
{
private:
    nlohmann::json json;
    const std::unordered_map<std::string, int> direction_encodings;
    const std::unordered_map<std::string, int> precision_encodings;
    const std::unordered_map<std::string, int> layout_encodings;

public:
    const std::vector<std::string> features;
    const size_t num_inputs;
    const size_t num_outputs;
    const size_t num_solvers;
    const std::unordered_map<size_t, std::string> solver_map;
    const std::vector<float> features_mean;
    const std::vector<float> features_std;
    const std::vector<float> test_features_mean;
    const std::vector<float> test_features_std;
    Metadata(const std::string& arch);
    size_t EncodeDirection(miopen::conv::Direction dir) const;
    size_t EncodePrecision(miopenDataType_t data_type) const;
    size_t EncodeLayout(const std::string& layout) const;
};
class Model;
MIOPEN_INTERNALS_EXPORT std::vector<uint64_t> PredictSolver(const conv::ProblemDescription& problem,
                                                            const ExecutionContext& ctx,
                                                            const std::string& device);
} // namespace immed_mode

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace tuning {
struct Metadata
{
    std::size_t predict_type;
    std::unordered_map<std::string, std::size_t> num_tuning_params;
    std::unordered_map<std::string, std::string> tuning_decodings;
    Metadata(const std::string& arch, const std::string& solver);
};

bool ModelSetParams(const std::string& arch,
                    const std::string& solver,
                    conv::Direction direction,
                    const std::vector<float>& features,
                    bool transform_features,
                    std::function<bool(std::size_t, std::string)> validator);
} // namespace tuning
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
} // namespace ai
} // namespace miopen
#endif
#endif // GAURD_MIOPEN_AI_HEURISTICS_HPP_
