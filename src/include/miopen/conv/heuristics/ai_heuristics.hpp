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
namespace common {
}
#endif
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
namespace immed_mode {
struct Metadata {
    public:
    std::vector<std::string> features;
    size_t num_inputs;
    size_t num_outputs;
    size_t num_solvers;
    std::unordered_map<size_t, std::string> solver_map;
    std::vector<float> features_mean;
    std::vector<float> features_std;
    Metadata (const std::string& arch);
    size_t MapDirection(const miopen::conv::Direction& dir) const;
    size_t MapPrecision(const miopenDataType_t& data_type) const;
    size_t MapLayout(const std::string& layout) const;

    private:
    std::unordered_map<std::string, int> direction_map;
    std::unordered_map<std::string, int> precision_map;
    std::unordered_map<std::string, int> layout_map;
};
class Model;
std::vector<uint64_t> PredictSolver(const ProblemDescription& problem,
                                    const ConvolutionContext& ctx,
                                    const std::string& device);
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
