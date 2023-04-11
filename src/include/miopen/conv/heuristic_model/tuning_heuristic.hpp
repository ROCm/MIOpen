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

#ifndef GAURD_MIOPEN_TUNING_HEURISTIC_HPP_
#define GAURD_MIOPEN_TUNING_HEURISTIC_HPP_

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
#if MIOPEN_ENABLE_AI_KERNEL_TUNING

namespace fdeep {
    class model;
}

namespace miopen {
namespace ai {
namespace tuning {

struct Metadata {
    std::size_t num_conv_params;
    std::size_t num_tuning_params;
    std::size_t sos_token;
    std::unordered_map<std::string, int> tuning_decodings;
    Metadata (const std::string& arch, const std::string& solver);
};

struct Model;

std::unordered_map<std::string, Model*> GetModels(const std::string& arch, const std::string& solver);

std::vector<float> TransformFeatures(const std::string& arch,
                                     const std::string& solver,
                                     const ProblemDescription& problem, 
                                     std::size_t n);

bool ModelSetParams(const std::string& arch,
                    const std::string& solver,
                    solver::PerformanceConfigConvAsm1x1U& config,
                    const ProblemDescription& problem);

} // namespace tuning
} // namespace ai
} // namespace miopen
#endif
#endif
