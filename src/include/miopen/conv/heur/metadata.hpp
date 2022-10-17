/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_METADATA_HPP_
#define GUARD_MIOPEN_METADATA_HPP_

#include <miopen/problem_description.hpp>
#include <fdeep/fdeep.hpp>
#include <fdeep/tensor.hpp>
#include <fdeep/tensor_shape.hpp>
#include <unordered_map>
#include <vector>
#include <miopen/db_path.hpp>

namespace miopen {
const std::unordered_map<size_t, std::string>& GetSolverMap(const std::string& arch);
void TransformFeatures(std::vector<float>& features, const std::string& arch);
const std::vector<std::string>& GetFeatureNames(const std::string& arch);
const size_t GetDirectionMap(const miopen::conv::Direction dir, const std::string& arch);
const size_t GetPrecisionMap(const miopenDataType_t data_type, const std::string& arch);
const size_t GetLayoutMap(const std::string& layout, const std::string& arch);
std::vector<float> CallModel(std::vector<float>& features, const std::string& arch);
} // namespace miopen
#endif
