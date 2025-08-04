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

#include <miopen/conv/heuristics/ai_heuristics.hpp>

#include <miopen/db_path.hpp>
#include <miopen/logger.hpp>

namespace miopen {
namespace ai {
namespace conv3d {

// Helper function to safely load JSON with error handling
static nlohmann::json LoadJSONSafe(const std::string& arch, bool& success)
{
    try {
        const auto file_path = GetSystemDbPath() / (arch + "_metadata.tn.model");
        auto json = common::LoadJSON(file_path);
        success = true;
        return json;
    } catch (const std::exception&) {
        success = false;
        return nlohmann::json{};
    } catch (...) {
        success = false;
        return nlohmann::json{};
    }
}

// Static helper functions for loading individual components
std::vector<std::string> Metadata3D::LoadFeatures(const std::string& arch, bool& success)
{
    auto json = LoadJSONSafe(arch, success);
    if (!success) return {};
    
    try {
        return json["conv_params_used_as_features"].get<std::vector<std::string>>();
    } catch (...) {
        success = false;
        return {};
    }
}

size_t Metadata3D::LoadNumInputs(const std::string& arch, bool& success)
{
    auto json = LoadJSONSafe(arch, success);
    if (!success) return 0;
    
    try {
        return json["num_inputs"].get<size_t>();
    } catch (...) {
        success = false;
        return 0;
    }
}

size_t Metadata3D::LoadNumOutputs(const std::string& arch, bool& success)
{
    auto json = LoadJSONSafe(arch, success);
    if (!success) return 0;
    
    try {
        return json["num_outputs"].get<size_t>();
    } catch (...) {
        success = false;
        return 0;
    }
}

size_t Metadata3D::LoadNumSolvers(const std::string& arch, bool& success)
{
    auto json = LoadJSONSafe(arch, success);
    if (!success) return 0;
    
    try {
        return json["num_solvers"].get<size_t>();
    } catch (...) {
        success = false;
        return 0;
    }
}

std::unordered_map<size_t, std::string> Metadata3D::LoadSolverMap(const std::string& arch, bool& success)
{
    auto json = LoadJSONSafe(arch, success);
    if (!success) return {};
    
    try {
        return common::ReverseMap<std::string, size_t>(json["encodings"]["solver"]);
    } catch (...) {
        success = false;
        return {};
    }
}

std::vector<float> Metadata3D::LoadFeaturesMean(const std::string& arch, size_t num_inputs, bool& success)
{
    // For now, return default values (could be enhanced to load from JSON stats)
    if (success) {
        return std::vector<float>(num_inputs, 0.0f);
    }
    return {};
}

std::vector<float> Metadata3D::LoadFeaturesStd(const std::string& arch, size_t num_inputs, bool& success)
{
    // For now, return default values (could be enhanced to load from JSON stats)
    if (success) {
        return std::vector<float>(num_inputs, 1.0f);
    }
    return {};
}

std::unordered_map<std::string, int> Metadata3D::LoadDirectionEncodings(const std::string& arch, bool& success)
{
    auto json = LoadJSONSafe(arch, success);
    if (!success) return {};
    
    try {
        return json["encodings"]["direction"].get<std::unordered_map<std::string, int>>();
    } catch (...) {
        success = false;
        return {};
    }
}

std::unordered_map<std::string, int> Metadata3D::LoadPrecisionEncodings(const std::string& arch, bool& success)
{
    auto json = LoadJSONSafe(arch, success);
    if (!success) return {};
    
    try {
        return json["encodings"]["precision"].get<std::unordered_map<std::string, int>>();
    } catch (...) {
        success = false;
        return {};
    }
}

std::unordered_map<std::string, int> Metadata3D::LoadInLayoutEncodings(const std::string& arch, bool& success)
{
    auto json = LoadJSONSafe(arch, success);
    if (!success) return {};
    
    try {
        return json["encodings"]["in_layout"].get<std::unordered_map<std::string, int>>();
    } catch (...) {
        success = false;
        return {};
    }
}

std::unordered_map<std::string, int> Metadata3D::LoadFilLayoutEncodings(const std::string& arch, bool& success)
{
    auto json = LoadJSONSafe(arch, success);
    if (!success) return {};
    
    try {
        return json["encodings"]["fil_layout"].get<std::unordered_map<std::string, int>>();
    } catch (...) {
        success = false;
        return {};
    }
}

std::unordered_map<std::string, int> Metadata3D::LoadOutLayoutEncodings(const std::string& arch, bool& success)
{
    auto json = LoadJSONSafe(arch, success);
    if (!success) return {};
    
    try {
        return json["encodings"]["out_layout"].get<std::unordered_map<std::string, int>>();
    } catch (...) {
        success = false;
        return {};
    }
}

// Constructor - loads all data immediately with error handling
Metadata3D::Metadata3D(const std::string& arch)
    : arch_name(arch),
      is_valid([&]() {
          bool success = true;
          
          // Load basic data first to get num_inputs for mean/std vectors
          LoadFeatures(arch, success);
          LoadNumInputs(arch, success);
          LoadNumOutputs(arch, success);
          LoadNumSolvers(arch, success);
          
          return success;
      }()),
      features(LoadFeatures(arch, const_cast<bool&>(is_valid))),
      num_inputs(LoadNumInputs(arch, const_cast<bool&>(is_valid))),
      num_outputs(LoadNumOutputs(arch, const_cast<bool&>(is_valid))),
      num_solvers(LoadNumSolvers(arch, const_cast<bool&>(is_valid))),
      solver_map(LoadSolverMap(arch, const_cast<bool&>(is_valid))),
      features_mean(LoadFeaturesMean(arch, num_inputs, const_cast<bool&>(is_valid))),
      features_std(LoadFeaturesStd(arch, num_inputs, const_cast<bool&>(is_valid))),
      direction_encodings_3d(LoadDirectionEncodings(arch, const_cast<bool&>(is_valid))),
      precision_encodings_3d(LoadPrecisionEncodings(arch, const_cast<bool&>(is_valid))),
      in_layout_encodings(LoadInLayoutEncodings(arch, const_cast<bool&>(is_valid))),
      fil_layout_encodings(LoadFilLayoutEncodings(arch, const_cast<bool&>(is_valid))),
      out_layout_encodings(LoadOutLayoutEncodings(arch, const_cast<bool&>(is_valid)))
{
    if (is_valid) {
        if (miopen::IsLogging(LoggingLevel::Info2)) {
            MIOPEN_LOG_I2("Metadata3D loaded for arch: " << arch << 
                         ", num_inputs=" << num_inputs << ", num_solvers=" << num_solvers);
        }
    } else {
        MIOPEN_LOG_I2("Metadata3D Failed to initialize metadata for: " << arch );
    }
}

// Encoding methods with safe error handling
size_t Metadata3D::EncodeDirection(miopen::conv::Direction dir) const
{
    if (!is_valid) return 0;
    
    try {
        if(dir == conv::Direction::BackwardWeights)
            return direction_encodings_3d.at("W");
        else if(dir == conv::Direction::BackwardData)
            return direction_encodings_3d.at("B");
        else
            return direction_encodings_3d.at("F");
    } catch (...) {
        MIOPEN_LOG_W("Direction encoding failed in 3D metadata, returning 0");
        return 0;
    }
}

size_t Metadata3D::EncodePrecision(miopenDataType_t data_type) const
{
    if (!is_valid) return 0;
    
    try {
        if(data_type == miopenBFloat16)
            return precision_encodings_3d.at("BF16");
        else if(data_type == miopenHalf)
            return precision_encodings_3d.at("FP16");
        else if(data_type == miopenFloat)
            return precision_encodings_3d.at("FP32");
        else {
            MIOPEN_LOG_W("Unsupported data type in 3D metadata, returning 0");
            return 0;
        }
    } catch (...) {
        MIOPEN_LOG_W("Precision encoding failed in 3D metadata, returning 0");
        return 0;
    }
}

size_t Metadata3D::EncodeLayout(const std::string& layout) const
{
    if (!is_valid) return 0;
    
    auto it = in_layout_encodings.find(layout);
    if(it != in_layout_encodings.end())
        return it->second;
    
    MIOPEN_LOG_W("Unsupported layout " << layout << " in 3D metadata, returning 0");
    return 0;
}

size_t Metadata3D::EncodeInLayout(const std::string& layout) const
{
    if (!is_valid) return 0;
    
    auto it = in_layout_encodings.find(layout);
    return (it != in_layout_encodings.end()) ? it->second : 0;
}

size_t Metadata3D::EncodeFilLayout(const std::string& layout) const
{
    if (!is_valid) return 0;
    
    auto it = fil_layout_encodings.find(layout);
    return (it != fil_layout_encodings.end()) ? it->second : 0;
}

size_t Metadata3D::EncodeOutLayout(const std::string& layout) const
{
    if (!is_valid) return 0;
    
    auto it = out_layout_encodings.find(layout);
    return (it != out_layout_encodings.end()) ? it->second : 0;
}

} // namespace conv3d
} // namespace ai
} // namespace miopen
