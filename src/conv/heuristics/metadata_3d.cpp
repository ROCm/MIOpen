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
#include <miopen/config.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>

#include <miopen/db_path.hpp>
#include <miopen/logger.hpp>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

namespace miopen {
namespace ai {
namespace conv3d {

// Helper function to safely load JSON with error handling
static std::optional<nlohmann::json> LoadJSONSafe(const std::string& arch)
{
    try
    {
        const auto file_path = GetSystemDbPath() / (arch + "_metadata.tn.model");
        return common::LoadJSON(file_path);
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_I2("Failed to load JSON for " << arch << ": " << e.what());
        return std::nullopt;
    }
    catch(...)
    {
        MIOPEN_LOG_I2("Failed to load JSON for " << arch << ": unknown error");
        return std::nullopt;
    }
}

// Static helper functions for loading individual components
std::optional<std::vector<std::string>> Metadata3D::LoadFeatures(const std::string& arch)
{
    auto json_opt = LoadJSONSafe(arch);
    if(!json_opt)
        return std::nullopt;

    try
    {
        return json_opt->at("conv_params_used_as_features").get<std::vector<std::string>>();
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_I2("Failed to load features for " << arch << ": " << e.what());
        return std::nullopt;
    }
}

std::optional<size_t> Metadata3D::LoadNumInputs(const std::string& arch)
{
    auto json_opt = LoadJSONSafe(arch);
    if(!json_opt)
        return std::nullopt;

    try
    {
        return json_opt->at("num_inputs").get<size_t>();
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_I2("Failed to load num_inputs for " << arch << ": " << e.what());
        return std::nullopt;
    }
}

std::optional<size_t> Metadata3D::LoadNumOutputs(const std::string& arch)
{
    auto json_opt = LoadJSONSafe(arch);
    if(!json_opt)
        return std::nullopt;

    try
    {
        return json_opt->at("num_outputs").get<size_t>();
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_I2("Failed to load num_outputs for " << arch << ": " << e.what());
        return std::nullopt;
    }
}

std::optional<size_t> Metadata3D::LoadNumSolvers(const std::string& arch)
{
    auto json_opt = LoadJSONSafe(arch);
    if(!json_opt)
        return std::nullopt;

    try
    {
        return json_opt->at("num_solvers").get<size_t>();
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_I2("Failed to load num_solvers for " << arch << ": " << e.what());
        return std::nullopt;
    }
}

std::optional<std::unordered_map<size_t, std::string>>
Metadata3D::LoadSolverMap(const std::string& arch)
{
    auto json_opt = LoadJSONSafe(arch);
    if(!json_opt)
        return std::nullopt;

    try
    {
        return common::ReverseMap<std::string, size_t>(json_opt->at("encodings").at("solver"));
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_I2("Failed to load solver_map for " << arch << ": " << e.what());
        return std::nullopt;
    }
}

std::optional<std::vector<float>>
Metadata3D::LoadFeaturesMean([[maybe_unused]] const std::string& arch, size_t num_inputs)
{
    // For now, return default values (could be enhanced to load from JSON stats)
    // This is a simplified version that returns zeros for mean
    return std::vector<float>(num_inputs, 0.0f);
}

std::optional<std::vector<float>>
Metadata3D::LoadFeaturesStd([[maybe_unused]] const std::string& arch, size_t num_inputs)
{
    // For now, return default values (could be enhanced to load from JSON stats)
    // This is a simplified version that returns ones for std
    return std::vector<float>(num_inputs, 1.0f);
}

std::optional<std::unordered_map<std::string, int>>
Metadata3D::LoadDirectionEncodings(const std::string& arch)
{
    auto json_opt = LoadJSONSafe(arch);
    if(!json_opt)
        return std::nullopt;

    try
    {
        return json_opt->at("encodings")
            .at("direction")
            .get<std::unordered_map<std::string, int>>();
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_I2("Failed to load direction_encodings for " << arch << ": " << e.what());
        return std::nullopt;
    }
}

std::optional<std::unordered_map<std::string, int>>
Metadata3D::LoadPrecisionEncodings(const std::string& arch)
{
    auto json_opt = LoadJSONSafe(arch);
    if(!json_opt)
        return std::nullopt;

    try
    {
        return json_opt->at("encodings")
            .at("precision")
            .get<std::unordered_map<std::string, int>>();
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_I2("Failed to load precision_encodings for " << arch << ": " << e.what());
        return std::nullopt;
    }
}

std::optional<std::unordered_map<std::string, int>>
Metadata3D::LoadInLayoutEncodings(const std::string& arch)
{
    auto json_opt = LoadJSONSafe(arch);
    if(!json_opt)
        return std::nullopt;

    try
    {
        return json_opt->at("encodings")
            .at("in_layout")
            .get<std::unordered_map<std::string, int>>();
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_I2("Failed to load in_layout_encodings for " << arch << ": " << e.what());
        return std::nullopt;
    }
}

std::optional<std::unordered_map<std::string, int>>
Metadata3D::LoadFilLayoutEncodings(const std::string& arch)
{
    auto json_opt = LoadJSONSafe(arch);
    if(!json_opt)
        return std::nullopt;

    try
    {
        return json_opt->at("encodings")
            .at("fil_layout")
            .get<std::unordered_map<std::string, int>>();
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_I2("Failed to load fil_layout_encodings for " << arch << ": " << e.what());
        return std::nullopt;
    }
}

std::optional<std::unordered_map<std::string, int>>
Metadata3D::LoadOutLayoutEncodings(const std::string& arch)
{
    auto json_opt = LoadJSONSafe(arch);
    if(!json_opt)
        return std::nullopt;

    try
    {
        return json_opt->at("encodings")
            .at("out_layout")
            .get<std::unordered_map<std::string, int>>();
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_I2("Failed to load out_layout_encodings for " << arch << ": " << e.what());
        return std::nullopt;
    }
}

// Constructor - loads all data immediately with error handling
MIOPEN_INTERNALS_EXPORT
Metadata3D::Metadata3D(const std::string& arch)
    : arch_name(arch),
      is_valid(false), // Initialize to false, will be set to true if all loads succeed
      features(),
      num_inputs(0),
      num_outputs(0),
      num_solvers(0),
      solver_map(),
      features_mean(),
      features_std(),
      direction_encodings_3d(),
      precision_encodings_3d(),
      in_layout_encodings(),
      fil_layout_encodings(),
      out_layout_encodings()
{
    // Load all components using std::optional pattern
    auto features_opt    = LoadFeatures(arch);
    auto num_inputs_opt  = LoadNumInputs(arch);
    auto num_outputs_opt = LoadNumOutputs(arch);
    auto num_solvers_opt = LoadNumSolvers(arch);
    auto solver_map_opt  = LoadSolverMap(arch);

    // Check if basic components loaded successfully
    if(!features_opt || !num_inputs_opt || !num_outputs_opt || !num_solvers_opt || !solver_map_opt)
    {
        MIOPEN_LOG_I2("Metadata3D: Failed to load basic components for " << arch);
        return;
    }

    // Now load mean/std with the known num_inputs
    auto features_mean_opt = LoadFeaturesMean(arch, *num_inputs_opt);
    auto features_std_opt  = LoadFeaturesStd(arch, *num_inputs_opt);

    // Load encoding maps
    auto direction_encodings_opt  = LoadDirectionEncodings(arch);
    auto precision_encodings_opt  = LoadPrecisionEncodings(arch);
    auto in_layout_encodings_opt  = LoadInLayoutEncodings(arch);
    auto fil_layout_encodings_opt = LoadFilLayoutEncodings(arch);
    auto out_layout_encodings_opt = LoadOutLayoutEncodings(arch);

    // Check if all components loaded successfully
    if(!features_mean_opt || !features_std_opt || !direction_encodings_opt ||
       !precision_encodings_opt || !in_layout_encodings_opt || !fil_layout_encodings_opt ||
       !out_layout_encodings_opt)
    {
        MIOPEN_LOG_I2("Metadata3D: Failed to load encoding components for " << arch);
        return;
    }

    // All components loaded successfully, now we can safely move the data
    // We need to const_cast because the members are const
    features               = std::move(*features_opt);
    num_inputs             = *num_inputs_opt;
    num_outputs            = *num_outputs_opt;
    num_solvers            = *num_solvers_opt;
    solver_map             = std::move(*solver_map_opt);
    features_mean          = std::move(*features_mean_opt);
    features_std           = std::move(*features_std_opt);
    direction_encodings_3d = std::move(*direction_encodings_opt);
    precision_encodings_3d = std::move(*precision_encodings_opt);
    in_layout_encodings    = std::move(*in_layout_encodings_opt);
    fil_layout_encodings   = std::move(*fil_layout_encodings_opt);
    out_layout_encodings   = std::move(*out_layout_encodings_opt);
    // Mark as valid after successful loading
    is_valid = true;

    if(miopen::IsLogging(LoggingLevel::Info2))
    {
        MIOPEN_LOG_I2("Metadata3D loaded successfully for arch: "
                      << arch << ", num_inputs=" << num_inputs << ", num_solvers=" << num_solvers);
    }
}

// Encoding methods with safe error handling
MIOPEN_INTERNALS_EXPORT
size_t Metadata3D::EncodeDirection(miopen::conv::Direction dir) const
{
    if(!is_valid)
        return 0;

    try
    {
        if(dir == conv::Direction::BackwardWeights)
            return direction_encodings_3d.at("W");
        else if(dir == conv::Direction::BackwardData)
            return direction_encodings_3d.at("B");
        else
            return direction_encodings_3d.at("F");
    }
    catch(...)
    {
        MIOPEN_LOG_W("Direction encoding failed in 3D metadata, returning 0");
        return 0;
    }
}

MIOPEN_INTERNALS_EXPORT
size_t Metadata3D::EncodePrecision(miopenDataType_t data_type) const
{
    if(!is_valid)
        return 0;

    try
    {
        if(data_type == miopenBFloat16)
            return precision_encodings_3d.at("BF16");
        else if(data_type == miopenHalf)
            return precision_encodings_3d.at("FP16");
        else if(data_type == miopenFloat)
            return precision_encodings_3d.at("FP32");
        else
        {
            MIOPEN_LOG_W("Unsupported data type in 3D metadata, returning 0");
            return 0;
        }
    }
    catch(...)
    {
        MIOPEN_LOG_W("Precision encoding failed in 3D metadata, returning 0");
        return 0;
    }
}

MIOPEN_INTERNALS_EXPORT
size_t Metadata3D::EncodeLayout(const std::string& layout) const
{
    if(!is_valid)
        return 0;

    auto it = in_layout_encodings.find(layout);
    if(it != in_layout_encodings.end())
        return it->second;

    MIOPEN_LOG_W("Unsupported layout " << layout << " in 3D metadata, returning 0");
    return 0;
}

MIOPEN_INTERNALS_EXPORT
size_t Metadata3D::EncodeInLayout(const std::string& layout) const
{
    if(!is_valid)
        return 0;

    auto it = in_layout_encodings.find(layout);
    return (it != in_layout_encodings.end()) ? it->second : 0;
}

MIOPEN_INTERNALS_EXPORT
size_t Metadata3D::EncodeFilLayout(const std::string& layout) const
{
    if(!is_valid)
        return 0;

    auto it = fil_layout_encodings.find(layout);
    return (it != fil_layout_encodings.end()) ? it->second : 0;
}

MIOPEN_INTERNALS_EXPORT
size_t Metadata3D::EncodeOutLayout(const std::string& layout) const
{
    if(!is_valid)
        return 0;

    auto it = out_layout_encodings.find(layout);
    return (it != out_layout_encodings.end()) ? it->second : 0;
}

} // namespace conv3d
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
