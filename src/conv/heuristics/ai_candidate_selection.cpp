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
 *******************************************************************************
 *
 * AI Candidate Selection Models for Kernel Tuning using a candidate selection approach.
 * Also known as a "Two Towers" model.
 * Contains: CandidateSelectionMetadata, CandidateSelectionModel, and helpers.
 *
 *******************************************************************************/

#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <nlohmann/json.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace miopen {
namespace ai {
namespace tuning {
namespace candidate_selection {

// --- CandidateSelectionMetadata ---------------------------------------------

MIOPEN_INTERNALS_EXPORT
CandidateSelectionMetadata::CandidateSelectionMetadata(const std::string& arch,
                                                       const std::string& solver)
{
    const auto path = GetSystemDbPath() / (arch + "_" + solver + "_metadata.tn.model");
    std::ifstream file(path);
    if(!file.is_open())
    {
        MIOPEN_THROW("Could not open metadata file: " + path.string());
    }
    nlohmann::json metadata;
    try
    {
        file >> metadata;
    }
    catch(const std::exception& ex)
    {
        MIOPEN_THROW("JSON parse error in metadata file: " + path.string() + ": " + ex.what());
    }

    input_params_  = metadata.value("input_params", std::vector<std::string>{});
    output_params_ = metadata.value("output_params", std::vector<std::string>{});

    for(size_t i = 0; i < input_params_.size(); ++i)
        input_param_indices_[input_params_[i]] = i;
    for(size_t i = 0; i < output_params_.size(); ++i)
        output_param_indices_[output_params_[i]] = i;

    if(metadata.contains("encodings"))
    {
        feature_encodings_ = metadata["encodings"].value("inputs", decltype(feature_encodings_){});
        sequence_encodings_ =
            metadata["encodings"].value("outputs", decltype(sequence_encodings_){});
    }
    else
    {
        MIOPEN_THROW("Metadata file does not contain 'encodings' section");
    }

    if(metadata.contains("decodings") && metadata["decodings"].contains("outputs"))
    {
        sequence_decodings_ = metadata["decodings"]["outputs"]
                                  .get<std::map<std::string, std::map<std::string, std::string>>>();
    }
    else
    {
        MIOPEN_THROW("Metadata file does not contain 'decodings' section for outputs");
    }

    if(metadata.contains("constants"))
    {
        constants_features_ =
            metadata["constants"].value("inputs", decltype(constants_features_){});
        constants_sequence_ =
            metadata["constants"].value("outputs", decltype(constants_sequence_){});
    }
    else
    {
        MIOPEN_THROW("Metadata file does not contain 'constants' section");
    }

    if(metadata.contains("missing_value_token"))
    {
        missing_value_token_ = metadata["missing_value_token"].get<float>();
    }
    else
    {
        MIOPEN_THROW("Metadata file does not contain 'missing_value_token' section");
    }

    if(metadata.contains("kernel_str_mapping"))
    {
        kernel_str_mapping_ = metadata["kernel_str_mapping"]
                                  .get<std::map<std::string, std::map<std::string, std::string>>>();
    }
    else
    {
        MIOPEN_THROW("Metadata file does not contain 'kernel_str_mapping' section");
    }

    if(metadata.contains("split_k_values"))
    {
        split_k_values_ = metadata["split_k_values"].get<std::vector<int>>();
    }
    else
    {
        split_k_values_ = {1}; // Default to 1 if not specified
    }
}

MIOPEN_INTERNALS_EXPORT
size_t CandidateSelectionMetadata::GetInputParamIndex(const std::string& name) const
{
    auto it = input_param_indices_.find(name);
    if(it == input_param_indices_.end())
        MIOPEN_THROW("Input parameter not found: " + name);
    return it->second;
}

MIOPEN_INTERNALS_EXPORT
size_t CandidateSelectionMetadata::GetOutputParamIndex(const std::string& name) const
{
    auto it = output_param_indices_.find(name);
    if(it == output_param_indices_.end())
        MIOPEN_THROW("Output parameter not found: " + name);
    return it->second;
}

MIOPEN_INTERNALS_EXPORT
std::optional<std::string>
CandidateSelectionMetadata::GetInputConstant(const std::string& name) const
{
    auto it = constants_features_.find(name);
    if(it != constants_features_.end())
        return it->second;
    return std::nullopt;
}

MIOPEN_INTERNALS_EXPORT
std::optional<std::string>
CandidateSelectionMetadata::GetOutputConstant(const std::string& name) const
{
    auto it = constants_sequence_.find(name);
    if(it != constants_sequence_.end())
        return it->second;
    return std::nullopt;
}

MIOPEN_INTERNALS_EXPORT
std::vector<size_t> CandidateSelectionMetadata::GetConstantInputIndices() const
{
    std::vector<size_t> indices;
    for(const auto& [name, value] : constants_features_)
    {
        auto it = input_param_indices_.find(name);
        if(it != input_param_indices_.end())
            indices.push_back(it->second);
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

MIOPEN_INTERNALS_EXPORT
std::vector<size_t> CandidateSelectionMetadata::GetConstantOutputIndices() const
{
    std::vector<size_t> indices;
    for(const auto& [name, value] : constants_sequence_)
    {
        auto it = output_param_indices_.find(name);
        if(it != output_param_indices_.end())
            indices.push_back(it->second);
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

MIOPEN_INTERNALS_EXPORT
std::map<std::string, std::string>
CandidateSelectionMetadata::GetKernelStrMapping(const std::string& kernel_name) const
{
    auto it = kernel_str_mapping_.find(kernel_name);
    if(it != kernel_str_mapping_.end())
    {
        return it->second;
    }
    else
    {
        MIOPEN_THROW("Kernel string mapping not found for kernel: " + kernel_name);
    }
}

MIOPEN_INTERNALS_EXPORT
const std::map<std::string, std::map<std::string, int>>&
CandidateSelectionMetadata::sequence_encodings() const
{
    return sequence_encodings_;
}
float CandidateSelectionMetadata::GetMissingValueToken() const { return missing_value_token_; }

MIOPEN_INTERNALS_EXPORT
const std::vector<int>& CandidateSelectionMetadata::GetSplitKValues() const
{
    return split_k_values_;
}

// --- CandidateSelectionModel ------------------------------------------------

MIOPEN_INTERNALS_EXPORT
CandidateSelectionModel::CandidateSelectionModel(const std::string& arch, const std::string& solver)
    : metadata_(arch, solver), arch_(arch), solver_(solver)
{
}

MIOPEN_INTERNALS_EXPORT
CandidateSelectionModel::~CandidateSelectionModel() = default;

MIOPEN_INTERNALS_EXPORT
std::vector<float>
CandidateSelectionModel::EncodeInputFeatures(const std::map<std::string, float>& features) const
{
    std::vector<float> filtered_features;
    const auto& input_params = metadata_.input_params();

    for(const auto& name : input_params)
    {
        // Skip constant features
        if(metadata_.GetInputConstant(name) != std::nullopt)
            continue;

        // Only add if present in the input map
        auto it = features.find(name);
        if(it != features.end())
        {
            filtered_features.push_back(it->second);
        }
        else
        {
            MIOPEN_THROW((std::ostringstream() << "Input parameter not found: " << name).str());
        }
    }

    // Pass the filtered vector to the encoding function
    return EncodeInputFeaturesWithFdeep(filtered_features, arch_, solver_);
}

MIOPEN_INTERNALS_EXPORT
std::vector<std::vector<float>> CandidateSelectionModel::EncodeKernelConfigs(
    const std::vector<std::vector<float>>& encoded_candidates) const
{
    return EncodeKernelConfigsWithFdeep(encoded_candidates, arch_, solver_);
}

MIOPEN_INTERNALS_EXPORT
std::vector<std::pair<int, float>> CandidateSelectionModel::SelectBestCandidateIndices(
    const std::vector<float>& encoded_features,
    const std::vector<std::vector<float>>& encoded_configs) const
{
    if(encoded_configs.empty() || encoded_features.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "Empty features or configs in SelectBestCandidateIndices");
    }

    size_t feature_dim    = encoded_features.size();
    size_t num_candidates = encoded_configs.size();

    std::vector<std::pair<int, float>> scored_candidates;
    scored_candidates.reserve(num_candidates);

    for(size_t i = 0; i < num_candidates; ++i)
    {
        if(encoded_configs[i].size() != feature_dim)
            MIOPEN_THROW(miopenStatusInternalError,
                         "Config dimension mismatch in SelectBestCandidateIndices");

        float score = std::inner_product(
            encoded_configs[i].begin(), encoded_configs[i].end(), encoded_features.begin(), 0.0f);
        scored_candidates.emplace_back(static_cast<int>(i), score);
    }

    // Sort by score in descending order (best to worst)
    std::sort(scored_candidates.begin(), scored_candidates.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    return scored_candidates;
}
// --- Factory and Helper Functions -------------------------------------------

// Helper: Expand kernel params with split_k and keep mapping
MIOPEN_INTERNALS_EXPORT
std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<int, int>>>
ExpandKernelParamsWithSplitK(const std::vector<std::vector<std::string>>& kernels,
                             const std::vector<int>& indexes,
                             const std::vector<int>& split_ks)
{
    std::vector<std::vector<std::string>> expanded;
    std::vector<std::pair<int, int>> mapping;
    for(size_t i = 0; i < kernels.size(); ++i)
    {
        for(int split_k : split_ks)
        {
            auto candidate = kernels[i];
            candidate.push_back(std::to_string(split_k));
            expanded.push_back(candidate);
            mapping.emplace_back(indexes[i], split_k);
        }
    }
    return {expanded, mapping};
}

MIOPEN_INTERNALS_EXPORT
const CandidateSelectionModel& GetCandidateSelectionModel(const std::string& arch,
                                                          const std::string& solver)
{
    static std::map<std::string, std::unique_ptr<CandidateSelectionModel>> models;
    static std::mutex models_mutex;
    std::string key = arch + "_" + solver;

    std::lock_guard<std::mutex> lock(models_mutex);
    try
    {
        auto [it, inserted] =
            models.try_emplace(key, std::make_unique<CandidateSelectionModel>(arch, solver));
        MIOPEN_LOG_I2("CandidateSelectionModel created for arch: " << arch
                                                                   << ", solver: " << solver);
        return *(it->second);
    }
    catch(const std::exception& ex)
    {
        {
            std::ostringstream oss;
            oss << "Failed to construct CandidateSelectionModel for arch: " << arch
                << ", solver: " << solver << ". Exception: " << ex.what();
            MIOPEN_THROW(miopenStatusInternalError, oss.str());
        }
    }
}

MIOPEN_INTERNALS_EXPORT
std::vector<std::vector<float>>
EncodeKernelParams(const std::vector<std::vector<std::string>>& valid_kernel_params,
                   const CandidateSelectionMetadata& metadata)
{
    std::vector<std::vector<float>> encoded_candidates;
    const auto& output_params          = metadata.output_params();
    const auto& sequence_encodings     = metadata.sequence_encodings();
    const float missing_value_encoding = metadata.GetMissingValueToken();

    for(const auto& candidate : valid_kernel_params)
    {
        // Get kernel_str_mapping for this candidate's kernel_name
        if(candidate.empty())
            MIOPEN_THROW("Candidate vector is empty, cannot extract kernel_name.");
        const std::string& kernel_name = candidate[0];
        const auto& kernel_str_mapping = metadata.GetKernelStrMapping(kernel_name);

        // Build a map from param_name to value for this candidate
        std::map<std::string, std::string> param_value_map;
        bool mapping_valid = true;
        for(const auto& kv : kernel_str_mapping)
        {
            try
            {
                // Use std::stoull for unsigned long long, then validate range
                unsigned long long ull_idx = std::stoull(kv.first);
                size_t idx                 = static_cast<size_t>(ull_idx);

                if(idx < candidate.size())
                    param_value_map[kv.second] = candidate[idx];
                else
                {
                    MIOPEN_LOG_W("Index " << idx << " out of bounds for candidate of size "
                                          << candidate.size() << " in kernel " << kernel_name);
                    mapping_valid = false;
                    break;
                }
            }
            catch(const std::exception& ex)
            {
                MIOPEN_LOG_W("Invalid index format in kernel_str_mapping: "
                             << kv.first << ", error: " << ex.what());
                mapping_valid = false;
                break;
            }
        }

        if(!mapping_valid)
        {
            // Skip this entire candidate rather than partial processing
            // also give a clear log message about the candidate being skipped
            std::ostringstream candidate_str;
            candidate_str << "[";
            for(size_t i = 0; i < candidate.size(); ++i)
            {
                if(i > 0)
                    candidate_str << ", ";
                candidate_str << "\"" << candidate[i] << "\"";
            }
            candidate_str << "]";

            MIOPEN_LOG_W("Skipping candidate due to invalid kernel string mapping. "
                         << "Kernel: " << kernel_name << ", Candidate: " << candidate_str.str()
                         << ", Total mappings: " << kernel_str_mapping.size());
            continue; // Continue to the next candidate
        }

        std::vector<float> encoded;
        for(const auto& param_name : output_params)
        {
            // Skip constant parameters
            if(metadata.GetOutputConstant(param_name).has_value())
                continue;

            float value = missing_value_encoding;

            auto val_it = param_value_map.find(param_name);
            if(val_it != param_value_map.end())
            {
                const std::string& param_value = val_it->second;

                // Handle "nan" token
                if(param_value == "nan")
                {
                    value = missing_value_encoding;
                }
                else
                {
                    // Encode using sequence_encodings
                    const auto enc_it = sequence_encodings.find(param_name);

                    if(enc_it == sequence_encodings.end())
                    {
                        // Try to cast param_value to float if no encoding is found
                        try
                        {
                            value = std::stof(param_value);
                        }
                        catch(const std::exception&)
                        {
                            std::ostringstream msg;
                            msg << "No sequence encoding found for output parameter: " << param_name
                                << " and value '" << param_value << "' is not a valid float.";
                            MIOPEN_THROW(msg.str());
                        }
                    }
                    else
                    {
                        const auto& value_map = enc_it->second;
                        const auto map_it     = value_map.find(param_value);

                        if(map_it == value_map.end())
                        {
                            // Secondary check: try matching param_value with all whitespace removed
                            std::string param_value_ws;
                            std::remove_copy_if(param_value.begin(),
                                                param_value.end(),
                                                std::back_inserter(param_value_ws),
                                                [](unsigned char c) { return std::isspace(c); });

                            bool found_ws = false;
                            for(const auto& kv : value_map)
                            {
                                std::string key_ws;
                                std::remove_copy_if(
                                    kv.first.begin(),
                                    kv.first.end(),
                                    std::back_inserter(key_ws),
                                    [](unsigned char c) { return std::isspace(c); });
                                if(param_value_ws == key_ws)
                                {
                                    value    = static_cast<float>(kv.second);
                                    found_ws = true;
                                    break;
                                }
                            }

                            if(!found_ws)
                            {
                                MIOPEN_LOG_WE("No encoding found in metadata for value '"
                                              << param_value
                                              << "' of output parameter: " << param_name);
                                MIOPEN_LOG_WE("setting it to the NaN value");
                                value = missing_value_encoding;
                            }
                        }
                        else
                        {
                            // Use the encoded value from the map
                            value = static_cast<float>(map_it->second);
                        }
                    }
                }
            }
            // If not present, value remains missing_value_encoding
            encoded.push_back(value);
        }
        encoded_candidates.push_back(encoded);
    }

    return encoded_candidates;
}

MIOPEN_INTERNALS_EXPORT
CandidateSelectionResult
ModelSelectBestCandidate(const std::string& arch,
                         const std::string& solver,
                         const std::map<std::string, float>& features,
                         const std::vector<std::vector<std::string>>& valid_kernel_params,
                         const bool use_split_k)
{
    try
    {
        const auto& model = GetCandidateSelectionModel(arch, solver);
        // debug: show that we successfully retrieved the model
        MIOPEN_LOG_I2("Retrieved CandidateSelectionModel for arch: " << arch
                                                                     << ", solver: " << solver);
        std::vector<std::vector<std::string>> expanded_params = valid_kernel_params;
        std::vector<std::pair<int, int>> mapping_pairs;
        std::vector<int> heuristic_indexes;
        heuristic_indexes.reserve(valid_kernel_params.size()); // Pre-allocate capacity
        for(size_t i = 0; i < valid_kernel_params.size(); ++i)
            heuristic_indexes.push_back(static_cast<int>(i));

        if(use_split_k)
        {
            // get split_k values from metadata
            const auto& split_ks = model.metadata().GetSplitKValues();

            // Expand kernel params with split_k and keep mapping
            std::tie(expanded_params, mapping_pairs) =
                ExpandKernelParamsWithSplitK(valid_kernel_params, heuristic_indexes, split_ks);
        }
        else
        {

            // If split_k is 0, we do not expand, just use the original kernels
            for(int heuristic_index : heuristic_indexes)
            {
                mapping_pairs.emplace_back(heuristic_index, 1); // Default split_k of 1
            }
        }
        const auto& encoded_candidates = EncodeKernelParams(expanded_params, model.metadata());

        if(encoded_candidates.empty())
        {
            MIOPEN_LOG_W("No valid encoded candidates available");
            return CandidateSelectionResult{{}, {}};
        }

        const auto& encoded_features = model.EncodeInputFeatures(features);
        const auto& encoded_configs  = model.EncodeKernelConfigs(encoded_candidates);

        // Get all candidates sorted by score (best to worst)
        auto scored_candidates =
            model.SelectBestCandidateIndices(encoded_features, encoded_configs);
        ;

        CandidateSelectionResult result;
        result.kernel_indices.reserve(scored_candidates.size());
        result.split_k_values.reserve(scored_candidates.size());

        for(const auto& [candidate_idx, score] : scored_candidates)
        {
            if(candidate_idx >= 0 && candidate_idx < static_cast<int>(mapping_pairs.size()))
            {
                result.kernel_indices.push_back(mapping_pairs[candidate_idx].first);
                result.split_k_values.push_back(mapping_pairs[candidate_idx].second);
            }
        }

        return result;
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_I2("[Warning] Candidate selection model failed: " << ex.what());
        return CandidateSelectionResult{{}, {}};
    }
    catch(const std::exception& ex)
    {
        MIOPEN_LOG_I2(
            "[Warning] Candidate selection model failed with std exception: " << ex.what());
        return CandidateSelectionResult{{}, {}};
    }
}

} // namespace candidate_selection
} // namespace tuning
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
