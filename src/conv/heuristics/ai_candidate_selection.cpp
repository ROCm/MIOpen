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

namespace miopen {
namespace ai {
namespace tuning {
namespace candidate_selection {

// --- CandidateSelectionMetadata ---------------------------------------------

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
        std::cerr << "JSON parse error: " << ex.what() << std::endl;
        std::cerr.flush();
        throw;
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
}

size_t CandidateSelectionMetadata::GetInputParamIndex(const std::string& name) const
{
    auto it = input_param_indices_.find(name);
    if(it == input_param_indices_.end())
        MIOPEN_THROW("Input parameter not found: " + name);
    return it->second;
}

size_t CandidateSelectionMetadata::GetOutputParamIndex(const std::string& name) const
{
    auto it = output_param_indices_.find(name);
    if(it == output_param_indices_.end())
        MIOPEN_THROW("Output parameter not found: " + name);
    return it->second;
}

std::optional<std::string>
CandidateSelectionMetadata::GetInputConstant(const std::string& name) const
{
    auto it = constants_features_.find(name);
    if(it != constants_features_.end())
        return it->second;
    return std::nullopt;
}

std::optional<std::string>
CandidateSelectionMetadata::GetOutputConstant(const std::string& name) const
{
    auto it = constants_sequence_.find(name);
    if(it != constants_sequence_.end())
        return it->second;
    return std::nullopt;
}

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

// --- CandidateSelectionModel ------------------------------------------------

CandidateSelectionModel::CandidateSelectionModel(const std::string& arch, const std::string& solver)
    : metadata_(arch, solver), arch_(arch), solver_(solver)
{
}

CandidateSelectionModel::~CandidateSelectionModel() = default;

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
            MIOPEN_THROW("Input feature not found in provided map: " + name);
        }
    }

    // Pass the filtered vector to the encoding function
    return EncodeInputFeaturesWithFdeep(filtered_features, arch_, solver_);
}

std::vector<std::vector<float>> CandidateSelectionModel::EncodeKernelConfigs(
    const std::vector<std::vector<float>>& encoded_candidates) const
{
    return EncodeKernelConfigsWithFdeep(encoded_candidates, arch_, solver_);
}

int CandidateSelectionModel::SelectBestCandidateIdx(
    const std::vector<float>& encoded_features,
    const std::vector<std::vector<float>>& encoded_configs) const
{
    if(encoded_configs.empty() || encoded_features.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "Empty features or configs in SelectBestCandidateIdx");
    }

    size_t feature_dim    = encoded_features.size();
    size_t num_candidates = encoded_configs.size();

    std::vector<float> selection_scores(num_candidates, 0.0f);

    for(size_t i = 0; i < num_candidates; ++i)
    {
        if(encoded_configs[i].size() != feature_dim)
            MIOPEN_THROW(miopenStatusInternalError,
                         "Config dimension mismatch in SelectBestCandidateIdx");
        selection_scores[i] = std::inner_product(
            encoded_configs[i].begin(), encoded_configs[i].end(), encoded_features.begin(), 0.0f);
    }

    return static_cast<int>(std::max_element(selection_scores.begin(), selection_scores.end()) -
                            selection_scores.begin());
}

// --- Factory and Helper Functions -------------------------------------------

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
        MIOPEN_THROW(miopenStatusInternalError,
                     "Failed to construct CandidateSelectionModel for arch: " + arch +
                         ", solver: " + solver + ". Exception: " + ex.what());
    }
}

std::vector<std::vector<float>>
EncodeKernelParams(const std::vector<std::vector<std::string>>& valid_kernel_params,
                   const CandidateSelectionMetadata& metadata)
{
    std::vector<std::vector<float>> encoded_candidates;
    const auto& output_params      = metadata.output_params();
    const auto& sequence_encodings = metadata.sequence_encodings();

    // NOTE: If candidate.size() < output_params.size(), extra output_params are ignored.
    // The order of candidate elements is assumed to match output_params.

    for(const auto& candidate : valid_kernel_params)
    {
        std::vector<float> encoded;
        for(size_t i = 0; i < candidate.size(); ++i)
        {
            if(i >= output_params.size())
                break; // Ignore extra candidate elements

            const std::string& param_name  = output_params[i];
            const std::string& param_value = candidate[i];

            // Skip constant parameters
            if(metadata.GetOutputConstant(param_name).has_value())
                continue;

            // Encode using sequence_encodings
            const auto enc_it = sequence_encodings.find(param_name);
            if(enc_it == sequence_encodings.end())
            {
                // Try to cast param_value to float if no encoding is found
                try
                {
                    float float_val = std::stof(param_value);
                    encoded.push_back(float_val);
                    continue;
                }
                catch(const std::exception&)
                {
                    MIOPEN_THROW("No sequence encoding found for output parameter: " + param_name +
                                 " and value '" + param_value + "' is not a valid float.");
                }
            }

            const auto& value_map = enc_it->second;
            const auto val_it     = value_map.find(param_value);
            if(val_it == value_map.end())
            {
                MIOPEN_THROW("No encoding found for value '" + param_value +
                             "' of output parameter: " + param_name);
            }

            encoded.push_back(static_cast<float>(val_it->second));
        }
        encoded_candidates.push_back(encoded);
    }

    return encoded_candidates;
}

int ModelSelectBestCandidate(const std::string& arch,
                             const std::string& solver,
                             const std::map<std::string, float>& features,
                             const std::vector<std::vector<std::string>>& valid_kernel_params)
{
    try
    {
        const auto& model = GetCandidateSelectionModel(arch, solver);

        const auto& encoded_candidates = EncodeKernelParams(valid_kernel_params, model.metadata());

        if(encoded_candidates.empty())
        {
            MIOPEN_LOG_W("No valid encoded candidates available");
            return -1;
        }

        const auto& encoded_features = model.EncodeInputFeatures(features);
        const auto& encoded_configs  = model.EncodeKernelConfigs(encoded_candidates);

        const int best_idx = model.SelectBestCandidateIdx(encoded_features, encoded_configs);

        if(best_idx >= 0 && best_idx < static_cast<int>(valid_kernel_params.size()))
        {
            return best_idx;
        }
        else
        {
            MIOPEN_LOG_W("Invalid candidate index returned: " << best_idx);
            return -1;
        }
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_I2("[Warning] Candidate selection model failed: " << ex.what());
        return -1;
    }
    catch(const std::exception& ex)
    {
        MIOPEN_LOG_I2(
            "[Warning] Candidate selection model failed with std exception: " << ex.what());
        return -1;
    }
}

} // namespace candidate_selection
} // namespace tuning
} // namespace ai
} // namespace miopen
