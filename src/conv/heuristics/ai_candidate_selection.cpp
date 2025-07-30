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
    if(!file)
        MIOPEN_THROW("Could not open metadata file: " + path.string());
    nlohmann::json metadata;
    file >> metadata;

    input_params  = metadata.value("input_params", std::vector<std::string>{});
    output_params = metadata.value("output_params", std::vector<std::string>{});

    for(size_t i = 0; i < input_params.size(); ++i)
        input_param_indices[input_params[i]] = i;
    for(size_t i = 0; i < output_params.size(); ++i)
        output_param_indices[output_params[i]] = i;

    if(metadata.contains("encodings"))
    {
        feature_encodings  = metadata["encodings"].value("inputs", decltype(feature_encodings){});
        sequence_encodings = metadata["encodings"].value("outputs", decltype(sequence_encodings){});
    }

    if(metadata.contains("decodings") && metadata["decodings"].contains("outputs"))
    {
        sequence_decodings =
            metadata["decodings"]["outputs"]
                .get<std::unordered_map<std::string,
                                        std::unordered_map<std::string, std::string>>>();
    }

    if(metadata.contains("constants"))
    {
        constants_features = metadata["constants"].value("inputs", decltype(constants_features){});
        constants_sequence = metadata["constants"].value("outputs", decltype(constants_sequence){});
    }
}

size_t CandidateSelectionMetadata::GetInputParamIndex(const std::string& name) const
{
    auto it = input_param_indices.find(name);
    if(it == input_param_indices.end())
        MIOPEN_THROW("Input parameter not found: " + name);
    return it->second;
}

size_t CandidateSelectionMetadata::GetOutputParamIndex(const std::string& name) const
{
    auto it = output_param_indices.find(name);
    if(it == output_param_indices.end())
        MIOPEN_THROW("Output parameter not found: " + name);
    return it->second;
}

std::optional<std::string>
CandidateSelectionMetadata::GetInputConstant(const std::string& name) const
{
    auto it = constants_features.find(name);
    if(it != constants_features.end())
        return it->second;
    return std::nullopt;
}

std::optional<std::string>
CandidateSelectionMetadata::GetOutputConstant(const std::string& name) const
{
    auto it = constants_sequence.find(name);
    if(it != constants_sequence.end())
        return it->second;
    return std::nullopt;
}

std::vector<size_t> CandidateSelectionMetadata::GetConstantInputIndices() const
{
    std::vector<size_t> indices;
    for(const auto& [name, value] : constants_features)
    {
        auto it = input_param_indices.find(name);
        if(it != input_param_indices.end())
            indices.push_back(it->second);
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

std::vector<size_t> CandidateSelectionMetadata::GetConstantOutputIndices() const
{
    std::vector<size_t> indices;
    for(const auto& [name, value] : constants_sequence)
    {
        auto it = output_param_indices.find(name);
        if(it != output_param_indices.end())
            indices.push_back(it->second);
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

// --- CandidateSelectionModel ------------------------------------------------

CandidateSelectionModel::CandidateSelectionModel(const std::string& arch, const std::string& solver)
    : metadata(arch, solver), arch_(arch), solver_(solver)
{
}

CandidateSelectionModel::~CandidateSelectionModel() = default;

std::vector<float>
CandidateSelectionModel::EncodeInputFeatures(const std::vector<float>& features) const
{
    return EncodeInputFeaturesWithFdeep(
        features, arch_, solver_, metadata.GetConstantInputIndices());
}

std::vector<std::vector<float>> CandidateSelectionModel::EncodeKernelConfigs(
    const std::vector<std::vector<float>>& encoded_candidates) const
{
    return EncodeKernelConfigsWithFdeep(
        encoded_candidates, arch_, solver_, metadata.GetConstantOutputIndices());
}

int CandidateSelectionModel::SelectBestCandidate(
    const std::vector<float>& encoded_features,
    const std::vector<std::vector<float>>& encoded_configs) const
{
    if(encoded_configs.empty() || encoded_features.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError, "Empty features or configs in SelectBestCandidate");
    }

    size_t feature_dim    = encoded_features.size();
    size_t num_candidates = encoded_configs.size();

    std::vector<float> selection_scores(num_candidates, 0.0f);

    for(size_t i = 0; i < num_candidates; ++i)
    {
        if(encoded_configs[i].size() != feature_dim)
            MIOPEN_THROW(miopenStatusInternalError,
                         "Config dimension mismatch in SelectBestCandidate");
        selection_scores[i] = std::inner_product(
            encoded_configs[i].begin(), encoded_configs[i].end(), encoded_features.begin(), 0.0f);
    }

    return static_cast<int>(std::max_element(selection_scores.begin(), selection_scores.end()) -
                            selection_scores.begin());
}

// --- Factory and Helper Functions -------------------------------------------

std::shared_ptr<CandidateSelectionModel> GetCandidateSelectionModel(const std::string& arch,
                                                                    const std::string& solver)
{
    static std::map<std::string, std::shared_ptr<CandidateSelectionModel>> models;
    std::string key = arch + "_" + solver;
    auto it         = models.find(key);
    if(it == models.end())
    {
        std::shared_ptr<CandidateSelectionModel> model =
            std::make_shared<CandidateSelectionModel>(arch, solver);
        models[key] = model;
        return model;
    }
    else
    {
        return it->second;
    }
}

std::vector<std::vector<float>>
EncodeKernelParams(const std::vector<std::vector<std::string>>& valid_kernel_params,
                   const CandidateSelectionMetadata& metadata)
{
    std::vector<std::vector<float>> encoded_candidates;
    const auto& output_params      = metadata.output_params;
    const auto& sequence_encodings = metadata.sequence_encodings;

    for(const auto& candidate : valid_kernel_params)
    {
        std::vector<float> encoded;
        for(size_t i = 0; i < candidate.size(); ++i)
        {
            const std::string& param_name  = output_params[i];
            const std::string& param_value = candidate[i];

            auto enc_it = sequence_encodings.find(param_name);
            if(enc_it != sequence_encodings.end())
            {
                const auto& value_map = enc_it->second;
                auto val_it           = value_map.find(param_value);
                if(val_it != value_map.end())
                {
                    encoded.push_back(static_cast<float>(val_it->second));
                    continue;
                }
            }

            try
            {
                encoded.push_back(std::stof(param_value));
            }
            catch(const std::exception&)
            {
                encoded.push_back(-1.0f);
            }
        }
        encoded_candidates.push_back(encoded);
    }

    return encoded_candidates;
}

int ModelSelectBestCandidate(const std::string& arch,
                             const std::string& solver,
                             const std::vector<float>& features,
                             const std::vector<std::vector<std::string>>& valid_kernel_params)
{
    try
    {
        auto model = GetCandidateSelectionModel(arch, solver);

        auto encoded_candidates = EncodeKernelParams(valid_kernel_params, model->metadata);

        if(encoded_candidates.empty())
        {
            MIOPEN_LOG_W("No valid encoded candidates available");
            return -1;
        }

        auto encoded_features = model->EncodeInputFeatures(features);
        auto encoded_configs  = model->EncodeKernelConfigs(encoded_candidates);

        int best_idx = model->SelectBestCandidate(encoded_features, encoded_configs);

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
