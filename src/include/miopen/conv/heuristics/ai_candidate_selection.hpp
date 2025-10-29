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
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <map>
#include <miopen/config.hpp>

namespace miopen {
namespace ai {
namespace tuning {
namespace candidate_selection {

// Forward declarations for the helpers implemented in ai_heuristics.cpp
std::vector<float> EncodeInputFeaturesWithFdeep(const std::vector<float>& features,
                                                const std::string& arch,
                                                const std::string& solver);

std::vector<std::vector<float>>
EncodeKernelConfigsWithFdeep(const std::vector<std::vector<float>>& encoded_candidates,
                             const std::string& arch,
                             const std::string& solver);

class CandidateSelectionMetadata
{
public:
    MIOPEN_INTERNALS_EXPORT CandidateSelectionMetadata(const std::string& arch,
                                                       const std::string& solver);
    MIOPEN_INTERNALS_EXPORT size_t GetInputParamIndex(const std::string& name) const;
    MIOPEN_INTERNALS_EXPORT size_t GetOutputParamIndex(const std::string& name) const;
    MIOPEN_INTERNALS_EXPORT std::optional<std::string>
    GetInputConstant(const std::string& name) const;
    MIOPEN_INTERNALS_EXPORT std::optional<std::string>
    GetOutputConstant(const std::string& name) const;
    MIOPEN_INTERNALS_EXPORT std::vector<size_t> GetConstantInputIndices() const;
    MIOPEN_INTERNALS_EXPORT std::vector<size_t> GetConstantOutputIndices() const;
    MIOPEN_INTERNALS_EXPORT std::map<std::string, std::string>
    GetKernelStrMapping(const std::string& kernel_name) const;
    // Getter functions for private members
    MIOPEN_INTERNALS_EXPORT const std::vector<std::string>& input_params() const
    {
        return input_params_;
    }
    MIOPEN_INTERNALS_EXPORT const std::vector<std::string>& output_params() const
    {
        return output_params_;
    }
    MIOPEN_INTERNALS_EXPORT const std::map<std::string, std::map<std::string, int>>&
    sequence_encodings() const;
    MIOPEN_INTERNALS_EXPORT float GetMissingValueToken() const;
    MIOPEN_INTERNALS_EXPORT const std::vector<int>& GetSplitKValues() const;

private:
    // Internal mappings and encodings
    std::vector<std::string> input_params_;
    std::vector<std::string> output_params_;
    std::map<std::string, std::map<std::string, int>> sequence_encodings_;
    std::map<std::string, size_t> input_param_indices_;
    std::map<std::string, size_t> output_param_indices_;
    std::map<std::string, std::map<std::string, int>> feature_encodings_;
    std::map<std::string, std::map<std::string, std::string>> sequence_decodings_;
    std::map<std::string, std::string> constants_features_;
    std::map<std::string, std::string> constants_sequence_;
    std::map<std::string, std::map<std::string, std::string>> kernel_str_mapping_;
    float missing_value_token_;
    std::vector<int> split_k_values_;
};

class CandidateSelectionModel
{
public:
    MIOPEN_INTERNALS_EXPORT CandidateSelectionModel(const std::string& arch,
                                                    const std::string& solver);
    MIOPEN_INTERNALS_EXPORT ~CandidateSelectionModel();

    MIOPEN_INTERNALS_EXPORT std::vector<float>
    EncodeInputFeatures(const std::map<std::string, float>& features) const;
    MIOPEN_INTERNALS_EXPORT std::vector<std::vector<float>>
    EncodeKernelConfigs(const std::vector<std::vector<float>>& encoded_candidates) const;
    MIOPEN_INTERNALS_EXPORT std::vector<std::pair<int, float>>
    SelectBestCandidateIndices(const std::vector<float>& encoded_features,
                               const std::vector<std::vector<float>>& encoded_configs) const;
    const CandidateSelectionMetadata& metadata() const { return metadata_; }

private:
    CandidateSelectionMetadata metadata_;
    std::string arch_;
    std::string solver_;
};

MIOPEN_INTERNALS_EXPORT const CandidateSelectionModel&
GetCandidateSelectionModel(const std::string& arch, const std::string& solver);

MIOPEN_INTERNALS_EXPORT std::vector<std::vector<float>>
EncodeKernelParams(const std::vector<std::vector<std::string>>& valid_kernel_params,
                   const CandidateSelectionMetadata& metadata);

MIOPEN_INTERNALS_EXPORT struct CandidateSelectionResult
{
    std::vector<int> kernel_indices; // Sorted list of kernel indices (best to worst)
    std::vector<int> split_k_values; // Corresponding split_k values

    // Helper methods for backward compatibility and convenience
    int GetBestKernelIndex() const { return kernel_indices.empty() ? -1 : kernel_indices[0]; }
    int GetBestSplitK() const { return split_k_values.empty() ? 1 : split_k_values[0]; }

    int GetFallbackKernelIndex(size_t fallback_level = 1) const
    {
        return (fallback_level < kernel_indices.size()) ? kernel_indices[fallback_level] : -1;
    }

    int GetFallbackSplitK(size_t fallback_level = 1) const
    {
        return (fallback_level < split_k_values.size()) ? split_k_values[fallback_level] : 1;
    }

    size_t GetNumCandidates() const { return kernel_indices.size(); }
    bool IsEmpty() const { return kernel_indices.empty(); }
};

MIOPEN_INTERNALS_EXPORT
CandidateSelectionResult
ModelSelectBestCandidate(const std::string& arch,
                         const std::string& solver,
                         const std::map<std::string, float>& features,
                         const std::vector<std::vector<std::string>>& valid_kernel_params,
                         bool use_split_k);

MIOPEN_INTERNALS_EXPORT
std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<int, int>>>
ExpandKernelParamsWithSplitK(const std::vector<std::vector<std::string>>& kernels,
                             const std::vector<int>& indexes,
                             const std::vector<int>& split_ks);

} // namespace candidate_selection
} // namespace tuning
} // namespace ai
} // namespace miopen
