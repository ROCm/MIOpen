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
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <map>

namespace miopen {
namespace ai {
namespace tuning {
namespace candidate_selection {

// Forward declarations for the helpers implemented in ai_heuristics.cpp
std::vector<float> EncodeInputFeaturesWithFdeep(const std::vector<float>& features,
                                                const std::string& arch,
                                                const std::string& solver,
                                                const std::vector<size_t>&& drop_indices);

std::vector<std::vector<float>>
EncodeKernelConfigsWithFdeep(const std::vector<std::vector<float>>& encoded_candidates,
                             const std::string& arch,
                             const std::string& solver,
                             const std::vector<size_t>&& drop_indices);

class CandidateSelectionMetadata
{
public:
    CandidateSelectionMetadata(const std::string& arch, const std::string& solver);
    size_t GetInputParamIndex(const std::string& name) const;
    size_t GetOutputParamIndex(const std::string& name) const;
    std::optional<std::string> GetInputConstant(const std::string& name) const;
    std::optional<std::string> GetOutputConstant(const std::string& name) const;
    std::vector<size_t> GetConstantInputIndices() const;
    std::vector<size_t> GetConstantOutputIndices() const;
    // Getter functions for private members
    const std::vector<std::string>& input_params() const { return input_params_; }
    const std::vector<std::string>& output_params() const { return output_params_; }
    const std::map<std::string, std::map<std::string, int>>& sequence_encodings() const
    {
        return sequence_encodings_;
    }

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
};

class CandidateSelectionModel
{
public:
    CandidateSelectionModel(const std::string& arch, const std::string& solver);
    ~CandidateSelectionModel();

    std::vector<float> EncodeInputFeatures(const std::vector<float>& features) const;
    std::vector<std::vector<float>>
    EncodeKernelConfigs(const std::vector<std::vector<float>>& encoded_candidates) const;
    int SelectBestCandidateIdx(const std::vector<float>& encoded_features,
                               const std::vector<std::vector<float>>& encoded_configs) const;
    const CandidateSelectionMetadata& metadata() const { return metadata_; }

private:
    CandidateSelectionMetadata metadata_;
    std::string arch_;
    std::string solver_;
};

const CandidateSelectionModel& GetCandidateSelectionModel(const std::string& arch,
                                                          const std::string& solver);

std::vector<std::vector<float>>
EncodeKernelParams(const std::vector<std::vector<std::string>>& valid_kernel_params,
                   const CandidateSelectionMetadata& metadata);

int ModelSelectBestCandidate(const std::string& arch,
                             const std::string& solver,
                             const std::vector<float>& features,
                             const std::vector<std::vector<std::string>>& valid_kernel_params);

} // namespace candidate_selection
} // namespace tuning
} // namespace ai
} // namespace miopen
