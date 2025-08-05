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

#include <gtest/gtest.h>
#include <iostream>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/filesystem.hpp>

using namespace miopen::ai::tuning::candidate_selection;

class CandidateSelectionTest : public ::testing::Test
{
protected:
    std::string arch   = "gfx942";
    std::string solver = "ConvHipImplicitGemm3DGroupWrwXdlops";
};

TEST_F(CandidateSelectionTest, FilesExist)
{
    auto db_path       = miopen::GetSystemDbPath();
    auto input_encoder = db_path / (arch + "_" + solver + "_input_encoder.tn.model");
    auto kernel_config_encoder =
        db_path / (arch + "_" + solver + "_kernel_config_encoder.tn.model");
    auto metadata = db_path / (arch + "_" + solver + "_metadata.tn.model");

    ASSERT_TRUE(miopen::fs::exists(input_encoder)) << "Input encoder file missing!";
    ASSERT_TRUE(miopen::fs::exists(kernel_config_encoder)) << "Kernel config encoder file missing!";
    ASSERT_TRUE(miopen::fs::exists(metadata)) << "Metadata file missing!";
}

TEST_F(CandidateSelectionTest, MetadataAndModelInit)
{
    ASSERT_NO_THROW({
        CandidateSelectionMetadata meta(arch, solver);
        CandidateSelectionModel model(arch, solver);
    });
}

TEST_F(CandidateSelectionTest, ModelCaching)
{
    auto& model1 = GetCandidateSelectionModel(arch, solver);
    auto& model2 = GetCandidateSelectionModel(arch, solver);
    ASSERT_EQ(&model1, &model2)
        << "GetCandidateSelectionModel did not return the same cached object!";
}

TEST_F(CandidateSelectionTest, EncodeInputFeatures)
{
    CandidateSelectionModel model(arch, solver);
    CandidateSelectionMetadata meta(arch, solver);

    // Initialize features as a map with all input_params set to 1.0f
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
    {
        features[name] = 1.0f;
    }

    auto encoded = model.EncodeInputFeatures(features);
    ASSERT_FALSE(encoded.empty()) << "EncodeInputFeatures returned empty vector!";
}
TEST_F(CandidateSelectionTest, EncodeKernelConfigs)
{
    CandidateSelectionModel model(arch, solver);
    CandidateSelectionMetadata meta(arch, solver);
    size_t feature_size = meta.output_params().size();
    std::vector<std::vector<float>> encoded_candidates(100, std::vector<float>(feature_size, 2.0f));

    auto encoded = model.EncodeKernelConfigs(encoded_candidates);
    ASSERT_FALSE(encoded.empty()) << "EncodeKernelConfigs returned empty vector!";
    for(const auto& vec : encoded)
    {
        ASSERT_FALSE(vec.empty()) << "EncodeKernelConfigs returned a candidate with empty vector!";
    }
}

TEST_F(CandidateSelectionTest, EncodeInputFeaturesEdgeCases)
{
    CandidateSelectionModel model(arch, solver);
    CandidateSelectionMetadata meta(arch, solver);

    // Empty input
    std::map<std::string, float> empty_features;
    EXPECT_THROW(model.EncodeInputFeatures(empty_features), std::exception);

    // Input smaller than expected (missing keys)
    std::map<std::string, float> short_features;
    if(!meta.input_params().empty())
    {
        // Add all but the last input param
        for(size_t i = 0; i < meta.input_params().size() - 1; ++i)
        {
            short_features[meta.input_params()[i]] = 1.0f;
        }
        EXPECT_THROW(model.EncodeInputFeatures(short_features), std::exception);
    }

    // Input larger than expected (extra keys)
    std::map<std::string, float> long_features;
    for(const auto& name : meta.input_params())
    {
        long_features[name] = 1.0f;
    }
    long_features["extra_param"] = 2.0f; // Add an extra key
    // Should not throw, extra keys are ignored
    EXPECT_NO_THROW({
        auto encoded = model.EncodeInputFeatures(long_features);
        ASSERT_FALSE(encoded.empty());
    });

    // Input containing constants (if any constants are defined)
    if(!meta.GetConstantInputIndices().empty())
    {
        std::map<std::string, float> features;
        for(const auto& name : meta.input_params())
        {
            features[name] = 1.0f;
        }
        for(auto idx : meta.GetConstantInputIndices())
        {
            if(idx < meta.input_params().size())
                features[meta.input_params()[idx]] = 42.0f;
        }
        EXPECT_NO_THROW({
            auto encoded = model.EncodeInputFeatures(features);
            ASSERT_FALSE(encoded.empty());
        });
    }
}

TEST_F(CandidateSelectionTest, EncodeKernelConfigsEdgeCases)
{
    CandidateSelectionModel model(arch, solver);
    CandidateSelectionMetadata meta(arch, solver);

    // Empty input
    std::vector<std::vector<float>> empty_candidates;
    EXPECT_THROW(model.EncodeKernelConfigs(empty_candidates), std::exception);

    // Candidate with wrong size (too short)
    std::vector<std::vector<float>> candidates_short(
        1,
        std::vector<float>(meta.output_params().size() > 0 ? meta.output_params().size() - 1 : 0,
                           2.0f));
    EXPECT_THROW(model.EncodeKernelConfigs(candidates_short), std::exception);

    // Candidate with wrong size (too long)
    std::vector<std::vector<float>> candidates_long(
        1, std::vector<float>(meta.output_params().size() + 1, 2.0f));
    EXPECT_THROW(model.EncodeKernelConfigs(candidates_long), std::exception);

    // Candidates containing constants (if any constants are defined)
    if(!meta.GetConstantOutputIndices().empty())
    {
        std::vector<std::vector<float>> candidates(
            2, std::vector<float>(meta.output_params().size(), 2.0f));
        for(auto idx : meta.GetConstantOutputIndices())
        {
            for(auto& candidate : candidates)
            {
                if(idx < candidate.size())
                    candidate[idx] = 99.0f;
            }
        }
        EXPECT_NO_THROW({
            auto encoded = model.EncodeKernelConfigs(candidates);
            ASSERT_FALSE(encoded.empty());
        });
    }
}

TEST_F(CandidateSelectionTest, SelectBestCandidateValid)
{
    CandidateSelectionModel model(arch, solver);
    CandidateSelectionMetadata meta(arch, solver);

    // Initialize features as a map with all input_params set to 1.0f
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
    {
        features[name] = 1.0f;
    }
    auto encoded_features = model.EncodeInputFeatures(features);

    // Prepare valid_kernel_params as vector<vector<string>>
    std::vector<std::vector<std::string>> valid_kernel_params(
        3, std::vector<std::string>(meta.output_params().size(), "2"));

    auto encoded_candidates = EncodeKernelParams(valid_kernel_params, meta);
    auto encoded_configs    = model.EncodeKernelConfigs(encoded_candidates);

    int idx = model.SelectBestCandidateIdx(encoded_features, encoded_configs);
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, static_cast<int>(valid_kernel_params.size()));
}

TEST_F(CandidateSelectionTest, SelectBestCandidateMismatchedDims)
{
    CandidateSelectionModel model(arch, solver);
    CandidateSelectionMetadata meta(arch, solver);

    // Initialize features as a map with all input_params set to 1.0f
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
    {
        features[name] = 1.0f;
    }
    auto encoded_features = model.EncodeInputFeatures(features);

    // Prepare mismatched kernel params (output_params.size() + 1)
    std::vector<std::vector<std::string>> valid_kernel_params(
        3, std::vector<std::string>(meta.output_params().size() + 1, "2"));

    auto encoded_candidates = EncodeKernelParams(valid_kernel_params, meta);
    auto encoded_configs    = encoded_candidates;

    EXPECT_THROW(model.SelectBestCandidateIdx(encoded_features, encoded_configs), std::exception);
}

TEST_F(CandidateSelectionTest, SelectBestCandidateEmptyInput)
{
    CandidateSelectionModel model(arch, solver);
    std::vector<float> encoded_features;
    std::vector<std::vector<float>> encoded_configs;
    EXPECT_THROW(model.SelectBestCandidateIdx(encoded_features, encoded_configs), std::exception);
}

TEST_F(CandidateSelectionTest, ModelSelectBestCandidate)
{
    CandidateSelectionMetadata meta(arch, solver);
    // Initialize features as a map with all input_params set to 1.0f
    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
    {
        features[name] = 1.0f;
    }
    std::vector<std::vector<std::string>> valid_kernel_params(
        3, std::vector<std::string>(meta.output_params().size(), "2"));

    int idx = ModelSelectBestCandidate(arch, solver, features, valid_kernel_params);
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, static_cast<int>(valid_kernel_params.size()));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
