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

#include <iostream>
#include <cassert>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/filesystem.hpp>

using namespace miopen::ai::tuning::candidate_selection;

// basic test functions for candidate selection model and metadata
void TestFilesExist(const std::string& arch, const std::string& solver)
{
    auto db_path       = miopen::GetSystemDbPath();
    auto input_encoder = db_path / (arch + "_" + solver + "_input_encoder.tn.model");
    auto kernel_config_encoder =
        db_path / (arch + "_" + solver + "_kernel_config_encoder.tn.model");
    auto metadata = db_path / (arch + "_" + solver + "_metadata.tn.model");

    std::cout << "Checking existence of:\n"
              << "  " << input_encoder << "\n"
              << "  " << kernel_config_encoder << "\n"
              << "  " << metadata << std::endl;

    if(!miopen::fs::exists(input_encoder))
    {
        std::cerr << "Input encoder file missing!" << std::endl;
        std::abort();
    }
    if(!miopen::fs::exists(kernel_config_encoder))
    {
        std::cerr << "Kernel config encoder file missing!" << std::endl;
        std::abort();
    }
    if(!miopen::fs::exists(metadata))
    {
        std::cerr << "Metadata file missing!" << std::endl;
        std::abort();
    }
}

void TestMetadataAndModelInit(const std::string& arch, const std::string& solver)
{
    try
    {
        CandidateSelectionMetadata meta(arch, solver);
        std::cout << "CandidateSelectionMetadata initialized successfully.\n";
        CandidateSelectionModel model(arch, solver);
        std::cout << "CandidateSelectionModel initialized successfully.\n";
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Initialization failed: " << ex.what() << std::endl;
        std::abort();
    }
}

// tests specific to the metadata class
// void TestMetadataConstants(const std::string& arch, const std::string& solver)
// {
//     CandidateSelectionMetadata meta(arch, solver);

//     std::cout << "Testing metadata constants for arch=" << arch << ", solver=" << solver
//               << std::endl;

//     // Print loaded constants
//     std::cout << "constants_features_.size(): " << meta.constants_features_.size() << std::endl;
//     std::cout << "constants_sequence_.size(): " << meta.constants_sequence_.size() << std::endl;

//     // Print indices
//     auto input_indices  = meta.GetConstantInputIndices();
//     auto output_indices = meta.GetConstantOutputIndices();

//     std::cout << "GetConstantInputIndices(): ";
//     for(auto idx : input_indices)
//         std::cout << idx << " ";
//     std::cout << std::endl;

//     std::cout << "GetConstantOutputIndices(): ";
//     for(auto idx : output_indices)
//         std::cout << idx << " ";
//     std::cout << std::endl;

//     // Fail if constants exist but indices are empty
//     if(!meta.constants_features_.empty() && input_indices.empty())
//     {
//         std::cerr << "constants_features_ present but GetConstantInputIndices() is empty!"
//                   << std::endl;
//         std::abort();
//     }
//     if(!meta.constants_sequence_.empty() && output_indices.empty())
//     {
//         std::cerr << "constants_sequence_ present but GetConstantOutputIndices() is empty!"
//                   << std::endl;
//         std::abort();
//     }
// }

void TestEncodeInputFeatures(const std::string& arch, const std::string& solver)
{
    try
    {
        CandidateSelectionModel model(arch, solver);

        // Prepare a dummy feature vector of the correct size
        CandidateSelectionMetadata meta(arch, solver);
        std::vector<float> features(meta.input_params().size(), 1.0f);

        auto encoded = model.EncodeInputFeatures(features);
        std::cout << "EncodeInputFeatures ran successfully. Output vector size: " << encoded.size()
                  << "\n";
        if(encoded.empty())
        {
            std::cerr << "EncodeInputFeatures returned empty vector!" << std::endl;
            std::abort();
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << "EncodeInputFeatures failed: " << ex.what() << std::endl;
        std::abort();
    }
}

void TestEncodeKernelConfigs(const std::string& arch, const std::string& solver)
{
    try
    {
        CandidateSelectionModel model(arch, solver);

        // Prepare dummy encoded candidates: 100 candidates, each with the correct feature size
        CandidateSelectionMetadata meta(arch, solver);
        size_t feature_size = meta.output_params().size();
        std::vector<std::vector<float>> encoded_candidates(100,
                                                           std::vector<float>(feature_size, 2.0f));

        auto encoded = model.EncodeKernelConfigs(encoded_candidates);
        std::cout << "EncodeKernelConfigs ran successfully. Output vector count: " << encoded.size()
                  << ", each of size: " << (encoded.empty() ? 0 : encoded[0].size()) << "\n";
        if(encoded.empty())
        {
            std::cerr << "EncodeKernelConfigs returned empty vector!" << std::endl;
            std::abort();
        }
        for(const auto& vec : encoded)
        {
            if(vec.empty())
            {
                std::cerr << "EncodeKernelConfigs returned a candidate with empty vector!"
                          << std::endl;
                std::abort();
            }
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << "EncodeKernelConfigs failed: " << ex.what() << std::endl;
        std::abort();
    }
}

void TestEncodeInputFeaturesEdgeCases(const std::string& arch, const std::string& solver)
{
    CandidateSelectionModel model(arch, solver);
    CandidateSelectionMetadata meta(arch, solver);

    // Edge case: empty input
    try
    {
        std::vector<float> empty_features;
        auto encoded = model.EncodeInputFeatures(empty_features);
        std::cerr << "EncodeInputFeatures (empty input) did not throw!" << std::endl;
        std::abort();
    }
    catch(const std::exception& ex)
    {
        std::cout << "EncodeInputFeatures (empty input) correctly threw: " << ex.what()
                  << std::endl;
    }

    // Edge case: input smaller than expected
    try
    {
        std::vector<float> short_features(
            meta.input_params().size() > 0 ? meta.input_params().size() - 1 : 0, 1.0f);
        auto encoded = model.EncodeInputFeatures(short_features);
        std::cerr << "EncodeInputFeatures (short input) did not throw!" << std::endl;
        std::abort();
    }
    catch(const std::exception& ex)
    {
        std::cout << "EncodeInputFeatures (short input) correctly threw: " << ex.what()
                  << std::endl;
    }

    // Edge case: input larger than expected
    try
    {
        std::vector<float> long_features(meta.input_params().size() + 1, 1.0f);
        auto encoded = model.EncodeInputFeatures(long_features);
        std::cerr << "EncodeInputFeatures (long input) did not throw!" << std::endl;
        std::abort();
    }
    catch(const std::exception& ex)
    {
        std::cout << "EncodeInputFeatures (long input) correctly threw: " << ex.what() << std::endl;
    }

    // Input containing constants (if any constants are defined)
    if(!meta.GetConstantInputIndices().empty())
    {
        std::vector<float> features(meta.input_params().size(), 1.0f);
        for(auto idx : meta.GetConstantInputIndices())
        {
            if(idx < features.size())
                features[idx] = 42.0f; // arbitrary constant value
        }
        try
        {
            auto encoded = model.EncodeInputFeatures(features);
            std::cout << "EncodeInputFeatures (with constants) ran. Output size: " << encoded.size()
                      << std::endl;
            if(encoded.empty())
            {
                std::cerr << "EncodeInputFeatures (with constants) returned empty vector!"
                          << std::endl;
                std::abort();
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << "EncodeInputFeatures (with constants) failed: " << ex.what() << std::endl;
            std::abort();
        }
    }
    else
    {
        std::cout << "No constants defined in metadata, skipping constant input test." << std::endl;
    }
}

void TestEncodeKernelConfigsEdgeCases(const std::string& arch, const std::string& solver)
{
    CandidateSelectionModel model(arch, solver);
    CandidateSelectionMetadata meta(arch, solver);

    // Edge case: empty input
    try
    {
        std::vector<std::vector<float>> empty_candidates;
        auto encoded = model.EncodeKernelConfigs(empty_candidates);
        std::cerr << "EncodeKernelConfigs (empty input) did not throw!" << std::endl;
        std::abort();
    }
    catch(const std::exception& ex)
    {
        std::cout << "EncodeKernelConfigs (empty input) correctly threw: " << ex.what()
                  << std::endl;
    }

    // Edge case: candidate with wrong size (too short)
    try
    {
        std::vector<std::vector<float>> candidates(
            1,
            std::vector<float>(
                meta.output_params().size() > 0 ? meta.output_params().size() - 1 : 0, 2.0f));
        auto encoded = model.EncodeKernelConfigs(candidates);
        std::cerr << "EncodeKernelConfigs (short candidate) did not throw!" << std::endl;
        std::abort();
    }
    catch(const std::exception& ex)
    {
        std::cout << "EncodeKernelConfigs (short candidate) correctly threw: " << ex.what()
                  << std::endl;
    }

    // Edge case: candidate with wrong size (too long)
    try
    {
        std::vector<std::vector<float>> candidates(
            1, std::vector<float>(meta.output_params().size() + 1, 2.0f));
        auto encoded = model.EncodeKernelConfigs(candidates);
        std::cerr << "EncodeKernelConfigs (long candidate) did not throw!" << std::endl;
        std::abort();
    }
    catch(const std::exception& ex)
    {
        std::cout << "EncodeKernelConfigs (long candidate) correctly threw: " << ex.what()
                  << std::endl;
    }

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
                    candidate[idx] = 99.0f; // arbitrary constant value
            }
        }
        try
        {
            auto encoded = model.EncodeKernelConfigs(candidates);
            std::cout << "EncodeKernelConfigs (with constants) ran. Output count: "
                      << encoded.size() << std::endl;
            if(encoded.empty())
            {
                std::cerr << "EncodeKernelConfigs (with constants) returned empty vector!"
                          << std::endl;
                std::abort();
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << "EncodeKernelConfigs (with constants) failed: " << ex.what() << std::endl;
            std::abort();
        }
    }
    else
    {
        std::cout << "No constants defined in metadata, skipping constant candidate test."
                  << std::endl;
    }
}

void TestSelectBestCandidateValid(const std::string& arch, const std::string& solver)
{
    try
    {
        CandidateSelectionModel model(arch, solver);

        // Prepare dummy encoded features and configs
        CandidateSelectionMetadata meta(arch, solver);
        std::vector<float> features(meta.input_params().size(), 1.0f);
        auto encoded_features = model.EncodeInputFeatures(features);

        // Prepare 3 dummy configs, each with the correct size
        std::vector<std::vector<float>> encoded_candidates(
            3, std::vector<float>(meta.output_params().size(), 2.0f));
        auto encoded_configs = model.EncodeKernelConfigs(encoded_candidates);

        int idx = model.SelectBestCandidate(encoded_features, encoded_configs);
        std::cout << "SelectBestCandidate (valid) returned: " << idx << std::endl;
        if(idx < 0 || idx >= static_cast<int>(encoded_candidates.size()))
        {
            std::cerr << "SelectBestCandidate returned invalid index!" << std::endl;
            std::abort();
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << "SelectBestCandidate (valid) failed: " << ex.what() << std::endl;
        std::abort();
    }
}

void TestSelectBestCandidateMismatchedDims(const std::string& arch, const std::string& solver)
{
    try
    {
        CandidateSelectionModel model(arch, solver);

        CandidateSelectionMetadata meta(arch, solver);
        std::vector<float> features(meta.input_params().size(), 1.0f);
        auto encoded_features = model.EncodeInputFeatures(features);

        // Prepare configs with mismatched size
        std::vector<std::vector<float>> encoded_candidates(
            3, std::vector<float>(meta.output_params().size() + 1, 2.0f));
        auto encoded_configs = encoded_candidates; // skip encoding for this test

        // Should throw or abort
        int idx = model.SelectBestCandidate(encoded_features, encoded_configs);
        std::cerr << "SelectBestCandidate (mismatched dims) did not throw, returned: " << idx
                  << std::endl;
        std::abort();
    }
    catch(const std::exception& ex)
    {
        std::cout << "SelectBestCandidate (mismatched dims) correctly threw: " << ex.what()
                  << std::endl;
    }
}

void TestSelectBestCandidateEmptyInput(const std::string& arch, const std::string& solver)
{
    try
    {
        CandidateSelectionModel model(arch, solver);

        std::vector<float> encoded_features;             // empty
        std::vector<std::vector<float>> encoded_configs; // empty

        int idx = model.SelectBestCandidate(encoded_features, encoded_configs);
        std::cerr << "SelectBestCandidate (empty input) did not throw, returned: " << idx
                  << std::endl;
        std::abort();
    }
    catch(const std::exception& ex)
    {
        std::cout << "SelectBestCandidate (empty input) correctly threw: " << ex.what()
                  << std::endl;
    }
}

void TestModelCaching(const std::string& arch, const std::string& solver)
{
    try
    {
        auto model1 = GetCandidateSelectionModel(arch, solver);
        auto model2 = GetCandidateSelectionModel(arch, solver);
        if(model1.get() != model2.get())
        {
            std::cerr << "GetCandidateSelectionModel did not return the same cached object!"
                      << std::endl;
            std::abort();
        }
        std::cout << "GetCandidateSelectionModel caching test passed." << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "GetCandidateSelectionModel caching test failed: " << ex.what() << std::endl;
        std::abort();
    }
}

int main()
{
    std::string arch   = "gfx942";
    std::string solver = "ConvHipImplicitGemm3DGroupWrwXdlops";

    // general setup and metadata tests
    TestFilesExist(arch, solver);
    TestMetadataAndModelInit(arch, solver);

    // specific tests for metadata

    // model caching test
    TestModelCaching(arch, solver);
    // specific tests for model encoding
    TestEncodeInputFeatures(arch, solver);
    TestEncodeKernelConfigs(arch, solver);
    TestEncodeInputFeaturesEdgeCases(arch, solver);
    TestEncodeKernelConfigsEdgeCases(arch, solver);

    // specific tests for candidate selection
    TestSelectBestCandidateValid(arch, solver);
    TestSelectBestCandidateMismatchedDims(arch, solver);
    TestSelectBestCandidateEmptyInput(arch, solver);

    std::cout << "All tests passed.\n";
    return 0;
}
