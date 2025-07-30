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

void TestEncodeInputFeatures(const std::string& arch, const std::string& solver)
{
    try
    {
        CandidateSelectionModel model(arch, solver);

        // Prepare a dummy feature vector of the correct size
        CandidateSelectionMetadata meta(arch, solver);
        std::vector<float> features(meta.input_params.size(), 1.0f);

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
        size_t feature_size = meta.output_params.size();
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

int main()
{
    std::string arch   = "gfx942";
    std::string solver = "ConvHipImplicitGemm3DGroupWrwXdlops";

    TestFilesExist(arch, solver);
    TestMetadataAndModelInit(arch, solver);
    TestEncodeInputFeatures(arch, solver);
    TestEncodeKernelConfigs(arch, solver);

    std::cout << "All tests passed.\n";
    return 0;
}
