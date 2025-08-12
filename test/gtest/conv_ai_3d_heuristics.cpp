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

// Tests for 3D convolution AI heuristics (TunaNet3D)
// Build: make test_conv_ai_3d_heuristics
// Run all: ./bin/test_conv_ai_3d_heuristics
// Run specific:
// ./bin/test_conv_ai_3d_heuristics
// --gtest_filter=Conv3DAIHeuristicsTest.BartTestMIOpenDriverEquivalent Enable logs:
// MIOPEN_LOG_LEVEL=6 ./bin/test_conv_ai_3d_heuristics

#include <gtest/gtest.h>
#include <miopen/config.h>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/logger.hpp>

namespace {

using namespace miopen;
using namespace miopen::ai;

class Conv3DAIHeuristicsTest : public ::testing::Test
{
protected:
    miopen::Handle handle;
    miopen::ExecutionContext ctx;

    Conv3DAIHeuristicsTest() : ctx(&handle) {}

    // Helper to create a 3D convolution problem
    miopen::conv::ProblemDescription
    Create3DProblem(int n                             = 1, // batch size
                    int c                             = 4, // input channels
                    int d                             = 8, // depth
                    int h                             = 8, // height
                    int w                             = 8, // width
                    int k                             = 8, // output channels
                    int z                             = 3, // filter depth
                    int y                             = 3, // filter height
                    int x                             = 3, // filter width
                    int pad_d                         = 0,
                    int pad_h                         = 0,
                    int pad_w                         = 0,
                    int stride_d                      = 1,
                    int stride_h                      = 1,
                    int stride_w                      = 1,
                    int dilation_d                    = 1,
                    int dilation_h                    = 1,
                    int dilation_w                    = 1,
                    miopen::conv::Direction direction = miopen::conv::Direction::Forward,
                    miopenDataType_t dataType         = miopenFloat)
    {
        // Create tensors for 3D convolution (NCDHW layout)
        miopen::TensorDescriptor inputTensor(dataType, {n, c, d, h, w});
        miopen::TensorDescriptor weightsTensor(dataType, {k, c, z, y, x});

        // Calculate output dimensions
        int out_d = (d + 2 * pad_d - dilation_d * (z - 1) - 1) / stride_d + 1;
        int out_h = (h + 2 * pad_h - dilation_h * (y - 1) - 1) / stride_h + 1;
        int out_w = (w + 2 * pad_w - dilation_w * (x - 1) - 1) / stride_w + 1;
        miopen::TensorDescriptor outputTensor(dataType, {n, k, out_d, out_h, out_w});

        // Create convolution descriptor
        miopen::ConvolutionDescriptor convDesc(3, // spatial_dim = 3 for 3D
                                               miopenConvolution,
                                               miopenPaddingDefault,
                                               {pad_d, pad_h, pad_w},
                                               {stride_d, stride_h, stride_w},
                                               {dilation_d, dilation_h, dilation_w},
                                               {0, 0, 0}, // trans_output_pads for 3D
                                               1,         // group_count
                                               1.0f);     // lowp_quant

        return miopen::conv::ProblemDescription(
            direction == miopen::conv::Direction::Forward ? inputTensor : outputTensor,
            weightsTensor,
            direction == miopen::conv::Direction::Forward ? outputTensor : inputTensor,
            convDesc,
            direction);
    }

    // Helper to check if required model files exist
    // Note: For 3D models, the arch parameter should already include "_3d" suffix
    // e.g., "gfx942_3d" will look for "gfx942_3d.tn.model" and "gfx942_3d_metadata.tn.model"
    bool ModelFilesExist(const std::string& arch)
    {
        auto model_path    = GetSystemDbPath() / (arch + ".tn.model");
        auto metadata_path = GetSystemDbPath() / (arch + "_metadata.tn.model");
        return fs::exists(model_path) && fs::exists(metadata_path);
    }
};

// Test Metadata3D class
TEST_F(Conv3DAIHeuristicsTest, Metadata3D_LoadValidArchitecture)
{
    // Skip if model files don't exist
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    conv3d::Metadata3D metadata("gfx942_3d");

    EXPECT_TRUE(metadata.IsValid());
    EXPECT_EQ(metadata.GetArchName(), "gfx942_3d");
    EXPECT_GT(metadata.GetNumInputs(), 0);
    EXPECT_GT(metadata.GetNumOutputs(), 0);
    EXPECT_GT(metadata.GetNumSolvers(), 0);
    EXPECT_FALSE(metadata.GetFeatures().empty());
    EXPECT_FALSE(metadata.GetSolverMap().empty());
}

// TEST_F(Conv3DAIHeuristicsTest, Metadata3D_LoadInvalidArchitecture)
// {
//     conv3d::Metadata3D metadata("nonexistent_arch");

//     EXPECT_FALSE(metadata.IsValid());
// }

TEST_F(Conv3DAIHeuristicsTest, Metadata3D_EncodeDirection)
{
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    conv3d::Metadata3D metadata("gfx942_3d");

    if(metadata.IsValid())
    {
        // Test direction encoding
        auto fwd_encoded = metadata.EncodeDirection(miopen::conv::Direction::Forward);
        auto bwd_encoded = metadata.EncodeDirection(miopen::conv::Direction::BackwardData);
        auto wrw_encoded = metadata.EncodeDirection(miopen::conv::Direction::BackwardWeights);

        // Each direction should have a unique encoding
        // EXPECT_NE(a, b) - Expects that a and b are NOT equal
        EXPECT_NE(fwd_encoded, bwd_encoded);
        EXPECT_NE(fwd_encoded, wrw_encoded);
        EXPECT_NE(bwd_encoded, wrw_encoded);
    }
}

TEST_F(Conv3DAIHeuristicsTest, Metadata3D_EncodePrecision)
{
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    conv3d::Metadata3D metadata("gfx942_3d");

    if(metadata.IsValid())
    {
        // Test precision encoding
        auto fp32_encoded = metadata.EncodePrecision(miopenFloat);
        auto fp16_encoded = metadata.EncodePrecision(miopenHalf);
        auto bf16_encoded = metadata.EncodePrecision(miopenBFloat16);

        // Each precision should have a unique encoding
        EXPECT_NE(fp32_encoded, fp16_encoded);
        EXPECT_NE(fp32_encoded, bf16_encoded);
        EXPECT_NE(fp16_encoded, bf16_encoded);
    }
}

TEST_F(Conv3DAIHeuristicsTest, Metadata3D_EncodeLayouts)
{
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    conv3d::Metadata3D metadata("gfx942_3d");

    if(metadata.IsValid())
    {
        // Test layout encoding for 3D layouts
        auto ncdhw_in  = metadata.EncodeInLayout("NCDHW");
        auto ndhwc_in  = metadata.EncodeInLayout("NDHWC");
        auto ncdhw_out = metadata.EncodeOutLayout("NCDHW");
        auto ndhwc_out = metadata.EncodeOutLayout("NDHWC");

        // NCDHW should encode to 0, NDHWC to 1 (based on metadata file)
        EXPECT_EQ(ncdhw_in, 0);
        EXPECT_EQ(ndhwc_in, 1);
        EXPECT_EQ(ncdhw_out, 0);
        EXPECT_EQ(ndhwc_out, 1);

        // Invalid layout should return 0 (but need different check since 0 is valid)
        // For invalid layouts, we check that it returns 0 AND is not in the valid set
        auto invalid = metadata.EncodeInLayout("INVALID_LAYOUT");
        EXPECT_EQ(invalid, 0);
        // To distinguish from valid NCDHW=0, we could check multiple invalid ones
        auto invalid2 = metadata.EncodeInLayout("NHWC"); // 2D layout, not 3D
        EXPECT_EQ(invalid2, 0);
    }
}

// Test Get3DModel factory function
TEST_F(Conv3DAIHeuristicsTest, Get3DModel_SupportedDevice)
{
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    auto model = conv3d::Get3DModel("gfx942");
    EXPECT_NE(model, nullptr);
}

// Test Model3D problem support
TEST_F(Conv3DAIHeuristicsTest, Model3D_IsProblemSupported_3DProblem)
{
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    auto model = conv3d::Get3DModel("gfx942");
    if(!model)
    {
        GTEST_SKIP() << "Failed to create 3D model";
    }

    // Create a 3D problem
    auto problem3d = Create3DProblem();
    EXPECT_TRUE(model->IsProblemSupported(problem3d, ctx));
}

TEST_F(Conv3DAIHeuristicsTest, Model3D_IsProblemSupported_2DProblem)
{
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    auto model = conv3d::Get3DModel("gfx942");
    if(!model)
    {
        GTEST_SKIP() << "Failed to create 3D model";
    }

    // Create a 2D problem (depth = 1)
    miopen::TensorDescriptor inputTensor2D(miopenFloat, {1, 4, 8, 8});
    miopen::TensorDescriptor weightsTensor2D(miopenFloat, {8, 4, 3, 3});
    miopen::TensorDescriptor outputTensor2D(miopenFloat, {1, 8, 6, 6});
    miopen::ConvolutionDescriptor convDesc2D(
        2, miopenConvolution, miopenPaddingDefault, {0, 0}, {1, 1}, {1, 1});

    miopen::conv::ProblemDescription problem2d(inputTensor2D,
                                               weightsTensor2D,
                                               outputTensor2D,
                                               convDesc2D,
                                               miopen::conv::Direction::Forward);

    // 3D model should not support 2D problems
    EXPECT_FALSE(model->IsProblemSupported(problem2d, ctx));
}

// Test Model3D forward inference
TEST_F(Conv3DAIHeuristicsTest, Model3D_Forward_ReturnsValidPredictions)
{
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    auto model = conv3d::Get3DModel("gfx942");
    if(!model)
    {
        GTEST_SKIP() << "Failed to create 3D model";
    }

    auto problem = Create3DProblem();

    if(model->IsProblemSupported(problem, ctx))
    {
        try
        {
            auto predictions = model->Forward(problem);

            // Should return predictions
            EXPECT_FALSE(predictions.empty());

            // Number of predictions should match number of solvers
            const auto& solver_map = model->GetSolverMap();
            EXPECT_EQ(predictions.size(), solver_map.size());

            // The model outputs raw logits, not probabilities
            // Logits can be any real number (positive or negative)
            // We just verify we got the expected number of outputs
            // The actual solver selection uses relative ranking, not absolute values

            // Find the solver with highest score (best prediction)
            auto max_it = std::max_element(predictions.begin(), predictions.end());
            EXPECT_NE(max_it, predictions.end());

            // Verify that not all predictions are the same
            // (model should differentiate between solvers)
            auto min_it = std::min_element(predictions.begin(), predictions.end());
            EXPECT_NE(*max_it, *min_it);

            // Log solver predictions for debugging (only visible with MIOPEN_LOG_LEVEL=6)
            size_t best_idx         = std::distance(predictions.begin(), max_it);
            std::string best_solver = "unknown";
            size_t idx              = 0;
            for(const auto& [solver_id, solver_name] : solver_map)
            {
                MIOPEN_LOG_I2("Solver[" << idx << "]: " << solver_name << " = " << predictions[idx]
                                        << (idx == best_idx ? " (BEST)" : ""));
                if(idx == best_idx)
                {
                    best_solver = solver_name;
                }
                idx++;
            }
            MIOPEN_LOG_I2("Best predicted solver: " << best_solver << " with score " << *max_it);
        }
        catch(const std::exception& e)
        {
            FAIL() << "Forward() threw exception: " << e.what();
        }
    }
}

// Test PredictSolver with 3D problem
TEST_F(Conv3DAIHeuristicsTest, PredictSolver_3DProblem_ReturnsSolvers)
{
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    auto problem = Create3DProblem();

    // Test with gfx942 (has 3D support)
    auto solvers = immed_mode::PredictSolver(problem, ctx, "gfx942");

    // Should return at least one solver
    EXPECT_FALSE(solvers.empty());

    // Solvers should be valid IDs
    for(auto solver_id : solvers)
    {
        EXPECT_GT(solver_id, 0);
    }
}

// Test PredictSolver caching
TEST_F(Conv3DAIHeuristicsTest, PredictSolver_3DProblem_UsesCaching)
{
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    auto problem = Create3DProblem();

    // First call - should compute
    auto solvers1 = immed_mode::PredictSolver(problem, ctx, "gfx942");

    // Second call with same problem - should use cache
    auto solvers2 = immed_mode::PredictSolver(problem, ctx, "gfx942");

    // Results should be identical
    EXPECT_EQ(solvers1, solvers2);
}

// Test different 3D problem configurations
TEST_F(Conv3DAIHeuristicsTest, Model3D_DifferentProblemSizes)
{
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    auto model = conv3d::Get3DModel("gfx942");
    if(!model)
    {
        GTEST_SKIP() << "Failed to create 3D model";
    }

    // Test various 3D problem sizes
    std::vector<std::tuple<int, int, int, int, int>> test_cases = {
        {1, 64, 16, 16, 16},  // Small
        {4, 128, 32, 32, 32}, // Medium
        {8, 256, 8, 8, 8},    // Different aspect
    };

    for(const auto& [n, c, d, h, w] : test_cases)
    {
        auto problem = Create3DProblem(n, c, d, h, w);

        if(model->IsProblemSupported(problem, ctx))
        {
            try
            {
                auto predictions = model->Forward(problem);
                EXPECT_FALSE(predictions.empty());
            }
            catch(const std::exception& e)
            {
                FAIL() << "Forward() failed for size (" << n << "," << c << "," << d << "," << h
                       << "," << w << "): " << e.what();
            }
        }
    }
}

// Test std::optional pattern in Metadata3D
TEST_F(Conv3DAIHeuristicsTest, Metadata3D_OptionalPattern)
{
    // Test with invalid architecture - should handle gracefully
    conv3d::Metadata3D invalid_metadata("nonexistent");

    // All methods should return safe defaults when invalid
    EXPECT_FALSE(invalid_metadata.IsValid());
    EXPECT_EQ(invalid_metadata.GetNumInputs(), 0);
    EXPECT_EQ(invalid_metadata.GetNumOutputs(), 0);
    EXPECT_EQ(invalid_metadata.GetNumSolvers(), 0);
    EXPECT_TRUE(invalid_metadata.GetFeatures().empty());
    EXPECT_TRUE(invalid_metadata.GetSolverMap().empty());

    // Encoding methods should return 0 for invalid metadata
    EXPECT_EQ(invalid_metadata.EncodeDirection(miopen::conv::Direction::Forward), 0);
    EXPECT_EQ(invalid_metadata.EncodePrecision(miopenFloat), 0);
    EXPECT_EQ(invalid_metadata.EncodeInLayout("NCDHW"), 0);
}

// Test matching MIOpenDriver command:
// ./bin/MIOpenDriver conv -F 1 -n 1 -c 4 -k 8 -H 8 -W 8 -! 8 -y 3 -x 3 -@ 3 -u 3 -v 3 -l 3 -j 3 -m
// 3 -g 1 -t fp32 -V 0
TEST_F(Conv3DAIHeuristicsTest, BartTestMIOpenDriverEquivalent)
{
    if(!ModelFilesExist("gfx942_3d"))
    {
        GTEST_SKIP() << "gfx942_3d model files not found";
    }

    // Create 3D problem matching MIOpenDriver parameters:
    // -n 1 (batch), -c 4 (input channels), -k 8 (output channels)
    // -H 8 -W 8 -! 8 (input height, width, depth)
    // -y 3 -x 3 -@ 3 (filter height, width, depth)
    // -u 3 -v 3 -l 3 (stride h, w, d)
    // -j 3 -m 3 -g 1 (pad h, w, d) - assuming padding is 0 since not specified
    // -t fp32 (data type)
    // -F 1 (forward direction)
    auto problem = Create3DProblem(1, // n - batch size
                                   4, // c - input channels
                                   8, // d - input depth
                                   8, // h - input height
                                   8, // w - input width
                                   8, // k - output channels
                                   3, // z - filter depth
                                   3, // y - filter height
                                   3, // x - filter width
                                   0, // pad_d - assuming 0
                                   0, // pad_h - assuming 0
                                   0, // pad_w - assuming 0
                                   3, // stride_d
                                   3, // stride_h
                                   3, // stride_w
                                   1, // dilation_d
                                   1, // dilation_h
                                   1, // dilation_w
                                   miopen::conv::Direction::Forward,
                                   miopenFloat // fp32
    );

    // Test PredictSolver function (as used by MIOpenDriver)
    // This is the actual function that MIOpenDriver calls
    MIOPEN_LOG_I2("=== Bart Test: 3D Convolution Solver Predictions ===");
    MIOPEN_LOG_I2("Problem: n=1 c=4 k=8 DHW=8x8x8 filter=3x3x3 stride=3x3x3 pad=0x0x0");
    auto solver_ids = immed_mode::PredictSolver(problem, ctx, "gfx942");
    EXPECT_FALSE(solver_ids.empty()) << "PredictSolver should return solvers";

    MIOPEN_LOG_I2("PredictSolver returned " << solver_ids.size() << " solver IDs");
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
