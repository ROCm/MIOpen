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
// --gtest_filter=Conv3DAIHeuristicsTest.TestMIOpenDriverEquivalent Enable logs:
// MIOPEN_LOG_LEVEL=6 ./bin/test_conv_ai_3d_heuristics

#include <gtest/gtest.h>
#include <miopen/config.h>
#include <miopen/conv/problem_description.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/logger.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

namespace {

using namespace miopen;
using namespace miopen::ai;

// Test suite for 3D convolution AI heuristics (TunaNet3D)
class GPU_Conv3DAIHeuristics_FP32 : public ::testing::Test
{
protected:
    miopen::Handle handle;
    miopen::ExecutionContext ctx;

    GPU_Conv3DAIHeuristics_FP32() : ctx(&handle) {}

    void SetUp() override
    {
        // Check if running on supported GPU arch (e.g., "gfx942")
        std::string device_name = handle.GetDeviceName();
        if(device_name != "gfx942")
        {
            GTEST_SKIP() << "Test requires gfx942 GPU, found: " << device_name;
        }

        // Early skip if model files don't exist (convention: early exit in SetUp)
        if(!ModelFilesExist("gfx942_3d"))
        {
            GTEST_SKIP() << "gfx942_3d model files not found";
        }
    }

    // Helper to create a 3D convolution problem (NCDHW layout)
    miopen::conv::ProblemDescription
    Create3DProblem(int n                             = 1,
                    int c                             = 4,
                    int d                             = 8,
                    int h                             = 8,
                    int w                             = 8,
                    int k                             = 8,
                    int z                             = 3,
                    int y                             = 3,
                    int x                             = 3,
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

// --- Metadata3D tests ---
// for some reason cppcheck raises a syntax error/warning here, but it compiles fine.
// cppcheck-suppress syntaxError
TEST_F(GPU_Conv3DAIHeuristics_FP32, Metadata3D_LoadValidArchitecture)
{
    conv3d::Metadata3D metadata("gfx942_3d");
    ASSERT_TRUE(metadata.IsValid());
    EXPECT_EQ(metadata.GetArchName(), "gfx942_3d");
    EXPECT_GT(metadata.GetNumInputs(), 0);
    EXPECT_GT(metadata.GetNumOutputs(), 0);
    EXPECT_GT(metadata.GetNumSolvers(), 0);
    EXPECT_FALSE(metadata.GetFeatures().empty());
    EXPECT_FALSE(metadata.GetSolverMap().empty());
}

TEST_F(GPU_Conv3DAIHeuristics_FP32, Metadata3D_EncodeDirection)
{
    conv3d::Metadata3D metadata("gfx942_3d");
    ASSERT_TRUE(metadata.IsValid());
    auto fwd_encoded = metadata.EncodeDirection(miopen::conv::Direction::Forward);
    auto bwd_encoded = metadata.EncodeDirection(miopen::conv::Direction::BackwardData);
    auto wrw_encoded = metadata.EncodeDirection(miopen::conv::Direction::BackwardWeights);
    EXPECT_NE(fwd_encoded, bwd_encoded);
    EXPECT_NE(fwd_encoded, wrw_encoded);
    EXPECT_NE(bwd_encoded, wrw_encoded);
}

TEST_F(GPU_Conv3DAIHeuristics_FP32, Metadata3D_EncodePrecision)
{
    conv3d::Metadata3D metadata("gfx942_3d");
    ASSERT_TRUE(metadata.IsValid());
    auto fp32_encoded = metadata.EncodePrecision(miopenFloat);
    auto fp16_encoded = metadata.EncodePrecision(miopenHalf);
    auto bf16_encoded = metadata.EncodePrecision(miopenBFloat16);
    EXPECT_NE(fp32_encoded, fp16_encoded);
    EXPECT_NE(fp32_encoded, bf16_encoded);
    EXPECT_NE(fp16_encoded, bf16_encoded);
}

TEST_F(GPU_Conv3DAIHeuristics_FP32, Metadata3D_EncodeLayouts)
{
    conv3d::Metadata3D metadata("gfx942_3d");
    ASSERT_TRUE(metadata.IsValid());
    auto ncdhw_in  = metadata.EncodeInLayout("NCDHW");
    auto ndhwc_in  = metadata.EncodeInLayout("NDHWC");
    auto ncdhw_out = metadata.EncodeOutLayout("NCDHW");
    auto ndhwc_out = metadata.EncodeOutLayout("NDHWC");
    EXPECT_EQ(ncdhw_in, 0);
    EXPECT_EQ(ndhwc_in, 1);
    EXPECT_EQ(ncdhw_out, 0);
    EXPECT_EQ(ndhwc_out, 1);
    auto invalid = metadata.EncodeInLayout("INVALID_LAYOUT");
    EXPECT_EQ(invalid, 0);
    auto invalid2 = metadata.EncodeInLayout("NHWC");
    EXPECT_EQ(invalid2, 0);
}

// --- Model3D tests ---

TEST_F(GPU_Conv3DAIHeuristics_FP32, Get3DModel_SupportedDevice)
{
    auto model = conv3d::Get3DModel("gfx942");
    ASSERT_NE(model, nullptr);
}

TEST_F(GPU_Conv3DAIHeuristics_FP32, Model3D_IsProblemSupported_3DProblem)
{
    auto model = conv3d::Get3DModel("gfx942");
    ASSERT_NE(model, nullptr);
    auto problem3d = Create3DProblem();
    EXPECT_TRUE(model->IsProblemSupported(problem3d, ctx));
}

TEST_F(GPU_Conv3DAIHeuristics_FP32, Model3D_IsProblemSupported_2DProblem)
{
    auto model = conv3d::Get3DModel("gfx942");
    ASSERT_NE(model, nullptr);
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
    EXPECT_FALSE(model->IsProblemSupported(problem2d, ctx));
}

TEST_F(GPU_Conv3DAIHeuristics_FP32, Model3D_Forward_ReturnsValidPredictions)
{
    auto model = conv3d::Get3DModel("gfx942");
    ASSERT_NE(model, nullptr);
    auto problem = Create3DProblem();
    ASSERT_TRUE(model->IsProblemSupported(problem, ctx));
    auto predictions = model->Forward(problem);
    EXPECT_FALSE(predictions.empty());
    const auto& solver_map = model->GetSolverMap();
    EXPECT_EQ(predictions.size(), solver_map.size());
    auto max_it = std::max_element(predictions.begin(), predictions.end());
    EXPECT_NE(max_it, predictions.end());
    auto min_it = std::min_element(predictions.begin(), predictions.end());
    EXPECT_NE(*max_it, *min_it);
}

TEST_F(GPU_Conv3DAIHeuristics_FP32, PredictSolver_3DProblem_ReturnsSolvers)
{
    auto problem = Create3DProblem();
    auto solvers = immed_mode::PredictSolver(problem, ctx, "gfx942");
    EXPECT_FALSE(solvers.empty());
    for(auto solver_id : solvers)
        EXPECT_GT(solver_id, 0);
}

TEST_F(GPU_Conv3DAIHeuristics_FP32, PredictSolver_3DProblem_UsesCaching)
{
    auto problem  = Create3DProblem();
    auto solvers1 = immed_mode::PredictSolver(problem, ctx, "gfx942");
    auto solvers2 = immed_mode::PredictSolver(problem, ctx, "gfx942");
    EXPECT_EQ(solvers1, solvers2);
}

TEST_F(GPU_Conv3DAIHeuristics_FP32, Model3D_DifferentProblemSizes)
{
    auto model = conv3d::Get3DModel("gfx942");
    ASSERT_NE(model, nullptr);
    std::vector<std::tuple<int, int, int, int, int>> test_cases = {
        {1, 64, 16, 16, 16},
        {4, 128, 32, 32, 32},
        {8, 256, 8, 8, 8},
    };
    for(const auto& [n, c, d, h, w] : test_cases)
    {
        auto problem = Create3DProblem(n, c, d, h, w);
        if(model->IsProblemSupported(problem, ctx))
        {
            auto predictions = model->Forward(problem);
            EXPECT_FALSE(predictions.empty());
        }
    }
}

TEST_F(GPU_Conv3DAIHeuristics_FP32, Metadata3D_OptionalPattern)
{
    conv3d::Metadata3D invalid_metadata("nonexistent");
    EXPECT_FALSE(invalid_metadata.IsValid());
    EXPECT_EQ(invalid_metadata.GetNumInputs(), 0);
    EXPECT_EQ(invalid_metadata.GetNumOutputs(), 0);
    EXPECT_EQ(invalid_metadata.GetNumSolvers(), 0);
    EXPECT_TRUE(invalid_metadata.GetFeatures().empty());
    EXPECT_TRUE(invalid_metadata.GetSolverMap().empty());
    EXPECT_EQ(invalid_metadata.EncodeDirection(miopen::conv::Direction::Forward), 0);
    EXPECT_EQ(invalid_metadata.EncodePrecision(miopenFloat), 0);
    EXPECT_EQ(invalid_metadata.EncodeInLayout("NCDHW"), 0);
}

// --- MIOpenDriver-equivalent test ---

TEST_F(GPU_Conv3DAIHeuristics_FP32, TestMIOpenDriverEquivalent)
{
    auto problem    = Create3DProblem(1,
                                   4,
                                   8,
                                   8,
                                   8,
                                   8,
                                   3,
                                   3,
                                   3,
                                   0,
                                   0,
                                   0,
                                   3,
                                   3,
                                   3,
                                   1,
                                   1,
                                   1,
                                   miopen::conv::Direction::Forward,
                                   miopenFloat);
    auto solver_ids = immed_mode::PredictSolver(problem, ctx, "gfx942");
    EXPECT_FALSE(solver_ids.empty()) << "PredictSolver should return solvers";
}

} // namespace

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
