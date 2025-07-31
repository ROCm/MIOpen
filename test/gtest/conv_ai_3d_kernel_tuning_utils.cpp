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
#include <miopen/conv/heuristics/ai_conv_3d_kernel_tuning_utils.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/tensor.hpp>
#include <miopen/convolution.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <miopen/conv/solvers.hpp>

using namespace miopen::solver::conv;

class Conv3DKernelTuningUtilsTest : public ::testing::Test
{
protected:
    miopen::conv::ProblemDescription GetReusableProblemDescription(
        miopenDataType_t dataType         = miopenFloat,
        miopen::conv::Direction direction = miopen::conv::Direction::BackwardWeights)
    {
        std::vector<int> in_lengths      = {2, 3, 8, 8, 8};
        std::vector<int> weights_lengths = {4, 3, 3, 3, 3};
        std::vector<int> out_lengths     = {2, 4, 6, 6, 6};

        miopen::TensorDescriptor in_desc(dataType, in_lengths);
        miopen::TensorDescriptor weights_desc(dataType, weights_lengths);
        miopen::TensorDescriptor out_desc(dataType, out_lengths);

        std::vector<int> pads              = {0, 0, 0};
        std::vector<int> strides           = {1, 1, 1};
        std::vector<int> dilations         = {1, 1, 1};
        std::vector<int> trans_output_pads = {0, 0, 0};

        miopen::ConvolutionDescriptor conv_desc(3,
                                                miopenConvolution,
                                                miopenPaddingDefault,
                                                pads,
                                                strides,
                                                dilations,
                                                trans_output_pads);

        return miopen::conv::ProblemDescription(
            in_desc, weights_desc, out_desc, conv_desc, direction);
    }
};

TEST_F(Conv3DKernelTuningUtilsTest, GetFeatures3D_Size)
{
    auto problem     = GetReusableProblemDescription();
    int max_cu       = 304;
    std::string arch = "gfx942";
    auto features    = miopen::solver::conv::GetFeatures3D(problem, max_cu, arch);
    ASSERT_EQ(features.size(), 22u) << "Unexpected feature vector size";
}

void CheckGetFeatures3D_Values(const std::vector<float>& features,
                               miopen::conv::Direction direction)
{
    int expected_in_c = 3, expected_in_d = 8, expected_in_h = 8, expected_in_w = 8;
    int expected_out_k = 4, expected_out_d = 6, expected_out_h = 6, expected_out_w = 6;
    int expected_batch_n       = 2;
    int expected_in_left_pad_d = 0, expected_in_left_pad_h = 0, expected_in_left_pad_w = 0;
    int expected_stride_d = 1, expected_stride_h = 1, expected_stride_w = 1;
    int expected_fil_d = 3, expected_fil_h = 3, expected_fil_w = 3;

    bool is_forward = (direction == miopen::conv::Direction::Forward);

    int in_c    = is_forward ? expected_in_c : expected_out_k;
    int in_d    = is_forward ? expected_in_d : expected_out_d;
    int in_h    = is_forward ? expected_in_h : expected_out_h;
    int in_w    = is_forward ? expected_in_w : expected_out_w;
    int out_k   = is_forward ? expected_out_k : expected_in_c;
    int out_d   = is_forward ? expected_out_d : expected_in_d;
    int out_h   = is_forward ? expected_out_h : expected_in_h;
    int out_w   = is_forward ? expected_out_w : expected_in_w;
    int batch_n = expected_batch_n;

    ASSERT_EQ(features[0], in_c);
    ASSERT_EQ(features[1], in_d);
    ASSERT_EQ(features[2], in_h);
    ASSERT_EQ(features[3], in_w);
    ASSERT_EQ(features[4], out_k);
    ASSERT_EQ(features[5], out_d);
    ASSERT_EQ(features[6], out_h);
    ASSERT_EQ(features[7], out_w);
    ASSERT_EQ(features[8], expected_fil_d);
    ASSERT_EQ(features[9], expected_fil_h);
    ASSERT_EQ(features[10], expected_fil_w);
    ASSERT_EQ(features[11], expected_in_left_pad_d);
    ASSERT_EQ(features[12], expected_in_left_pad_h);
    ASSERT_EQ(features[13], expected_in_left_pad_w);
    ASSERT_EQ(features[14], expected_stride_d);
    ASSERT_EQ(features[15], expected_stride_h);
    ASSERT_EQ(features[16], expected_stride_w);
    ASSERT_EQ(features[17], batch_n);
    ASSERT_EQ(features[18], 0.0f);                            // InputLayout
    ASSERT_EQ(features[19], 0.0f);                            // FilterLayout
    ASSERT_EQ(features[20], 0.0f);                            // OutputLayout
    ASSERT_EQ(features[21], static_cast<float>(miopenFloat)); // DataType
}

TEST_F(Conv3DKernelTuningUtilsTest, GetFeatures3D_ValueChecks)
{
    int max_cu                                            = 304;
    std::string arch                                      = "gfx942";
    const std::vector<miopen::conv::Direction> directions = {
        miopen::conv::Direction::Forward,
        miopen::conv::Direction::BackwardData,
        miopen::conv::Direction::BackwardWeights};
    for(const auto direction : directions)
    {
        auto problem  = GetReusableProblemDescription(miopenFloat, direction);
        auto features = miopen::solver::conv::GetFeatures3D(problem, max_cu, arch);
        ASSERT_EQ(features.size(), 22u);
        CheckGetFeatures3D_Values(features, direction);
    }
}

TEST_F(Conv3DKernelTuningUtilsTest, GetFeatures3D_Directions)
{
    int max_cu       = 304;
    std::string arch = "gfx942";
    auto problem_fwd = GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::Forward);
    auto features_fwd = miopen::solver::conv::GetFeatures3D(problem_fwd, max_cu, arch);

    auto problem_bwd =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardData);
    auto features_bwd = miopen::solver::conv::GetFeatures3D(problem_bwd, max_cu, arch);

    auto problem_wrw =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);
    auto features_wrw = miopen::solver::conv::GetFeatures3D(problem_wrw, max_cu, arch);

    ASSERT_EQ(features_fwd.size(), features_bwd.size());
    ASSERT_EQ(features_fwd.size(), features_wrw.size());
}

TEST_F(Conv3DKernelTuningUtilsTest, GetFeatures3D_DataTypes)
{
    int max_cu       = 304;
    std::string arch = "gfx942";

    auto problem_f  = GetReusableProblemDescription(miopenFloat);
    auto features_f = miopen::solver::conv::GetFeatures3D(problem_f, max_cu, arch);
    ASSERT_EQ(features_f[21], static_cast<float>(miopenFloat));

    auto problem_h  = GetReusableProblemDescription(miopenHalf);
    auto features_h = miopen::solver::conv::GetFeatures3D(problem_h, max_cu, arch);
    ASSERT_EQ(features_h[21], static_cast<float>(miopenHalf));

    auto problem_b  = GetReusableProblemDescription(miopenBFloat16);
    auto features_b = miopen::solver::conv::GetFeatures3D(problem_b, max_cu, arch);
    ASSERT_EQ(features_b[21], static_cast<float>(miopenBFloat16));
}

TEST_F(Conv3DKernelTuningUtilsTest, TokenizeKernel)
{
    auto tokens = miopen::solver::conv::TokenizeKernel("type_param1_param2");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "type");
    EXPECT_EQ(tokens[1], "param1");
    EXPECT_EQ(tokens[2], "param2");

    auto empty = miopen::solver::conv::TokenizeKernel("");
    ASSERT_TRUE(empty.empty());
}

TEST_F(Conv3DKernelTuningUtilsTest, FilterHeuristicKernels)
{
    std::vector<std::string> kernels = {"typeA_param1", "typeB_param2", "typeA_param3"};
    std::vector<int> indexes;
    std::vector<std::vector<std::string>> tokens;
    miopen::solver::conv::FilterHeuristicKernels("typeA", kernels, indexes, tokens);

    ASSERT_EQ(indexes.size(), 2u);
    ASSERT_EQ(tokens.size(), 2u);
    ASSERT_EQ(indexes[0], 0);
    ASSERT_EQ(indexes[1], 2);
}

TEST_F(Conv3DKernelTuningUtilsTest, GenerateSplitK)
{
    auto split_ks             = miopen::solver::conv::GenerateSplitK(8);
    std::vector<int> expected = {1, 2, 4, 8};
    ASSERT_EQ(split_ks, expected);
}

TEST_F(Conv3DKernelTuningUtilsTest, ExpandKernelParamsWithSplitK)
{
    std::vector<std::vector<std::string>> kernels = {{"typeA", "p1"}, {"typeB", "p2"}};
    std::vector<int> indexes                      = {0, 1};
    std::vector<int> split_ks                     = miopen::solver::conv::GenerateSplitK(8);
    auto [expanded, mapping] =
        miopen::solver::conv::ExpandKernelParamsWithSplitK(kernels, indexes, split_ks);

    ASSERT_EQ(expanded.size(), 8u);
    ASSERT_EQ(mapping.size(), 8u);

    std::vector<std::vector<std::string>> expected_expanded = {
        {"typeA", "p1", "1"},
        {"typeA", "p1", "2"},
        {"typeA", "p1", "4"},
        {"typeA", "p1", "8"},
        {"typeB", "p2", "1"},
        {"typeB", "p2", "2"},
        {"typeB", "p2", "4"},
        {"typeB", "p2", "8"},
    };
    std::vector<std::pair<int, int>> expected_mapping = {
        {0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 1}, {1, 2}, {1, 4}, {1, 8}};

    for(size_t i = 0; i < expanded.size(); ++i)
    {
        ASSERT_EQ(expanded[i], expected_expanded[i]);
        ASSERT_EQ(mapping[i], expected_mapping[i]);
    }
}

TEST_F(Conv3DKernelTuningUtilsTest, RunParameterPredictionModel)
{
    miopen::Handle handle;
    miopen::ExecutionContext ctx(&handle);

    std::string device_name = handle.GetDeviceName();
    int max_cu              = handle.GetMaxComputeUnits();
    std::cout << "Device name: " << device_name << std::endl;
    std::cout << "Max compute units: " << max_cu << std::endl;

    auto problem =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);

    std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)>
        fill_valid_kernels = [&ctx](const miopen::conv::ProblemDescription& problem) {
            miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops solver;
            if(!solver.IsApplicable(ctx, problem))
                return std::vector<std::string>{};
            auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
            auto solution = solver.GetSolution(ctx, problem, perf_cfg);
            std::vector<std::string> kernel_names;
            for(const auto& cp : solution.construction_params)
                kernel_names.push_back(cp.kernel_name);
            return kernel_names;
        };

    std::vector<std::string> valid_kernels;
    int index = 0, split_k = 1;
    std::string kernel_id;
    std::string solver_name = "ConvHipImplicitGemm3DGroupWrwXdlops";

    bool result = miopen::solver::conv::RunParameterPredictionModel<float>(
        ctx, problem, valid_kernels, index, split_k, kernel_id, fill_valid_kernels, solver_name);

    ASSERT_TRUE(result) << "Model did not return a valid result.";
    ASSERT_GE(index, 0);
    ASSERT_GE(split_k, 1);
    ASSERT_FALSE(kernel_id.empty());
    std::cout << "RunParameterPredictionModel: index=" << index << ", split_k=" << split_k
              << ", kernel_id=" << kernel_id << std::endl;
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
