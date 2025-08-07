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
#include <miopen/filesystem.hpp>

using namespace miopen::solver::conv;
// Helper function for layout string to code (must match GetFeatures3D)
int LayoutStringToCode(const std::string& layout)
{
    if(layout == "NCDHW")
        return 0.0;
    if(layout == "NDHWC")
        return 1.0;
    return -1.0; // Unknown
}

// dummy kernels for testing
const std::vector<std::string> dummy_kernels = {
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64,64,64,4,Default,4,2,2,1,4,1,4,1,1,1>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<128,128,32,4,Default,4,2,1,4,4,1,1,1,1,1>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64,32,64,4,Default,4,1,2,1,2,4,4,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256,256,128,4,Default,4,4,2,4,4,4,2,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256,128,256,4,Default,4,2,4,4,2,4,4,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<128,128,128,4,Default,4,4,2,4,4,4,4,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256,128,128,4,Default,4,2,2,4,2,4,2,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<128,128,64,4,Default,4,2,2,4,4,4,2,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<128,64,128,4,Default,4,2,2,4,2,4,4,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64,64,64,4,Default,4,2,2,4,4,4,4,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256,128,64,4,Default,4,2,1,4,2,4,1,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256,64,128,4,Default,4,1,2,4,1,4,2,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<128,128,32,4,Default,4,2,1,4,4,4,1,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<128,32,128,4,Default,4,1,2,4,1,4,4,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64,64,32,4,Default,4,2,1,4,4,4,2,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64,32,64,4,Default,4,1,2,4,2,4,4,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256,64,64,8,Default,8,1,1,4,4,4,4,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256,64,64,8,Default,8,1,1,4,4,1,4,1,1,1>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256,64,64,8,Default,8,1,1,1,4,4,4,1,1,4>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffle<256,64,64,8,Default,8,1,1,1,4,1,4,1,1,1>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffleV3<64,16,16,32,Default,8,1,1,1,4,1,4,1,1,2>",
    "DeviceGroupedConvBwdWeight_Xdl_CShuffleV3<64,16,16,32,Default,8,1,1,1,4,1,4,1,1,2>",
};

// test version of fill_valid_kernels with a dummy function
static std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)>
    fill_valid_kernels = [](const miopen::conv::ProblemDescription&) { return dummy_kernels; };

class Conv3DKernelTuningUtilsTest : public ::testing::Test
{
protected:
    miopen::Handle handle;
    miopen::ExecutionContext ctx;

    void SetUp() override { ctx = miopen::ExecutionContext(&handle); }

    miopen::conv::ProblemDescription GetReusableProblemDescription(
        miopenDataType_t dataType         = miopenFloat,
        miopen::conv::Direction direction = miopen::conv::Direction::BackwardWeights)
    {
        std::vector<int> in_lengths      = {1, 512, 11, 130, 66};
        std::vector<int> weights_lengths = {256, 512, 3, 3, 3};
        std::vector<int> out_lengths     = {1, 256, 9, 128, 64};

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
    auto features    = GetFeatures3D(problem, max_cu, arch);
    ASSERT_EQ(features.size(), 29u) << "Unexpected feature vector size";
}

TEST_F(Conv3DKernelTuningUtilsTest, GetFeatures3D_Directions)
{
    int max_cu       = 304;
    std::string arch = "gfx942";
    auto problem_fwd = GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::Forward);
    auto features_fwd = GetFeatures3D(problem_fwd, max_cu, arch);

    auto problem_bwd =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardData);
    auto features_bwd = GetFeatures3D(problem_bwd, max_cu, arch);

    auto problem_wrw =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);
    auto features_wrw = GetFeatures3D(problem_wrw, max_cu, arch);

    ASSERT_EQ(features_fwd.size(), features_bwd.size());
    ASSERT_EQ(features_fwd.size(), features_wrw.size());
}

TEST_F(Conv3DKernelTuningUtilsTest, GetKernelAsTokens)
{
    auto tokens = GetKernelAsTokens("type<param1,param2>");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "type");
    EXPECT_EQ(tokens[1], "param1");
    EXPECT_EQ(tokens[2], "param2");

    auto empty = GetKernelAsTokens("");
    ASSERT_TRUE(empty.empty());
}

TEST_F(Conv3DKernelTuningUtilsTest, GenerateSplitK)
{
    auto split_ks             = GenerateSplitK(8);
    std::vector<int> expected = {1, 2, 4, 8};
    ASSERT_EQ(split_ks, expected);
}

TEST_F(Conv3DKernelTuningUtilsTest, ExpandKernelParamsWithSplitK)
{
    std::vector<std::vector<std::string>> kernels = {{"typeA", "p1"}, {"typeB", "p2"}};
    std::vector<int> indexes                      = {0, 1};
    std::vector<int> split_ks                     = GenerateSplitK(8);
    auto [expanded, mapping] = ExpandKernelParamsWithSplitK(kernels, indexes, split_ks);

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

TEST_F(Conv3DKernelTuningUtilsTest, CandidateSelectionFilesExist)
{
    std::string db_path     = miopen::GetSystemDbPath();
    std::string solver_name = "ConvHipImplicitGemm3DGroupWrwXdlops";
    std::string arch        = "gfx942";

    auto metadata      = db_path + "/" + arch + "_" + solver_name + "_metadata.tn.model";
    auto input_encoder = db_path + "/" + arch + "_" + solver_name + "_input_encoder.tn.model";
    auto kernel_config_encoder =
        db_path + "/" + arch + "_" + solver_name + "_kernel_config_encoder.tn.model";

    ASSERT_TRUE(miopen::fs::exists(metadata)) << "Missing metadata file: " << metadata;
    ASSERT_TRUE(miopen::fs::exists(input_encoder))
        << "Missing input encoder file: " << input_encoder;
    ASSERT_TRUE(miopen::fs::exists(kernel_config_encoder))
        << "Missing kernel config encoder file: " << kernel_config_encoder;
}

TEST_F(Conv3DKernelTuningUtilsTest, CandidateSelectionModelInitialization)
{
    std::string arch        = "gfx942";
    std::string solver_name = "ConvHipImplicitGemm3DGroupWrwXdlops";
    EXPECT_NO_THROW({
        miopen::ai::tuning::candidate_selection::CandidateSelectionModel model(arch, solver_name);
    });

    try
    {
        auto& model =
            miopen::ai::tuning::candidate_selection::GetCandidateSelectionModel(arch, solver_name);
        const auto& meta = model.metadata();
        ASSERT_FALSE(meta.input_params().empty());
        ASSERT_FALSE(meta.output_params().empty());
    }
    catch(const std::exception& ex)
    {
        FAIL() << "Exception during model construction: " << ex.what();
    }
}

TEST_F(Conv3DKernelTuningUtilsTest, ExpandKernelParamsWithSplitKFunctionality)
{
    std::vector<std::vector<std::string>> kernels = {
        {"DeviceGroupedConvBwdWeight_Xdl_CShuffle", "p1"}};
    std::vector<int> indexes  = {0};
    std::vector<int> split_ks = GenerateSplitK(8);
    auto [expanded, mapping]  = ExpandKernelParamsWithSplitK(kernels, indexes, split_ks);

    ASSERT_EQ(expanded.size(), split_ks.size());
    ASSERT_EQ(mapping.size(), split_ks.size());
    for(size_t i = 0; i < split_ks.size(); ++i)
    {
        ASSERT_EQ(expanded[i][0], "DeviceGroupedConvBwdWeight_Xdl_CShuffle");
        ASSERT_EQ(expanded[i][2], std::to_string(split_ks[i]));
        ASSERT_EQ(mapping[i].first, 0);
        ASSERT_EQ(mapping[i].second, split_ks[i]);
    }
}

TEST_F(Conv3DKernelTuningUtilsTest, RunParameterPredictionModelReturnsValidResult)
{
    std::string arch = handle.GetDeviceName();
    auto problem =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);

    int index = 0, split_k = 1;
    std::string kernel_id;
    std::string solver_name = "ConvHipImplicitGemm3DGroupWrwXdlops";
    std::vector<std::string> valid_kernels;

    bool result = miopen::solver::conv::RunParameterPredictionModel<float>(
        ctx, problem, valid_kernels, index, split_k, kernel_id, fill_valid_kernels, solver_name);

    ASSERT_TRUE(result);
    ASSERT_FALSE(kernel_id.empty()); // Optionally check kernel_id was set
    // for debugging, print out selected kernel_id
    std::cout << "Selected kernel_id: " << kernel_id << std::endl;
}

TEST_F(Conv3DKernelTuningUtilsTest, RunParameterPredictionModel_Fallback)
{
    // Use a fill_valid_kernels that returns an empty list
    std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)> empty_kernels =
        [](const miopen::conv::ProblemDescription&) { return std::vector<std::string>{}; };

    auto problem =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);
    int index = 0, split_k = 1;
    std::string kernel_id;
    std::string solver_name = "ConvHipImplicitGemm3DGroupWrwXdlops";
    std::vector<std::string> valid_kernels;

    bool result = miopen::solver::conv::RunParameterPredictionModel<float>(
        ctx, problem, valid_kernels, index, split_k, kernel_id, empty_kernels, solver_name);

    ASSERT_FALSE(result);
    ASSERT_TRUE(kernel_id.empty());
}

void CheckGetFeatures3D_MapValues(const std::map<std::string, float>& features,
                                  const miopen::conv::ProblemDescription& problem,
                                  miopen::conv::Direction direction)
{
    std::map<std::string, float> expected;
    expected["spatial_dim"] = 3.0f;
    expected["in_channels"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputChannelC(problem));
    expected["in_d"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputDepthDi(problem));
    expected["in_h"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputHeightHi(problem));
    expected["in_w"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputWidthWi(problem));
    expected["out_channels"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetOutputChannelK(problem));
    expected["out_d"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetOutputDepthDo(problem));
    expected["out_h"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetOutputHeightHo(problem));
    expected["out_w"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetOutputWidthWo(problem));
    expected["fil_d"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetFilterDepthZ(problem));
    expected["fil_h"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetFilterHeightY(problem));
    expected["fil_w"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetFilterWidthX(problem));
    expected["pad_d"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputLeftPadD(problem));
    expected["pad_h"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputLeftPadH(problem));
    expected["pad_w"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetInputLeftPadW(problem));
    expected["conv_stride_d"] = static_cast<float>(
        miopen::solver::ProblemInterpreter::GetAdjustedConvolutionStrideD(problem));
    expected["conv_stride_h"] = static_cast<float>(
        miopen::solver::ProblemInterpreter::GetAdjustedConvolutionStrideH(problem));
    expected["conv_stride_w"] = static_cast<float>(
        miopen::solver::ProblemInterpreter::GetAdjustedConvolutionStrideW(problem));
    expected["dilation_d"] = static_cast<float>(problem.GetDilationD());
    expected["dilation_h"] = static_cast<float>(problem.GetDilationH());
    expected["dilation_w"] = static_cast<float>(problem.GetDilationW());
    expected["batchsize"] =
        static_cast<float>(miopen::solver::ProblemInterpreter::GetBatchN(problem));
    expected["bias"]      = static_cast<float>(problem.GetBias());
    expected["in_layout"] = static_cast<float>(
        LayoutStringToCode(miopen::solver::ProblemInterpreter::GetInputLayout(problem)));
    expected["fil_layout"] = static_cast<float>(
        LayoutStringToCode(miopen::solver::ProblemInterpreter::GetFilterLayout(problem)));
    expected["out_layout"] = static_cast<float>(
        LayoutStringToCode(miopen::solver::ProblemInterpreter::GetOutputLayout(problem)));
    expected["precision"] = static_cast<float>(problem.GetInDataType());
    expected["direction"] =
        static_cast<float>(direction == miopen::conv::Direction::Forward           ? 0.0f
                           : direction == miopen::conv::Direction::BackwardData    ? 1.0f
                           : direction == miopen::conv::Direction::BackwardWeights ? 2.0f
                                                                                   : -1.0f);
    expected["group_count"] = static_cast<float>(problem.GetGroupCount());

    for(const auto& kv : expected)
    {
        ASSERT_TRUE(features.count(kv.first)) << "Missing key: " << kv.first;
        EXPECT_FLOAT_EQ(features.at(kv.first), kv.second) << "Mismatch for key: " << kv.first;
    }
}

TEST_F(Conv3DKernelTuningUtilsTest, GetFeatures3D_MapValueChecks)
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
        auto features = GetFeatures3D(problem, max_cu, arch);
        ASSERT_EQ(features.size(), 29u);
        CheckGetFeatures3D_MapValues(features, problem, direction);
    }
}

TEST_F(Conv3DKernelTuningUtilsTest, GetFeatures3D_DataTypes)
{
    int max_cu       = 304;
    std::string arch = "gfx942";

    auto problem_f  = GetReusableProblemDescription(miopenFloat);
    auto features_f = GetFeatures3D(problem_f, max_cu, arch);
    ASSERT_EQ(features_f.at("precision"), static_cast<float>(miopenFloat));

    auto problem_h  = GetReusableProblemDescription(miopenHalf);
    auto features_h = GetFeatures3D(problem_h, max_cu, arch);
    ASSERT_EQ(features_h.at("precision"), static_cast<float>(miopenHalf));

    auto problem_b  = GetReusableProblemDescription(miopenBFloat16);
    auto features_b = GetFeatures3D(problem_b, max_cu, arch);
    ASSERT_EQ(features_b.at("precision"), static_cast<float>(miopenBFloat16));
}

TEST_F(Conv3DKernelTuningUtilsTest, FullSolverPathway_ConvHipImplicitGemm3DGroupWrwXdlops)
{
    // Set up the problem and context
    auto problem =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);

    // Set device name to gfx942 for the handle/context
    ctx = miopen::ExecutionContext(&handle);

    // Instantiate the solver
    ConvHipImplicitGemm3DGroupWrwXdlops solver;

    // Check applicability
    ASSERT_TRUE(solver.IsApplicable(ctx, problem)) << "Solver not applicable for this problem";

    // Get default performance config
    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg))
        << "Invalid performance config";

    // Get solution
    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    // Check solution validity
    ASSERT_FALSE(solution.construction_params.empty()) << "Solution construction_params is empty";
    ASSERT_TRUE(solution.invoker_factory) << "Solution invoker_factory is not set";
    ASSERT_GE(solution.workspace_sz, 0u) << "Workspace size should be non-negative";

    std::cout << "Selected CK kernel_id: " << perf_cfg.kernel_id << std::endl;
}

TEST_F(Conv3DKernelTuningUtilsTest, FullSolverPathway_ConvHipImplicitGemm3DGroupFwdXdlops)
{
    auto problem = GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::Forward);

    ctx = miopen::ExecutionContext(&handle);

    ConvHipImplicitGemm3DGroupFwdXdlops solver;

    ASSERT_TRUE(solver.IsApplicable(ctx, problem)) << "FWD solver not applicable for this problem";

    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg))
        << "Invalid FWD performance config";

    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    ASSERT_FALSE(solution.construction_params.empty())
        << "FWD solution construction_params is empty";
    ASSERT_TRUE(solution.invoker_factory) << "FWD solution invoker_factory is not set";
    ASSERT_GE(solution.workspace_sz, 0u) << "FWD workspace size should be non-negative";

    std::cout << "Selected FWD CK kernel_id: " << perf_cfg.kernel_id << std::endl;
}

TEST_F(Conv3DKernelTuningUtilsTest, FullSolverPathway_ConvHipImplicitGemm3DGroupBwdXdlops)
{
    auto problem =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardData);

    ctx = miopen::ExecutionContext(&handle);

    ConvHipImplicitGemm3DGroupBwdXdlops solver;

    ASSERT_TRUE(solver.IsApplicable(ctx, problem)) << "BWD solver not applicable for this problem";

    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg))
        << "Invalid BWD performance config";

    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    ASSERT_FALSE(solution.construction_params.empty())
        << "BWD solution construction_params is empty";
    ASSERT_TRUE(solution.invoker_factory) << "BWD solution invoker_factory is not set";
    ASSERT_GE(solution.workspace_sz, 0u) << "BWD workspace size should be non-negative";

    std::cout << "Selected BWD CK kernel_id: " << perf_cfg.kernel_id << std::endl;
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
