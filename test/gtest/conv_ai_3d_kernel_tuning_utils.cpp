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

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
using namespace miopen::solver::conv;
using namespace miopen::ai::tuning::candidate_selection;

namespace {
// Helper: layout string to code (must match GetFeatures3D)
int LayoutStringToCode(const std::string& layout)
{
    if(layout == "NCDHW")
        return 0;
    if(layout == "NDHWC")
        return 1;
    return -1; // Unknown
}

// Dummy kernels for testing
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

// Dummy fill_valid_kernels for testing
static std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)>
    fill_valid_kernels = [](const miopen::conv::ProblemDescription&) { return dummy_kernels; };

// Default validation function: accepts all kernel/split_k combinations
inline constexpr auto accept_all_combinations = [](int, int) { return true; };

// Helper: reusable problem description
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

    miopen::ConvolutionDescriptor conv_desc(
        3, miopenConvolution, miopenPaddingDefault, pads, strides, dilations, trans_output_pads);

    return miopen::conv::ProblemDescription(in_desc, weights_desc, out_desc, conv_desc, direction);
}

// Helper: check GetFeatures3D map values
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

// Helper: check if model files exist for architecture
bool CheckModelFilesExist(const std::string& arch, const std::string& solver_name)
{
    std::string db_path = miopen::GetSystemDbPath().string();
    auto metadata       = db_path + "/" + arch + "_" + solver_name + "_metadata.tn.model";
    auto input_encoder  = db_path + "/" + arch + "_" + solver_name + "_input_encoder.tn.model";
    auto kernel_config_encoder =
        db_path + "/" + arch + "_" + solver_name + "_kernel_config_encoder.tn.model";

    return miopen::fs::exists(metadata) && miopen::fs::exists(input_encoder) &&
           miopen::fs::exists(kernel_config_encoder);
}

struct Conv3DKernelTuningTestParam
{
    miopenDataType_t data_type;
    miopen::conv::Direction direction;
    std::string test_name;
};

// Test case generators
std::vector<Conv3DKernelTuningTestParam> GenSmokeTestCases()
{
    return {
        {miopenFloat, miopen::conv::Direction::Forward, "Fwd"},
        {miopenFloat, miopen::conv::Direction::BackwardWeights, "Wrw"},
    };
}

std::vector<Conv3DKernelTuningTestParam> GenFullTestCases()
{
    return {
        {miopenFloat, miopen::conv::Direction::Forward, "Fwd"},
        {miopenFloat, miopen::conv::Direction::BackwardData, "Bwd"},
        {miopenFloat, miopen::conv::Direction::BackwardWeights, "Wrw"},
        {miopenHalf, miopen::conv::Direction::BackwardWeights, "Half"},
        {miopenBFloat16, miopen::conv::Direction::BackwardWeights, "BFloat16"},
    };
}

// Test name generator
std::string
Conv3DKernelTuningTestName(const ::testing::TestParamInfo<Conv3DKernelTuningTestParam>& info)
{
    return info.param.test_name;
}

// Helper function for metadata encoding validation
// Validates that all CK kernel instances can be encoded without errors
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
void ValidateMetadataEncoding(const std::string& solver_name,
                              const std::vector<std::string>& all_ck_kernels,
                              const std::string& device_arch)
{
    ASSERT_FALSE(all_ck_kernels.empty()) << "No CK instances found for " << solver_name;
    MIOPEN_LOG_I("Testing " << all_ck_kernels.size() << " total CK instances for " << solver_name);

    const auto& model    = GetCandidateSelectionModel(device_arch, solver_name);
    const auto& metadata = model.metadata();

    // Get supported kernel names from metadata (check both encodings and constants)
    std::set<std::string> supported_kernel_names;

    auto encodings_it = metadata.sequence_encodings().find("000_kernel_name");
    if(encodings_it != metadata.sequence_encodings().end())
    {
        // Variable kernel names - get all encoded values
        for(const auto& [name, value] : encodings_it->second)
        {
            supported_kernel_names.insert(name);
        }
    }
    else
    {
        // Check if it's a constant (single kernel type)
        auto constant_kernel = metadata.GetOutputConstant("000_kernel_name");
        if(constant_kernel.has_value())
        {
            supported_kernel_names.insert(constant_kernel.value());
            MIOPEN_LOG_I("Kernel name is constant: " << constant_kernel.value());
        }
    }

    // Separate kernels into supported and unsupported
    std::vector<std::string> supported_kernels;
    std::vector<std::string> unsupported_kernels;

    if(!supported_kernel_names.empty())
    {
        for(const auto& typestring : all_ck_kernels)
        {
            auto tokens = GetKernelAsTokens(typestring);
            if(!tokens.empty())
            {
                const std::string& kernel_name = tokens[0];
                if(supported_kernel_names.count(kernel_name) > 0)
                {
                    supported_kernels.push_back(typestring);
                }
                else
                {
                    unsupported_kernels.push_back(typestring);
                }
            }
        }
    }
    else
    {
        // No kernel name filtering available - test all kernels
        MIOPEN_LOG_I("No kernel name filtering available - testing all kernels");
        supported_kernels = all_ck_kernels;
    }

    // Log unsupported kernels (expected - these are CK kernels not in the model)
    if(!unsupported_kernels.empty())
    {
        MIOPEN_LOG_I("Found " << unsupported_kernels.size()
                              << " CK kernels not in metadata (expected)");
    }

    ASSERT_FALSE(supported_kernels.empty()) << "No supported kernels found";
    MIOPEN_LOG_I("Testing " << supported_kernels.size() << " supported kernels");

    // Create kernel parameter vectors from supported TypeStrings
    std::vector<std::vector<std::string>> kernel_params;
    for(const auto& typestring : supported_kernels)
    {
        auto tokens = GetKernelAsTokens(typestring);

        // Check if this kernel type requires split_k to be appended
        // by checking if the kernel_str_mapping includes a parameter with "splitk" in its name
        if(!tokens.empty())
        {
            const std::string& kernel_name = tokens[0];
            try
            {
                auto kernel_mapping = metadata.GetKernelStrMapping(kernel_name);

                // Check if any parameter name contains "splitk" (case-insensitive)
                bool has_split_k_mapping = false;
                for(const auto& [idx, param_name] : kernel_mapping)
                {
                    std::string param_lower = param_name;
                    std::transform(param_lower.begin(),
                                   param_lower.end(),
                                   param_lower.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if(param_lower.find("splitk") != std::string::npos)
                    {
                        has_split_k_mapping = true;
                        break;
                    }
                }

                // If split_k is expected in the mapping, append a default value
                if(has_split_k_mapping)
                {
                    tokens.push_back("1"); // Add default split_k value
                }
            }
            catch(const std::exception&)
            {
                // Kernel not in metadata - skip split_k append
            }
        }

        kernel_params.push_back(tokens);
    }

    // Encode all supported kernels
    auto encoded = EncodeKernelParams(kernel_params, metadata);

    // Verify no NaN values (which indicate encoding failures)
    int nan_count = 0;
    std::vector<std::string> failed_kernels;

    for(size_t i = 0; i < encoded.size(); ++i)
    {
        bool has_nan = false;
        for(size_t j = 0; j < encoded[i].size(); ++j)
        {
            if(std::isnan(encoded[i][j]))
            {
                nan_count++;
                has_nan = true;
            }
        }
        if(has_nan)
        {
            failed_kernels.push_back(supported_kernels[i]);
        }
    }

    if(nan_count > 0)
    {
        MIOPEN_LOG_E("Found " << nan_count << " NaN encodings across " << failed_kernels.size()
                              << " kernels");
        for(const auto& kernel : failed_kernels)
        {
            MIOPEN_LOG_E("  Failed kernel: " << kernel);
        }
    }

    EXPECT_EQ(nan_count, 0) << "Found " << nan_count
                            << " NaN encodings - indicates missing metadata entries";
}
#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
} // anonymous namespace

// ------------------- Base Test Fixture -------------------

class GPU_Conv3DKernelTuning_Base : public ::testing::Test
{
protected:
    miopen::Handle handle;
    miopen::ExecutionContext ctx;
    std::string device_arch;
    std::string solver_name = "ConvHipImplicitGemm3DGroupWrwXdlops";

    void SetUp() override
    {
        // Early exit for architecture compatibility
        device_arch = handle.GetDeviceName();
        ctx         = miopen::ExecutionContext(&handle);

        // Check if this is an AI-dependent test that requires model files
        if(RequiresAIModels() && !CheckModelFilesExist(device_arch, solver_name))
        {
            GTEST_SKIP() << "Test requires AI model files for " << device_arch
                         << " (found device: " << device_arch
                         << ", no compatible model files available)";
        }
    }

    virtual bool RequiresAIModels() const { return false; }
};

// ------------------- Platform-Agnostic Tests -------------------

class GPU_Conv3DKernelTuning_FP32
    : public GPU_Conv3DKernelTuning_Base,
      public ::testing::WithParamInterface<Conv3DKernelTuningTestParam>
{
protected:
    bool RequiresAIModels() const override { return false; }
};

TEST_P(GPU_Conv3DKernelTuning_FP32, GetFeatures3D_Test)
{
    const auto& param = GetParam();
    auto problem      = GetReusableProblemDescription(param.data_type, param.direction);
    int max_cu        = 304;
    auto features     = GetFeatures3D(problem, max_cu, device_arch);

    ASSERT_EQ(features.size(), 29u);
    CheckGetFeatures3D_MapValues(features, problem, param.direction);
}

TEST_F(GPU_Conv3DKernelTuning_FP32, GetKernelAsTokens_Test)
{
    auto tokens = GetKernelAsTokens("type<param1,param2>");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "type");
    EXPECT_EQ(tokens[1], "param1");
    EXPECT_EQ(tokens[2], "param2");

    auto empty = GetKernelAsTokens("");
    ASSERT_TRUE(empty.empty());
}

TEST_F(GPU_Conv3DKernelTuning_FP32, GenerateSplitK_Test)
{
    auto split_ks             = GenerateSplitK(8);
    std::vector<int> expected = {1, 2, 4, 8};
    ASSERT_EQ(split_ks, expected);
}

// ------------------- AI Model Tests (Architecture-Dependent) -------------------

class GPU_Conv3DKernelTuningAI_FP32 : public GPU_Conv3DKernelTuning_Base
{
protected:
    bool RequiresAIModels() const override { return true; }
};

TEST_F(GPU_Conv3DKernelTuningAI_FP32, CandidateSelectionModel_Test)
{
    EXPECT_NO_THROW({
        miopen::ai::tuning::candidate_selection::CandidateSelectionModel model(device_arch,
                                                                               solver_name);
    });

    auto& model = miopen::ai::tuning::candidate_selection::GetCandidateSelectionModel(device_arch,
                                                                                      solver_name);
    const auto& meta = model.metadata();
    ASSERT_FALSE(meta.input_params().empty());
    ASSERT_FALSE(meta.output_params().empty());
}

TEST_F(GPU_Conv3DKernelTuningAI_FP32, RunParameterPredictionModel_Test)
{
    auto problem =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);

    int index = 0, split_k = 1;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;

    auto [ai_success, result] =
        miopen::solver::conv::RunParameterPredictionModel<float>(ctx,
                                                                 problem,
                                                                 valid_kernels,
                                                                 index,
                                                                 split_k,
                                                                 kernel_id,
                                                                 fill_valid_kernels,
                                                                 solver_name,
                                                                 accept_all_combinations);

    ASSERT_TRUE(ai_success);
    ASSERT_FALSE(kernel_id.empty());
}

TEST_F(GPU_Conv3DKernelTuningAI_FP32, RunParameterPredictionModel_Fallback_Test)
{
    std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)> empty_kernels =
        [](const miopen::conv::ProblemDescription&) { return std::vector<std::string>{}; };

    auto problem =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);
    int index = 0, split_k = 1;
    std::string kernel_id;
    std::vector<std::string> valid_kernels;

    auto [ai_success, result] =
        miopen::solver::conv::RunParameterPredictionModel<float>(ctx,
                                                                 problem,
                                                                 valid_kernels,
                                                                 index,
                                                                 split_k,
                                                                 kernel_id,
                                                                 empty_kernels,
                                                                 solver_name,
                                                                 accept_all_combinations);

    ASSERT_FALSE(ai_success);
    ASSERT_TRUE(kernel_id.empty());
}

// ------------------- Metadata Encoding Validation Tests -------------------

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

TEST_F(GPU_Conv3DKernelTuningAI_FP32, MetadataEncodingValidation_AllCKInstances_Wrw_Test)
{
    ValidateMetadataEncoding(
        "ConvHipImplicitGemm3DGroupWrwXdlops", GetAllWrwKernelTypeStrings(), device_arch);
}

TEST_F(GPU_Conv3DKernelTuningAI_FP32, MetadataEncodingValidation_AllCKInstances_Fwd_Test)
{
    ValidateMetadataEncoding(
        "ConvHipImplicitGemm3DGroupFwdXdlops", GetAllFwdKernelTypeStrings(), device_arch);
}

TEST_F(GPU_Conv3DKernelTuningAI_FP32, MetadataEncodingValidation_AllCKInstances_Bwd_Test)
{
    ValidateMetadataEncoding(
        "ConvHipImplicitGemm3DGroupBwdXdlops", GetAllBwdKernelTypeStrings(), device_arch);
}

#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL

// ------------------- Full Solver Tests -------------------

TEST_F(GPU_Conv3DKernelTuningAI_FP32, FullSolverPathway_Wrw_Test)
{
    auto problem =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);
    ConvHipImplicitGemm3DGroupWrwXdlops solver;

    ASSERT_TRUE(solver.IsApplicable(ctx, problem));
    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg));
    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    ASSERT_FALSE(solution.construction_params.empty());
    ASSERT_TRUE(solution.invoker_factory);
    ASSERT_GE(solution.workspace_sz, 0u);
}

TEST_F(GPU_Conv3DKernelTuningAI_FP32, FullSolverPathway_Fwd_Test)
{
    auto problem = GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::Forward);
    ConvHipImplicitGemm3DGroupFwdXdlops solver;

    ASSERT_TRUE(solver.IsApplicable(ctx, problem));
    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg));
    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    ASSERT_FALSE(solution.construction_params.empty());
    ASSERT_TRUE(solution.invoker_factory);
    ASSERT_GE(solution.workspace_sz, 0u);
}

TEST_F(GPU_Conv3DKernelTuningAI_FP32, FullSolverPathway_Bwd_Test)
{
    auto problem =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardData);
    ConvHipImplicitGemm3DGroupBwdXdlops solver;

    ASSERT_TRUE(solver.IsApplicable(ctx, problem));
    auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
    ASSERT_TRUE(solver.IsValidPerformanceConfig(ctx, problem, perf_cfg));
    auto solution = solver.GetSolution(ctx, problem, perf_cfg);

    ASSERT_FALSE(solution.construction_params.empty());
    ASSERT_TRUE(solution.invoker_factory);
    ASSERT_GE(solution.workspace_sz, 0u);
}

// ------------------- Cross-Platform Diagnostic Tests -------------------

class CPU_Conv3DKernelTuningDiagnostic_NONE : public GPU_Conv3DKernelTuning_Base
{
protected:
    bool RequiresAIModels() const override { return false; }
};

TEST_F(CPU_Conv3DKernelTuningDiagnostic_NONE, ArchitectureCompatibility_Test)
{
    std::vector<std::string> test_architectures = {"gfx942", "gfx90a", "gfx908", "gfx1101"};

    std::cout << "=== Architecture Compatibility Report ===" << std::endl;
    std::cout << "Current device: " << device_arch << std::endl;

    for(const auto& arch : test_architectures)
    {
        bool files_exist = CheckModelFilesExist(arch, solver_name);
        std::cout << arch << ": " << (files_exist ? "SUPPORTED" : "NOT SUPPORTED") << std::endl;
    }
}

TEST_F(CPU_Conv3DKernelTuningDiagnostic_NONE, GetFeatures3D_CrossPlatform_Test)
{
    auto problem                        = GetReusableProblemDescription();
    int max_cu                          = 304;
    std::vector<std::string> test_archs = {"gfx942", "gfx90a", "gfx908"};

    for(const auto& arch : test_archs)
    {
        auto features = GetFeatures3D(problem, max_cu, arch);
        EXPECT_EQ(features.size(), 29u) << "Feature count inconsistent for " << arch;
        EXPECT_EQ(features.at("spatial_dim"), 3.0f) << "Spatial dim incorrect for " << arch;
    }
}

// ------------------- Test Instantiations -------------------

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv3DKernelTuning_FP32,
                         ::testing::ValuesIn(GenSmokeTestCases()),
                         Conv3DKernelTuningTestName);

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv3DKernelTuning_FP32,
                         ::testing::ValuesIn(GenFullTestCases()),
                         Conv3DKernelTuningTestName);

#else
// Dummy test when AI kernel tuning is disabled
TEST(CPU_Conv3DKernelTuningDisabled_NONE, FeatureDisabled)
{
    GTEST_SKIP() << "AI kernel tuning features are disabled in this build";
}
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
