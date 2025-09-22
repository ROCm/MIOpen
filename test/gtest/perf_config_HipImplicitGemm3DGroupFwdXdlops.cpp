// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <gtest/group_conv.hpp>

#include <miopen/tensor.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <sstream>

using Problem = miopen::conv::ProblemDescription;
using Config  = miopen::solver::conv::PerformanceConfigHipImplicitGemm3DGroupFwdXdlops;

struct PerfConfigTestCase
{
    struct group_conv::GroupConvTestConfig<3u> conv;
    miopenDataType_t data_type;
    miopenTensorLayout_t layout;
    std::string arch;
};

// helper functions to set and unset environment variables in a cross-platform way
// if required by other tests, these can be moved to a common utility file
#if defined(_WIN32)
#include <Windows.h>
inline void set_env_var(const char* name, const char* value)
{
    SetEnvironmentVariableA(name, value);
}
inline void unset_env_var(const char* name)
{
    SetEnvironmentVariableA(name, nullptr);
}
#else
#include <cstdlib>
inline void set_env_var(const char* name, const char* value)
{
    setenv(name, value, 1);
}
inline void unset_env_var(const char* name)
{
    unsetenv(name);
}
#endif

std::vector<PerfConfigTestCase> GetPerfConfigTestCases(miopenDataType_t data_type, std::string arch)
{
    return {{{1, 128, 64, 32, {3, 28, 28}, {3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
             data_type,
             miopenTensorNCDHW,
             arch},
            {{1, 128, 64, 192, {3, 28, 28}, {3, 3, 3}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}},
             data_type,
             miopenTensorNCDHW,
             arch}};
}

template <miopenDataType_t date_type>
class PerfConfig_HipImplicitGemm3DGroupFwdXdlops
    : public ::testing::TestWithParam<PerfConfigTestCase>
{
protected:
    void TestConfigs()
    {
        auto test_case = GetParam();

        auto&& handle = get_handle();
        miopen::ExecutionContext ctx(&handle);
        if(test_case.arch != ctx.GetStream().GetDeviceName())
            GTEST_SKIP();

        auto input_tensor_desc =
            miopen::TensorDescriptor(test_case.data_type, test_case.conv.GetInput());

        auto weights_tensor_desc = miopen::TensorDescriptor(
            test_case.data_type, test_case.layout, test_case.conv.GetWeights());

        auto conv_desc = test_case.conv.GetConv();

        auto output_desc = conv_desc.GetForwardOutputTensor(
            input_tensor_desc, weights_tensor_desc, test_case.data_type);

        auto problem = miopen::conv::ProblemDescription(input_tensor_desc,
                                                        weights_tensor_desc,
                                                        output_desc,
                                                        conv_desc,
                                                        miopen::conv::Direction::Forward);

        Config cfg;
        cfg.HeuristicInit(ctx, problem);
        EXPECT_TRUE(cfg.index != 0) << "index is 0:" << test_case.conv;
    }

    void TestOverrideEnvVar()
    {
        auto test_case = GetParam();

        auto&& handle = get_handle();
        miopen::ExecutionContext ctx(&handle);
        if(test_case.arch != ctx.GetStream().GetDeviceName())
            GTEST_SKIP();

        auto input_tensor_desc =
            miopen::TensorDescriptor(test_case.data_type, test_case.conv.GetInput());

        auto weights_tensor_desc = miopen::TensorDescriptor(
            test_case.data_type, test_case.layout, test_case.conv.GetWeights());

        auto conv_desc = test_case.conv.GetConv();

        auto output_desc = conv_desc.GetForwardOutputTensor(
            input_tensor_desc, weights_tensor_desc, test_case.data_type);

        auto problem = miopen::conv::ProblemDescription(input_tensor_desc,
                                                        weights_tensor_desc,
                                                        output_desc,
                                                        conv_desc,
                                                        miopen::conv::Direction::Forward);

        // Check if hardcoded heuristics conditions are met
        bool will_use_hardcoded =
            (test_case.data_type == miopenBFloat16 || test_case.data_type == miopenHalf) &&
            problem.GetInChannels() > 8 && problem.GetGroupCount() == 1 &&
            problem.GetAlphaBetaCase() == DEFAULT;

        // Test override with value 1 (should select index 1 if hardcoded heuristics don't trigger)
        {
            set_env_var("MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_IDX_OVERRIDE", "1");

            Config cfg;
            cfg.HeuristicInit(ctx, problem);

            if(will_use_hardcoded)
            {
                // When hardcoded heuristics trigger, they override the simple index setting
                EXPECT_TRUE(cfg.index != 1)
                    << "Hardcoded heuristics should override simple index setting";
                EXPECT_TRUE(!cfg.kernel_id.empty())
                    << "Should have selected a kernel via hardcoded heuristics";
            }
            else
            {
                // When hardcoded heuristics don't trigger, simple index override should work
                if(!cfg.valid_kernels.empty() && cfg.valid_kernels.size() > 1)
                {
                    EXPECT_EQ(cfg.index, 1) << "Override should set index to 1";
                    EXPECT_EQ(cfg.kernel_id, cfg.valid_kernels[1])
                        << "kernel_id should match valid_kernels[1]";
                }
            }

            unset_env_var("MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_IDX_OVERRIDE");
        }

        // Test hardcoded heuristics for BF16/FP16 with appropriate conditions
        if(will_use_hardcoded)
        {
            set_env_var("MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_IDX_OVERRIDE", "0");

            Config cfg;
            cfg.HeuristicInit(ctx, problem);

            // Verify that hardcoded heuristics were applied
            EXPECT_TRUE(!cfg.kernel_id.empty()) << "Hardcoded heuristics should select a kernel";

            // Verify the kernel selected is reasonable (contains expected patterns)
            bool has_expected_pattern =
                cfg.kernel_id.find("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3") !=
                    std::string::npos &&
                (cfg.kernel_id.find("BlkGemmPipelineScheduler: Intrawave") != std::string::npos ||
                 cfg.kernel_id.find("BlkGemmPipelineScheduler: Interwave") != std::string::npos);

            EXPECT_TRUE(has_expected_pattern)
                << "Selected kernel should match hardcoded heuristic pattern: " << cfg.kernel_id;

            unset_env_var("MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_IDX_OVERRIDE");
        }

        // Test that override affects kernel selection by using a high index value
        // that won't trigger hardcoded heuristics
        {
            // First get normal result (without override)
            Config cfg_normal;
            cfg_normal.HeuristicInit(ctx, problem);

            // Use a high index that's less likely to trigger hardcoded heuristics
            // but still tests the override functionality
            set_env_var("MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_IDX_OVERRIDE", "50");

            Config cfg_override;
            cfg_override.HeuristicInit(ctx, problem);

            if(will_use_hardcoded)
            {
                // Even with index 50, hardcoded heuristics should still trigger
                // for BF16/FP16 with the right conditions
                EXPECT_TRUE(!cfg_override.kernel_id.empty()) << "Should select a kernel";
                // Both normal and override will use hardcoded heuristics, so they might be the same
                // The important thing is that the override path was taken
            }
            else
            {
                // For other cases, the override should work normally
                if(cfg_override.valid_kernels.size() > 50)
                {
                    EXPECT_EQ(cfg_override.index, 50) << "Override should select index 50";
                    EXPECT_NE(cfg_override.kernel_id, cfg_normal.kernel_id)
                        << "Override should select different kernel than normal path";
                }
                else
                {
                    // If there aren't enough kernels, it should fall back to index 0
                    EXPECT_EQ(cfg_override.index, 0)
                        << "Should fall back to index 0 when override index is out of range";
                }
            }

            unset_env_var("MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_FWD_XDLOPS_IDX_OVERRIDE");
        }
    }
};

using GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_BFP16 =
    PerfConfig_HipImplicitGemm3DGroupFwdXdlops<miopenBFloat16>;
using GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_FP16 =
    PerfConfig_HipImplicitGemm3DGroupFwdXdlops<miopenHalf>;

TEST_P(GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_BFP16, All) { TestConfigs(); }
TEST_P(GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_FP16, All) { TestConfigs(); }

TEST_P(GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_BFP16, OverrideEnvVar)
{
    TestOverrideEnvVar();
}
TEST_P(GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_FP16, OverrideEnvVar)
{
    TestOverrideEnvVar();
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_BFP16,
                         testing::ValuesIn(GetPerfConfigTestCases(miopenBFloat16, "gfx942")));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_PerfConfig_HipImplicitGemm3DGroupFwdXdlops_FP16,
                         testing::ValuesIn(GetPerfConfigTestCases(miopenHalf, "gfx942")));
