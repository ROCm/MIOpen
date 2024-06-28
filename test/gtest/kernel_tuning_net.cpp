#include <gtest/ai_heuristics.hpp>
#include "../tensor_holder.hpp"
#include "get_handle.hpp"
#include <miopen/solver.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>

struct KernelTuningNetTestCase : AIModelTestCase
{
    std::string expected_config;
    std::string arch;
};

std::vector<KernelTuningNetTestCase> GetConvAsm1x1UTestCases()
{
    return {{{{1, 512, 192, 288, {56, 56}, {1, 1}, {0, 0}, {1, 1}, {1, 1}},
              miopen::conv::Direction::BackwardData,
              miopenFloat,
              miopenTensorNCHW},
             "1,16,1,64,2,2,1,4",
             "gfx908"},
            {{{1, 256, 2048, 512, {7, 7}, {1, 1}, {0, 0}, {1, 1}, {1, 1}},
              miopen::conv::Direction::Forward,
              miopenHalf,
              miopenTensorNCHW},
             "2,8,4,16,1,4,1,4",
             "gfx908"}};
}

std::vector<KernelTuningNetTestCase> GetConvHipIgemmGroupFwdXdlopsTestCases()
{
    return {
        {{{1, 128, 64, 128, {209, 209}, {3, 3}, {0, 0}, {2, 2}, {1, 1}},
          miopen::conv::Direction::Forward,
          miopenFloat,
          miopenTensorNHWC},
         "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Default, 32, 32, 2, 2, "
         "4, 4, 4, 1, 1>",
         "gfx90a"},
        {{{16, 256, 2016, 192, {7, 7}, {1, 1}, {0, 0}, {1, 1}, {1, 1}},
          miopen::conv::Direction::Forward,
          miopenHalf,
          miopenTensorNHWC},
         "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Stride1Pad0, 32, "
         "32, 2, 2, 1, "
         "1, 1, 1, 1>",
         "gfx942"},
    };
}

std::vector<KernelTuningNetTestCase> GetConvHipIgemmGroupBwdXdlopsTestCases()
{
    return {{{{32, 4, 256, 256, {59, 59}, {3, 3}, {1, 1}, {2, 2}, {1, 1}},
              miopen::conv::Direction::BackwardData,
              miopenHalf,
              miopenTensorNHWC},
             "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<128, 128, 32, 32, 8, 8, Default, "
             "32, 32, 2, 1, 8, 8, 1, 1>",
             "gfx90a"},
            {{{64, 96, 64, 64, {224, 224}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              miopen::conv::Direction::BackwardData,
              miopenFloat,
              miopenTensorNHWC},
             "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 64, 64, 32, 8, 8, Default, 32, "
             "32, 2, 2, 1, 1, 1, 1>",
             "gfx942"}};
}

std::vector<KernelTuningNetTestCase> GetConvHipIgemmGroupWrwXdlopsTestCases()
{
    return {
        {{{1, 512, 3, 64, {219, 219}, {11, 11}, {2, 2}, {4, 4}, {1, 1}},
          miopen::conv::Direction::BackwardWeights,
          miopenFloat,
          miopenTensorNHWC},
         "DeviceGroupedConvBwdWeight_Xdl_CShuffle<128, 128, 32, 4, Default, 4, 2, 1, 4, 4, 1, 1, "
         "1, 1, 1>+128",
         "gfx942"},
        {{{32, 1024, 480, 64, {14, 14}, {1, 1}, {0, 0}, {1, 1}, {1, 1}},
          miopen::conv::Direction::BackwardWeights,
          miopenHalf,
          miopenTensorNHWC},
         "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<64, 16, 16, 32, Default, 8, 1, 1, 1, 4, "
         "1, 4, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v5, 1>+128",
         "gfx942"},
        {{{1, 16, 128, 256, {27, 27}, {3, 3}, {0, 0}, {1, 2}, {1, 1}},
          miopen::conv::Direction::BackwardWeights,
          miopenHalf,
          miopenTensorNHWC},
         "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64, 64, 32, 4, Default, 8, 2, 1, 8, 4, 8, 2, "
         "1, 1, 8>+1",
         "gfx90a"}};
}

struct KernelTuningNetTest : public ::testing::TestWithParam<KernelTuningNetTestCase>
{
protected:
    void SetUp() override
    {
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
        auto test_case                             = GetParam();
        miopen::TensorDescriptor input_tensor_desc = miopen::TensorDescriptor(
            test_case.data_type, test_case.layout, test_case.conv.GetInput());
        miopen::TensorDescriptor weights_tensor_desc = miopen::TensorDescriptor(
            test_case.data_type, test_case.layout, test_case.conv.GetWeights());
        auto conv_desc                       = test_case.conv.GetConv();
        miopen::TensorDescriptor output_desc = conv_desc.GetForwardOutputTensor(
            input_tensor_desc, weights_tensor_desc, test_case.data_type);
        problem  = (test_case.direction == miopen::conv::Direction::Forward)
                       ? miopen::conv::ProblemDescription(input_tensor_desc,
                                                         weights_tensor_desc,
                                                         output_desc,
                                                         conv_desc,
                                                         test_case.direction)
                       : miopen::conv::ProblemDescription(output_desc,
                                                         weights_tensor_desc,
                                                         input_tensor_desc,
                                                         conv_desc,
                                                         test_case.direction);
        expected = test_case.expected_config;
        arch     = test_case.arch;
#else
        GTEST_SKIP();
#endif
    }
    miopen::conv::ProblemDescription problem;
    std::string arch;
    std::string expected;
};

template <typename T>
void TestParameterPredictionModel(miopen::conv::ProblemDescription problem,
                                  std::string expected,
                                  std::string arch)
{
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    auto&& handle = get_handle();
    miopen::ExecutionContext ctx;
    ctx.SetStream(&handle);
    T perf_config;
    if(arch != ctx.GetStream().GetDeviceName())
        GTEST_SKIP();
    if(!perf_config.IsModelApplicable(ctx, problem))
        GTEST_SKIP();
    perf_config.HeuristicInit(ctx, problem);
    EXPECT_EQ(perf_config.ToString(), expected)
        << "Expected parameters: " << expected
        << "\nPredicted parameters: " << perf_config.ToString();
#else
    std::ignore = problem;
    std::ignore = expected;
    std::ignore = arch;
    GTEST_SKIP();
#endif
}

struct KernelTuningNetTestConvAsm1x1U : KernelTuningNetTest
{
};

struct KernelTuningNetTestConvHipIgemmGroupFwdXdlops : KernelTuningNetTest
{
};

struct KernelTuningNetTestConvHipIgemmGroupBwdXdlops : KernelTuningNetTest
{
};

struct KernelTuningNetTestConvHipIgemmGroupWrwXdlops : KernelTuningNetTest
{
};

TEST_P(KernelTuningNetTestConvAsm1x1U, ConvAsm1x1UParameterPredictionModel)
{
    TestParameterPredictionModel<miopen::solver::conv::PerformanceConfigConvAsm1x1U>(
        problem, expected, arch);
}

TEST_P(KernelTuningNetTestConvHipIgemmGroupFwdXdlops,
       ConvHipIgemmGroupFwdXdlopsParameterPredictionModel)
{
    TestParameterPredictionModel<
        miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupFwdXdlops>(
        problem, expected, arch);
}

TEST_P(KernelTuningNetTestConvHipIgemmGroupBwdXdlops,
       ConvHipIgemmGroupBwdXdlopsParameterPredictionModel)
{
    TestParameterPredictionModel<
        miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupBwdXdlops>(
        problem, expected, arch);
}

TEST_P(KernelTuningNetTestConvHipIgemmGroupWrwXdlops,
       ConvHipIgemmGroupWrwXdlopsParameterPredictionModel)
{
    TestParameterPredictionModel<
        miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupWrwXdlops>(
        problem, expected, arch);
}

INSTANTIATE_TEST_SUITE_P(ConvAsm1x1UParameterPredictionModelTest,
                         KernelTuningNetTestConvAsm1x1U,
                         testing::ValuesIn(GetConvAsm1x1UTestCases()));

INSTANTIATE_TEST_SUITE_P(ConvHipIgemmGroupFwdXdlopsParameterPredictionModelTest,
                         KernelTuningNetTestConvHipIgemmGroupFwdXdlops,
                         testing::ValuesIn(GetConvHipIgemmGroupFwdXdlopsTestCases()));

INSTANTIATE_TEST_SUITE_P(ConvHipIgemmGroupBwdXdlopsParameterPredictionModelTest,
                         KernelTuningNetTestConvHipIgemmGroupBwdXdlops,
                         testing::ValuesIn(GetConvHipIgemmGroupBwdXdlopsTestCases()));

INSTANTIATE_TEST_SUITE_P(ConvHipIgemmGroupWrwXdlopsParameterPredictionModelTest,
                         KernelTuningNetTestConvHipIgemmGroupWrwXdlops,
                         testing::ValuesIn(GetConvHipIgemmGroupWrwXdlopsTestCases()));
