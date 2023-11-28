#include <gtest/ai_heuristics.hpp>
#include "../tensor_holder.hpp"
#include "get_handle.hpp"
#include <miopen/solver.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>

struct KernelTuningNetTestCase : AIModelTestCase
{
    std::string expected_config;
};

std::vector<KernelTuningNetTestCase> GetConvAsm1x1UTestCases()
{
    return {{{{512, 192, 56, 56, 288, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
              miopen::conv::Direction::BackwardData,
              miopenFloat,
              miopenTensorNCHW},
             "1,16,1,64,2,2,1,4"},
            {{{256, 2048, 7, 7, 512, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
              miopen::conv::Direction::Forward,
              miopenHalf,
              miopenTensorNCHW},
             "2,8,4,16,1,4,1,4"}};
}

std::vector<KernelTuningNetTestCase> GetConvHipIgemmGroupFwdXdlopsTestCases()
{
    return {{{{128, 64, 209, 209, 128, 3, 3, 0, 0, 2, 2, 1, 1, miopenConvolution},
              miopen::conv::Direction::Forward,
              miopenFloat,
              miopenTensorNHWC},
             "DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<256, 128, 128, 16, Default, 32, 32, 2, 2, "
             "4, 4, 1, 1>"}};
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
#else
        GTEST_SKIP();
#endif
    }
    miopen::conv::ProblemDescription problem;
    std::string expected;
};

template <typename T>
void TestParameterPredictionModel(miopen::conv::ProblemDescription problem, std::string expected)
{
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    auto&& handle = get_handle();
    miopen::ExecutionContext ctx;
    ctx.SetStream(&handle);
    T perf_config;
    if(!perf_config.IsModelApplicable(ctx, problem))
        GTEST_SKIP();
    perf_config.HeuristicInit(ctx, problem);
    EXPECT_EQ(perf_config.ToString(), expected)
        << "Expected parameters: " << expected
        << "\nPredicted parameters: " << perf_config.ToString();
#else
    std::ignore = problem;
    std::ignore = expected;
    GTEST_SKIP();
#endif
}

struct KernelTuningNetTestConvAsm1x1U : KernelTuningNetTest
{
};

struct KernelTuningNetTestConvHipIgemmGroupFwdXdlops : KernelTuningNetTest
{
};

TEST_P(KernelTuningNetTestConvAsm1x1U, ConvAsm1x1UParameterPredictionModel)
{
    TestParameterPredictionModel<miopen::solver::conv::PerformanceConfigConvAsm1x1U>(problem,
                                                                                     expected);
}

TEST_P(KernelTuningNetTestConvHipIgemmGroupFwdXdlops,
       ConvHipIgemmGroupFwdXdlopsParameterPredictionModel)
{
    TestParameterPredictionModel<
        miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupFwdXdlops>(problem, expected);
}

INSTANTIATE_TEST_SUITE_P(ConvAsm1x1UParameterPredictionModelTest,
                         KernelTuningNetTestConvAsm1x1U,
                         testing::ValuesIn(GetConvAsm1x1UTestCases()));

INSTANTIATE_TEST_SUITE_P(ConvHipIgemmGroupFwdXdlopsParameterPredictionModelTest,
                         KernelTuningNetTestConvHipIgemmGroupFwdXdlops,
                         testing::ValuesIn(GetConvHipIgemmGroupFwdXdlopsTestCases()));
