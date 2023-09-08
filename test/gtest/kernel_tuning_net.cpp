#include <gtest/ai_heuristics.hpp>
#include "../tensor_holder.hpp"
#include "get_handle.hpp"
#include <miopen/solver.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>

struct KernelTuningNetTestCase : AIModelTestCase
{
    bool expected_valid;
    std::string expected_config;
};

std::vector<KernelTuningNetTestCase> GetConvAsm1x1UFloatTestCases()
{
    return {{{{512, 192, 56, 56, 288, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
              miopen::conv::Direction::BackwardData,
              miopenFloat,
              miopenTensorNCHW},
             true,
             "1,16,1,64,2,2,1,4"},

            {{{1, 4, 2, 2, 4, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
              miopen::conv::Direction::Forward,
              miopenFloat,
              miopenTensorNCHW},
             false,
             ""}};
}

std::vector<KernelTuningNetTestCase> GetConvAsm1x1UHalfTestCases()
{
    return {{{{256, 2048, 7, 7, 512, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
              miopen::conv::Direction::Forward,
              miopenHalf,
              miopenTensorNCHW},
             true,
             "2,8,4,16,1,4,1,4"}};
}

template <typename G>
struct KernelTuningNetTest : public ::testing::TestWithParam<KernelTuningNetTestCase>
{
protected:
    void SetUp() override
    {
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
        auto test_case = GetParam();
        tensor<G> input_tensor =
            tensor<G>(test_case.data_type, test_case.layout, test_case.conv.GetInput());
        tensor<G> weights_tensor =
            tensor<G>(test_case.data_type, test_case.layout, test_case.conv.GetWeights());
        auto conv_desc                       = test_case.conv.GetConv();
        miopen::TensorDescriptor output_desc = conv_desc.GetForwardOutputTensor(
            input_tensor.desc, weights_tensor.desc, test_case.data_type);

        problem = (test_case.direction == miopen::conv::Direction::Forward)
                      ? miopen::conv::ProblemDescription(input_tensor.desc,
                                                         weights_tensor.desc,
                                                         output_desc,
                                                         conv_desc,
                                                         test_case.direction)
                      : miopen::conv::ProblemDescription(output_desc,
                                                         weights_tensor.desc,
                                                         input_tensor.desc,
                                                         conv_desc,
                                                         test_case.direction);

        expected_valid = test_case.expected_valid;
        expected       = test_case.expected_config;
#else
        GTEST_SKIP();
#endif
    }
    miopen::ProblemDescription problem;
    bool expected_valid;
    std::string expected;
};

struct KernelTuningNetTestFloat : KernelTuningNetTest<float>
{
};

struct KernelTuningNetTestHalf : KernelTuningNetTest<half_float::half>
{
};

template <typename T>
void TestParameterPredictionModel(miopen::ProblemDescription problem,
                                  bool expected_valid,
                                  std::string expected)
{
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    auto&& handle = get_handle();
    if(handle.GetDeviceName() != "gfx908")
        GTEST_SKIP();
    miopen::ExecutionContext ctx;
    ctx.SetStream(&handle);
    T perf_config;
    bool valid = false;
    perf_config.RunParmeterPredictionModel(ctx, problem, valid);
    ASSERT_EQ(valid, expected_valid)
        << "Expected parameters to be "
        << (expected_valid ? std::string("valid") : std::string("invalid")) << " but were "
        << (valid ? std::string("valid") : std::string("invalid"));
    if(expected_valid)
    {
        EXPECT_EQ(perf_config.ToString(), expected)
            << "Expected parameters: " << expected
            << "\nPredicted parameters: " << perf_config.ToString();
    }
#else
    std::ignore = problem;
    std::ignore = expected_valid;
    std::ignore = expected;
    GTEST_SKIP();
#endif
}

TEST_P(KernelTuningNetTestFloat, ConvAsm1x1UParameterPredictionModelFloat)
{
    TestParameterPredictionModel<miopen::solver::PerformanceConfigConvAsm1x1U>(
        problem, expected_valid, expected);
}

TEST_P(KernelTuningNetTestHalf, ConvAsm1x1UParameterPredictionModelHalf)
{
    TestParameterPredictionModel<miopen::solver::PerformanceConfigConvAsm1x1U>(
        problem, expected_valid, expected);
}

INSTANTIATE_TEST_SUITE_P(ConvAsm1x1UParameterPredictionModelFloatTest,
                         KernelTuningNetTestFloat,
                         testing::ValuesIn(GetConvAsm1x1UFloatTestCases()));

INSTANTIATE_TEST_SUITE_P(ConvAsm1x1UParameterPredictionModelHalfTest,
                         KernelTuningNetTestHalf,
                         testing::ValuesIn(GetConvAsm1x1UHalfTestCases()));
