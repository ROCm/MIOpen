#include <miopen/miopen.h>
#include "../tensor_holder.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <miopen/convolution.hpp>
#include <miopen/problem_description.hpp>
#include <miopen/solver.hpp>
#include "get_handle.hpp"
#include <unordered_map>

struct KernelTuningNetTestCase
{
    std::vector<size_t> input;
    std::vector<size_t> weight;
    std::vector<size_t> output;
    size_t pad_x;
    size_t pad_y;
    size_t stride_x;
    size_t stride_y;
    size_t dilation_x;
    size_t dilation_y;
    miopen::conv::Direction direction;
    miopenDataType_t data_type;
    miopenTensorLayout_t layout;
    bool expected_valid;
    std::string expected_config;
    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            {static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_y), static_cast<int>(dilation_x)}};
    }
};

template <typename G>
struct KernelTuningNetTest : public ::testing::TestWithParam<KernelTuningNetTestCase>
{
protected:
    void SetUp() override
    {
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
        auto test_case         = GetParam();
        tensor<G> input_tensor = tensor<G>(test_case.data_type, test_case.layout, test_case.input);
        tensor<G> weight_tensor =
            tensor<G>(test_case.data_type, test_case.layout, test_case.weight);
        tensor<G> output_tensor =
            tensor<G>(test_case.data_type, test_case.layout, test_case.output);
        problem        = miopen::ProblemDescription(input_tensor.desc,
                                             weight_tensor.desc,
                                             output_tensor.desc,
                                             test_case.GetConv(),
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

std::vector<KernelTuningNetTestCase> GetConvAsm1x1UFloatTestCases()
{
    return {{{512, 192, 56, 56},
             {288, 192, 1, 1},
             {512, 288, 56, 56},
             0,
             0,
             1,
             1,
             1,
             1,
             miopen::conv::Direction::BackwardData,
             miopenFloat,
             miopenTensorNCHW,
             true,
             "1,16,1,64,2,2,1,4"},
            {{1, 4, 2, 2},
             {4, 4, 1, 1},
             {1, 4, 2, 2},
             0,
             0,
             1,
             1,
             1,
             1,
             miopen::conv::Direction::Forward,
             miopenFloat,
             miopenTensorNCHW,
             false,
             ""}};
}

std::vector<KernelTuningNetTestCase> GetConvAsm1x1UHalfTestCases()
{
    return {{{256, 2048, 7, 7},
             {512, 2048, 1, 1},
             {256, 512, 7, 7},
             0,
             0,
             1,
             1,
             1,
             1,
             miopen::conv::Direction::Forward,
             miopenHalf,
             miopenTensorNCHW,
             true,
             "2,8,4,16,1,4,1,4"}};
}

template <typename T, typename G>
void TestParameterPredictionModel(miopen::ProblemDescription problem,
                                  bool expected_valid,
                                  std::string expected)
{
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
    auto&& handle = get_handle();
    if(handle.GetDeviceName() != "gfx908")
        GTEST_SKIP();
    miopen::ConvolutionContext ctx;
    ctx.SetStream(&handle);
    ctx.DetectRocm();
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
    TestParameterPredictionModel<miopen::solver::PerformanceConfigConvAsm1x1U, float>(
        problem, expected_valid, expected);
}

TEST_P(KernelTuningNetTestHalf, ConvAsm1x1UParameterPredictionModelHalf)
{
    TestParameterPredictionModel<miopen::solver::PerformanceConfigConvAsm1x1U, half_float::half>(
        problem, expected_valid, expected);
}

INSTANTIATE_TEST_SUITE_P(ConvAsm1x1UParameterPredictionModelFloatTest,
                         KernelTuningNetTestFloat,
                         testing::ValuesIn(GetConvAsm1x1UFloatTestCases()));

INSTANTIATE_TEST_SUITE_P(ConvAsm1x1UParameterPredictionModelHalfTest,
                         KernelTuningNetTestHalf,
                         testing::ValuesIn(GetConvAsm1x1UHalfTestCases()));
