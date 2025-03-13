#include <gtest/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include "../tensor_holder.hpp"
#include "get_handle.hpp"

struct TunaNetTestCase : AIModelTestCase
{
    std::size_t expected_solver;
    std::string device_architecture;
};

std::vector<TunaNetTestCase> GetGfx908FloatTestCases()
{
    return {{{{1, 5, 256, 64, {267, 300}, {1, 1}, {0, 0}, {1, 1}, {1, 1}},
              miopen::conv::Direction::Forward,
              miopenFloat,
              miopenTensorNCHW},
             4,
             "gfx908"}};
}

std::vector<TunaNetTestCase> GetGfx908HalfTestCases()
{
    return {{{{1, 16, 256, 512, {20, 84}, {5, 5}, {1, 1}, {1, 1}, {1, 1}},
              miopen::conv::Direction::Forward,
              miopenHalf,
              miopenTensorNCHW},
             3,
             "gfx908"}};
}

std::vector<TunaNetTestCase> GetGfx908BF16TestCases()
{
    return {{{{1, 32, 1024, 512, {15, 15}, {1, 1}, {0, 0}, {1, 1}, {1, 1}},
              miopen::conv::Direction::Forward,
              miopenBFloat16,
              miopenTensorNCHW},
             4,
             "gfx908"}};
}

std::vector<TunaNetTestCase> GetGfx90aFloatTestCases()
{
    return {{{{1, 5, 3, 64, {1301, 1333}, {7, 7}, {3, 3}, {2, 2}, {1, 1}},
              miopen::conv::Direction::Forward,
              miopenFloat,
              miopenTensorNCHW},
             6,
             "gfx90a"}};
}

std::vector<TunaNetTestCase> GetGfx90aHalfTestCases()
{
    return {{{{1, 24, 1024, 2048, {14, 14}, {1, 1}, {0, 0}, {2, 2}, {1, 1}},
              miopen::conv::Direction::Forward,
              miopenHalf,
              miopenTensorNCHW},
             4,
             "gfx90a"}};
}

std::vector<TunaNetTestCase> GetGfx90aBF16TestCases()
{
    return {{{{1, 2, 480, 192, {28, 28}, {1, 1}, {0, 0}, {1, 1}, {1, 1}},
              miopen::conv::Direction::Forward,
              miopenBFloat16,
              miopenTensorNCHW},
             6,
             "gfx90a"}};
}

template <typename G>
struct TunaNetTest : public ::testing::TestWithParam<TunaNetTestCase>
{
protected:
    void SetUp() override
    {
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
        auto test_case           = GetParam();
        tensor<G> input_tensor   = tensor<G>(test_case.layout, test_case.conv.GetInput());
        tensor<G> weights_tensor = tensor<G>(test_case.layout, test_case.conv.GetWeights());
        auto conv_desc           = test_case.conv.GetConv();
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

        expected_solver     = test_case.expected_solver;
        device_architecture = test_case.device_architecture;
#else
        GTEST_SKIP();
#endif
    }
    miopen::conv::ProblemDescription problem;
    std::size_t expected_solver;
    std::string device_architecture;
};

struct GPU_TunaNetTest_FP32 : TunaNetTest<float>
{
};

struct GPU_TunaNetTest_FP16 : TunaNetTest<half_float::half>
{
};

struct GPU_TunaNetTest_BFP16 : TunaNetTest<bfloat16>
{
};

void TestSolverPredictionModel(miopen::conv::ProblemDescription& problem,
                               std::size_t expected_solver,
                               std::string device_architecture)
{
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
    auto&& handle      = get_handle();
    std::string device = handle.GetDeviceName();
    if(device != device_architecture)
        GTEST_SKIP();
    miopen::ExecutionContext ctx;
    ctx.SetStream(&handle);
    std::vector<std::size_t> solvers = miopen::ai::immed_mode::PredictSolver(problem, ctx, device);
    std::size_t solver =
        std::distance(solvers.begin(), std::max_element(solvers.begin(), solvers.end()));
    ASSERT_EQ(solver, expected_solver)
        << "TunaNet predicted solver: " << solver
        << " when it should've predicted solver: " << expected_solver << std::endl;
#else
    std::ignore = problem;
    std::ignore = expected_solver;
    std::ignore = device_architecture;
    GTEST_SKIP();
#endif
}

TEST_P(GPU_TunaNetTest_FP32, TestSolverPredictionModelFloat)
{
    TestSolverPredictionModel(problem, expected_solver, device_architecture);
}

TEST_P(GPU_TunaNetTest_FP16, TestSolverPredictionModelHalf)
{
    TestSolverPredictionModel(problem, expected_solver, device_architecture);
}

TEST_P(GPU_TunaNetTest_BFP16, TestSolverPredictionModelBF16)
{
    TestSolverPredictionModel(problem, expected_solver, device_architecture);
}

INSTANTIATE_TEST_SUITE_P(SmokeGfx908,
                         GPU_TunaNetTest_FP32,
                         testing::ValuesIn(GetGfx908FloatTestCases()));

INSTANTIATE_TEST_SUITE_P(SmokeGfx908,
                         GPU_TunaNetTest_FP16,
                         testing::ValuesIn(GetGfx908HalfTestCases()));

INSTANTIATE_TEST_SUITE_P(SmokeGfx908,
                         GPU_TunaNetTest_BFP16,
                         testing::ValuesIn(GetGfx908BF16TestCases()));

INSTANTIATE_TEST_SUITE_P(SmokeGfx90a,
                         GPU_TunaNetTest_FP32,
                         testing::ValuesIn(GetGfx90aFloatTestCases()));

INSTANTIATE_TEST_SUITE_P(SmokeGfx90a,
                         GPU_TunaNetTest_FP16,
                         testing::ValuesIn(GetGfx90aHalfTestCases()));

INSTANTIATE_TEST_SUITE_P(SmokeGfx90a,
                         GPU_TunaNetTest_BFP16,
                         testing::ValuesIn(GetGfx90aBF16TestCases()));
