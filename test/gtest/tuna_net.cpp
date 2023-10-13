#include <gtest/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include "../tensor_holder.hpp"
#include "get_handle.hpp"

struct TunaNetTestCase : AIModelTestCase
{
    std::size_t expected_solver;
};

std::vector<TunaNetTestCase> GetGfx908FloatTestCases()
{
    return {{{{5, 256, 267, 300, 64, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
              miopen::conv::Direction::Forward,
              miopenFloat,
              miopenTensorNCHW},
             4}};
}

std::vector<TunaNetTestCase> GetGfx908HalfTestCases()
{
    return {{{{16, 256, 20, 84, 512, 5, 5, 1, 1, 1, 1, 1, 1, miopenConvolution},
              miopen::conv::Direction::Forward,
              miopenHalf,
              miopenTensorNCHW},
             3}};
}

std::vector<TunaNetTestCase> GetGfx908BF16TestCases()
{
    return {{{{32, 1024, 15, 15, 512, 1, 1, 0, 0, 1, 1, 1, 1, miopenConvolution},
              miopen::conv::Direction::Forward,
              miopenBFloat16,
              miopenTensorNCHW},
             4}};
}

template <typename G>
struct TunaNetTest : public ::testing::TestWithParam<TunaNetTestCase>
{
protected:
    void SetUp() override
    {
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
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

        expected_solver = test_case.expected_solver;
#else
        GTEST_SKIP();
#endif
    }
    miopen::ProblemDescription problem;
    std::size_t expected_solver;
};

struct TunaNetTestFloat : TunaNetTest<float>
{
};

struct TunaNetTestHalf : TunaNetTest<half_float::half>
{
};

struct TunaNetTestBF16 : TunaNetTest<bfloat16>
{
};

void TestSolverPredictionModel(miopen::ProblemDescription& problem, std::size_t expected_solver)
{
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
    auto&& handle      = get_handle();
    std::string device = handle.GetDeviceName();
    if(device != "gfx908")
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
    GTEST_SKIP();
#endif
}

TEST_P(TunaNetTestFloat, Gfx908TestSolverPredictionModelFloat)
{
    TestSolverPredictionModel(problem, expected_solver);
}

TEST_P(TunaNetTestHalf, Gfx908TestSolverPredictionModelHalf)
{
    TestSolverPredictionModel(problem, expected_solver);
}

TEST_P(TunaNetTestBF16, Gfx908TestSolverPredictionModelBF16)
{
    TestSolverPredictionModel(problem, expected_solver);
}

INSTANTIATE_TEST_SUITE_P(Gfx908TestSolverPredictionModelFloatTest,
                         TunaNetTestFloat,
                         testing::ValuesIn(GetGfx908FloatTestCases()));

INSTANTIATE_TEST_SUITE_P(Gfx908TestSolverPredictionModelHalfTest,
                         TunaNetTestHalf,
                         testing::ValuesIn(GetGfx908HalfTestCases()));

INSTANTIATE_TEST_SUITE_P(Gfx908TestSolverPredictionModelBF16Test,
                         TunaNetTestBF16,
                         testing::ValuesIn(GetGfx908BF16TestCases()));
