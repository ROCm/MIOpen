/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <gtest/ai_heuristics.hpp>
#include "../tensor_holder.hpp"
#include "get_handle.hpp"
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include "../../driver/driver.hpp"

struct KernelTuningNetTestCase : AIModelTestCase
{
    std::string expected_config;
    std::string arch;
};

std::vector<KernelTuningNetTestCase> GetConvAsm1x1UTestCases_FP32()
{
    return {{{{1, 512, 192, 288, {56, 56}, {1, 1}, {0, 0}, {1, 1}, {1, 1}},
              miopen::conv::Direction::BackwardData,
              miopenFloat,
              miopenTensorNCHW},
             "1,16,1,64,2,2,1,4",
             "gfx908"}};
}
std::vector<KernelTuningNetTestCase> GetConvAsm1x1UTestCases_FP16()
{
    return {{{{1, 256, 2048, 512, {7, 7}, {1, 1}, {0, 0}, {1, 1}, {1, 1}},
              miopen::conv::Direction::Forward,
              miopenHalf,
              miopenTensorNCHW},
             "2,8,4,16,1,4,1,4",
             "gfx908"}};
}

std::vector<KernelTuningNetTestCase> GetConvHipIgemmGroupFwdXdlopsTestCases_FP32()
{
    return {{{{1, 128, 64, 128, {209, 209}, {3, 3}, {0, 0}, {2, 2}, {1, 1}},
              miopen::conv::Direction::Forward,
              miopenFloat,
              miopenTensorNHWC},
             "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<256, 128, 128, 16, Default, 32, 32, 2, "
             "2, 4, 4, 4, 1, 1, 1>",
             "gfx90a"}};
}

std::vector<KernelTuningNetTestCase> GetConvHipIgemmGroupFwdXdlopsTestCases_FP16()
{
    return {{{{16, 256, 2016, 192, {7, 7}, {1, 1}, {0, 0}, {1, 1}, {1, 1}},
              miopen::conv::Direction::Forward,
              miopenHalf,
              miopenTensorNHWC},
             "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<64, 64, 64, 32, Filter1x1Pad0, "
             "32, 32, 2, 2, 1, 1, 1, 1, 1, 1>",
             "gfx942"}};
}

std::vector<KernelTuningNetTestCase> GetConvHipIgemmGroupBwdXdlopsTestCases_FP32()
{
    return {{{{64, 96, 64, 64, {224, 224}, {3, 3}, {1, 1}, {1, 1}, {1, 1}},
              miopen::conv::Direction::BackwardData,
              miopenFloat,
              miopenTensorNHWC},
             "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<64, 64, 64, 32, 8, 8, Default, 32, "
             "32, 2, 2, 1, 1, 1, 1>",
             "gfx942"}};
}

std::vector<KernelTuningNetTestCase> GetConvHipIgemmGroupBwdXdlopsTestCases_FP16()
{
    return {{{{32, 4, 256, 256, {59, 59}, {3, 3}, {1, 1}, {2, 2}, {1, 1}},
              miopen::conv::Direction::BackwardData,
              miopenHalf,
              miopenTensorNHWC},
             "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<128, 128, 32, 32, 8, 8, Default, "
             "32, 32, 2, 1, 8, 8, 1, 1>",
             "gfx90a"}};
}

std::vector<KernelTuningNetTestCase> GetConvHipIgemmGroupWrwXdlopsTestCases_FP32()
{
    return {{{{1, 512, 3, 64, {219, 219}, {11, 11}, {2, 2}, {4, 4}, {1, 1}},
              miopen::conv::Direction::BackwardWeights,
              miopenFloat,
              miopenTensorNHWC},
             "DeviceGroupedConvBwdWeight_Xdl_CShuffle<128, 128, 32, 4, Default, 4, 2, 1, 4, 4, 1, "
             "1, 1, 1, 1>+128",
             "gfx942"},
            {{{1, 2, 2, 1, {9, 1}, {1, 1}, {1, 0}, {3, 1}, {2, 1}},
              miopen::conv::Direction::BackwardWeights,
              miopenFloat,
              miopenTensorNHWC},
             "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64, 64, 64, 4, Default, 4, 2, 2, 1, 4, 1, 4, "
             "1, 1, 1>+1",
             "gfx942"}};
}

std::vector<KernelTuningNetTestCase> GetConvHipIgemmGroupWrwXdlopsTestCases_FP16()
{
    return {
        {{{32, 1024, 480, 64, {14, 14}, {1, 1}, {0, 0}, {1, 1}, {1, 1}},
          miopen::conv::Direction::BackwardWeights,
          miopenHalf,
          miopenTensorNHWC},
         "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<64, 16, 16, 32, Default, 8, 1, 1, 1, 4, "
         "1, 4, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>+1",
         "gfx942"},
        {{{1, 2, 2, 1, {9, 1}, {1, 1}, {1, 0}, {3, 1}, {2, 1}},
          miopen::conv::Direction::BackwardWeights,
          miopenHalf,
          miopenTensorNHWC},
         "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<64, 16, 16, 32, Default, 8, 1, 1, 1, 4, "
         "1, 4, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>+1",
         "gfx942"},
        {{{1, 16, 128, 256, {27, 27}, {3, 3}, {0, 0}, {1, 2}, {1, 1}},
          miopen::conv::Direction::BackwardWeights,
          miopenHalf,
          miopenTensorNHWC},
         "DeviceGroupedConvBwdWeight_Xdl_CShuffle<64, 64, 32, 4, Default, 8, 2, 1, 8, 4, 8, 2, 1, "
         "1, 8>+1",
         "gfx90a"}};
}

template <typename Solver>
class KernelTuningNetTest : public ::testing::TestWithParam<KernelTuningNetTestCase>
{
protected:
    void TestParameterPredictionModel()
    {
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
        auto test_case = GetParam();

        auto&& handle = get_handle();
        miopen::ExecutionContext ctx(&handle);

        if(test_case.arch != ctx.GetStream().GetDeviceName())
            GTEST_SKIP();

        auto input_tensor_desc = miopen::TensorDescriptor(
            test_case.data_type, test_case.layout, test_case.conv.GetInput());

        auto weights_tensor_desc = miopen::TensorDescriptor(
            test_case.data_type, test_case.layout, test_case.conv.GetWeights());

        auto conv_desc = test_case.conv.GetConv();

        auto output_desc = conv_desc.GetForwardOutputTensor(
            input_tensor_desc, weights_tensor_desc, test_case.data_type);

        auto problem = (test_case.direction == miopen::conv::Direction::Forward)
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

        auto data_type = test_case.data_type;
        auto in_tensor = GPUMem{0, input_tensor_desc.GetNumBytes() / sizeof(data_type), sizeof(data_type)};
        auto wt_tensor = GPUMem{0, weights_tensor_desc.GetNumBytes() / sizeof(data_type), sizeof(data_type)};
        auto out_tensor = GPUMem{0, output_desc.GetNumBytes() / sizeof(data_type), sizeof(data_type)};
        auto workSpaceSize = conv_desc.GetWorkSpaceSize(ctx, problem);
        auto workSpace = GPUMem{0, workSpaceSize / sizeof(data_type), sizeof(data_type)};

        miopen::AnyInvokeParams invoke_ctx;
        if(test_case.direction == miopen::conv::Direction::Forward)
            invoke_ctx = miopen::conv::DataInvokeParams{{input_tensor_desc, in_tensor.GetMem(), weights_tensor_desc, wt_tensor.GetMem(), output_desc, out_tensor.GetMem()}, 
                                                workSpace.GetMem(), workSpaceSize, conv_desc.attribute.gfx90aFp16alt.GetFwd()};
        else if(test_case.direction == miopen::conv::Direction::BackwardData)
            invoke_ctx = miopen::conv::DataInvokeParams{{output_desc, out_tensor.GetMem(), weights_tensor_desc, wt_tensor.GetMem(), input_tensor_desc, in_tensor.GetMem()},
                                               workSpace.GetMem(),
                                               workSpaceSize,
                                               conv_desc.attribute.gfx90aFp16alt.GetBwd()};
        else
            invoke_ctx = miopen::conv::WrWInvokeParams{{output_desc, out_tensor.GetMem(), input_tensor_desc, in_tensor.GetMem(), weights_tensor_desc, wt_tensor.GetMem()},
                                                  workSpace.GetMem(),
                                                  workSpaceSize,
                                                  conv_desc.attribute.gfx90aFp16alt.GetWrW()};

        Solver perf_config;
        ASSERT_TRUE(perf_config.IsModelApplicable(ctx, problem));

        perf_config.HeuristicInit(ctx, problem);

        std::vector<uint64_t> solvers;
        solvers = miopen::ai::immed_mode::PredictSolver(problem, ctx, test_case.arch);

        auto conv_sols = std::vector<miopenConvSolution_t>{};
        const auto ai_time = [](const int& idx) {
            return 10.0f * static_cast<float>(idx); // Assume idx == 1 (best solver) is 10 ms.
        };
        int idx = 1;
        for(const auto kinder : solvers)
        {
            const auto solver_id = miopen::solver::Id{kinder};
            const auto sol       = solver_id.GetSolver();
            const auto algo      = solver_id.GetAlgo();
            //MIOPEN_LOG_I2("Testing solver: " << solver_id.ToString());
            if(miopen::conv::IsAlgorithmDisabled(algo))
                continue;
            if(!sol.IsDynamic())
                continue; // branch should never be taken
            if(!sol.IsApplicable(ctx, problem))
                continue;
            const auto ws = sol.GetWorkspaceSize(ctx, problem);
            if(!miopen::conv::IsEnoughWorkspace("GetSolutionsFallback AI", solver_id, ws, &invoke_ctx))
                continue;
            conv_sols.emplace_back(
                miopenConvSolution_t{ai_time(idx), ws, solver_id.Value(), algo});
            ++idx;
        }

        ASSERT_FALSE(conv_sols.empty());

        miopen::PerformanceDb db = {miopen::DbKinds::PerfDb, fs::path{"/tmp"}, fs::path{"/tmp"}}; //empty db
        for(const auto& entry : conv_sols)
        {
            const auto id = miopen::solver::Id{entry.solution_id};
            const auto& s = id.GetSolver();
            //CompileSolution(id, ctx, problem);
            //results.push_back({id, sol->time, s.GetWorkspaceSize(ctx, problem)});
            MIOPEN_LOG_I(id.ToString()
                         << "\t" << entry.time << "\t" << s.GetWorkspaceSize(ctx, problem));

            miopen::solver::ConvSolution sol =
                s.FindSolution(ctx, problem, db, {}); // auto tune is not expected here
            const auto invoker = handle.PrepareInvoker(*sol.invoker_factory,
                                                   sol.construction_params);
            invoker(handle, invoke_ctx);
            MIOPEN_LOG_I(id.ToString() << " convolution completed");

        }

#else
        GTEST_SKIP();
#endif
    }
};

using GPU_KernelTuningNetTestConvAsm1x1U_FP32 =
    KernelTuningNetTest<miopen::solver::conv::PerformanceConfigConvAsm1x1U>;
using GPU_KernelTuningNetTestConvAsm1x1U_FP16 =
    KernelTuningNetTest<miopen::solver::conv::PerformanceConfigConvAsm1x1U>;

TEST_P(GPU_KernelTuningNetTestConvAsm1x1U_FP32, ConvAsm1x1UParameterPredictionModel)
{
    TestParameterPredictionModel();
}

TEST_P(GPU_KernelTuningNetTestConvAsm1x1U_FP16, ConvAsm1x1UParameterPredictionModel)
{
    TestParameterPredictionModel();
}

using GPU_KernelTuningNetTestConvHipIgemmGroupFwdXdlops_FP32 =
    KernelTuningNetTest<miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupFwdXdlops>;

using GPU_KernelTuningNetTestConvHipIgemmGroupFwdXdlops_FP16 =
    KernelTuningNetTest<miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupFwdXdlops>;

TEST_P(GPU_KernelTuningNetTestConvHipIgemmGroupFwdXdlops_FP32,
       ConvHipIgemmGroupFwdXdlopsParameterPredictionModel)
{
    TestParameterPredictionModel();
}

TEST_P(GPU_KernelTuningNetTestConvHipIgemmGroupFwdXdlops_FP16,
       ConvHipIgemmGroupFwdXdlopsParameterPredictionModel)
{
    TestParameterPredictionModel();
}

using GPU_KernelTuningNetTestConvHipIgemmGroupBwdXdlops_FP32 =
    KernelTuningNetTest<miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupBwdXdlops>;

using GPU_KernelTuningNetTestConvHipIgemmGroupBwdXdlops_FP16 =
    KernelTuningNetTest<miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupBwdXdlops>;

TEST_P(GPU_KernelTuningNetTestConvHipIgemmGroupBwdXdlops_FP32,
       ConvHipIgemmGroupBwdXdlopsParameterPredictionModel)
{
    TestParameterPredictionModel();
}

TEST_P(GPU_KernelTuningNetTestConvHipIgemmGroupBwdXdlops_FP16,
       ConvHipIgemmGroupBwdXdlopsParameterPredictionModel)
{
    TestParameterPredictionModel();
}

using GPU_KernelTuningNetTestConvHipIgemmGroupWrwXdlops_FP32 =
    KernelTuningNetTest<miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupWrwXdlops>;

using GPU_KernelTuningNetTestConvHipIgemmGroupWrwXdlops_FP16 =
    KernelTuningNetTest<miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupWrwXdlops>;

TEST_P(GPU_KernelTuningNetTestConvHipIgemmGroupWrwXdlops_FP32,
       ConvHipIgemmGroupWrwXdlopsParameterPredictionModel)
{
    TestParameterPredictionModel();
}

TEST_P(GPU_KernelTuningNetTestConvHipIgemmGroupWrwXdlops_FP16,
       ConvHipIgemmGroupWrwXdlopsParameterPredictionModel)
{
    TestParameterPredictionModel();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GPU_KernelTuningNetTestConvAsm1x1U_FP32);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GPU_KernelTuningNetTestConvAsm1x1U_FP16);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    GPU_KernelTuningNetTestConvHipIgemmGroupFwdXdlops_FP32);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    GPU_KernelTuningNetTestConvHipIgemmGroupFwdXdlops_FP16);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    GPU_KernelTuningNetTestConvHipIgemmGroupBwdXdlops_FP32);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    GPU_KernelTuningNetTestConvHipIgemmGroupBwdXdlops_FP16);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    GPU_KernelTuningNetTestConvHipIgemmGroupWrwXdlops_FP32);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    GPU_KernelTuningNetTestConvHipIgemmGroupWrwXdlops_FP16);

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_KernelTuningNetTestConvAsm1x1U_FP32,
                         testing::ValuesIn(GetConvAsm1x1UTestCases_FP32()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_KernelTuningNetTestConvAsm1x1U_FP16,
                         testing::ValuesIn(GetConvAsm1x1UTestCases_FP16()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_KernelTuningNetTestConvHipIgemmGroupFwdXdlops_FP32,
                         testing::ValuesIn(GetConvHipIgemmGroupFwdXdlopsTestCases_FP32()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_KernelTuningNetTestConvHipIgemmGroupFwdXdlops_FP16,
                         testing::ValuesIn(GetConvHipIgemmGroupFwdXdlopsTestCases_FP16()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_KernelTuningNetTestConvHipIgemmGroupBwdXdlops_FP32,
                         testing::ValuesIn(GetConvHipIgemmGroupBwdXdlopsTestCases_FP32()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_KernelTuningNetTestConvHipIgemmGroupBwdXdlops_FP16,
                         testing::ValuesIn(GetConvHipIgemmGroupBwdXdlopsTestCases_FP16()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_KernelTuningNetTestConvHipIgemmGroupWrwXdlops_FP32,
                         testing::ValuesIn(GetConvHipIgemmGroupWrwXdlopsTestCases_FP32()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_KernelTuningNetTestConvHipIgemmGroupWrwXdlops_FP16,
                         testing::ValuesIn(GetConvHipIgemmGroupWrwXdlopsTestCases_FP16()));
