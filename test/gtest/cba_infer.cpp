/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>
#include <fusionHost.hpp>
#include <miopen/fusion.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>
#include <miopen/env.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "cba.hpp"

namespace env = miopen::env;

namespace cba_infer {

struct GPU_ConvBiasActivInfer_FP32 : ConvBiasActivInferTest<float>
{
};

struct GPU_ConvBiasActivInferFusionCompileStep_FP32 : ConvBiasActivInferTest<float>
{
};

struct GPU_ConvBiasActivInfer_FP16 : ConvBiasActivInferTest<half_float::half>
{
};

template <typename Solver, typename TestCase>
void RunSolver(miopen::FusionPlanDescriptor& fusePlanDesc,
               const std::unique_ptr<miopen::fusion::FusionInvokeParams>& plan_params,
               const TestCase& conv_config,
               bool& test_skipped)
{
    auto& handle = get_handle();
    Solver solv{};
    const auto fusion_problem = miopen::FusionDescription{&fusePlanDesc};
    auto fusion_ctx           = miopen::FusionContext{handle};
    if(!solv.IsApplicable(fusion_ctx, fusion_problem))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << conv_config;
    }
    ASSERT_TRUE(solv.IsApplicable(fusion_ctx, fusion_problem));
    auto sol = solv.GetSolution(fusion_ctx, fusion_problem);
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, *(plan_params.get()));
    handle.Finish();
}
template <typename Solver>
void RunTunableSolver(miopen::FusionPlanDescriptor& fusePlanDesc,
                      const std::unique_ptr<miopen::fusion::FusionInvokeParams>& plan_params,
                      const ConvTestCaseBase& conv_config,
                      bool& test_skipped)
{
    auto& handle = get_handle();
    Solver solv{};
    const auto fusion_problem = miopen::FusionDescription{&fusePlanDesc};
    auto fusion_ctx           = miopen::FusionContext{handle};
    if(!solv.IsApplicable(fusion_ctx, fusion_problem))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << conv_config;
    }
    ASSERT_TRUE(solv.IsApplicable(fusion_ctx, fusion_problem));
    auto sol = solv.GetSolution(
        fusion_ctx, fusion_problem, solv.GetDefaultPerformanceConfig(fusion_ctx, fusion_problem));
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, *(plan_params.get()));
    handle.Finish();
}
} // namespace cba_infer
using namespace cba_infer;

TEST_P(GPU_ConvBiasActivInfer_FP32, ConvBiasActivAsm1x1UFloat)
{
    const auto plan_params = std::make_unique<miopen::fusion::FusionInvokeParams>(
        params, input.desc, in_dev.get(), output.desc, out_dev.get(), false);
    RunTunableSolver<miopen::solver::fusion::ConvBiasActivAsm1x1U>(
        fusePlanDesc, plan_params, conv_config, test_skipped);
}
TEST_P(GPU_ConvBiasActivInfer_FP32, ConvOclDirectFwdFused)
{
    const auto plan_params = std::make_unique<miopen::fusion::FusionInvokeParams>(
        params, input.desc, in_dev.get(), output.desc, out_dev.get(), false);
    RunTunableSolver<miopen::solver::fusion::ConvOclDirectFwdFused>(
        fusePlanDesc, plan_params, conv_config, test_skipped);
}
TEST_P(GPU_ConvBiasActivInfer_FP32, ConvBinWinogradRxSFused)
{
    const auto plan_params = std::make_unique<miopen::fusion::FusionInvokeParams>(
        params, input.desc, in_dev.get(), output.desc, out_dev.get(), false);
    RunSolver<miopen::solver::fusion::ConvBinWinogradRxSFused>(
        fusePlanDesc, plan_params, conv_config, test_skipped);
}
TEST_P(GPU_ConvBiasActivInfer_FP32, ConvBinWinogradRxSf2x3g1Fused)
{
    const auto plan_params = std::make_unique<miopen::fusion::FusionInvokeParams>(
        params, input.desc, in_dev.get(), output.desc, out_dev.get(), false);
    RunSolver<miopen::solver::fusion::ConvBinWinogradRxSf2x3g1Fused>(
        fusePlanDesc, plan_params, conv_config, test_skipped);
}
TEST_P(GPU_ConvBiasActivInfer_FP16, ConvWinoFuryRxSf2x3Fused)
{
    const auto plan_params = std::make_unique<miopen::fusion::FusionInvokeParams>(
        params, input.desc, in_dev.get(), output.desc, out_dev.get(), false);
    RunSolver<miopen::solver::fusion::ConvWinoFuryRxSFused<2, 3>>(
        fusePlanDesc, plan_params, conv_config, test_skipped);
}

TEST_P(GPU_ConvBiasActivInfer_FP16, ConvCKIgemmFwdBiasActivFused)
{
    const auto plan_params = std::make_unique<miopen::fusion::FusionInvokeParams>(
        params, input.desc, in_dev.get(), output.desc, out_dev.get(), false);
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmFwdBiasActivFused>(
        fusePlanDesc, plan_params, conv_config, test_skipped);
}

#if MIOPEN_BACKEND_HIP

TEST_P(GPU_ConvBiasActivInferFusionCompileStep_FP32, ConvBiasActivAsm1x1UFloat_testCompile)
{
    env::setEnvironmentVariable("MIOPEN_FIND_ENFORCE", "SEARCH_DB_UPDATE");
    env::update(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 5);
    fusePlanDesc.Compile(get_handle());
    const auto plan_params = std::make_unique<miopen::fusion::FusionInvokeParams>(
        params, input.desc, in_dev.get(), output.desc, out_dev.get(), false);
    RunTunableSolver<miopen::solver::fusion::ConvBiasActivAsm1x1U>(
        fusePlanDesc, plan_params, conv_config, test_skipped);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvBiasActivInferFusionCompileStep_FP32,
    testing::Combine(testing::Values(miopenActivationRELU),
                     testing::ValuesIn(GetNetworkForFusionCompileStepTest<ConvTestCaseBase>()),
                     testing::Values(miopenTensorNCHW)));

#endif

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvBiasActivInfer_FP32,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenTensorNCHW)));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvBiasActivInfer_FP16,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenTensorNHWC)));
