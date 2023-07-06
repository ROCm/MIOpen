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

#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>

#include "na.hpp"
#include "bn.hpp"

struct BNActivInferFloat : BNActivInferTest<float>
{
};

struct BNActivInferHalf : BNActivInferTest<half_float::half>
{
};

struct BNInferFloat : BNInferTest<float>
{
};

struct BNInferHalf : BNInferTest<half_float::half>
{
};

template <typename Solver, typename TestCase>
void RunSolver(miopen::FusionPlanDescriptor& fusePlanDesc,
               const miopen::fusion::FusionInvokeParams& plan_params,
               const TestCase& config,
               bool& test_skipped)
{
    auto& handle = get_handle();
    Solver solv{};
    const auto fusion_problem = miopen::FusionDescription{&fusePlanDesc};
    auto fusion_ctx           = miopen::FusionContext{handle};
    fusion_ctx.DetectRocm();
    if(!solv.IsApplicable(fusion_ctx, fusion_problem))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << config;
    }
    ASSERT_TRUE(solv.IsApplicable(fusion_ctx, fusion_problem));
    auto sol = solv.GetSolution(fusion_ctx, fusion_problem);
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, plan_params);
    handle.Finish();
}

template <typename Solver, typename TestCase>
void RunTunableSolver(miopen::FusionPlanDescriptor& fusePlanDesc,
                      const std::unique_ptr<miopen::fusion::FusionInvokeParams>& plan_params,
                      const TestCase& config,
                      bool& test_skipped)
{
    auto& handle = get_handle();
    Solver solv{};
    const auto fusion_problem = miopen::FusionDescription{&fusePlanDesc};
    auto fusion_ctx           = miopen::FusionContext{handle};
    fusion_ctx.DetectRocm();
    if(!solv.IsApplicable(fusion_ctx, fusion_problem))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << config;
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

TEST_P(BNActivInferFloat, BnFwdInferActivationFused)
{
    const auto plan_params = miopen::fusion::FusionInvokeParams(params,
                                                                bn_infer_data.input.desc,
                                                                bn_infer_data.in_dev.get(),
                                                                bn_infer_data.output.desc,
                                                                bn_infer_data.out_dev.get(),
                                                                false);
    RunSolver<miopen::solver::fusion::BnFwdInferActivationFused>(
        fusePlanDesc, plan_params, bn_config, test_skipped);
}

TEST_P(BNInferFloat, CKBnFwdInference)
{
    const auto plan_params =
        std::make_unique<miopen::fusion::FusionInvokeParams>(params,
                                                             bn_infer_data.input.desc,
                                                             bn_infer_data.in_dev.get(),
                                                             bn_infer_data.output.desc,
                                                             bn_infer_data.out_dev.get(),
                                                             false);
    RunTunableSolver<miopen::solver::fusion::CKBnFwdInference>(
        fusePlanDesc, plan_params, bn_config, test_skipped);
}

INSTANTIATE_TEST_SUITE_P(BNInferFloatSuite,
                         BNInferFloat,
                         testing::Combine(testing::ValuesIn(Network1()),
                                          testing::Values(miopenTensorNHWC)));

INSTANTIATE_TEST_SUITE_P(BNActivInferFloatSuite,
                         BNActivInferFloat,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(Network1()),
                                          testing::Values(miopenTensorNCHW)));

TEST_P(BNActivInferHalf, DISABLED_BnFwdInferActivationFused)
{
    const auto plan_params = miopen::fusion::FusionInvokeParams(params,
                                                                bn_infer_data.input.desc,
                                                                bn_infer_data.in_dev.get(),
                                                                bn_infer_data.output.desc,
                                                                bn_infer_data.out_dev.get(),
                                                                false);
    RunSolver<miopen::solver::fusion::BnFwdInferActivationFused>(
        fusePlanDesc, plan_params, bn_config, test_skipped);
}

INSTANTIATE_TEST_SUITE_P(BNActivInferHalfSuite,
                         BNActivInferHalf,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(Network1()),
                                          testing::Values(miopenTensorNHWC)));
