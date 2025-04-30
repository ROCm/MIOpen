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

struct GPU_BNActivInfer_FP32 : BNActivInferTest<float>
{
};

struct GPU_BNActivInfer_FP16 : BNActivInferTest<half_float::half>
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
TEST_P(GPU_BNActivInfer_FP32, BnFwdInferActivationFused)
{
    const auto plan_params = miopen::fusion::FusionInvokeParams(
        params, input.desc, in_dev.get(), output.desc, out_dev.get(), false);
    RunSolver<miopen::solver::fusion::BnFwdInferActivationFused>(
        fusePlanDesc, plan_params, bn_config, test_skipped);
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BNActivInfer_FP32,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(Networkna1())));
TEST_P(GPU_BNActivInfer_FP16, DISABLED_BnFwdInferActivationFused)
{
    const auto plan_params = miopen::fusion::FusionInvokeParams(
        params, input.desc, in_dev.get(), output.desc, out_dev.get(), false);
    RunSolver<miopen::solver::fusion::BnFwdInferActivationFused>(
        fusePlanDesc, plan_params, bn_config, test_skipped);
}

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BNActivInfer_FP16,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(Networkna1())));
