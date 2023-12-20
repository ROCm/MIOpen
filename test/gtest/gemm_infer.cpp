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

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "gemm_test.hpp"

struct GemmTestHalf : GemmTest<half_float::half>
{
};
//

template <typename Solver>
void RunTunableSolver(miopen::FusionPlanDescriptor& fusePlanDesc,
                      const std::unique_ptr<miopen::fusion::FusionInvokeParams>& plan_params,
                      const GemmTestCase& gemm_config,
                      bool& test_skipped)
{
    auto& handle = get_handle();
    Solver solv{};
    const auto fusion_problem = miopen::FusionDescription{&fusePlanDesc};
    auto fusion_ctx           = miopen::FusionContext{handle};
    if(!solv.IsApplicable(fusion_ctx, fusion_problem))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << gemm_config;
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

TEST_P(GemmTestHalf, CKGEMMAddActiv)
{
    const auto plan_params = std::make_unique<miopen::fusion::FusionInvokeParams>(
        params, A_tensor.desc, a_dev.get(), C_tensor.desc, c_dev.get(), false);

    RunTunableSolver<miopen::solver::fusion::CKGEMMAddActiv>(
        fusePlanDesc, plan_params, gemm_config, test_skipped);
}

INSTANTIATE_TEST_SUITE_P(GemmSolverTest,
                         GemmTestHalf,
                         testing::Combine(testing::Values(miopenActivationFGELU),
                                          testing::ValuesIn(GetTestData()),
                                          testing::Values(miopenTensorRowMajor)));
