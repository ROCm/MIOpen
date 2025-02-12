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
#include <gtest/gtest_common.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/miopen.h>
#include <miopen/search_options.hpp>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>
#include <fusionHost.hpp>
#include <miopen/fusion.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/fusion/fusion_invoke_params.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "cba_find2.hpp"

namespace cba_find2_infer {

bool IsTestSupportedForDevice()
{
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    // gfx120X is not enabled due to WORKAROUND_SWDEV_479810
    using d_mask = disabled<Gpu::None>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
}

struct GPU_ConvBiasActivFind2Infer_FP32 : ConvBiasActivInferFind2Test<float>
{
};

struct GPU_ConvBiasActivFind2InferFusionFind_FP32 : ConvBiasActivInferFind2Test<float>
{
};

struct GPU_ConvBiasActivFind2Infer_FP16 : ConvBiasActivInferFind2Test<half_float::half>
{
};

template <typename Solver, typename TestCase>
void RunSolver(miopen::FusedProblem& problem,
               const miopen::AnyInvokeParams& invoke_ctx,
               const TestCase& conv_config,
               bool& test_skipped)
{
    auto& handle = get_handle();
    Solver solv{};
    const auto plan        = problem.AsFusionPlan();
    const auto fusion_desc = miopen::FusionDescription{&plan};
    auto fusion_ctx        = miopen::FusionContext{handle};
    if(!solv.IsApplicable(fusion_ctx, fusion_desc))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << conv_config;
    }
    ASSERT_TRUE(solv.IsApplicable(fusion_ctx, fusion_desc));
    auto sol = solv.GetSolution(fusion_ctx, fusion_desc);
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_ctx);
    handle.Finish();
}

template <typename Solver>
void RunTunableSolver(miopen::FusedProblem& problem,
                      const miopen::AnyInvokeParams& invoke_ctx,
                      const ConvTestCaseBase& conv_config,
                      bool& test_skipped)
{
    auto& handle = get_handle();
    Solver solv{};
    const auto plan        = problem.AsFusionPlan();
    const auto fusion_desc = miopen::FusionDescription{&plan};
    auto fusion_ctx        = miopen::FusionContext{handle};
    if(!solv.IsApplicable(fusion_ctx, fusion_desc))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << conv_config;
    }
    ASSERT_TRUE(solv.IsApplicable(fusion_ctx, fusion_desc));
    auto sol = solv.GetSolution(
        fusion_ctx, fusion_desc, solv.GetDefaultPerformanceConfig(fusion_ctx, fusion_desc));
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_ctx);
    handle.Finish();
}

bool SkipTest() { return get_handle_xnack(); }

} // namespace cba_find2_infer
using namespace cba_find2_infer;

TEST_P(GPU_ConvBiasActivFind2Infer_FP32, ConvBiasActivAsm1x1UFind2Float)
{
    RunTunableSolver<miopen::solver::fusion::ConvBiasActivAsm1x1U>(
        fused_problem, invoke_params, conv_config, test_skipped);
}
TEST_P(GPU_ConvBiasActivFind2Infer_FP32, ConvOclDirectFwdFind2Fused)
{
    RunTunableSolver<miopen::solver::fusion::ConvOclDirectFwdFused>(
        fused_problem, invoke_params, conv_config, test_skipped);
}
TEST_P(GPU_ConvBiasActivFind2Infer_FP32, ConvBinWinogradRxSFind2Fused)
{
    RunSolver<miopen::solver::fusion::ConvBinWinogradRxSFused>(
        fused_problem, invoke_params, conv_config, test_skipped);
}
TEST_P(GPU_ConvBiasActivFind2Infer_FP32, ConvBinWinogradRxSf2x3g1Find2Fused)
{
    RunSolver<miopen::solver::fusion::ConvBinWinogradRxSf2x3g1Fused>(
        fused_problem, invoke_params, conv_config, test_skipped);
}
TEST_P(GPU_ConvBiasActivFind2Infer_FP16, ConvWinoFuryRxSf2x3Find2Fused)
{
    RunSolver<miopen::solver::fusion::ConvWinoFuryRxSFused<2, 3>>(
        fused_problem, invoke_params, conv_config, test_skipped);
}

TEST_P(GPU_ConvBiasActivFind2Infer_FP16, ConvCKIgemmFwdBiasActivFind2Fused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmFwdBiasActivFused>(
        fused_problem, invoke_params, conv_config, test_skipped);
}

#if MIOPEN_BACKEND_HIP
TEST_P(GPU_ConvBiasActivFind2InferFusionFind_FP32, ConvBiasActivFind2Float_testFind)
{
    if(SkipTest())
    {
        test_skipped = true;
        GTEST_SKIP() << "Fusion does not support xnack";
    }
    if(!IsTestSupportedForDevice())
    {
        test_skipped = true;
        GTEST_SKIP() << "Fusion not supported for this device";
    }
    miopen::solver::debug::TuningIterationScopedLimiter tuning_limit{5};

    std::vector<miopen::Solution> solutions;
    auto options         = miopen::FindOptions{};
    options.find_enforce = miopen::FindEnforce{miopen::FindEnforceAction::SearchDbUpdate};

    ASSERT_NO_THROW(solutions = fused_problem.FindSolutions(get_handle(), options, 10));

    auto tensors = std::unordered_map<miopenTensorArgumentId_t, miopen::Solution::RunInput>{
        {miopenTensorConvolutionX, in_dev.get()},
        {miopenTensorConvolutionW, wei_dev.get()},
        {miopenTensorActivationY, out_dev.get()},
        {miopenTensorBias, bias_dev.get()},
    };

    for(auto& solution : solutions)
    {
        ASSERT_NO_THROW(solution.Run(get_handle(), tensors, nullptr, 0));
        ValidateResult();
    }
}

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvBiasActivFind2InferFusionFind_FP32,
    testing::Combine(testing::Values(miopenActivationRELU),
                     testing::ValuesIn(GetNetworkForFusionCompileStepTest<ConvTestCaseBase>()),
                     testing::Values(miopenTensorNCHW)));

#endif

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvBiasActivFind2Infer_FP32,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenTensorNCHW)));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvBiasActivFind2Infer_FP16,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenTensorNHWC)));
