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
#include "cba.hpp"

struct ConvBiasActivInferTestFloat : ConvBiasActivInferTest<float>
{
};

TEST_P(ConvBiasActivInferTestFloat, SolverTest)
{
    auto& handle            = get_handle();
    const float alpha       = static_cast<float>(1.0f);
    const float beta        = static_cast<float>(0);
    const float activ_alpha = static_cast<double>(0.5f);
    const float activ_beta  = static_cast<double>(0.5f);
    const float activ_gamma = static_cast<double>(0.5f);
    miopen::FusionPlanDescriptor fusePlanDesc(miopenVerticalFusion, input.desc);
    auto convOp  = std::make_shared<miopen::ConvForwardOpDescriptor>(conv_desc, weights.desc);
    auto biasOp  = std::make_shared<miopen::BiasFusionOpDescriptor>(bias.desc);
    auto activOp = std::make_shared<miopen::ActivFwdFusionOpDescriptor>(activ_desc.GetMode());
    EXPECT_EQ(fusePlanDesc.AddOp(convOp), miopenStatusSuccess);
    convOp->SetArgs(&alpha, &beta, wei_dev.get());
    EXPECT_EQ(fusePlanDesc.AddOp(biasOp), miopenStatusSuccess);
    biasOp->SetArgs(&alpha, &beta, bias_dev.get());
    EXPECT_EQ(fusePlanDesc.AddOp(activOp), miopenStatusSuccess);
    activOp->SetArgs(&alpha, &beta, activ_alpha, activ_beta, activ_gamma);
    miopen::solver::fusion::ConvBiasActivAsm1x1U solv{};

    auto fusion_ctx = miopen::FusionContext{&fusePlanDesc, handle};
    fusion_ctx.DetectRocm();
    if(!solv.IsApplicable(fusion_ctx))
    {
        test_skipped = true;
        GTEST_SKIP() << "ConvBiasActivAsm1x1U Not Applicable" << conv_config;
    }
    ASSERT_TRUE(solv.IsApplicable(fusion_ctx));
    auto sol = solv.GetSolution(fusion_ctx, solv.GetDefaultPerformanceConfig(fusion_ctx));
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    std::vector<std::shared_ptr<miopen::fusion::FusionOpInvokeParamBase>> params;
    for(const auto& op : fusePlanDesc.op_map)
        params.push_back(op->GetArgs());
    const auto plan_params = miopen::fusion::FusionInvokeParams{
        params, input.desc, in_dev.get(), output.desc, out_dev.get(), false};
    (invoker)(handle, plan_params);
    handle.Finish();
}

INSTANTIATE_TEST_SUITE_P(CBAInferSolverTest,
                         ConvBiasActivInferTestFloat,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1())));
