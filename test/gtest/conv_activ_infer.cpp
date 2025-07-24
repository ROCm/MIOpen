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
#include <half/half.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "conv_activ.hpp"

namespace {

using float16 = half_float::half;

template <typename T, typename TestCaseType>
struct CAInferBase : ConvActivInferTest<T, TestCaseType>
{
    void RunSolver(const miopen::solver::fusion::FusionSolverBase& solv)
    {
        auto& handle              = get_handle();
        const auto fusion_problem = miopen::FusionDescription{&this->fusePlanDesc};
        auto fusion_ctx           = miopen::FusionContext{handle};
        if(!solv.IsApplicable(fusion_ctx, fusion_problem))
        {
            this->test_skipped = true;
            GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << this->conv_config;
        }
        ASSERT_TRUE(solv.IsApplicable(fusion_ctx, fusion_problem));
        auto sol = solv.GetSolution(fusion_ctx, fusion_problem);
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);

        const auto plan_params =
            std::make_unique<miopen::fusion::FusionInvokeParams>(this->params,
                                                                 this->input.desc,
                                                                 this->in_dev.get(),
                                                                 this->output.desc,
                                                                 this->out_dev.get(),
                                                                 false);

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        (invoker)(handle, *(plan_params.get()));
        handle.Finish();
    }

    std::unique_ptr<miopen::fusion::FusionInvokeParams> createFusionInvokeParams(
        const miopen::FusionDescription& fusion_desc,
        const miopen::FusionContext& fusion_ctx,
        const miopen::solver::SolverInterfaceTunable<miopen::FusionContext,
                                                     miopen::FusionDescription>& solv,
        bool useWorkspace = false)
    {
        if(useWorkspace)
        {
            this->wspace.resize(solv.GetWorkspaceSize(fusion_ctx, fusion_desc));

            return std::make_unique<miopen::fusion::FusionInvokeParams>(this->params,
                                                                        this->input.desc,
                                                                        this->in_dev.get(),
                                                                        this->output.desc,
                                                                        this->out_dev.get(),
                                                                        false,
                                                                        this->wspace.ptr(),
                                                                        this->wspace.size());
        }
        else
        {
            return std::make_unique<miopen::fusion::FusionInvokeParams>(this->params,
                                                                        this->input.desc,
                                                                        this->in_dev.get(),
                                                                        this->output.desc,
                                                                        this->out_dev.get(),
                                                                        false);
        }
    }

    // Have to keep it a template besause of GetDefaultPerformanceConfig() call
    template <typename Solver>
    void RunTunableSolver()
    {
        auto& handle = get_handle();
        Solver solv{};
        const auto fusion_problem = miopen::FusionDescription{&this->fusePlanDesc};
        auto fusion_ctx           = miopen::FusionContext{handle};
        if(!solv.IsApplicable(fusion_ctx, fusion_problem))
        {
            this->test_skipped = true;
            GTEST_SKIP() << solv.SolverDbId() << " Not Applicable" << this->conv_config;
        }
        ASSERT_TRUE(solv.IsApplicable(fusion_ctx, fusion_problem));
        auto sol = solv.GetSolution(fusion_ctx,
                                    fusion_problem,
                                    solv.GetDefaultPerformanceConfig(fusion_ctx, fusion_problem));
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);

        auto plan_params =
            createFusionInvokeParams(fusion_problem, fusion_ctx, solv, solv.MayNeedWorkspace());

        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        (invoker)(handle, *(plan_params.get()));
        handle.Finish();
    }
};

template <typename Configs, typename TensorTypes>
inline auto gcaInferParamGenSmoke(Configs configs, TensorTypes tensorTypes)
{
    return ::testing::Combine(testing::Values(miopenActivationRELU, miopenActivationCLIPPEDRELU),
                              testing::ValuesIn(configs),
                              tensorTypes,
                              testing::Values(0.5f),
                              testing::Values(1.0f),
                              testing::Values(0.5f));
}

template <typename Configs, typename TensorTypes>
inline auto gcaInferParamGenFull(Configs configs, TensorTypes tensorTypes)
{
    return ::testing::Combine(testing::Values(miopenActivationCLAMP),
                              testing::ValuesIn(configs),
                              tensorTypes,
                              testing::Values(0.5f),
                              testing::Values(1.0f),
                              testing::Values(0.5f));
}

using GPU_ConvGrpActivInfer_BFP16 = CAInferBase<bfloat16, GroupConvTestConfig<2u>>;
using GPU_ConvGrpActivInfer_FP16  = CAInferBase<float16, GroupConvTestConfig<2u>>;
using GPU_ConvGrpActivInfer_FP32  = CAInferBase<float, GroupConvTestConfig<2u>>;

using GPU_ConvGrpActivInfer3D_BFP16 = CAInferBase<bfloat16, GroupConvTestConfig<3u>>;
using GPU_ConvGrpActivInfer3D_FP16  = CAInferBase<float16, GroupConvTestConfig<3u>>;
using GPU_ConvGrpActivInfer3D_FP32  = CAInferBase<float, GroupConvTestConfig<3u>>;

} // namespace

TEST_P(GPU_ConvGrpActivInfer_BFP16, ConvCKIgemmGrpFwdActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdActivFused>();
};
TEST_P(GPU_ConvGrpActivInfer3D_BFP16, ConvCKIgemmGrpFwdActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdActivFused>();
};
TEST_P(GPU_ConvGrpActivInfer_FP16, ConvCKIgemmGrpFwdActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdActivFused>();
};
TEST_P(GPU_ConvGrpActivInfer3D_FP16, ConvCKIgemmGrpFwdActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdActivFused>();
};
TEST_P(GPU_ConvGrpActivInfer_FP32, ConvCKIgemmGrpFwdActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdActivFused>();
};
TEST_P(GPU_ConvGrpActivInfer3D_FP32, ConvCKIgemmGrpFwdActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdActivFused>();
};

// Instantiate test suites for BFP16
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpActivInfer_BFP16,
    gcaInferParamGenSmoke(GroupConvTestConfig<2>::GetSmokeConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNHWC, miopenTensorNCHW)));
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpActivInfer3D_BFP16,
    gcaInferParamGenSmoke(GroupConvTestConfig<3>::GetSmokeConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpActivInfer_BFP16,
    gcaInferParamGenFull(GroupConvTestConfig<2>::GetConfigs<Direction::Forward>(),
                         testing::Values(miopenTensorNHWC, miopenTensorNCHW)));
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpActivInfer3D_BFP16,
    gcaInferParamGenFull(GroupConvTestConfig<3>::GetConfigs<Direction::Forward>(),
                         testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)));

// Instantiate test suites for FP16
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpActivInfer_FP16,
    gcaInferParamGenSmoke(GroupConvTestConfig<2>::GetSmokeConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNHWC, miopenTensorNCHW)));
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpActivInfer3D_FP16,
    gcaInferParamGenSmoke(GroupConvTestConfig<3>::GetSmokeConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpActivInfer_FP16,
    gcaInferParamGenFull(GroupConvTestConfig<2>::GetConfigs<Direction::Forward>(),
                         testing::Values(miopenTensorNHWC, miopenTensorNCHW)));
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpActivInfer3D_FP16,
    gcaInferParamGenFull(GroupConvTestConfig<3>::GetConfigs<Direction::Forward>(),
                         testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)));

// Instantiate test suites for FP32
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpActivInfer_FP32,
    gcaInferParamGenSmoke(GroupConvTestConfig<2>::GetSmokeConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNHWC, miopenTensorNCHW)));
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpActivInfer3D_FP32,
    gcaInferParamGenSmoke(GroupConvTestConfig<3>::GetSmokeConfigs<Direction::Forward>(),
                          testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpActivInfer_FP32,
    gcaInferParamGenFull(GroupConvTestConfig<2>::GetConfigs<Direction::Forward>(),
                         testing::Values(miopenTensorNHWC, miopenTensorNCHW)));
INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpActivInfer3D_FP32,
    gcaInferParamGenFull(GroupConvTestConfig<3>::GetConfigs<Direction::Forward>(),
                         testing::Values(miopenTensorNDHWC, miopenTensorNCDHW)));
