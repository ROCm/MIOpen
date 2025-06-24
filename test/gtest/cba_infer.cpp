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
#include <miopen/conv_algo_name.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "cba.hpp"

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

struct GPU_ConvGrpBiasActivInfer_BFP16 : ConvBiasActivInferTest<bfloat16, GroupConvTestConfig<2u>>
{
};

struct GPU_ConvGrpBiasActivInfer3D_BFP16 : ConvBiasActivInferTest<bfloat16, GroupConvTestConfig<3u>>
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
std::unique_ptr<miopen::fusion::FusionInvokeParams>
createFusionInvokeParams(const miopen::FusionDescription& fusion_desc,
                         const miopen::FusionContext& fusion_ctx,
                         const miopen::OperatorArgs& params,
                         const Solver& solv,
                         const miopen::TensorDescriptor& input_desc,
                         ConstData_t in_dev_ptr,
                         const miopen::TensorDescriptor& output_desc,
                         Data_t out_dev_ptr,
                         Workspace& wspace,
                         bool useWorkspace = false)
{
    if(useWorkspace)
    {
        wspace.resize(solv.GetWorkspaceSize(fusion_ctx, fusion_desc));

        return std::make_unique<miopen::fusion::FusionInvokeParams>(params,
                                                                    input_desc,
                                                                    in_dev_ptr,
                                                                    output_desc,
                                                                    out_dev_ptr,
                                                                    false,
                                                                    wspace.ptr(),
                                                                    wspace.size());
    }
    else
    {
        return std::make_unique<miopen::fusion::FusionInvokeParams>(
            params, input_desc, in_dev_ptr, output_desc, out_dev_ptr, false);
    }
}

template <typename Solver, typename TCase = ConvTestCaseBase>
void RunTunableSolver(miopen::FusionPlanDescriptor& fusePlanDesc,
                      const miopen::OperatorArgs& params,
                      const TCase& conv_config,
                      bool& test_skipped,
                      const miopen::TensorDescriptor& input_desc,
                      ConstData_t in_dev_ptr,
                      const miopen::TensorDescriptor& output_desc,
                      Data_t out_dev_ptr,
                      Workspace& wspace)
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

    auto plan_params = createFusionInvokeParams<Solver>(fusion_problem,
                                                        fusion_ctx,
                                                        params,
                                                        solv,
                                                        input_desc,
                                                        in_dev_ptr,
                                                        output_desc,
                                                        out_dev_ptr,
                                                        wspace,
                                                        solv.MayNeedWorkspace());

    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, *(plan_params.get()));
    handle.Finish();
}

} // namespace cba_infer
using namespace cba_infer;

TEST_P(GPU_ConvBiasActivInfer_FP32, ConvBiasActivAsm1x1UFloat)
{
    RunTunableSolver<miopen::solver::fusion::ConvBiasActivAsm1x1U>(fusePlanDesc,
                                                                   params,
                                                                   conv_config,
                                                                   test_skipped,
                                                                   input.desc,
                                                                   in_dev.get(),
                                                                   output.desc,
                                                                   out_dev.get(),
                                                                   wspace);
}
TEST_P(GPU_ConvBiasActivInfer_FP32, ConvOclDirectFwdFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvOclDirectFwdFused>(fusePlanDesc,
                                                                    params,
                                                                    conv_config,
                                                                    test_skipped,
                                                                    input.desc,
                                                                    in_dev.get(),
                                                                    output.desc,
                                                                    out_dev.get(),
                                                                    wspace);
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
TEST_P(GPU_ConvBiasActivInfer_FP16, ConvWinoRageRxSf2x3Fused)
{
    const auto plan_params = std::make_unique<miopen::fusion::FusionInvokeParams>(
        params, input.desc, in_dev.get(), output.desc, out_dev.get(), false);
    RunSolver<miopen::solver::fusion::ConvWinoRageRxSFused<2, 3>>(
        fusePlanDesc, plan_params, conv_config, test_skipped);
}

TEST_P(GPU_ConvBiasActivInfer_FP16, ConvCKIgemmFwdBiasActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmFwdBiasActivFused>(fusePlanDesc,
                                                                           params,
                                                                           conv_config,
                                                                           test_skipped,
                                                                           input.desc,
                                                                           in_dev.get(),
                                                                           output.desc,
                                                                           out_dev.get(),
                                                                           wspace);
}

TEST_P(GPU_ConvGrpBiasActivInfer_BFP16, ConvCKIgemmGrpFwdBiasActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdBiasActivFused>(fusePlanDesc,
                                                                              params,
                                                                              conv_config,
                                                                              test_skipped,
                                                                              input.desc,
                                                                              in_dev.get(),
                                                                              output.desc,
                                                                              out_dev.get(),
                                                                              wspace);
}

TEST_P(GPU_ConvGrpBiasActivInfer3D_BFP16, ConvCKIgemmGrpFwdBiasActiv3DFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdBiasActivFused>(fusePlanDesc,
                                                                              params,
                                                                              conv_config,
                                                                              test_skipped,
                                                                              input.desc,
                                                                              in_dev.get(),
                                                                              output.desc,
                                                                              out_dev.get(),
                                                                              wspace);
}

#if MIOPEN_BACKEND_HIP

TEST_P(GPU_ConvBiasActivInferFusionCompileStep_FP32, ConvBiasActivAsm1x1UFloat_testCompile)
{
    ScopedEnvironment<std::string> find_enforce_env(MIOPEN_FIND_ENFORCE, "SEARCH_DB_UPDATE");
    ScopedEnvironment<int> find_enforce_tuning_iter_env(wa::MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, 5);

    fusePlanDesc.Compile(get_handle());
    RunTunableSolver<miopen::solver::fusion::ConvOclDirectFwdFused>(fusePlanDesc,
                                                                    params,
                                                                    conv_config,
                                                                    test_skipped,
                                                                    input.desc,
                                                                    in_dev.get(),
                                                                    output.desc,
                                                                    out_dev.get(),
                                                                    wspace);
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvBiasActivInferFusionCompileStep_FP32,
    testing::Combine(testing::Values(miopenActivationRELU),
                     testing::ValuesIn(GetNetworkForFusionCompileStepTest<ConvTestCaseBase>()),
                     testing::Values(miopenTensorNCHW),
                     testing::Values(0.25f),
                     testing::Values(0.75f),
                     testing::Values(0.5f)));

#endif

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvBiasActivInfer_FP32,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenTensorNCHW),
                                          testing::Values(0.25f),
                                          testing::Values(0.75f),
                                          testing::Values(0.5f)));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_ConvBiasActivInfer_FP16,
                         testing::Combine(testing::Values(miopenActivationRELU),
                                          testing::ValuesIn(GetNetwork1<ConvTestCaseBase>()),
                                          testing::Values(miopenTensorNHWC),
                                          testing::Values(0.25f),
                                          testing::Values(0.75f),
                                          testing::Values(0.5f)));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpBiasActivInfer_BFP16,
    testing::Combine(
        testing::Values(miopenActivationRELU, miopenActivationCLIPPEDRELU),
        testing::ValuesIn(GroupConvTestConfig<2u>::GetSmokeConfigs<Direction::Forward>()),
        testing::ValuesIn({miopenTensorNHWC /*, miopenTensorNCHW*/}),
        testing::Values(0.5f),
        testing::Values(1.0f),
        testing::Values(0.5f)));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpBiasActivInfer_BFP16,
    testing::Combine(testing::Values(miopenActivationCLAMP),
                     testing::ValuesIn(GroupConvTestConfig<2u>::GetConfigs<Direction::Forward>()),
                     testing::ValuesIn({miopenTensorNHWC /*, miopenTensorNCHW*/}),
                     testing::Values(0.5f),
                     testing::Values(1.0f),
                     testing::Values(0.5f)));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpBiasActivInfer3D_BFP16,
    testing::Combine(
        testing::Values(miopenActivationRELU, miopenActivationCLIPPEDRELU),
        testing::ValuesIn(GroupConvTestConfig<3u>::GetSmokeConfigs<Direction::Forward>()),
        testing::ValuesIn({miopenTensorNDHWC /*, miopenTensorNCDHW*/}),
        testing::Values(0.5f),
        testing::Values(1.0f),
        testing::Values(0.5f)));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpBiasActivInfer3D_BFP16,
    testing::Combine(testing::Values(miopenActivationCLAMP),
                     testing::ValuesIn(GroupConvTestConfig<3u>::GetConfigs<Direction::Forward>()),
                     testing::ValuesIn({miopenTensorNDHWC /*, miopenTensorNCDHW*/}),
                     testing::Values(0.5f),
                     testing::Values(1.0f),
                     testing::Values(0.5f)));
