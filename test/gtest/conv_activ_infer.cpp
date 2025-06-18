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

namespace ca_infer {

using float16 = half_float::half;

template <typename T>
struct GPU_ConvGrpActivInfer : ConvActivInferTest<T, GroupConvTestConfig<2u>>
{
};

using GPU_ConvGrpActivInfer_BFP16 = GPU_ConvGrpActivInfer<bfloat16>;

template <typename T>
struct GPU_ConvGrpActivInfer3D : ConvActivInferTest<T, GroupConvTestConfig<3u>>
{
};

using GPU_ConvGrpActivInfer3D_BFP16 = GPU_ConvGrpActivInfer3D<bfloat16>;

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

} // namespace ca_infer

using namespace ca_infer;

TEST_P(GPU_ConvGrpActivInfer_BFP16, ConvCKIgemmGrpFwdActivFused)
{
    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdActivFused>(fusePlanDesc,
                                                                          params,
                                                                          conv_config,
                                                                          test_skipped,
                                                                          input.desc,
                                                                          in_dev.get(),
                                                                          output.desc,
                                                                          out_dev.get(),
                                                                          wspace);
}

TEST_P(GPU_ConvGrpActivInfer3D_BFP16, ConvCKIgemmGrpFwdActiv3DFused)
{

    RunTunableSolver<miopen::solver::fusion::ConvCKIgemmGrpFwdActivFused>(fusePlanDesc,
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
    GPU_ConvGrpActivInfer_BFP16,
    testing::Combine(
        testing::Values(miopenActivationRELU, miopenActivationCLIPPEDRELU),
        testing::ValuesIn(GroupConvTestConfig<2>::GetSmokeConfigs<Direction::Forward>()),
        testing::Values(miopenTensorNHWC /*, miopenTensorNCHW*/),
        testing::Values(0.5f),
        testing::Values(1.0f),
        testing::Values(0.5f)));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpActivInfer_BFP16,
    testing::Combine(testing::Values(miopenActivationCLAMP),
                     testing::ValuesIn(GroupConvTestConfig<2>::GetConfigs<Direction::Forward>()),
                     testing::Values(miopenTensorNHWC /*, miopenTensorNCHW*/),
                     testing::Values(0.5f),
                     testing::Values(1.0f),
                     testing::Values(0.5f)));

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    GPU_ConvGrpActivInfer3D_BFP16,
    testing::Combine(
        testing::Values(miopenActivationRELU, miopenActivationCLIPPEDRELU),
        testing::ValuesIn(GroupConvTestConfig<3>::GetSmokeConfigs<Direction::Forward>()),
        testing::Values(miopenTensorNDHWC /*, miopenTensorNCDHW*/),
        testing::Values(0.5f),
        testing::Values(1.0f),
        testing::Values(0.5f)));

INSTANTIATE_TEST_SUITE_P(
    Full,
    GPU_ConvGrpActivInfer3D_BFP16,
    testing::Combine(testing::Values(miopenActivationCLAMP),
                     testing::ValuesIn(GroupConvTestConfig<3>::GetConfigs<Direction::Forward>()),
                     testing::Values(miopenTensorNDHWC /*, miopenTensorNCDHW*/),
                     testing::Values(0.5f),
                     testing::Values(1.0f),
                     testing::Values(0.5f)));
