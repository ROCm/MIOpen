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
#pragma once

#include <gtest/gtest.h>
#include "conv_common.hpp"
#include "get_handle.hpp"
#include "tensor_util.hpp"
#include <fusionHost.hpp>
#include <miopen/conv/data_invoke_params.hpp>

#include <miopen/hip_float8.hpp>
#include <miopen/type_name.hpp>
#include <miopen/rank.hpp>

#include "conv_test_base.hpp"
#include "get_solver.hpp"

template <typename T = float, typename Tref = float, bool use_cpu_ref = false>
struct ConvFwdSolverTest
    : public ::testing::TestWithParam<
          std::tuple<miopenConvFwdAlgorithm_t, ConvTestCase, miopenTensorLayout_t>>,
      ConvFwdSolverTestBase<T, Tref, use_cpu_ref>
{
    template <typename Solver>
    void SolverFwd(Solver solv)
    {
        auto&& handle = get_handle();

        const auto tensors = miopen::ConvFwdTensors{this->input.desc,
                                                    this->in_dev.get(),
                                                    this->weights.desc,
                                                    this->wei_dev.get(),
                                                    this->output.desc,
                                                    this->out_dev.get()};
        const auto problem = miopen::ProblemDescription(
            miopen::conv::ProblemDescription{this->input.desc,
                                             this->weights.desc,
                                             this->output.desc,
                                             this->conv_desc,
                                             miopen::conv::Direction::Forward});
        const miopen::ConvolutionContext ctx = [&] {
            auto tmp = miopen::ConvolutionContext{&handle};
            problem.conv_problem.SetupFloats(tmp);
            return tmp;
        }();

        // const auto network_config = problem.BuildConfKey();

        if(!solv.IsApplicable(ctx, problem))
        {
            test_skipped = true;
            GTEST_SKIP() << solv.SolverDbId() << ": Not Applicable for this problem" << conv_config;
        }
        if(solv.MayNeedWorkspace())
        {
            const auto cur_sol_ws = solv.GetWorkspaceSize(ctx, problem);
            workspace_dev         = handle.Create<T>(cur_sol_ws);
            workspace_size        = cur_sol_ws;
        }

        const auto invoke_params =
            miopen::conv::DataInvokeParams{tensors,
                                           workspace_dev.get(),
                                           workspace_size,
                                           this->conv_desc.attribute.gfx90aFp16alt.GetFwd()};

        // auto sol = solv.GetSolution(ctx, problem);
        // This is complicated due to the split between tunable and non-tunable solvers
        // since the signature for solver.GetSolution needs a consutructed tuning params
        // in the tunable case and not otherwise
        const auto sol = GetSolution(solv, ctx, problem);
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);
        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        (invoker)(handle, invoke_params);
        handle.Finish();
    }

protected:
    void SetUp() override
    {
        test_skipped                               = false;
        std::tie(algo, conv_config, tensor_layout) = GetParam();
        this->SetUpImpl(conv_config, tensor_layout);
    }

    void TearDown() override
    {
        if(test_skipped)
            return;
        this->TearDownConv();
        this->ThresholdChecks();
    }

    ConvTestCase conv_config;
    miopen::Allocator::ManageDataPtr workspace_dev;
    size_t workspace_size;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoDirect;
    bool test_skipped             = false;
    miopenTensorLayout_t tensor_layout;
};
