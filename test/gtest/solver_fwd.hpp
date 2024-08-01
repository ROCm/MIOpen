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
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver.hpp>

#include "conv_test_base.hpp"
#include "../workspace.hpp"

template <typename T = float, typename Tref = float, bool use_cpu_ref = false>
struct ConvFwdSolverTest
    : public ::testing::TestWithParam<
          std::tuple<miopenConvFwdAlgorithm_t, ConvTestCaseBase, miopenTensorLayout_t>>,
      ConvFwdSolverTestBase<T, Tref, use_cpu_ref>
{
    void SolverFwd(const miopen::solver::conv::ConvSolverBase& solv)
    {
        // SetUpTest() and TearDownTest() are moved here so that if a test is skipped, time is not
        // wasted.
        this->SetUpTest();
        this->RunSolver(solv);
        this->TearDownTest();
    }

protected:
    void SetUp() override
    {
        // this->SetUpTest();
    }

    void TearDown() override
    {
        // this->TearDownTest();
    }

private:
    void RunSolver(const miopen::solver::conv::ConvSolverBase& solv)
    {
        auto&& handle = get_handle();

        const auto tensors = miopen::ConvFwdTensors{this->input.desc,
                                                    this->in_dev.get(),
                                                    this->weights.desc,
                                                    this->wei_dev.get(),
                                                    this->output.desc,
                                                    this->out_dev.get()};

        const auto problem                 = miopen::conv::ProblemDescription(this->input.desc,
                                                              this->weights.desc,
                                                              this->output.desc,
                                                              this->conv_desc,
                                                              miopen::conv::Direction::Forward);
        const miopen::ExecutionContext ctx = [&] {
            auto tmp = miopen::ExecutionContext{&handle};
            problem.SetupFloats(tmp);
            return tmp;
        }();

        if(!solv.IsApplicable(ctx, problem))
        {
            // Do not put GTEST_SKIP here! An inappropriate config should cause the test to fail,
            // not skip. Otherwise, such testing is pointless.
            GTEST_FAIL();
        }

        if(solv.MayNeedWorkspace())
        {
            const auto cur_sol_ws = solv.GetWorkspaceSize(ctx, problem);
            wspace.resize(cur_sol_ws);
        }

        const auto invoke_params = miopen::conv::DataInvokeParams{
            tensors, wspace.ptr(), wspace.size(), this->conv_desc.attribute.gfx90aFp16alt.GetFwd()};

        const auto sol = solv.GetDefaultSolution(ctx, problem);
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);
        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        (invoker)(handle, invoke_params);
        handle.Finish();

        test_skipped = false;
    }

    void SetUpTest()
    {
        test_skipped                               = true;
        std::tie(algo, conv_config, tensor_layout) = GetParam();
        this->SetUpImpl(conv_config, tensor_layout);
    }

    void TearDownTest()
    {
        if(test_skipped)
            return;
        this->TearDownConv();
        this->ThresholdChecks();
    }

    ConvTestCaseBase conv_config;
    Workspace wspace{};
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoDirect;
    bool test_skipped;
    miopenTensorLayout_t tensor_layout;
};
