/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/solver.hpp>

#include "conv_test_base.hpp"
#include "conv_tensor_gen.hpp"

#include "../workspace.hpp"

template <typename T = float, typename Tref = float, bool use_cpu_ref = false>
struct ConvWrwSolverTest
    : public ::testing::TestWithParam<std::tuple<miopenConvFwdAlgorithm_t, ConvTestCaseBase>>
{
    void SolverWrw(const miopen::solver::conv::ConvSolverBase& solv)
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

        const auto tensors = miopen::ConvWrwTensors{
            output.desc, out_dev.get(), input.desc, in_dev.get(), weights.desc, wei_dev.get()};

        const auto problem =
            miopen::conv::ProblemDescription(output.desc,
                                             weights.desc,
                                             input.desc,
                                             conv_desc,
                                             miopen::conv::Direction::BackwardWeights);
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

        const auto invoke_params = miopen::conv::WrWInvokeParams{
            tensors, wspace.ptr(), wspace.size(), conv_desc.attribute.gfx90aFp16alt.GetWrW()};

        auto sol = solv.GetDefaultSolution(ctx, problem);
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);
        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        (invoker)(handle, invoke_params);
        handle.Finish();

        test_skipped = false;
    }

    void SetUpTest()
    {
        test_skipped = true;

        ConvTestCaseBase conv_config;
        std::tie(algo, conv_config) = GetParam();
        input                       = tensor<T>{conv_config.GetInput()};
        weights                     = tensor<T>{conv_config.GetWeights()};
        input.generate(GenData<T>{});

        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});

        output = tensor<T>{output_desc.GetLengths()};
        output.generate(GenData<T>{});

        std::fill(weights.begin(), weights.end(), T(0));

        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
    }

    void TearDownTest()
    {
        if(test_skipped)
            return;

        ref_weights = tensor<Tref>{weights.desc.GetLengths()};
        if(use_cpu_ref)
        {
            cpu_convolution_backward_weight(conv_desc.GetSpatialDimension(),
                                            input,
                                            ref_weights,
                                            output,
                                            conv_desc.GetConvPads(),
                                            conv_desc.GetConvStrides(),
                                            conv_desc.GetConvDilations(),
                                            conv_desc.GetGroupCount());
        }
        else
        {
            ref_weights = ref_conv_wrw(input, ref_weights, output, conv_desc);
        }

        auto&& handle = get_handle();
        weights.data  = handle.Read<T>(wei_dev, weights.data.size());

        ASSERT_FALSE(miopen::range_zero(ref_weights)) << "Cpu data is all zeros";
        ASSERT_FALSE(miopen::range_zero(weights)) << "Gpu data is all zeros";
        ASSERT_TRUE(miopen::range_distance(ref_weights) == miopen::range_distance(weights));

        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_weights, weights);

        ASSERT_FALSE(miopen::find_idx(ref_weights, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        ASSERT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }

    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<Tref> ref_weights;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    Workspace wspace{};
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoDirect;
    bool test_skipped;
};
