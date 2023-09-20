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
#include "tensor_util.hpp"
#include <fusionHost.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>

#include <miopen/hip_float8.hpp>
#include <miopen/type_name.hpp>
#include <miopen/rank.hpp>

#include "conv_test_base.hpp"
#include "conv_tensor_gen.hpp"

#include "get_solver.hpp"

template <typename T = float, typename Tref = float, bool use_cpu_ref = false>
struct ConvWrwSolverTest
    : public ::testing::TestWithParam<std::tuple<miopenConvFwdAlgorithm_t, ConvTestCase>>
{

    template <typename Solver>
    void SolverWrw(Solver solv)
    {
        auto&& handle = get_handle();

        const auto tensors = miopen::ConvWrwTensors{
            output.desc, out_dev.get(), input.desc, in_dev.get(), weights.desc, wei_dev.get()};
        const auto problem = miopen::ProblemDescription(
            miopen::conv::ProblemDescription{output.desc,
                                             weights.desc,
                                             input.desc,
                                             conv_desc,
                                             miopen::conv::Direction::BackwardWeights});
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
            miopen::conv::WrWInvokeParams{tensors,
                                          workspace_dev.get(),
                                          workspace_size,
                                          conv_desc.attribute.gfx90aFp16alt.GetBwd()};

        auto sol = GetSolution(solv, ctx, problem);
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);
        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        (invoker)(handle, invoke_params);
        handle.Finish();
    }

protected:
    void SetUp() override
    {
        test_skipped                = false;
        std::tie(algo, conv_config) = GetParam();
        input   = tensor<T>{conv_config.N, conv_config.C, conv_config.H, conv_config.W};
        weights = tensor<T>{conv_config.k, conv_config.C, conv_config.y, conv_config.x};
        input.generate(GenData<T>{});

        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, GetDataType<T>());

        output = tensor<T>{output_desc.GetLengths()};
        output.generate(GenData<T>{});

        std::fill(weights.begin(), weights.end(), std::numeric_limits<T>::quiet_NaN());

        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
    }
    void TearDown() override
    {
        if(test_skipped)
            return;

        auto&& handle = get_handle();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, GetDataType<T>());
        ref_weights = tensor<Tref>{output_desc.GetLengths()};
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
        weights.data = handle.Read<T>(in_dev, input.data.size());
#if defined(__clang__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
        const auto zero_chk = [](T x) { return static_cast<T>(x) == static_cast<T>(0.0); };
#if defined(__clang__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

        EXPECT_FALSE(std::all_of(ref_weights.begin(), ref_weights.end(), [](float x) {
            return x == 0.0f;
        })) << "Cpu data is all zeros";
        EXPECT_FALSE(std::all_of(weights.begin(), weights.end(), zero_chk))
            << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_weights) == miopen::range_distance(weights));

        const double tolerance = 80;
        double threshold       = static_cast<float>(std::numeric_limits<T>::epsilon()) * tolerance;
        auto error             = miopen::rms_range(ref_weights, weights);

        EXPECT_FALSE(miopen::find_idx(ref_weights, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
    ConvTestCase conv_config;
    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<Tref> ref_weights;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    size_t workspace_size;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoDirect;
    bool test_skipped             = false;
};
