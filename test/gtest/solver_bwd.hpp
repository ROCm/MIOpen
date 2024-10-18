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

#include "gtest_common.hpp"
#include "conv_common.hpp"
#include "get_handle.hpp"
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>

#include "conv_test_base.hpp"
#include "conv_tensor_gen.hpp"

#include "../workspace.hpp"

template <typename T = float, typename Tref = float, bool use_cpu_ref = false>
struct ConvBwdSolverTest
    : public ::testing::TestWithParam<std::tuple<Gpu, miopenConvAlgorithm_t, ConvTestCaseBase>>
{
    void SolverBwd(const miopen::solver::conv::ConvSolverInterface& solv)
    {
        auto&& handle = get_handle();

        const auto tensors = miopen::ConvBwdTensors{
            output.desc, out_dev.get(), weights.desc, wei_dev.get(), input.desc, in_dev.get()};

        const auto problem =
            miopen::conv::ProblemDescription(input.desc,
                                             weights.desc,
                                             output.desc,
                                             conv_desc,
                                             miopen::conv::Direction::BackwardData);
        const miopen::ExecutionContext ctx = [&] {
            auto tmp = miopen::ExecutionContext{&handle};
            problem.SetupFloats(tmp);
            return tmp;
        }();

        if(!solv.IsApplicable(ctx, problem))
        {
            // Do not put GTEST_SKIP here.
            // The usage of non-applicable config should be considered as a bug in the test.
            GTEST_FAIL();
        }

        if(solv.MayNeedWorkspace())
        {
            const auto cur_sol_ws = solv.GetWorkspaceSize(ctx, problem);
            wspace.resize(cur_sol_ws);
        }

        const auto invoke_params = miopen::conv::DataInvokeParams{
            tensors, wspace.ptr(), wspace.size(), conv_desc.attribute.gfx90aFp16alt.GetBwd()};

        // \todo add path for tunable solvers
        const auto& conv_solv = dynamic_cast<const miopen::solver::conv::ConvSolver&>(solv);

        const auto sol = conv_solv.GetSolution(ctx, problem);
        ASSERT_TRUE(sol.Succeeded());
        ASSERT_TRUE(sol.invoker_factory);
        const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
        (invoker)(handle, invoke_params);
        handle.Finish();

        this->Verify();
    }

protected:
    void SetUp() override
    {
        Gpu supported_devs;
        ConvTestCaseBase conv_config;
        std::tie(supported_devs, algo, conv_config) = GetParam();

        if(!IsTestSupportedByDevice(supported_devs))
        {
            GTEST_SKIP();
        }

        input   = tensor<T>{conv_config.GetInput()};
        weights = tensor<T>{conv_config.GetWeights()};
        weights.generate(GenWeights<T>{});

        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, miopen_type<T>{});

        output = tensor<T>{output_desc.GetLengths()};
        output.generate(GenData<T>{});

        std::fill(input.begin(), input.end(), T(0));

        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
    }

private:
    void Verify()
    {
        ref_in = tensor<Tref>{input.desc.GetLengths()};
        if(use_cpu_ref)
        {
            cpu_convolution_backward_data(conv_desc.GetSpatialDimension(),
                                          ref_in,
                                          weights,
                                          output,
                                          conv_desc.GetConvPads(),
                                          conv_desc.GetConvStrides(),
                                          conv_desc.GetConvDilations(),
                                          conv_desc.GetGroupCount());
        }
        else
        {
            ref_in = ref_conv_bwd(ref_in, weights, output, conv_desc);
        }

        auto&& handle = get_handle();
        input.data    = handle.Read<T>(in_dev, input.data.size());

        ASSERT_FALSE(miopen::range_zero(ref_in)) << "Cpu data is all zeros";
        ASSERT_FALSE(miopen::range_zero(input)) << "Gpu data is all zeros";
        ASSERT_EQ(miopen::range_distance(ref_in), miopen::range_distance(input));

        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_in, input);

        ASSERT_LT(miopen::find_idx(ref_in, miopen::not_finite), 0)
            << "Non finite number found in the CPU data";

        ASSERT_LT(error, threshold) << "Error beyond tolerance";
    }

    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<Tref> ref_in;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    Workspace wspace{};
    miopenConvAlgorithm_t algo = miopenConvolutionAlgoDirect;
};
