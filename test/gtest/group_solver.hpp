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

#include <random>

#include "get_handle.hpp"
#include <miopen/conv/data_invoke_params.hpp>

#include "../driver/tensor_driver.hpp"
#include "conv_common.hpp"

template <typename T>
miopenDataType_t GetDataType();

template <>
miopenDataType_t GetDataType<float>()
{
    return miopenFloat;
}

template <>
miopenDataType_t GetDataType<half_float::half>()
{
    return miopenHalf;
}

template <>
miopenDataType_t GetDataType<int8_t>()
{
    return miopenInt8;
}

struct ConvTestCase
{
    size_t G;
    size_t N;
    size_t C;
    size_t H;
    size_t W;
    size_t k;
    size_t y;
    size_t x;
    size_t pad_x;
    size_t pad_y;
    size_t stride_x;
    size_t stride_y;
    size_t dilation_x;
    size_t dilation_y;
    miopenConvolutionMode_t conv_mode;
    friend std::ostream& operator<<(std::ostream& os, const ConvTestCase& tc)
    {
        return os << " G:" << tc.G << " N:" << tc.N << " C:" << tc.C << " H:" << tc.H
                  << " W:" << tc.W << " k:" << tc.k << " y:" << tc.y << " x:" << tc.x
                  << " pad_y:" << tc.pad_y << " pad_x:" << tc.pad_x << " stride_y:" << tc.stride_y
                  << " dilation_y:" << tc.dilation_y << " conv_mode:" << tc.conv_mode;
    }

    std::vector<size_t> GetInput() { return {N, C, H, W}; }
    std::vector<size_t> GetWeights()
    {
        EXPECT_EQUAL(C % G, 0);
        return {k, C / G, y, x};
    }

    miopen::ConvolutionDescriptor GetConv()
    {
        return miopen::ConvolutionDescriptor{
            2,
            miopenConvolution,
            miopenPaddingDefault,
            {static_cast<int>(pad_y), static_cast<int>(pad_x)},
            {static_cast<int>(stride_y), static_cast<int>(stride_x)},
            {static_cast<int>(dilation_y), static_cast<int>(dilation_x)},
            {0, 0},
            static_cast<int>(G),
            1.0};
    }
};

std::vector<ConvTestCase> ConvTestConfigs()
{ // g  n  c   h   w   k   y  x pad_x pad_y stri_x stri_y dia_x dia_y
    return {{1, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 256, 12, 28, 28, 12, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {4, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {8, 256, 192, 28, 28, 192, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {8, 256, 384, 28, 28, 384, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {32, 256, 1024, 28, 28, 2048, 3, 3, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

inline int SetTensorLayout(miopen::TensorDescriptor& desc)
{
    // get layout string names
    std::string layout_str = desc.GetLayout_str();

    std::vector<std::size_t> lens = desc.GetLengths();
    std::vector<int> int_lens(lens.begin(), lens.end());

    // set the strides for the tensor
    return SetTensorNd(&desc, int_lens, layout_str, desc.GetType());
}

using Direction = miopen::conv::Direction;

template <typename T, Direction CONV_DIR>
struct Conv2DGroupTestFix
    : public ::testing::TestWithParam<
          std::tuple<miopenConvFwdAlgorithm_t, ConvTestCase, miopenTensorLayout_t>>
{
private:

  template <typename F>
  void SetupFwd(F&& gen_value) {
        input.generate(gen_value);
        weights.generate(gen_value);
        std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());
  }

  template <typename F>
  void SetupBwd(F&& gen_value) {
        output.generate(gen_value);
        weights.generate(gen_value);
        std::fill(input.begin(), input.end(), std::numeric_limits<double>::quiet_NaN());
  }

  template <typename F>
  void SetupWrw(F&& gen_value) {
        input.generate(gen_value);
        output.generate(gen_value);
        std::fill(weights.begin(), weights.end(), T{0});
  }

  void verify(const tensor<T>& computed) {
        EXPECT_FALSE(miopen::range_zero(ref)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(computed)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref) == miopen::range_distance(computed));

        double threshold       = 1.0e-5;
        auto error             = miopen::rms_range(ref, computed);

        EXPECT_FALSE(miopen::find_idx(ref, miopen::not_finite) >= 0)
            << "Non finite number found in the reference output";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
  }

  template <typename Solver, typename ConvTensorsType, typename InvokeParamType>
  void RunSolverImpl() {
    auto&& handle = get_handle();

    Solver solv{};

    const auto tensors =
        ConvTensorsType{inputDesc, input, wDesc, weight, outputDesc, output};

    const auto problem = miopen::conv::ProblemDescription{
        inputDesc, wDesc, outputDesc, convDesc, CONV_DIR};
    auto ctx = miopen::ExecutionContext{};

    ctx.SetStream(&handle);

    if(!solv.IsApplicable(ctx, problem))
    {
        test_skipped = true;
        GTEST_SKIP() << solv.SolverDbId()
                     << "Not Applicable for this problem"
                     << conv_config;
    }

    if (solver.MayNeedWorkspace()) {
      wspace.resize(solver.GetWorkSpaceSize(ctx, problem));
    }

    const auto invoke_params = InvokeParamType{tensors, wspace.ptr(), wspace.size(), false};

    ASSERT_TRUE(solv.IsApplicable(ctx, problem));
    auto sol = solv.GetSolution(ctx, problem, solv.GetDefaultPerformanceConfig(ctx, problem));
    ASSERT_TRUE(sol.Succeeded());
    ASSERT_TRUE(sol.invoker_factory);
    const auto invoker = handle.PrepareInvoker(*sol.invoker_factory, sol.construction_params);
    (invoker)(handle, invoke_params);
    handle.Finish();
  }

  void RunSolver() {
        if constexpr (CONV_DIR == Direction::Forward) {
          RunSolverImpl<
            miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops,
            miopen::>();

        } else if constexpr (CONV_DIR == Direction::BackwardData) {
          SetupBwd(gen_value);
        } else {
          static_assert(CONV_DIR == Direction::BackwardWeights);
          SetupWrw(gen_value);
        }
  }

protected:
    void SetUp() override
    {
        test_skipped                               = false;
        std::tie(algo, conv_config, tensor_layout) = GetParam();

        input   = tensor<T>{tensor_layout, conv_config.GetInput()};
        weights = tensor<T>{tensor_layout, conv_config.GetWeights()};
        SetTensorLayout(input.desc);
        SetTensorLayout(weights.desc);

        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensorWithLayout(input.desc, weights.desc, tensor_layout, GetDataType<T>());
        output = tensor<T>{tensor_layout, output_desc.GetLengths()};
        SetTensorLayout(output.desc);

        auto gen_value = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };

        if constexpr (CONV_DIR == Direction::Forward) {
          SetFwd(gen_value);
        } else if constexpr (CONV_DIR == Direction::BackwardData) {
          SetupBwd(gen_value);
        } else {
          static_assert(CONV_DIR == Direction::BackwardWeights);
          SetupWrw(gen_value);
        }


        auto& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
    }

    void TearDown() override
    {
        if(test_skipped)
            return;

        auto& handle = get_handle();


        if constexpr (CONV_DIR == Direction::Forward) {
          ref     = ref_conv_fwd(input, weights, output, conv_desc);
          handle.ReadToVec(out_dev, output);
          verify(output);
        } else if constexpr (CONV_DIR == Direction::BackwardData) {
          ref     = ref_conv_bwd(input, weights, output, conv_desc);
          handle.ReadToVec(in_dev, input);
          verify(input);
        } else {
          static_assert(CONV_DIR == Direction::BackwardWeights);
          ref     = ref_conv_wrw(input, weights, output, conv_desc);
          handle.ReadToVec(wei_dev, weights);
          verify(weights);
        }

    }

    ConvTestCase conv_config;
    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> ref;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoImplicitGEMM;
    bool test_skipped             = false;
    miopenTensorLayout_t tensor_layout;
    Workspace wspace{};
};
