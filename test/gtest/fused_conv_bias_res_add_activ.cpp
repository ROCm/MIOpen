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
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/env.hpp>
#if MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#endif

#include "tensor_util.hpp"
#include "get_handle.hpp"

#include "conv3d_test_case.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)

#if MIOPEN_USE_COMPOSABLEKERNEL
#define WORAROUND_ISSUE_2533 1
#endif

namespace conv_bias_act_res_add_fwd {

bool TestIsApplicable()
{
#if MIOPEN_USE_COMPOSABLEKERNEL
    const auto float_arg = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    return
#if WORAROUND_ISSUE_2533
        miopen::solver::ck_utility::is_ck_whitelist(get_handle().GetDeviceName()) //
#else
    /// \todo Check against specific ASCIs.
#endif
        && (float_arg == "--half" // So far only test for fp16 is implemented.
            || float_arg.empty()) // Empty when gtest is run without parameters.
        && !miopen::IsDisabled(
               ENV(MIOPEN_TEST_ALL)); // Not disabled when gtest is run without parameters.
#else
    return false;
#endif
}

std::vector<Conv3DTestCase> ConvTestConfigs()
{ //         g, n, c, d,  h,  w, k,  z, y, x, pad_x pad_y pad_z stri_x stri_y stri_z dia_x dia_y
  //         dia_z
    return {{1, 1, 4, 14, 11, 1, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 1, 1, 1, 4, 4, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 1, 1, 8, 8, 8, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 1, 1, 8, 8, 8, 1, 2, 2, 2, 0, 0, 0, 2, 2, 2, 1, 1, 1, miopenConvolution},
            {2, 8, 8, 12, 14, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {4, 8, 8, 11, 11, 11, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {6, 8, 18, 11, 11, 11, 18, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {8, 8, 8, 11, 11, 11, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {4, 8, 4, 11, 11, 11, 8, 3, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {2, 8, 2, 11, 11, 11, 2, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

template <typename T = float>
struct ConvFwdBiasResAddFixture
    : public ::testing::TestWithParam<
          std::tuple<miopenConvFwdAlgorithm_t, Conv3DTestCase, float, float, miopenTensorLayout_t>>
{

protected:
    void SetUp() override
    {
        if(!TestIsApplicable())
            return;

        std::tie(algo, conv_config, alpha1, alpha2, tensor_layout) = GetParam();

        input          = tensor<T>{tensor_layout, conv_config.GetInput()};
        weights        = tensor<T>{tensor_layout, conv_config.GetWeights()};
        auto gen_value = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };
        input.generate(gen_value);
        weights.generate(gen_value);
        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, GetDataType<T>());
        output = tensor<T>{tensor_layout, output_desc.GetLengths()};
        std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());

        z = tensor<T>{tensor_layout, output_desc.GetLengths()};
        z.generate(gen_value);

        const std::vector<std::size_t>& strides = {1, 1, 1, 1, 1};
        bias = tensor<T>{tensor_layout, {1, 1, 1, 1, conv_config.k}, strides};

        bias.generate(gen_value);

        auto& handle = get_handle();
        in_dev       = handle.Write(input.data);
        wei_dev      = handle.Write(weights.data);
        out_dev      = handle.Write(output.data);
        z_dev        = handle.Write(z.data);
        bias_dev     = handle.Write(bias.data);

        miopenCreateActivationDescriptor(&activ_desc);
        miopenSetActivationDescriptor(activ_desc, miopenActivationRELU, 1.0f, 1.0f, 1.0f);
    }
    void TearDown() override
    {
        if(!TestIsApplicable())
            return;

        miopenDestroyActivationDescriptor(activ_desc);

        auto&& handle = get_handle();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, GetDataType<T>());

        ref_out = tensor<T>{tensor_layout, output_desc.GetLengths()};
        ref_out = ref_conv_fwd(input, weights, output, conv_desc);

        // implement equation out = act(conv(in) * alpah1 + z * alpha2 + bias);
        ref_out.par_for_each([&](auto n, auto k, auto... dhw) {
            auto& o = ref_out(n, k, dhw...);

            o *= alpha1;
            o += alpha2 * z(n, k, dhw...) + bias(0, k, 0, 0, 0);
            o = (o > T{0}) ? o : T{0}; // TODO: hardcoded relu. Todo: use
                                       // activationHostInfer
        });

        output.data = handle.Read<T>(out_dev, output.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(ref_out) == miopen::range_distance(output));

        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(ref_out, output);

        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }

    Conv3DTestCase conv_config;
    float alpha1 = 1.0f;
    float alpha2 = 1.0f;
    miopen::ConvolutionDescriptor conv_desc;
    tensor<T> input;
    tensor<T> weights;
    tensor<T> output;
    tensor<T> z;
    tensor<T> bias;
    tensor<T> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopen::Allocator::ManageDataPtr z_dev;
    miopen::Allocator::ManageDataPtr bias_dev;

    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoImplicitGEMM;
    miopenTensorLayout_t tensor_layout;
    miopenActivationDescriptor_t activ_desc;
};

struct ConvFwdBiasResAddActivTest : ConvFwdBiasResAddFixture<half_float::half>
{
};

} // end namespace conv_bias_act_res_add_fwd
  //

using namespace conv_bias_act_res_add_fwd;

TEST_P(ConvFwdBiasResAddActivTest, ConvFusedAPI)
{
    if(TestIsApplicable())
    {
        auto status = miopenConvolutionBiasActivationForward(&get_handle(),
                                                             &alpha1,
                                                             &input.desc,
                                                             in_dev.get(),
                                                             &weights.desc,
                                                             wei_dev.get(),
                                                             &conv_desc,
                                                             algo,
                                                             nullptr, // workspace
                                                             0ull,    // workspace size
                                                             &alpha2,
                                                             &z.desc,
                                                             z_dev.get(),
                                                             &bias.desc,
                                                             bias_dev.get(),
                                                             activ_desc,
                                                             &output.desc,
                                                             out_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);
    }
    else
    {
        GTEST_SKIP();
    }
}

INSTANTIATE_TEST_SUITE_P(ConvFwdBiasActivAPI,
                         ConvFwdBiasResAddActivTest,
                         testing::Combine(testing::Values(miopenConvolutionFwdAlgoImplicitGEMM),
                                          testing::ValuesIn(ConvTestConfigs()),
                                          testing::ValuesIn({1.0f, 2.0f}), // alpha1
                                          testing::ValuesIn({1.0f, 2.0f}), // alpha2
                                          testing::Values(miopenTensorNDHWC)));
