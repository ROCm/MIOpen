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

#include "tensor_util.hpp"
#include "get_handle.hpp"

#include "conv3d_test_case.hpp"

// TODO: uncomment when PR 2599 is merged
// namespace conv_bias_act_res_add_fwd  {

std::vector<Conv3DTestCase> ConvTestConfigs()
{ // g    n   c   d    h   w   k   z  y  x pad_x pad_y pad_z stri_x stri_y stri_z dia_x dia_y dia_z
    return {{1, 128, 64, 14, 28, 28, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 64, 32, 28, 28, 28, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {32, 128, 32, 28, 28, 28, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {16, 128, 16, 28, 28, 28, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {8, 128, 8, 28, 28, 28, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {4, 128, 4, 28, 28, 28, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {2, 128, 2, 28, 28, 28, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution}};
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

template <typename T = float>
struct ConvFwdBiasResAddFixture
    : public ::testing::TestWithParam<std::tuple<miopenConvBwdDataAlgorithm_t,
                                                 Conv3DTestCase,
                                                 float,
                                                 float,
                                                 miopenTensorLayout_t>>
{

protected:
    void SetUp() override
    {

        std::tie(algo, conv_config, alpha1, alpha2, tensor_layout) = GetParam();
        input   = tensor<T>{tensor_layout, conv_config.GetInput()};
        weights = tensor<T>{tensor_layout, conv_config.GetWeights()};
        SetTensorLayout(input.desc);
        SetTensorLayout(weights.desc);
        auto gen_value = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };
        input.generate(gen_value);
        weights.generate(gen_value);
        conv_desc = conv_config.GetConv();

        miopen::TensorDescriptor output_desc =
            conv_desc.GetForwardOutputTensor(input.desc, weights.desc, GetDataType<T>());
        output = tensor<T>{tensor_layout, output_desc.GetLengths()};
        SetTensorLayout(output.desc);
        std::fill(output.begin(), output.end(), std::numeric_limits<double>::quiet_NaN());

        z = tensor<T>{tensor_layout, output_desc.GetLengths()};
        SetTensorLayout(z.desc);
        z.generate(gen_value);

        bias = tensor<T>{tensor_layout,
                         {1, 1, 1, 1, conv_config.k}, // NDHWK for lengths
                         {0, 1, 0, 0, 0}};            // NKDHW order for strides

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

        miopenDestroyActivationDescriptor(activ_desc);

        auto&& handle = get_handle();

        ref_out = tensor<T>{tensor_layout, output_desc.GetLengths()};
        ref_out = ref_conv_fwd(input, weights, output, conv_desc);

        // implement equation out = act(conv(in) * alpah1 + z * alpha2 + bias);
        ref_out.par_for_each([&](auto n, auto k, auto... dhw) {
            auto& o = ref_out(n, k, dhw...);

            o *= alpha1;
            o += alpha2 * z(n, k, dhw...) + bias(n, k, dhw...);
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
    miopenActivationDescriptor activ_desc;
};

TEST_P(ConvFwdBiasResAddActivTest, ConvFwdBiasResAddFixture)
{
    auto status = miopenConvolutionBiasActivationForward(&alpha1,
                                                         &input.desc,
                                                         in_dev.get(),
                                                         &weights.desc,
                                                         wei_dev.get(),
                                                         conv_desc,
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

INSTANTIATE_TEST_SUITE_P(ConvFwdBiasActivAPI,
                         ConvBwdSolverTest3D,
                         testing::Combine(testing::Values(miopenConvolutionFwdAlgoImplicitGEMM),
                                          testing::ValuesIn(ConvTestConfigs()),
                                          testing::ValuesIn({1.0f, 2.0f}), // alpha1
                                          testing::ValuesIn({1.0f, 2.0f}), // alpha2
                                          testing::Values(miopenTensorNDHWC)));

// TODO: uncomment when PR 2599 is merged
// }// end namespace conv_bias_act_res_add_fwd
