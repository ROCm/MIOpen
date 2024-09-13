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
#include <miopen/miopen.h>

#define WORKAROUND_ISSUE_2212 1

#if MIOPEN_BACKEND_HIP
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>
#include <serialize.hpp>
#include <fusionHost.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"
#include "gtest_common.hpp"

struct CBATestCase
{
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
    size_t dialtion_x;
    size_t dilation_y;
    miopenActivationMode_t activ_mode;
    miopenConvolutionMode_t conv_mode;
    friend std::ostream& operator<<(std::ostream& os, const CBATestCase& tc)
    {
        return os << "N: " << tc.N << " C:" << tc.C << " H:" << tc.H << " W:" << tc.W
                  << " k: " << tc.k << " y:" << tc.y << " x:" << tc.x << " pad_y:" << tc.pad_y
                  << " pad_x:" << tc.pad_x << " stride_y:" << tc.stride_y
                  << " dilation_y:" << tc.dilation_y << " activ_mode:" << tc.activ_mode
                  << " conv_mode:" << tc.conv_mode;
    }
};

bool IsTestSupportedForDevice()
{
#if WORKAROUND_ISSUE_2212
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A, Gpu::gfx103X>;
    using d_mask = disabled<Gpu::gfx110X, Gpu::gfx94X>;
    return ::IsTestSupportedForDevMask<d_mask, e_mask>();
#else
    return true;
#endif
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GPU_ConvBiasActivFwd_FP32);
struct GPU_ConvBiasActivFwd_FP32
    : public ::testing::TestWithParam<std::tuple<miopenConvFwdAlgorithm_t, CBATestCase>>
{
protected:
    void SetUp() override
    {
        if(!IsTestSupportedForDevice())
        {
            GTEST_SKIP();
        }
        std::tie(algo, cba_config) = GetParam();
        const double double_zero   = 0.0f;
        input   = tensor<float>{cba_config.N, cba_config.C, cba_config.H, cba_config.W};
        weights = tensor<float>{cba_config.k, cba_config.C, cba_config.x, cba_config.y};
        input.generate(tensor_elem_gen_integer{17});
        weights.generate(tensor_elem_gen_integer{17});
        miopenCreateConvolutionDescriptor(&conv_desc);
        miopenCreateActivationDescriptor(&activ_desc);
        miopenSetActivationDescriptor(
            activ_desc, cba_config.activ_mode, double_zero, double_zero, double_zero);
        miopenInitConvolutionDescriptor(conv_desc,
                                        cba_config.conv_mode,
                                        cba_config.pad_y,
                                        cba_config.pad_x,
                                        cba_config.stride_y,
                                        cba_config.stride_x,
                                        cba_config.dilation_y,
                                        cba_config.dialtion_x);
        int n, c, h, w;
        miopenGetConvolutionForwardOutputDim(conv_desc, &input.desc, &weights.desc, &n, &c, &h, &w);
        output  = tensor<float>{static_cast<size_t>(n),
                               static_cast<size_t>(c),
                               static_cast<size_t>(h),
                               static_cast<size_t>(w)};
        ref_out = tensor<float>{static_cast<size_t>(n),
                                static_cast<size_t>(c),
                                static_cast<size_t>(h),
                                static_cast<size_t>(w)};
        bias    = tensor<float>{1, static_cast<size_t>(c), 1, 1};
        bias.generate(tensor_elem_gen_integer{17});
        std::fill(output.begin(), output.end(), 0.0f);
        std::fill(ref_out.begin(), ref_out.end(), 0.0f);
        std::fill(bias.begin(), bias.end(), 0.0f);
        auto&& handle = get_handle();
        in_dev        = handle.Write(input.data);
        wei_dev       = handle.Write(weights.data);
        out_dev       = handle.Write(output.data);
        bias_dev      = handle.Write(bias.data);
    }
    void TearDown() override
    {
        const double double_zero = 0.0f;
        int bias_mode            = 1; // zero disables bias
        convHostForward(input, ref_out, weights, bias_mode, bias, conv_desc);
        activationHostInfer(cba_config.activ_mode,
                            double_zero,
                            double_zero,
                            double_zero,
                            ref_out.data,
                            ref_out.data);
        auto&& handle = get_handle();
        output.data   = handle.Read<float>(out_dev, output.data.size());
        EXPECT_FALSE(miopen::range_zero(ref_out)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "Gpu data is all zeros";
        const auto mxdiff = miopen::max_diff(output, ref_out);
        std::ignore       = mxdiff;
        auto idx          = miopen::mismatch_idx(ref_out, output, miopen::float_equal);
        EXPECT_FALSE(miopen::find_idx(ref_out, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";
        EXPECT_FALSE(idx < miopen::range_distance(ref_out));
        miopenDestroyConvolutionDescriptor(conv_desc);
        miopenDestroyActivationDescriptor(activ_desc);
    }
    CBATestCase cba_config;
    miopenConvolutionDescriptor_t conv_desc;
    miopenActivationDescriptor_t activ_desc;
    tensor<float> input;
    tensor<float> weights;
    tensor<float> output;
    tensor<float> bias;
    tensor<float> ref_out;
    miopen::Allocator::ManageDataPtr in_dev;
    miopen::Allocator::ManageDataPtr wei_dev;
    miopen::Allocator::ManageDataPtr out_dev;
    miopen::Allocator::ManageDataPtr bias_dev;
    miopenConvFwdAlgorithm_t algo = miopenConvolutionFwdAlgoDirect;
};

TEST_P(GPU_ConvBiasActivFwd_FP32, DISABLED_DriveAPI)
{

    tensor<float> z{};
    const float alpha = 1.0f;
    const auto status = miopenConvolutionBiasActivationForward(&get_handle(),
                                                               &alpha,
                                                               &input.desc,
                                                               in_dev.get(),
                                                               &weights.desc,
                                                               wei_dev.get(),
                                                               conv_desc,
                                                               algo,
                                                               nullptr,
                                                               0,
                                                               &alpha,
                                                               &z.desc,
                                                               nullptr,
                                                               &bias.desc,
                                                               bias_dev.get(),
                                                               activ_desc,
                                                               &output.desc,
                                                               out_dev.get());
    EXPECT_EQ(status, miopenStatusSuccess);
}

// Extra layer of indirection introduced since GTEST_SKIP() cannot be called from non-void function.
std::vector<CBATestCase> GetTestValues()
{
    return {
        {16, 128, 16, 16, 128, 3, 3, 0, 0, 1, 1, 1, 1, miopenActivationRELU, miopenConvolution}};
}
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvBiasActivFwd_FP32,
                         testing::Combine(testing::Values(miopenConvolutionFwdAlgoDirect,
                                                          miopenConvolutionFwdAlgoWinograd),
                                          testing::ValuesIn(GetTestValues())));
#endif
