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

#include "../driver/tensor_driver.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/layernorm.hpp>
#include <miopen/miopen.h>

template <class T>
void cpu_layernorm_forward(tensor<T> input,
                           tensor<T> weight,
                           tensor<T> bias,
                           tensor<T>& ref_output,
                           tensor<T>& ref_mean,
                           tensor<T>& ref_rstd,
                           float eps,
                           int32_t dim,
                           miopenNormMode_t mode)
{
    auto dims         = input.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = 1;
    size_t i          = 0;
    for(; i < dim; i++)
    {
        outer_size *= dims[i];
    }

    for(; i < dims.size(); i++)
    {
        inner_size *= dims[i];
    }

    par_ford(outer_size)([&](int32_t o) {
        float mean_v = 0;
        float var_v  = 0;

        ford(inner_size)([&](int32_t i) {
            float tmp = static_cast<float>(input[o * inner_size + i]);
            mean_v += tmp;
            var_v += tmp * tmp;
        });

        mean_v       = mean_v / inner_size;
        var_v        = var_v / inner_size - mean_v * mean_v;
        float rstd_v = 1 / sqrt(var_v + eps);

        ref_mean[o] = static_cast<T>(mean_v);
        ref_rstd[o] = static_cast<T>(rstd_v);

        ford(inner_size)([&](int32_t i) {
            float weight_v =
                (mode == MIOPEN_ELEMENTWISE_AFFINE) ? 1 : static_cast<float>(weight[i]);
            float bias_v = (mode == MIOPEN_ELEMENTWISE_AFFINE) ? 0 : static_cast<float>(bias[i]);
            ref_output[o * inner_size + i] = static_cast<T>(
                (static_cast<float>(input[o * inner_size + i]) - mean_v) * rstd_v * weight_v +
                bias_v);
        });
    });
}

struct LayerNormTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    size_t nomalized_dim;
    float eps;
    miopenNormMode_t ln_mode;
    friend std::ostream& operator<<(std::ostream& os, const LayerNormTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " dim:" << tc.nomalized_dim << " eps:" << tc.eps
                  << " LayerNorm_mode:" << tc.ln_mode;
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, D, H, W});
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, H, W});
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, W});
        }
        else if((N != 0) && (W != 0))
        {
            return std::vector<size_t>({N, W});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }
};

std::vector<LayerNormTestCase> LayerNormTestConfigs()
{ // n c d h w nomalized_dim eps ln_mode
    // clang-format off
    return {
        { 32,   1,   32,  32,  32  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},   // 32x32x32 based on VoxNet arch
        { 32,   1,   14,  14,  14  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 32,  32,   14,  14,  14  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 32,  32,   12,  12,  12  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 32,  32,   6,   6,   6   , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 256,  1,   32,  32,  32  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},   // 32x32x32 based on VoxNet arch
        { 256, 32,   14,  14,  14  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 256, 32,   12,  12,  12  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 256, 32,   6,   6,   6   , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 512,  1,   32,  32,  32  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},   // 32x32x32 based on VoxNet arch
        { 512, 32,   14,  14,  14  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 512, 32,   12,  12,  12  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 512, 32,   6,   6,   6   , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 32,   2,   32,  57,  125 , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},    // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        { 32,  32,   14,  25,  59  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 32,  32,   6,   10,  27  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 32,  32,   4,   6,   11  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 32,  32,   2,   2,   3   , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 32,  32,   32,  28,  62  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},    // Hand-gesture recognition CVPR 2015 paper Low Res Net Path
        { 32,  32,   14,  12,  29  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 32,  32,   6,   4,   12  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 32,  32,   4,   2,   2   , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        { 16,  32,   6,   50,  50  , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},    // Multi-view 3D convnet
        { 1,    3,   8,   240, 320 , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},     // 3D convet on video
        { 1,    3,   16,  240, 320 , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},     // 3D convet on video
        { 1,    3,   8,   128, 171 , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},     // 3D convet on video
        { 1,    3,   16,  128, 171 , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},     // 3D convet on video
        { 1,    3,   8,   112, 112 , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},     // 3D convet on video
        { 1,    3,   16,  112, 112 , 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},     // 3D convet on video
        { 32,   1,   32,  32,  32  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},          // 32x32x32 based on VoxNet arch
        { 32,   1,   14,  14,  14  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 32,  32,   14,  14,  14  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 32,  32,   12,  12,  12  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 32,  32,   6,   6,   6   , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 256,  1,   32,  32,  32  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},          // 32x32x32 based on VoxNet arch
        { 256, 32,   14,  14,  14  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 256, 32,   12,  12,  12  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 256, 32,   6,   6,   6   , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 512,  1,   32,  32,  32  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},          // 32x32x32 based on VoxNet arch
        { 512, 32,   14,  14,  14  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 512, 32,   12,  12,  12  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 512, 32,   6,   6,   6   , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 32,   2,   32,  57,  125 , 4, 1e-5, MIOPEN_WEIGHT_BIAS},           // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        { 32,  32,   14,  25,  59  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 32,  32,   6,   10,  27  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 32,  32,   4,   6,   11  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 32,  32,   2,   2,   3   , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 32,  32,   32,  28,  62  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},           // Hand-gesture recognition CVPR 2015 paper Low Res Net Path
        { 32,  32,   14,  12,  29  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 32,  32,   6,   4,   12  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 32,  32,   4,   2,   2   , 4, 1e-5, MIOPEN_WEIGHT_BIAS},
        { 16,  32,   6,   50,  50  , 4, 1e-5, MIOPEN_WEIGHT_BIAS},           // Multi-view 3D convnet
        { 1,   3,    8,   240, 320 , 4, 1e-5, MIOPEN_WEIGHT_BIAS},            // 3D convet on video
        { 1,   3,    16,  240, 320 , 4, 1e-5, MIOPEN_WEIGHT_BIAS},            // 3D convet on video
        { 1,   3,    8,   128, 171 , 4, 1e-5, MIOPEN_WEIGHT_BIAS},            // 3D convet on video
        { 1,   3,    16,  128, 171 , 4, 1e-5, MIOPEN_WEIGHT_BIAS},            // 3D convet on video
        { 1,   3,    8,   112, 112 , 4, 1e-5, MIOPEN_WEIGHT_BIAS},            // 3D convet on video
        { 1,   3,    16,  112, 112 , 4, 1e-5, MIOPEN_WEIGHT_BIAS},            // 3D convet on video
        {32,   4,    0,   4,   256 , 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        {64,   4,    0,   4,   256 , 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        {32,   4,    0,   4,   256 , 1, 1e-5, MIOPEN_WEIGHT_BIAS},
        {64,   4,    0,   4,   256 , 1, 1e-5, MIOPEN_WEIGHT_BIAS},
        {32,   0,    0,   0,   256 , 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        {64,   0,    0,   0,   256 , 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
        {32,   0,    0,   0,   256 , 1, 1e-5, MIOPEN_WEIGHT_BIAS},
        {64,   0,    0,   0,   256 , 1, 1e-5, MIOPEN_WEIGHT_BIAS}
      };
    // clang-format on
}

template <typename T = float>
struct LayerNormTest : public ::testing::TestWithParam<LayerNormTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle    = get_handle();
        layernorm_config = GetParam();
        auto gen_value   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        nomalized_dim = layernorm_config.nomalized_dim;
        eps           = layernorm_config.eps;
        ln_mode       = layernorm_config.ln_mode;

        auto in_dim = layernorm_config.GetInput();

        input = tensor<T>{in_dim}.generate(gen_value);

        std::vector<size_t> inner_dim;
        if(nomalized_dim == in_dim.size())
            inner_dim = {1};
        else
            inner_dim = {in_dim.begin() + nomalized_dim, in_dim.end()};

        if(ln_mode == MIOPEN_ELEMENTWISE_AFFINE)
        {
            auto gen_one  = [&](auto...) { return 1; };
            auto gen_zero = [&](auto...) { return 0; };
            weight        = tensor<T>{inner_dim}.generate(gen_one);
            bias          = tensor<T>{inner_dim}.generate(gen_zero);
        }
        else
        {
            weight = tensor<T>{inner_dim}.generate(gen_value);
            bias   = tensor<T>{inner_dim}.generate(gen_value);
        }

        std::vector<size_t> outer_dim;
        if(nomalized_dim == 0)
            outer_dim = {1};
        else
            outer_dim = {in_dim.begin(), in_dim.end() - (in_dim.size() - nomalized_dim)};

        output = tensor<T>{in_dim};
        mean   = tensor<T>{outer_dim};
        rstd   = tensor<T>{outer_dim};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(mean.begin(), mean.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(rstd.begin(), rstd.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{in_dim};
        ref_mean   = tensor<T>{outer_dim};
        ref_rstd   = tensor<T>{outer_dim};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(ref_mean.begin(), ref_mean.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(ref_rstd.begin(), ref_rstd.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        weight_dev = handle.Write(weight.data);
        bias_dev   = handle.Write(bias.data);
        output_dev = handle.Write(output.data);
        mean_dev   = handle.Write(mean.data);
        rstd_dev   = handle.Write(rstd.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_layernorm_forward<T>(
            input, weight, bias, ref_output, ref_mean, ref_rstd, eps, nomalized_dim, ln_mode);
        miopenStatus_t status;

        status = miopen::LayerNormForward(handle,
                                          input.desc,
                                          input_dev.get(),
                                          weight.desc,
                                          weight_dev.get(),
                                          bias.desc,
                                          bias_dev.get(),
                                          output.desc,
                                          output_dev.get(),
                                          mean.desc,
                                          mean_dev.get(),
                                          rstd.desc,
                                          rstd_dev.get(),
                                          ln_mode,
                                          eps,
                                          nomalized_dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
        mean.data   = handle.Read<T>(mean_dev, mean.data.size());
        rstd.data   = handle.Read<T>(rstd_dev, rstd.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            threshold *= 8.0;
        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold)
            << "Error output beyond tolerance Error:" << error << ",  Threshold: " << threshold;

        error = miopen::rms_range(ref_mean, mean);
        EXPECT_TRUE(miopen::range_distance(ref_mean) == miopen::range_distance(mean));
        EXPECT_TRUE(error < threshold)
            << "Error mean beyond tolerance Error:" << error << ",  Threshold: " << threshold;

        error = miopen::rms_range(ref_rstd, rstd);
        EXPECT_TRUE(miopen::range_distance(ref_rstd) == miopen::range_distance(rstd));
        EXPECT_TRUE(error < threshold * 4) << "Error rstd beyond tolerance Error:" << error
                                           << ",  Threshold x 4: " << threshold * 4;
    }
    LayerNormTestCase layernorm_config;

    tensor<T> input;
    tensor<T> weight;
    tensor<T> bias;
    tensor<T> output;
    tensor<T> mean;
    tensor<T> rstd;

    tensor<T> ref_output;
    tensor<T> ref_mean;
    tensor<T> ref_rstd;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr bias_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr mean_dev;
    miopen::Allocator::ManageDataPtr rstd_dev;

    size_t nomalized_dim;
    float eps;
    miopenNormMode_t ln_mode;
};
