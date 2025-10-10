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

template <class T>
void cpu_layernorm_backward(tensor<T> dy,
                            tensor<T> x,
                            tensor<T> weight,
                            tensor<T> mean,
                            tensor<T> rstd,
                            tensor<T>& ref_dx,
                            int32_t dim,
                            miopenNormMode_t mode)
{
    auto dims         = dy.desc.GetLengths();
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
        float sum_dy_weight   = 0;
        float sum_dy_weight_x = 0;

        ford(inner_size)([&](int32_t i) {
            float pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE) ? 1 : static_cast<float>(weight[i]);
            float pdy     = (dy.GetSize() != 0) ? static_cast<float>(dy[o * inner_size + i]) : 0;
            float px      = static_cast<float>(x[o * inner_size + i]);
            sum_dy_weight += pdy * pweight;
            sum_dy_weight_x += pdy * px * pweight;
        });

        float scale = 1.0f / static_cast<float>(inner_size);
        float prstd = static_cast<float>(rstd[o]);
        float pmean = static_cast<float>(mean[o]);
        float a     = prstd * prstd * prstd * scale * (sum_dy_weight_x - sum_dy_weight * pmean);
        float b     = prstd * sum_dy_weight * scale - a * pmean;

        ford(inner_size)([&](int32_t i) {
            float pweight = (mode == MIOPEN_ELEMENTWISE_AFFINE) ? 1 : static_cast<float>(weight[i]);
            float pdy     = (dy.GetSize() != 0) ? static_cast<float>(dy[o * inner_size + i]) : 0;

            float val = prstd * pdy * pweight - a * static_cast<float>(x[o * inner_size + i]) - b;
            ref_dx[o * inner_size + i] = static_cast<T>(val);
        });
    });
}

template <class T>
void cpu_layernorm_backward_weight_bias(tensor<T> dy,
                                        tensor<T> x,
                                        tensor<T> mean,
                                        tensor<T> rstd,
                                        tensor<T>& ref_dw,
                                        tensor<T>& ref_db,
                                        int32_t dim)
{
    auto dims         = dy.desc.GetLengths();
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

    par_ford(inner_size)([&](int32_t i) {
        float sum_dw = 0;
        float sum_db = 0;

        ford(outer_size)([&](int32_t o) {
            float prstd = static_cast<float>(rstd[o]);
            float pmean = static_cast<float>(mean[o]);
            float pdy   = (dy.GetSize() != 0) ? static_cast<float>(dy[o * inner_size + i]) : 0;
            float px    = static_cast<float>(x[o * inner_size + i]);

            sum_dw += pdy * (px - pmean) * prstd;
            sum_db += pdy;
        });

        ref_dw[i] = sum_dw;
        ref_db[i] = sum_db;
    });
}

struct LayerNormTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    size_t normalized_dim;
    float eps;
    miopenNormMode_t ln_mode;
    friend std::ostream& operator<<(std::ostream& os, const LayerNormTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " dim:" << tc.normalized_dim << " eps:" << tc.eps
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
{ // n c d h w normalized_dim eps ln_mode
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
struct LayerNormFwdTest : public ::testing::TestWithParam<LayerNormTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle    = get_handle();
        layernorm_config = GetParam();
        auto gen_value   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        normalized_dim = layernorm_config.normalized_dim;
        eps            = layernorm_config.eps;
        ln_mode        = layernorm_config.ln_mode;

        auto in_dim = layernorm_config.GetInput();

        input = tensor<T>{in_dim}.generate(gen_value);

        std::vector<size_t> inner_dim;
        if(normalized_dim == in_dim.size())
            inner_dim = {1};
        else
            inner_dim = {in_dim.begin() + normalized_dim, in_dim.end()};

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
        if(normalized_dim == 0)
            outer_dim = {1};
        else
            outer_dim = {in_dim.begin(), in_dim.begin() + normalized_dim};

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
            input, weight, bias, ref_output, ref_mean, ref_rstd, eps, normalized_dim, ln_mode);
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
                                          normalized_dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
        mean.data   = handle.Read<T>(mean_dev, mean.data.size());
        rstd.data   = handle.Read<T>(rstd_dev, rstd.data.size());
    }

    void Verify()
    {
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 4e-3;

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

    size_t normalized_dim;
    float eps;
    miopenNormMode_t ln_mode;
};

template <typename T = float>
struct LayerNormBwdTest : public ::testing::TestWithParam<LayerNormTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle    = get_handle();
        layernorm_config = GetParam();
        auto gen_value   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        normalized_dim = layernorm_config.normalized_dim;
        ln_mode        = layernorm_config.ln_mode;

        auto in_dim = layernorm_config.GetInput();

        x = tensor<T>{in_dim}.generate(gen_value);

        std::vector<size_t> inner_dim;
        if(normalized_dim == in_dim.size())
            inner_dim = {1};
        else
            inner_dim = {in_dim.begin() + normalized_dim, in_dim.end()};

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
        if(normalized_dim == 0)
            outer_dim = {1};
        else
            outer_dim = {in_dim.begin(), in_dim.begin() + normalized_dim};

        x    = tensor<T>{in_dim}.generate(gen_value);
        dy   = tensor<T>{in_dim}.generate(gen_value);
        mean = tensor<T>{outer_dim}.generate(gen_value);
        rstd = tensor<T>{outer_dim}.generate(gen_value);

        dx = tensor<T>{in_dim};
        dw = tensor<T>{inner_dim};
        db = tensor<T>{inner_dim};
        std::fill(dx.begin(), dx.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(dw.begin(), dw.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(db.begin(), db.end(), std::numeric_limits<T>::quiet_NaN());

        ref_dx = tensor<T>{in_dim};
        ref_dw = tensor<T>{inner_dim};
        ref_db = tensor<T>{inner_dim};
        std::fill(ref_dx.begin(), ref_dx.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(ref_dw.begin(), ref_dw.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(ref_db.begin(), ref_db.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> workspace_dims;

        ws_sizeInBytes = miopen::GetLayerNormBackwardWorkspaceSize(handle,
                                                                   dy.desc,
                                                                   x.desc,
                                                                   weight.desc,
                                                                   mean.desc,
                                                                   rstd.desc,
                                                                   dx.desc,
                                                                   dw.desc,
                                                                   db.desc,
                                                                   ln_mode,
                                                                   normalized_dim);

        workspace_dims.push_back(ws_sizeInBytes / sizeof(T));
        if(ws_sizeInBytes != 0)
        {
            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), std::numeric_limits<T>::quiet_NaN());
            workspace_dev = handle.Write(workspace.data);
        }

        x_dev      = handle.Write(x.data);
        weight_dev = handle.Write(weight.data);
        mean_dev   = handle.Write(mean.data);
        rstd_dev   = handle.Write(rstd.data);
        dy_dev     = handle.Write(dy.data);
        dx_dev     = handle.Write(dx.data);
        dw_dev     = handle.Write(dw.data);
        db_dev     = handle.Write(db.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_layernorm_backward<T>(dy, x, weight, mean, rstd, ref_dx, normalized_dim, ln_mode);
        cpu_layernorm_backward_weight_bias<T>(dy, x, mean, rstd, ref_dw, ref_db, normalized_dim);

        miopenStatus_t status;

        status = miopen::LayerNormBackward(handle,
                                           workspace_dev.get(),
                                           ws_sizeInBytes,
                                           dy.desc,
                                           dy_dev.get(),
                                           x.desc,
                                           x_dev.get(),
                                           weight.desc,
                                           weight_dev.get(),
                                           mean.desc,
                                           mean_dev.get(),
                                           rstd.desc,
                                           rstd_dev.get(),
                                           dx.desc,
                                           dx_dev.get(),
                                           dw.desc,
                                           dw_dev.get(),
                                           db.desc,
                                           db_dev.get(),
                                           ln_mode,
                                           normalized_dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        dx.data = handle.Read<T>(dx_dev, dx.data.size());
        dw.data = handle.Read<T>(dw_dev, dw.data.size());
        db.data = handle.Read<T>(db_dev, db.data.size());
    }

    void Verify()
    {
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 4e-3;

        auto error = miopen::rms_range(ref_dx, dx);
        EXPECT_TRUE(miopen::range_distance(ref_dx) == miopen::range_distance(dx));
        EXPECT_TRUE(error < threshold)
            << "Error dx beyond tolerance Error:" << error << ",  Threshold: " << threshold;
        error = miopen::rms_range(ref_dw, dw);
        EXPECT_TRUE(miopen::range_distance(ref_dw) == miopen::range_distance(dw));
        EXPECT_TRUE(error < threshold * 2)
            << "Error dw beyond tolerance Error:" << error << ",  Threshold x 2: " << threshold * 2;
        error = miopen::rms_range(ref_db, db);
        EXPECT_TRUE(miopen::range_distance(ref_db) == miopen::range_distance(db));
        EXPECT_TRUE(error < threshold * 2)
            << "Error db beyond tolerance Error:" << error << ",  Threshold x 2: " << threshold * 2;
    }

    LayerNormTestCase layernorm_config;

    tensor<T> x;
    tensor<T> weight;
    tensor<T> bias;
    tensor<T> mean;
    tensor<T> rstd;
    tensor<T> dy;
    tensor<T> dx;
    tensor<T> dw;
    tensor<T> db;
    tensor<T> workspace;

    tensor<T> ref_dx;
    tensor<T> ref_dw;
    tensor<T> ref_db;

    miopen::Allocator::ManageDataPtr x_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr mean_dev;
    miopen::Allocator::ManageDataPtr rstd_dev;
    miopen::Allocator::ManageDataPtr dy_dev;
    miopen::Allocator::ManageDataPtr dx_dev;
    miopen::Allocator::ManageDataPtr dw_dev;
    miopen::Allocator::ManageDataPtr db_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;

    int32_t normalized_dim;
    miopenNormMode_t ln_mode;
};
