/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include <miopen/t5layernorm.hpp>
#include <miopen/miopen.h>

template <class T>
void cpu_t5layernorm_forward(tensor<T> x,
                             tensor<T> weight,
                             tensor<T>& ref_y,
                             tensor<T>& ref_rstd,
                             float eps,
                             miopenNormMode_t mode)
{
    auto dims         = x.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = dims[dims.size() - 1];

    for(size_t i = 0ULL; i < dims.size() - 1; ++i)
    {
        outer_size *= dims[i];
    }

    par_ford(outer_size)([&](int32_t o) {
        float pvar = 0;

        ford(inner_size)([&](int32_t i) {
            float tmp = static_cast<float>(x[o * inner_size + i]);
            pvar += tmp * tmp;
        });

        pvar        = pvar / inner_size;
        float prstd = 1 / sqrt(pvar + eps);

        ref_rstd[o] = static_cast<T>(prstd);

        ford(inner_size)([&](int32_t i) {
            float pweight = mode ? static_cast<float>(weight[i]) : 1;
            ref_y[o * inner_size + i] =
                static_cast<T>(static_cast<float>(x[o * inner_size + i]) * prstd * pweight);
        });
    });
}

template <class T>
void cpu_t5layernorm_backward(tensor<T> dy,
                              tensor<T> x,
                              tensor<T> weight,
                              tensor<T> rstd,
                              tensor<T>& ref_dx,
                              miopenNormMode_t mode)
{
    auto dims         = dy.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = dims[dims.size() - 1];

    for(size_t i = 0ULL; i < dims.size() - 1; ++i)
    {
        outer_size *= dims[i];
    }

    par_ford(outer_size)([&](int32_t o) {
        float sum = 0;

        ford(inner_size)([&](int32_t i) {
            float pweight = mode ? static_cast<float>(weight[i]) : 1;
            float pdy     = (dy.GetSize() != 0) ? static_cast<float>(dy[o * inner_size + i]) : 0;
            float px      = static_cast<float>(x[o * inner_size + i]);
            sum += pdy * px * pweight;
        });

        float s     = 1 / static_cast<float>(inner_size);
        float prstd = static_cast<float>(rstd[o]);
        float a     = sum * prstd * prstd * prstd * s;

        ford(inner_size)([&](int32_t i) {
            float pweight = mode ? static_cast<float>(weight[i]) : 1;
            float pdy     = (dy.GetSize() != 0) ? static_cast<float>(dy[o * inner_size + i]) : 0;

            float val = prstd * pdy * pweight - a * static_cast<float>(x[o * inner_size + i]);
            ref_dx[o * inner_size + i] = static_cast<T>(val);
        });
    });
}

template <class T>
void cpu_t5layernorm_backward_weight(
    tensor<T> dy, tensor<T> x, tensor<T> rstd, tensor<T>& ref_dw, miopenNormMode_t mode)
{
    auto dims         = dy.desc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = dims[dims.size() - 1];

    for(size_t i = 0ULL; i < dims.size() - 1; ++i)
    {
        outer_size *= dims[i];
    }

    par_ford(inner_size)([&](int32_t o) {
        float sum = 0;

        ford(outer_size)([&](int32_t i) {
            float prstd = static_cast<float>(rstd[i]);
            float pdy   = (dy.GetSize() != 0) ? static_cast<float>(dy[i * inner_size + o]) : 0;
            float px    = static_cast<float>(x[i * inner_size + o]);

            sum += pdy * px * prstd;
        });

        ref_dw[o] = sum;
    });
}

struct T5LayerNormTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    float eps;
    miopenNormMode_t ln_mode;
    friend std::ostream& operator<<(std::ostream& os, const T5LayerNormTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " eps:" << tc.eps << " LayerNorm_mode:" << tc.ln_mode;
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

std::vector<T5LayerNormTestCase> T5LayerNormTestConfigs()
{ // n c d h w eps ln_mode
    // clang-format off
    return {
        { 32,   1,   32,  32,  32, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},   // 32x32x32 based on VoxNet arch
        { 32,   1,   14,  14,  14, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 32,  32,   14,  14,  14, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 32,  32,   12,  12,  12, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 32,  32,    6,   6,   6, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 256,  1,   32,  32,  32, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},   // 32x32x32 based on VoxNet arch
        { 256, 32,   14,  14,  14, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 256, 32,   12,  12,  12, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 256, 32,    6,   6,   6, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 512,  1,   32,  32,  32, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},   // 32x32x32 based on VoxNet arch
        { 512, 32,   14,  14,  14, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 512, 32,   12,  12,  12, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 512, 32,    6,   6,   6, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 32,   2,   32,  57, 125, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},    // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        { 32,  32,   14,  25,  59, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 32,  32,    6,  10,  27, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 32,  32,    4,   6,  11, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 32,  32,    2,   2,   3, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 32,  32,   32,  28,  62, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},    // Hand-gesture recognition CVPR 2015 paper Low Res Net Path
        { 32,  32,   14,  12,  29, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 32,  32,    6,   4,  12, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 32,  32,    4,   2,   2, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        { 16,  32,    6,  50,  50, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},    // Multi-view 3D convnet
        { 1,    3,    8, 240, 320, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},     // 3D convet on video
        { 1,    3,   16, 240, 320, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},     // 3D convet on video
        { 1,    3,    8, 128, 171, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},     // 3D convet on video
        { 1,    3,   16, 128, 171, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},     // 3D convet on video
        { 1,    3,    8, 112, 112, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},     // 3D convet on video
        { 1,    3,   16, 112, 112, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},     // 3D convet on video
        { 32,   1,   32,  32,  32, 1e-5, MIOPEN_WEIGHT_BIAS_T5},          // 32x32x32 based on VoxNet arch
        { 32,   1,   14,  14,  14, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 32,  32,   14,  14,  14, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 32,  32,   12,  12,  12, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 32,  32,    6,   6,   6, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 256,  1,   32,  32,  32, 1e-5, MIOPEN_WEIGHT_BIAS_T5},          // 32x32x32 based on VoxNet arch
        { 256, 32,   14,  14,  14, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 256, 32,   12,  12,  12, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 256, 32,    6,   6,   6, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 512,  1,   32,  32,  32, 1e-5, MIOPEN_WEIGHT_BIAS_T5},          // 32x32x32 based on VoxNet arch
        { 512, 32,   14,  14,  14, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 512, 32,   12,  12,  12, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 512, 32,    6,   6,   6, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 32,   2,   32,  57, 125, 1e-5, MIOPEN_WEIGHT_BIAS_T5},           // Hand-gesture recognition CVPR 2015 paper High Res Net Path
        { 32,  32,   14,  25,  59, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 32,  32,    6,  10,  27, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 32,  32,    4,   6,  11, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 32,  32,    2,   2,   3, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 32,  32,   32,  28,  62, 1e-5, MIOPEN_WEIGHT_BIAS_T5},           // Hand-gesture recognition CVPR 2015 paper Low Res Net Path
        { 32,  32,   14,  12,  29, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 32,  32,    6,   4,  12, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 32,  32,    4,   2,   2, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        { 16,  32,    6,  50,  50, 1e-5, MIOPEN_WEIGHT_BIAS_T5},           // Multi-view 3D convnet
        { 1,    3,    8, 240, 320, 1e-5, MIOPEN_WEIGHT_BIAS_T5},            // 3D convet on video
        { 1,    3,   16, 240, 320, 1e-5, MIOPEN_WEIGHT_BIAS_T5},            // 3D convet on video
        { 1,    3,    8, 128, 171, 1e-5, MIOPEN_WEIGHT_BIAS_T5},            // 3D convet on video
        { 1,    3,   16, 128, 171, 1e-5, MIOPEN_WEIGHT_BIAS_T5},            // 3D convet on video
        { 1,    3,    8, 112, 112, 1e-5, MIOPEN_WEIGHT_BIAS_T5},            // 3D convet on video
        { 1,    3,   16, 112, 112, 1e-5, MIOPEN_WEIGHT_BIAS_T5},            // 3D convet on video
        {32,    4,    0,   4, 256, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        {64,    4,    0,   4, 256, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        {32,    4,    0,   4, 256, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        {64,    4,    0,   4, 256, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        {32,    0,    0,   0, 256, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        {64,    0,    0,   0, 256, 1e-5, MIOPEN_ELEMENTWISE_AFFINE_T5},
        {32,    0,    0,   0, 256, 1e-5, MIOPEN_WEIGHT_BIAS_T5},
        {64,    0,    0,   0, 256, 1e-5, MIOPEN_WEIGHT_BIAS_T5}
      };
    // clang-format on
}

template <typename T = float>
struct T5LayerNormFwdTest : public ::testing::TestWithParam<T5LayerNormTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle      = get_handle();
        t5layernorm_config = GetParam();
        auto gen_value     = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        eps     = t5layernorm_config.eps;
        ln_mode = t5layernorm_config.ln_mode;

        auto in_dim = t5layernorm_config.GetInput();
        x           = tensor<T>{in_dim}.generate(gen_value);

        std::vector<size_t> inner_dim = {in_dim[in_dim.size() - 1]};

        if(ln_mode == MIOPEN_ELEMENTWISE_AFFINE_T5)
        {
            auto gen_one = [&](auto...) { return 1; };
            weight       = tensor<T>{inner_dim}.generate(gen_one);
        }
        else
        {
            weight = tensor<T>{inner_dim}.generate(gen_value);
        }

        std::vector<size_t> outer_dim;

        outer_dim = {in_dim.begin(), in_dim.end() - 1};

        y    = tensor<T>{in_dim};
        rstd = tensor<T>{outer_dim};
        std::fill(y.begin(), y.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(rstd.begin(), rstd.end(), std::numeric_limits<T>::quiet_NaN());

        ref_y    = tensor<T>{in_dim};
        ref_rstd = tensor<T>{outer_dim};
        std::fill(ref_y.begin(), ref_y.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(ref_rstd.begin(), ref_rstd.end(), std::numeric_limits<T>::quiet_NaN());

        x_dev      = handle.Write(x.data);
        weight_dev = handle.Write(weight.data);
        y_dev      = handle.Write(y.data);
        rstd_dev   = handle.Write(rstd.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_t5layernorm_forward<T>(x, weight, ref_y, ref_rstd, eps, ln_mode);

        miopenStatus_t status;
        status = miopen::T5LayerNormForward(handle,
                                            x.desc,
                                            x_dev.get(),
                                            weight.desc,
                                            weight_dev.get(),
                                            y.desc,
                                            y_dev.get(),
                                            rstd.desc,
                                            rstd_dev.get(),
                                            ln_mode,
                                            eps);
        EXPECT_EQ(status, miopenStatusSuccess);

        y.data    = handle.Read<T>(y_dev, y.data.size());
        rstd.data = handle.Read<T>(rstd_dev, rstd.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        // In the case of layernorm, there is a cumulative sum operation, and in the case of
        // floating point operation, the result value can change if the order of the summed values
        // is changed. So apply a threshold that is 10 times larger than other operations.
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            threshold *= 8.0;
        auto error = miopen::rms_range(ref_y, y);
        EXPECT_TRUE(miopen::range_distance(ref_y) == miopen::range_distance(y));
        EXPECT_TRUE(error < threshold)
            << "Error y beyond tolerance Error:" << error << ",  Threshold: " << threshold;

        error = miopen::rms_range(ref_rstd, rstd);
        EXPECT_TRUE(miopen::range_distance(ref_rstd) == miopen::range_distance(rstd));
        EXPECT_TRUE(error < threshold * 4) << "Error rstd beyond tolerance Error:" << error
                                           << ",  Threshold x 4: " << threshold * 4;
    }
    T5LayerNormTestCase t5layernorm_config;

    tensor<T> x;
    tensor<T> weight;
    tensor<T> y;
    tensor<T> rstd;

    tensor<T> ref_y;
    tensor<T> ref_rstd;

    miopen::Allocator::ManageDataPtr x_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr y_dev;
    miopen::Allocator::ManageDataPtr rstd_dev;

    float eps;
    miopenNormMode_t ln_mode;
};

template <typename T = float>
struct T5LayerNormBwdTest : public ::testing::TestWithParam<T5LayerNormTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle      = get_handle();
        t5layernorm_config = GetParam();
        auto gen_value     = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        ln_mode = t5layernorm_config.ln_mode;

        auto in_dim                   = t5layernorm_config.GetInput();
        std::vector<size_t> outer_dim = {in_dim.begin(), in_dim.end() - 1};

        x    = tensor<T>{in_dim}.generate(gen_value);
        dy   = tensor<T>{in_dim}.generate(gen_value);
        rstd = tensor<T>{outer_dim}.generate(gen_value);

        std::vector<size_t> inner_dim = {in_dim[in_dim.size() - 1]};

        if(ln_mode == MIOPEN_ELEMENTWISE_AFFINE_T5)
        {
            auto gen_one = [&](auto...) { return 1; };
            weight       = tensor<T>{inner_dim}.generate(gen_one);
        }
        else
        {
            weight = tensor<T>{inner_dim}.generate(gen_value);
        }

        dx = tensor<T>{in_dim};
        dw = tensor<T>{inner_dim};
        std::fill(dx.begin(), dx.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(dw.begin(), dw.end(), std::numeric_limits<T>::quiet_NaN());

        ref_dx = tensor<T>{in_dim};
        ref_dw = tensor<T>{inner_dim};
        std::fill(ref_dx.begin(), ref_dx.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(ref_dw.begin(), ref_dw.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> workspace_dims;

        ws_sizeInBytes = miopen::GetT5LayerNormBackwardWorkspaceSize(
            handle, dy.desc, x.desc, weight.desc, rstd.desc, dx.desc, dw.desc, ln_mode);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        workspace_dims.push_back(ws_sizeInBytes / sizeof(T));
        if(ws_sizeInBytes != 0)
        {
            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), std::numeric_limits<T>::quiet_NaN());
            workspace_dev = handle.Write(workspace.data);
        }

        x_dev      = handle.Write(x.data);
        weight_dev = handle.Write(weight.data);
        rstd_dev   = handle.Write(rstd.data);
        dy_dev     = handle.Write(dy.data);
        dx_dev     = handle.Write(dx.data);
        dw_dev     = handle.Write(dw.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_t5layernorm_backward<T>(dy, x, weight, rstd, ref_dx, ln_mode);
        cpu_t5layernorm_backward_weight<T>(dy, x, rstd, ref_dw, ln_mode);

        miopenStatus_t status;

        status = miopen::T5LayerNormBackward(handle,
                                             workspace_dev.get(),
                                             ws_sizeInBytes,
                                             dy.desc,
                                             dy_dev.get(),
                                             x.desc,
                                             x_dev.get(),
                                             weight.desc,
                                             weight_dev.get(),
                                             rstd.desc,
                                             rstd_dev.get(),
                                             dx.desc,
                                             dx_dev.get(),
                                             dw.desc,
                                             dw_dev.get(),
                                             ln_mode);

        EXPECT_EQ(status, miopenStatusSuccess);

        dx.data = handle.Read<T>(dx_dev, dx.data.size());
        dw.data = handle.Read<T>(dw_dev, dw.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        // In the case of layernorm, there is a cumulative sum operation, and in the case of
        // floating point operation, the result value can change if the order of the summed values
        // is changed. So apply a threshold that is 10 times larger than other operations.
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            threshold *= 80.0;

        auto error = miopen::rms_range(ref_dx, dx);
        EXPECT_TRUE(miopen::range_distance(ref_dx) == miopen::range_distance(dx));
        EXPECT_TRUE(error < threshold)
            << "Error dx beyond tolerance Error:" << error << ",  Threshold: " << threshold;
        error = miopen::rms_range(ref_dw, dw);
        EXPECT_TRUE(miopen::range_distance(ref_dw) == miopen::range_distance(dw));
        EXPECT_TRUE(error < threshold * 2)
            << "Error dw beyond tolerance Error:" << error << ",  Threshold x 2: " << threshold * 2;
    }
    T5LayerNormTestCase t5layernorm_config;

    tensor<T> x;
    tensor<T> weight;
    tensor<T> rstd;
    tensor<T> dy;
    tensor<T> dx;
    tensor<T> dw;
    tensor<T> workspace;

    tensor<T> ref_dx;
    tensor<T> ref_dw;

    miopen::Allocator::ManageDataPtr x_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr rstd_dev;
    miopen::Allocator::ManageDataPtr dy_dev;
    miopen::Allocator::ManageDataPtr dx_dev;
    miopen::Allocator::ManageDataPtr dw_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;

    miopenNormMode_t ln_mode;
};
