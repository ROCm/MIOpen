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
#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/groupnorm.hpp>

#include "tensor_holder.hpp"
#include "cpu_groupnorm.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "../driver/tensor_driver.hpp"
#include "verify.hpp"
#include <random>

struct GroupNormTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    size_t num_groups;
    float eps;
    miopenNormMode_t mode;
    friend std::ostream& operator<<(std::ostream& os, const GroupNormTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " num_groups:" << tc.num_groups << " eps:" << tc.eps
                  << " mode:" << tc.mode;
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
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }
};

std::vector<GroupNormTestCase> GroupNormTestConfigs()
{ // n c d h w num_groups eps mode

    return {{32, 1, 32, 32, 32, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 1, 14, 14, 14, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 32, 14, 14, 14, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 32, 12, 12, 12, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 32, 6, 6, 6, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {256, 1, 32, 32, 32, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {256, 32, 14, 14, 14, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {256, 32, 12, 12, 12, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {256, 32, 6, 6, 6, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {512, 1, 32, 32, 32, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {512, 32, 14, 14, 14, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {512, 32, 12, 12, 12, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {512, 32, 6, 6, 6, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 2, 32, 57, 125, 2, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 32, 14, 25, 59, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 32, 6, 10, 27, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 32, 4, 6, 11, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 32, 2, 2, 3, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 32, 32, 28, 62, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 32, 14, 12, 29, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 32, 6, 4, 12, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 32, 4, 2, 2, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {16, 32, 6, 50, 50, 4, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            // {1, 3, 8, 240, 320, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            // {1, 3, 16, 240, 320, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            // {1, 3, 8, 128, 171, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            // {1, 3, 16, 128, 171, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            // {1, 3, 8, 112, 112, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            // {1, 3, 16, 112, 112, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 1, 32, 32, 32, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 1, 14, 14, 14, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 32, 14, 14, 14, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 32, 12, 12, 12, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 32, 6, 6, 6, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {256, 1, 32, 32, 32, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            {256, 32, 14, 14, 14, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {256, 32, 12, 12, 12, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {256, 32, 6, 6, 6, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {512, 1, 32, 32, 32, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            {512, 32, 14, 14, 14, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {512, 32, 12, 12, 12, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {512, 32, 6, 6, 6, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 2, 32, 57, 125, 2, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 32, 14, 25, 59, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 32, 6, 10, 27, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 32, 4, 6, 11, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 32, 2, 2, 3, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 32, 32, 28, 62, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 32, 14, 12, 29, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 32, 6, 4, 12, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 32, 4, 2, 2, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            {16, 32, 6, 50, 50, 4, 1e-5, MIOPEN_WEIGHT_BIAS},
            // {1, 3, 8, 240, 320, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            // {1, 3, 16, 240, 320, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            // {1, 3, 8, 128, 171, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            // {1, 3, 16, 128, 171, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            // {1, 3, 8, 112, 112, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            // {1, 3, 16, 112, 112, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 4, 0, 4, 256, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {64, 4, 0, 4, 256, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 4, 0, 4, 256, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            {64, 4, 0, 4, 256, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            {32, 1, 0, 0, 256, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {64, 1, 0, 0, 256, 1, 1e-5, MIOPEN_ELEMENTWISE_AFFINE},
            {32, 1, 0, 0, 256, 1, 1e-5, MIOPEN_WEIGHT_BIAS},
            {64, 1, 0, 0, 256, 1, 1e-5, MIOPEN_WEIGHT_BIAS}};
}

template <typename T = float>
struct GroupNormTest : public ::testing::TestWithParam<GroupNormTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle    = get_handle();
        groupnorm_config = GetParam();
        auto gen_value   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        num_groups = groupnorm_config.num_groups;
        eps        = groupnorm_config.eps;
        mode       = groupnorm_config.mode;

        std::vector<size_t> inout_dim       = groupnorm_config.GetInput();
        std::vector<size_t> weight_bias_dim = {inout_dim[1]};
        std::vector<size_t> mean_rstd_dim   = {inout_dim[0], num_groups};

        input = tensor<T>{inout_dim}.generate(gen_value);

        if(mode == MIOPEN_ELEMENTWISE_AFFINE)
        {
            auto gen_one  = [&](auto...) { return 1; };
            auto gen_zero = [&](auto...) { return 0; };
            weight        = tensor<T>{weight_bias_dim}.generate(gen_one);
            bias          = tensor<T>{weight_bias_dim}.generate(gen_zero);
        }
        else
        {
            weight = tensor<T>{weight_bias_dim}.generate(gen_value);
            bias   = tensor<T>{weight_bias_dim}.generate(gen_value);
        }
        output = tensor<T>{inout_dim};
        mean   = tensor<T>{mean_rstd_dim};
        rstd   = tensor<T>{mean_rstd_dim};

        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(mean.begin(), mean.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(rstd.begin(), rstd.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{inout_dim};
        ref_mean   = tensor<T>{mean_rstd_dim};
        ref_rstd   = tensor<T>{mean_rstd_dim};

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

        cpu_groupnorm_forward<T>(
            input, weight, bias, ref_output, ref_mean, ref_rstd, num_groups, eps, mode);
        miopenStatus_t status;

        status = miopen::GroupNormForward(handle,
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
                                          mode,
                                          num_groups,
                                          eps);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
        mean.data   = handle.Read<T>(mean_dev, mean.data.size());
        rstd.data   = handle.Read<T>(rstd_dev, rstd.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 1000) << "Error output beyond tolerance Error:" << error
                                              << ",  Thresholdx1000: " << threshold * 1000;

        error = miopen::rms_range(ref_mean, mean);
        EXPECT_TRUE(miopen::range_distance(ref_mean) == miopen::range_distance(mean));
        EXPECT_TRUE(error < threshold * 50) << "Error mean beyond tolerance Error:" << error
                                            << ",  Thresholdx50: " << threshold * 50;

        error = miopen::rms_range(ref_rstd, rstd);
        EXPECT_TRUE(miopen::range_distance(ref_rstd) == miopen::range_distance(rstd));
        EXPECT_TRUE(error < threshold * 2000) << "Error rstd beyond tolerance Error:" << error
                                              << ",  Thresholdx2000: " << threshold * 2000;
    }
    GroupNormTestCase groupnorm_config;

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

    size_t num_groups;
    float eps;
    miopenNormMode_t mode;
};
