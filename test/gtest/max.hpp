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
#include "cpu_max.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/max.hpp>
#include <miopen/miopen.h>

struct MaxTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int32_t dim;
    friend std::ostream& operator<<(std::ostream& os, const MaxTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " dim:" << tc.dim;
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
        else if((N != 0))
        {
            return std::vector<size_t>({N});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }
};

std::vector<MaxTestCase> MaxTestConfigs()
{ // n c d h w dim
    // clang-format off
    return {
        { 16, 21,   0, 513, 513, 1 },   //deeplabv3m
        { 24, 21,   0, 513, 513, 1 },   //deeplabv3r
        { 64, 21,   0, 230, 333, 1 },   //fcn_resnet_lraspp
        { 64, 21,   0, 215, 288, 1 },
        { 1,  21,   0, 333, 500, 1 },   //stdc
        { 1,  21,   0, 375, 500, 1 },
        { 15, 21,   0, 256, 256, 1 },   //unet
        { 22, 21,   0, 256, 256, 1 },
        { 21, 412,  0, 0,   500, 0 },
        { 21, 333,  0, 0,   500, 0 }
      };
    // clang-format on
}

template <typename T = float>
struct MaxTest : public ::testing::TestWithParam<MaxTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        max_config     = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        dim = max_config.dim;

        auto in_dims = max_config.GetInput();

        input = tensor<T>{in_dims}.generate(gen_value);

        std::vector<size_t> out_dims;

        for(int i = 0; i < in_dims.size(); i++)
        {
            if(i != dim)
            {
                out_dims.push_back(in_dims[i]);
            }
        }

        output = tensor<T>{out_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        indice = tensor<int>{out_dims};
        std::fill(indice.begin(), indice.end(), std::numeric_limits<int>::quiet_NaN());

        ref_indice = tensor<int>{out_dims};
        std::fill(ref_indice.begin(), ref_indice.end(), std::numeric_limits<int>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
        indice_dev = handle.Write(indice.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_max_forward<T>(input, ref_output, ref_indice, dim);
        miopenStatus_t status;

        status = miopen::MaxForward(handle,
                                    input.desc,
                                    input_dev.get(),
                                    output.desc,
                                    output_dev.get(),
                                    indice.desc,
                                    indice_dev.get(),
                                    dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
        indice.data = handle.Read<int>(indice_dev, indice.data.size());
    }

    void Verify()
    {
        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(std::abs(static_cast<float>(error)) == 0.0f)
            << "Error output beyond tolerance Error:" << error;

        auto indice_error = miopen::rms_range(ref_indice, indice);

        EXPECT_TRUE(miopen::range_distance(ref_indice) == miopen::range_distance(indice));
        EXPECT_TRUE(std::abs(static_cast<float>(indice_error)) == 0.0f)
            << "Error indice beyond tolerance Error:" << error;
    }
    MaxTestCase max_config;

    tensor<T> input;
    tensor<T> output;
    tensor<int> indice;

    tensor<T> ref_output;
    tensor<int> ref_indice;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr indice_dev;

    int32_t dim;
};
