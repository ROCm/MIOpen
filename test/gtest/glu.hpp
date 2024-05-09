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
#include "cpu_glu.hpp"
#include "get_handle.hpp"
#include "miopen/allocator.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/glu.hpp>
#include <numeric>

struct GLUTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int32_t dim;
    friend std::ostream& operator<<(std::ostream& os, const GLUTestCase& tc)
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

std::vector<GLUTestCase> GLUTestConfigs()
{ // n c d h w dim
    // clang-format off
    return {
        { 2,    320,   4,  4,   4,    0},
        { 32,    64,   3,  3,    3,     0},
        { 64,    3,  0,  11,    11,     0},
        { 256,    256,  0,  1,    1,     0},
        { 64,    64,  0,  7,    7,     0},
        { 64,    32,  0,  7,    7,     0},
        { 32,    32,  0,  7,    7,     0}
      };
    // clang-format on
}

template <typename T>
struct GLUFwdTest : public ::testing::TestWithParam<GLUTestCase>
{
protected:
    void SetUp() override
    {

        auto&& handle  = get_handle();
        glu_config     = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        dim = glu_config.dim;

        auto in_dims = glu_config.GetInput();

        input = tensor<T>{in_dims}.generate(gen_value);

        std::vector<size_t> out_dims;

        for(int i = 0; i < in_dims.size(); i++)
        {
            if(i != dim)
            {
                out_dims.push_back(in_dims[i]);
            }
            else
            {
                out_dims.push_back(in_dims[i] / 2);
            }
        }

        output = tensor<T>{out_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_glu_forward<T>(input, ref_output);
        miopenStatus_t status;

        status = miopen::GLUForward(
            handle, input.desc, input_dev.get(), dim, output.desc, output_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    double GetTolerance()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        auto error       = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    GLUTestCase glu_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    int32_t dim;
};

template <typename T>
struct GLUBwdTest : public ::testing::TestWithParam<GLUTestCase>
{
protected:
    void SetUp() override
    {

        auto&& handle  = get_handle();
        glu_config     = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        dim = glu_config.dim;

        auto in_dims = glu_config.GetInput();

        input     = tensor<T>{in_dims}.generate(gen_value);
        inputGrad = tensor<T>{in_dims};
        std::fill(inputGrad.begin(), inputGrad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_inputGrad = tensor<T>{in_dims};
        std::fill(ref_inputGrad.begin(), ref_inputGrad.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> out_dims;

        for(int i = 0; i < in_dims.size(); i++)
        {
            if(i != dim)
            {
                out_dims.push_back(in_dims[i]);
            }
            else
            {
                out_dims.push_back(in_dims[i] / 2);
            }
        }

        outputGrad = tensor<T>{out_dims}.generate(gen_value);

        input_dev      = handle.Write(input.data);
        inputGrad_dev  = handle.Write(inputGrad.data);
        outputGrad_dev = handle.Write(outputGrad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_glu_backward<T>(input, outputGrad, ref_inputGrad);
        miopenStatus_t status;

        status = miopen::GLUBackward(handle,
                                     input.desc,
                                     input_dev.get(),
                                     inputGrad.desc,
                                     inputGrad_dev.get(),
                                     outputGrad.desc,
                                     outputGrad_dev.get(),
                                     dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        inputGrad.data = handle.Read<T>(inputGrad_dev, inputGrad.data.size());
    }

    double GetTolerance()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        auto error       = miopen::rms_range(ref_inputGrad, inputGrad);

        EXPECT_TRUE(miopen::range_distance(ref_inputGrad) == miopen::range_distance(inputGrad));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    GLUTestCase glu_config;

    tensor<T> input;
    tensor<T> inputGrad;
    tensor<T> outputGrad;

    tensor<T> ref_inputGrad;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr inputGrad_dev;
    miopen::Allocator::ManageDataPtr outputGrad_dev;

    int32_t dim;
};
