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
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/glu.hpp>

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
        { 8,    120,  0,  0,   1,     0},  //bart
        { 8,    120,  0,  0,   1,     0},
        { 8,    1023, 0,  0,   1,     0},  //gpt_neo
        { 8,    1024, 0,  0,   768,   0},
        { 8,    1023, 0,  0,   1,     0},
        { 8,    1024, 0,  0,   768,   0},
        { 16,   1024, 0,  0,   768,   0},  //gpt2
        { 16,   1024, 0,  0,   768,   0},
        { 48,   8,    0,  512, 512,   0},  //t5
        { 48,   8,    0,  512, 512,   0},
        { 16, 311,    0,  98,  512,   2},  //rnnt
        { 16, 311,    0,  98,  512,   2}
      };
    // clang-format on
}

template <typename T = float>
struct GLUTest : public ::testing::TestWithParam<GLUTestCase>
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

        inputFirstHalf = tensor<T>{out_dims};
        std::fill(
            inputFirstHalf.begin(), inputFirstHalf.end(), std::numeric_limits<T>::quiet_NaN());

        inputSecondHalf = tensor<T>{out_dims};
        std::fill(
            inputSecondHalf.begin(), inputSecondHalf.end(), std::numeric_limits<T>::quiet_NaN());

        output = tensor<T>{out_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        splitInput();

        inputFirstHalf_dev  = handle.Write(inputFirstHalf.data);
        inputSecondHalf_dev = handle.Write(inputSecondHalf.data);
        output_dev          = handle.Write(output.data);
    }

    void splitInput()
    {
        auto input_dims  = input.desc.GetLengths();
        auto output_dims = output.desc.GetLengths();

        auto splitDim_size   = input_dims[dim];
        auto splitedDim_size = output_dims[dim];
        auto output_numel =
            std::accumulate(output_dims.begin(), output_dims.end(), 1L, std::multiplies<int64_t>());

        auto inner_size = 1ULL;
        for(int32_t i = dim + 1; i < input_dims.size(); i++)
        {
            inner_size *= input_dims[i];
        }

        for(size_t o = 0; o < output_numel; o++)
        {
            size_t innerIdx       = o % inner_size;
            size_t splittedDimIdx = ((o - innerIdx) / inner_size) % splitedDim_size;
            size_t outerIdx =
                (o - innerIdx - splittedDimIdx * inner_size) / (inner_size * splitedDim_size);
            size_t inputIdx1 =
                outerIdx * splitDim_size * inner_size + splittedDimIdx * inner_size + innerIdx;
            size_t inputIdx2 = outerIdx * splitDim_size * inner_size +
                               (splittedDimIdx + splitedDim_size) * inner_size + innerIdx;
            inputFirstHalf[o]  = input[inputIdx1];
            inputSecondHalf[o] = input[inputIdx2];
        }
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_glu_forward<T>(inputFirstHalf, inputSecondHalf, ref_output);
        miopenStatus_t status;

        status = miopen::GLUForward(handle,
                                    input.desc,
                                    inputFirstHalf.desc,
                                    inputFirstHalf_dev.get(),
                                    inputSecondHalf_dev.get(),
                                    dim,
                                    output.desc,
                                    output_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    GLUTestCase glu_config;

    tensor<T> input;
    tensor<T> inputFirstHalf;
    tensor<T> inputSecondHalf;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr inputFirstHalf_dev;
    miopen::Allocator::ManageDataPtr inputSecondHalf_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    int32_t dim;
};
