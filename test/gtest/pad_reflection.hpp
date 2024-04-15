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
#include "cpu_pad_reflection.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/pad_reflection.hpp>

struct PadReflectionCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int padding[4];
    friend std::ostream& operator<<(std::ostream& os, const PadReflectionCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W;
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

    std::vector<size_t> GetPadding()
    {
        std::vector<size_t> paddingVector;
        for(int i = 0; i < 4; ++i)
        {
            paddingVector.push_back(static_cast<size_t>(padding[i]));
        }
        return paddingVector;
    }
};

std::vector<PadReflectionCase> PadReflectionTestFloatConfigs()
{ // n c d h w padding
    // clang-format off
    return {
        { 1,   1,    0,  3, 3, {2, 2, 2, 2}},
        { 48,   8,    0,  512, 512, {1, 1, 1, 1}},
        { 48,   8,    0,  512, 512, {1, 1, 3, 3}},
        { 48,   8,    0,  512, 512, {0, 0, 2, 2}},
        { 16, 311,    0,  98,  512, {1, 1, 1, 1}},
        { 16, 311,    0,  98,  512, {1, 1, 3, 3}},
        { 16, 311,    0,  98,  512, {0, 0, 2, 2}},
      };
    // clang-format on
}

template <typename T = float>
struct PadReflectionTest : public ::testing::TestWithParam<PadReflectionCase>
{
protected:
    void SetUp() override
    {
        auto&& handle         = get_handle();
        pad_reflection_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dims = pad_reflection_config.GetInput();
        auto padding = pad_reflection_config.GetPadding();
        input        = tensor<T>{in_dims}.generate(gen_value);
        std::vector<size_t> out_dims;

        for(int i = 0; i < in_dims.size(); i++)
        {
            if(i == 2)
            {
                out_dims.push_back(in_dims[i] + 2 * padding[2]);
            }
            else if(i == 3)
            {
                out_dims.push_back(in_dims[i] + 2 * padding[0]);
            }
            else
            {
                out_dims.push_back(in_dims[i]);
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
        auto padding  = pad_reflection_config.GetPadding();

        cpu_pad_reflection<T>(input, ref_output, padding);
        miopenStatus_t status;

        status = miopen::PadReflection(
            handle, input.desc, input_dev.get(), output.desc, output_dev.get(), padding);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        for(int i = 0; i < output.data.size() - 1; ++i)
        {
            EXPECT_EQ(output.data[i], ref_output.data[i]);
        }
    }
    PadReflectionCase pad_reflection_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};
