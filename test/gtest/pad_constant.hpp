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

#include <miopen/miopen.h>
#include <miopen/pad_constant.hpp>

#include <gtest/gtest.h>

#include "cpu_pad_constant.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

struct PadConstantTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;

    friend std::ostream& operator<<(std::ostream& os, const PadConstantTestCase& tc)
    {
        return os << "(N: " << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << ")";
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
        else if(N != 0)
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

std::vector<PadConstantTestCase> PadConstantTestConfigs()
{
    return {
        // 2D
        {8, 0, 0, 0, 8},

        // 3D
        {8, 512, 0, 0, 384},
        {8, 511, 0, 0, 1},
        {8, 512, 0, 0, 384},
        {16, 512, 0, 0, 384},
        {16, 512, 0, 0, 8},

        // 4D
        {8, 16, 0, 32, 32},

        // 5D
        {8, 4, 16, 32, 32},
    };
}

inline std::vector<size_t> GetStrides(std::vector<size_t> input, bool contiguous)
{
    if(!contiguous)
        std::swap(input.front(), input.back());
    std::vector<size_t> strides(input.size());
    strides.back() = 1;
    for(int i = input.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * input[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename T>
struct PadConstantTest : public ::testing::TestWithParam<PadConstantTestCase>
{
protected:
    PadConstantTestCase pad_constant_config;
    tensor<T> input;
    tensor<T> output;
    tensor<T> backward_output;

    tensor<T> ref_output;
    tensor<T> ref_backward_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr backward_output_dev;

    std::vector<size_t> padding = std::vector<size_t>(2, 0);

    void SetUp() override
    {
        auto&& handle       = get_handle();
        pad_constant_config = GetParam();
        auto gen_value      = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dims = pad_constant_config.GetInput();

        auto strides = GetStrides(in_dims, false);
        input        = tensor<T>{in_dims, strides}.generate(gen_value);
        input_dev    = handle.Write(input.data);

        // Generate random padding for the first 3 dims
        for(size_t i = 2; i < input.desc.GetLengths().size() * 2; i++)
        {
            padding.push_back(prng::gen_descreet_unsigned<size_t>(1, 5));
        }

        std::vector<size_t> out_dims;
        for(size_t i = 0; i < input.desc.GetLengths().size(); i++)
        {
            out_dims.push_back(in_dims[i] + padding[2 * i] + padding[2 * i + 1]);
        }

        output = tensor<T>{out_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        output_dev = handle.Write(output.data);

        backward_output = tensor<T>{in_dims, strides};
        std::fill(
            backward_output.begin(), backward_output.end(), std::numeric_limits<T>::quiet_NaN());
        backward_output_dev = handle.Write(backward_output.data);

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_backward_output = tensor<T>{in_dims, strides};
        std::fill(ref_backward_output.begin(),
                  ref_backward_output.end(),
                  std::numeric_limits<T>::quiet_NaN());
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        auto out_dims = output.desc.GetLengths();

        T padding_value = static_cast<T>(0);

        cpu_pad_constant_fwd<T>(input.data.data(),
                                ref_output.data.data(),
                                &input.desc,
                                &output.desc,
                                padding,
                                padding_value);
        miopenStatus_t status;

        status = miopen::PadConstantForward(handle,
                                            input.desc,
                                            output.desc,
                                            input_dev.get(),
                                            output_dev.get(),
                                            padding.data(),
                                            padding.size(),
                                            padding_value);
        EXPECT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<T>(output_dev, output.data.size());

        // We're feeding output as backward input gradient
        // On PadConstant this *should* cause backward output to be equal to input
        cpu_pad_constant_bwd<T>(output.data.data(),
                                ref_backward_output.data.data(),
                                &input.desc,
                                &output.desc,
                                padding);

        status = miopen::PadConstantBackward(handle,
                                             backward_output.desc,
                                             output.desc,
                                             backward_output_dev.get(),
                                             output_dev.get(),
                                             padding.data(),
                                             padding.size());

        EXPECT_EQ(status, miopenStatusSuccess);

        backward_output.data = handle.Read<T>(backward_output_dev, backward_output.data.size());

        hipDeviceSynchronize();
    }

    void Verify()
    {
        auto error = miopen::rms_range(ref_output, output);
        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error == 0) << "Outputs do not match each other. Error:" << error;

        error = miopen::rms_range(backward_output, ref_backward_output);
        EXPECT_TRUE(miopen::range_distance(backward_output) ==
                    miopen::range_distance(ref_backward_output));
        EXPECT_TRUE(error == 0) << "Outputs do not match each other. Error:" << error;
    }
};
