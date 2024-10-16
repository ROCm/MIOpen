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

#pragma once

#include "cpu_glu.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include <miopen/allocator.hpp>
#include <miopen/miopen.h>
#include <miopen/glu.hpp>

struct GLUTestCase
{
    std::vector<size_t> dimsLength;
    uint32_t dim;
    friend std::ostream& operator<<(std::ostream& os, const GLUTestCase& tc)
    {
        os << "Dims Length: ";
        for(auto len : tc.dimsLength)
        {
            os << len << " ";
        }
        return os << " dim:" << tc.dim;
    }

    std::vector<size_t> GetDimsLength() const { return dimsLength; }

    GLUTestCase() {}

    GLUTestCase(std::vector<size_t> dimsLength_, uint32_t dim_) : dimsLength(dimsLength_), dim(dim_)
    {
    }
};

inline std::vector<GLUTestCase> GenFullTestCases()
{ // n c d h w dim
    // clang-format off
    return {
        GLUTestCase({2, 320, 4, 4, 4}, 0),
        GLUTestCase({32, 64, 3, 3, 3}, 0),
        GLUTestCase({64, 3, 11, 11}, 0),
        GLUTestCase({256, 256, 1, 1}, 0),
        GLUTestCase({64, 64, 7, 7}, 0),
        GLUTestCase({64, 32, 7, 7}, 0),
        GLUTestCase({32, 32, 7, 7}, 0)
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

        auto in_dims = glu_config.GetDimsLength();

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

        cpu_glu_contiguous_dim0_forward<T>(input, ref_output);
        miopenStatus_t status;

        status = miopen::glu::GLUForward(
            handle, input.desc, input_dev.get(), output.desc, output_dev.get(), dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
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

    uint32_t dim;
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

        auto in_dims = glu_config.GetDimsLength();

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

        cpu_glu_contiguous_dim0_backward<T>(input, outputGrad, ref_inputGrad);
        miopenStatus_t status;

        status = miopen::glu::GLUBackward(handle,
                                          input.desc,
                                          input_dev.get(),
                                          outputGrad.desc,
                                          outputGrad_dev.get(),
                                          inputGrad.desc,
                                          inputGrad_dev.get(),
                                          dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        inputGrad.data = handle.Read<T>(inputGrad_dev, inputGrad.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
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

    uint32_t dim;
};
