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

#include "cpu_where.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>
#include <miopen/allocator.hpp>
#include <miopen/miopen.h>
#include <miopen/where.hpp>

struct WhereTestCase
{
    std::vector<size_t> inDims;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const WhereTestCase& tc)
    {
        os << "Input dim: ";
        for(auto inDim : tc.inDims)
        {
            os << inDim << " ";
        }
        os << std::endl;

        return os;
    }

    std::vector<size_t> GetInputDim() const { return inDims; }

    WhereTestCase() {}

    WhereTestCase(std::vector<size_t> input_dim, bool isContiguous_)
        : inDims(input_dim), isContiguous(isContiguous_)
    {
    }
};

inline std::vector<WhereTestCase> GenFullTestCases()
{
    return {WhereTestCase({6, 2, 2, 2}, true),
            WhereTestCase({4, 4, 8, 8, 2}, true),
            WhereTestCase({2, 8, 2}, true),
            WhereTestCase({20, 20, 12, 12}, true)};
}

template <typename T>
struct WhereBwdTest : public ::testing::TestWithParam<WhereTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        where_config   = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_bool  = [](auto...) { return rand() % 2; };

        auto in_dims = where_config.GetInputDim();

        cond       = tensor<uint8_t>{in_dims}.generate(gen_bool);
        outputGrad = tensor<T>{in_dims}.generate(gen_value);

        inputGrad = tensor<T>{in_dims};
        std::fill(inputGrad.begin(), inputGrad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_inputGrad = tensor<T>{in_dims};
        std::fill(ref_inputGrad.begin(), ref_inputGrad.end(), static_cast<T>(0));

        otherGrad = tensor<T>{in_dims};
        std::fill(otherGrad.begin(), otherGrad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_otherGrad = tensor<T>{in_dims};
        std::fill(ref_otherGrad.begin(), ref_otherGrad.end(), static_cast<T>(0));

        inputGrad_dev  = handle.Write(inputGrad.data);
        otherGrad_dev  = handle.Write(otherGrad.data);
        cond_dev       = handle.Write(cond.data);
        outputGrad_dev = handle.Write(outputGrad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        size_t size = outputGrad.GetSize();

        cpu_where_backward<T>(outputGrad, cond, ref_inputGrad, ref_otherGrad, size);
        miopenStatus_t status;

        status = miopen::where::WhereBackward(handle,
                                              outputGrad.desc,
                                              outputGrad_dev.get(),
                                              cond.desc,
                                              cond_dev.get(),
                                              inputGrad.desc,
                                              inputGrad_dev.get(),
                                              otherGrad.desc,
                                              otherGrad_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        inputGrad.data = handle.Read<T>(inputGrad_dev, inputGrad.data.size());
        otherGrad.data = handle.Read<T>(otherGrad_dev, otherGrad.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        auto error1      = miopen::rms_range(ref_inputGrad, inputGrad);
        auto error2      = miopen::rms_range(ref_otherGrad, otherGrad);

        EXPECT_TRUE(miopen::range_distance(ref_inputGrad) == miopen::range_distance(inputGrad));
        EXPECT_TRUE(miopen::range_distance(ref_otherGrad) == miopen::range_distance(otherGrad));

        EXPECT_TRUE(error1 < threshold)
            << "Error output (input grad) beyond tolerance Error:" << error1
            << ",  Threshold: " << threshold << std::endl;

        EXPECT_TRUE(error2 < threshold)
            << "Error output (other grad) beyond tolerance Error:" << error2
            << ",  Threshold: " << threshold << std::endl;
    }

    WhereTestCase where_config;

    tensor<T> inputGrad;
    tensor<T> otherGrad;
    tensor<uint8_t> cond;
    tensor<T> outputGrad;

    tensor<T> ref_inputGrad;
    tensor<T> ref_otherGrad;

    miopen::Allocator::ManageDataPtr inputGrad_dev;
    miopen::Allocator::ManageDataPtr otherGrad_dev;
    miopen::Allocator::ManageDataPtr cond_dev;
    miopen::Allocator::ManageDataPtr outputGrad_dev;
};
