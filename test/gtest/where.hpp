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
#include <vector>

#include <gtest/gtest.h>
#include <miopen/allocator.hpp>
#include <miopen/miopen.h>
#include <miopen/where.hpp>

struct WhereTestCase
{
    std::vector<size_t> inDims;
    std::vector<size_t> otherDims;
    std::vector<size_t> condDims;
    friend std::ostream& operator<<(std::ostream& os, const WhereTestCase& tc)
    {
        os << "Input dim: ";
        for(auto inDim : tc.inDims)
        {
            os << inDim << " ";
        }
        os << std::endl;
        os << "Other dim: ";
        for(auto otherDim : tc.otherDims)
        {
            os << otherDim << " ";
        }
        os << std::endl;
        os << "Cond dim: ";
        for(auto condDim : tc.condDims)
        {
            os << condDim << " ";
        }
        os << std::endl;
        return os;
    }

    std::vector<size_t> GetInputDim() const
    {
        if(inDims.empty())
        {
            return std::vector<size_t>{0};
        }
        return inDims;
    }

    std::vector<size_t> GetOtherDim() const
    {
        if(otherDims.empty())
        {
            return std::vector<size_t>{0};
        }
        return otherDims;
    }

    std::vector<size_t> GetCondDim() const
    {
        if(condDims.empty())
        {
            return std::vector<size_t>{0};
        }
        return condDims;
    }

    WhereTestCase() {}

    WhereTestCase(std::vector<size_t> input_dim)
        : inDims(input_dim), otherDims(input_dim), condDims(input_dim)
    {
    }

    WhereTestCase(std::vector<size_t> input_dim,
                  std::vector<size_t> other_dim,
                  std::vector<size_t> cond_dim)
        : inDims(input_dim), otherDims(other_dim), condDims(cond_dim)
    {
    }
};

inline std::vector<WhereTestCase> GenFullTestCases()
{ // n c d h w dim
    // clang-format off
    return {
        WhereTestCase({6, 2, 2, 2, 2}),
        WhereTestCase({4, 4, 8, 8, 2}),
        WhereTestCase({1, 2, 8, 2, 2}),
        WhereTestCase({16, 20, 20, 12, 12}),
    };
    // clang-format on
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

        auto in_dims    = where_config.GetInputDim();
        auto other_dims = where_config.GetOtherDim();
        auto cond_dims  = where_config.GetCondDim();

        isInputGradRequired = !in_dims.empty();
        isOtherGradRequired = !other_dims.empty();

        cond = tensor<uint8_t>{cond_dims};
        for(auto i = 0; i < cond.GetSize(); i++)
        {
            cond[i] = rand() % 2;
        }

        outputGrad = tensor<T>{cond_dims}.generate(gen_value);

        if(isInputGradRequired)
        {
            inputGrad = tensor<T>{in_dims};
            std::fill(inputGrad.begin(), inputGrad.end(), std::numeric_limits<T>::quiet_NaN());

            ref_inputGrad = tensor<T>{in_dims};
            std::fill(ref_inputGrad.begin(), ref_inputGrad.end(), static_cast<T>(0));

            inputGrad_dev = handle.Write(inputGrad.data);
        }

        if(isOtherGradRequired)
        {
            otherGrad = tensor<T>{other_dims};
            std::fill(otherGrad.begin(), otherGrad.end(), std::numeric_limits<T>::quiet_NaN());

            ref_otherGrad = tensor<T>{other_dims};
            std::fill(ref_otherGrad.begin(), ref_otherGrad.end(), static_cast<T>(0));

            otherGrad_dev = handle.Write(otherGrad.data);
        }

        cond_dev       = handle.Write(cond.data);
        outputGrad_dev = handle.Write(outputGrad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        size_t size = outputGrad.GetSize();

        tensor<T> dummy;
        tensor<T>& refInput = isInputGradRequired ? ref_inputGrad : dummy;
        tensor<T>& refOther = isOtherGradRequired ? ref_otherGrad : dummy;
        cpu_where_backward<T>(outputGrad, cond, refInput, refOther, size);
        miopenStatus_t status;

        auto inputGradMem = isInputGradRequired ? inputGrad_dev.get() : nullptr;
        auto otherGradMem = isOtherGradRequired ? otherGrad_dev.get() : nullptr;
        status            = miopen::where::WhereBackward(handle,
                                              outputGrad.desc,
                                              outputGrad_dev.get(),
                                              cond.desc,
                                              cond_dev.get(),
                                              inputGrad.desc,
                                              inputGradMem,
                                              otherGrad.desc,
                                              otherGradMem);

        EXPECT_EQ(status, miopenStatusSuccess);

        if(isInputGradRequired)
        {
            inputGrad.data = handle.Read<T>(inputGrad_dev, inputGrad.data.size());
        }
        if(isOtherGradRequired)
        {
            otherGrad.data = handle.Read<T>(otherGrad_dev, otherGrad.data.size());
        }
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        auto error1      = isInputGradRequired ? miopen::rms_range(ref_inputGrad, inputGrad) : 0;
        auto error2      = isOtherGradRequired ? miopen::rms_range(ref_otherGrad, otherGrad) : 0;

        if(isInputGradRequired)
        {
            EXPECT_TRUE(miopen::range_distance(ref_inputGrad) == miopen::range_distance(inputGrad));
        }
        if(isOtherGradRequired)
        {
            EXPECT_TRUE(miopen::range_distance(ref_otherGrad) == miopen::range_distance(otherGrad));
        }
        EXPECT_TRUE(error1 < threshold)
            << "Error output (input grad) beyond tolerance Error:" << error1
            << ",  Threshold: " << threshold << std::endl;

        EXPECT_TRUE(error2 < threshold)
            << "Error output (other grad) beyond tolerance Error:" << error2
            << ",  Threshold: " << threshold << std::endl;
    }

    WhereTestCase where_config;
    bool isInputGradRequired;
    bool isOtherGradRequired;

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
