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

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/generate_random_bit_mask.hpp>

#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <rocrand/rocrand_xorwow.h>

#include <algorithm>

// GenerateBitMaskTestCase
struct GBMTestCase
{
    std::vector<size_t> input_shape;
    std::vector<size_t> mask_shape;
    float p;

    // Add mask size

    friend std::ostream& operator<<(std::ostream& os, const GBMTestCase& tc)
    {
        os << "input_shape: (";
        for(auto i : tc.input_shape)
        {
            os << i << ", ";
        }

        os << ")";
        os << " p: " << tc.p;

        return os;
    }

    std::vector<size_t> GetInputShape() const { return input_shape; }
    std::vector<size_t> GetMaskShape() const { return mask_shape; }
    float GetProb() const { return p; }

    GBMTestCase() {}

    GBMTestCase(std::vector<size_t> input_shape_, std::vector<size_t> mask_shape_, float p_ = 0.5)
        : input_shape(input_shape_), mask_shape(mask_shape_), p(p_)
    {
    }

    GBMTestCase(std::vector<size_t> input_shape_, float p_ = 0.5) : input_shape(input_shape_), p(p_)
    {
        mask_shape        = input_shape;
        mask_shape.back() = (input_shape.back() + 7) / 8;
    }
};

inline std::vector<GBMTestCase> GBMTestConfigs()
{
    return {
        // GBMTestCase({std::pow(2,0)}),
        // GBMTestCase({2, 3, 4}),
        GBMTestCase({800}, {100}, 0), // drop nothing
        GBMTestCase({800}, {100}, 1), // drop all
        // GBMTestCase({800}, {100}, 0.5),
        GBMTestCase({16, 2048, 4096}, {16, 2048, 512}, 0.7),
        // GBMTestCase({16, 32, 2048, 2048}, 0.7),
        // GBMTestCase({50, 700} , 0.7),
        // GBMTestCase({std::pow(2,2)}),
    };
}

struct GBMTest : public ::testing::TestWithParam<GBMTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto input_shape = config.GetInputShape();
        auto mask_shape  = config.GetMaskShape();
        p                = config.GetProb();

        auto in_gen_value = [](auto...) {
            return prng::gen_descreet_uniform_sign<float>(1e-2, 100);
        };

        // NOTE: This function only need inputDescriptor to get input_shape, so other values (e.g.
        // dtype, strides, values are not important, just random choose them)
        input = tensor<float>{input_shape}.generate(in_gen_value);

        mask = tensor<unsigned char>{mask_shape};
        std::fill(mask.begin(), mask.end(), 0);

        ref_mask = tensor<unsigned char>{mask_shape};
        std::fill(ref_mask.begin(), ref_mask.end(), 0);

        mask_dev = handle.Write(mask.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        // Run kernel
        // Step 1: Initialize pstate
        status = miopenGetGenerateRandomBitMaskStatesSize(&handle, &stateSizeInBytes);

        ASSERT_EQ(status, miopenStatusSuccess);

        pstate = tensor<uchar>{stateSizeInBytes / sizeof(rocrand_state_xorwow)};
        std::fill(pstate.begin(), pstate.end(), 0);

        pstate_dev = handle.Write(pstate.data);

        status = miopen::generate_random_bit_mask::InitGenerateRandomBitMaskStates(
            handle, pstate_dev.get(), stateSizeInBytes, 0);

        ASSERT_EQ(status, miopenStatusSuccess);

        // Step 2: Generate random bit mask
        status = miopen::generate_random_bit_mask::GenerateRandomBitMask(
            handle, pstate.desc, pstate_dev.get(), mask.desc, mask_dev.get(), p);

        ASSERT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        mask.data = handle.Read<unsigned char>(mask_dev, mask.data.size());
    }

    void Verify()
    {
        // Print mask.data
        // std::cout << "mask.data: ";
        // for(auto i : mask.data) {
        //     std::cout << static_cast<int>(i) << " ";
        // }
        // std::cout << std::endl;

        // auto output_mask = tensor<unsigned char>{mask.desc.GetLengths()};
        // std::fill(output_mask.begin(), output_mask.end(), 1);

        // counting number 1 in mask
        int64_t count_1 = 0;
        for(size_t i = 0; i < mask.data.size(); i++)
        {
            unsigned char val = mask[i];
            for(int j = 0; j < 8; j++)
            {
                count_1 += val & 1; // Add the least significant bit
                val >>= 1;          // Right shift to process the next bit
            }
        }

        auto input_numel = input.desc.GetElementSize();

        // NOTE: 5% is heuristic number
        // +- 5% allowed
        double min_expected = (1 - p) * 0.95;
        double max_expected = (1 - p) * 1.05;

        double actual = static_cast<double>(count_1) / input_numel;

        EXPECT_TRUE(actual >= min_expected && actual <= max_expected)
            << "Error output beyond tolerance. Actual: " << actual << ". Expected in range: ["
            << min_expected << ", " << max_expected << "]";
    }

    GBMTestCase config;

    tensor<float> input;
    tensor<uchar> pstate;
    tensor<unsigned char> mask;

    tensor<unsigned char> ref_mask;

    miopen::Allocator::ManageDataPtr pstate_dev;
    miopen::Allocator::ManageDataPtr mask_dev;

    float p;
    int rng_mode_cmd = 0;

    size_t stateSizeInBytes;
};
