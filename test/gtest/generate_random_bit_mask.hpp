/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

struct GenerateRandomBitMaskTestCase
{
    std::vector<size_t> mask_shape;
    float p;

    friend std::ostream& operator<<(std::ostream& os, const GenerateRandomBitMaskTestCase& tc)
    {
        os << "mask_shape: (";
        for(auto i : tc.mask_shape)
        {
            os << i << ", ";
        }

        os << ")";
        os << " p: " << tc.p;

        return os;
    }

    std::vector<size_t> GetMaskShape() const { return mask_shape; }
    float GetProb() const { return p; }

    GenerateRandomBitMaskTestCase() {}

    GenerateRandomBitMaskTestCase(std::vector<size_t> mask_shape_, float p_ = 0.5)
        : mask_shape(mask_shape_), p(p_)
    {
    }
};

inline std::vector<GenerateRandomBitMaskTestCase> GBMTestConfigs()
{
    return {
        GenerateRandomBitMaskTestCase({400}, 0), // drop nothing
        GenerateRandomBitMaskTestCase({400}, 0.7),
        GenerateRandomBitMaskTestCase({400}, 1), // drop all
        GenerateRandomBitMaskTestCase({4, 400}, 0.7),
        GenerateRandomBitMaskTestCase({2, 4, 400}, 0.7),
        GenerateRandomBitMaskTestCase({1, 2, 4, 400}, 0.7),
        GenerateRandomBitMaskTestCase({700, 700}, 0.7), // not div 8 size
    };
}

struct GBMTest : public ::testing::TestWithParam<GenerateRandomBitMaskTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto mask_shape = config.GetMaskShape();
        p               = config.GetProb();

        // Initialize pstate
        auto status = miopenGetGenerateRandomBitMaskStatesSize(&handle, &stateSizeInBytes);

        ASSERT_EQ(status, miopenStatusSuccess);

        pstate_dev = handle.Create<rocrand_state_xorwow>(stateSizeInBytes);

        status = miopen::generate_random_bit_mask::InitPRNGState(
            handle, pstate_dev.get(), stateSizeInBytes, 0);

        ASSERT_EQ(status, miopenStatusSuccess);

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
        status = miopen::generate_random_bit_mask::GenerateRandomBitMask(
            handle, pstate_dev.get(), stateSizeInBytes, mask.desc, mask_dev.get(), p);

        ASSERT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        mask.data = handle.Read<unsigned char>(mask_dev, mask.data.size());
    }

    void Verify()
    {

        // counting number bit 1 in mask
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

        auto input_numel = mask.desc.GetElementSize() * 8;

        // NOTE: 5% is heuristic number
        // +- 5% allowed
        double min_expected = (1 - p) * 0.95;
        double max_expected = (1 - p) * 1.05;

        double actual = static_cast<double>(count_1) / input_numel;

        EXPECT_TRUE(actual >= min_expected && actual <= max_expected)
            << "Error output beyond tolerance. Actual: " << actual << ". Expected in range: ["
            << min_expected << ", " << max_expected << "]";
    }

    GenerateRandomBitMaskTestCase config;

    // pstate.type is `rocrand_state_xorwow` but use `uchar` instead, to be able to create a tensor
    // type
    tensor<unsigned char> pstate;
    tensor<unsigned char> mask;

    tensor<unsigned char> ref_mask;

    miopen::Allocator::ManageDataPtr pstate_dev;
    miopen::Allocator::ManageDataPtr mask_dev;

    float p;
    int rng_mode_cmd = 0;

    size_t stateSizeInBytes;
};
