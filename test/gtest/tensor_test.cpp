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
#include <miopen/tensor.hpp>
#include <miopen/datatype.hpp>

#include <gtest/gtest.h>

namespace {
struct TestCase
{
    std::vector<int> lens;
    std::vector<int> strides;
    miopenDataType_t datatype;
    bool packed;
    std::string name;

    std::function<miopenStatus_t(miopenTensorDescriptor_t,
                                 miopenDataType_t,
                                 const std::vector<int>&,
                                 const std::vector<int>&)>
        setDecriptor;

    miopenStatus_t SetDecriptor(miopenTensorDescriptor_t tensor) const
    {
        return setDecriptor(tensor, datatype, lens, strides);
    }

    int Size() const noexcept { return static_cast<int>(lens.size()); }

    friend std::ostream& operator<<(std::ostream& os, const TestCase& tc)
    {
        os << tc.name << (tc.packed ? "_packed" : "_non-packed") << " lens: ";
        std::copy(tc.lens.begin(), tc.lens.end(), std::ostream_iterator<int>(os, "x"));
        os << " strides: ";
        std::copy(tc.strides.begin(), tc.strides.end(), std::ostream_iterator<int>(os, "x"));
        return os;
    }
};

miopenStatus_t fixture_nxd(miopenTensorDescriptor_t tensor,
                           miopenDataType_t datatype,
                           const std::vector<int>& lens,
                           const std::vector<int>&)
{
    return miopenSetTensorDescriptor(tensor, datatype, lens.size(), lens.data(), nullptr);
};

miopenStatus_t fixture_nxd_strides(miopenTensorDescriptor_t tensor,
                                   miopenDataType_t datatype,
                                   const std::vector<int>& lens,
                                   const std::vector<int>& strides)
{
    return miopenSetTensorDescriptor(tensor, datatype, lens.size(), lens.data(), strides.data());
};

miopenStatus_t fixture_n4d_vector(miopenTensorDescriptor_t tensor,
                                  miopenDataType_t datatype,
                                  const std::vector<int>& lens,
                                  const std::vector<int>&)
{
    auto vec_lens = lens;
    vec_lens[1] *= 4;
    return miopenSetNdTensorDescriptorWithLayout(
        tensor, datatype, miopenTensorNCHWc4, vec_lens.data(), vec_lens.size());
};

miopenStatus_t fixture_n4d(miopenTensorDescriptor_t tensor,
                           miopenDataType_t datatype,
                           const std::vector<int>& lens,
                           const std::vector<int>&)
{
    return lens.size() == 4
               ? miopenSet4dTensorDescriptor(tensor, datatype, lens[0], lens[1], lens[2], lens[3])
               : miopenStatusBadParm;
};

class CPU_tensor_nxd_NONE : public ::testing::TestWithParam<TestCase>
{
public:
    static void SetUpTestSuite()
    {
        miopenTensorDescriptor_t tensor;
        ASSERT_EQ(miopenCreateTensorDescriptor(&tensor), miopenStatusSuccess);
        ASSERT_EQ(miopenSet4dTensorDescriptor(tensor, miopenHalf, 100, 32, 8, 8),
                  miopenStatusSuccess);
        ASSERT_EQ(miopenDestroyTensorDescriptor(tensor), miopenStatusSuccess);

        ASSERT_NE(miopenSet4dTensorDescriptor(nullptr, miopenHalf, 100, 32, 8, 8),
                  miopenStatusSuccess);
    }

protected:
    miopenTensorDescriptor_t tensor;
    void SetUp() override
    {
        ASSERT_EQ(miopenCreateTensorDescriptor(&tensor), miopenStatusSuccess);
        ASSERT_NE(tensor, nullptr);
        ASSERT_EQ(GetParam().SetDecriptor(tensor), miopenStatusSuccess);

        int size;
        ASSERT_EQ(miopenGetTensorDescriptorSize(tensor, &size), miopenStatusSuccess);
        ASSERT_EQ(size, GetParam().Size());

        EXPECT_EQ(miopen::get_object(*tensor).IsPacked(), GetParam().packed);
    }
    void TearDown() override
    {
        ASSERT_EQ(miopenDestroyTensorDescriptor(tensor), miopenStatusSuccess);
    }
};

} // namespace
TEST_P(CPU_tensor_nxd_NONE, TestGetTensor)
{
    std::vector<int> lens(GetParam().Size(), 0);
    std::vector<int> strides(GetParam().Size(), 0);
    miopenDataType_t dt;
    ASSERT_EQ(miopenGetTensorDescriptor(tensor, &dt, lens.data(), strides.data()),
              miopenStatusSuccess);
    EXPECT_EQ(dt, GetParam().datatype);
    EXPECT_EQ(lens, GetParam().lens);
    EXPECT_EQ(strides, GetParam().strides);
};

TEST_P(CPU_tensor_nxd_NONE, TestGetTensorLengths)
{
    std::vector<int> lens(GetParam().Size(), 0);
    miopenDataType_t dt;
    ASSERT_EQ(miopenGetTensorDescriptor(tensor, &dt, lens.data(), nullptr), miopenStatusSuccess);
    EXPECT_EQ(dt, GetParam().datatype);
    EXPECT_EQ(lens, GetParam().lens);

    const auto& l = miopen::get_object(*tensor).GetLengths();
    std::copy(l.begin(), l.end(), lens.begin());
    EXPECT_EQ(lens, GetParam().lens);
};

TEST_P(CPU_tensor_nxd_NONE, TestGetTensorStrides)
{
    std::vector<int> strides(GetParam().Size(), 0);
    miopenDataType_t dt;
    ASSERT_EQ(miopenGetTensorDescriptor(tensor, &dt, nullptr, strides.data()), miopenStatusSuccess);
    EXPECT_EQ(dt, GetParam().datatype);
    EXPECT_EQ(strides, GetParam().strides);

    const auto& s = miopen::get_object(*tensor).GetStrides();
    std::copy(s.begin(), s.end(), strides.begin());
    EXPECT_EQ(strides, GetParam().strides);
};

TEST_P(CPU_tensor_nxd_NONE, TestGetTensorBytes)
{
    auto vector_len = miopen::get_object(*tensor).GetVectorLength();
    auto byte_size  = miopen::get_data_size(GetParam().datatype) *
                     std::inner_product(GetParam().lens.begin(),
                                        GetParam().lens.end(),
                                        GetParam().strides.begin(),
                                        vector_len,
                                        std::plus<std::size_t>(),
                                        [](int v1, int v2) { return (v1 - 1) * v2; });

    size_t numBytes;
    ASSERT_EQ(miopenGetTensorNumBytes(tensor, &numBytes), miopenStatusSuccess);
    EXPECT_EQ(numBytes, byte_size);
};

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_tensor_nxd_NONE,
    ::testing::Values(
        // clang-format off
        // 1-DIMENSIONAL -------------------//
        TestCase{{10}, {1}, miopenBFloat16, true, "n1d", fixture_nxd},
        TestCase{{10}, {2}, miopenFloat, false, "n1d_strides", fixture_nxd_strides},
        TestCase{{10}, {1}, miopenHalf, true, "n1d_strides", fixture_nxd_strides},

        // 2-DIMENSIONAL -------------------//
        TestCase{{10, 32}, {32, 1}, miopenBFloat16, true, "n2d", fixture_nxd},
        TestCase{{10, 32}, {64, 2}, miopenFloat, false, "n2d_strides", fixture_nxd_strides},
        TestCase{{10, 32}, {32, 1}, miopenHalf, true, "n2d_strides", fixture_nxd_strides},
        TestCase{{8, 8}, {14, 1}, miopenHalf, false, "n2d_strides", fixture_nxd_strides},

        // 3-DIMENSIONAL -------------------//
        TestCase{{10, 32, 8}, {256, 8, 1}, miopenBFloat16, true, "n3d", fixture_nxd},
        TestCase{{10, 32, 8}, {512, 16, 2}, miopenFloat, false, "n3d_strides", fixture_nxd_strides},
        TestCase{{10, 32, 8}, {256, 8, 1}, miopenHalf, true, "n3d_strides", fixture_nxd_strides},
        TestCase{{32, 8, 8}, {112, 14, 1}, miopenHalf, false, "n3d_strides", fixture_nxd_strides},

        // 4-DIMENSIONAL -------------------//
        TestCase{{10, 32, 8, 4}, {1024, 32, 4, 1}, miopenBFloat16, true, "n4d", fixture_nxd},
        TestCase{{10, 32, 8, 4}, {1024, 32, 4, 1}, miopenFloat, true, "n4d_direct", fixture_n4d},
        TestCase{{10, 32, 8, 4}, {2048, 64, 8, 2}, miopenFloat, false, "n4d_strides", fixture_nxd_strides},
        TestCase{{10, 32, 8, 4}, {1024, 32, 4, 1}, miopenHalf, true, "n4d_strides", fixture_nxd_strides},
        TestCase{{100, 32, 8, 8}, {4704, 112, 14, 1}, miopenHalf, false, "n4d_strides", fixture_nxd_strides},

        // 4-DIMENSIONAL - vector ----------//
        TestCase{{32, 16, 8, 4}, {2048, 128, 16, 4}, miopenHalf, true, "n4d_vector", fixture_n4d_vector},

        // 5-DIMENSIONAL -------------------//
        TestCase{{10, 32, 5, 4, 2}, {1280, 40, 8, 2, 1}, miopenBFloat16, true, "n5d", fixture_nxd},
        TestCase{{10, 32, 5, 4, 2}, {2580, 80, 16, 4, 2}, miopenFloat, false, "n5d_strides", fixture_nxd_strides},
        TestCase{{10, 32, 5, 4, 2}, {1280, 40, 8, 2, 1}, miopenHalf, true, "n5d_strides", fixture_nxd_strides},
        TestCase{{128, 100, 32, 8, 8}, {493920, 4704, 112, 14, 1}, miopenHalf, false, "n5d_strides", fixture_nxd_strides} // clang-format on
        ));

using CPU_tensor_n4d_NONE = CPU_tensor_nxd_NONE;

TEST_P(CPU_tensor_n4d_NONE, TestGet4DTensor)
{
    std::vector<int> lens(4, -1);
    std::vector<int> strides(4, -1);
    miopenDataType_t dt;
    ASSERT_EQ(miopenGet4dTensorDescriptor(tensor,
                                          &dt,
                                          &lens[0],
                                          &lens[1],
                                          &lens[2],
                                          &lens[3],
                                          &strides[0],
                                          &strides[1],
                                          &strides[2],
                                          &strides[3]),
              miopenStatusSuccess);
    EXPECT_EQ(dt, GetParam().datatype);
    EXPECT_EQ(lens, GetParam().lens);
    EXPECT_EQ(strides, GetParam().strides);
};

TEST_P(CPU_tensor_n4d_NONE, TestGetTensorIndex)
{
    auto [nStride, cStride, hStride, wStride] = miopen::tien<4>(GetParam().strides);
    auto vector_len                           = miopen::get_object(*tensor).GetVectorLength();

    if(vector_len == 1)
    {
        EXPECT_EQ(miopen::get_object(*tensor).GetIndex({0, 0, 0, 0}), 0);
        EXPECT_EQ(miopen::get_object(*tensor).GetIndex({0, 0, 0, 1}), wStride);
        EXPECT_EQ(miopen::get_object(*tensor).GetIndex({0, 0, 0, 2}), 2 * wStride);
        EXPECT_EQ(miopen::get_object(*tensor).GetIndex({0, 0, 1, 0}), hStride);
        EXPECT_EQ(miopen::get_object(*tensor).GetIndex({0, 0, 1, 1}), hStride + wStride);
    }
    else if(vector_len == 4)
    {
        EXPECT_EQ(miopen::get_object(*tensor).GetIndex({0, 0, 0, 0, 0}), 0);
        EXPECT_EQ(miopen::get_object(*tensor).GetIndex({1, 0, 0, 0, 0}), 1);
        EXPECT_EQ(miopen::get_object(*tensor).GetIndex({2, 0, 0, 0, 0}), 2);
        EXPECT_EQ(miopen::get_object(*tensor).GetIndex({3, 0, 0, 0, 0}), 3);
        EXPECT_EQ(miopen::get_object(*tensor).GetIndex({0, 0, 0, 0, 1}), wStride);
    }
    else
    {
        FAIL() << "Test expects not vectorized or c4 layouts";
    }
};

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_tensor_n4d_NONE,
    ::testing::Values(
        // clang-format off
        // 4-DIMENSIONAL -------------------//
        TestCase{{10, 32, 8, 4}, {1024, 32, 4, 1}, miopenBFloat16, true, "n4d", fixture_nxd},
        TestCase{{10, 32, 8, 4}, {1024, 32, 4, 1}, miopenFloat, true, "n4d_direct", fixture_n4d},
        TestCase{{10, 32, 8, 4}, {2048, 64, 8, 2}, miopenFloat, false, "n4d_strides", fixture_nxd_strides},
        TestCase{{10, 32, 8, 4}, {1024, 32, 4, 1}, miopenHalf, true, "n4d_strides", fixture_nxd_strides},
        TestCase{{100, 32, 8, 8}, {4704, 112, 14, 1}, miopenHalf, false, "n4d_strides", fixture_nxd_strides},

        // 4-DIMENSIONAL - vector ----------//
        TestCase{{32, 16, 8, 4}, {2048, 128, 16, 4}, miopenHalf, true, "n4d_vector", fixture_n4d_vector}
        // clang-format on
        ));
