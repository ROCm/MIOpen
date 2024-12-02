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

#include <numeric>

#include <miopen/inline_vector.hpp>
#include <miopen/tensor.hpp>

namespace {

TEST(CPU_InlineVectorSizeAndAccumulate_NONE, Test)
{
    miopen::InlineVector<std::size_t, 5> in_v1{4, 2, 1};
    std::vector<size_t> v1{4, 2, 1};

    EXPECT_EQ(in_v1.size(), v1.size());

    for(uint8_t i = 0; i < in_v1.size(); i++)
    {
        EXPECT_EQ(in_v1[i], v1[i]);
    }

    int sum_in_v1 = std::accumulate(in_v1.begin(), in_v1.end(), 0);
    int sum_v1    = std::accumulate(v1.begin(), v1.end(), 0);

    EXPECT_EQ(sum_in_v1, sum_v1);
}

TEST(CPU_InlineVectorFindIfAndDistance_NONE, Test)
{
    std::initializer_list<size_t> init_list_2{4, 1, 2, 2};
    miopen::InlineVector<size_t, 5> in_v2 = init_list_2;
    std::vector<size_t> v2                = init_list_2;

    auto first_not_one_in_v2 =
        std::find_if(in_v2.rbegin(), in_v2.rend(), [](int i) { return i != 1; });
    auto first_note_one_v2 = std::find_if(v2.rbegin(), v2.rend(), [](int i) { return i != 1; });
    EXPECT_EQ(*first_not_one_in_v2, *first_note_one_v2);

    auto d_in_v2 = std::distance(in_v2.begin(), first_not_one_in_v2.base());
    auto d_v2    = std::distance(v2.begin(), first_note_one_v2.base());
    EXPECT_EQ(d_in_v2, d_v2);
}

TEST(CPU_InlineVecotrTie_NONE, Test)
{
    std::initializer_list<size_t> init_list_3{4, 1, 2, 2};
    miopen::InlineVector<size_t, 5> in_v3 = init_list_3;
    std::vector<size_t> v3                = init_list_3;

    std::array<size_t, 4> arr_in_v3;
    std::array<size_t, 4> arr_v3;
    std::tie(arr_in_v3[0], arr_in_v3[1], arr_in_v3[2], arr_in_v3[3]) = miopen::tien<4>(in_v3);
    std::tie(arr_v3[0], arr_v3[1], arr_v3[2], arr_v3[3])             = miopen::tien<4>(v3);

    for(uint8_t i = 0; i < in_v3.size(); i++)
    {
        EXPECT_EQ(arr_in_v3[i], arr_v3[i]);
    }
}

TEST(CPU_InlineVectorCapacityAndEmpty_NONE, Test)
{
    miopen::InlineVector<size_t, 5> in_v4{};
    std::vector<size_t> v4{};

    ASSERT_EQ(in_v4.capacity(), 5);

    EXPECT_EQ(in_v4.empty(), v4.empty());
    EXPECT_EQ(in_v4.begin(), in_v4.end());
}

TEST(CPU_InlineVectorIteratorsConstructor_NONE, Test)
{
    std::vector<size_t> vv = {1, 2, 4, 1};
    miopen::InlineVector<size_t, 5> in_v5(vv.begin(), vv.end());
    std::vector<size_t> v5(vv.begin(), vv.end());

    for(uint8_t i = 0; i < in_v5.size(); i++)
    {
        EXPECT_EQ(in_v5[i], v5[i]);
    }
}

TEST(CPU_InlineVectorConstructorException_NONE, Test)
{
    std::initializer_list<size_t> init_list_v6{1, 2, 3, 4, 5, 6};
    auto constructor_1 = [init_list_v6]() {
        miopen::InlineVector<size_t, 5> v6(init_list_v6.begin(), init_list_v6.end());
    };
    auto constructor_2 = [init_list_v6]() { miopen::InlineVector<size_t, 5> v6(init_list_v6); };
    ASSERT_ANY_THROW(constructor_1());
    ASSERT_ANY_THROW(constructor_2());
}

} // namespace
