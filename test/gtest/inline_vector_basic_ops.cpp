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

#include <miopen/tensor_layout.hpp>
#include <miopen/inline_vector.hpp>
#include <miopen/tensor.hpp>

TEST(CPU_InlineVectorSizeAndAccumulate_NONE, Test)
{
    miopen::InlineVector<int, 5> in_v1{4, 2, 1};
    std::vector<int> v1{4, 2, 1};

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

    auto d_in_v2 = std::distance(in_v2.begin(), first_not_one_in_v2.base());
    auto d_v2    = std::distance(v2.begin(), first_note_one_v2.base());

    ASSERT_EQ(d_in_v2, d_v2);
    EXPECT_EQ(*first_not_one_in_v2, *first_note_one_v2);
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

TEST(CPU_InlineVectorAllOf_NONE, Test)
{
    miopen::InlineVector<size_t, 5> in_v7({3, 1, 1});
    std::vector<size_t> v7{3, 2, 1};

    bool all_of_in_v7 = std::all_of(in_v7.cbegin(), in_v7.cend(), [](size_t x) { return x > 0; });
    bool all_of_v7    = std::all_of(v7.cbegin(), v7.cend(), [](size_t x) { return x > 0; });

    EXPECT_EQ(all_of_in_v7, all_of_v7);
}

TEST(CPU_InlineVectorResize_NONE, Test)
{
    miopen::InlineVector<size_t, 5> in_v8({2, 2, 2, 2, 2});
    in_v8.resize(2);

    EXPECT_EQ(in_v8.size(), 2);

    in_v8.resize(4, 1);

    std::vector<size_t> v8{2, 2, 1, 1};

    EXPECT_EQ(in_v8.size(), v8.size());

    for(uint8_t i = 0; i < in_v8.size(); i++)
    {
        EXPECT_EQ(in_v8[i], v8[i]);
    }
}

TEST(CPU_InlineVectorPushBackPopBack_NONE, Test)
{
    miopen::InlineVector<size_t, 5> in_v9 = {8, 7, 6};
    std::vector<size_t> v9{8, 7, 6, 5};

    in_v9.push_back(5);

    EXPECT_EQ(in_v9.size(), v9.size());
    for(uint8_t i = 0; i < in_v9.size(); i++)
    {
        EXPECT_EQ(in_v9[i], v9[i]);
    }

    v9.pop_back();
    in_v9.pop_back();

    EXPECT_EQ(in_v9.size(), v9.size());
    for(uint8_t i = 0; i < in_v9.size(); i++)
    {
        EXPECT_EQ(in_v9[i], v9[i]);
    }

    in_v9.push_back(5);
    in_v9.push_back(4);
    EXPECT_ANY_THROW({ in_v9.push_back(3); });
}

TEST(CPU_InlineVectorAt_NONE, Test)
{
    miopen::InlineVector<size_t, 5> in_v10{2, 4, 6};
    std::vector<size_t> v10{2, 4, 6};

    EXPECT_ANY_THROW(in_v10.at(3));
    EXPECT_ANY_THROW(in_v10.at(5));
    EXPECT_EQ(in_v10.at(1), v10.at(1));
}

TEST(CPU_InlineVectorFrontBack_NONE, Test)
{
    miopen::InlineVector<size_t, 5> in_v11{};

    EXPECT_ANY_THROW(in_v11.front());
    EXPECT_ANY_THROW(in_v11.back());

    in_v11.push_back(10);
    EXPECT_EQ(in_v11.front(), in_v11.back());
}

TEST(CPU_InlineVectorClear_NONE, Test)
{
    miopen::InlineVector<size_t, 5> in_v12{1, 2, 3, 4, 5};
    in_v12.clear();
    EXPECT_EQ(in_v12.size(), 0);
}

TEST(CPU_InlineVectorInsert_NONE, Test)
{
    miopen::InlineVector<size_t, 5> iv13_1{1, 2, 3};
    std::vector<size_t> v13_1{1, 2, 3};
    iv13_1.insert(iv13_1.begin(), 0);
    v13_1.insert(v13_1.begin(), 0);
    for(int i = 0; i < iv13_1.size(); i++)
    {
        EXPECT_EQ(iv13_1[i], v13_1[i]);
    }

    miopen::InlineVector<size_t, 5> iv13_2{1, 2, 3};
    std::vector<size_t> v13_2{1, 2, 3};
    iv13_2.insert(iv13_2.end(), 4);
    v13_2.insert(v13_2.end(), 4);
    for(int i = 0; i < iv13_2.size(); i++)
    {
        EXPECT_EQ(iv13_2[i], v13_2[i]);
    }

    miopen::InlineVector<size_t, 5> iv13_3{1, 2, 3, 4};
    std::vector<size_t> v13_3{1, 2, 3, 4};
    iv13_3.insert(iv13_3.begin() + 2, 0);
    v13_3.insert(v13_3.begin() + 2, 0);
    for(int i = 0; i < iv13_3.size(); i++)
    {
        EXPECT_EQ(iv13_3[i], v13_3[i]);
    }

    miopen::InlineVector<size_t, 5> iv13_4{1, 2, 3};
    std::vector<size_t> v13_4{1, 2, 3};
    iv13_4.insert(iv13_4.begin() + iv13_4.size(), 4);
    v13_4.insert(v13_4.begin() + v13_4.size(), 4);
    for(int i = 0; i < iv13_4.size(); i++)
    {
        EXPECT_EQ(iv13_4[i], v13_4[i]);
    }
}

#include <chrono>

TEST(CPU_InlineVectorPerf1_NONE, Test)
{
    std::vector<float> iv_times;
    std::vector<float> v_times;

    for(int i = 0; i < 1000; i++)
    {
        auto start = std::chrono::steady_clock::now();
        miopen::InlineVector<size_t, 5> iv{1, 2, 3, 4, 5};
        auto end = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<float, std::nano>>(end - start)
                .count();
        iv_times.push_back(elapsed);

        auto start1 = std::chrono::steady_clock::now();
        std::vector<size_t> v{1, 2, 3, 4, 5};
        auto end1 = std::chrono::steady_clock::now();
        auto elapsed1 =
            std::chrono::duration_cast<std::chrono::duration<float, std::nano>>(end1 - start1)
                .count();
        v_times.push_back(elapsed1);
    }

    std::cout << "IV min: " << *(std::min_element(iv_times.begin(), iv_times.end()))
              << " avg: " << std::reduce(iv_times.begin(), iv_times.end()) / 1000.0 << std::endl;
    std::cout << "VE min: " << *(std::min_element(v_times.begin(), v_times.end()))
              << " avg: " << std::reduce(v_times.begin(), v_times.end()) / 1000.0 << std::endl;
}

TEST(CPU_InlineVectorPerf2_NONE, Test)
{
    std::vector<float> iv_times;
    std::vector<float> v_times;

    std::initializer_list<size_t> il{1, 2, 3, 4, 5};

    for(int i = 0; i < 1000; i++)
    {
        auto start = std::chrono::steady_clock::now();
        miopen::InlineVector<size_t, 5> iv(il);
        auto end = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<float, std::nano>>(end - start)
                .count();
        iv_times.push_back(elapsed);

        auto start1 = std::chrono::steady_clock::now();
        std::vector<size_t> v(il);
        auto end1 = std::chrono::steady_clock::now();
        auto elapsed1 =
            std::chrono::duration_cast<std::chrono::duration<float, std::nano>>(end1 - start1)
                .count();
        v_times.push_back(elapsed1);
    }

    std::cout << "IV min: " << *(std::min_element(iv_times.begin(), iv_times.end()))
              << " avg: " << std::reduce(iv_times.begin(), iv_times.end()) / 1000.0 << std::endl;
    std::cout << "VE min: " << *(std::min_element(v_times.begin(), v_times.end()))
              << " avg: " << std::reduce(v_times.begin(), v_times.end()) / 1000.0 << std::endl;
}

TEST(CPU_InlineVectorPerf3_NONE, Test)
{
    std::vector<float> iv_times;
    std::vector<float> v_times;

    std::initializer_list<size_t> il{1, 2, 3, 4, 5};
    size_t sum = 0;

    for(int i = 0; i < 1000; i++)
    {
        sum        = 0;
        auto start = std::chrono::steady_clock::now();
        miopen::InlineVector<size_t, 5> iv(il.begin(), il.end());
        for(int j = 0; j < iv.size(); j++)
        {
            sum += iv[j];
        }
        auto end = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<float, std::nano>>(end - start)
                .count();
        iv_times.push_back(elapsed);
        sum         = 0;
        auto start1 = std::chrono::steady_clock::now();
        std::vector<size_t> v(il.begin(), il.end());
        for(int j = 0; j < v.size(); j++)
        {
            sum += v[j];
        }
        auto end1 = std::chrono::steady_clock::now();
        auto elapsed1 =
            std::chrono::duration_cast<std::chrono::duration<float, std::nano>>(end1 - start1)
                .count();
        v_times.push_back(elapsed1);
    }

    std::cout << "IV min: " << *(std::min_element(iv_times.begin(), iv_times.end()))
              << " avg: " << std::reduce(iv_times.begin(), iv_times.end()) / 1000.0 << std::endl;
    std::cout << "VE min: " << *(std::min_element(v_times.begin(), v_times.end()))
              << " avg: " << std::reduce(v_times.begin(), v_times.end()) / 1000.0 << std::endl;
}

TEST(CPU_InlineVectorPerf4_NONE, Test)
{
    std::vector<float> iv_times;
    std::vector<float> v_times;

    std::initializer_list<size_t> il{1, 2, 3, 4, 5};
    size_t sum = 0;

    for(int i = 0; i < 1000; i++)
    {
        sum        = 0;
        auto start = std::chrono::steady_clock::now();
        miopen::InlineVector<size_t, 5> iv(il);
        auto first_not_one = std::find_if(iv.rbegin(), iv.rend(), [](int j) { return j != 1; });
        auto d             = std::distance(iv.begin(), first_not_one.base());
        int work_per_wg    = std::accumulate(iv.begin() + d, iv.end(), 1, std::multiplies<int>());
        auto end           = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<float, std::nano>>(end - start)
                .count();
        iv_times.push_back(elapsed);
        sum         = 0;
        auto start1 = std::chrono::steady_clock::now();
        std::vector<size_t> v(il);
        auto first_not_one1 = std::find_if(v.rbegin(), v.rend(), [](int j) { return j != 1; });
        auto d1             = std::distance(v.begin(), first_not_one1.base());
        int work_per_wg1    = std::accumulate(v.begin() + d1, v.end(), 1, std::multiplies<int>());
        auto end1           = std::chrono::steady_clock::now();
        auto elapsed1 =
            std::chrono::duration_cast<std::chrono::duration<float, std::nano>>(end1 - start1)
                .count();
        v_times.push_back(elapsed1);
    }

    std::cout << "IV min: " << *(std::min_element(iv_times.begin(), iv_times.end()))
              << " avg: " << std::reduce(iv_times.begin(), iv_times.end()) / 1000.0 << std::endl;
    std::cout << "VE min: " << *(std::min_element(v_times.begin(), v_times.end()))
              << " avg: " << std::reduce(v_times.begin(), v_times.end()) / 1000.0 << std::endl;
}
