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
#include <miopen/sequences.hpp>
#include <gtest/gtest.h>

namespace Seq = miopen::seq;

TEST(CPU_sequence_NONE, SimpleListTest)
{
    using TestSequence = Seq::Sequence<int, 1, 2, 4, 5>;
    TestSequence test_seq;
    auto it = test_seq.begin();
    EXPECT_EQ(*it, 1);
    EXPECT_EQ(*++it, 2);
    EXPECT_EQ(*++it, 4);
    EXPECT_EQ(*++it, 5);
    EXPECT_EQ(++it, test_seq.end());
}

TEST(CPU_sequence_NONE, LinearListTest)
{
    using TestSequence = Seq::Span<int, 1, 3>;
    TestSequence test_seq;

    auto it = test_seq.begin();
    EXPECT_EQ(*it, 1);
    EXPECT_EQ(*++it, 2);
    EXPECT_EQ(*++it, 3);
    EXPECT_EQ(++it, test_seq.end());
}

TEST(CPU_sequence_NONE, TwoPowersListTest)
{
    using TestSequence = Seq::TwoPowersSpan<int, 4, 16>;
    TestSequence test_seq;
    auto it = test_seq.begin();
    EXPECT_EQ(*it, 4);
    EXPECT_EQ(*++it, 8);
    EXPECT_EQ(*++it, 16);
    EXPECT_EQ(++it, test_seq.end());
}

TEST(CPU_sequence_NONE, JoinTest)
{
    using TestSequence = Seq::Join<Seq::Sequence<int, 1>, Seq::TwoPowersSpan<int, 4, 8>>;
    TestSequence test_seq;
    auto it = test_seq.begin();
    EXPECT_EQ(*it, 1);
    EXPECT_EQ(*++it, 4);
    EXPECT_EQ(*++it, 8);
    EXPECT_EQ(++it, test_seq.end());
}

TEST(CPU_sequence_NONE, DividedTest)
{
    using TestSequence = Seq::Multiplied<Seq::Sequence<int, 1, 2, 4, 5>, 3>;
    TestSequence test_seq;
    auto it = test_seq.begin();
    EXPECT_EQ(*it, 3);
    EXPECT_EQ(*++it, 6);
    EXPECT_EQ(*++it, 12);
    EXPECT_EQ(*++it, 15);
    EXPECT_EQ(++it, test_seq.end());
}

namespace {
class CPU_sequenceRule_NONE : public ::testing::Test
{
protected:
    struct TestData
    {
        int x;
        int y;
    };

    static const auto& TestRule()
    {
        static const auto instance =
            Seq::MakeRule(Seq::MakeMemberPtr(&TestData::x), Seq::Sequence<int, 1, 2>{});
        return instance;
    }
};
}; // namespace

TEST_F(CPU_sequenceRule_NONE, IsInTest)
{
    EXPECT_TRUE(TestRule().IsIn({1, 2}));
    EXPECT_TRUE(TestRule().IsIn({2, 3}));
    EXPECT_FALSE(TestRule().IsIn({3, 1}));
}

TEST_F(CPU_sequenceRule_NONE, NextTest)
{
    TestData data{-1, 2};
    TestRule().FillBegin(data);
    EXPECT_TRUE(TestRule().IsEqualToBegin(data));
    EXPECT_EQ(data.x, 1);
    EXPECT_EQ(data.y, 2);
    EXPECT_FALSE(TestRule().Next(data));
    EXPECT_FALSE(TestRule().IsEqualToBegin(data));
    EXPECT_EQ(data.x, 2);
    EXPECT_EQ(data.y, 2);
    EXPECT_TRUE(TestRule().Next(data));
    EXPECT_EQ(data.x, 1);
    EXPECT_EQ(data.y, 2);
}

TEST_F(CPU_sequenceRule_NONE, CompareTest)
{
    TestData data1{-1, 2};
    TestData data2{-1, 1};
    TestData data3{1, 2};
    EXPECT_TRUE(TestRule().Compare(data1, data1));
    EXPECT_TRUE(TestRule().Compare(data1, data2));
    EXPECT_FALSE(TestRule().Compare(data1, data3));
}

namespace {
class CPU_sequenceRuleSet_NONE : public ::testing::Test
{
protected:
    struct TestData
    {
        int x;
        int y;
        int z;
    };

    static const auto& TestRule()
    {
        static const auto instance =
            Seq::MakeRuleSet(std::make_tuple(Seq::Sequence<int, 1, 2>{}, &TestData::x),
                             std::make_tuple(Seq::Sequence<int, 2, 3>{}, &TestData::y));
        return instance;
    }
};
}; // namespace

TEST_F(CPU_sequenceRuleSet_NONE, IsInTest)
{
    EXPECT_TRUE(TestRule().IsIn(TestData{1, 2, 5}));
    EXPECT_TRUE(TestRule().IsIn(TestData{2, 3, 5}));
    EXPECT_FALSE(TestRule().IsIn(TestData{3, 2, 2}));
    EXPECT_FALSE(TestRule().IsIn(TestData{2, 1, 2}));
}

TEST_F(CPU_sequenceRuleSet_NONE, NextTest)
{
    TestData data{-1, -1, 5};
    TestRule().FillBegin(data);
    EXPECT_TRUE(TestRule().IsEqualToBegin(data));
    EXPECT_EQ(data.x, 1);
    EXPECT_EQ(data.y, 2);
    EXPECT_EQ(data.z, 5);
    EXPECT_FALSE(TestRule().Next(data));
    EXPECT_FALSE(TestRule().IsEqualToBegin(data));
    EXPECT_EQ(data.x, 2);
    EXPECT_EQ(data.y, 2);
    EXPECT_EQ(data.z, 5);
    EXPECT_FALSE(TestRule().Next(data));
    EXPECT_EQ(data.x, 1);
    EXPECT_EQ(data.y, 3);
    EXPECT_EQ(data.z, 5);
    EXPECT_FALSE(TestRule().Next(data));
    EXPECT_EQ(data.x, 2);
    EXPECT_EQ(data.y, 3);
    EXPECT_EQ(data.z, 5);
    EXPECT_TRUE(TestRule().Next(data));
    EXPECT_EQ(data.x, 1);
    EXPECT_EQ(data.y, 2);
    EXPECT_EQ(data.z, 5);
}

TEST_F(CPU_sequenceRuleSet_NONE, CompareTest)
{
    const TestData data1{-1, 2, 3};
    const TestData data2{-1, 2, 2};
    const TestData data3{1, 2, 3};
    const TestData data4{-1, 1, 2};
    EXPECT_TRUE(TestRule().Compare(data1, data1));
    EXPECT_TRUE(TestRule().Compare(data1, data2));
    EXPECT_FALSE(TestRule().Compare(data1, data3));
    EXPECT_FALSE(TestRule().Compare(data1, data4));
}
