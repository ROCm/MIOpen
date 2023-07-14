/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include "test.hpp"
#include <miopen/sequences.hpp>

namespace miopen {
namespace seq {
namespace tests {
struct SimpleListTest
{
    void Run() const
    {
        auto it = test_seq.begin();
        EXPECT_EQUAL(*it, 1);
        EXPECT_EQUAL(*++it, 2);
        EXPECT_EQUAL(*++it, 4);
        EXPECT_EQUAL(*++it, 5);
        EXPECT(++it == test_seq.end());
    }

private:
    using TestSequence = Sequence<int, 1, 2, 4, 5>;
    TestSequence test_seq;
};

struct LinearListTest
{
    void Run() const
    {
        auto it = test_seq.begin();
        EXPECT_EQUAL(*it, 1);
        EXPECT_EQUAL(*++it, 2);
        EXPECT_EQUAL(*++it, 3);
        EXPECT(++it == test_seq.end());
    }

private:
    using TestSequence = Span<int, 1, 3>;
    TestSequence test_seq;
};

struct TwoPowersListTest
{
    void Run() const
    {
        auto it = test_seq.begin();
        EXPECT_EQUAL(*it, 4);
        EXPECT_EQUAL(*++it, 8);
        EXPECT_EQUAL(*++it, 16);
        EXPECT(++it == test_seq.end());
    }

private:
    using TestSequence = TwoPowersSpan<int, 4, 16>;
    TestSequence test_seq;
};

struct JoinTest
{
    void Run() const
    {
        auto it = test_seq.begin();
        EXPECT_EQUAL(*it, 1);
        EXPECT_EQUAL(*++it, 4);
        EXPECT_EQUAL(*++it, 8);
        EXPECT(++it == test_seq.end());
    }

private:
    using TestSequence = Join<Sequence<int, 1>, TwoPowersSpan<int, 4, 8>>;
    TestSequence test_seq;
};

struct DividedTest
{
    void Run() const
    {
        auto it = test_seq.begin();
        EXPECT_EQUAL(*it, 3);
        EXPECT_EQUAL(*++it, 6);
        EXPECT_EQUAL(*++it, 12);
        EXPECT_EQUAL(*++it, 15);
        EXPECT(++it == test_seq.end());
    }

private:
    using TestSequence = Multiplied<Sequence<int, 1, 2, 4, 5>, 3>;
    TestSequence test_seq;
};

struct RuleTest
{
    void Run() const
    {
        IsInTest();
        NextTest();
        CompareTest();
    }

private:
    struct TestData
    {
        int x;
        int y;
    };

    static const auto& TestRule()
    {
        static const auto instance = MakeRule(MakeMemberPtr(&TestData::x), Sequence<int, 1, 2>{});
        return instance;
    }

    void IsInTest() const
    {
        EXPECT(TestRule().IsIn({1, 2}));
        EXPECT(TestRule().IsIn({2, 3}));
        EXPECT(!TestRule().IsIn({3, 1}));
    }

    void NextTest() const
    {
        TestData data{-1, 2};
        TestRule().FillBegin(data);
        EXPECT(TestRule().IsEqualToBegin(data));
        EXPECT_EQUAL(data.x, 1);
        EXPECT_EQUAL(data.y, 2);
        EXPECT(!TestRule().Next(data));
        EXPECT(!TestRule().IsEqualToBegin(data));
        EXPECT_EQUAL(data.x, 2);
        EXPECT_EQUAL(data.y, 2);
        EXPECT(TestRule().Next(data));
        EXPECT_EQUAL(data.x, 1);
        EXPECT_EQUAL(data.y, 2);
    }

    void CompareTest() const
    {
        TestData data1{-1, 2};
        TestData data2{-1, 1};
        TestData data3{1, 2};
        EXPECT(TestRule().Compare(data1, data1));
        EXPECT(TestRule().Compare(data1, data2));
        EXPECT(!TestRule().Compare(data1, data3));
    }
};

struct RuleSetTest
{
    void Run() const
    {
        IsInTest();
        NextTest();
        CompareTest();
    }

private:
    struct TestData
    {
        int x;
        int y;
        int z;
    };

    // clang-format off
    static const auto& TestRuleSet()
    {
        static const auto instance = MakeRuleSet(
            std::make_tuple(Sequence<int, 1, 2>{}, &TestData::x),
            std::make_tuple(Sequence<int, 2, 3>{}, &TestData::y)
        );
        return instance;
    }
    // clang-format on

    void IsInTest() const
    {
        EXPECT(TestRuleSet().IsIn(TestData{1, 2, 5}));
        EXPECT(TestRuleSet().IsIn(TestData{2, 3, 5}));
        EXPECT(!TestRuleSet().IsIn(TestData{3, 2, 2}));
        EXPECT(!TestRuleSet().IsIn(TestData{2, 1, 2}));
    }

    struct S
    {
        float y;
    };

    void NextTest() const
    {
        TestData data{-1, -1, 5};
        TestRuleSet().FillBegin(data);
        EXPECT(TestRuleSet().IsEqualToBegin(data));
        EXPECT_EQUAL(data.x, 1);
        EXPECT_EQUAL(data.y, 2);
        EXPECT_EQUAL(data.z, 5);
        EXPECT(!TestRuleSet().Next(data));
        EXPECT(!TestRuleSet().IsEqualToBegin(data));
        EXPECT_EQUAL(data.x, 2);
        EXPECT_EQUAL(data.y, 2);
        EXPECT_EQUAL(data.z, 5);
        EXPECT(!TestRuleSet().Next(data));
        EXPECT_EQUAL(data.x, 1);
        EXPECT_EQUAL(data.y, 3);
        EXPECT_EQUAL(data.z, 5);
        EXPECT(!TestRuleSet().Next(data));
        EXPECT_EQUAL(data.x, 2);
        EXPECT_EQUAL(data.y, 3);
        EXPECT_EQUAL(data.z, 5);
        EXPECT(TestRuleSet().Next(data));
        EXPECT_EQUAL(data.x, 1);
        EXPECT_EQUAL(data.y, 2);
        EXPECT_EQUAL(data.z, 5);
    }

    void CompareTest() const
    {
        const TestData data1{-1, 2, 3};
        const TestData data2{-1, 2, 2};
        const TestData data3{1, 2, 3};
        const TestData data4{-1, 1, 2};
        EXPECT(TestRuleSet().Compare(data1, data1));
        EXPECT(TestRuleSet().Compare(data1, data2));
        EXPECT(!TestRuleSet().Compare(data1, data3));
        EXPECT(!TestRuleSet().Compare(data1, data4));
    }
};

} // namespace tests
} // namespace seq
} // namespace miopen

int main()
{
    miopen::seq::tests::SimpleListTest().Run();
    miopen::seq::tests::LinearListTest().Run();
    miopen::seq::tests::TwoPowersListTest().Run();
    miopen::seq::tests::JoinTest().Run();
    miopen::seq::tests::DividedTest().Run();
    miopen::seq::tests::RuleTest().Run();
    miopen::seq::tests::RuleSetTest().Run();

    return 0;
}
