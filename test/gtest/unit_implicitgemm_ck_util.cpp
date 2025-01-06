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

#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace unit_implicitgemm_ck_util_test {
struct ParsingTestCase
{
    std::string instanceToCheck;
    std::string suffix;
    bool expectedSupported;
    bool checkSplitK;

    friend std::ostream& operator<<(std::ostream& os, const ParsingTestCase& tc)
    {
        return os << "(instanceToCheck: " << tc.instanceToCheck << " suffix: " << tc.suffix
                  << " expectedSupported: " << tc.expectedSupported
                  << " checkSplitK: " << tc.checkSplitK << ")";
    }
};

static std::vector<ParsingTestCase> GetTestCases()
{
    return {{"test0", "+5", true, true},
            {"test1", "+", false, true},
            {"test2", "", false, true},
            {"test3", "+2+3", false, true},
            {"test4", "+9999999999999999999999999", false, true},
            {"test5", "", true, false},
            {"test6", "+", false, false},
            {"test7", "+2", false, false},
            {"test8", "+2+3", false, false}};
}

using ProblemDescription = miopen::conv::ProblemDescription;

struct StubbedDeviceOp
{
    StubbedDeviceOp(const std::string& op) : typeString(op) {}

    std::string typeString;

    std::string GetTypeString() { return typeString; }
};

struct StubbedDeviceOps
{
    static std::vector<std::string> deviceOps;

    static std::vector<std::unique_ptr<StubbedDeviceOp>> GetInstances()
    {
        std::vector<std::unique_ptr<StubbedDeviceOp>> ops;
        ops.reserve(deviceOps.size());

        std::transform(deviceOps.begin(),
                       deviceOps.end(),
                       std::back_inserter(ops),
                       [&](auto& deviceOp) { return std::make_unique<StubbedDeviceOp>(deviceOp); });

        return ops;
    }
};

std::vector<std::string> StubbedDeviceOps::deviceOps = {};

struct StubbedCKArgs
{
    StubbedCKArgs(const ProblemDescription& problem) {}

    template <typename ConvPtr>
    bool IsSupportedBy(const ConvPtr&) const
    {
        return true;
    }

    template <typename ConvPtr>
    bool IsSupportedBySplitK(const ConvPtr&, int) const
    {
        return true;
    }
};

template <typename CKArgsType, typename DeviceOpType>
class CKArgParsingTest : public ::testing::TestWithParam<ParsingTestCase>
{
protected:
    void TestParsing()
    {
        auto testCase = GetParam();

        // Set up stubbed instances to match test
        DeviceOpType::deviceOps.clear();
        DeviceOpType::deviceOps.push_back(testCase.instanceToCheck);

        bool success;
        if(testCase.checkSplitK)
        {
            success = miopen::solver::
                IsCKArgsSupported<DeviceOpType, CKArgsType, ProblemDescription, true>(
                    ProblemDescription{}, testCase.instanceToCheck + testCase.suffix);
        }
        else
        {
            success = miopen::solver::IsCKArgsSupported<DeviceOpType, CKArgsType>(
                ProblemDescription{}, testCase.instanceToCheck + testCase.suffix);
        }
        EXPECT_EQ(success, testCase.expectedSupported);
    }
};
} // namespace unit_implicitgemm_ck_util_test

using namespace unit_implicitgemm_ck_util_test;

struct CPU_UnitTestImplicitGemmCKUtil_NONE : CKArgParsingTest<StubbedCKArgs, StubbedDeviceOps>
{
};

TEST_P(CPU_UnitTestImplicitGemmCKUtil_NONE, TestParsing) { this->TestParsing(); };

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_UnitTestImplicitGemmCKUtil_NONE,
                         testing::ValuesIn(GetTestCases()));
