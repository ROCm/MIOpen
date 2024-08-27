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

#include <miopen/graphapi/reduction.hpp>

#include <tuple>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using DescriptorTuple = std::tuple<miopenReduceTensorOp_t, miopenDataType_t>;
using miopen::graphapi::Reduction;
using miopen::graphapi::ReductionBuilder;

} // namespace

class CPU_GraphApiReductionBuilder_NONE : public testing::TestWithParam<DescriptorTuple>
{
protected:
    miopenReduceTensorOp_t mReductionOperator;
    miopenDataType_t mCompType;

    void SetUp() override { std::tie(mReductionOperator, mCompType) = GetParam(); }
};

TEST_P(CPU_GraphApiReductionBuilder_NONE, MissingSetter)
{
    EXPECT_NO_THROW({
        ReductionBuilder().setReductionOperator(mReductionOperator).setCompType(mCompType).build();
    }) << "Builder failed on valid attributes";
    EXPECT_ANY_THROW({ ReductionBuilder().setCompType(mCompType).build(); })
        << "Builder validated attributes despite missing setReductionOperator() call";
    EXPECT_ANY_THROW({ ReductionBuilder().setReductionOperator(mReductionOperator).build(); })
        << "Builder validated attributes despite missing setCompType() call";
}

namespace {

using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;

} // namespace

class CPU_GraphApiReduction_NONE : public testing::TestWithParam<DescriptorTuple>
{
private:
    // Pointers to these are stored inside 'mExecute' object (below)
    GTestDescriptorSingleValueAttribute<miopenReduceTensorOp_t, char> mReductionOperator;
    GTestDescriptorSingleValueAttribute<miopenDataType_t, char> mCompType;

protected:
    GTestGraphApiExecute<GTestDescriptorAttribute*> mExecute;

    void SetUp() override
    {
        auto [reductionOperator, compType] = GetParam();

        mReductionOperator = {true,
                              "MIOPEN_ATTR_REDUCTION_OPERATOR",
                              MIOPEN_ATTR_REDUCTION_OPERATOR,
                              MIOPEN_TYPE_REDUCTION_OPERATOR_TYPE,
                              MIOPEN_TYPE_CHAR,
                              2,
                              reductionOperator};

        mCompType = {true,
                     "MIOPEN_ATTR_REDUCTION_COMP_TYPE",
                     MIOPEN_ATTR_REDUCTION_COMP_TYPE,
                     MIOPEN_TYPE_DATA_TYPE,
                     MIOPEN_TYPE_CHAR,
                     2,
                     compType};

        mExecute.descriptor.attributes = {&mReductionOperator, &mCompType};

        mExecute.descriptor.attrsValid = true;
        mExecute.descriptor.textName   = "MIOPEN_BACKEND_REDUCTION_DESCRIPTOR";
        mExecute.descriptor.type       = MIOPEN_BACKEND_REDUCTION_DESCRIPTOR;
    }
};

TEST_P(CPU_GraphApiReduction_NONE, CFunctions) { mExecute(); }

static auto testCases =
    testing::Combine(testing::Values(MIOPEN_REDUCE_TENSOR_ADD, MIOPEN_REDUCE_TENSOR_MUL),
                     testing::Values(miopenFloat, miopenHalf));

INSTANTIATE_TEST_SUITE_P(Unit, CPU_GraphApiReductionBuilder_NONE, testCases);
INSTANTIATE_TEST_SUITE_P(Unit, CPU_GraphApiReduction_NONE, testCases);
