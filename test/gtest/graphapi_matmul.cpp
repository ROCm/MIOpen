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

#include <miopen/graphapi/matmul.hpp>

#include <tuple>
#include <gtest/gtest.h>
#include "graphapi_gtest_common.hpp"

TEST(CPU_GraphApiMatmulBuilder_NONE, Attributes)
{
    EXPECT_ANY_THROW({ miopen::graphapi::MatmulBuilder().build(); })
        << "Builder produced Matmul despite missing setComputeType call";
    EXPECT_NO_THROW({ miopen::graphapi::MatmulBuilder().setComputeType(miopenDouble).build(); })
        << "Builder failed to produce Matmul with valid attributes";
}

namespace {

using miopen::graphapi::GTestDescriptor;
using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;

class ComputeType : public GTestDescriptorSingleValueAttribute<miopenDataType_t, char>
{
public:
    ComputeType() = default;
    ComputeType(miopenDataType_t computeType)
        : GTestDescriptorSingleValueAttribute<miopenDataType_t, char>(
              true,
              "MIOPEN_ATTR_MATMUL_COMP_TYPE",
              MIOPEN_ATTR_MATMUL_COMP_TYPE,
              MIOPEN_TYPE_DATA_TYPE,
              MIOPEN_TYPE_CHAR,
              2,
              computeType)
    {
    }
};
} // namespace

void PrintTo(const miopenDataType_t& v, std::ostream* os) { *os << "compute type: " << v; }

class CPU_GraphApiMatMul_NONE : public testing::TestWithParam<miopenDataType_t>
{
protected:
    GTestGraphApiExecute<GTestDescriptorAttribute*> execute;

    ComputeType computeType;

    void SetUp() override
    {
        auto compType                 = GetParam();
        computeType                   = compType;
        execute.descriptor.attributes = {&computeType};
        execute.descriptor.attrsValid = true;
        execute.descriptor.textName   = "MIOPEN_BACKEND_MATMUL_DESCRIPTOR";
        execute.descriptor.type       = MIOPEN_BACKEND_MATMUL_DESCRIPTOR;
    }
};

TEST_P(CPU_GraphApiMatMul_NONE, CFuncions) { execute(); }

INSTANTIATE_TEST_SUITE_P(Unit, CPU_GraphApiMatMul_NONE, testing::Values(miopenFloat, miopenDouble));
