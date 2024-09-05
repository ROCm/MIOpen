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
#include <miopen/tensor.hpp>
#include <miopen/graphapi/tensor.hpp>

TEST(CPU_BackendApi_NONE, Tensor)
{
    miopenBackendDescriptor_t tensorDescriptor;

    miopenStatus_t status =
        miopenBackendCreateDescriptor(MIOPEN_BACKEND_TENSOR_DESCRIPTOR, &tensorDescriptor);
    ASSERT_TRUE(status == miopenStatusSuccess);

    status = miopenBackendFinalize(tensorDescriptor);
    EXPECT_FALSE(status == miopenStatusSuccess);

    int64_t theId = 1;
    status        = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_BOOLEAN, 1, &theId);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_INT64, 2, &theId);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_INT64, 1, &theId);
    EXPECT_TRUE(status == miopenStatusSuccess);

    miopenDataType_t theDataType = miopenFloat;
    status                       = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_CHAR, 1, &theDataType);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_DATA_TYPE, 2, &theDataType);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &theDataType);
    EXPECT_TRUE(status == miopenStatusSuccess);

    int64_t dimensions[4] = {4, 1, 16, 16};
    status                = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT32, 4, dimensions);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT64, 0, dimensions);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT64, 4, dimensions);
    EXPECT_TRUE(status == miopenStatusSuccess);

    int64_t strides[4] = {1 * 16 * 16, 16 * 16, 16, 1};
    status             = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT32, 4, strides);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT64, 0, strides);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT64, 4, strides);
    EXPECT_TRUE(status == miopenStatusSuccess);

    bool isVirtual = true;
    status         = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_IS_VIRTUAL, MIOPEN_TYPE_CHAR, 1, &isVirtual);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_IS_VIRTUAL, MIOPEN_TYPE_BOOLEAN, 2, &isVirtual);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendSetAttribute(
        tensorDescriptor, MIOPEN_ATTR_TENSOR_IS_VIRTUAL, MIOPEN_TYPE_BOOLEAN, 1, &isVirtual);
    EXPECT_TRUE(status == miopenStatusSuccess);

    int64_t elementCount = 0;

    int64_t retrievedId = 0;
    status              = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_UNIQUE_ID,
                                       MIOPEN_TYPE_INT64,
                                       1,
                                       &elementCount,
                                       &retrievedId);
    EXPECT_FALSE(status == miopenStatusSuccess);

    status = miopenBackendFinalize(tensorDescriptor);
    ASSERT_TRUE(status == miopenStatusSuccess);

    status = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_UNIQUE_ID,
                                       MIOPEN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &retrievedId);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_UNIQUE_ID,
                                       MIOPEN_TYPE_INT64,
                                       2,
                                       &elementCount,
                                       &retrievedId);
    EXPECT_FALSE(status == miopenStatusSuccess);
    elementCount = 0;
    status       = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_UNIQUE_ID,
                                       MIOPEN_TYPE_INT64,
                                       1,
                                       &elementCount,
                                       &retrievedId);
    ASSERT_TRUE(status == miopenStatusSuccess);
    EXPECT_EQ(elementCount, 1);
    EXPECT_TRUE(retrievedId == theId);

    miopenDataType_t retrievedDataType = miopenHalf;
    status                             = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_DATA_TYPE,
                                       MIOPEN_TYPE_CHAR,
                                       1,
                                       &elementCount,
                                       &retrievedDataType);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_DATA_TYPE,
                                       MIOPEN_TYPE_DATA_TYPE,
                                       2,
                                       &elementCount,
                                       &retrievedDataType);
    EXPECT_FALSE(status == miopenStatusSuccess);
    elementCount = 0;
    status       = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_DATA_TYPE,
                                       MIOPEN_TYPE_DATA_TYPE,
                                       1,
                                       &elementCount,
                                       &retrievedDataType);
    ASSERT_TRUE(status == miopenStatusSuccess);
    EXPECT_EQ(elementCount, 1);
    EXPECT_TRUE(retrievedDataType == theDataType);

    int64_t retrievedDimensions[4] = {0, 0, 0, 0};
    status                         = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_DIMENSIONS,
                                       MIOPEN_TYPE_INT32,
                                       4,
                                       &elementCount,
                                       retrievedDimensions);
    EXPECT_FALSE(status == miopenStatusSuccess);
    elementCount = 0;
    status       = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_DIMENSIONS,
                                       MIOPEN_TYPE_INT64,
                                       4,
                                       &elementCount,
                                       retrievedDimensions);
    ASSERT_TRUE(status == miopenStatusSuccess);
    EXPECT_EQ(elementCount, 4);
    for(int i = 0; i < elementCount; ++i)
    {
        EXPECT_TRUE(retrievedDimensions[i] == dimensions[i]);
    }

    int64_t retrievedStrides[4] = {0, 0, 0, 0};
    status                      = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_STRIDES,
                                       MIOPEN_TYPE_INT32,
                                       4,
                                       &elementCount,
                                       retrievedStrides);
    EXPECT_FALSE(status == miopenStatusSuccess);
    elementCount = 0;
    status       = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_STRIDES,
                                       MIOPEN_TYPE_INT64,
                                       4,
                                       &elementCount,
                                       retrievedStrides);
    ASSERT_TRUE(status == miopenStatusSuccess);
    EXPECT_EQ(elementCount, 4);
    for(int i = 0; i < elementCount; ++i)
    {
        EXPECT_TRUE(retrievedStrides[i] == strides[i]);
    }

    elementCount            = 0;
    bool retrievedIsVirtual = false;
    status                  = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_IS_VIRTUAL,
                                       MIOPEN_TYPE_CHAR,
                                       1,
                                       &elementCount,
                                       &retrievedIsVirtual);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_IS_VIRTUAL,
                                       MIOPEN_TYPE_BOOLEAN,
                                       2,
                                       &elementCount,
                                       &retrievedIsVirtual);
    EXPECT_FALSE(status == miopenStatusSuccess);
    status = miopenBackendGetAttribute(tensorDescriptor,
                                       MIOPEN_ATTR_TENSOR_IS_VIRTUAL,
                                       MIOPEN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &retrievedIsVirtual);
    ASSERT_TRUE(status == miopenStatusSuccess);
    EXPECT_EQ(elementCount, 1);
    EXPECT_TRUE(retrievedIsVirtual == isVirtual);
}

namespace gr = miopen::graphapi;

namespace graph_api_tensor_test {

static bool TestIsApplicable() { return true; }

using TestCase = std::tuple<std::vector<size_t>,
                            std::vector<size_t>,
                            miopenDataType_t,
                            std::optional<miopenTensorLayout_t>>;
static std::vector<TestCase> TestConfigs()
{
    return {{{1, 4, 14, 11, 1}, {616, 1, 44, 4, 4}, miopenFloat, miopenTensorNDHWC},
            {{1, 4, 14, 11, 1}, {616, 154, 11, 1, 1}, miopenFloat, miopenTensorNCDHW},
            {{1, 4, 11, 1}, {44, 1, 4, 4}, miopenFloat, miopenTensorNHWC},
            {{1, 4, 11, 1}, {44, 11, 1, 1}, miopenFloat, miopenTensorNCHW},
            {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, miopenFloat, miopenTensorNCDHW},
            {{1, 1, 1, 1}, {1, 1, 1, 1}, miopenFloat, miopenTensorNCHW},
            {{3, 5, 6}, {30, 6, 1}, miopenFloat, std::nullopt}};
}

class CPU_GraphTensor_NONE : public ::testing::TestWithParam<TestCase>
{
public:
    void SetUp() override
    {
        if(!TestIsApplicable())
        {
            GTEST_SKIP();
        }
    }

    void Run()
    {
        std::vector<std::size_t> dimensions;
        std::vector<std::size_t> strides;
        miopenDataType_t dataType;
        std::optional<miopenTensorLayout_t> layout;

        std::tie(dimensions, strides, dataType, layout) = GetParam();

        auto descriptor =
            layout ? miopen::TensorDescriptor(dataType, layout.value(), dimensions, strides)
                   : miopen::TensorDescriptor(dataType, dimensions, strides);

        gr::Tensor graphTensorFromDescription(descriptor, 0, false);
        EXPECT_EQ(graphTensorFromDescription, descriptor);
        EXPECT_EQ(graphTensorFromDescription.GetLayoutEnum(), descriptor.GetLayoutEnum());

        gr::Tensor graphTensorFromParams(dataType, dimensions, strides, 0, false);
        EXPECT_EQ(graphTensorFromParams, descriptor);
        EXPECT_EQ(graphTensorFromParams.GetLayoutEnum(), descriptor.GetLayoutEnum());
    }
};

} // namespace graph_api_tensor_test
using namespace graph_api_tensor_test;

TEST_P(CPU_GraphTensor_NONE, Test) { Run(); }

INSTANTIATE_TEST_SUITE_P(Unit, CPU_GraphTensor_NONE, testing::ValuesIn(TestConfigs()));
