#include <miopen/miopen.h>
#include <gtest/gtest.h>

TEST(BackendApi, Tensor)
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
