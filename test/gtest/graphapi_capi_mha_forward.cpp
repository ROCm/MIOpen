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

#include <miopen/graphapi/tensor.hpp>
#include <miopen/miopen.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include <gtest/gtest.h>

miopenStatus_t CheckStatusAndThrow(miopenStatus_t status, const std::string& msg)
{
    if(status != miopenStatusSuccess)
    {
        MIOPEN_THROW(status, msg);
    }

    return status;
}

class DescriptorWrapper
{
public:
    DescriptorWrapper(miopenBackendDescriptorType_t descriptorType)
        : m_descriptorType(descriptorType), m_descriptor(nullptr)
    {
        CheckStatusAndThrow(miopenBackendCreateDescriptor(descriptorType, &m_descriptor),
                            "miopenBackendCreateDescriptor failed: type=" +
                                std::to_string(descriptorType));
    }

    ~DescriptorWrapper()
    {
        EXPECT_NE(m_descriptor, nullptr) << "m_descriptor is nullptr";

        miopenStatus_t status = miopenBackendDestroyDescriptor(m_descriptor);
        EXPECT_EQ(status, miopenStatusSuccess)
            << "Error while destroying descriptor, type: " << m_descriptorType;
    }

    miopenBackendDescriptor_t GetDescriptor() const { return m_descriptor; }
    miopenBackendDescriptorType_t GetDescriptorType() const { return m_descriptorType; }

private:
    miopenBackendDescriptorType_t m_descriptorType;
    miopenBackendDescriptor_t m_descriptor;
};

typedef std::unique_ptr<DescriptorWrapper> DescriptorWrapperPtr;

DescriptorWrapperPtr
MakeTensorDescriptor(int64_t uniqueId, int64_t n = 1, int64_t h = 1, int64_t s = 1, int64_t d = 1)
{
    DescriptorWrapperPtr descWrapperPtr =
        std::make_unique<DescriptorWrapper>(MIOPEN_BACKEND_TENSOR_DESCRIPTOR);

    miopenBackendDescriptor_t descriptor = descWrapperPtr->GetDescriptor();

    miopenDataType_t dtype = miopenFloat;

    CheckStatusAndThrow(
        miopenBackendSetAttribute(
            descriptor, MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &dtype),
        "miopenBackendSetAttribute for MIOPEN_ATTR_TENSOR_DATA_TYPE failed");

    int64_t dims[]    = {n, h, s, d};
    int64_t strides[] = {0, 0, 0, 0};
    int64_t alignment = 4;

    CheckStatusAndThrow(miopenBackendSetAttribute(
                            descriptor, MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT64, 4, dims),
                        "miopenBackendSetAttribute for MIOPEN_ATTR_TENSOR_DIMENSIONS failed");

    CheckStatusAndThrow(miopenBackendSetAttribute(
                            descriptor, MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT64, 4, strides),
                        "miopenBackendSetAttribute for MIOPEN_ATTR_TENSOR_STRIDES failed");

    CheckStatusAndThrow(
        miopenBackendSetAttribute(
            descriptor, MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_INT64, 1, &uniqueId),
        "miopenBackendSetAttribute for MIOPEN_ATTR_TENSOR_UNIQUE_ID failed");

    CheckStatusAndThrow(
        miopenBackendSetAttribute(
            descriptor, MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT, MIOPEN_TYPE_INT64, 1, &alignment),
        "miopenBackendSetAttribute for MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT failed");

    CheckStatusAndThrow(miopenBackendFinalize(descriptor),
                        "miopenBackendFinalize for Tensor descriptor failed");

    return descWrapperPtr;
}

TEST(TestCGraphApi, MhaForward)
{
    try
    {
    }
    catch(const miopen::Exception& ex)
    {
        FAIL() << ex.what();
    }
}
