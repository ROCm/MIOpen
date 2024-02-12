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
#include <miopen/graphapi_tensor.hpp>
#include <miopen/errors.hpp>

#include <algorithm>

namespace miopen {

namespace graphapi {

TensorBuilder& TensorBuilder::setDataType(miopenDataType_t dataType)
{
    mDataType    = dataType;
    mDataTypeSet = true;
    return *this;
}

TensorBuilder& TensorBuilder::setDim(int64_t numberDimensions, int64_t* dimensions)
{
    mDimensions    = std::vector<size_t>(dimensions, dimensions + numberDimensions);
    mDimensionsSet = true;
    return *this;
}

TensorBuilder& TensorBuilder::setId(int64_t id)
{
    mUniqueId    = id;
    mUniqueIdSet = true;
    return *this;
}

TensorBuilder& TensorBuilder::setStride(int64_t numberStrides, int64_t* strides)
{
    mStrides    = std::vector<size_t>(strides, strides + numberStrides);
    mStridesSet = true;
    return *this;
}

TensorBuilder& TensorBuilder::setVirtual(bool isVirtual)
{
    mVirtual = isVirtual;
    return *this;
}

std::shared_ptr<TensorDescriptorEx> TensorBuilder::build() const
{
    if(!mUniqueIdSet || !mDataTypeSet || !mDimensionsSet || !mStridesSet)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    return std::make_shared<TensorDescriptorEx>(
        mUniqueId, mVirtual, mDataType, mDimensions, mStrides);
}

BackendTensorDescriptor::BackendTensorDescriptor() : mBuilder(std::in_place), mDescriptor(nullptr)
{
}

BackendTensorDescriptor::~BackendTensorDescriptor() {}

void BackendTensorDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
                                           miopenBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           void* arrayOfElements)
{
    if(mFinalized || !mBuilder.has_value())
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    switch(attributeName)
    {
    case MIOPEN_ATTR_TENSOR_UNIQUE_ID:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            mBuilder->setId(*static_cast<int64_t*>(arrayOfElements));
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_DATA_TYPE:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && elementCount == 1)
        {
            mBuilder->setDataType(*static_cast<miopenDataType_t*>(arrayOfElements));
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_DIMENSIONS:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount > 0)
        {
            mBuilder->setDim(elementCount, static_cast<int64_t*>(arrayOfElements));
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_STRIDES:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount > 0)
        {
            mBuilder->setStride(elementCount, static_cast<int64_t*>(arrayOfElements));
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_IS_VIRTUAL:
        if(attributeType == MIOPEN_TYPE_BOOLEAN && elementCount == 1)
        {
            mBuilder->setVirtual(*static_cast<bool*>(arrayOfElements));
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT:
    case MIOPEN_ATTR_TENSOR_VECTOR_COUNT:
    case MIOPEN_ATTR_TENSOR_VECTORIZED_DIMENSION:
    case MIOPEN_ATTR_TENSOR_RAGGED_OFFSET_DESC: MIOPEN_THROW(miopenStatusNotImplemented);

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendTensorDescriptor::finalize()
{
    if(mFinalized || !mBuilder.has_value())
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mDescriptor = mBuilder->build();
    mBuilder.reset();
    mFinalized = true;
}

void BackendTensorDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
                                           miopenBackendAttributeType_t attributeType,
                                           int64_t requestedElementCount,
                                           int64_t* elementCount,
                                           void* arrayOfElements)
{
    if(!mFinalized || !mDescriptor)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    switch(attributeName)
    {
    case MIOPEN_ATTR_TENSOR_UNIQUE_ID:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
        {
            *static_cast<int64_t*>(arrayOfElements) = mDescriptor->getId();
            *elementCount                           = 1;
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_DATA_TYPE:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && requestedElementCount == 1)
        {
            *static_cast<miopenDataType_t*>(arrayOfElements) = mDescriptor->GetType();
            *elementCount                                    = 1;
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_DIMENSIONS:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount >= 0)
        {
            const auto& dimensions = mDescriptor->GetLengths();
            *elementCount          = dimensions.size();
            std::copy_n(dimensions.begin(),
                        std::min(*elementCount, requestedElementCount),
                        static_cast<int64_t*>(arrayOfElements));
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_STRIDES:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount >= 0)
        {
            const auto& strides = mDescriptor->GetStrides();
            *elementCount       = strides.size();
            std::copy_n(strides.begin(),
                        std::min(*elementCount, requestedElementCount),
                        static_cast<int64_t*>(arrayOfElements));
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_IS_VIRTUAL:
        if(attributeType == MIOPEN_TYPE_BOOLEAN && requestedElementCount == 1)
        {
            *static_cast<bool*>(arrayOfElements) = mDescriptor->getVirtual();
            *elementCount                        = 1;
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT:
    case MIOPEN_ATTR_TENSOR_VECTOR_COUNT:
    case MIOPEN_ATTR_TENSOR_VECTORIZED_DIMENSION:
    case MIOPEN_ATTR_TENSOR_RAGGED_OFFSET_DESC: MIOPEN_THROW(miopenStatusNotImplemented);

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // namespace graphapi

} // namespace miopen
