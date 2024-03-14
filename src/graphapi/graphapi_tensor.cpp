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
#include <miopen/graphapi/graphapi_tensor.hpp>
#include <miopen/errors.hpp>

#include <algorithm>

namespace miopen {

namespace graphapi {

TensorBuilder& TensorBuilder::setDataType(miopenDataType_t dataType) &
{
    mTensor.setDataType(dataType);
    mDataTypeSet = true;
    return *this;
}

TensorBuilder& TensorBuilder::setDim(const std::vector<int64_t>& dimensions) &
{
    if(dimensions.empty() ||
       std::any_of(dimensions.cbegin(), dimensions.cend(), [](int64_t val) { return val <= 0; }))
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    mTensor.setDimensions(dimensions);
    mDimensionsSet = true;
    return *this;
}

TensorBuilder& TensorBuilder::setDim(std::vector<int64_t>&& dimensions) &
{
    if(dimensions.empty() ||
       std::any_of(dimensions.cbegin(), dimensions.cend(), [](int64_t val) { return val <= 0; }))
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    mTensor.setDimensions(std::move(dimensions));
    mDimensionsSet = true;
    return *this;
}

TensorBuilder& TensorBuilder::setStride(const std::vector<int64_t>& strides) &
{
    if(strides.empty() ||
       std::any_of(strides.cbegin(), strides.cend(), [](int64_t val) { return val <= 0; }))
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    mTensor.setStrides(strides);
    mStridesSet = true;
    return *this;
}

TensorBuilder& TensorBuilder::setStride(std::vector<int64_t>&& strides) &
{
    if(strides.empty() ||
       std::any_of(strides.cbegin(), strides.cend(), [](int64_t val) { return val <= 0; }))
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    mTensor.setStrides(std::move(strides));
    mStridesSet = true;
    return *this;
}

TensorBuilder& TensorBuilder::setId(int64_t id) &
{
    mTensor.setId(id);
    mUniqueIdSet = true;
    return *this;
}

TensorBuilder& TensorBuilder::setVirtual(bool isVirtual) &
{
    mTensor.setVirtual(isVirtual);
    return *this;
}

Tensor TensorBuilder::build() const&
{
    if(!mUniqueIdSet || !mDataTypeSet || !mDimensionsSet || !mStridesSet ||
       mTensor.getDimensions().size() != mTensor.getStrides().size())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    return mTensor;
}

Tensor TensorBuilder::build() &&
{
    if(!mUniqueIdSet || !mDataTypeSet || !mDimensionsSet || !mStridesSet ||
       mTensor.getDimensions().size() != mTensor.getStrides().size())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    return std::move(mTensor);
}

BackendTensorDescriptor::~BackendTensorDescriptor() = default;

void BackendTensorDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
                                           miopenBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           void* arrayOfElements)
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    switch(attributeName)
    {
    case MIOPEN_ATTR_TENSOR_UNIQUE_ID:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            mBuilder.setId(*static_cast<int64_t*>(arrayOfElements));
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_DATA_TYPE:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && elementCount == 1)
        {
            mBuilder.setDataType(*static_cast<miopenDataType_t*>(arrayOfElements));
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_DIMENSIONS:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount > 0)
        {
            mBuilder.setDim(
                std::vector<int64_t>(static_cast<int64_t*>(arrayOfElements),
                                     static_cast<int64_t*>(arrayOfElements) + elementCount));
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_STRIDES:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount > 0)
        {
            mBuilder.setStride(
                std::vector<int64_t>(static_cast<int64_t*>(arrayOfElements),
                                     static_cast<int64_t*>(arrayOfElements) + elementCount));
            return;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

    case MIOPEN_ATTR_TENSOR_IS_VIRTUAL:
        if(attributeType == MIOPEN_TYPE_BOOLEAN && elementCount == 1)
        {
            mBuilder.setVirtual(*static_cast<bool*>(arrayOfElements));
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
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mDescriptor = std::move(mBuilder).build();
    mFinalized  = true;
}

void BackendTensorDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
                                           miopenBackendAttributeType_t attributeType,
                                           int64_t requestedElementCount,
                                           int64_t* elementCount,
                                           void* arrayOfElements)
{
    if(!mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    switch(attributeName)
    {
    case MIOPEN_ATTR_TENSOR_UNIQUE_ID:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
        {
            *static_cast<int64_t*>(arrayOfElements) = mDescriptor.getId();
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
            *static_cast<miopenDataType_t*>(arrayOfElements) = mDescriptor.getDataType();
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
            const auto& dimensions = mDescriptor.getDimensions();
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
            const auto& strides = mDescriptor.getStrides();
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
            *static_cast<bool*>(arrayOfElements) = mDescriptor.getVirtual();
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
