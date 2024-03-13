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
#pragma once

#include <miopen/graphapi/graphapi.hpp>
#include <miopen/tensor.hpp>

#include <memory>
#include <vector>
#include <optional>

namespace miopen {

namespace graphapi {

class TensorDescriptorEx : public TensorDescriptor
{
public:
    TensorDescriptorEx() = default;
    TensorDescriptorEx(int64_t id,
                       bool isVirtual,
                       miopenDataType_t dataType,
                       const std::vector<size_t>& dimensions,
                       const std::vector<size_t>& strides)
        : TensorDescriptor(dataType, dimensions, strides), mUniqueId(id), mVirtual(isVirtual)
    {
    }
    int64_t getId() const noexcept { return mUniqueId; }
    bool getVirtual() const noexcept { return mVirtual; }

private:
    int64_t mUniqueId = 0;
    bool mVirtual     = false;
};

class TensorBuilder
{
public:
    TensorBuilder& setDataType(miopenDataType_t dataType);
    TensorBuilder& setDim(int64_t numberDimensions, int64_t* dimensions);
    TensorBuilder& setId(int64_t id);
    TensorBuilder& setStride(int64_t numberStrides, int64_t* strides);
    TensorBuilder& setVirtual(bool isVirtual);

    std::shared_ptr<TensorDescriptorEx> build() const;

private:
    std::vector<size_t> mDimensions;
    std::vector<size_t> mStrides;
    int64_t mUniqueId          = 0;
    bool mVirtual              = false;
    miopenDataType_t mDataType = miopenFloat;
    bool mUniqueIdSet          = false;
    bool mDataTypeSet          = false;
    bool mDimensionsSet        = false;
    bool mStridesSet           = false;
};

class BackendTensorDescriptor : public BackendDescriptor
{
public:
    BackendTensorDescriptor();
    virtual ~BackendTensorDescriptor() override;
    virtual void setAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              void* arrayOfElements) override;
    virtual void finalize() override;
    virtual void getAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) override;

    std::shared_ptr<TensorDescriptorEx> tensorDescriptor() { return mDescriptor; }

private:
    std::optional<TensorBuilder> mBuilder;
    std::shared_ptr<TensorDescriptorEx> mDescriptor;
};

} // namespace graphapi

} // namespace miopen
