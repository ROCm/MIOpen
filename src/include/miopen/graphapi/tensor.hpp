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

#include <cstdint>
#include <vector>

namespace miopen {

namespace graphapi {

class Tensor : public TensorDescriptor
{
private:
    int64_t mId   = 0;
    bool mVirtual = false;

    // Deprecated
    using TensorDescriptor::GetLayout_t;

public:
    Tensor() noexcept         = default;
    Tensor(const Tensor&)     = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    Tensor(const TensorDescriptor& other, int64_t id, bool isVirtual)
        : TensorDescriptor(other), mId(id), mVirtual(isVirtual)
    {
    }
    Tensor(TensorDescriptor&& other, int64_t id, bool isVirtual)
        : TensorDescriptor(std::move(other)), mId(id), mVirtual(isVirtual)
    {
    }
    Tensor(miopenDataType_t dataType,
           const std::vector<std::size_t>& dimensions,
           const std::vector<std::size_t>& strides,
           int64_t id,
           bool isVirtual)
        : TensorDescriptor(dataType, dimensions, strides), mId(id), mVirtual(isVirtual)
    {
    }
    Tensor(miopenDataType_t dataType,
           std::vector<std::size_t>&& dimensions,
           std::vector<std::size_t>&& strides,
           int64_t id,
           bool isVirtual) noexcept
        : TensorDescriptor(dataType, std::move(dimensions), std::move(strides)),
          mId(id),
          mVirtual(isVirtual)
    {
    }

    int64_t getId() const noexcept { return mId; }
    bool isVirtual() const noexcept { return mVirtual; }
};

class MIOPEN_INTERNALS_EXPORT TensorBuilder
{
private:
    std::vector<std::size_t> mDimensions;
    std::vector<std::size_t> mStrides;
    int64_t mId                = 0;
    miopenDataType_t mDataType = miopenFloat;
    bool mVirtual              = false;
    bool mUniqueIdSet          = false;
    bool mDataTypeSet          = false;
    bool mDimensionsSet        = false;
    bool mStridesSet           = false;

public:
    TensorBuilder& setDataType(miopenDataType_t dataType) &;
    TensorBuilder& setDim(const std::vector<std::size_t>& dimensions) &;
    TensorBuilder& setDim(std::vector<std::size_t>&& dimensions) &;
    TensorBuilder& setStride(const std::vector<std::size_t>& strides) &;
    TensorBuilder& setStride(std::vector<std::size_t>&& strides) &;
    TensorBuilder& setId(int64_t id) &;
    TensorBuilder& setVirtual(bool isVirtual) &;

    TensorBuilder&& setDataType(miopenDataType_t dataType) &&
    {
        return std::move(setDataType(dataType));
    }
    TensorBuilder&& setDim(const std::vector<std::size_t>& dimensions) &&
    {
        return std::move(setDim(dimensions));
    }
    TensorBuilder&& setDim(std::vector<std::size_t>&& dimensions) &&
    {
        return std::move(setDim(std::move(dimensions)));
    }
    TensorBuilder&& setStride(const std::vector<std::size_t>& strides) &&
    {
        return std::move(setStride(strides));
    }
    TensorBuilder&& setStride(std::vector<std::size_t>&& strides) &&
    {
        return std::move(setStride(std::move(strides)));
    }
    TensorBuilder&& setId(int64_t id) && { return std::move(setId(id)); }
    TensorBuilder&& setVirtual(bool isVirtual) && { return std::move(setVirtual(isVirtual)); }

    Tensor build() const&;
    Tensor build() &&;
};

class MIOPEN_INTERNALS_EXPORT BackendTensorDescriptor : public BackendDescriptor
{
private:
    TensorBuilder mBuilder;
    Tensor mDescriptor;

public:
    BackendTensorDescriptor() = default;
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

    const Tensor* getTensor() const { return &mDescriptor; }
    Tensor* getTensor() { return &mDescriptor; }
};

} // namespace graphapi

} // namespace miopen
