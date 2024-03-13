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

#include <vector>

namespace miopen {

namespace graphapi {

class Tensor
{
public:
    Tensor() noexcept         = default;
    Tensor(const Tensor&)     = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    Tensor(miopenDataType_t dataType,
           const std::vector<int64_t>& dimensions,
           const std::vector<int64_t>& strides,
           int64_t id,
           bool isVirtual)
        : mDimensions(dimensions),
          mStrides(strides),
          mId(id),
          mDataType(dataType),
          mVirtual(isVirtual)
    {
    }
    Tensor(miopenDataType_t dataType,
           std::vector<int64_t>&& dimensions,
           std::vector<int64_t>&& strides,
           int64_t id,
           bool isVirtual) noexcept
        : mDimensions(std::move(dimensions)),
          mStrides(std::move(strides)),
          mId(id),
          mDataType(dataType),
          mVirtual(isVirtual)
    {
    }

    miopenDataType_t getDataType() const noexcept { return mDataType; }
    const std::vector<int64_t>& getDimensions() const noexcept { return mDimensions; }
    const std::vector<int64_t>& getStrides() const noexcept { return mStrides; }
    int64_t getId() const noexcept { return mId; }
    bool getVirtual() const noexcept { return mVirtual; }

    Tensor& setDataType(miopenDataType_t dataType) noexcept
    {
        mDataType = dataType;
        return *this;
    }
    Tensor& setDimensions(const std::vector<int64_t>& dimensions)
    {
        mDimensions = dimensions;
        return *this;
    }
    Tensor& setDimensions(std::vector<int64_t>&& dimensions) noexcept
    {
        mDimensions = std::move(dimensions);
        return *this;
    }
    Tensor& setStrides(const std::vector<int64_t>& strides)
    {
        mStrides = strides;
        return *this;
    }
    Tensor& setStrides(std::vector<int64_t>&& strides) noexcept
    {
        mStrides = std::move(strides);
        return *this;
    }
    Tensor& setId(int64_t id) noexcept
    {
        mId = id;
        return *this;
    }
    Tensor& setVirtual(bool isVirtual) noexcept
    {
        mVirtual = isVirtual;
        return *this;
    }

private:
    std::vector<int64_t> mDimensions;
    std::vector<int64_t> mStrides;
    int64_t mId                = 0;
    miopenDataType_t mDataType = miopenFloat;
    bool mVirtual              = false;
};

class TensorBuilder
{
public:
    TensorBuilder& setDataType(miopenDataType_t dataType) &;
    TensorBuilder& setDim(const std::vector<int64_t>& dimensions) &;
    TensorBuilder& setStride(const std::vector<int64_t>& strides) &;
    TensorBuilder& setId(int64_t id) &;
    TensorBuilder& setVirtual(bool isVirtual) &;

    TensorBuilder&& setDataType(miopenDataType_t dataType) &&
    {
        return std::move(setDataType(dataType));
    }
    TensorBuilder&& setDim(const std::vector<int64_t>& dimensions) &&
    {
        return std::move(setDim(dimensions));
    }
    TensorBuilder&& setStride(const std::vector<int64_t>& strides) &&
    {
        return std::move(setStride(strides));
    }
    TensorBuilder&& setId(int64_t id) && { return std::move(setId(id)); }
    TensorBuilder&& setVirtual(bool isVirtual) && { return std::move(setVirtual(isVirtual)); }

    Tensor build() const&;
    Tensor build() &&;

private:
    std::vector<int64_t> mDimensions;
    std::vector<int64_t> mStrides;
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

    Tensor& getTensor() { return mDescriptor; }

private:
    TensorBuilder mBuilder;
    Tensor mDescriptor;
};

} // namespace graphapi

} // namespace miopen
