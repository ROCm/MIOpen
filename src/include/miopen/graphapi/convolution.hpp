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
#include <miopen/graphapi/operation.hpp>
#include <miopen/graphapi/tensor.hpp>

#include <cstdint>

namespace miopen {

namespace graphapi {

class Convolution
{
public:
    Convolution() noexcept              = default;
    Convolution(const Convolution&)     = default;
    Convolution(Convolution&&) noexcept = default;
    Convolution& operator=(const Convolution&) = default;
    Convolution& operator=(Convolution&&) noexcept = default;
    Convolution(miopenDataType_t compType,
                miopenConvolutionMode_t mode,
                size_t spatialDims,
                const std::vector<int64_t>& prePaddings,
                const std::vector<int64_t>& filterStrides,
                const std::vector<int64_t>& dilations,
                const std::vector<int64_t>& postPaddings)
        : mSpatialDims(spatialDims),
          mDilations(dilations),
          mFilterStrides(filterStrides),
          mPrePaddings(prePaddings),
          mPostPaddings(postPaddings),
          mCompType(compType),
          mMode(mode)
    {
    }
    Convolution(miopenDataType_t compType,
                miopenConvolutionMode_t mode,
                size_t spatialDims,
                std::vector<int64_t>&& prePaddings,
                std::vector<int64_t>&& filterStrides,
                std::vector<int64_t>&& dilations,
                std::vector<int64_t>&& postPaddings)
        : mSpatialDims(spatialDims),
          mDilations(std::move(dilations)),
          mFilterStrides(std::move(filterStrides)),
          mPrePaddings(std::move(prePaddings)),
          mPostPaddings(std::move(postPaddings)),
          mCompType(compType),
          mMode(mode)
    {
    }

    miopenDataType_t getCompType() const noexcept { return mCompType; }
    miopenConvolutionMode_t getMode() const noexcept { return mMode; }
    int64_t getSpatialDims() const noexcept { return mSpatialDims; }
    const std::vector<int64_t>& getDilations() const noexcept { return mDilations; }
    const std::vector<int64_t>& getFilterStrides() const noexcept { return mFilterStrides; }
    const std::vector<int64_t>& getPrePaddings() const noexcept { return mPrePaddings; }
    const std::vector<int64_t>& getPostPaddings() const noexcept { return mPostPaddings; }

private:
    int64_t mSpatialDims = 0;
    std::vector<int64_t> mDilations;
    std::vector<int64_t> mFilterStrides;
    std::vector<int64_t> mPrePaddings;
    std::vector<int64_t> mPostPaddings;
    miopenDataType_t mCompType    = miopenFloat;
    miopenConvolutionMode_t mMode = miopenConvolution;
};

class ConvolutionBuilder
{
public:
    ConvolutionBuilder& setCompType(miopenDataType_t compType) & noexcept;
    ConvolutionBuilder& setMode(miopenConvolutionMode_t mode) & noexcept;
    ConvolutionBuilder& setSpatialDims(int64_t spatialDims) & noexcept;
    ConvolutionBuilder& setDilations(const std::vector<int64_t>& dilations) &;
    ConvolutionBuilder& setDilations(std::vector<int64_t>&& dilations) & noexcept;
    ConvolutionBuilder& setFilterStrides(const std::vector<int64_t>& filterStrides) &;
    ConvolutionBuilder& setFilterStrides(std::vector<int64_t>&& filterStrides) & noexcept;
    ConvolutionBuilder& setPrePaddings(const std::vector<int64_t>& prePaddings) &;
    ConvolutionBuilder& setPrePaddings(std::vector<int64_t>&& prePaddings) & noexcept;
    ConvolutionBuilder& setPostPaddings(const std::vector<int64_t>& postPaddings) &;
    ConvolutionBuilder& setPostPaddings(std::vector<int64_t>&& postPaddings) & noexcept;

    ConvolutionBuilder&& setCompType(miopenDataType_t compType) && noexcept
    {
        return std::move(setCompType(compType));
    }
    ConvolutionBuilder&& setMode(miopenConvolutionMode_t mode) && noexcept
    {
        return std::move(setMode(mode));
    }
    ConvolutionBuilder&& setSpatialDims(int64_t spatialDims) && noexcept
    {
        return std::move(setSpatialDims(spatialDims));
    }
    ConvolutionBuilder&& setDilations(const std::vector<int64_t>& dilations) &&
    {
        return std::move(setDilations(dilations));
    }
    ConvolutionBuilder&& setDilations(std::vector<int64_t>&& dilations) && noexcept
    {
        return std::move(setDilations(std::move(dilations)));
    }
    ConvolutionBuilder&& setFilterStrides(const std::vector<int64_t>& filterStrides) &&
    {
        return std::move(setFilterStrides(filterStrides));
    }
    ConvolutionBuilder&& setFilterStrides(std::vector<int64_t>&& filterStrides) && noexcept
    {
        return std::move(setFilterStrides(std::move(filterStrides)));
    }
    ConvolutionBuilder&& setPrePaddings(const std::vector<int64_t>& prePaddings) &&
    {
        return std::move(setPrePaddings(prePaddings));
    }
    ConvolutionBuilder&& setPrePaddings(std::vector<int64_t>&& prePaddings) && noexcept
    {
        return std::move(setPrePaddings(std::move(prePaddings)));
    }
    ConvolutionBuilder&& setPostPaddings(const std::vector<int64_t>& postPaddings) &&
    {
        return std::move(setPostPaddings(postPaddings));
    }
    ConvolutionBuilder&& setPostPaddings(std::vector<int64_t>&& postPaddings) && noexcept
    {
        return std::move(setPostPaddings(std::move(postPaddings)));
    }

    Convolution build() const&;
    Convolution build() &&;

private:
    bool validate() const;

    int64_t mSpatialDims = 0;
    std::vector<int64_t> mDilations;
    std::vector<int64_t> mFilterStrides;
    std::vector<int64_t> mPrePaddings;
    std::vector<int64_t> mPostPaddings;
    miopenDataType_t mCompType    = miopenFloat;
    miopenConvolutionMode_t mMode = miopenConvolution;
    bool mCompTypeSet             = false;
    bool mModeSet                 = false;
    bool mSpatialDimsSet          = false;
    bool mDilationsSet            = false;
    bool mFilterStridesSet        = false;
    bool mPrePaddingsSet          = false;
    bool mPostPaddingsSet         = false;
};

class BackendConvolutionDescriptor : public BackendDescriptor
{
public:
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

    const Convolution* getConvolution() const noexcept { return &mConvolution; }
    Convolution* getConvolution() noexcept { return &mConvolution; }

private:
    void setCompType(miopenBackendAttributeType_t attributeType,
                     int64_t elementCount,
                     void* arrayOfElements);
    void setMode(miopenBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 void* arrayOfElements);
    void setSpatialDims(miopenBackendAttributeType_t attributeType,
                        int64_t elementCount,
                        void* arrayOfElements);
    void setDilations(miopenBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      void* arrayOfElements);
    void setFilterStrides(miopenBackendAttributeType_t attributeType,
                          int64_t elementCount,
                          void* arrayOfElements);
    void setPrePaddings(miopenBackendAttributeType_t attributeType,
                        int64_t elementCount,
                        void* arrayOfElements);
    void setPostPaddings(miopenBackendAttributeType_t attributeType,
                         int64_t elementCount,
                         void* arrayOfElements);

    void getCompType(miopenBackendAttributeType_t attributeType,
                     int64_t requestedElementCount,
                     int64_t* elementCount,
                     void* arrayOfElements);
    void getMode(miopenBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements);
    void getSpatialDims(miopenBackendAttributeType_t attributeType,
                        int64_t requestedElementCount,
                        int64_t* elementCount,
                        void* arrayOfElements);
    void getDilations(miopenBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements);
    void getFilterStrides(miopenBackendAttributeType_t attributeType,
                          int64_t requestedElementCount,
                          int64_t* elementCount,
                          void* arrayOfElements);
    void getPrePaddings(miopenBackendAttributeType_t attributeType,
                        int64_t requestedElementCount,
                        int64_t* elementCount,
                        void* arrayOfElements);
    void getPostPaddings(miopenBackendAttributeType_t attributeType,
                         int64_t requestedElementCount,
                         int64_t* elementCount,
                         void* arrayOfElements);

    ConvolutionBuilder mBuilder;
    Convolution mConvolution;
};

class OperationConvolution : public Operation
{
public:
    OperationConvolution() = default;
    OperationConvolution(Convolution* convolution,
                         Tensor* x,
                         Tensor* y,
                         Tensor* w,
                         double alpha,
                         double beta) noexcept
        : mConvolution(convolution), mX(x), mY(y), mW(w), mAlpha(alpha), mBeta(beta)
    {
    }

    Convolution* getConvolution() const noexcept { return mConvolution; }
    Tensor* getX() const noexcept { return mX; }
    Tensor* getY() const noexcept { return mY; }
    Tensor* getW() const noexcept { return mW; }
    double getAlpha() const noexcept { return mAlpha; }
    double getBeta() const noexcept { return mBeta; }

private:
    Convolution* mConvolution;
    Tensor* mX;
    Tensor* mY;
    Tensor* mW;
    double mAlpha;
    double mBeta;
};

class OperationConvolutionForward : public OperationConvolution
{
public:
    OperationConvolutionForward() = default;
    OperationConvolutionForward(Convolution* convolution,
                                Tensor* x,
                                Tensor* y,
                                Tensor* w,
                                double alpha,
                                double beta) noexcept
        : OperationConvolution(convolution, x, y, w, alpha, beta)
    {
        addInputTensor(x);
        addInputTensor(w);
        addOutputTensor(y);
        if(beta != 0.0)
        {
            addInputTensor(y);
        }
    }
};

class OperationConvolutionBuilder
{
public:
    OperationConvolutionBuilder& setConvolution(Convolution* convolution) noexcept
    {
        mConvolution = convolution;
        return *this;
    }
    OperationConvolutionBuilder& setX(Tensor* x) noexcept
    {
        mX = x;
        return *this;
    }
    OperationConvolutionBuilder& setY(Tensor* y) noexcept
    {
        mY = y;
        return *this;
    }
    OperationConvolutionBuilder& setW(Tensor* w) noexcept
    {
        mW = w;
        return *this;
    }
    OperationConvolutionBuilder& setAlpha(double alpha) noexcept
    {
        mAlpha    = alpha;
        mAlphaSet = true;
        return *this;
    }
    OperationConvolutionBuilder& setBeta(double beta) noexcept
    {
        mBeta    = beta;
        mBetaSet = true;
        return *this;
    }

protected:
    OperationConvolutionBuilder() = default;

    Convolution* mConvolution = nullptr;
    Tensor* mX                = nullptr;
    Tensor* mY                = nullptr;
    Tensor* mW                = nullptr;
    double mAlpha             = 1.0;
    double mBeta              = 0.0;
    bool mAlphaSet            = false;
    bool mBetaSet             = false;
};

class OperationConvolutionForwardBuilder : public OperationConvolutionBuilder
{
public:
    OperationConvolutionForwardBuilder& setConvolution(Convolution* convolution) noexcept
    {
        OperationConvolutionBuilder::setConvolution(convolution);
        return *this;
    }
    OperationConvolutionForwardBuilder& setX(Tensor* x) noexcept
    {
        OperationConvolutionBuilder::setX(x);
        return *this;
    }
    OperationConvolutionForwardBuilder& setY(Tensor* y) noexcept
    {
        OperationConvolutionBuilder::setY(y);
        return *this;
    }
    OperationConvolutionForwardBuilder& setW(Tensor* w) noexcept
    {
        OperationConvolutionBuilder::setW(w);
        return *this;
    }
    OperationConvolutionForwardBuilder& setAlpha(double alpha) noexcept
    {
        OperationConvolutionBuilder::setAlpha(alpha);
        return *this;
    }
    OperationConvolutionForwardBuilder& setBeta(double beta) noexcept
    {
        OperationConvolutionBuilder::setBeta(beta);
        return *this;
    }

    OperationConvolutionForward build() const
    {
        if(mConvolution == nullptr || mX == nullptr || mY == nullptr || mW == nullptr ||
           !mAlphaSet || !mBetaSet)
        {
            MIOPEN_THROW(miopenStatusNotInitialized);
        }
        return {mConvolution, mX, mY, mW, mAlpha, mBeta};
    }
};

class BackendOperationConvolutionDescriptor : public BackendDescriptor
{
protected:
    virtual OperationConvolutionBuilder& getBuilder()       = 0;
    virtual OperationConvolution& getOperationConvolution() = 0;

    miopenBackendDescriptor_t mConvolutionDescriptor = nullptr;
    miopenBackendDescriptor_t mXDescriptor           = nullptr;
    miopenBackendDescriptor_t mWDescriptor           = nullptr;
    miopenBackendDescriptor_t mYDescriptor           = nullptr;

    void setConvolution(miopenBackendAttributeType_t attributeType,
                        int64_t elementCount,
                        void* arrayOfElements);
    void
    setX(miopenBackendAttributeType_t attributeType, int64_t elementCount, void* arrayOfElements);
    void
    setW(miopenBackendAttributeType_t attributeType, int64_t elementCount, void* arrayOfElements);
    void
    setY(miopenBackendAttributeType_t attributeType, int64_t elementCount, void* arrayOfElements);
    void setAlpha(miopenBackendAttributeType_t attributeType,
                  int64_t elementCount,
                  void* arrayOfElements);
    void setBeta(miopenBackendAttributeType_t attributeType,
                 int64_t elementCount,
                 void* arrayOfElements);
    void getConvolution(miopenBackendAttributeType_t attributeType,
                        int64_t requestedElementCount,
                        int64_t* elementCount,
                        void* arrayOfElements);
    void getX(miopenBackendAttributeType_t attributeType,
              int64_t requestedElementCount,
              int64_t* elementCount,
              void* arrayOfElements);
    void getW(miopenBackendAttributeType_t attributeType,
              int64_t requestedElementCount,
              int64_t* elementCount,
              void* arrayOfElements);
    void getY(miopenBackendAttributeType_t attributeType,
              int64_t requestedElementCount,
              int64_t* elementCount,
              void* arrayOfElements);
    void getAlpha(miopenBackendAttributeType_t attributeType,
                  int64_t requestedElementCount,
                  int64_t* elementCount,
                  void* arrayOfElements);
    void getBeta(miopenBackendAttributeType_t attributeType,
                 int64_t requestedElementCount,
                 int64_t* elementCount,
                 void* arrayOfElements);
};

class BackendOperationConvolutionForwardDescriptor : public BackendOperationConvolutionDescriptor
{
public:
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
    virtual Operation* getOperation() override;

protected:
    virtual OperationConvolutionBuilder& getBuilder() override;
    virtual OperationConvolution& getOperationConvolution() override;

    OperationConvolutionForwardBuilder mBuilder;
    OperationConvolutionForward mOperation;
};

class OperationConvolutionBackwardData : public OperationConvolution
{
public:
    OperationConvolutionBackwardData() = default;
    OperationConvolutionBackwardData(Convolution* convolution,
                                     Tensor* x,
                                     Tensor* y,
                                     Tensor* w,
                                     double alpha,
                                     double beta) noexcept
        : OperationConvolution(convolution, x, y, w, alpha, beta)
    {
        addOutputTensor(x);
        addInputTensor(w);
        addInputTensor(y);
        if(beta != 0.0)
        {
            addInputTensor(x);
        }
    }
};

class OperationConvolutionBackwardDataBuilder : public OperationConvolutionBuilder
{
public:
    OperationConvolutionBackwardDataBuilder& setConvolution(Convolution* convolution) noexcept
    {
        OperationConvolutionBuilder::setConvolution(convolution);
        return *this;
    }
    OperationConvolutionBackwardDataBuilder& setX(Tensor* x) noexcept
    {
        OperationConvolutionBuilder::setX(x);
        return *this;
    }
    OperationConvolutionBackwardDataBuilder& setY(Tensor* y) noexcept
    {
        OperationConvolutionBuilder::setY(y);
        return *this;
    }
    OperationConvolutionBackwardDataBuilder& setW(Tensor* w) noexcept
    {
        OperationConvolutionBuilder::setW(w);
        return *this;
    }
    OperationConvolutionBackwardDataBuilder& setAlpha(double alpha) noexcept
    {
        OperationConvolutionBuilder::setAlpha(alpha);
        return *this;
    }
    OperationConvolutionBackwardDataBuilder& setBeta(double beta) noexcept
    {
        OperationConvolutionBuilder::setBeta(beta);
        return *this;
    }

    OperationConvolutionBackwardData build() const
    {
        if(mConvolution == nullptr || mX == nullptr || mY == nullptr || mW == nullptr ||
           !mAlphaSet || !mBetaSet)
        {
            MIOPEN_THROW(miopenStatusNotInitialized);
        }
        return {mConvolution, mX, mY, mW, mAlpha, mBeta};
    }
};

class BackendOperationConvolutionBackwardDataDescriptor
    : public BackendOperationConvolutionDescriptor
{
public:
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
    virtual Operation* getOperation() override;

private:
    virtual OperationConvolutionBuilder& getBuilder() override;
    virtual OperationConvolution& getOperationConvolution() override;

    OperationConvolutionBackwardDataBuilder mBuilder;
    OperationConvolutionBackwardData mOperation;
};

class OperationConvolutionBackwardFilter : public OperationConvolution
{
public:
    OperationConvolutionBackwardFilter() = default;
    OperationConvolutionBackwardFilter(Convolution* convolution,
                                       Tensor* x,
                                       Tensor* y,
                                       Tensor* w,
                                       double alpha,
                                       double beta) noexcept
        : OperationConvolution(convolution, x, y, w, alpha, beta)
    {
        addInputTensor(x);
        addOutputTensor(w);
        addInputTensor(y);
        if(beta != 0.0)
        {
            addInputTensor(w);
        }
    }
};

class OperationConvolutionBackwardFilterBuilder : public OperationConvolutionBuilder
{
public:
    OperationConvolutionBackwardFilterBuilder& setConvolution(Convolution* convolution) noexcept
    {
        OperationConvolutionBuilder::setConvolution(convolution);
        return *this;
    }
    OperationConvolutionBackwardFilterBuilder& setX(Tensor* x) noexcept
    {
        OperationConvolutionBuilder::setX(x);
        return *this;
    }
    OperationConvolutionBackwardFilterBuilder& setY(Tensor* y) noexcept
    {
        OperationConvolutionBuilder::setY(y);
        return *this;
    }
    OperationConvolutionBackwardFilterBuilder& setW(Tensor* w) noexcept
    {
        OperationConvolutionBuilder::setW(w);
        return *this;
    }
    OperationConvolutionBackwardFilterBuilder& setAlpha(double alpha) noexcept
    {
        OperationConvolutionBuilder::setAlpha(alpha);
        return *this;
    }
    OperationConvolutionBackwardFilterBuilder& setBeta(double beta) noexcept
    {
        OperationConvolutionBuilder::setBeta(beta);
        return *this;
    }

    OperationConvolutionBackwardFilter build() const
    {
        if(mConvolution == nullptr || mX == nullptr || mY == nullptr || mW == nullptr ||
           !mAlphaSet || !mBetaSet)
        {
            MIOPEN_THROW(miopenStatusNotInitialized);
        }
        return {mConvolution, mX, mY, mW, mAlpha, mBeta};
    }
};

class BackendOperationConvolutionBackwardFilterDescriptor
    : public BackendOperationConvolutionDescriptor
{
public:
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
    virtual Operation* getOperation() override;

private:
    virtual OperationConvolutionBuilder& getBuilder() override;
    virtual OperationConvolution& getOperationConvolution() override;

    OperationConvolutionBackwardFilterBuilder mBuilder;
    OperationConvolutionBackwardFilter mOperation;
};

} // namespace graphapi

} // namespace miopen
