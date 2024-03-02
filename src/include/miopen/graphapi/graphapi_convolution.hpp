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
#include <miopen/graphapi/graphapi_tensor.hpp>
#include <miopen/convolution.hpp>
#include <miopen/problem.hpp>

#include <memory>
#include <vector>
#include <unordered_map>
#include <optional>

namespace miopen {

namespace graphapi {

class ConvolutionDescriptorEx : public ConvolutionDescriptor
{
public:
    ConvolutionDescriptorEx(miopenDataType_t theCompType,
                            size_t theSpatialDims,
                            miopenConvolutionMode_t theMode,
                            miopenPaddingMode_t thePadMode,
                            const std::vector<int>& thePrePaddings,
                            const std::vector<int>& theFilterStrides,
                            const std::vector<int>& theDilations,
                            const std::vector<int>& thePostPaddings);

    miopenDataType_t getCompType() const noexcept { return mCompType; }

private:
    miopenDataType_t mCompType;
};

class ConvolutionBuilder
{
public:
    ConvolutionBuilder& setCompType(miopenDataType_t dataType);
    ConvolutionBuilder& setMode(miopenConvolutionMode_t mode);
    ConvolutionBuilder& setSpatialDims(int64_t spatialDims);
    ConvolutionBuilder& setDilations(int64_t numberDilations, int64_t* dilations);
    ConvolutionBuilder& setFilterStrides(int64_t numberFilterStrides, int64_t* filterStrides);
    ConvolutionBuilder& setPrePaddings(int64_t numberPrePaddings, int64_t* prePaddings);
    ConvolutionBuilder& setPostPaddings(int64_t numberPostPaddings, int64_t* postPaddings);

    std::shared_ptr<ConvolutionDescriptorEx> build() const;

private:
    std::vector<int> mDilations;
    std::vector<int> mFilterStrides;
    std::vector<int> mPrePaddings;
    std::vector<int> mPostPaddings;
    int64_t mSpatialDims          = 0;
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
    BackendConvolutionDescriptor();
    virtual ~BackendConvolutionDescriptor() override;
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

    std::shared_ptr<ConvolutionDescriptorEx> getDescriptor() const noexcept { return mDescriptor; }

private:
    std::optional<ConvolutionBuilder> mBuilder;
    std::shared_ptr<ConvolutionDescriptorEx> mDescriptor;

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
};

class DirectedGraphNode
{
public:
    Problem& getProblem() noexcept { return mProblem; }

    void setInput(miopenTensorArgumentId_t name, std::shared_ptr<TensorDescriptorEx> descriptor);
    int getInputCount(std::shared_ptr<TensorDescriptorEx> descriptor);

    void setOutput(miopenTensorArgumentId_t name, std::shared_ptr<TensorDescriptorEx> descriptor);
    int getOutputCount(std::shared_ptr<TensorDescriptorEx> descriptor);

private:
    Problem mProblem;

    using EdgeSet = std::unordered_map<std::shared_ptr<TensorDescriptorEx>, int>;
    EdgeSet mInputs;
    EdgeSet mOutputs;
};

class OperationConvolution : public DirectedGraphNode
{
public:
    OperationConvolution(std::shared_ptr<ConvolutionDescriptorEx> convolution,
                         miopenProblemDirection_t direction,
                         double alpha,
                         double beta);

    double getAlpha() const noexcept { return mAlpha; }
    double getBeta() const noexcept { return mBeta; }

private:
    std::shared_ptr<ConvolutionDescriptorEx> mConvolution;
    double mAlpha; // TODO: propagate field to problem
    double mBeta;  // TODO: propagate field to problem
};

class OperationConvolutionBuilder
{
public:
    OperationConvolutionBuilder(miopenProblemDirection_t direction) : mDirection(direction) {}
    std::shared_ptr<OperationConvolution> build() const;

protected:
    void setConvolution(std::shared_ptr<ConvolutionDescriptorEx> convolution)
    {
        mConvolution    = convolution;
        mConvolutionSet = true;
    }
    void setX(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        mX    = descriptor;
        mXSet = true;
    }
    void setW(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        mW    = descriptor;
        mWSet = true;
    }
    void setY(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        mY    = descriptor;
        mYSet = true;
    }
    void setAlpha(double value)
    {
        mAlpha    = value;
        mAlphaSet = true;
    }
    void setBeta(double value)
    {
        mBeta    = value;
        mBetaSet = true;
    }

private:
    std::shared_ptr<ConvolutionDescriptorEx> mConvolution;
    std::shared_ptr<TensorDescriptorEx> mX;
    std::shared_ptr<TensorDescriptorEx> mW;
    std::shared_ptr<TensorDescriptorEx> mY;
    double mAlpha = 1.0;
    double mBeta  = 0.0;
    miopenProblemDirection_t mDirection;
    bool mConvolutionSet = false;
    bool mXSet           = false;
    bool mWSet           = false;
    bool mYSet           = false;
    bool mAlphaSet       = false;
    bool mBetaSet        = false;

    friend class BackendOperationConvolutionDescriptor;
};

class OperationConvolutionForwardBuilder : public OperationConvolutionBuilder
{
public:
    OperationConvolutionForwardBuilder()
        : OperationConvolutionBuilder(miopenProblemDirectionForward)
    {
    }
    OperationConvolutionForwardBuilder&
    setConvolution(std::shared_ptr<ConvolutionDescriptorEx> convolution)
    {
        OperationConvolutionBuilder::setConvolution(convolution);
        return *this;
    }
    OperationConvolutionForwardBuilder& setX(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        OperationConvolutionBuilder::setX(descriptor);
        return *this;
    }
    OperationConvolutionForwardBuilder& setW(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        OperationConvolutionBuilder::setW(descriptor);
        return *this;
    }
    OperationConvolutionForwardBuilder& setY(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        OperationConvolutionBuilder::setY(descriptor);
        return *this;
    }
    OperationConvolutionForwardBuilder& setAlpha(double value)
    {
        OperationConvolutionBuilder::setAlpha(value);
        return *this;
    }
    OperationConvolutionForwardBuilder& setBeta(double value)
    {
        OperationConvolutionBuilder::setBeta(value);
        return *this;
    }
};

class OperationConvolutionBackwardDataBuilder : public OperationConvolutionBuilder
{
public:
    OperationConvolutionBackwardDataBuilder()
        : OperationConvolutionBuilder(miopenProblemDirectionBackward)
    {
    }
    OperationConvolutionBackwardDataBuilder&
    setConvolution(std::shared_ptr<ConvolutionDescriptorEx> convolution)
    {
        OperationConvolutionBuilder::setConvolution(convolution);
        return *this;
    }
    OperationConvolutionBackwardDataBuilder& setDX(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        OperationConvolutionBuilder::setX(descriptor);
        return *this;
    }
    OperationConvolutionBackwardDataBuilder& setW(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        OperationConvolutionBuilder::setW(descriptor);
        return *this;
    }
    OperationConvolutionBackwardDataBuilder& setDY(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        OperationConvolutionBuilder::setY(descriptor);
        return *this;
    }
    OperationConvolutionBackwardDataBuilder& setAlpha(double value)
    {
        OperationConvolutionBuilder::setAlpha(value);
        return *this;
    }
    OperationConvolutionBackwardDataBuilder& setBeta(double value)
    {
        OperationConvolutionBuilder::setBeta(value);
        return *this;
    }
};

class OperationConvolutionBackwardFilterBuilder : public OperationConvolutionBuilder
{
public:
    OperationConvolutionBackwardFilterBuilder()
        : OperationConvolutionBuilder(miopenProblemDirectionBackwardWeights)
    {
    }
    OperationConvolutionBackwardFilterBuilder&
    setConvolution(std::shared_ptr<ConvolutionDescriptorEx> convolution)
    {
        OperationConvolutionBuilder::setConvolution(convolution);
        return *this;
    }
    OperationConvolutionBackwardFilterBuilder& setX(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        OperationConvolutionBuilder::setX(descriptor);
        return *this;
    }
    OperationConvolutionBackwardFilterBuilder& setDW(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        OperationConvolutionBuilder::setW(descriptor);
        return *this;
    }
    OperationConvolutionBackwardFilterBuilder& setDY(std::shared_ptr<TensorDescriptorEx> descriptor)
    {
        OperationConvolutionBuilder::setY(descriptor);
        return *this;
    }
    OperationConvolutionBackwardFilterBuilder& setAlpha(double value)
    {
        OperationConvolutionBuilder::setAlpha(value);
        return *this;
    }
    OperationConvolutionBackwardFilterBuilder& setBeta(double value)
    {
        OperationConvolutionBuilder::setBeta(value);
        return *this;
    }
};

class BackendOperationConvolutionDescriptor : public BackendDescriptor
{
public:
    BackendOperationConvolutionDescriptor(miopenProblemDirection_t direction)
        : mBuilder(std::in_place, direction)
    {
    }
    virtual ~BackendOperationConvolutionDescriptor() override;
    virtual void finalize() override;

    std::shared_ptr<OperationConvolution> getDescriptor() const noexcept { return mDescriptor; }

protected:
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

    std::optional<OperationConvolutionBuilder> mBuilder;
    std::shared_ptr<OperationConvolution> mDescriptor    = nullptr;
    miopenBackendDescriptor_t mBaseConvolutionDescriptor = nullptr;
    miopenBackendDescriptor_t mBaseXDescriptor           = nullptr;
    miopenBackendDescriptor_t mBaseWDescriptor           = nullptr;
    miopenBackendDescriptor_t mBaseYDescriptor           = nullptr;
};

class BackendOperationConvolutionForwardDescriptor : public BackendOperationConvolutionDescriptor
{
public:
    BackendOperationConvolutionForwardDescriptor()
        : BackendOperationConvolutionDescriptor(miopenProblemDirectionForward)
    {
    }
    virtual ~BackendOperationConvolutionForwardDescriptor() override;
    virtual void setAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              void* arrayOfElements) override;
    virtual void getAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) override;
};

class BackendOperationConvolutionBackwardDataDescriptor
    : public BackendOperationConvolutionDescriptor
{
public:
    BackendOperationConvolutionBackwardDataDescriptor()
        : BackendOperationConvolutionDescriptor(miopenProblemDirectionBackward)
    {
    }
    virtual ~BackendOperationConvolutionBackwardDataDescriptor() override;
    virtual void setAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              void* arrayOfElements) override;
    virtual void getAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) override;
};

class BackendOperationConvolutionBackwardFilterDescriptor
    : public BackendOperationConvolutionDescriptor
{
public:
    BackendOperationConvolutionBackwardFilterDescriptor()
        : BackendOperationConvolutionDescriptor(miopenProblemDirectionBackwardWeights)
    {
    }
    virtual ~BackendOperationConvolutionBackwardFilterDescriptor() override;
    virtual void setAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              void* arrayOfElements) override;
    virtual void getAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) override;
};

} // namespace graphapi

} // namespace miopen
