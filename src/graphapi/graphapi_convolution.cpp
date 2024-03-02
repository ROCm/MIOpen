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
#include <miopen/graphapi/graphapi_convolution.hpp>
#include <miopen/errors.hpp>

#include <limits>
#include <algorithm>

namespace miopen {

namespace graphapi {

ConvolutionDescriptorEx::ConvolutionDescriptorEx(miopenDataType_t theCompType,
                                                 size_t theSpatialDims,
                                                 miopenConvolutionMode_t theMode,
                                                 miopenPaddingMode_t thePadMode,
                                                 const std::vector<int>& thePrePaddings,
                                                 const std::vector<int>& theFilterStrides,
                                                 const std::vector<int>& theDilations,
                                                 const std::vector<int>& thePostPaddings)
    : ConvolutionDescriptor(theSpatialDims,
                            theMode,
                            thePadMode,
                            thePrePaddings,
                            theFilterStrides,
                            theDilations,
                            thePostPaddings),
      mCompType(theCompType)
{
}

ConvolutionBuilder& ConvolutionBuilder::setCompType(miopenDataType_t dataType)
{
    mCompType    = dataType;
    mCompTypeSet = true;
    return *this;
}

ConvolutionBuilder& ConvolutionBuilder::setMode(miopenConvolutionMode_t mode)
{
    mMode    = mode;
    mModeSet = true;
    return *this;
}

ConvolutionBuilder& ConvolutionBuilder::setSpatialDims(int64_t spatialDims)
{
    mSpatialDims    = spatialDims;
    mSpatialDimsSet = true;
    return *this;
}

ConvolutionBuilder& ConvolutionBuilder::setDilations(int64_t numberDilations, int64_t* dilations)
{
    mDilationsSet = std::all_of(dilations, dilations + numberDilations, [](auto value) {
        return value >= std::numeric_limits<int>::lowest() &&
               value <= std::numeric_limits<int>::max();
    });

    if(!mDilationsSet)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    mDilations = std::vector<int>(dilations, dilations + numberDilations);

    return *this;
}

ConvolutionBuilder& ConvolutionBuilder::setFilterStrides(int64_t numberFilterStrides,
                                                         int64_t* filterStrides)
{
    mFilterStridesSet =
        std::all_of(filterStrides, filterStrides + numberFilterStrides, [](auto value) {
            return value >= std::numeric_limits<int>::lowest() &&
                   value <= std::numeric_limits<int>::max();
        });

    if(!mFilterStridesSet)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    mFilterStrides = std::vector<int>(filterStrides, filterStrides + numberFilterStrides);

    return *this;
}

ConvolutionBuilder& ConvolutionBuilder::setPrePaddings(int64_t numberPrePaddings,
                                                       int64_t* prePaddings)
{
    mDilationsSet = std::all_of(prePaddings, prePaddings + numberPrePaddings, [](auto value) {
        return value >= std::numeric_limits<int>::lowest() &&
               value <= std::numeric_limits<int>::max();
    });

    if(!mPrePaddingsSet)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    mPrePaddings = std::vector<int>(prePaddings, prePaddings + numberPrePaddings);

    return *this;
}

ConvolutionBuilder& ConvolutionBuilder::setPostPaddings(int64_t numberPostPaddings,
                                                        int64_t* postPaddings)
{
    mDilationsSet = std::all_of(postPaddings, postPaddings + numberPostPaddings, [](auto value) {
        return value >= std::numeric_limits<int>::lowest() &&
               value <= std::numeric_limits<int>::max();
    });

    if(!mPostPaddingsSet)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    mPostPaddings = std::vector<int>(postPaddings, postPaddings + numberPostPaddings);

    return *this;
}

std::shared_ptr<ConvolutionDescriptorEx> ConvolutionBuilder::build() const
{
    if(!mCompTypeSet || !mModeSet || !mSpatialDimsSet || !mDilationsSet || !mFilterStridesSet ||
       !mPrePaddingsSet || !mPostPaddingsSet ||
       (mDilations.size() != mSpatialDims && mFilterStrides.size() != mSpatialDims &&
        mPrePaddings.size() != mSpatialDims && mPostPaddings.size() != mSpatialDims))
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    return std::make_shared<ConvolutionDescriptorEx>(mCompType,
                                                     mSpatialDims,
                                                     mMode,
                                                     miopenPaddingDefault,
                                                     mPrePaddings,
                                                     mFilterStrides,
                                                     mDilations,
                                                     mPostPaddings);
}

BackendConvolutionDescriptor::BackendConvolutionDescriptor()
    : mBuilder(std::in_place), mDescriptor(nullptr)
{
}

BackendConvolutionDescriptor::~BackendConvolutionDescriptor() {}

void BackendConvolutionDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_CONVOLUTION_COMP_TYPE:
        setCompType(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_CONV_MODE:
        setMode(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS:
        setSpatialDims(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_DILATIONS:
        setDilations(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES:
        setFilterStrides(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS:
        setPrePaddings(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS:
        setPostPaddings(attributeType, elementCount, arrayOfElements);
        return;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::finalize()
{
    if(mFinalized || !mBuilder.has_value())
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mDescriptor = mBuilder->build();
    mBuilder.reset();
    mFinalized = true;
}

void BackendConvolutionDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_CONVOLUTION_COMP_TYPE:
        getCompType(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_CONV_MODE:
        getMode(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS:
        getSpatialDims(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_DILATIONS:
        getDilations(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES:
        getFilterStrides(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS:
        getPrePaddings(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS:
        getPostPaddings(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::setCompType(miopenBackendAttributeType_t attributeType,
                                               int64_t elementCount,
                                               void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_DATA_TYPE && elementCount == 1)
    {
        if(mBuilder)
            mBuilder->setCompType(*static_cast<miopenDataType_t*>(arrayOfElements));
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::setMode(miopenBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_CONVOLUTION_MODE && elementCount == 1)
    {
        if(mBuilder)
            mBuilder->setMode(*static_cast<miopenConvolutionMode_t*>(arrayOfElements));
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::setSpatialDims(miopenBackendAttributeType_t attributeType,
                                                  int64_t elementCount,
                                                  void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
    {
        if(mBuilder)
            mBuilder->setSpatialDims(*static_cast<int64_t*>(arrayOfElements));
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::setDilations(miopenBackendAttributeType_t attributeType,
                                                int64_t elementCount,
                                                void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_INT64 && elementCount > 0)
    {
        if(mBuilder)
            mBuilder->setDilations(elementCount, static_cast<int64_t*>(arrayOfElements));
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::setFilterStrides(miopenBackendAttributeType_t attributeType,
                                                    int64_t elementCount,
                                                    void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_INT64 && elementCount > 0)
    {
        if(mBuilder)
            mBuilder->setFilterStrides(elementCount, static_cast<int64_t*>(arrayOfElements));
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::setPrePaddings(miopenBackendAttributeType_t attributeType,
                                                  int64_t elementCount,
                                                  void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_INT64 && elementCount > 0)
    {
        if(mBuilder)
            mBuilder->setPrePaddings(elementCount, static_cast<int64_t*>(arrayOfElements));
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::setPostPaddings(miopenBackendAttributeType_t attributeType,
                                                   int64_t elementCount,
                                                   void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_INT64 && elementCount > 0)
    {
        if(mBuilder)
            mBuilder->setPostPaddings(elementCount, static_cast<int64_t*>(arrayOfElements));
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::getCompType(miopenBackendAttributeType_t attributeType,
                                               int64_t requestedElementCount,
                                               int64_t* elementCount,
                                               void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_DATA_TYPE && requestedElementCount == 1)
    {
        *static_cast<miopenDataType_t*>(arrayOfElements) = mDescriptor->getCompType();
        *elementCount                                    = 1;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::getMode(miopenBackendAttributeType_t attributeType,
                                           int64_t requestedElementCount,
                                           int64_t* elementCount,
                                           void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_CONVOLUTION_MODE && requestedElementCount == 1)
    {
        *static_cast<miopenConvolutionMode_t*>(arrayOfElements) = mDescriptor->mode;
        *elementCount                                           = 1;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::getSpatialDims(miopenBackendAttributeType_t attributeType,
                                                  int64_t requestedElementCount,
                                                  int64_t* elementCount,
                                                  void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
    {
        *static_cast<int64_t*>(arrayOfElements) = mDescriptor->spatialDim;
        *elementCount                           = 1;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::getDilations(miopenBackendAttributeType_t attributeType,
                                                int64_t requestedElementCount,
                                                int64_t* elementCount,
                                                void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount >= 0)
    {
        const auto& dilations = mDescriptor->GetConvDilations();
        *elementCount         = dilations.size();
        std::copy_n(dilations.begin(),
                    std::min(*elementCount, requestedElementCount),
                    static_cast<int64_t*>(arrayOfElements));
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::getFilterStrides(miopenBackendAttributeType_t attributeType,
                                                    int64_t requestedElementCount,
                                                    int64_t* elementCount,
                                                    void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount >= 0)
    {
        const auto& strides = mDescriptor->GetConvStrides();
        *elementCount       = strides.size();
        std::copy_n(strides.begin(),
                    std::min(*elementCount, requestedElementCount),
                    static_cast<int64_t*>(arrayOfElements));
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::getPrePaddings(miopenBackendAttributeType_t attributeType,
                                                  int64_t requestedElementCount,
                                                  int64_t* elementCount,
                                                  void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount >= 0)
    {
        const auto& pads = mDescriptor->GetConvPads();
        *elementCount    = pads.size();
        std::copy_n(pads.begin(),
                    std::min(*elementCount, requestedElementCount),
                    static_cast<int64_t*>(arrayOfElements));
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendConvolutionDescriptor::getPostPaddings(miopenBackendAttributeType_t attributeType,
                                                   int64_t requestedElementCount,
                                                   int64_t* elementCount,
                                                   void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount >= 0)
    {
        const auto& postPads = mDescriptor->GetTransposeConvPads();
        *elementCount        = postPads.size();
        std::copy_n(postPads.begin(),
                    std::min(*elementCount, requestedElementCount),
                    static_cast<int64_t*>(arrayOfElements));
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void DirectedGraphNode::setInput(miopenTensorArgumentId_t name,
                                 std::shared_ptr<TensorDescriptorEx> descriptor)
{
    mProblem.RegisterTensorDescriptor(name, miopen::deref(descriptor));
    ++mInputs[descriptor];
}

int DirectedGraphNode::getInputCount(std::shared_ptr<TensorDescriptorEx> descriptor)
{
    auto iter = mInputs.find(descriptor);
    return iter != mInputs.end() ? iter->second : 0;
}

void DirectedGraphNode::setOutput(miopenTensorArgumentId_t name,
                                  std::shared_ptr<TensorDescriptorEx> descriptor)
{
    mProblem.RegisterTensorDescriptor(name, miopen::deref(descriptor));
    ++mOutputs[descriptor];
}

int DirectedGraphNode::getOutputCount(std::shared_ptr<TensorDescriptorEx> descriptor)
{
    auto iter = mOutputs.find(descriptor);
    return iter != mOutputs.end() ? iter->second : 0;
}

OperationConvolution::OperationConvolution(std::shared_ptr<ConvolutionDescriptorEx> convolution,
                                           miopenProblemDirection_t direction,
                                           double alpha,
                                           double beta)
    : mConvolution(std::move(convolution)), mAlpha(alpha), mBeta(beta)
{
    auto& problem = getProblem();
    problem.SetDirection(direction);
    problem.SetOperatorDescriptor(*mConvolution);
}

std::shared_ptr<OperationConvolution> OperationConvolutionBuilder::build() const
{
    if(!mConvolutionSet || !mXSet || !mWSet || !mYSet || !mAlphaSet || !mBetaSet)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    auto graphNode =
        std::make_shared<OperationConvolution>(mConvolution, mDirection, mAlpha, mBeta);
    graphNode->setInput(miopenTensorConvolutionX, mX);
    graphNode->setInput(miopenTensorConvolutionW, mW);
    graphNode->setOutput(miopenTensorConvolutionY, mY);

    if(mBeta != 0.0)
    {
        graphNode->setInput(miopenTensorConvolutionY, mY);
    }

    return graphNode;
}

BackendOperationConvolutionDescriptor::~BackendOperationConvolutionDescriptor() {}

void BackendOperationConvolutionDescriptor::finalize()
{
    if(mFinalized || !mBuilder.has_value())
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mDescriptor = mBuilder->build();
    mBuilder.reset();
    mFinalized = true;
}

void BackendOperationConvolutionDescriptor::setConvolution(
    miopenBackendAttributeType_t attributeType, int64_t elementCount, void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
    {
        mBaseConvolutionDescriptor =
            deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
        auto& baseDescr = deref(mBaseConvolutionDescriptor);
        auto& convDescr = dynamic_cast<BackendConvolutionDescriptor&>(baseDescr);
        if(mBuilder)
        {
            mBuilder->setConvolution(convDescr.getDescriptor());
        }
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionDescriptor::setX(miopenBackendAttributeType_t attributeType,
                                                 int64_t elementCount,
                                                 void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
    {
        mBaseXDescriptor = deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
        auto& baseDescr  = deref(mBaseXDescriptor);
        auto& tensDescr  = dynamic_cast<BackendTensorDescriptor&>(baseDescr);
        if(mBuilder)
        {
            mBuilder->setX(tensDescr.tensorDescriptor());
        }
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionDescriptor::setW(miopenBackendAttributeType_t attributeType,
                                                 int64_t elementCount,
                                                 void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
    {
        mBaseWDescriptor = deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
        auto& baseDescr  = deref(mBaseWDescriptor);
        auto& tensDescr  = dynamic_cast<BackendTensorDescriptor&>(baseDescr);
        if(mBuilder)
        {
            mBuilder->setW(tensDescr.tensorDescriptor());
        }
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionDescriptor::setY(miopenBackendAttributeType_t attributeType,
                                                 int64_t elementCount,
                                                 void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
    {
        mBaseYDescriptor = deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
        auto& baseDescr  = deref(mBaseYDescriptor);
        auto& tensDescr  = dynamic_cast<BackendTensorDescriptor&>(baseDescr);
        if(mBuilder)
        {
            mBuilder->setY(tensDescr.tensorDescriptor());
        }
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionDescriptor::setAlpha(miopenBackendAttributeType_t attributeType,
                                                     int64_t elementCount,
                                                     void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_DOUBLE && elementCount > 0)
    {
        if(mBuilder)
        {
            mBuilder->setAlpha(deref(static_cast<double*>(arrayOfElements)));
        }
    }
    else if(attributeType == MIOPEN_TYPE_FLOAT && elementCount > 0)
    {
        if(mBuilder)
        {
            mBuilder->setAlpha(deref(static_cast<float*>(arrayOfElements)));
        }
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionDescriptor::setBeta(miopenBackendAttributeType_t attributeType,
                                                    int64_t elementCount,
                                                    void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_DOUBLE && elementCount > 0)
    {
        if(mBuilder)
        {
            mBuilder->setBeta(deref(static_cast<double*>(arrayOfElements)));
        }
    }
    else if(attributeType == MIOPEN_TYPE_FLOAT && elementCount > 0)
    {
        if(mBuilder)
        {
            mBuilder->setBeta(deref(static_cast<float*>(arrayOfElements)));
        }
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionDescriptor::getConvolution(
    miopenBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t* elementCount,
    void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount > 0)
    {
        *elementCount                                             = 1;
        *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mBaseConvolutionDescriptor;
        return;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionDescriptor::getX(miopenBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount > 0)
    {
        *elementCount                                             = 1;
        *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mBaseXDescriptor;
        return;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionDescriptor::getW(miopenBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount > 0)
    {
        *elementCount                                             = 1;
        *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mBaseWDescriptor;
        return;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionDescriptor::getY(miopenBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount > 0)
    {
        *elementCount                                             = 1;
        *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mBaseYDescriptor;
        return;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionDescriptor::getAlpha(miopenBackendAttributeType_t attributeType,
                                                     int64_t requestedElementCount,
                                                     int64_t* elementCount,
                                                     void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_DOUBLE && requestedElementCount > 0)
    {
        *elementCount                          = 1;
        *static_cast<double*>(arrayOfElements) = mDescriptor->getAlpha();
        return;
    }
    else if(attributeType == MIOPEN_TYPE_FLOAT && requestedElementCount > 0)
    {
        *elementCount                         = 1;
        *static_cast<float*>(arrayOfElements) = mDescriptor->getAlpha();
        return;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionDescriptor::getBeta(miopenBackendAttributeType_t attributeType,
                                                    int64_t requestedElementCount,
                                                    int64_t* elementCount,
                                                    void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_DOUBLE && requestedElementCount > 0)
    {
        *elementCount                          = 1;
        *static_cast<double*>(arrayOfElements) = mDescriptor->getBeta();
        return;
    }
    else if(attributeType == MIOPEN_TYPE_FLOAT && requestedElementCount > 0)
    {
        *elementCount                         = 1;
        *static_cast<float*>(arrayOfElements) = mDescriptor->getBeta();
        return;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

BackendOperationConvolutionForwardDescriptor::~BackendOperationConvolutionForwardDescriptor() {}

void BackendOperationConvolutionForwardDescriptor::setAttribute(
    miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC:
        setConvolution(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X:
        setX(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
        setW(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
        setY(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA:
        setAlpha(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA:
        setBeta(attributeType, elementCount, arrayOfElements);
        return;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionForwardDescriptor::getAttribute(
    miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC:
        getConvolution(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X:
        getX(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
        getW(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
        getY(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA:
        getAlpha(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA:
        getBeta(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

BackendOperationConvolutionBackwardDataDescriptor::
    ~BackendOperationConvolutionBackwardDataDescriptor()
{
}

void BackendOperationConvolutionBackwardDataDescriptor::setAttribute(
    miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC:
        setConvolution(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX:
        setX(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W:
        setW(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY:
        setY(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA:
        setAlpha(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA:
        setBeta(attributeType, elementCount, arrayOfElements);
        return;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionBackwardDataDescriptor::getAttribute(
    miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC:
        getConvolution(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX:
        getX(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W:
        getW(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY:
        getY(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA:
        getAlpha(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA:
        getBeta(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

BackendOperationConvolutionBackwardFilterDescriptor::
    ~BackendOperationConvolutionBackwardFilterDescriptor()
{
}

void BackendOperationConvolutionBackwardFilterDescriptor::setAttribute(
    miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC:
        setConvolution(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X:
        setX(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW:
        setW(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY:
        setY(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA:
        setAlpha(attributeType, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA:
        setBeta(attributeType, elementCount, arrayOfElements);
        return;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionBackwardFilterDescriptor::getAttribute(
    miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC:
        getConvolution(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X:
        getX(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW:
        getW(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY:
        getY(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA:
        getAlpha(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    case MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA:
        getBeta(attributeType, requestedElementCount, elementCount, arrayOfElements);
        return;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // namespace graphapi

} // namespace miopen
