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

// WORKAROUND: building on Windows is failing due to conflicting definitions
// of std::min() between the MSVC standard library and HIP Clang wrappers.
// this avoids including cuda_wrappers version of <algorithm>
#define __CLANG_CUDA_WRAPPERS_ALGORITHM
// end of workaround.

#include <miopen/algorithm.hpp>
#include <miopen/graphapi/graphapi_convolution.hpp>
#include <miopen/errors.hpp>

namespace miopen {

namespace graphapi {

ConvolutionBuilder& ConvolutionBuilder::setCompType(miopenDataType_t compType) & noexcept
{
    mCompType    = compType;
    mCompTypeSet = true;
    return *this;
}
ConvolutionBuilder& ConvolutionBuilder::setMode(miopenConvolutionMode_t mode) & noexcept
{
    mMode    = mode;
    mModeSet = true;
    return *this;
}
ConvolutionBuilder& ConvolutionBuilder::setSpatialDims(int64_t spatialDims) & noexcept
{
    mSpatialDims    = spatialDims;
    mSpatialDimsSet = true;
    return *this;
}
ConvolutionBuilder& ConvolutionBuilder::setDilations(const std::vector<int64_t>& dilations) &
{
    mDilations    = dilations;
    mDilationsSet = true;
    return *this;
}
ConvolutionBuilder& ConvolutionBuilder::setDilations(std::vector<int64_t>&& dilations) & noexcept
{
    mDilations    = std::move(dilations);
    mDilationsSet = true;
    return *this;
}
ConvolutionBuilder&
ConvolutionBuilder::setFilterStrides(const std::vector<int64_t>& filterStrides) &
{
    mFilterStrides    = filterStrides;
    mFilterStridesSet = true;
    return *this;
}
ConvolutionBuilder&
ConvolutionBuilder::setFilterStrides(std::vector<int64_t>&& filterStrides) & noexcept
{
    mFilterStrides    = std::move(filterStrides);
    mFilterStridesSet = true;
    return *this;
}
ConvolutionBuilder& ConvolutionBuilder::setPrePaddings(const std::vector<int64_t>& prePaddings) &
{
    mPrePaddings    = prePaddings;
    mPrePaddingsSet = true;
    return *this;
}
ConvolutionBuilder&
ConvolutionBuilder::setPrePaddings(std::vector<int64_t>&& prePaddings) & noexcept
{
    mPrePaddings    = std::move(prePaddings);
    mPrePaddingsSet = true;
    return *this;
}
ConvolutionBuilder& ConvolutionBuilder::setPostPaddings(const std::vector<int64_t>& postPaddings) &
{
    mPostPaddings    = postPaddings;
    mPostPaddingsSet = true;
    return *this;
}
ConvolutionBuilder&
ConvolutionBuilder::setPostPaddings(std::vector<int64_t>&& postPaddings) & noexcept
{
    mPostPaddings    = std::move(postPaddings);
    mPostPaddingsSet = true;
    return *this;
}

bool ConvolutionBuilder::validate() const
{
    return mCompTypeSet && mModeSet && mSpatialDimsSet && mDilationsSet && mFilterStridesSet &&
           mPrePaddingsSet && mPostPaddingsSet && mSpatialDims >= 1 &&
           mDilations.size() == mSpatialDims && mFilterStrides.size() == mSpatialDims &&
           mPrePaddings.size() == mSpatialDims && mPostPaddings.size() == mSpatialDims &&
           miopen::all_of(mDilations, [](auto value) { return value > 0; }) &&
           miopen::all_of(mFilterStrides, [](auto value) { return value > 0; }) &&
           miopen::all_of(mPrePaddings, [](auto value) { return value >= 0; }) &&
           miopen::all_of(mPostPaddings, [](auto value) { return value >= 0; });
}

Convolution ConvolutionBuilder::build() const&
{
    if(!validate())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    return {
        mCompType, mMode, mSpatialDims, mPrePaddings, mFilterStrides, mDilations, mPostPaddings};
}

Convolution ConvolutionBuilder::build() &&
{
    if(!validate())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    return {mCompType,
            mMode,
            mSpatialDims,
            std::move(mPrePaddings),
            std::move(mFilterStrides),
            std::move(mDilations),
            std::move(mPostPaddings)};
}

void BackendConvolutionDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
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
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mConvolution = std::move(mBuilder).build();
    mFinalized   = true;
}

void BackendConvolutionDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
        mBuilder.setCompType(*static_cast<miopenDataType_t*>(arrayOfElements));
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
        mBuilder.setMode(*static_cast<miopenConvolutionMode_t*>(arrayOfElements));
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
        mBuilder.setSpatialDims(*static_cast<int64_t*>(arrayOfElements));
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
        mBuilder.setDilations({static_cast<int64_t*>(arrayOfElements),
                               static_cast<int64_t*>(arrayOfElements) + elementCount});
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
        mBuilder.setFilterStrides({static_cast<int64_t*>(arrayOfElements),
                                   static_cast<int64_t*>(arrayOfElements) + elementCount});
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
        mBuilder.setPrePaddings({static_cast<int64_t*>(arrayOfElements),
                                 static_cast<int64_t*>(arrayOfElements) + elementCount});
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
        mBuilder.setPostPaddings({static_cast<int64_t*>(arrayOfElements),
                                  static_cast<int64_t*>(arrayOfElements) + elementCount});
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
        *static_cast<miopenDataType_t*>(arrayOfElements) = mConvolution.getCompType();
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
        *static_cast<miopenConvolutionMode_t*>(arrayOfElements) = mConvolution.getMode();
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
        *static_cast<int64_t*>(arrayOfElements) = mConvolution.getSpatialDims();
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
    if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount > 0)
    {
        const auto& dilations = mConvolution.getDilations();
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
    if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount > 0)
    {
        const auto& strides = mConvolution.getFilterStrides();
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
    if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount > 0)
    {
        const auto& pads = mConvolution.getPrePaddings();
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
    if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount > 0)
    {
        const auto& postPads = mConvolution.getPostPaddings();
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

void BackendOperationConvolutionDescriptor::setConvolution(
    miopenBackendAttributeType_t attributeType, int64_t elementCount, void* arrayOfElements)
{
    if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
    {
        mConvolutionDescriptor = deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
        auto& baseDescr        = deref(mConvolutionDescriptor);
        auto& convDescr        = dynamic_cast<BackendConvolutionDescriptor&>(baseDescr);
        getBuilder().setConvolution(convDescr.getConvolution());
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
        mXDescriptor    = deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
        auto& baseDescr = deref(mXDescriptor);
        auto& tensDescr = dynamic_cast<BackendTensorDescriptor&>(baseDescr);
        getBuilder().setX(tensDescr.getTensor());
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
        mWDescriptor    = deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
        auto& baseDescr = deref(mWDescriptor);
        auto& tensDescr = dynamic_cast<BackendTensorDescriptor&>(baseDescr);
        getBuilder().setW(tensDescr.getTensor());
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
        mYDescriptor    = deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
        auto& baseDescr = deref(mYDescriptor);
        auto& tensDescr = dynamic_cast<BackendTensorDescriptor&>(baseDescr);
        getBuilder().setY(tensDescr.getTensor());
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
        getBuilder().setAlpha(deref(static_cast<double*>(arrayOfElements)));
    }
    else if(attributeType == MIOPEN_TYPE_FLOAT && elementCount > 0)
    {
        getBuilder().setAlpha(deref(static_cast<float*>(arrayOfElements)));
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
        getBuilder().setBeta(deref(static_cast<double*>(arrayOfElements)));
    }
    else if(attributeType == MIOPEN_TYPE_FLOAT && elementCount > 0)
    {
        getBuilder().setBeta(deref(static_cast<float*>(arrayOfElements)));
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
        *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mConvolutionDescriptor;
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
        *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mXDescriptor;
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
        *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mWDescriptor;
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
        *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mYDescriptor;
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
        *static_cast<double*>(arrayOfElements) = getOperationConvolution().getAlpha();
        return;
    }
    else if(attributeType == MIOPEN_TYPE_FLOAT && requestedElementCount > 0)
    {
        *elementCount                         = 1;
        *static_cast<float*>(arrayOfElements) = getOperationConvolution().getAlpha();
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
        *static_cast<double*>(arrayOfElements) = getOperationConvolution().getBeta();
        return;
    }
    else if(attributeType == MIOPEN_TYPE_FLOAT && requestedElementCount > 0)
    {
        *elementCount                         = 1;
        *static_cast<float*>(arrayOfElements) = getOperationConvolution().getBeta();
        return;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationConvolutionForwardDescriptor::setAttribute(
    miopenBackendAttributeName_t attributeName,
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

void BackendOperationConvolutionForwardDescriptor::finalize()
{
    if(mFinalized || !deref(mConvolutionDescriptor).getFinalized() ||
       !deref(mXDescriptor).getFinalized() || !deref(mYDescriptor).getFinalized() ||
       !deref(mWDescriptor).getFinalized())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    mOperation = mBuilder.build();
    mFinalized = true;
}

void BackendOperationConvolutionForwardDescriptor::getAttribute(
    miopenBackendAttributeName_t attributeName,
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

Operation* BackendOperationConvolutionForwardDescriptor::getOperation()
{
    if(!mFinalized)
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }
    return &mOperation;
}

OperationConvolutionBuilder& BackendOperationConvolutionForwardDescriptor::getBuilder()
{
    return mBuilder;
}

OperationConvolution& BackendOperationConvolutionForwardDescriptor::getOperationConvolution()
{
    return mOperation;
}

void BackendOperationConvolutionBackwardDataDescriptor::setAttribute(
    miopenBackendAttributeName_t attributeName,
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

void BackendOperationConvolutionBackwardDataDescriptor::finalize()
{
    if(mFinalized || !deref(mConvolutionDescriptor).getFinalized() ||
       !deref(mXDescriptor).getFinalized() || !deref(mYDescriptor).getFinalized() ||
       !deref(mWDescriptor).getFinalized())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    mOperation = mBuilder.build();
    mFinalized = true;
}

void BackendOperationConvolutionBackwardDataDescriptor::getAttribute(
    miopenBackendAttributeName_t attributeName,
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

Operation* BackendOperationConvolutionBackwardDataDescriptor::getOperation()
{
    if(!mFinalized)
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }
    return &mOperation;
}

OperationConvolutionBuilder& BackendOperationConvolutionBackwardDataDescriptor::getBuilder()
{
    return mBuilder;
}

OperationConvolution& BackendOperationConvolutionBackwardDataDescriptor::getOperationConvolution()
{
    return mOperation;
}

void BackendOperationConvolutionBackwardFilterDescriptor::setAttribute(
    miopenBackendAttributeName_t attributeName,
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

void BackendOperationConvolutionBackwardFilterDescriptor::finalize()
{
    if(mFinalized || !deref(mConvolutionDescriptor).getFinalized() ||
       !deref(mXDescriptor).getFinalized() || !deref(mYDescriptor).getFinalized() ||
       !deref(mWDescriptor).getFinalized())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    mOperation = mBuilder.build();
    mFinalized = true;
}

void BackendOperationConvolutionBackwardFilterDescriptor::getAttribute(
    miopenBackendAttributeName_t attributeName,
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

Operation* BackendOperationConvolutionBackwardFilterDescriptor::getOperation()
{
    if(!mFinalized)
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }
    return &mOperation;
}

OperationConvolutionBuilder& BackendOperationConvolutionBackwardFilterDescriptor::getBuilder()
{
    return mBuilder;
}

OperationConvolution& BackendOperationConvolutionBackwardFilterDescriptor::getOperationConvolution()
{
    return mOperation;
}

} // namespace graphapi

} // namespace miopen
