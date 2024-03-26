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

#include <miopen/errors.hpp>
#include <miopen/graphapi/pointwise.hpp>

namespace miopen {

namespace graphapi {

void BackendPointwiseDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_POINTWISE_MODE:
        if(attributeType == MIOPEN_TYPE_POINTWISE_MODE && elementCount == 1)
        {
            mBuilder.setMode(*static_cast<miopenPointwiseMode_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_MATH_PREC:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && elementCount == 1)
        {
            mBuilder.setMathPrecision(*static_cast<miopenDataType_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_NAN_PROPAGATION:
        if(attributeType == MIOPEN_TYPE_NAN_PROPOGATION && elementCount == 1)
        {
            mBuilder.setNanPropagation(*static_cast<miopenNanPropagation_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP:
        setFloatOrDouble(
            &PointwiseBuilder::setReluLowerClip, attributeType, elementCount, arrayOfElements);
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP:
        setFloatOrDouble(
            &PointwiseBuilder::setReluUpperClip, attributeType, elementCount, arrayOfElements);
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE:
        setFloatOrDouble(
            &PointwiseBuilder::setReluLowerClipSlope, attributeType, elementCount, arrayOfElements);
        break;

    case MIOPEN_ATTR_POINTWISE_ELU_ALPHA:
        setFloatOrDouble(
            &PointwiseBuilder::setEluAlpha, attributeType, elementCount, arrayOfElements);
        break;

    case MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA:
        setFloatOrDouble(
            &PointwiseBuilder::setSoftPlusBeta, attributeType, elementCount, arrayOfElements);
        break;

    case MIOPEN_ATTR_POINTWISE_AXIS:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            mBuilder.setAxis(*static_cast<int64_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendPointwiseDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mPointwise = mBuilder.build();
    mFinalized = true;
}

void BackendPointwiseDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_POINTWISE_MODE:
        if(attributeType == MIOPEN_TYPE_POINTWISE_MODE && requestedElementCount == 1)
        {
            *elementCount                                         = 1;
            *static_cast<miopenPointwiseMode_t*>(arrayOfElements) = mPointwise.getMode();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_MATH_PREC:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && requestedElementCount == 1)
        {
            *elementCount                                    = 1;
            *static_cast<miopenDataType_t*>(arrayOfElements) = mPointwise.getMathPrecision();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_NAN_PROPAGATION:
        if(attributeType == MIOPEN_TYPE_NAN_PROPOGATION && requestedElementCount == 1)
        {
            *elementCount                                          = 1;
            *static_cast<miopenNanPropagation_t*>(arrayOfElements) = mPointwise.getNanPropagation();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP:
        getFloatOrDouble(&Pointwise::getReluLowerClip,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements);
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP:
        getFloatOrDouble(&Pointwise::getReluUpperClip,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements);
        break;

    case MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE:
        getFloatOrDouble(&Pointwise::getReluLowerClipSlope,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements);
        break;

    case MIOPEN_ATTR_POINTWISE_ELU_ALPHA:
        getFloatOrDouble(&Pointwise::getEluAlpha,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements);
        break;

    case MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA:
        getFloatOrDouble(&Pointwise::getSoftPlusBeta,
                         attributeType,
                         requestedElementCount,
                         elementCount,
                         arrayOfElements);
        break;

    case MIOPEN_ATTR_POINTWISE_AXIS:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int64_t*>(arrayOfElements) = mPointwise.getAxis();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // namespace graphapi

} // namespace miopen
