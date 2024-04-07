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
#include <miopen/graphapi/reduction.hpp>

#include <algorithm>

namespace miopen {

namespace graphapi {

Reduction ReductionBuilder::build()
{
    if(mReductionOperatorSet && mCompTypeSet)
    {
        return mReduction;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendReductionDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_REDUCTION_OPERATOR:
        if(attributeType == MIOPEN_TYPE_REDUCTION_OPERATOR_TYPE && elementCount == 1)
        {
            mBuilder.setReductionOperator(*static_cast<miopenReduceTensorOp_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_REDUCTION_COMP_TYPE:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && elementCount == 1)
        {
            mBuilder.setCompType(*static_cast<miopenDataType_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendReductionDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mReduction = mBuilder.build();
    mFinalized = true;
}

void BackendReductionDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_REDUCTION_OPERATOR:
        if(attributeType == MIOPEN_TYPE_REDUCTION_OPERATOR_TYPE && requestedElementCount == 1)
        {
            *elementCount = 1;
            *static_cast<miopenReduceTensorOp_t*>(arrayOfElements) =
                mReduction.getReductionOperator();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_REDUCTION_COMP_TYPE:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && requestedElementCount == 1)
        {
            *elementCount                                    = 1;
            *static_cast<miopenDataType_t*>(arrayOfElements) = mReduction.getCompType();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

std::vector<Tensor*> OperationReduction::getInTensors() const { return {mX}; }

std::vector<Tensor*> OperationReduction::getOutTensors() const { return {mY}; }

namespace {

template <typename Ptr>
void assignPtr(Ptr src, Ptr& dst)
{
    if(src != nullptr)
    {
        dst = src;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // namespace

OperationReductionBuilder& OperationReductionBuilder::setReduction(Reduction* reduction)
{
    assignPtr(reduction, mOperationReduction.mReduction);
    return *this;
}

OperationReductionBuilder& OperationReductionBuilder::setX(Tensor* x)
{
    assignPtr(x, mOperationReduction.mX);
    return *this;
}

OperationReductionBuilder& OperationReductionBuilder::setY(Tensor* y)
{
    assignPtr(y, mOperationReduction.mY);
    return *this;
}

OperationReduction OperationReductionBuilder::build()
{
    if(mOperationReduction.mReduction != nullptr && mOperationReduction.mX != nullptr &&
       mOperationReduction.mY != nullptr &&
       std::equal(mOperationReduction.mX->getDimensions().cbegin(),
                  mOperationReduction.mX->getDimensions().cend(),
                  mOperationReduction.mY->getDimensions().cbegin(),
                  mOperationReduction.mY->getDimensions().cend(),
                  [](auto inputDim, auto outputDim) {
                      return outputDim == 1 || outputDim == inputDim || outputDim % inputDim == 0;
                  }))
    {
        return mOperationReduction;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // namespace graphapi

} // namespace miopen
