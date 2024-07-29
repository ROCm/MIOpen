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

const std::string& OperationReduction::signName() const
{
    switch(mReduction->getReductionOperator())
    {
    case MIOPEN_REDUCE_TENSOR_ADD: {
        static const std::string name = "OP_REDUCTION:ADD";
        return name;
    }
    case MIOPEN_REDUCE_TENSOR_MUL: {
        static const std::string name = "OP_REDUCTION:MUL";
        return name;
    }
    case MIOPEN_REDUCE_TENSOR_MIN: {
        static const std::string name = "OP_REDUCTION:MIN";
        return name;
    }
    case MIOPEN_REDUCE_TENSOR_MAX: {
        static const std::string name = "OP_REDUCTION:MAX";
        return name;
    }
    case MIOPEN_REDUCE_TENSOR_AMAX: {
        static const std::string name = "OP_REDUCTION:AMAX";
        return name;
    }
    case MIOPEN_REDUCE_TENSOR_AVG: {
        static const std::string name = "OP_REDUCTION:AVG";
        return name;
    }
    case MIOPEN_REDUCE_TENSOR_NORM1: {
        static const std::string name = "OP_REDUCTION:NORM1";
        return name;
    }
    case MIOPEN_REDUCE_TENSOR_NORM2: {
        static const std::string name = "OP_REDUCTION:NORM2";
        return name;
    }
    default: MIOPEN_THROW(miopenStatusNotImplemented);
    }
}

std::vector<Tensor*> OperationReduction::getInTensors() const { return {mX}; }

std::vector<Tensor*> OperationReduction::getOutTensors() const { return {mY}; }

OperationReductionBuilder& OperationReductionBuilder::setReduction(Reduction* reduction)
{
    mOperationReduction.mReduction = checkPtr(reduction);
    return *this;
}

OperationReductionBuilder& OperationReductionBuilder::setX(Tensor* x)
{
    mOperationReduction.mX = checkPtr(x);
    return *this;
}

OperationReductionBuilder& OperationReductionBuilder::setY(Tensor* y)
{
    mOperationReduction.mY = checkPtr(y);
    return *this;
}

OperationReduction OperationReductionBuilder::build()
{
    if(mOperationReduction.mReduction != nullptr && mOperationReduction.mX != nullptr &&
       mOperationReduction.mY != nullptr &&
       std::equal(mOperationReduction.mX->GetLengths().cbegin(),
                  mOperationReduction.mX->GetLengths().cend(),
                  mOperationReduction.mY->GetLengths().cbegin(),
                  mOperationReduction.mY->GetLengths().cend(),
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

void BackendOperationReductionDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
                                                       miopenBackendAttributeType_t attributeType,
                                                       int64_t elementCount,
                                                       void* arrayOfElements)
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    using Setter = OperationReductionBuilder& (OperationReductionBuilder::*)(Tensor * tensor);

    auto callSetter = [=](Setter setter, miopenBackendDescriptor_t& outDescriptor) {
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
        {
            miopenBackendDescriptor_t apiDescriptor =
                deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
            BackendDescriptor& backendDescriptor = deref(apiDescriptor);

            if(!backendDescriptor.isFinalized())
            {
                MIOPEN_THROW(miopenStatusBadParm);
            }

            BackendTensorDescriptor& tensorDescriptor =
                dynamic_cast<BackendTensorDescriptor&>(backendDescriptor);
            (mBuilder.*setter)(tensorDescriptor.getTensor());
            outDescriptor = apiDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
    };

    switch(attributeName)
    {
    case MIOPEN_ATTR_OPERATION_REDUCTION_DESC:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
        {
            miopenBackendDescriptor_t apiDescriptor =
                deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
            BackendDescriptor& backendDescriptor = deref(apiDescriptor);

            if(!backendDescriptor.isFinalized())
            {
                MIOPEN_THROW(miopenStatusBadParm);
            }

            BackendReductionDescriptor& reductionDescriptor =
                dynamic_cast<BackendReductionDescriptor&>(backendDescriptor);
            mBuilder.setReduction(reductionDescriptor.getReduction());
            mReductionDescriptor = apiDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATION_REDUCTION_XDESC:
        callSetter(&OperationReductionBuilder::setX, mXDescriptor);
        break;

    case MIOPEN_ATTR_OPERATION_REDUCTION_YDESC:
        callSetter(&OperationReductionBuilder::setY, mYDescriptor);
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationReductionDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mOperationReduction = mBuilder.build();
    mFinalized          = true;
}

void BackendOperationReductionDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
                                                       miopenBackendAttributeType_t attributeType,
                                                       int64_t requestedElementCount,
                                                       int64_t* elementCount,
                                                       void* arrayOfElements)
{
    if(!mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    auto storeDescriptor = [=](miopenBackendDescriptor_t descriptor) {
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = descriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
    };

    switch(attributeName)
    {
    case MIOPEN_ATTR_OPERATION_REDUCTION_DESC: storeDescriptor(mReductionDescriptor); break;

    case MIOPEN_ATTR_OPERATION_REDUCTION_XDESC: storeDescriptor(mXDescriptor); break;

    case MIOPEN_ATTR_OPERATION_REDUCTION_YDESC: storeDescriptor(mYDescriptor); break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

OpNode* BackendOperationReductionDescriptor::getOperation() { return &mOperationReduction; }

} // namespace graphapi

} // namespace miopen
