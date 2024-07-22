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
#include <miopen/graphapi/reshape.hpp>

#include <algorithm>

namespace miopen {

namespace graphapi {

const std::string& OperationReshape::signName() const
{
    static const std::string name = "OP_RESHAPE";
    return name;
}

std::vector<Tensor*> OperationReshape::getInTensors() const { return {mX}; }

std::vector<Tensor*> OperationReshape::getOutTensors() const { return {mY}; }

OperationReshapeBuilder& OperationReshapeBuilder::setX(Tensor* x)
{
    mOperationReshape.mX = checkPtr(x);
    return *this;
}

OperationReshapeBuilder& OperationReshapeBuilder::setY(Tensor* y)
{
    mOperationReshape.mY = checkPtr(y);
    return *this;
}

OperationReshape OperationReshapeBuilder::build()
{
    const bool valid = mOperationReshape.mX != nullptr && mOperationReshape.mY != nullptr;

    if(!valid)
        MIOPEN_THROW(miopenStatusBadParm);

    const auto& inputDims     = mOperationReshape.mX->GetLengths();
    const auto& inputStrides  = mOperationReshape.mX->GetStrides();
    const auto& outputDims    = mOperationReshape.mY->GetLengths();
    const auto& outputStrides = mOperationReshape.mY->GetStrides();
    const auto size           = inputDims.size();

    /* Detect a case of transpose operation for the last 2 dimensions:
     * [b, h, s, d] -> [b, h, d, s]
     * which is important for MHA graphs
     */
    const bool transpose =
        outputDims.size() == size && size >= 2 && inputDims[size - 1] == outputDims[size - 2] &&
        inputStrides[size - 1] == outputStrides[size - 2] &&
        inputDims[size - 2] == outputDims[size - 1] &&
        inputStrides[size - 2] == outputStrides[size - 1] &&
        std::equal(inputDims.cbegin(), inputDims.cbegin() + size - 2, outputDims.cbegin()) &&
        std::equal(inputStrides.cbegin(), inputStrides.cbegin() + size - 2, outputStrides.cbegin());

    if(transpose)
    {
        mOperationReshape.mOpKind = OperationReshape::OpKind::TRANSPOSE;
    }
    else
    {
        mOperationReshape.mOpKind = OperationReshape::OpKind::GENERIC;
    }

    return mOperationReshape;
}

void BackendOperationReshapeDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
                                                     miopenBackendAttributeType_t attributeType,
                                                     int64_t elementCount,
                                                     void* arrayOfElements)
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    using Setter = OperationReshapeBuilder& (OperationReshapeBuilder::*)(Tensor * tensor);

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
    case MIOPEN_ATTR_OPERATION_RESHAPE_XDESC:
        callSetter(&OperationReshapeBuilder::setX, mXDescriptor);
        break;

    case MIOPEN_ATTR_OPERATION_RESHAPE_YDESC:
        callSetter(&OperationReshapeBuilder::setY, mYDescriptor);
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    };
}

void BackendOperationReshapeDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mOperationReshape = mBuilder.build();
    mFinalized        = true;
}

void BackendOperationReshapeDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_OPERATION_RESHAPE_XDESC: storeDescriptor(mXDescriptor); break;

    case MIOPEN_ATTR_OPERATION_RESHAPE_YDESC: storeDescriptor(mYDescriptor); break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

OpNode* BackendOperationReshapeDescriptor::getOperation() { return &mOperationReshape; }

} // namespace graphapi

} // namespace miopen
