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
#include <miopen/graphapi/matmul.hpp>

namespace miopen {
namespace graphapi {

Matmul MatmulBuilder::build() const
{
    if(!mComputeTypeSet)
        MIOPEN_THROW(miopenStatusBadParm);
    return mMatmul;
}

void BackendMatmulDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_MATMUL_COMP_TYPE:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && elementCount == 1)
        {
            mBuilder.setComputeType(*static_cast<miopenDataType_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendMatmulDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mMatmul    = mBuilder.build();
    mFinalized = true;
}

void BackendMatmulDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_MATMUL_COMP_TYPE:
        if(attributeType == MIOPEN_TYPE_DATA_TYPE && requestedElementCount == 1)
        {
            *elementCount                                    = 1;
            *static_cast<miopenDataType_t*>(arrayOfElements) = mMatmul.getComputeType();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

OperationMatmul OperationMatmulBuilder::build()
{
    if(!aSet || !bSet || !cSet || !matmulSet)
        MIOPEN_THROW(miopenStatusBadParm);

    if(!(mOperationMatmul.mA->getDimensions().size() == mOperationMatmul.mB->getDimensions().size() == mOperationMatmul.mC->getDimensions().size()))
        MIOPEN_THROW(miopenStatusBadParm);
    int Am = mOperationMatmul.mA->getDimensions().end()[-2];
    int An = mOperationMatmul.mA->getDimensions().end()[-1];

    int Bn = mOperationMatmul.mB->getDimensions().end()[-2];
    int Bk = mOperationMatmul.mB->getDimensions().end()[-1];

    int Cm = mOperationMatmul.mB->getDimensions().end()[-2];
    int Ck = mOperationMatmul.mB->getDimensions().end()[-1];
    
    if(Am != Cm || An != Bn || Bk != Ck)
        MIOPEN_THROW(miopenStatusBadParm);

    // TODO: need broadcsat checks
    //

    return mOperationMatmul;
}

void BackendOperationMatmulDescriptor::finalize()
{
    if(mFinalized || !deref(mA).isFinalized() || !deref(mB).isFinalized() ||
       !deref(mC).isFinalized() || !deref(mGemmMOverride).isFinalized() ||
       !deref(mGemmNOverride).isFinalized() || !deref(mGemmKOverride).isFinalized())
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mMatmul    = mBuilder.build();
    mFinalized = true;
}

void BackendOperationMatmulDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_OPERATION_MATMUL_ADESC:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mA;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_BDESC:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mB;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_CDESC:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mC;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_DESC:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mMatmuDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mGemmMOverride;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mGemmNOverride;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mGemmKOverride;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int64_t*>(arrayOfElements) = mMatmul.getBatchCount();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationMatmulDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
                                                    miopenBackendAttributeType_t attributeType,
                                                    int64_t elementCount,
                                                    void* arrayOfElements)
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    using TensorSetter = OperationMatmulBuilder& (OperationMatmulBuilder::*)(Tensor*);

    auto callTensorSetter = [=](TensorSetter setter, miopenBackendDescriptor_t& outApiDescriptor) {
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
            outApiDescriptor = apiDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
    };

    switch(attributeName)
    {
    case MIOPEN_ATTR_OPERATION_MATMUL_ADESC:
        callTensorSetter(&OperationMatmulBuilder::setA, mA);
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_BDESC:
        callTensorSetter(&OperationMatmulBuilder::setB, mB);
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_CDESC:
        callTensorSetter(&OperationMatmulBuilder::setC, mC);
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_DESC:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
        {
            miopenBackendDescriptor_t apiDescriptor =
                deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
            BackendDescriptor& backendDescriptor = deref(apiDescriptor);

            if(!backendDescriptor.isFinalized())
            {
                MIOPEN_THROW(miopenStatusBadParm);
            }
            BackendMatmulDescriptor& matmulDescriptor =
                dynamic_cast<BackendMatmulDescriptor&>(backendDescriptor);
            mBuilder.setMatmulDescriptor(matmulDescriptor.getMatmul());
            mMatmuDescriptor = apiDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC:
        callTensorSetter(&OperationMatmulBuilder::setGemmMOverride, mGemmMOverride);
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC:
        callTensorSetter(&OperationMatmulBuilder::setGemmNOverride, mGemmNOverride);
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC:
        callTensorSetter(&OperationMatmulBuilder::setGemmKOverride, mGemmKOverride);
        break;

    case MIOPEN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            mBuilder.setBatchCount(*static_cast<int64_t*>(arrayOfElements));
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
