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
#include <miopen/graphapi/convolution.hpp>
#include <miopen/graphapi/graphapi.hpp>
#include <miopen/graphapi/tensor.hpp>
#include <miopen/logger.hpp>

#include <memory>

extern "C" miopenStatus_t
miopenBackendCreateDescriptor(miopenBackendDescriptorType_t descriptorType,
                              miopenBackendDescriptor_t* descriptor)
{
    MIOPEN_LOG_FUNCTION(descriptorType, descriptor);
    return miopen::try_([&] {
        auto& outputDesciptor = miopen::deref(descriptor);

        switch(descriptorType)
        {
        case MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR:
            outputDesciptor = new miopen::graphapi::BackendConvolutionDescriptor();
            break;
        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            outputDesciptor = new miopen::graphapi::BackendOperationConvolutionForwardDescriptor();
            break;
        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            outputDesciptor =
                new miopen::graphapi::BackendOperationConvolutionBackwardFilterDescriptor();
            break;
        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            outputDesciptor =
                new miopen::graphapi::BackendOperationConvolutionBackwardDataDescriptor();
            break;
        case MIOPEN_BACKEND_TENSOR_DESCRIPTOR:
            outputDesciptor = new miopen::graphapi::BackendTensorDescriptor();
            break;
        default: MIOPEN_THROW(miopenStatus_t::miopenStatusUnsupportedOp);
        }
    });
}

extern "C" miopenStatus_t miopenBackendSetAttribute(miopenBackendDescriptor_t descriptor,
                                                    miopenBackendAttributeName_t attributeName,
                                                    miopenBackendAttributeType_t attributeType,
                                                    int64_t elementCount,
                                                    void* arrayOfElements)
{
    MIOPEN_LOG_FUNCTION(attributeName, attributeType, elementCount);

    if(arrayOfElements == nullptr)
    {
        return miopenStatusBadParm;
    }

    return miopen::try_(
        [&] {
            auto& theDescriptor = miopen::deref(descriptor);
            theDescriptor.setAttribute(attributeName, attributeType, elementCount, arrayOfElements);
        },
        false);
}

extern "C" miopenStatus_t miopenBackendFinalize(miopenBackendDescriptor_t descriptor)
{
    return miopen::try_(
        [&] {
            auto& theDescriptor = miopen::deref(descriptor);
            theDescriptor.finalize();
        },
        false);
}

extern "C" miopenStatus_t miopenBackendGetAttribute(miopenBackendDescriptor_t descriptor,
                                                    miopenBackendAttributeName_t attributeName,
                                                    miopenBackendAttributeType_t attributeType,
                                                    int64_t requestedElementCount,
                                                    int64_t* elementCount,
                                                    void* arrayOfElements)
{
    if(elementCount == nullptr || arrayOfElements == nullptr)
    {
        return miopenStatusBadParm;
    }

    return miopen::try_(
        [&] {
            auto& theDescriptor = miopen::deref(descriptor);
            theDescriptor.getAttribute(
                attributeName, attributeType, requestedElementCount, elementCount, arrayOfElements);
        },
        false);
}

extern "C" miopenStatus_t miopenBackendExecute(miopenHandle_t handle,
                                               miopenBackendDescriptor_t executionPlan,
                                               miopenBackendDescriptor_t variantPack)
{
    return miopen::try_(
        [&] {
            auto& theDescriptor = miopen::deref(executionPlan);
            theDescriptor.execute(handle, variantPack);
        },
        false);
}

extern "C" miopenStatus_t miopenBackendDestroyDescriptor(miopenBackendDescriptor_t descriptor)
{
    return miopen::try_([&] { miopen_destroy_object(descriptor); }, false);
}

extern "C" miopenStatus_t miopenBackendInitialize(miopenBackendDescriptor_t descriptor,
                                                  miopenBackendDescriptorType_t descriptorType,
                                                  size_t sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(descriptorType, sizeInBytes);

    if(descriptor == nullptr)
    {
        return miopenStatusBadParm;
    }

    return miopen::try_([&] {
        void* address = descriptor;

        switch(descriptorType)
        {
        case MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR:
            if(std::align(alignof(miopen::graphapi::BackendConvolutionDescriptor),
                          sizeof(miopen::graphapi::BackendConvolutionDescriptor),
                          address,
                          sizeInBytes) != nullptr &&
               address == descriptor)
            {
                new(descriptor) miopen::graphapi::BackendConvolutionDescriptor();
            }
            else
            {
                MIOPEN_THROW(miopenStatus_t::miopenStatusUnsupportedOp);
            }
            break;
        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            if(std::align(alignof(miopen::graphapi::BackendOperationConvolutionForwardDescriptor),
                          sizeof(miopen::graphapi::BackendOperationConvolutionForwardDescriptor),
                          address,
                          sizeInBytes) != nullptr &&
               address == descriptor)
            {
                new(descriptor) miopen::graphapi::BackendOperationConvolutionForwardDescriptor();
            }
            else
            {
                MIOPEN_THROW(miopenStatus_t::miopenStatusUnsupportedOp);
            }
            break;
        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
            if(std::align(
                   alignof(miopen::graphapi::BackendOperationConvolutionBackwardFilterDescriptor),
                   sizeof(miopen::graphapi::BackendOperationConvolutionBackwardFilterDescriptor),
                   address,
                   sizeInBytes) != nullptr &&
               address == descriptor)
            {
                new(descriptor)
                    miopen::graphapi::BackendOperationConvolutionBackwardFilterDescriptor();
            }
            else
            {
                MIOPEN_THROW(miopenStatus_t::miopenStatusUnsupportedOp);
            }
            break;
        case MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
            if(std::align(
                   alignof(miopen::graphapi::BackendOperationConvolutionBackwardDataDescriptor),
                   sizeof(miopen::graphapi::BackendOperationConvolutionBackwardDataDescriptor),
                   address,
                   sizeInBytes) != nullptr &&
               address == descriptor)
            {
                new(descriptor)
                    miopen::graphapi::BackendOperationConvolutionBackwardDataDescriptor();
            }
            else
            {
                MIOPEN_THROW(miopenStatus_t::miopenStatusUnsupportedOp);
            }
            break;
        case MIOPEN_BACKEND_TENSOR_DESCRIPTOR:
            if(std::align(alignof(miopen::graphapi::BackendTensorDescriptor),
                          sizeof(miopen::graphapi::BackendTensorDescriptor),
                          address,
                          sizeInBytes) != nullptr &&
               address == descriptor)
            {
                new(descriptor) miopen::graphapi::BackendTensorDescriptor();
            }
            else
            {
                MIOPEN_THROW(miopenStatus_t::miopenStatusUnsupportedOp);
            }
            break;
        default: MIOPEN_THROW(miopenStatusUnsupportedOp);
        }
    });
}

namespace miopen {

namespace graphapi {

BackendDescriptor::~BackendDescriptor() {}

void BackendDescriptor::execute([[maybe_unused]] miopenHandle_t handle,
                                [[maybe_unused]] miopenBackendDescriptor_t variantPack)
{
    MIOPEN_THROW(miopenStatusBadParm);
}

OpNode* BackendDescriptor::getOperation() { return nullptr; }

} // namespace graphapi

} // namespace miopen
