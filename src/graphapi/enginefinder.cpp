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
#include <miopen/graphapi/enginefinder.hpp>

#include <algorithm>

void miopen::graphapi::BackendOperationGraphDescriptor::setAttribute(
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
    case MIOPEN_ATTR_OPERATIONGRAPH_HANDLE:
        if(attributeType == MIOPEN_TYPE_HANDLE && elementCount == 1)
        {
            mHandle = *static_cast<miopenHandle_t*>(arrayOfElements);
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATIONGRAPH_OPS:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount > 0)
        {
            std::vector<miopenBackendDescriptor_t> ops;
            ops.reserve(elementCount);

            OpGraphBuilder builder;

            std::for_each_n(static_cast<miopenBackendDescriptor_t*>(arrayOfElements),
                            elementCount,
                            [&ops, &builder](miopenBackendDescriptor_t apiDescriptor) {
                                BackendDescriptor& backendDescriptor = deref(apiDescriptor);
                                if(backendDescriptor.isFinalized())
                                {
                                    OpNode* node = backendDescriptor.getOperation();
                                    if(node != nullptr && !builder.hasNode(node))
                                    {
                                        builder.addNode(node);
                                        ops.push_back(apiDescriptor);
                                        return;
                                    }
                                }
                                MIOPEN_THROW(miopenStatusBadParm);
                            });

            mOps            = std::move(ops);
            mOpGraphBuilder = std::move(builder);
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void miopen::graphapi::BackendOperationGraphDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    if(mHandle == nullptr || mOps.empty())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    mOpGraph = std::move(mOpGraphBuilder).build();

    // TODO: Find solutions for the operation graph

    mFinalized = true;
}

void miopen::graphapi::BackendOperationGraphDescriptor::getAttribute(
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
    case MIOPEN_ATTR_OPERATIONGRAPH_HANDLE:
        if(attributeType == MIOPEN_TYPE_HANDLE && requestedElementCount == 1)
        {
            *elementCount                                  = 1;
            *static_cast<miopenHandle_t*>(arrayOfElements) = mHandle;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATIONGRAPH_OPS:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount >= 0)
        {
            *elementCount = mOps.size();
            std::copy_n(mOps.cbegin(),
                        std::min(*elementCount, requestedElementCount),
                        static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int64_t*>(arrayOfElements) = mSolutions.size();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}
