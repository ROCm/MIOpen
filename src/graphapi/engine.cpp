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
#include <miopen/graphapi/engine.hpp>
#include <miopen/graphapi/opgraph.hpp>

namespace miopen {

namespace graphapi {

EngineBuilder& EngineBuilder::setOpGraph(const OpGraph* opGraph)
{
    mOpGraph = checkPtr(opGraph);
    return *this;
}

EngineBuilder& EngineBuilder::setGlobalIndex(int64_t globalIndex)
{
    if(globalIndex >= 0)
    {
        mGlobalIndex = globalIndex;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    return *this;
}

EngineBuilder& EngineBuilder::setSmCount(int32_t smCount)
{
    if(smCount >= 0)
    {
        mSmCount = smCount;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    return *this;
}

Engine EngineBuilder::build()
{
    if(mOpGraph != nullptr && mGlobalIndexSet && mGlobalIndex < mOpGraph->getEngines().size())
    {
        // TODO: validate mSmCount
        Engine engine       = mOpGraph->getEngines()[mGlobalIndex];
        engine.mGlobalIndex = mGlobalIndex;
        engine.mSmCount     = mSmCount;
        return engine;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendEngineDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_ENGINE_OPERATION_GRAPH:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
        {
            miopenBackendDescriptor_t& apiDescriptor =
                deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
            BackendDescriptor& backendDescriptor = deref(apiDescriptor);

            if(!backendDescriptor.isFinalized())
            {
                MIOPEN_THROW(miopenStatusBadParm);
            }

            BackendOperationGraphDescriptor& operationGraphDescriptor =
                dynamic_cast<BackendOperationGraphDescriptor&>(backendDescriptor);
            mBuilder.setOpGraph(operationGraphDescriptor.getOperationGraph());
            mOpGraphDescriptor = apiDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINE_GLOBAL_INDEX:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount == 1)
        {
            mBuilder.setGlobalIndex(*static_cast<int64_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINE_SM_COUNT_TARGET:
        if(attributeType == MIOPEN_TYPE_INT32 && elementCount == 1)
        {
            mBuilder.setSmCount(*static_cast<int32_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendEngineDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }
}

void BackendEngineDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_ENGINE_OPERATION_GRAPH:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mOpGraphDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINE_GLOBAL_INDEX:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int64_t*>(arrayOfElements) = mEngine.getGlobalIndex();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINE_SM_COUNT_TARGET:
        if(attributeType == MIOPEN_TYPE_INT32 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int32_t*>(arrayOfElements) = mEngine.getSmCount();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINE_BEHAVIOR_NOTE:
    case MIOPEN_ATTR_ENGINE_KNOB_INFO:
    case MIOPEN_ATTR_ENGINE_LAYOUT_INFO:
    case MIOPEN_ATTR_ENGINE_NUMERICAL_NOTE:
        // TODO: figure out what we can return here
        *elementCount = 0;
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // namespace graphapi

} // namespace miopen
