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
#include <miopen/graphapi/enginecfg.hpp>

namespace miopen {

namespace graphapi {

EngineCfg EngineCfgBuilder::build() &
{
    if(mEngineSet)
    {
        return mEngineCfg;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

EngineCfg EngineCfgBuilder::build() &&
{
    if(mEngineSet)
    {
        return std::move(mEngineCfg);
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendEngineCfgDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_ENGINECFG_ENGINE:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
        {
            miopenBackendDescriptor_t apiDescriptor =
                deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
            BackendDescriptor& backendDescriptor = deref(apiDescriptor);

            if(!backendDescriptor.isFinalized())
            {
                MIOPEN_THROW(miopenStatusBadParm);
            }

            BackendEngineDescriptor& engineDescriptor =
                dynamic_cast<BackendEngineDescriptor&>(backendDescriptor);
            mBuilder.setEngine(engineDescriptor.getEngine());
            mEngineDescriptor = apiDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINECFG_KNOB_CHOICES:
        if(attributeType != MIOPEN_TYPE_BACKEND_DESCRIPTOR || elementCount < 0)
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendEngineCfgDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }
    mEngineCfg = mBuilder.build();
    mFinalized = true;
}

void BackendEngineCfgDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_ENGINECFG_ENGINE:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mEngineDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINECFG_KNOB_CHOICES:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR || requestedElementCount >= 0)
        {
            *elementCount = 0;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINECFG_INTERMEDIATE_INFO: MIOPEN_THROW(miopenStatusUnsupportedOp);

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // namespace graphapi

} // namespace miopen
