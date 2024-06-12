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
#include <miopen/graphapi/engineheur.hpp>

#include <algorithm>

namespace miopen {

namespace graphapi {

EngineHeurBuilder& EngineHeurBuilder::setOpGraph(OpGraph* opGraph)
{
    mEngineHeur.mOpGraph = checkPtr(opGraph);
    return *this;
}

EngineHeurBuilder& EngineHeurBuilder::setMode(miopenBackendHeurMode_t mode)
{
    mEngineHeur.mMode = mode;
    mModeSet          = true;
    return *this;
}

EngineHeurBuilder& EngineHeurBuilder::setSmCount(int32_t smCount)
{
    mEngineHeur.mSmCount = smCount;
    return *this;
}

EngineHeur EngineHeurBuilder::build()
{
    if(mEngineHeur.mOpGraph == nullptr || !mModeSet)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    const auto& engines = mEngineHeur.mOpGraph->getEngines();
    std::for_each(engines.begin(), engines.end(), [this](const Engine& engine) {
        mEngineHeur.mResults.emplace_back(engine);
    });

    return mEngineHeur;
}

void BackendEngineHeurDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_ENGINEHEUR_MODE:
        if(attributeType == MIOPEN_TYPE_HEUR_MODE && elementCount == 1)
        {
            mBuilder.setMode(*static_cast<miopenBackendHeurMode_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
        {
            miopenBackendDescriptor_t apiDescriptor =
                deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
            BackendDescriptor& backendDescriptor = deref(apiDescriptor);

            if(!backendDescriptor.isFinalized())
            {
                MIOPEN_THROW(miopenStatusBadParm);
            }

            BackendOperationGraphDescriptor& opGraphDescriptor =
                dynamic_cast<BackendOperationGraphDescriptor&>(backendDescriptor);
            mBuilder.setOpGraph(opGraphDescriptor.getOperationGraph());
            mOpGraphDescriptor = apiDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINEHEUR_SM_COUNT_TARGET:
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

void BackendEngineHeurDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    mEngineHeur = mBuilder.build();

    const auto& engineCfgs = mEngineHeur.getResults();
    mResults.reserve(engineCfgs.size());

    std::for_each(engineCfgs.begin(), engineCfgs.end(), [this](const EngineCfg& engineCfg) {
        mResults.emplace_back(engineCfg, mOpGraphDescriptor);
    });

    mFinalized = true;
}

void BackendEngineHeurDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_ENGINEHEUR_MODE:
        if(attributeType == MIOPEN_TYPE_HEUR_MODE && requestedElementCount == 1)
        {
            *elementCount                                           = 1;
            *static_cast<miopenBackendHeurMode_t*>(arrayOfElements) = mEngineHeur.getMode();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
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

    case MIOPEN_ATTR_ENGINEHEUR_RESULTS:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount >= 0)
        {
            *elementCount = mResults.size();
            std::transform(mResults.begin(),
                           mResults.begin() + minimum(*elementCount, requestedElementCount),
                           static_cast<miopenBackendDescriptor_t*>(arrayOfElements),
                           [](auto& descriptor) { return &descriptor; });
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_ENGINEHEUR_SM_COUNT_TARGET:
        if(attributeType == MIOPEN_TYPE_INT32 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int32_t*>(arrayOfElements) = mEngineHeur.getSmCount();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

// TODO(Amber): delete
/*
BackendEngineHeurDescriptor::OwnedEngineCfgDescriptor::OwnedEngineCfgDescriptor(
    EngineCfg&& engineCfg, miopenBackendDescriptor_t opGraphDescriptor)
    : BackendEngineCfgDescriptor(std::move(engineCfg), &mOwnedEngineDescriptorInstance),
      mOwnedEngineDescriptorInstance(getEngineCfg().getEngine(), opGraphDescriptor)
{
}

BackendEngineHeurDescriptor::OwnedEngineCfgDescriptor::OwnedEngineCfgDescriptor(
    const OwnedEngineCfgDescriptor& other)
    : BackendEngineCfgDescriptor(other.getEngineCfg(), &mOwnedEngineDescriptorInstance),
      mOwnedEngineDescriptorInstance(other.mOwnedEngineDescriptorInstance)
{
}

BackendEngineHeurDescriptor::OwnedEngineCfgDescriptor::OwnedEngineCfgDescriptor(
    OwnedEngineCfgDescriptor&& other) noexcept
    : BackendEngineCfgDescriptor(std::move(other.getEngineCfg()), &mOwnedEngineDescriptorInstance),
      mOwnedEngineDescriptorInstance(std::move(other.mOwnedEngineDescriptorInstance))
{
}

BackendEngineHeurDescriptor::OwnedEngineCfgDescriptor&
BackendEngineHeurDescriptor::OwnedEngineCfgDescriptor::operator=(
    const OwnedEngineCfgDescriptor& other)
{
    if(this != &other)
    {
        BackendEngineCfgDescriptor::operator=(other);
        mEngineDescriptor                   = &mOwnedEngineDescriptorInstance;
        mOwnedEngineDescriptorInstance      = other.mOwnedEngineDescriptorInstance;
    }
    return *this;
}

BackendEngineHeurDescriptor::OwnedEngineCfgDescriptor&
BackendEngineHeurDescriptor::OwnedEngineCfgDescriptor::operator=(
    OwnedEngineCfgDescriptor&& other) noexcept
{
    if(this != &other)
    {
        BackendEngineCfgDescriptor::operator=(std::move(other));
        mEngineDescriptor                   = &mOwnedEngineDescriptorInstance;
        mOwnedEngineDescriptorInstance      = std::move(other.mOwnedEngineDescriptorInstance);
    }
    return *this;
}
*/

} // namespace graphapi

} // namespace miopen
