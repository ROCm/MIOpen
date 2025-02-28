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

#include <miopen/graphapi/execution_plan.hpp>

namespace miopen {

namespace graphapi {

std::string ExecutionPlan::getJsonRepresentation() const
{
    std::string result;
    nlohmann::json json = *this;
    // We don't support text/json as we have binary data inside
    nlohmann::json::to_msgpack(json, result);
    return result;
}

ExecutionPlanBuilder& ExecutionPlanBuilder::setHandle(miopenHandle_t handle) &
{
    mExecutionPlan.mHandle = checkPtr(handle);
    return *this;
}

ExecutionPlanBuilder& ExecutionPlanBuilder::setEngineCfg(const EngineCfg& engineCfg) &
{
    mExecutionPlan.mEngineCfg = engineCfg;
    mEngineCfgSet             = true;
    return *this;
}

ExecutionPlanBuilder& ExecutionPlanBuilder::setEngineCfg(EngineCfg&& engineCfg) &
{
    mExecutionPlan.mEngineCfg = std::move(engineCfg);
    mEngineCfgSet             = true;
    return *this;
}

ExecutionPlanBuilder& ExecutionPlanBuilder::setIntermediateIds(const std::vector<int64_t>& ids) &
{
    mExecutionPlan.mIntermediateIds = ids;
    return *this;
}

ExecutionPlanBuilder& ExecutionPlanBuilder::setIntermediateIds(std::vector<int64_t>&& ids) &
{
    mExecutionPlan.mIntermediateIds = std::move(ids);
    return *this;
}

ExecutionPlanBuilder& ExecutionPlanBuilder::setJsonRepresentation(const std::string_view& s) &
{
    nlohmann::json json = nlohmann::json::from_msgpack(s);
    json.get_to(mExecutionPlan);
    mJsonRepresentationSet = true;
    return *this;
}

ExecutionPlan ExecutionPlanBuilder::build() &
{
    if((mExecutionPlan.mHandle != nullptr && mEngineCfgSet) || mJsonRepresentationSet)
    {
        return mExecutionPlan;
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

ExecutionPlan ExecutionPlanBuilder::build() &&
{
    if((mExecutionPlan.mHandle != nullptr && mEngineCfgSet) || mJsonRepresentationSet)
    {
        return std::move(mExecutionPlan);
    }
    else
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendExecutionPlanDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_EXECUTION_PLAN_HANDLE:
        if(attributeType == MIOPEN_TYPE_HANDLE && elementCount == 1)
        {
            mBuilder.setHandle(*static_cast<miopenHandle_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount == 1)
        {
            miopenBackendDescriptor_t apiDescriptor =
                deref(static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
            BackendDescriptor& backendDescriptor = deref(apiDescriptor);

            if(!backendDescriptor.isFinalized())
            {
                MIOPEN_THROW(miopenStatusBadParm);
            }

            BackendEngineCfgDescriptor& engineCfgDescriptor =
                dynamic_cast<BackendEngineCfgDescriptor&>(backendDescriptor);
            mBuilder.setEngineCfg(engineCfgDescriptor.getEngineCfg());
            mEngineCfgDescriptor = apiDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
        if(attributeType == MIOPEN_TYPE_INT64 && elementCount >= 0)
        {
            mBuilder.setIntermediateIds({static_cast<int64_t*>(arrayOfElements),
                                         static_cast<int64_t*>(arrayOfElements) + elementCount});
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION:
        if(attributeType == MIOPEN_TYPE_CHAR && elementCount > 0)
        {
            std::string_view s(static_cast<char*>(arrayOfElements), elementCount);
            mBuilder.setJsonRepresentation(s);
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendExecutionPlanDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }
    mExecutionPlan = std::move(mBuilder).build();
    mFinalized     = true;
}

void BackendExecutionPlanDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
    case MIOPEN_ATTR_EXECUTION_PLAN_HANDLE:
        if(attributeType == MIOPEN_TYPE_HANDLE && requestedElementCount == 1)
        {
            *elementCount                                  = 1;
            *static_cast<miopenHandle_t*>(arrayOfElements) = mExecutionPlan.getHandle();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount == 1)
        {
            *elementCount                                             = 1;
            *static_cast<miopenBackendDescriptor_t*>(arrayOfElements) = mEngineCfgDescriptor;
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;
    case MIOPEN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int64_t*>(arrayOfElements) = mExecutionPlan.getWorkspaceSize();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount >= 0)
        {
            const auto& vec = mExecutionPlan.getIntermediateIds();
            *elementCount   = vec.size();
            std::copy_n(vec.begin(),
                        minimum(requestedElementCount, *elementCount),
                        static_cast<int64_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION:
        if(attributeType == MIOPEN_TYPE_CHAR && requestedElementCount > 0)
        {
            std::string s = mExecutionPlan.getJsonRepresentation();
            *elementCount = s.size() + 1;
            std::copy_n(s.c_str(),
                        minimum(requestedElementCount, *elementCount),
                        static_cast<char*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendExecutionPlanDescriptor::execute(miopenHandle_t handle,
                                             miopenBackendDescriptor_t variantPack)
{
    BackendDescriptor& bd = deref(variantPack);
    auto& bendvp          = dynamic_cast<BackendVariantPackDescriptor&>(bd);
    assert(&bendvp);
    mExecutionPlan.execute(handle, *bendvp.getVariantPack());
}

} // namespace graphapi

} // namespace miopen
