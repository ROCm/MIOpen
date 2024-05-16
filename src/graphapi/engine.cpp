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

GraphPatternExecutor::~GraphPatternExecutor() = default;

size_t GraphExecutorFind20::getWorkspaceSize() const {
  size_t workspace_size = 0;
  auto s = miopenGetSolutionWorkspaceSize(mSolution, &workspace_size);
  MIOPEN_THROW_IF(s != miopenStatusSuccess, "get solution workspace size failed");

  return workspace_size;
}

void GraphExecutorFind20::execute(miopenHandle_t handle, const VariantPack& vpk) {

  std::vector<miopenTensorArgument_t> tens_args;

  auto num = vpk.getTensorIds().size();
  assert(num == vpk.getDataPtrs().size());

  // TODO(amber) : verify that variant pack has all the expected input and output
  // tensors
  for (auto i = 0ull; i < num; ++i) {
    auto tens_id = vpk.getTensorIds()[i];
    auto* gpu_ptr = vpk.getDataPtrs()[i];
    assert(gpu_ptr);

    MIOPEN_THROW_IF(mTensorInfoMap->find(tens_id) == mTensorInfoMap->cend(), 
        "couldn't find a variant pack tensor id in the map");

    auto& v = (*mTensorInfoMap)[tens_id];

    miopenTensorArgument_t targ{
      .id = v.mEnumId,
      // .descriptor = &(v.mTensDesc),
      .descriptor = nullptr,
      .buffer = gpu_ptr
    };

    tens_args.emplace_back(targ);
  }

  auto s = miopenRunSolution(
      handle, 
      mSolution,
      tens_args.size(),
      tens_args.data(),
      vpk.getWorkspace(),
      getWorkspaceSize());

  MIOPEN_THROW_IF(s != miopenStatusSuccess, "Run Solution failed");

}


EngineBuilder& EngineBuilder::setGraph(OpGraph* g)
{
    assert(g);
    mEngine.mGraph = checkPtr(g);
    mGraphSet = true;
    return *this;
}

EngineBuilder& EngineBuilder::setGlobalIndex(int64_t globalIndex)
{
    MIOPEN_THROW_IF(globalIndex < 0, "globalIndex must be >= 0");
    mEngine.mGlobalIndex = globalIndex;
    mIndexSet = true;
    return *this;
}

EngineBuilder& EngineBuilder::setSmCount(int32_t smCount)
{
    MIOPEN_THROW_IF(smCount <= 0, "SM count must be positive");
    mEngine.mSmCount = smCount;
    return *this;
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
            mBuilder.setGraph(operationGraphDescriptor.getOperationGraph());
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
