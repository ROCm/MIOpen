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
#include <miopen/graphapi/operationgraph_descriptor.hpp>

#include <algorithm>

namespace miopen {

namespace graphapi {

OperationGraph::OperationGraph(miopenHandle_t handle, const OpGraph& opGraph) : mHandle(checkPtr(handle))
{
    /* TODO: to be implemented in
     * [MHA] Implement MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR read only attributes #2914
     * https://github.com/ROCm/MIOpen/issues/2914
     */
    (void)opGraph;
}

OperationGraph::OperationGraph(miopenHandle_t handle, OpGraph&& opGraph) : mHandle(checkPtr(handle))
{
    /* TODO: to be implemented in
     * [MHA] Implement MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR read only attributes #2914
     * https://github.com/ROCm/MIOpen/issues/2914
     */
    (void)opGraph;
}

OperationGraphBuilder& OperationGraphBuilder::setHandle(miopenHandle_t handle) &
{
    mHandle = checkPtr(handle);
    return *this;
}

OperationGraphBuilder& OperationGraphBuilder::setOps(const std::vector<OpNode*>& ops) &
{
    if(ops.empty())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    OpGraphBuilder builder;
    std::for_each(ops.begin(), ops.end(), [&builder](OpNode* op) {
        if(builder.hasNode(checkPtr(op)))
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        builder.addNode(op);
    });
    mOpGraphBuilder = std::move(builder);
    mOpsSet         = true;
    return *this;
}

OperationGraphBuilder& OperationGraphBuilder::addOp(OpNode* op) &
{
    if(mOpGraphBuilder.hasNode(checkPtr(op)))
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    mOpGraphBuilder.addNode(op);
    mOpsSet = true;
    return *this;
}

OperationGraph OperationGraphBuilder::build() &
{
    if(mHandle == nullptr || !mOpsSet)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    return {mHandle, OpGraphBuilder(mOpGraphBuilder).build()};
}

OperationGraph OperationGraphBuilder::build() &&
{
    if(mHandle == nullptr || !mOpsSet)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    return {mHandle, std::move(mOpGraphBuilder).build()};
}

void BackendOperationGraphDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
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
            mBuilder.setHandle(*static_cast<miopenHandle_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATIONGRAPH_OPS:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount > 0)
        {
            std::vector<miopenBackendDescriptor_t> descriptors;
            descriptors.reserve(elementCount);
            std::vector<OpNode*> nodes;
            nodes.reserve(elementCount);

            std::for_each_n(static_cast<miopenBackendDescriptor_t*>(arrayOfElements),
                            elementCount,
                            [&descriptors, &nodes](miopenBackendDescriptor_t apiDescriptor) {
                                BackendDescriptor& backendDescriptor = deref(apiDescriptor);
                                if(backendDescriptor.isFinalized())
                                {
                                    descriptors.push_back(apiDescriptor);
                                    nodes.push_back(backendDescriptor.getOperation());
                                }
                                else
                                {
                                    MIOPEN_THROW(miopenStatusBadParm);
                                }
                            });

            mBuilder.setOps(nodes);
            mOps = std::move(descriptors);
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationGraphDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }
    mOperationGraph = std::move(mBuilder).build();
    mFinalized      = true;
}

void BackendOperationGraphDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
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
            *static_cast<miopenHandle_t*>(arrayOfElements) = mOperationGraph.getHandle();
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
            *static_cast<int64_t*>(arrayOfElements) = mOperationGraph.getEngines().size();
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
