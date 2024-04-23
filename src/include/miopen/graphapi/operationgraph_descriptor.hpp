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
#pragma once

#include <miopen/graphapi/engine.hpp>
#include <miopen/graphapi/graphapi.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <miopen/miopen.h>

#include <cstdint>
#include <vector>

namespace miopen {

namespace graphapi {

/* The class represents MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
 * and holds a list of engines; it should have been called something
 * like EngineList, but to be closer to Graph API terminology
 * it was called as below. Do not confuse it with OpGraph
 * representing a graph of operations.
 */
class OperationGraph
{
private:
    miopenHandle_t mHandle = nullptr;
    std::vector<Engine> mEngines;

public:
    OperationGraph() noexcept             = default;
    OperationGraph(const OperationGraph&) = default;
    OperationGraph(OperationGraph&&)      = default;
    OperationGraph& operator=(const OperationGraph&) = default;
    OperationGraph& operator=(OperationGraph&&) = default;

    OperationGraph(miopenHandle_t handle, const OpGraph& opGraph);
    OperationGraph(miopenHandle_t handle, OpGraph&& opGraph);

    miopenHandle_t getHandle() const noexcept { return mHandle; }
    const std::vector<Engine>& getEngines() const noexcept { return mEngines; }
};

class OperationGraphBuilder
{
private:
    OpGraphBuilder mOpGraphBuilder;
    miopenHandle_t mHandle = nullptr;
    bool mOpsSet           = false;

public:
    OperationGraphBuilder& setHandle(miopenHandle_t handle) &;
    OperationGraphBuilder& setOps(const std::vector<OpNode*>& ops) &;
    OperationGraphBuilder& addOp(OpNode* op) &;

    OperationGraphBuilder&& setHandle(miopenHandle_t handle) &&
    {
        return std::move(setHandle(handle));
    }
    OperationGraphBuilder&& setOps(const std::vector<OpNode*>& ops) &&
    {
        return std::move(setOps(ops));
    }
    OperationGraphBuilder&& addOp(OpNode* op) && { return std::move(addOp(op)); }

    OperationGraph build() &;
    OperationGraph build() &&;
};

class BackendOperationGraphDescriptor : public BackendDescriptor
{
private:
    OperationGraphBuilder mBuilder;
    OperationGraph mOperationGraph;
    std::vector<miopenBackendDescriptor_t> mOps; // to return them in getAttribute

public:
    void setAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      void* arrayOfElements) override;
    void finalize() override;
    void getAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) override;

    const OperationGraph* getOperationGraph() const noexcept { return &mOperationGraph; }
    OperationGraph* getOperationGraph() noexcept { return &mOperationGraph; }
};

} // namespace graphapi

} // namespace miopen
