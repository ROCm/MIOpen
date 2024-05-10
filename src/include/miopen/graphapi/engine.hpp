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

#include <miopen/graphapi/graphapi.hpp>
#include <miopen/solution.hpp>

namespace miopen {

namespace graphapi {

class Engine
{
private:
    Solution mSolution;
    int64_t mGlobalIndex = -1;
    int32_t mSmCount     = 0;
    friend class EngineBuilder;

public:
    Engine()              = default;
    Engine(const Engine&) = default;
    Engine(Engine&&)      = default;
    Engine& operator=(const Engine&) = default;
    Engine& operator=(Engine&&) = default;

    Engine(const Solution& solution) : mSolution(solution) {}
    Engine(Solution&& solution) : mSolution(std::move(solution)) {}

    const Solution& getSolution() const noexcept { return mSolution; }
    Solution& getSolution() noexcept { return mSolution; }

    int64_t getGlobalIndex() const noexcept { return mGlobalIndex; }
    int32_t getSmCount() const noexcept { return mSmCount; }
};

class OpGraph;

class EngineBuilder
{
private:
    const OpGraph* mOpGraph = nullptr;
    int64_t mGlobalIndex    = -1;
    int32_t mSmCount        = 0;
    bool mGlobalIndexSet    = false;

public:
    EngineBuilder& setOpGraph(const OpGraph* opGraph);
    EngineBuilder& setGlobalIndex(int64_t globalIndex);
    EngineBuilder& setSmCount(int32_t smCount);
    Engine build();
};

class BackendEngineDescriptor : public BackendDescriptor
{
private:
    EngineBuilder mBuilder;
    Engine mEngine;

    miopenBackendDescriptor_t mOpGraphDescriptor = nullptr;

public:
    BackendEngineDescriptor() = default;
    BackendEngineDescriptor(const Engine& engine, miopenBackendDescriptor_t opGraphDescriptor)
        : mEngine(engine), mOpGraphDescriptor(opGraphDescriptor)
    {
    }

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

    const Engine& getEngine() const noexcept { return mEngine; }
    Engine& getEngine() noexcept { return mEngine; }
};

} // namespace graphapi

} // namespace miopen
