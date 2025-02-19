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
#include <miopen/graphapi/tensor.hpp>
#include <miopen/graphapi/variant_pack.hpp>
#include <miopen/solution.hpp>
#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <string_view>

namespace miopen {

namespace graphapi {

class Engine;
class OpGraph;

// Pattern is a family of solvers for the same graph shape
class MIOPEN_INTERNALS_EXPORT GraphPatternMatcher
{

public:
    virtual bool matches(const OpGraph* graph) const             = 0;
    virtual std::vector<Engine> getEngines(OpGraph* graph) const = 0;
    virtual std::string_view name() const                        = 0;

    virtual ~GraphPatternMatcher();
};

struct TensorInfo
{
    miopenTensorArgumentId_t mEnumId = miopenTensorArgumentIdInvalid;
    Tensor* mGraphTensor             = nullptr;
    Data_t mDevBuf                   = nullptr;

    TensorInfo(miopenTensorArgumentId_t enum_id, Tensor* tens_ptr)
        : mEnumId(enum_id), mGraphTensor(tens_ptr)
    {
        assert(mEnumId != miopenTensorArgumentIdInvalid);
    }

    void setDevBuf(Data_t ptr)
    {
        assert(ptr);
        mDevBuf = ptr;
    }
};

// int64_t is the graph tensor id
using TensorInfoMap = std::unordered_map<int64_t, TensorInfo>;

class MIOPEN_INTERNALS_EXPORT GraphPatternExecutor
{
public:
    virtual void execute(miopenHandle_t handle, const VariantPack& vpk) = 0;
    virtual size_t getWorkspaceSize() const                             = 0;
    virtual nlohmann::json getJson()                                    = 0;
    virtual ~GraphPatternExecutor();

    struct JsonFields
    {
        static constexpr const char* Name = "name";
    };
};

// generic executor that uses Find 2.0 Solution
class GraphExecutorFind20 : public GraphPatternExecutor
{
    Solution mSolution;
    std::shared_ptr<TensorInfoMap> mTensorInfoMap;

public:
    GraphExecutorFind20(const Solution& sol, const std::shared_ptr<TensorInfoMap>& tmap)
        : GraphPatternExecutor(), mSolution(sol), mTensorInfoMap(tmap)
    {
        MIOPEN_THROW_IF(tmap.get() == nullptr,
                        "TensorInfoMap should be allocated before GraphExecutorFind20 creation");
    }

    GraphExecutorFind20(Solution&& sol, const std::shared_ptr<TensorInfoMap>& tmap)
        : GraphPatternExecutor(), mSolution(std::move(sol)), mTensorInfoMap(tmap)
    {
        MIOPEN_THROW_IF(tmap.get() == nullptr,
                        "TensorInfoMap should be allocated before GraphExecutorFind20 creation");
    }

    GraphExecutorFind20(const nlohmann::json& json);

    void execute(miopenHandle_t handle, const VariantPack& vpk) final;

    size_t getWorkspaceSize() const final;

    nlohmann::json getJson() final;

    static constexpr const char* name = "GraphExecutorFind20";

    struct JsonFields
    {
        static constexpr const char* Solution       = "solution";
        static constexpr const char* Id2ArgumentMap = "id_to_argument_map";
    };
};

class Engine
{
private:
    std::shared_ptr<GraphPatternExecutor> mExecutor;
    OpGraph* mGraph      = nullptr;
    int64_t mGlobalIndex = -1;
    int32_t mSmCount     = 0;
    friend class EngineBuilder;

public:
    Engine()              = default;
    Engine(const Engine&) = default;
    Engine(Engine&&)      = default;
    Engine& operator=(const Engine&) = default;
    Engine& operator=(Engine&&) = default;

    GraphPatternExecutor* getExecutor() noexcept { return mExecutor.get(); }

    const std::shared_ptr<GraphPatternExecutor>& getExecutor() const noexcept { return mExecutor; }

    int64_t getGlobalIndex() const noexcept { return mGlobalIndex; }
    int32_t getSmCount() const noexcept { return mSmCount; }

    const OpGraph* getOpGraph() const { return mGraph; }
    OpGraph* getOpGraph() { return mGraph; }

    friend void to_json(nlohmann::json& json, const Engine& engine);
    friend void from_json(const nlohmann::json& json, Engine& engine);

    struct JsonFields
    {
        static constexpr const char* Executor    = "executor";
        static constexpr const char* GlobalIndex = "global_index";
        static constexpr const char* SmCount     = "sm_count";
    };
};

class MIOPEN_INTERNALS_EXPORT EngineBuilder
{
    friend class BackendEngineDescriptor;

    std::shared_ptr<GraphPatternExecutor> mExecutor = nullptr;
    OpGraph* mGraph                                 = nullptr;
    int64_t mGlobalIndex                            = -1;
    int32_t mSmCount                                = 0;
    bool mGraphSet                                  = false;
    bool mExecSet                                   = false;
    bool mIndexSet                                  = false;

public:
    EngineBuilder& setGraph(OpGraph* g);

    EngineBuilder& setGlobalIndex(int64_t globalIndex);

    EngineBuilder& setSmCount(int32_t smCount);

    EngineBuilder& setExecutor(const std::shared_ptr<GraphPatternExecutor>& e);

    Engine build();
};

class MIOPEN_INTERNALS_EXPORT BackendEngineDescriptor : public BackendDescriptor
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

MIOPEN_INTERNALS_EXPORT std::vector<Engine> findEngines(OpGraph*);

} // namespace graphapi

} // namespace miopen
