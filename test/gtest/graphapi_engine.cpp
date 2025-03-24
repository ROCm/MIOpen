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

#include <miopen/graphapi/engine.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::EngineBuilder;
using miopen::graphapi::GraphPatternExecutor;
using miopen::graphapi::OpGraph;
using miopen::graphapi::VariantPack;

class MockPatternExecutor : public GraphPatternExecutor
{
public:
    void execute([[maybe_unused]] miopenHandle_t handle,
                 [[maybe_unused]] const VariantPack& vpk) override
    {
    }
    size_t getWorkspaceSize() const override { return 0; }
    nlohmann::json getJson() override { return {}; };
};

} // namespace

TEST(CPU_GraphApiEngineBuilder_NONE, EngineBuilder)
{
    OpGraph opGraph;
    auto executor = std::make_shared<MockPatternExecutor>();

    EXPECT_NO_THROW({
        EngineBuilder().setGraph(&opGraph).setGlobalIndex(0).setExecutor(executor).build();
    }) << "EngineBuilder failed on valid attributes";

    EXPECT_ANY_THROW({ EngineBuilder().setGlobalIndex(0).setExecutor(executor).build(); })
        << "EngineBuilder failed on missing setGraph() call";

    EXPECT_ANY_THROW({ EngineBuilder().setGraph(&opGraph).setExecutor(executor).build(); })
        << "EngineBuilder failed on missing setGlobalIndex() call";

    EXPECT_ANY_THROW({ EngineBuilder().setGraph(&opGraph).setGlobalIndex(0).build(); })
        << "EngineBuilder failed on missing setExecutor() call";
}

namespace {

using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;

class MockOpGraphDescriptor : public miopen::graphapi::BackendOperationGraphDescriptor
{
public:
    MockOpGraphDescriptor() { mFinalized = true; }
};

} // namespace

TEST(CPU_GraphApiEngine_NONE, Engine)
{
    MockOpGraphDescriptor opGraphDescriptor;

    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> attrOpGraph(
        true,
        "MIOPEN_ATTR_ENGINE_OPERATION_GRAPH",
        MIOPEN_ATTR_ENGINE_OPERATION_GRAPH,
        MIOPEN_TYPE_BACKEND_DESCRIPTOR,
        MIOPEN_TYPE_CHAR,
        2,
        &opGraphDescriptor);

    GTestDescriptorSingleValueAttribute<int64_t, char> attrGlobalIndex(
        true,
        "MIOPEN_ATTR_ENGINE_GLOBAL_INDEX",
        MIOPEN_ATTR_ENGINE_GLOBAL_INDEX,
        MIOPEN_TYPE_INT64,
        MIOPEN_TYPE_CHAR,
        2,
        0);

    GTestDescriptorSingleValueAttribute<int32_t, char> attrSmCount(
        true,
        "MIOPEN_ATTR_ENGINE_SM_COUNT_TARGET",
        MIOPEN_ATTR_ENGINE_SM_COUNT_TARGET,
        MIOPEN_TYPE_INT32,
        MIOPEN_TYPE_CHAR,
        2,
        1);

    GTestGraphApiExecute<GTestDescriptorAttribute*> execute;

    /* BackendOperationGraphDescriptor and the whole API design lacks dependency inversion
     * to be able to achieve a full coverage in tests for BackendOperationGraphDescriptor
     * without touching other parts of the code. We cannot mock OpGraph and make it return
     * a non-empty mock list of Engines, so here we test only attrsValid=false case.
     *
     * We have other tests (for example MHA) that cover attrsValid=true case but they
     * cover the whole API code.
     */

    execute.descriptor.attrsValid = false;
    execute.descriptor.textName   = "MIOPEN_BACKEND_ENGINE_DESCRIPTOR";
    execute.descriptor.type       = MIOPEN_BACKEND_ENGINE_DESCRIPTOR;
    execute.descriptor.attributes = {&attrOpGraph, &attrGlobalIndex, &attrSmCount};

    execute();
}
