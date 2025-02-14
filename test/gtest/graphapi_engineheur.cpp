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

#include <miopen/graphapi/engineheur.hpp>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::EngineHeurBuilder;
using miopen::graphapi::OpGraph;

} // namespace

TEST(CPU_GraphApiEngineHeurBuilder_NONE, EngineHeurBuilder)
{
    OpGraph opGraph;

    EXPECT_NO_THROW({
        EngineHeurBuilder().setOpGraph(&opGraph).setMode(MIOPEN_HEUR_MODE_B).setSmCount(1).build();
    }) << "EngineHeurBuilder failed on valid attributes";

    EXPECT_NO_THROW({
        EngineHeurBuilder().setOpGraph(&opGraph).setMode(MIOPEN_HEUR_MODE_B).build();
    }) << "EngineHeurBuilder failed on valid attributes";

    EXPECT_ANY_THROW({ EngineHeurBuilder().setMode(MIOPEN_HEUR_MODE_B).setSmCount(1).build(); })
        << "EngineHeurBuilder failed on missing setOpGraph() call";

    EXPECT_ANY_THROW({ EngineHeurBuilder().setOpGraph(&opGraph).setSmCount(1).build(); })
        << "EngineHeurBuilder failed on missing setMode() call";
}

namespace {

class MockOpGraphDescriptor : public miopen::graphapi::BackendOperationGraphDescriptor
{
public:
    MockOpGraphDescriptor() { mFinalized = true; }
};

using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;

} // namespace

TEST(CPU_GraphApiEngineHeur_NONE, EngineHeur)
{
    MockOpGraphDescriptor opGraphDescriptor;

    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> attrOpGraph(
        true,
        "MIOPEN_ATTR_ENGINEHEUR_OPERATION_GRAPH",
        MIOPEN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
        MIOPEN_TYPE_BACKEND_DESCRIPTOR,
        MIOPEN_TYPE_CHAR,
        2,
        &opGraphDescriptor);

    GTestDescriptorSingleValueAttribute<miopenBackendHeurMode_t, char> attrMode(
        true,
        "MIOPEN_ATTR_ENGINEHEUR_MODE",
        MIOPEN_ATTR_ENGINEHEUR_MODE,
        MIOPEN_TYPE_HEUR_MODE,
        MIOPEN_TYPE_CHAR,
        2,
        MIOPEN_HEUR_MODE_B);

    GTestDescriptorSingleValueAttribute<int32_t, char> attrSmCount(
        true,
        "MIOPEN_ATTR_ENGINEHEUR_SM_COUNT_TARGET",
        MIOPEN_ATTR_ENGINEHEUR_SM_COUNT_TARGET,
        MIOPEN_TYPE_INT32,
        MIOPEN_TYPE_CHAR,
        2,
        1);

    GTestGraphApiExecute<GTestDescriptorAttribute*> execute;

    execute.descriptor.attrsValid = false;
    execute.descriptor.textName   = "MIOPEN_BACKEND_ENGINEHEUR_DESCRIPTOR";
    execute.descriptor.type       = MIOPEN_BACKEND_ENGINEHEUR_DESCRIPTOR;
    execute.descriptor.attributes = {&attrOpGraph, &attrSmCount};
    execute();

    execute.descriptor.attributes = {&attrMode, &attrSmCount};
    execute();

    execute.descriptor.attrsValid = true;
    execute.descriptor.attributes = {&attrOpGraph, &attrMode, &attrSmCount};
    execute();
}
