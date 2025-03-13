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

#include <miopen/graphapi/enginecfg.hpp>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::Engine;
using miopen::graphapi::EngineCfgBuilder;

} // namespace

TEST(CPU_GraphApiEngineCfgBuilder_NONE, EngineCfgBuilder)
{
    Engine engine;

    EXPECT_NO_THROW({ EngineCfgBuilder().setEngine(engine).build(); })
        << "EngineCfgBuilder failed on valid attributes";

    EXPECT_ANY_THROW({ EngineCfgBuilder().build(); })
        << "EngineCfgBuilder failed on missing setEngine() call";
}

namespace {

class MockEngineDescriptor : public miopen::graphapi::BackendEngineDescriptor
{
public:
    MockEngineDescriptor() { mFinalized = true; }
};

using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;

} // namespace

TEST(CPU_GraphApiEngineCfg_NONE, EngineCfg)
{
    MockEngineDescriptor engineDescriptor;

    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> attrEngine(
        true,
        "MIOPEN_ATTR_ENGINECFG_ENGINE",
        MIOPEN_ATTR_ENGINECFG_ENGINE,
        MIOPEN_TYPE_BACKEND_DESCRIPTOR,
        MIOPEN_TYPE_CHAR,
        2,
        &engineDescriptor);

    GTestGraphApiExecute<GTestDescriptorAttribute*> execute;

    execute.descriptor.attrsValid = true;
    execute.descriptor.textName   = "MIOPEN_BACKEND_ENGINECFG_DESCRIPTOR";
    execute.descriptor.type       = MIOPEN_BACKEND_ENGINECFG_DESCRIPTOR;
    execute.descriptor.attributes = {&attrEngine};

    execute();
}
