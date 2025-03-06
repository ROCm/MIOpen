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

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::EngineCfg;
using miopen::graphapi::ExecutionPlanBuilder;

} // namespace

TEST(CPU_GraphApiExecutionPlanBuilder_NONE, ExecutionPlanBuilder)
{
    miopenHandle_t handle;
    auto status = miopenCreate(&handle);
    ASSERT_EQ(status, miopenStatusSuccess) << "miopenCreate() failed";

    EngineCfg engineCfg;

    EXPECT_NO_THROW({ ExecutionPlanBuilder().setHandle(handle).setEngineCfg(engineCfg).build(); })
        << "ExecutionPlanBuilder failed on valid attributes";

    EXPECT_NO_THROW({
        ExecutionPlanBuilder()
            .setHandle(handle)
            .setEngineCfg(engineCfg)
            .setIntermediateIds({1, 2, 3})
            .build();
    }) << "ExecutionPlanBuilder failed on valid attributes";

    EXPECT_ANY_THROW({ ExecutionPlanBuilder().setEngineCfg(engineCfg).build(); })
        << "ExecutionPlanBuilder failed on missing setHandle() call";

    EXPECT_ANY_THROW({ ExecutionPlanBuilder().setHandle(handle).build(); })
        << "ExecutionPlanBuilder failed on missing setEngineCfg() call";
}

namespace {

class MockBackendEngineCfgDescriptor : public miopen::graphapi::BackendEngineCfgDescriptor
{
public:
    MockBackendEngineCfgDescriptor() { mFinalized = true; }
};

using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;

} // namespace

TEST(CPU_GraphApiExecutionPlanBuilder_NONE, ExecutionPlan)
{
    miopenHandle_t handle;
    auto status = miopenCreate(&handle);
    ASSERT_EQ(status, miopenStatusSuccess) << "miopenCreate() failed";

    MockBackendEngineCfgDescriptor engineCfgDescriptor;

    GTestDescriptorSingleValueAttribute<miopenHandle_t, char> attrHandle(
        true,
        "MIOPEN_ATTR_EXECUTION_PLAN_HANDLE",
        MIOPEN_ATTR_EXECUTION_PLAN_HANDLE,
        MIOPEN_TYPE_HANDLE,
        MIOPEN_TYPE_CHAR,
        2,
        handle);

    GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char> attrEngineCfg(
        true,
        "MIOPEN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG",
        MIOPEN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
        MIOPEN_TYPE_BACKEND_DESCRIPTOR,
        MIOPEN_TYPE_CHAR,
        2,
        &engineCfgDescriptor);

    GTestGraphApiExecute<GTestDescriptorAttribute*> execute;

    execute.descriptor.attrsValid = true;
    execute.descriptor.textName   = "MIOPEN_BACKEND_EXECUTION_PLAN_DESCRIPTOR";
    execute.descriptor.type       = MIOPEN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
    execute.descriptor.attributes = {&attrHandle, &attrEngineCfg};

    execute();
}
