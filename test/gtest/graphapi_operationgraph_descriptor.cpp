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

#include <miopen/graphapi/opgraph.hpp>

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::BackendDescriptor;
using miopen::graphapi::OpNode;
using miopen::graphapi::Tensor;

static int64_t id = 0;

class GMockNode : public OpNode
{
private:
    std::shared_ptr<Tensor> mIn;
    std::shared_ptr<Tensor> mOut;

public:
    GMockNode()
        : mIn(std::make_shared<Tensor>(miopenFloat,
                                       std::vector<std::size_t>{8, 64, 64},
                                       std::vector<std::size_t>{64 * 64, 64, 1},
                                       ++id,
                                       false)),
          mOut(std::make_shared<Tensor>(miopenFloat,
                                        std::vector<std::size_t>{8, 64, 64},
                                        std::vector<std::size_t>{64 * 64, 64, 1},
                                        ++id,
                                        false))
    {
    }

    std::vector<Tensor*> getInTensors() const override { return {mIn.get()}; }

    std::vector<Tensor*> getOutTensors() const override { return {mOut.get()}; }

    const std::string& signName() const override
    {
        static const std::string name = "OP_MOCK";
        return name;
    }
};

class GMockBackendOperationDescriptor : public BackendDescriptor
{
private:
    GMockNode mNode;

public:
    GMockBackendOperationDescriptor(bool finalized) { mFinalized = finalized; }
    void setAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      void* arrayOfElements) override
    {
    }
    void finalize() override {}
    void getAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) override
    {
    }
    OpNode* getOperation() override { return &mNode; }
};

using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestDescriptorVectorAttribute;
using miopen::graphapi::GTestGraphApiExecute;

} // namespace

TEST(CPU_GraphApiOperationGraphDescriptor_NONE, CFunctions)
{
    miopenHandle_t handle = nullptr;
    auto status           = miopenCreate(&handle);
    ASSERT_EQ(status, miopenStatusSuccess) << "Handle wasn't obtained, cannot continue";

    GTestDescriptorSingleValueAttribute<miopenHandle_t, char> handleAttribute{
        true,
        "MIOPEN_ATTR_OPERATIONGRAPH_HANDLE",
        MIOPEN_ATTR_OPERATIONGRAPH_HANDLE,
        MIOPEN_TYPE_HANDLE,
        MIOPEN_TYPE_CHAR,
        0,
        handle};

    GMockBackendOperationDescriptor goodNode(true);

    GTestDescriptorVectorAttribute<miopenBackendDescriptor_t, char> goodOp{
        true,
        "MIOPEN_ATTR_OPERATIONGRAPH_OPS",
        MIOPEN_ATTR_OPERATIONGRAPH_OPS,
        MIOPEN_TYPE_BACKEND_DESCRIPTOR,
        MIOPEN_TYPE_CHAR,
        -1,
        {&goodNode}};
    ;

    GTestGraphApiExecute<GTestDescriptorAttribute*> execute{
        {"MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR",
         MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
         true,
         {&handleAttribute, &goodOp}}};

    execute();

    execute.descriptor.attributes = {&handleAttribute, &goodOp, &goodOp};
    execute();

    GMockBackendOperationDescriptor badNode(false);

    GTestDescriptorVectorAttribute<miopenBackendDescriptor_t, char> badOp{
        false,
        "MIOPEN_ATTR_OPERATIONGRAPH_OPS",
        MIOPEN_ATTR_OPERATIONGRAPH_OPS,
        MIOPEN_TYPE_BACKEND_DESCRIPTOR,
        MIOPEN_TYPE_CHAR,
        -1,
        {&badNode}};
    ;

    execute.descriptor.attrsValid = false;

    execute.descriptor.attributes = {&handleAttribute};
    execute();

    execute.descriptor.attributes = {&goodOp};
    execute();

    execute.descriptor.attributes = {&handleAttribute, &badOp};
    execute();

    badOp = {false,
             "MIOPEN_ATTR_OPERATIONGRAPH_OPS",
             MIOPEN_ATTR_OPERATIONGRAPH_OPS,
             MIOPEN_TYPE_BACKEND_DESCRIPTOR,
             MIOPEN_TYPE_CHAR,
             -1,
             {&goodNode, &goodNode}};
    execute();

    miopenDestroy(handle);
}
