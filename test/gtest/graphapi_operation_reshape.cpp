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

#include <miopen/graphapi/reshape.hpp>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::OperationReshape;
using miopen::graphapi::OperationReshapeBuilder;
using miopen::graphapi::Tensor;

} // namespace

class CPU_GraphApiOperationReshape_NONE : public testing::Test
{
protected:
    Tensor mX{miopenFloat, {8, 64, 128}, {64 * 128, 128, 1}, 1, false};
    Tensor mY{miopenFloat, {8, 128, 64}, {64 * 128, 1, 128}, 1, false};
};

TEST_F(CPU_GraphApiOperationReshape_NONE, Builder)
{
    EXPECT_NO_THROW({ OperationReshapeBuilder().setX(&mX).setY(&mY).build(); })
        << "Builder failed on valid attributes";
    EXPECT_ANY_THROW({ OperationReshapeBuilder().setX(nullptr); })
        << "OperationReshapeBuilder::setX failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationReshapeBuilder().setY(nullptr); })
        << "OperationReshapeBuilder::setY failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationReshapeBuilder().setY(&mY).build(); })
        << "Builder failed on missing setX call";
    EXPECT_ANY_THROW({ OperationReshapeBuilder().setX(&mX).build(); })
        << "Builder failed on missing setY call";
}

TEST_F(CPU_GraphApiOperationReshape_NONE, Transpose)
{
    OperationReshape transpose, notTranspose;

    ASSERT_NO_THROW({
        transpose    = OperationReshapeBuilder().setX(&mX).setY(&mY).build();
        notTranspose = OperationReshapeBuilder().setX(&mX).setY(&mX).build();
    }) << "OperationReshapeBuilder failed on valid attributes";

    EXPECT_EQ(transpose.getOpKind(), OperationReshape::OpKind::TRANSPOSE)
        << "False negative detection for reshape transpose";

    EXPECT_NE(notTranspose.getOpKind(), OperationReshape::OpKind::TRANSPOSE)
        << "False positive detection for reshape transpose";
}

namespace {

using miopen::graphapi::GMockBackendTensorDescriptor;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::ValidatedValue;

class XAttribute : public GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char>
{
private:
    GMockBackendTensorDescriptor mTensor;

public:
    XAttribute()
        : GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char>(
              false,
              "MIOPEN_ATTR_OPERATION_RESHAPE_XDESC",
              MIOPEN_ATTR_OPERATION_RESHAPE_XDESC,
              MIOPEN_TYPE_BACKEND_DESCRIPTOR,
              MIOPEN_TYPE_CHAR,
              2,
              &mTensor)
    {
    }
    XAttribute(Tensor* tensor) : XAttribute() { *this = tensor; }

    XAttribute& operator=(Tensor* tensor)
    {
        try
        {
            mTensor = ValidatedValue<Tensor*>{tensor != nullptr, tensor};
        }
        catch(...)
        {
        }
        mTestCase.isCorrect = mTensor.isFinalized();
        return *this;
    }
};

class YAttribute : public GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char>
{
private:
    GMockBackendTensorDescriptor mTensor;

public:
    YAttribute()
        : GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char>(
              false,
              "MIOPEN_ATTR_OPERATION_RESHAPE_YDESC",
              MIOPEN_ATTR_OPERATION_RESHAPE_YDESC,
              MIOPEN_TYPE_BACKEND_DESCRIPTOR,
              MIOPEN_TYPE_CHAR,
              2,
              &mTensor)
    {
    }
    YAttribute(Tensor* tensor) : YAttribute() { *this = tensor; }

    YAttribute& operator=(Tensor* tensor)
    {
        try
        {
            mTensor = ValidatedValue<Tensor*>{tensor != nullptr, tensor};
        }
        catch(...)
        {
        }
        mTestCase.isCorrect = mTensor.isFinalized();
        return *this;
    }
};

using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestGraphApiExecute;

} // namespace

TEST_F(CPU_GraphApiOperationReshape_NONE, CFunctions)
{
    XAttribute x{&mX};
    YAttribute y{&mY};

    GTestGraphApiExecute<GTestDescriptorAttribute*> execute{
        {"MIOPEN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR",
         MIOPEN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR,
         true,
         {&x, &y}}};

    execute();
}
