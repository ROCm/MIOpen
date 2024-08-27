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

#include <miopen/graphapi/reduction.hpp>

#include <gtest/gtest.h>

#include "graphapi_gtest_common.hpp"

namespace {

using miopen::graphapi::OperationReduction;
using miopen::graphapi::OperationReductionBuilder;
using miopen::graphapi::Reduction;
using miopen::graphapi::Tensor;

} // namespace

class CPU_GraphApiOperationReduction_NONE : public testing::Test
{
protected:
    Reduction mReduction{MIOPEN_REDUCE_TENSOR_ADD, miopenFloat};
    Tensor mX{miopenFloat, {8, 64, 64}, {64 * 64, 64, 1}, 1, false};
    Tensor mYs[5]{{miopenFloat, {8, 64, 64}, {64 * 64, 64, 1}, 2, false},
                  {miopenFloat, {8, 1, 64}, {64, 64, 1}, 2, false},
                  {miopenFloat, {8, 64, 1}, {64, 1, 1}, 2, false},
                  {miopenFloat, {8, 1, 1}, {1, 1, 1}, 2, false},
                  {miopenFloat, {8, 128, 1}, {128, 1, 1}, 2, false}};
    Tensor mBadY{miopenFloat, {8, 32, 32}, {32 * 32, 32, 1}, 2, false};
};

TEST_F(CPU_GraphApiOperationReduction_NONE, Builder)
{
    for(Tensor& y : mYs)
    {
        EXPECT_NO_THROW({
            OperationReductionBuilder().setReduction(&mReduction).setX(&mX).setY(&y).build();
        }) << "Builder failed on valid attributes";
    }
    EXPECT_ANY_THROW({
        OperationReductionBuilder().setReduction(&mReduction).setX(&mX).setY(&mBadY).build();
    }) << "Builder failed on invalid attributes";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setReduction(nullptr); })
        << "OperationReductionBuilder::setReduction failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setX(nullptr); })
        << "OperationReductionBuilder::setX failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setY(nullptr); })
        << "OperationReductionBuilder::setY failed on an invalid attribute";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setX(&mX).setY(mYs).build(); })
        << "Builder failed to detect missing setReduction call";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setReduction(&mReduction).setY(mYs).build(); })
        << "Builder failed to detect missing setX call";
    EXPECT_ANY_THROW({ OperationReductionBuilder().setReduction(&mReduction).setX(&mX).build(); })
        << "Builder failed to detect missing setY call";
}

namespace {

using miopen::graphapi::BackendReductionDescriptor;
using miopen::graphapi::GMockBackendTensorDescriptor;
using miopen::graphapi::GTestDescriptorAttribute;
using miopen::graphapi::GTestDescriptorSingleValueAttribute;
using miopen::graphapi::GTestGraphApiExecute;
using miopen::graphapi::ValidatedValue;

class GMockBackendReductionDescriptor : public BackendReductionDescriptor
{
public:
    GMockBackendReductionDescriptor& operator=(Reduction* reduction)
    {
        if(reduction == nullptr)
        {
            return *this;
        }

        auto reductionOperator = reduction->getReductionOperator();
        setAttribute(MIOPEN_ATTR_REDUCTION_OPERATOR,
                     MIOPEN_TYPE_REDUCTION_OPERATOR_TYPE,
                     1,
                     &reductionOperator);

        auto compType = reduction->getCompType();
        setAttribute(MIOPEN_ATTR_REDUCTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &compType);

        finalize();

        return *this;
    }
};

class ReductionAttribute
    : public GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char>
{
private:
    GMockBackendReductionDescriptor mReduction;

public:
    ReductionAttribute()
        : GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char>(
              false,
              "MIOPEN_ATTR_OPERATION_REDUCTION_DESC",
              MIOPEN_ATTR_OPERATION_REDUCTION_DESC,
              MIOPEN_TYPE_BACKEND_DESCRIPTOR,
              MIOPEN_TYPE_CHAR,
              2,
              &mReduction)
    {
    }
    ReductionAttribute(Reduction* reduction) : ReductionAttribute() { *this = reduction; }

    ReductionAttribute& operator=(Reduction* reduction)
    {
        try
        {
            mReduction = reduction;
        }
        catch(...)
        {
        }
        mTestCase.isCorrect = mReduction.isFinalized();
        return *this;
    }
};

class XAttribute : public GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char>
{
private:
    GMockBackendTensorDescriptor mTensor;

public:
    XAttribute()
        : GTestDescriptorSingleValueAttribute<miopenBackendDescriptor_t, char>(
              false,
              "MIOPEN_ATTR_OPERATION_REDUCTION_XDESC",
              MIOPEN_ATTR_OPERATION_REDUCTION_XDESC,
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
              "MIOPEN_ATTR_OPERATION_REDUCTION_YDESC",
              MIOPEN_ATTR_OPERATION_REDUCTION_YDESC,
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

} // namespace

TEST_F(CPU_GraphApiOperationReduction_NONE, CFunctions)
{
    ReductionAttribute invalidReduction;

    ReductionAttribute goodReduction(&mReduction);

    XAttribute invalidX;
    YAttribute invalidY;

    XAttribute x(&mX);

    YAttribute ys[5];
    std::transform(std::begin(mYs), std::end(mYs), std::begin(ys), [](Tensor& tensor) -> Tensor* {
        return &tensor;
    });

    YAttribute badY(&mBadY);

    GTestGraphApiExecute<GTestDescriptorAttribute*> execute{
        {"MIOPEN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR",
         MIOPEN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
         true,
         {}}};

    for(auto& y : ys)
    {
        execute.descriptor.attributes = {&goodReduction, &x, &y};
        execute();
    }

    execute.descriptor.attrsValid = false;

    execute.descriptor.attributes = {&goodReduction, &x, &badY};
    execute();

    execute.descriptor.attributes = {&invalidReduction};
    execute();

    execute.descriptor.attributes = {&invalidX};
    execute();

    execute.descriptor.attributes = {&invalidY};
    execute();

    execute.descriptor.attributes = {&x, ys};
    execute();

    execute.descriptor.attributes = {&goodReduction, ys};
    execute();

    execute.descriptor.attributes = {&goodReduction, &x};
    execute();
}
