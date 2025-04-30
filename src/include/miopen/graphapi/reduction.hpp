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

#include <miopen/miopen.h>
#include <miopen/graphapi/graphapi.hpp>
#include <miopen/graphapi/opgraph.hpp>

namespace miopen {

namespace graphapi {

class Reduction
{
private:
    miopenReduceTensorOp_t mReductionOperator = MIOPEN_REDUCE_TENSOR_ADD;
    miopenDataType_t mCompType                = miopenFloat;

    friend class ReductionBuilder;

public:
    Reduction() noexcept = default;
    Reduction(miopenReduceTensorOp_t reductionOperator, miopenDataType_t compType) noexcept
        : mReductionOperator(reductionOperator), mCompType(compType)
    {
    }

    miopenReduceTensorOp_t getReductionOperator() const { return mReductionOperator; }
    miopenDataType_t getCompType() const { return mCompType; }
};

class MIOPEN_INTERNALS_EXPORT ReductionBuilder
{
private:
    Reduction mReduction;
    bool mReductionOperatorSet = false;
    bool mCompTypeSet          = false;

public:
    ReductionBuilder& setReductionOperator(miopenReduceTensorOp_t reductionOperator) noexcept
    {
        mReduction.mReductionOperator = reductionOperator;
        mReductionOperatorSet         = true;
        return *this;
    }

    ReductionBuilder& setCompType(miopenDataType_t compType) noexcept
    {
        mReduction.mCompType = compType;
        mCompTypeSet         = true;
        return *this;
    }

    Reduction build();
};

class MIOPEN_INTERNALS_EXPORT BackendReductionDescriptor : public BackendDescriptor
{
private:
    ReductionBuilder mBuilder;
    Reduction mReduction;

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

    const Reduction* getReduction() const { return &mReduction; }
    Reduction* getReduction() { return &mReduction; }
};

class MIOPEN_INTERNALS_EXPORT OperationReduction : public OpNode
{
private:
    Reduction* mReduction = nullptr;
    Tensor* mX            = nullptr;
    Tensor* mY            = nullptr;

    friend class OperationReductionBuilder;

public:
    OperationReduction() noexcept = default;
    OperationReduction(Reduction* reduction, Tensor* x, Tensor* y) noexcept
        : mReduction(reduction), mX(x), mY(y)
    {
    }

    Reduction* getReduction() const noexcept { return mReduction; }
    Tensor* getX() const noexcept { return mX; }
    Tensor* getY() const noexcept { return mY; }

    const std::string& signName() const override;
    std::vector<Tensor*> getInTensors() const override;
    std::vector<Tensor*> getOutTensors() const override;
};

class MIOPEN_INTERNALS_EXPORT OperationReductionBuilder
{
private:
    OperationReduction mOperationReduction;

public:
    OperationReductionBuilder& setReduction(Reduction* reduction);
    OperationReductionBuilder& setX(Tensor* x);
    OperationReductionBuilder& setY(Tensor* y);
    OperationReduction build();
};

class MIOPEN_INTERNALS_EXPORT BackendOperationReductionDescriptor : public BackendDescriptor
{
private:
    OperationReductionBuilder mBuilder;
    OperationReduction mOperationReduction;

    miopenBackendDescriptor_t mReductionDescriptor = nullptr;
    miopenBackendDescriptor_t mXDescriptor         = nullptr;
    miopenBackendDescriptor_t mYDescriptor         = nullptr;

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
    OpNode* getOperation() override;

    const OperationReduction* getOperationReduction() const { return &mOperationReduction; }
    OperationReduction* getOperationReduction() { return &mOperationReduction; }
};

} // namespace graphapi

} // namespace miopen
