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

class MIOPEN_INTERNALS_EXPORT OperationReshape : public OpNode
{
public:
    enum class OpKind
    {
        GENERIC,
        TRANSPOSE
    };

private:
    Tensor* mX     = nullptr;
    Tensor* mY     = nullptr;
    OpKind mOpKind = OpKind::GENERIC;

    friend class OperationReshapeBuilder;

public:
    OperationReshape() noexcept = default;
    OperationReshape(Tensor* x, Tensor* y) : mX(x), mY(y) {}

    Tensor* getX() const noexcept { return mX; }
    Tensor* getY() const noexcept { return mY; }
    OpKind getOpKind() const noexcept { return mOpKind; }

    const std::string& signName() const override;
    std::vector<Tensor*> getInTensors() const override;
    std::vector<Tensor*> getOutTensors() const override;
};

class MIOPEN_INTERNALS_EXPORT OperationReshapeBuilder
{
private:
    OperationReshape mOperationReshape;

public:
    OperationReshapeBuilder& setX(Tensor* x);
    OperationReshapeBuilder& setY(Tensor* y);
    OperationReshape build();
};

class MIOPEN_INTERNALS_EXPORT BackendOperationReshapeDescriptor : public BackendDescriptor
{
private:
    OperationReshapeBuilder mBuilder;
    OperationReshape mOperationReshape;

    miopenBackendDescriptor_t mXDescriptor = nullptr;
    miopenBackendDescriptor_t mYDescriptor = nullptr;

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

    const OperationReshape* getOperationReshape() const { return &mOperationReshape; }
    OperationReshape* getOperationReshape() { return &mOperationReshape; }
};

} // namespace graphapi

} // namespace miopen
