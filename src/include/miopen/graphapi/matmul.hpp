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
#include <miopen/graphapi/graphapi.hpp>

#include <miopen/graphapi/tensor.hpp>
#include <miopen/graphapi/opgraph.hpp>

namespace miopen {
namespace graphapi {

class Matmul
{
private:
    miopenDataType_t mCompType;

public:
    Matmul() = default;
    Matmul(miopenDataType_t computeType) : mCompType(computeType) {}
    miopenDataType_t getComputeType() { return mCompType; }

private:
    friend class MatmulBuilder;
};

class MIOPEN_INTERNALS_EXPORT MatmulBuilder
{

private:
    Matmul mMatmul;
    bool mComputeTypeSet = false;

public:
    MatmulBuilder& setComputeType(miopenDataType_t computeType)
    {
        mMatmul.mCompType = computeType;
        mComputeTypeSet   = true;
        return *this;
    }

    Matmul build() const;
};

class MIOPEN_INTERNALS_EXPORT BackendMatmulDescriptor : public BackendDescriptor
{
private:
    MatmulBuilder mBuilder;
    Matmul mMatmul;

public:
    virtual void setAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              void* arrayOfElements) override;
    virtual void finalize() override;
    virtual void getAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) override;

    const Matmul* getMatmul() const noexcept { return &mMatmul; }
    Matmul* getMatmul() noexcept { return &mMatmul; }
};

class OperationMatmul : public OpNode
{
private:
    Tensor* mA;
    Tensor* mB;
    Tensor* mC;
    int64_t mBatchCount = 1;
    Tensor* mGemmMOverride;
    Tensor* mGemmNOverride;
    Tensor* mGemmKOverride;
    Matmul* mMatmul;

public:
    OperationMatmul(Tensor* A,
                    Tensor* B,
                    Tensor* C,
                    int batchCount,
                    Tensor* MOverride,
                    Tensor* NOverride,
                    Tensor* KOverride,
                    Matmul* matmul) noexcept
        : mA(A),
          mB(B),
          mC(C),
          mBatchCount(batchCount),
          mGemmMOverride(MOverride),
          mGemmNOverride(NOverride),
          mGemmKOverride(KOverride),
          mMatmul(matmul)
    {
    }

    OperationMatmul() = default;
    Tensor* getA() const { return mA; }
    Tensor* getB() const { return mB; }
    Tensor* getC() const { return mC; }
    int64_t getBatchCount() const { return mBatchCount; }
    Tensor* getMOverride() { return mGemmMOverride; }
    Tensor* getNOverride() { return mGemmNOverride; }
    Tensor* getKOverride() { return mGemmKOverride; }
    Matmul* getMatmul() { return mMatmul; }
    virtual std::vector<Tensor*> getInTensors() const override { return {getA(), getB()}; }
    virtual std::vector<Tensor*> getOutTensors() const override { return {getC()}; }
    virtual const std::string& signName() const override
    {
        static const std::string name = "OP_MATMUL";
        return name;
    }

private:
    friend class OperationMatmulBuilder;
};

class MIOPEN_INTERNALS_EXPORT OperationMatmulBuilder
{
private:
    OperationMatmul mOperationMatmul;
    bool mASet      = false;
    bool mBSet      = false;
    bool mCSet      = false;
    bool mMatmulSet = false;

public:
    OperationMatmulBuilder& setA(Tensor* A);

    OperationMatmulBuilder& setB(Tensor* B);

    OperationMatmulBuilder& setC(Tensor* C);

    OperationMatmulBuilder& setBatchCount(int64_t count);

    OperationMatmulBuilder& setGemmMOverride(Tensor* overrideTensor);

    OperationMatmulBuilder& setGemmNOverride(Tensor* overrideTensor);

    OperationMatmulBuilder& setGemmKOverride(Tensor* overrideTensor);

    OperationMatmulBuilder& setMatmulDescriptor(Matmul* mMatmul);

    OperationMatmul build();
};

class MIOPEN_INTERNALS_EXPORT BackendOperationMatmulDescriptor : public BackendDescriptor
{
private:
    OperationMatmulBuilder mBuilder;
    OperationMatmul mMatmul;
    miopenBackendDescriptor_t mA               = nullptr;
    miopenBackendDescriptor_t mB               = nullptr;
    miopenBackendDescriptor_t mC               = nullptr;
    miopenBackendDescriptor_t mGemmMOverride   = nullptr;
    miopenBackendDescriptor_t mGemmNOverride   = nullptr;
    miopenBackendDescriptor_t mGemmKOverride   = nullptr;
    miopenBackendDescriptor_t mMatmuDescriptor = nullptr;

public:
    virtual void setAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t elementCount,
                              void* arrayOfElements) override;
    virtual void finalize() override;
    virtual void getAttribute(miopenBackendAttributeName_t attributeName,
                              miopenBackendAttributeType_t attributeType,
                              int64_t requestedElementCount,
                              int64_t* elementCount,
                              void* arrayOfElements) override;
    OpNode* getOperation() override;
};

} // namespace graphapi
} // namespace miopen
