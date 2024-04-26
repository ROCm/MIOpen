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

#include <miopen/graphapi/tensor.hpp>
#include <miopen/miopen.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include <gtest/gtest.h>

miopenStatus_t CheckStatusAndThrow(miopenStatus_t status, const std::string& msg, bool addStatusToMessage = true)
{
    std::string newMsg = msg;

    if (addStatusToMessage)
    {
        newMsg = "StatusCode=" + std::to_string(status) + ". " + newMsg;
    }

    if (status == miopenStatusNotImplemented)
    {
        std::cerr << "Not Implemented: " << newMsg << std::endl;
    }
    else if(status != miopenStatusSuccess)
    {
        MIOPEN_THROW(status, newMsg);
    }

    return status;
}

class DescriptorWrapper;
typedef std::shared_ptr<DescriptorWrapper> DescriptorWrapperPtr;

class DescriptorWrapper
{
public:
    DescriptorWrapper(miopenBackendDescriptorType_t descriptorType)
        : m_descriptorType(descriptorType), m_descriptor(nullptr)
    {
        CheckStatusAndThrow(miopenBackendCreateDescriptor(descriptorType, &m_descriptor),
                            "miopenBackendCreateDescriptor failed: type=" +
                                std::to_string(descriptorType));
    }

    ~DescriptorWrapper()
    {
        m_refsToKeep.clear();

        EXPECT_NE(m_descriptor, nullptr) << "m_descriptor is nullptr";

        miopenStatus_t status = miopenBackendDestroyDescriptor(m_descriptor);
        EXPECT_EQ(status, miopenStatusSuccess)
            << "Error while destroying descriptor, type: " << m_descriptorType;
    }

    void SetAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      void* arrayOfElements)
    {
        miopenStatus_t status = miopenBackendSetAttribute(
            m_descriptor, attributeName, attributeType, elementCount, arrayOfElements);

        CheckStatusAndThrow(status,
                            "miopenBackendSetAttribute failed: descriptorType = " +
                                std::to_string(m_descriptorType) +
                                ", attributeName=" + std::to_string(attributeName) +
                                ", attributeType=" + std::to_string(attributeType));
    }

    void Finalize()
    {
        CheckStatusAndThrow(miopenBackendFinalize(m_descriptor),
                            "miopenBackendFinalize failed: descriptorType = " +
                                std::to_string(m_descriptorType));
    }

    void AddRef(DescriptorWrapperPtr refToKeep) { m_refsToKeep.push_back(refToKeep); }

    miopenBackendDescriptor_t GetDescriptor() const { return m_descriptor; }
    miopenBackendDescriptorType_t GetDescriptorType() const { return m_descriptorType; }

private:
    miopenBackendDescriptorType_t m_descriptorType;
    miopenBackendDescriptor_t m_descriptor;

    std::vector<DescriptorWrapperPtr> m_refsToKeep;
};

DescriptorWrapperPtr MakeDescriptor(miopenBackendDescriptorType_t descriptorType)
{
    return std::make_shared<DescriptorWrapper>(descriptorType);
}

DescriptorWrapperPtr MakeTensorDescriptor(int64_t uniqueId,
                                          bool isVirtual = false,
                                          int64_t n      = 1,
                                          int64_t h      = 1,
                                          int64_t s      = 1,
                                          int64_t d      = 1)
{
    DescriptorWrapperPtr descWrapperPtr = MakeDescriptor(MIOPEN_BACKEND_TENSOR_DESCRIPTOR);

    miopenDataType_t dtype = miopenFloat;

    descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &dtype);

    int64_t dims[]    = {n, h, s, d};
    int64_t strides[] = {1, 1, 1, 1};
    int64_t alignment = 4;

    descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT64, 4, dims);

    descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT64, 4, strides);
    descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_INT64, 1, &uniqueId);

    // commented this out as Not Implemented
    // descWrapperPtr->SetAttribute(
    //    MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT, MIOPEN_TYPE_INT64, 1, &alignment);

    descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_IS_VIRTUAL, MIOPEN_TYPE_BOOLEAN, 1, &isVirtual);
    descWrapperPtr->Finalize();

    return descWrapperPtr;
}

// just a simple id generator, might be redone if necessary
int64_t GetNextId()
{
    static int64_t counter = 0;

    return counter++;
}

DescriptorWrapperPtr
MakeMatmul(DescriptorWrapperPtr tensor1, DescriptorWrapperPtr tensor2, DescriptorWrapperPtr output)
{
    DescriptorWrapperPtr matmul = MakeDescriptor(MIOPEN_BACKEND_MATMUL_DESCRIPTOR);

    miopenDataType_t dType = miopenFloat;
    matmul->SetAttribute(MIOPEN_ATTR_MATMUL_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &dType);
    matmul->Finalize();

    miopenBackendDescriptor_t childDesc = matmul->GetDescriptor();

    miopenBackendDescriptor_t tensor1Desc = tensor1->GetDescriptor();
    miopenBackendDescriptor_t tensor2Desc = tensor2->GetDescriptor();
    miopenBackendDescriptor_t outputDesc  = output->GetDescriptor();

    DescriptorWrapperPtr matmulOperation =
        MakeDescriptor(MIOPEN_BACKEND_OPERATION_MATMUL_DESCRIPTOR);
    matmulOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_MATMUL_DESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &childDesc);
    matmulOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_MATMUL_ADESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor1Desc);
    matmulOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_MATMUL_BDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor2Desc);
    matmulOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_MATMUL_CDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &outputDesc);

    matmulOperation->AddRef(matmul);

    matmulOperation->Finalize();

    return matmulOperation;
}

DescriptorWrapperPtr MakePointwise(miopenPointwiseMode_t mode,
                                   DescriptorWrapperPtr tensor1,
                                   DescriptorWrapperPtr tensor2,
                                   DescriptorWrapperPtr output)
{
    DescriptorWrapperPtr pointwise = MakeDescriptor(MIOPEN_BACKEND_POINTWISE_DESCRIPTOR);

    pointwise->SetAttribute(MIOPEN_ATTR_POINTWISE_MODE, MIOPEN_TYPE_POINTWISE_MODE, 1, &mode);
    pointwise->Finalize();

    miopenBackendDescriptor_t childDesc = pointwise->GetDescriptor();

    miopenBackendDescriptor_t tensor1Desc = tensor1->GetDescriptor();
    miopenBackendDescriptor_t tensor2Desc = nullptr;

    if(tensor2)
    {
        tensor2Desc = tensor2->GetDescriptor();
    }

    miopenBackendDescriptor_t outputDesc = output->GetDescriptor();

    DescriptorWrapperPtr pointwiseOperation =
        MakeDescriptor(MIOPEN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    pointwiseOperation->SetAttribute(MIOPEN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                                     MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                                     1,
                                     &childDesc);
    pointwiseOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_POINTWISE_XDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor1Desc);

    if(tensor2)
    {
        pointwiseOperation->SetAttribute(
            MIOPEN_ATTR_OPERATION_POINTWISE_BDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor2Desc);
    }

    pointwiseOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_POINTWISE_YDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &outputDesc);

    pointwiseOperation->AddRef(pointwise);

    pointwiseOperation->Finalize();

    return pointwiseOperation;
}

DescriptorWrapperPtr MakeReduction(miopenReduceTensorOp_t opType,
                                   DescriptorWrapperPtr tensor1,
                                   DescriptorWrapperPtr output)
{
    DescriptorWrapperPtr reduction = MakeDescriptor(MIOPEN_BACKEND_REDUCTION_DESCRIPTOR);

    miopenDataType_t dType = miopenFloat;
    reduction->SetAttribute(MIOPEN_ATTR_REDUCTION_COMP_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &dType);

    reduction->SetAttribute(
        MIOPEN_ATTR_REDUCTION_OPERATOR, MIOPEN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &opType);
    reduction->Finalize();

    miopenBackendDescriptor_t childDesc = reduction->GetDescriptor();

    miopenBackendDescriptor_t tensor1Desc = tensor1->GetDescriptor();
    miopenBackendDescriptor_t outputDesc  = output->GetDescriptor();

    DescriptorWrapperPtr reductionOperation =
        MakeDescriptor(MIOPEN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
    reductionOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_REDUCTION_XDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &childDesc);
    reductionOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_REDUCTION_YDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor1Desc);
    reductionOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_REDUCTION_DESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &outputDesc);

    reductionOperation->AddRef(reduction);

    reductionOperation->Finalize();

    return reductionOperation;
}

DescriptorWrapperPtr MakeRNG(DescriptorWrapperPtr probability,
                             DescriptorWrapperPtr seed,
                             DescriptorWrapperPtr offset,
                             DescriptorWrapperPtr output)
{
    DescriptorWrapperPtr rng = MakeDescriptor(MIOPEN_BACKEND_RNG_DESCRIPTOR);
    rng->Finalize();

    miopenBackendDescriptor_t childDesc = rng->GetDescriptor();

    miopenBackendDescriptor_t probabilityDesc = probability->GetDescriptor();
    miopenBackendDescriptor_t seedDesc        = seed->GetDescriptor();
    miopenBackendDescriptor_t offsetDesc      = offset->GetDescriptor();
    miopenBackendDescriptor_t outputDesc      = output->GetDescriptor();

    DescriptorWrapperPtr rngOperation = MakeDescriptor(MIOPEN_BACKEND_OPERATION_RNG_DESCRIPTOR);
    rngOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_RNG_DESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &childDesc);
    rngOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_RNG_SEED, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &seedDesc);
    rngOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_RNG_OFFSET_DESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &offsetDesc);
    rngOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_RNG_YDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &outputDesc);

    rngOperation->AddRef(rng);

    rngOperation->Finalize();

    return rngOperation;
}

TEST(TestCGraphApi, MhaForward)
{
    const int64_t test_n = 2;
    const int64_t test_h = 4;
    const int64_t test_s = 8;
    const int64_t test_d = 16;

    try
    {
        // real tensors
        auto k        = MakeTensorDescriptor(GetNextId(), false, test_n, test_h, test_s, test_d);
        auto v        = MakeTensorDescriptor(GetNextId(), false, test_n, test_h, test_s, test_d);
        auto q        = MakeTensorDescriptor(GetNextId(), false, test_n, test_h, test_s, test_d);
        auto descaleK = MakeTensorDescriptor(GetNextId());
        auto descaleQ = MakeTensorDescriptor(GetNextId());
        auto descaleV = MakeTensorDescriptor(GetNextId());
        auto descaleS = MakeTensorDescriptor(GetNextId());
        auto scaleS   = MakeTensorDescriptor(GetNextId());
        auto scaleO   = MakeTensorDescriptor(GetNextId());

        auto dp   = MakeTensorDescriptor(GetNextId());
        auto ds   = MakeTensorDescriptor(GetNextId());
        auto doff = MakeTensorDescriptor(GetNextId());

        // This scale param is just a float in Find 2.0
        auto atnScl = MakeTensorDescriptor(GetNextId());

        auto o     = MakeTensorDescriptor(GetNextId(), false, test_n, test_h, test_s, test_d);
        auto amaxO = MakeTensorDescriptor(GetNextId());
        auto amaxS = MakeTensorDescriptor(GetNextId());
        auto m     = MakeTensorDescriptor(GetNextId(), false, test_n, test_h, test_s, 1);
        auto zInv  = MakeTensorDescriptor(GetNextId(), false, test_n, test_h, test_s, 1);

        // virtual tensors
        auto tMM0 = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_s);
        auto pwS0 = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_s);
        auto pwS1 = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_s);
        auto pwS2 = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_s);

        auto tSub   = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_s);
        auto tExp   = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_s);
        auto tSum   = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, 1);
        auto tMult0 = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_s);
        auto tRnd   = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_s);
        auto tMult1 = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_s);
        auto pwS3   = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_s);
        auto pwS4   = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_s);

        auto tMM1 = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_d);
        auto pwS5 = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_d);
        auto pwS6 = MakeTensorDescriptor(GetNextId(), true, test_n, test_h, test_s, test_d);

        // Node creation

        auto matmul0  = MakeMatmul(q, k, tMM0);
        auto pwScale0 = MakePointwise(MIOPEN_POINTWISE_MUL, tMM0, atnScl, pwS0);
        auto pwScale1 = MakePointwise(MIOPEN_POINTWISE_MUL, pwS0, descaleQ, pwS1);
        auto pwScale2 = MakePointwise(MIOPEN_POINTWISE_MUL, pwS1, descaleK, pwS2);

        auto reduction0 = MakeReduction(MIOPEN_REDUCE_TENSOR_MAX, pwScale2, m);
        auto pwSubtract = MakePointwise(MIOPEN_POINTWISE_SUB, pwScale2, m, tSub);
        auto pwExp      = MakePointwise(MIOPEN_POINTWISE_EXP, tSub, DescriptorWrapperPtr(), tExp);
        auto reduction1 = MakeReduction(MIOPEN_REDUCE_TENSOR_ADD, tExp, tSum);
        auto pwInv      = MakePointwise(MIOPEN_POINTWISE_EXP, tSum, DescriptorWrapperPtr(), zInv);
        auto pwMult0    = MakePointwise(MIOPEN_POINTWISE_MUL, tExp, zInv, tMult0);

        auto reduction2 = MakeReduction(MIOPEN_REDUCE_TENSOR_MAX, tMult0, amaxS);

        auto rng = MakeRNG(dp, ds, doff, tRnd);

        auto pwMult1  = MakePointwise(MIOPEN_POINTWISE_MUL, tMult0, tRnd, tMult1);
        auto pwScale3 = MakePointwise(MIOPEN_POINTWISE_MUL, tMult1, dp, pwS3);
        auto pwScale4 = MakePointwise(MIOPEN_POINTWISE_MUL, pwS3, scaleS, pwS4);

        auto matmul1    = MakeMatmul(pwS4, v, tMM1);
        auto pwScale5   = MakePointwise(MIOPEN_POINTWISE_MUL, tMM1, descaleS, pwS5);
        auto pwScale6   = MakePointwise(MIOPEN_POINTWISE_MUL, pwS5, descaleV, pwS6);
        auto reduction3 = MakeReduction(MIOPEN_REDUCE_TENSOR_MAX, pwS6, amaxO);
        auto pwScale7   = MakePointwise(MIOPEN_POINTWISE_MUL, pwS6, scaleO, o);
    }
    catch(const miopen::Exception& ex)
    {
        FAIL() << ex.what();
    }
}
