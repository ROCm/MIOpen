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

#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "../workspace.hpp"
#include "gtest/mha_helper.hpp"

miopenStatus_t
CheckStatusAndThrow(miopenStatus_t status, const std::string& msg, bool addStatusToMessage = true)
{
    if(status == miopenStatusSuccess)
    {
        return status;
    }

    std::string newMsg = msg;

    if(addStatusToMessage)
    {
        newMsg = "StatusCode=" + std::to_string(status) + ". " + newMsg;
    }

    if(status == miopenStatusNotImplemented)
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

    void GetAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements)
    {
        miopenStatus_t status = miopenBackendGetAttribute(m_descriptor,
                                                          attributeName,
                                                          attributeType,
                                                          requestedElementCount,
                                                          elementCount,
                                                          arrayOfElements);

        CheckStatusAndThrow(status,
                            "miopenBackendGetAttribute failed: descriptorType = " +
                                std::to_string(m_descriptorType) +
                                ", attributeName=" + std::to_string(attributeName) +
                                ", attributeType=" + std::to_string(attributeType) +
                                ", requestedElementCount=" + std::to_string(requestedElementCount));
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

struct TensorData
{
    DescriptorWrapperPtr m_gapiDesc;
    tensor<float> m_tensor;
    miopen::Allocator::ManageDataPtr m_gpuBuffer;

    void Init(tensor<float>&& tens_val) { m_tensor = std::move(tens_val); }

    void Init(float val)
    {
        m_tensor.generate([=](auto...) { return val; });
    }
};

typedef std::shared_ptr<TensorData> TensorDataPtr;

DescriptorWrapperPtr MakeDescriptor(miopenBackendDescriptorType_t descriptorType)
{
    return std::make_shared<DescriptorWrapper>(descriptorType);
}

DescriptorWrapperPtr MakeGapiTensorDesc(int64_t uniqueId,
                                        bool isVirtual = false,
                                        int64_t n      = 1,
                                        int64_t h      = 1,
                                        int64_t s      = 1,
                                        int64_t d      = 1,
                                        bool transpose = false)
{
    DescriptorWrapperPtr descWrapperPtr = MakeDescriptor(MIOPEN_BACKEND_TENSOR_DESCRIPTOR);

    miopenDataType_t dtype = miopenFloat;

    descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &dtype);

    std::vector<int64_t> dims    = {n, h, s, d};
    std::vector<int64_t> strides = {1, n, n * h, n * h * s};

    if(transpose)
    {
        dims    = {n, h, d, s};
        strides = {1, n, n * h * s, n * h};
    }

    // commented this out as Not Implemented
    // int64_t alignment = 4;

    descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT64, 4, dims.data());

    descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT64, 4, strides.data());
    descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_INT64, 1, &uniqueId);

    // commented this out as Not Implemented
    // descWrapperPtr->SetAttribute(
    //    MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT, MIOPEN_TYPE_INT64, 1, &alignment);

    descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_IS_VIRTUAL, MIOPEN_TYPE_BOOLEAN, 1, &isVirtual);
    descWrapperPtr->Finalize();

    return descWrapperPtr;
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
    matmulOperation->AddRef(tensor1);
    matmulOperation->AddRef(tensor2);
    matmulOperation->AddRef(output);

    matmulOperation->Finalize();

    return matmulOperation;
}

DescriptorWrapperPtr MakePointwise(miopenPointwiseMode_t mode,
                                   DescriptorWrapperPtr tensor1,
                                   DescriptorWrapperPtr tensor2,
                                   DescriptorWrapperPtr output,
                                   bool setAlpha1Param = false,
                                   float alpha1Param   = 0.0f)
{
    DescriptorWrapperPtr pointwise = MakeDescriptor(MIOPEN_BACKEND_POINTWISE_DESCRIPTOR);

    pointwise->SetAttribute(MIOPEN_ATTR_POINTWISE_MODE, MIOPEN_TYPE_POINTWISE_MODE, 1, &mode);

    miopenDataType_t dType = miopenFloat;
    pointwise->SetAttribute(MIOPEN_ATTR_POINTWISE_MATH_PREC, MIOPEN_TYPE_DATA_TYPE, 1, &dType);
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

        pointwiseOperation->AddRef(tensor2);
    }

    pointwiseOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_POINTWISE_YDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &outputDesc);

    if(setAlpha1Param)
    {
        pointwiseOperation->SetAttribute(
            MIOPEN_ATTR_OPERATION_POINTWISE_ALPHA1, MIOPEN_TYPE_FLOAT, 1, &alpha1Param);
    }

    pointwiseOperation->AddRef(pointwise);
    pointwiseOperation->AddRef(tensor1);
    pointwiseOperation->AddRef(output);

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
        MIOPEN_ATTR_OPERATION_REDUCTION_XDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor1Desc);
    reductionOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_REDUCTION_YDESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &outputDesc);
    reductionOperation->SetAttribute(
        MIOPEN_ATTR_OPERATION_REDUCTION_DESC, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &childDesc);

    reductionOperation->AddRef(reduction);
    reductionOperation->AddRef(tensor1);
    reductionOperation->AddRef(output);

    reductionOperation->Finalize();

    return reductionOperation;
}

DescriptorWrapperPtr MakeRNG(double probability,
                             DescriptorWrapperPtr seed,
                             DescriptorWrapperPtr offset,
                             DescriptorWrapperPtr output)
{
    DescriptorWrapperPtr rng = MakeDescriptor(MIOPEN_BACKEND_RNG_DESCRIPTOR);

    rng->SetAttribute(
        MIOPEN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY, MIOPEN_TYPE_DOUBLE, 1, &probability);
    rng->Finalize();

    miopenBackendDescriptor_t childDesc = rng->GetDescriptor();

    miopenBackendDescriptor_t seedDesc   = seed->GetDescriptor();
    miopenBackendDescriptor_t offsetDesc = offset->GetDescriptor();
    miopenBackendDescriptor_t outputDesc = output->GetDescriptor();

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
    rngOperation->AddRef(seed);
    rngOperation->AddRef(offset);
    rngOperation->AddRef(output);

    rngOperation->Finalize();

    return rngOperation;
}

class MhaForwardTest
{
public:
    void Run()
    {
        try
        {
            miopen::Handle& handle = get_handle();

            MakeRealTensors();
            MakeVirtualTensorsAndNodes();

            PrepareOpGraphAndEngines(handle);

            MakeVariantPackAndRun(handle);
            RunCPUverify(handle);
        }
        catch(const miopen::Exception& ex)
        {
            FAIL() << ex.what();
        }
    }

private:
    enum class GenerateType
    {
        DontGenerate,
        GenerateRandom,
        GenerateConstant
    };

    void MakeRealTensors()
    {
        // We use identifiers from Find 2.0 enum to have sopmething unique for the test purposes
        MakeAndAddRealTensorDescriptor(miopenTensorMhaQ, false, m_testN, m_testH, m_testS, m_testD);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaK,
                                       false,
                                       m_testN,
                                       m_testH,
                                       m_testS,
                                       m_testD,
                                       false); // no transpose for now

        MakeAndAddRealTensorDescriptor(miopenTensorMhaV, false, m_testN, m_testH, m_testS, m_testD);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleK);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleQ);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleV);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleS);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleS);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleO);

        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutProbability);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutSeed);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutOffset);

        // output real tensors
        MakeAndAddRealTensorDescriptor(miopenTensorMhaO, false, m_testN, m_testH, m_testS, m_testD);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxO);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxS);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaM, false, m_testN, m_testH, m_testS, 1);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaZInv, false, m_testN, m_testH, m_testS, 1);

        InitTensorValues();

        // get next value for the rest of the tensors (which don't have any particular enum value)
        m_nextTensorId++;
    }

    void InitTensorValues()
    {
        using namespace test::cpu;

        ScaledTensor Q = GenScaledTensor(m_testN, m_testH, m_testS, m_testD);
        ScaledTensor K = GenScaledTensor(m_testN, m_testH, m_testS, m_testD);
        ScaledTensor V = GenScaledTensor(m_testN, m_testH, m_testS, m_testD);

        for(auto& [k, v] : m_realTensorMap)
        {
            if(k == miopenTensorMhaQ)
            {
                v->Init(std::move(Q.mTensor));
            }
            else if(k == miopenTensorMhaDescaleQ)
            {
                v->Init(Q.mDescale);
            }
            else if(k == miopenTensorMhaK)
            {
                v->Init(std::move(K.mTensor));
            }
            else if(k == miopenTensorMhaDescaleK)
            {
                v->Init(K.mDescale);
            }
            else if(k == miopenTensorMhaV)
            {
                v->Init(std::move(V.mTensor));
            }
            else if(k == miopenTensorMhaDescaleV)
            {
                v->Init(V.mDescale);
            }
            else if(k == miopenTensorMhaScaleO || k == miopenTensorMhaScaleS ||
                    k == miopenTensorMhaDescaleS)
            {
                v->Init(1.0f);
            }
            else if(k == miopenTensorMhaDropoutProbability)
            {
                v->Init(m_bernulliProbability);
            }
            else if(k == miopenTensorMhaDropoutSeed || k == miopenTensorMhaDropoutOffset)
            {
                v->Init(1.0f);
            }
            else if(k == miopenTensorMhaM || k == miopenTensorMhaO || k == miopenTensorMhaZInv ||
                    k == miopenTensorMhaAmaxO || k == miopenTensorMhaAmaxS)
            {
                // these are outputs
                v->Init(0.0f);
            }
            else
            {
                FAIL() << "Uninitialized input or output: " << k;
            }
        }
    }

    void MakeVirtualTensorsAndNodes()
    {
        // virtual tensors
        auto tMM0 = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto pwS0 = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto pwS1 = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto pwS2 = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);

        auto tSub   = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto tExp   = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto tSum   = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, 1);
        auto tMult0 = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto tRnd   = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto tMult1 = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto pwS3   = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto pwS4   = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);

        auto tMM1 = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testD);
        auto pwS5 = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testD);
        auto pwS6 = MakeGapiTensorDesc(GetNextId(), true, m_testN, m_testH, m_testS, m_testD);

        // Node creation
        AddGraphNode(MakeMatmul(m_realTensorMap[miopenTensorMhaQ]->m_gapiDesc,
                                m_realTensorMap[miopenTensorMhaK]->m_gapiDesc,
                                tMM0));
        AddGraphNode(
            MakePointwise(MIOPEN_POINTWISE_IDENTITY, tMM0, nullptr, pwS0, m_attentionScale));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL,
                                   pwS0,
                                   m_realTensorMap[miopenTensorMhaDescaleQ]->m_gapiDesc,
                                   pwS1));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL,
                                   pwS1,
                                   m_realTensorMap[miopenTensorMhaDescaleK]->m_gapiDesc,
                                   pwS2));

        AddGraphNode(MakeReduction(
            MIOPEN_REDUCE_TENSOR_MAX, pwS2, m_realTensorMap[miopenTensorMhaM]->m_gapiDesc));
        AddGraphNode(MakePointwise(
            MIOPEN_POINTWISE_SUB, pwS2, m_realTensorMap[miopenTensorMhaM]->m_gapiDesc, tSub));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_EXP, tSub, DescriptorWrapperPtr(), tExp));
        AddGraphNode(MakeReduction(MIOPEN_REDUCE_TENSOR_ADD, tExp, tSum));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_RECIPROCAL,
                                   tSum,
                                   DescriptorWrapperPtr(),
                                   m_realTensorMap[miopenTensorMhaZInv]->m_gapiDesc));
        AddGraphNode(MakePointwise(
            MIOPEN_POINTWISE_MUL, tExp, m_realTensorMap[miopenTensorMhaZInv]->m_gapiDesc, tMult0));

        AddGraphNode(MakeReduction(
            MIOPEN_REDUCE_TENSOR_MAX, tMult0, m_realTensorMap[miopenTensorMhaAmaxS]->m_gapiDesc));

        AddGraphNode(MakeRNG(m_bernulliProbability,
                             m_realTensorMap[miopenTensorMhaDropoutSeed]->m_gapiDesc,
                             m_realTensorMap[miopenTensorMhaDropoutOffset]->m_gapiDesc,
                             tRnd));

        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL, tMult0, tRnd, tMult1));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL,
                                   tMult1,
                                   m_realTensorMap[miopenTensorMhaDropoutProbability]->m_gapiDesc,
                                   pwS3));
        AddGraphNode(MakePointwise(
            MIOPEN_POINTWISE_MUL, pwS3, m_realTensorMap[miopenTensorMhaScaleS]->m_gapiDesc, pwS4));

        AddGraphNode(MakeMatmul(pwS4, m_realTensorMap[miopenTensorMhaV]->m_gapiDesc, tMM1));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL,
                                   tMM1,
                                   m_realTensorMap[miopenTensorMhaDescaleS]->m_gapiDesc,
                                   pwS5));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL,
                                   pwS5,
                                   m_realTensorMap[miopenTensorMhaDescaleV]->m_gapiDesc,
                                   pwS6));
        AddGraphNode(MakeReduction(
            MIOPEN_REDUCE_TENSOR_MAX, pwS6, m_realTensorMap[miopenTensorMhaAmaxO]->m_gapiDesc));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL,
                                   pwS6,
                                   m_realTensorMap[miopenTensorMhaScaleO]->m_gapiDesc,
                                   m_realTensorMap[miopenTensorMhaO]->m_gapiDesc));
    }

    void PrepareOpGraphAndEngines(miopen::Handle& handle)
    {
        miopenHandle_t rawHandle = &handle;

        // Setup an operation graph
        DescriptorWrapperPtr operationGraph =
            MakeDescriptor(MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);

        miopenBackendDescriptor_t opGraphDesc = operationGraph->GetDescriptor();

        operationGraph->SetAttribute(
            MIOPEN_ATTR_OPERATIONGRAPH_HANDLE, MIOPEN_TYPE_HANDLE, 1, &rawHandle);

        std::vector<miopenBackendDescriptor_t> descs;
        descs.reserve(m_nodeVector.size());

        for(const DescriptorWrapperPtr& descWrapper : m_nodeVector)
        {
            descs.push_back(descWrapper->GetDescriptor());
        }

        operationGraph->SetAttribute(MIOPEN_ATTR_OPERATIONGRAPH_OPS,
                                     MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                                     descs.size(),
                                     descs.data());
        operationGraph->Finalize();

        // Setup an execution engine
        DescriptorWrapperPtr engine = MakeDescriptor(MIOPEN_BACKEND_ENGINE_DESCRIPTOR);

        miopenBackendDescriptor_t engineDesc = engine->GetDescriptor();

        engine->SetAttribute(
            MIOPEN_ATTR_ENGINE_OPERATION_GRAPH, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraphDesc);
        int64_t gidx = 0;
        engine->SetAttribute(MIOPEN_ATTR_ENGINE_GLOBAL_INDEX, MIOPEN_TYPE_INT64, 1, &gidx);
        engine->Finalize();

        // Setup an engine config
        DescriptorWrapperPtr engineConfig = MakeDescriptor(MIOPEN_BACKEND_ENGINECFG_DESCRIPTOR);

        miopenBackendDescriptor_t engineConfigDesc = engineConfig->GetDescriptor();

        engineConfig->SetAttribute(
            MIOPEN_ATTR_ENGINECFG_ENGINE, MIOPEN_TYPE_BACKEND_DESCRIPTOR, 1, &engineDesc);

        engineConfig->Finalize();

        // Setup a plan
        m_executionPlan = MakeDescriptor(MIOPEN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);

        m_executionPlan->SetAttribute(
            MIOPEN_ATTR_EXECUTION_PLAN_HANDLE, MIOPEN_TYPE_HANDLE, 1, &rawHandle);
        m_executionPlan->SetAttribute(MIOPEN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                      MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                                      1,
                                      &engineConfigDesc);
        m_executionPlan->Finalize();

        int64_t ws_count = 0;
        m_executionPlan->GetAttribute(MIOPEN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                      MIOPEN_TYPE_INT64,
                                      1,
                                      &ws_count,
                                      &m_workspaceSize);

        // save references to prevent them from being released
        m_executionPlan->AddRef(operationGraph);
        m_executionPlan->AddRef(engine);
        m_executionPlan->AddRef(engineConfig);
    }

    void MakeVariantPackAndRun(miopen::Handle& handle)
    {
        miopenHandle_t rawHandle = &handle;

        size_t numTensors = m_realTensorMap.size();

        std::vector<void*> devPtrs;
        devPtrs.reserve(numTensors);

        std::vector<int64_t> uids;
        uids.reserve(numTensors);

        for(const auto& it : m_realTensorMap)
        {
            it.second->m_gpuBuffer = handle.Write(it.second->m_tensor.data);
            devPtrs.push_back(it.second->m_gpuBuffer.get());

            uids.push_back(it.first);
        }

        Workspace workspace;
        workspace.resize(m_workspaceSize);

        DescriptorWrapperPtr varpack = MakeDescriptor(MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR);

        varpack->SetAttribute(MIOPEN_ATTR_VARIANT_PACK_DATA_POINTERS,
                              MIOPEN_TYPE_VOID_PTR,
                              numTensors,
                              devPtrs.data());
        varpack->SetAttribute(
            MIOPEN_ATTR_VARIANT_PACK_UNIQUE_IDS, MIOPEN_TYPE_INT64, numTensors, uids.data());

        auto ptr = workspace.ptr();

        varpack->SetAttribute(MIOPEN_ATTR_VARIANT_PACK_WORKSPACE, MIOPEN_TYPE_VOID_PTR, 1, &ptr);
        varpack->Finalize();

        m_executionPlan->AddRef(varpack);

        // Execute the plan with a variant pack.
        miopenStatus_t status = miopenBackendExecute(
            rawHandle, m_executionPlan->GetDescriptor(), varpack->GetDescriptor());
        CheckStatusAndThrow(status, "miopenBackendExecute failed!");
    }

    void RunCPUverify(miopen::Handle& handle)
    {
        auto softmaxRef  = tensor<float>{m_testN, m_testH, m_testS, m_testS};
        auto oDescRef    = tensor<float>{m_testN, m_testH, m_testS, m_testD};
        auto mDescRef    = tensor<float>{m_testN, m_testH, m_testS, 1};
        auto zInvDescRef = tensor<float>{m_testN, m_testH, m_testS, 1};
        float amaxSRef   = 0;
        float amaxORef   = 0;

        auto lookup = [this](const int64_t id) -> const tensor<float>& {
            auto it = m_realTensorMap.find(id);
            assert(it != m_realTensorMap.cend());
            return it->second->m_tensor;
        };

        test::cpu::MultiHeadAttentionfp8(
            lookup(miopenTensorMhaQ),
            lookup(miopenTensorMhaK),
            lookup(miopenTensorMhaV),
            softmaxRef,
            mDescRef,
            zInvDescRef,
            lookup(miopenTensorMhaDescaleQ)[0],
            lookup(miopenTensorMhaDescaleK)[0],
            lookup(miopenTensorMhaDescaleV)[0],
            lookup(miopenTensorMhaDescaleS)[0],
            lookup(miopenTensorMhaScaleS)[0],
            lookup(miopenTensorMhaScaleO)[0],
            lookup(miopenTensorMhaDropoutProbability)[0],
            static_cast<uint64_t>(lookup(miopenTensorMhaDropoutSeed)[0]),
            static_cast<uint64_t>(lookup(miopenTensorMhaDropoutOffset)[0]),
            amaxSRef,
            amaxORef,
            oDescRef);

        auto GetResult = [&](const int64_t& id) -> const tensor<float>& {
            auto it = m_realTensorMap.find(id);
            assert(it != m_realTensorMap.cend());

            TensorDataPtr ptr  = it->second;
            ptr->m_tensor.data = handle.Read<float>(ptr->m_gpuBuffer, ptr->m_tensor.data.size());
            return ptr->m_tensor;
        };

        const double errorThreshold = 5e-6;

        const auto& resAmaxS = GetResult(miopenTensorMhaAmaxS);
        auto amaxSAbsDiff    = std::abs(amaxSRef - resAmaxS[0]);
        EXPECT_LT(amaxSAbsDiff, errorThreshold)
            << " ref: " << amaxSRef << " result: " << resAmaxS[0];

        const auto& resAmaxO = GetResult(miopenTensorMhaAmaxO);
        auto amaxOAbsDiff    = std::abs(amaxORef - resAmaxO[0]);
        EXPECT_LT(amaxOAbsDiff, errorThreshold)
            << " ref: " << amaxORef << " result: " << resAmaxO[0];

        double mError = miopen::rms_range(mDescRef, GetResult(miopenTensorMhaM));
        EXPECT_LT(mError, errorThreshold);

        double zInvError = miopen::rms_range(zInvDescRef, GetResult(miopenTensorMhaZInv));
        EXPECT_LT(zInvError, errorThreshold);

        double oError = miopen::rms_range(oDescRef, GetResult(miopenTensorMhaO));
        EXPECT_LT(oError, errorThreshold);
    }

    void AddGraphNode(DescriptorWrapperPtr node) { m_nodeVector.push_back(node); }

    // For real tensors we use values from enum miopenTensorArgumentId_t (miopen.h) jsut to have
    // some unique and named values. For virtual tensors we use identifiers starting from "max id
    // from real tensors" + 1
    void MakeAndAddRealTensorDescriptor(int64_t tensorId,
                                        bool isVirtual = false,
                                        int64_t n      = 1,
                                        int64_t h      = 1,
                                        int64_t s      = 1,
                                        int64_t d      = 1,
                                        bool transpose = false)
    {
        DescriptorWrapperPtr realTensorGapiDesc =
            MakeGapiTensorDesc(tensorId, isVirtual, n, h, s, d, transpose);

        TensorDataPtr tensorDataPtr = std::make_shared<TensorData>();
        tensorDataPtr->m_gapiDesc   = realTensorGapiDesc;
        tensorDataPtr->m_tensor = tensor<float>{n, h, s, d};

        m_realTensorMap[tensorId] = tensorDataPtr;

        // Here we memorize maximum id from real tensors set -to start from this value + 1 for
        // virtual tensors set.
        m_nextTensorId = std::max(m_nextTensorId, tensorId);
    }

    // just a simple id generator, might be redone if necessary
    int64_t GetNextId() { return m_nextTensorId++; }

private:
    const int64_t m_testN = 2;
    const int64_t m_testH = 4;
    const int64_t m_testS = 8;
    const int64_t m_testD = 16;

    double m_bernulliProbability = 0.0;

    // to be fed into OperationGraph
    std::vector<DescriptorWrapperPtr> m_nodeVector;

    std::map<int64_t, TensorDataPtr> m_realTensorMap;

    int64_t m_nextTensorId = 0;

    float m_attentionScale = 1.0f;

    int64_t m_workspaceSize = 0;

    DescriptorWrapperPtr m_executionPlan;
};

TEST(TestCGraphApi, MhaForward)
{
    MhaForwardTest forwardTest;

    forwardTest.Run();
}
