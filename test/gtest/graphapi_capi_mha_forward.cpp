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
    if(status == miopenStatusSuccess)
    {
        return status;
    }

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
                                          int64_t d      = 1,
                                          bool transpose = false)
{
    DescriptorWrapperPtr descWrapperPtr = MakeDescriptor(MIOPEN_BACKEND_TENSOR_DESCRIPTOR);

    miopenDataType_t dtype = miopenFloat;

    descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &dtype);

    std::vector<int64_t> dims    = {n, h, s, d};
    std::vector<int64_t> strides = {1, n, n*h, n*h*s};

    if (transpose)
    {
        dims = {n, h, d, s};       
        strides = {1, n, n*h*s, n*h};
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
                                   DescriptorWrapperPtr output)
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

    rng->SetAttribute(MIOPEN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY, MIOPEN_TYPE_DOUBLE, 1, &probability);
    rng->Finalize();

    miopenBackendDescriptor_t childDesc = rng->GetDescriptor();

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
            MakeRealTensors();
            MakeVirtualTensorsAndNodes();

            // Run OperationGraph
            RunTheGraph();
        }
        catch(const miopen::Exception& ex)
        {
            FAIL() << ex.what();
        }
    }
 
private:
    void MakeRealTensors()
    {
        // We use identifiers from Find 2.0 enum to have sopmething unique for the test purposes
        MakeAndAddRealTensorDescriptor(miopenTensorMhaQ, false, m_testN, m_testH, m_testS, m_testD);        
        MakeAndAddRealTensorDescriptor(miopenTensorMhaK, false, m_testN, m_testH, m_testS, m_testD, true); // transpose this tensor
        MakeAndAddRealTensorDescriptor(miopenTensorMhaV, false, m_testN, m_testH, m_testS, m_testD);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleK);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleQ);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleV);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDescaleS);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleS);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaScaleO);

        // we have only double input for probability in RNG node (m_bernulliProbability), however
        // for pointwise pwScale3 we need to have it as a tensor, so we need to have these values synced
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutProbability);

        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutSeed);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaDropoutOffset);

        // output real tensors
        MakeAndAddRealTensorDescriptor(miopenTensorMhaO, false, m_testN, m_testH, m_testS, m_testD);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxO);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaAmaxS);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaM, false, m_testN, m_testH, m_testS, 1);
        MakeAndAddRealTensorDescriptor(miopenTensorMhaZInv, false, m_testN, m_testH, m_testS, 1);

        // This attention scale param is just a float in Find 2.0, so no particular enum value, just use m_nextTensorId, it equals to max value + 1 at this point
        // If it's needed we could save this value here
        m_attentionScaleId = GetNextId();
        MakeAndAddRealTensorDescriptor(m_attentionScaleId);
    }

    void MakeVirtualTensorsAndNodes()
    {
        // virtual tensors
        auto tMM0 = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto pwS0 = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto pwS1 = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto pwS2 = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);

        auto tSub   = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto tExp   = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto tSum   = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, 1);
        auto tMult0 = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto tRnd   = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto tMult1 = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto pwS3   = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);
        auto pwS4   = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testS);

        auto tMM1 = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testD);
        auto pwS5 = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testD);
        auto pwS6 = MakeTensorDescriptor(GetNextId(), true, m_testN, m_testH, m_testS, m_testD);

        // Node creation
        AddGraphNode(MakeMatmul(m_realTensorMap[miopenTensorMhaQ], m_realTensorMap[miopenTensorMhaK], tMM0));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL, tMM0, m_realTensorMap[m_attentionScaleId], pwS0));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL, pwS0, m_realTensorMap[miopenTensorMhaDescaleQ], pwS1));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL, pwS1, m_realTensorMap[miopenTensorMhaDescaleK], pwS2));

        AddGraphNode(MakeReduction(MIOPEN_REDUCE_TENSOR_MAX, pwS2, m_realTensorMap[miopenTensorMhaM]));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_SUB, pwS2, m_realTensorMap[miopenTensorMhaM], tSub));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_EXP, tSub, DescriptorWrapperPtr(), tExp));
        AddGraphNode(MakeReduction(MIOPEN_REDUCE_TENSOR_ADD, tExp, tSum));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_EXP, tSum, DescriptorWrapperPtr(), m_realTensorMap[miopenTensorMhaZInv]));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL, tExp, m_realTensorMap[miopenTensorMhaZInv], tMult0));

        AddGraphNode(MakeReduction(MIOPEN_REDUCE_TENSOR_MAX, tMult0, m_realTensorMap[miopenTensorMhaAmaxS]));

        AddGraphNode(MakeRNG(m_bernulliProbability, m_realTensorMap[miopenTensorMhaDropoutSeed], m_realTensorMap[miopenTensorMhaDropoutOffset], tRnd));

        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL, tMult0, tRnd, tMult1));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL, tMult1, m_realTensorMap[miopenTensorMhaDropoutProbability], pwS3));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL, pwS3, m_realTensorMap[miopenTensorMhaScaleS], pwS4));

        AddGraphNode(MakeMatmul(pwS4, m_realTensorMap[miopenTensorMhaV], tMM1));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL, tMM1, m_realTensorMap[miopenTensorMhaDescaleS], pwS5));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL, pwS5, m_realTensorMap[miopenTensorMhaDescaleV], pwS6));
        AddGraphNode(MakeReduction(MIOPEN_REDUCE_TENSOR_MAX, pwS6, m_realTensorMap[miopenTensorMhaAmaxO]));
        AddGraphNode(MakePointwise(MIOPEN_POINTWISE_MUL, pwS6, m_realTensorMap[miopenTensorMhaScaleO], m_realTensorMap[miopenTensorMhaO]));
    }

    void RunTheGraph()
    {
        DescriptorWrapperPtr operationGraph = MakeDescriptor(MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);

    }

    void AddGraphNode(DescriptorWrapperPtr node)
    {
        m_nodeVector.push_back(node);
    }


    // For real tensors we use values from enum miopenTensorArgumentId_t (miopen.h) jsut to have some unique and named values.
    // For virtual tensors we use identifiers starting from "max id from real tensors" + 1
    void MakeAndAddRealTensorDescriptor( int64_t tensorId,
                                bool isVirtual = false,
                                int64_t n      = 1,
                                int64_t h      = 1,
                                int64_t s      = 1,
                                int64_t d      = 1,
                                bool transpose = false)
    {
        auto realTensor = MakeTensorDescriptor(tensorId, isVirtual, n, h, s, d, transpose);
        m_realTensorMap[tensorId] = realTensor;

        // Here we memorize maximum id from real tensors set -to start from this value + 1 for virtual tensors set.
        m_nextTensorId = std::max(m_nextTensorId, tensorId) + 1;
    }

    // just a simple id generator, might be redone if necessary
    int64_t GetNextId()
    {
        return m_nextTensorId++;
    }

private:
    const int64_t m_testN = 2;
    const int64_t m_testH = 4;
    const int64_t m_testS = 8;
    const int64_t m_testD = 16;    

    double m_bernulliProbability = 0.5;

    // to be fed into OperationGraph
    std::vector<DescriptorWrapperPtr> m_nodeVector;

    std::map<int64_t, DescriptorWrapperPtr> m_realTensorMap;

    int64_t m_nextTensorId = 0;

    // In Find 2.0 attention scale is just a float value, lets save the id here
    int64_t m_attentionScaleId = 0;
};

TEST(TestCGraphApi, MhaForward)
{
    MhaForwardTest forwardTest;

    forwardTest.Run();
}
