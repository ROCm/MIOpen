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

#include "gtest_common.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "../workspace.hpp"
#include "gtest/mha_helper.hpp"

namespace capi_test_mha_common {

class DescriptorWrapper;
typedef std::shared_ptr<DescriptorWrapper> DescriptorWrapperPtr;

class Helpers
{
public:
    static miopenStatus_t CheckStatusAndThrow(miopenStatus_t status,
                                              const std::string& msg,
                                              bool addStatusToMessage = true)
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
};

class DescriptorWrapper
{
public:
    DescriptorWrapper(miopenBackendDescriptorType_t descriptorType)
        : m_descriptorType(descriptorType), m_descriptor(nullptr)
    {
        Helpers::CheckStatusAndThrow(miopenBackendCreateDescriptor(descriptorType, &m_descriptor),
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

        Helpers::CheckStatusAndThrow(status,
                                     "miopenBackendSetAttribute failed: descriptorType = " +
                                         std::to_string(m_descriptorType) +
                                         ", attributeName=" + std::to_string(attributeName) +
                                         ", attributeType=" + std::to_string(attributeType));
    }

    void GetAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) const
    {
        miopenStatus_t status = miopenBackendGetAttribute(m_descriptor,
                                                          attributeName,
                                                          attributeType,
                                                          requestedElementCount,
                                                          elementCount,
                                                          arrayOfElements);

        Helpers::CheckStatusAndThrow(
            status,
            "miopenBackendGetAttribute failed: descriptorType = " +
                std::to_string(m_descriptorType) + ", attributeName=" +
                std::to_string(attributeName) + ", attributeType=" + std::to_string(attributeType) +
                ", requestedElementCount=" + std::to_string(requestedElementCount));
    }

    void Finalize()
    {
        Helpers::CheckStatusAndThrow(miopenBackendFinalize(m_descriptor),
                                     "miopenBackendFinalize failed: descriptorType = " +
                                         std::to_string(m_descriptorType));
    }

    void AddRef(DescriptorWrapperPtr refToKeep) { m_refsToKeep.push_back(refToKeep); }

    miopenBackendDescriptor_t GetDescriptor() const { return m_descriptor; }
    miopenBackendDescriptorType_t GetDescriptorType() const { return m_descriptorType; }

    // helper for tensor
    void GetTensorDims(std::vector<size_t>& dims) const
    {
        assert(m_descriptorType == MIOPEN_BACKEND_TENSOR_DESCRIPTOR);

        dims.resize(4);
        int64_t count = 0;
        GetAttribute(
            MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT64, dims.size(), &count, dims.data());
    }

private:
    miopenBackendDescriptorType_t m_descriptorType;
    miopenBackendDescriptor_t m_descriptor;

    std::vector<DescriptorWrapperPtr> m_refsToKeep;
};

using TensorVariant = std::variant<tensor<float>, tensor<float8>, tensor<bfloat8>, tensor<int64_t>>;

template <typename T>
tensor<T>& GetTensor(TensorVariant& var)
{
    return std::get<tensor<T>>(var);
}

struct TensorData
{
    DescriptorWrapperPtr m_gapiDesc;
    TensorVariant m_tensorVariant;
    miopen::Allocator::ManageDataPtr m_gpuBuffer;

    template <typename T>
    void InitAndWriteToGPU(miopen::Handle& handle, tensor<T>&& tensor)
    {
        m_tensorVariant = std::move(tensor);
        m_gpuBuffer     = handle.Write(GetTensor<T>(m_tensorVariant).data);
    }

    template <typename T>
    void InitAndWriteToGPU(miopen::Handle& handle, T val)
    {
        GetTensor<T>(m_tensorVariant).generate([=](auto...) { return val; });
        m_gpuBuffer = handle.Write(GetTensor<T>(m_tensorVariant).data);
    }
};

typedef std::shared_ptr<TensorData> TensorDataPtr;

template <typename T>
miopenDataType_t GetMainType()
{
    if(std::is_same_v<T, float8>)
    {
        return miopenFloat8;
    }
    else if(std::is_same_v<T, float>)
    {
        return miopenFloat;
    }

    assert(false);
    return miopenFloat;
}

class MhaCommonTest : public testing::TestWithParam<std::tuple<int, int, int, int, float>>
{
public:
    void Run()
    {
        auto [n, h, s, d, p] = GetParam();

        m_testN               = n;
        m_testH               = h;
        m_testS               = s;
        m_testD               = d;
        m_bernulliProbability = p;

        miopen::Handle& handle = get_handle();

        if((p > 0.0f) && (s % handle.GetWavefrontWidth() != 0))
        {
            GTEST_SKIP() << "CPU Dropout currently supprorts only fully occupied warps. n=" << n
                         << ", h=" << h << ", s=" << s << ", d=" << d << ", bernulli p=" << p;
        }

        try
        {
            MakeRealTensorsAndFillData(handle);
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

protected:
    virtual void MakeRealTensorsAndFillData(miopen::Handle& handle) = 0;

    virtual void MakeVirtualTensorsAndNodes() = 0;

    virtual void PrepareOpGraphAndEngines(miopen::Handle& handle)
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

    virtual void MakeVariantPackAndRun(miopen::Handle& handle)
    {
        miopenHandle_t rawHandle = &handle;

        size_t numTensors = m_realTensorMap.size();

        std::vector<void*> devPtrs;
        devPtrs.reserve(numTensors);

        std::vector<int64_t> uids;
        uids.reserve(numTensors);

        for(const auto& it : m_realTensorMap)
        {
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
        Helpers::CheckStatusAndThrow(status, "miopenBackendExecute failed!");
    }

    template <typename ResultT>
    tensor<ResultT>& GetResult(const int64_t& id, miopen::Handle& handle)
    {
        auto it = m_realTensorMap.find(id);
        assert(it != m_realTensorMap.cend());

        TensorDataPtr ptr    = it->second;
        tensor<ResultT>& ret = GetTensor<ResultT>(ptr->m_tensorVariant);

        ret.data = handle.Read<ResultT>(ptr->m_gpuBuffer, ret.data.size());
        return ret;
    };

    virtual void RunCPUverify(miopen::Handle& handle) = 0;

    // just a simple id generator, might be redone if necessary
    int64_t GetNextId() { return m_nextTensorId++; }

    static DescriptorWrapperPtr MakeDescriptor(miopenBackendDescriptorType_t descriptorType)
    {
        return std::make_shared<DescriptorWrapper>(descriptorType);
    }

    DescriptorWrapperPtr MakeMatmul(DescriptorWrapperPtr tensor1,
                                    DescriptorWrapperPtr tensor2,
                                    DescriptorWrapperPtr output)
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

        AddGraphNode(matmulOperation);

        return matmulOperation;
    }

    // if output is nullptr, let's automatically create virtual tensor with size of input tensor
    DescriptorWrapperPtr MakePointwise(miopenPointwiseMode_t mode,
                                       DescriptorWrapperPtr tensor1,
                                       DescriptorWrapperPtr tensor2,
                                       DescriptorWrapperPtr& output,
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

        if(output == nullptr)
        {
            std::vector<size_t> dims;
            tensor1->GetTensorDims(dims);

            output = MakeGapiVirtualTensorDesc(dims[0], dims[1], dims[2], dims[3]);
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
            pointwiseOperation->SetAttribute(MIOPEN_ATTR_OPERATION_POINTWISE_BDESC,
                                             MIOPEN_TYPE_BACKEND_DESCRIPTOR,
                                             1,
                                             &tensor2Desc);

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

        AddGraphNode(pointwiseOperation);

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

        AddGraphNode(reductionOperation);

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

        AddGraphNode(rngOperation);

        return rngOperation;
    }

    DescriptorWrapperPtr MakeGapiVirtualTensorDesc(size_t n               = 1,
                                                   size_t h               = 1,
                                                   size_t s               = 1,
                                                   size_t d               = 1,
                                                   miopenDataType_t dtype = miopenFloat,
                                                   bool transpose         = false)
    {
        return MakeGapiTensorDesc(GetNextId(), true, n, h, s, d, dtype, transpose);
    }

    // For real tensors we use values from enum miopenTensorArgumentId_t (miopen.h) jsut to have
    // some unique and named values. For virtual tensors we use identifiers starting from "max id
    // from real tensors" + 1
    TensorData& MakeAndAddRealTensorDescriptor(int64_t tensorId,
                                               size_t n               = 1,
                                               size_t h               = 1,
                                               size_t s               = 1,
                                               size_t d               = 1,
                                               miopenDataType_t dtype = miopenFloat,
                                               bool transpose         = false)
    {
        DescriptorWrapperPtr realTensorGapiDesc =
            MakeGapiTensorDesc(tensorId, false, n, h, s, d, dtype, transpose);

        TensorDataPtr tensorDataPtr = std::make_shared<TensorData>();
        tensorDataPtr->m_gapiDesc   = realTensorGapiDesc;

        if(dtype == miopenFloat)
        {
            tensorDataPtr->m_tensorVariant = tensor<float>{n, h, s, d};
        }
        else if(dtype == miopenInt64)
        {
            tensorDataPtr->m_tensorVariant = tensor<int64_t>{n, h, s, d};
        }
        else if(dtype == miopenFloat8)
        {
            tensorDataPtr->m_tensorVariant = tensor<float8>{n, h, s, d};
        }
        else
        {
            assert(false);
        }

        m_realTensorMap[tensorId] = tensorDataPtr;

        // Here we memorize maximum id from real tensors set -to start from this value + 1 for
        // virtual tensors set.
        m_nextTensorId = std::max(m_nextTensorId, tensorId);

        return *tensorDataPtr;
    }

    DescriptorWrapperPtr MakeGapiTensorDesc(int64_t uniqueId,
                                            bool isVirtual         = false,
                                            size_t n               = 1,
                                            size_t h               = 1,
                                            size_t s               = 1,
                                            size_t d               = 1,
                                            miopenDataType_t dtype = miopenFloat,
                                            bool transpose         = false)
    {
        DescriptorWrapperPtr descWrapperPtr = MakeDescriptor(MIOPEN_BACKEND_TENSOR_DESCRIPTOR);

        descWrapperPtr->SetAttribute(
            MIOPEN_ATTR_TENSOR_DATA_TYPE, MIOPEN_TYPE_DATA_TYPE, 1, &dtype);

        std::vector<int64_t> dims = {n, h, s, d};

        miopen::TensorDescriptor td(dtype, {n, h, s, d});
        const std::vector<std::size_t>& tdStrides = td.GetStrides();

        std::vector<int64_t> strides(tdStrides.size());
        std::copy_n(tdStrides.begin(), tdStrides.size(), strides.begin());

        if(transpose)
        {
            std::swap(dims[2], dims[3]);
            std::swap(strides[2], strides[3]);
        }

        // commented this out as Not Implemented
        // int64_t alignment = 4;

        descWrapperPtr->SetAttribute(
            MIOPEN_ATTR_TENSOR_DIMENSIONS, MIOPEN_TYPE_INT64, 4, dims.data());

        descWrapperPtr->SetAttribute(
            MIOPEN_ATTR_TENSOR_STRIDES, MIOPEN_TYPE_INT64, 4, strides.data());
        descWrapperPtr->SetAttribute(MIOPEN_ATTR_TENSOR_UNIQUE_ID, MIOPEN_TYPE_INT64, 1, &uniqueId);

        // commented this out as Not Implemented
        // descWrapperPtr->SetAttribute(
        //    MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT, MIOPEN_TYPE_INT64, 1, &alignment);

        descWrapperPtr->SetAttribute(
            MIOPEN_ATTR_TENSOR_IS_VIRTUAL, MIOPEN_TYPE_BOOLEAN, 1, &isVirtual);
        descWrapperPtr->Finalize();

        return descWrapperPtr;
    }

protected:
    size_t m_testN = 0;
    size_t m_testH = 0;
    size_t m_testS = 0;
    size_t m_testD = 0;

    float m_bernulliProbability = 0.0f;
    float m_attentionScale      = 1.0f;

    // to be fed into OperationGraph
    std::vector<DescriptorWrapperPtr> m_nodeVector;

    std::map<int64_t, TensorDataPtr> m_realTensorMap;

    int64_t m_nextTensorId  = 0;
    int64_t m_workspaceSize = 0;

    DescriptorWrapperPtr m_executionPlan;

private:
    void AddGraphNode(DescriptorWrapperPtr node) { m_nodeVector.push_back(node); }
};

} // namespace capi_test_mha_common
