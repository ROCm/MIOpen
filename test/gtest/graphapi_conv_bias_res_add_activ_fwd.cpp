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
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/env.hpp>

#include <miopen/graphapi/convolution.hpp>
#include <miopen/graphapi/execution_plan.hpp>
#include <miopen/graphapi/matmul.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <miopen/graphapi/pointwise.hpp>
#include <miopen/graphapi/rng.hpp>
#include <miopen/graphapi/util.hpp>
#include <miopen/graphapi/variant_pack.hpp>

#include "tensor_util.hpp"
#include "get_handle.hpp"

#include "conv3d_test_case.hpp"

namespace gr = miopen::graphapi;

namespace conv_graph_api_test {

bool TestIsApplicable() { return true; }

std::vector<Conv3DTestCase> ConvTestConfigs()
{ //         g, n, c, d,  h,  w, k,  z, y, x, pad_x pad_y pad_z stri_x stri_y stri_z dia_x dia_y
  //         dia_z
    return {{1, 1, 4, 14, 11, 1, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 1, 1, 1, 4, 4, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 1, 1, 8, 8, 8, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {1, 1, 1, 8, 8, 8, 1, 2, 2, 2, 0, 0, 0, 2, 2, 2, 1, 1, 1, miopenConvolution},
            {2, 8, 8, 12, 14, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {4, 8, 8, 11, 11, 11, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {6, 8, 18, 11, 11, 11, 18, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {8, 8, 8, 11, 11, 11, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {4, 8, 4, 11, 11, 11, 8, 3, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution},
            {2, 8, 2, 11, 11, 11, 2, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, miopenConvolution}};
}

template <typename T>
class GPU_ConvBiasResAddActivation_fwd
    : public ::testing::TestWithParam<
          std::tuple<Conv3DTestCase, float, float, miopenTensorLayout_t>>
{

    static gr::OperationPointwise* MakeAddNode(gr::AutoDeleteAllocator& nodeAllocator,
                                               miopenDataType_t dataType,
                                               gr::Tensor* x,
                                               gr::Tensor* b,
                                               gr::Tensor* y,
                                               gr::OperationPointwise::Alpha alpha1 = 1.0f,
                                               gr::OperationPointwise::Alpha alpha2 = 1.0f)
    {
        auto add = nodeAllocator.allocate(gr::PointwiseBuilder{}
                                              .setMode(MIOPEN_POINTWISE_ADD)
                                              .setMathPrecision(dataType)
                                              .build());

        return nodeAllocator.allocate(gr::OperationPointwiseBuilder{}
                                          .setPointwise(add)
                                          .setX(x)
                                          .setB(b)
                                          .setAlpha1(alpha1)
                                          .setAlpha2(alpha2)
                                          .setY(y)
                                          .build());
    }

    template <typename DataT>
    static std::vector<int64_t> Convert(const std::vector<DataT>& value)
    {
        return {value.cbegin(), value.cend()};
    }

    struct TensorData
    {
        gr::Tensor* mTensPtr;
        tensor<T> mCpuTensor;
        miopen::Allocator::ManageDataPtr mGpuBuf;

        explicit TensorData(gr::Tensor* tensorPtr, const miopen::TensorDescriptor& desc)
            : mTensPtr(tensorPtr), mCpuTensor(tensor<T>(desc))
        {
            assert(mTensPtr);
            assert(mTensPtr->GetType() == miopen_type<T>());
        }

        template <typename G>
        void Init(G generator)
        {
            mCpuTensor.generate(generator);
            auto& handle = get_handle();
            mGpuBuf      = handle.Write(mCpuTensor.data);
        }

        void CopyBack()
        {
            auto& handle = get_handle();
            handle.ReadToVec(mGpuBuf, mCpuTensor.data);
        }

        TensorData(const TensorData&) = delete;
        TensorData(TensorData&&)      = default;

        TensorData& operator=(const TensorData&) = delete;
        TensorData& operator=(TensorData&&) = default;

        ~TensorData() = default;
    };

    struct GraphTensorAllocator
    {
        gr::AutoDeleteAllocator mAlloc;
        std::unordered_map<std::string, TensorData> mFilledTensors;

        GraphTensorAllocator() {}

        template <bool isVirtual>
        gr::Tensor* MakeTensor(std::string_view name,
                               miopenDataType_t dataType,
                               const miopen::TensorDescriptor& tensorDesc)
        {
            auto ptr = mAlloc.allocate(gr::makeTensor<isVirtual>(
                name, dataType, tensorDesc.GetLengths(), tensorDesc.GetStrides()));
            if constexpr(!isVirtual)
            {
                mFilledTensors.emplace(std::string(name), TensorData(ptr, tensorDesc));
            }
            return ptr;
        }

        TensorData& LookupTensorData(const std::string& name)
        {
            auto it = mFilledTensors.find(name);
            assert(it != mFilledTensors.end());
            return it->second;
        }

        gr::VariantPack MakeVariantPack(void* workspacePtr)
        {
            std::vector<int64_t> tensorIds;
            std::vector<void*> gpuPtrs;

            for(const auto& [k, v] : mFilledTensors)
            {
                tensorIds.emplace_back(v.mTensPtr->getId());
                assert(v.mGpuBuf.get());
                gpuPtrs.emplace_back(v.mGpuBuf.get());
            }

            return gr::VariantPackBuilder()
                .setTensorIds(tensorIds)
                .setDataPointers(gpuPtrs)
                .setWorkspace(workspacePtr)
                .build();
        }

        GraphTensorAllocator(const GraphTensorAllocator&) = delete;
        GraphTensorAllocator(GraphTensorAllocator&&)      = default;

        GraphTensorAllocator& operator=(const GraphTensorAllocator&) = delete;
        GraphTensorAllocator& operator=(GraphTensorAllocator&&) = default;

        ~GraphTensorAllocator() = default;
    };

public:
    void SetUp() override
    {
        if(!TestIsApplicable())
        {
            GTEST_SKIP();
        }

        prng::reset_seed();
    }

    void Run()
    {
        Conv3DTestCase convConfig;
        float alpha1 = 1.0f;
        float alpha2 = 1.0f;
        miopenTensorLayout_t tensorLayout;
        auto dataType = miopen_type<T>();

        std::tie(convConfig, alpha1, alpha2, tensorLayout) = GetParam();

        gr::OpGraphBuilder graphBuilder;
        gr::AutoDeleteAllocator graphNodeAllocator;
        GraphTensorAllocator graphTensorAllocator;

        // Graph of Y = activation(Conv(X,W) * alpha1 + Z * alpha2 + B)
        const std::string convInputName        = "X";
        const std::string convWeightName       = "W";
        const std::string convOutputName       = "Tmp0";
        const std::string addInputName         = "Z";
        const std::string addOutputName        = "Tmp1";
        const std::string biasInputName        = "B";
        const std::string biasOutputName       = "Tmp2";
        const std::string activationOutputName = "Y";

        auto convInput = graphTensorAllocator.template MakeTensor<false>(
            convInputName,
            dataType,
            miopen::TensorDescriptor{dataType, tensorLayout, convConfig.GetInput()});
        auto convWeight = graphTensorAllocator.template MakeTensor<false>(
            convWeightName,
            dataType,
            miopen::TensorDescriptor{dataType, tensorLayout, convConfig.GetWeights()});
        auto convInputDescription = convConfig.GetConv();
        auto& convInputData       = graphTensorAllocator.LookupTensorData(convInputName);
        auto& convWeightData      = graphTensorAllocator.LookupTensorData(convWeightName);
        auto convOutputDesc       = convInputDescription.GetForwardOutputTensor(
            convInputData.mCpuTensor.desc, convWeightData.mCpuTensor.desc, dataType);
        auto convOutput = graphTensorAllocator.template MakeTensor<true>(
            convOutputName, dataType, convOutputDesc);

        gr::Convolution* convolution = nullptr;
        ASSERT_NO_THROW(
            convolution = graphNodeAllocator.allocate(
                gr::ConvolutionBuilder{}
                    .setCompType(dataType)
                    .setMode(convConfig.conv_mode)
                    .setSpatialDims(convInputDescription.GetSpatialDimension())
                    .setDilations(Convert(convInputDescription.GetConvDilations()))
                    .setFilterStrides(Convert(convInputDescription.GetConvStrides()))
                    .setPrePaddings(Convert(convInputDescription.GetConvPads()))
                    .setPostPaddings(Convert(convInputDescription.GetTransposeConvPads()))
                    .build()));

        ASSERT_NO_THROW(graphBuilder.addNode(
            graphNodeAllocator.allocate(gr::OperationConvolutionForwardBuilder()
                                            .setConvolution(convolution)
                                            .setX(convInput)
                                            .setY(convOutput)
                                            .setW(convWeight)
                                            .setAlpha(1.0)
                                            .setBeta(0)
                                            .build())));

        auto addInput =
            graphTensorAllocator.template MakeTensor<false>(addInputName, dataType, convOutputDesc);
        auto addOutput =
            graphTensorAllocator.template MakeTensor<true>(addOutputName, dataType, convOutputDesc);

        ASSERT_NO_THROW(graphBuilder.addNode(MakeAddNode(
            graphNodeAllocator, dataType, convOutput, addInput, addOutput, alpha1, alpha2)));

        miopen::TensorDescriptor biasInputTensorDesc{
            dataType, tensorLayout, {1, convConfig.k, 1, 1, 1}};
        auto biasInput = graphTensorAllocator.template MakeTensor<false>(
            biasInputName, dataType, biasInputTensorDesc);
        auto biasOutput = graphTensorAllocator.template MakeTensor<true>(
            biasOutputName, dataType, convOutputDesc);

        ASSERT_NO_THROW(graphBuilder.addNode(
            MakeAddNode(graphNodeAllocator, dataType, addOutput, biasInput, biasOutput)));

        auto activationOutput = graphTensorAllocator.template MakeTensor<false>(
            activationOutputName, dataType, convOutputDesc);

        gr::Pointwise* activation = nullptr;
        ASSERT_NO_THROW(activation =
                            graphNodeAllocator.allocate(gr::PointwiseBuilder{}
                                                            .setMode(MIOPEN_POINTWISE_RELU_FWD)
                                                            .setMathPrecision(dataType)
                                                            .build()));
        ASSERT_NO_THROW(
            graphBuilder.addNode(graphNodeAllocator.allocate(gr::OperationPointwiseBuilder{}
                                                                 .setPointwise(activation)
                                                                 .setX(biasOutput)
                                                                 .setY(activationOutput)
                                                                 .build())));

        auto& addTensorData  = graphTensorAllocator.LookupTensorData(addInputName);
        auto& biasTensorData = graphTensorAllocator.LookupTensorData(biasInputName);

        auto genValue = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };

        convInputData.Init(genValue);
        convWeightData.Init(genValue);
        addTensorData.Init(genValue);
        biasTensorData.Init(genValue);

        auto& handle   = get_handle();
        auto handlePtr = static_cast<miopenHandle_t>(&handle);
        graphBuilder.setHandle(handlePtr);
        gr::OpGraph graph;
        ASSERT_NO_THROW(graph = std::move(graphBuilder).build());
        auto engines = gr::findEngines(&graph);

        // No engines exist currently that can handle graphAPI fused convolution
        EXPECT_EQUAL(engines.size(), 0);

        /// \todo uncomment below to execute plan, and run verification once engine implemented
        /// --BrianHarrisonAMD July 2024
        // ASSERT_GT(engines.size(), 0);

        // gr::EngineCfg engineConfig;
        // ASSERT_NO_THROW(engineConfig = gr::EngineCfgBuilder().setEngine(engines[0]).build());

        // gr::ExecutionPlan plan;
        // ASSERT_NO_THROW(plan =
        // gr::ExecutionPlanBuilder().setEngineCfg(engineConfig).setHandle(handlePtr).build());

        // Workspace ws(plan.getWorkspaceSize());

        // gr::VariantPack variantPack;
        // ASSERT_NO_THROW(variantPack = graphTensorAllocator.MakeVariantPack(ws.ptr()));

        // ASSERT_NO_THROW(plan.execute(handlePtr, variantPack));

        // // Reference implementation for Y = activation(Conv(X,W) * alpha1 + Z * alpha2 + B)
        // auto referenceOutput = tensor<T>(convOutputDesc);
        // referenceOutput = ref_conv_fwd(convInputData.mCpuTensor, convWeightData.mCpuTensor,
        // referenceOutput, convInputDescription);

        // auto& z = addTensorData.mCpuTensor;
        // auto& bias = biasTensorData.mCpuTensor;

        // referenceOutput.par_for_each([&](auto n, auto k, auto... dhw) {
        //     auto& o = referenceOutput(n, k, dhw...);

        //     o *= alpha1;
        //     o += alpha2 * z(n, k, dhw...) + bias(0, k, 0, 0, 0);
        //     o = (o > T{0}) ? o : T{0};
        // });

        // auto& activationOutputData = graphTensorAllocator.LookupTensorData(activationOutputName);
        // activationOutputData.CopyBack();
        // auto& output = activationOutputData.mCpuTensor;

        // EXPECT_FALSE(miopen::range_zero(referenceOutput)) << "Cpu data is all zeros";
        // EXPECT_FALSE(miopen::range_zero(output)) << "Gpu data is all zeros";
        // EXPECT_TRUE(miopen::range_distance(referenceOutput) == miopen::range_distance(output));

        // const double tolerance = 80;
        // double threshold = std::numeric_limits<T>::epsilon() * tolerance;
        // auto error = miopen::rms_range(referenceOutput, output);

        // EXPECT_FALSE(miopen::find_idx(referenceOutput, miopen::not_finite) >= 0)
        //     << "Non finite number found in the CPU data";

        // EXPECT_FALSE(miopen::find_idx(output, miopen::not_finite) >= 0)
        //     << "Non finite number found in the GPU data";

        // EXPECT_TRUE(error < threshold)
        //     << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }
};

struct GPU_ConvBiasResAddActivation_fwd_FP16 : GPU_ConvBiasResAddActivation_fwd<half_float::half>
{
};

} // end namespace conv_graph_api_test
using namespace conv_graph_api_test;

TEST_P(GPU_ConvBiasResAddActivation_fwd_FP16, Test) { Run(); }

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_ConvBiasResAddActivation_fwd_FP16,
                         testing::Combine(testing::ValuesIn(ConvTestConfigs()),
                                          testing::ValuesIn({1.0f, 2.5f}), // alpha1
                                          testing::ValuesIn({1.0f, 2.0f}), // alpha2
                                          testing::Values(miopenTensorNDHWC)));
