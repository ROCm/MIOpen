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
#include <gtest/gtest_common.hpp>
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

bool IsTestSupportedForDevice()
{
    return IsTestSupportedByDevice(Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X);
}

static bool TestIsApplicable() { return true; }

static std::vector<Conv3DTestCase> ConvTestConfigs()
{ // g   n   c   k   image   filter   pad   stride   dilation
    // clang-format off
    return {{1, 1, 4, 4, {14, 11, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 1, 1, 1, {1, 4, 4}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 1, 1, 1, {8, 8, 8}, {2, 2, 2}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {1, 1, 1, 1, {8, 8, 8}, {2, 2, 2}, {0, 0, 0}, {2, 2, 2}, {1, 1, 1}, miopenConvolution},
            {2, 8, 8, 4, {12, 14, 4}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {4, 8, 8, 16, {11, 11, 11}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {6, 8, 18, 18, {11, 11, 11}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {8, 8, 8, 8, {11, 11, 11}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {4, 8, 4, 8, {11, 11, 11}, {3, 4, 5}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution},
            {2, 8, 2, 2, {11, 11, 11}, {4, 4, 4}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, miopenConvolution}};
    // clang-format on
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

    // Graph of Y = activation(Conv(X,W) * alpha1 + Z * alpha2 + B)
    inline static const std::string convInputName        = "X";
    inline static const std::string convWeightName       = "W";
    inline static const std::string convOutputName       = "Tmp0";
    inline static const std::string addInputName         = "Z";
    inline static const std::string addOutputName        = "Tmp1";
    inline static const std::string biasInputName        = "B";
    inline static const std::string biasOutputName       = "Tmp2";
    inline static const std::string activationOutputName = "Y";

public:
    void SetUp() override
    {
        if(!TestIsApplicable())
        {
            GTEST_SKIP();
        }
        if(!IsTestSupportedForDevice())
        {
            GTEST_SKIP() << "CBA graph Fusion not supported in this device";
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
            dataType, tensorLayout, {1, convConfig.K, 1, 1, 1}};
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

        auto& addTensorData   = graphTensorAllocator.LookupTensorData(addInputName);
        auto& biasTensorData  = graphTensorAllocator.LookupTensorData(biasInputName);
        auto& activTensorData = graphTensorAllocator.LookupTensorData(activationOutputName);

        auto genValue = [](auto...) {
            return prng::gen_A_to_B(static_cast<T>(-3.0), static_cast<T>(3.0));
        };

        convInputData.Init(genValue);
        convWeightData.Init(genValue);
        addTensorData.Init(genValue);
        biasTensorData.Init(genValue);
        activTensorData.Init([](auto...) { return 0; });

        auto& handle   = get_handle();
        auto handlePtr = static_cast<miopenHandle_t>(&handle);
        graphBuilder.setHandle(handlePtr);
        gr::OpGraph graph;
        ASSERT_NO_THROW(graph = std::move(graphBuilder).build());
        std::vector<gr::Engine> engines;
        ASSERT_NO_THROW(engines = gr::findEngines(&graph));

        ASSERT_GT(engines.size(), 0);

        gr::EngineCfg engineConfig;
        ASSERT_NO_THROW(engineConfig = gr::EngineCfgBuilder().setEngine(engines[0]).build());

        gr::ExecutionPlan plan;
        ASSERT_NO_THROW(
            plan =
                gr::ExecutionPlanBuilder().setEngineCfg(engineConfig).setHandle(handlePtr).build());

        Workspace ws(plan.getWorkspaceSize());

        gr::VariantPack variantPack;
        ASSERT_NO_THROW(variantPack = graphTensorAllocator.MakeVariantPack(ws.ptr()));

        ASSERT_NO_THROW(plan.execute(handlePtr, variantPack));

        // Reference implementation for Y = activation(Conv(X,W) * alpha1 + Z * alpha2 + B)
        auto referenceOutput = tensor<T>(convOutputDesc);
        referenceOutput      = ref_conv_fwd(convInputData.mCpuTensor,
                                       convWeightData.mCpuTensor,
                                       referenceOutput,
                                       convInputDescription);

        auto& z    = addTensorData.mCpuTensor;
        auto& bias = biasTensorData.mCpuTensor;

        referenceOutput.par_for_each([&](auto n, auto k, auto... dhw) {
            auto& o = referenceOutput(n, k, dhw...);

            o *= alpha1;
            o += alpha2 * z(n, k, dhw...) + bias(0, k, 0, 0, 0);
            o = (o > T{0}) ? o : T{0};
        });

        auto& activationOutputData = graphTensorAllocator.LookupTensorData(activationOutputName);
        activationOutputData.CopyBack();
        auto& output = activationOutputData.mCpuTensor;

        EXPECT_FALSE(miopen::range_zero(referenceOutput)) << "Cpu data is all zeros";
        EXPECT_FALSE(miopen::range_zero(output)) << "Gpu data is all zeros";
        EXPECT_TRUE(miopen::range_distance(referenceOutput) == miopen::range_distance(output));

        const double tolerance = 80;
        double threshold       = std::numeric_limits<T>::epsilon() * tolerance;
        auto error             = miopen::rms_range(referenceOutput, output);

        EXPECT_FALSE(miopen::find_idx(referenceOutput, miopen::not_finite) >= 0)
            << "Non finite number found in the CPU data";

        EXPECT_FALSE(miopen::find_idx(output, miopen::not_finite) >= 0)
            << "Non finite number found in the GPU data";

        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }

    // Test that finding engine fails due to invalid Convolution tensor layout (input_c % wei_c !=
    // 0)
    void RunExceptionCheck()
    {
        float alpha1                      = 1.0f;
        float alpha2                      = 1.0f;
        miopenTensorLayout_t tensorLayout = miopenTensorNDHWC;
        auto dataType                     = miopen_type<T>();

        gr::OpGraphBuilder graphBuilder;
        gr::AutoDeleteAllocator allocator;

        miopen::TensorDescriptor convInputTensorDesc{dataType, tensorLayout, {1, 4, 14, 11, 1}};
        auto convInput =
            allocator.allocate(gr::makeTensor<false>(convInputName,
                                                     dataType,
                                                     convInputTensorDesc.GetLengths(),
                                                     convInputTensorDesc.GetStrides()));

        miopen::TensorDescriptor convInvalidCWeightTensorDesc{
            dataType, tensorLayout, {1, 3, 4, 3, 3}};
        auto convWeight =
            allocator.allocate(gr::makeTensor<false>(convWeightName,
                                                     dataType,
                                                     convInvalidCWeightTensorDesc.GetLengths(),
                                                     convInvalidCWeightTensorDesc.GetStrides()));

        std::vector<size_t> allOnes{size_t{1}, size_t{1}, size_t{1}, size_t{1}, size_t{1}};
        auto convOutput =
            allocator.allocate(gr::makeTensor<true>(convOutputName, dataType, allOnes, allOnes));

        gr::Convolution* convolution = nullptr;
        ASSERT_NO_THROW(convolution = allocator.allocate(gr::ConvolutionBuilder{}
                                                             .setCompType(dataType)
                                                             .setMode(miopenConvolution)
                                                             .setSpatialDims(3)
                                                             .setDilations({1, 1, 1})
                                                             .setFilterStrides({1, 1, 1})
                                                             .setPrePaddings({1, 1, 1})
                                                             .setPostPaddings({1, 1, 1})
                                                             .build()));

        ASSERT_NO_THROW(
            graphBuilder.addNode(allocator.allocate(gr::OperationConvolutionForwardBuilder()
                                                        .setConvolution(convolution)
                                                        .setX(convInput)
                                                        .setY(convOutput)
                                                        .setW(convWeight)
                                                        .setAlpha(1.0)
                                                        .setBeta(0)
                                                        .build())));

        auto addInput =
            allocator.allocate(gr::makeTensor<false>(addInputName, dataType, allOnes, allOnes));
        auto addOutput =
            allocator.allocate(gr::makeTensor<true>(addOutputName, dataType, allOnes, allOnes));

        ASSERT_NO_THROW(graphBuilder.addNode(
            MakeAddNode(allocator, dataType, convOutput, addInput, addOutput, alpha1, alpha2)));

        miopen::TensorDescriptor biasInputTensorDesc{dataType, tensorLayout, allOnes};
        auto biasInput =
            allocator.allocate(gr::makeTensor<false>(biasInputName, dataType, allOnes, allOnes));
        auto biasOutput =
            allocator.allocate(gr::makeTensor<true>(biasOutputName, dataType, allOnes, allOnes));

        ASSERT_NO_THROW(graphBuilder.addNode(
            MakeAddNode(allocator, dataType, addOutput, biasInput, biasOutput)));

        auto activationOutput = allocator.allocate(
            gr::makeTensor<false>(activationOutputName, dataType, allOnes, allOnes));

        gr::Pointwise* activation = nullptr;
        ASSERT_NO_THROW(activation = allocator.allocate(gr::PointwiseBuilder{}
                                                            .setMode(MIOPEN_POINTWISE_RELU_FWD)
                                                            .setMathPrecision(dataType)
                                                            .build()));
        ASSERT_NO_THROW(graphBuilder.addNode(allocator.allocate(gr::OperationPointwiseBuilder{}
                                                                    .setPointwise(activation)
                                                                    .setX(biasOutput)
                                                                    .setY(activationOutput)
                                                                    .build())));

        auto& handle   = get_handle();
        auto handlePtr = static_cast<miopenHandle_t>(&handle);
        graphBuilder.setHandle(handlePtr);
        gr::OpGraph graph;
        ASSERT_NO_THROW(graph = std::move(graphBuilder).build());
        std::vector<gr::Engine> engines;
        ASSERT_THROW(engines = gr::findEngines(&graph), miopen::Exception);
    }
};

} // end namespace conv_graph_api_test
using namespace conv_graph_api_test;

#define DEFINE_GRAPH_API_CONV_BIAS_ACTIV_TEST(type, datatype, dir)                  \
    struct GPU_ConvBiasResAddActivation_##dir##_##type                              \
        : GPU_ConvBiasResAddActivation_##dir<datatype>                              \
    {                                                                               \
    };                                                                              \
    TEST_P(GPU_ConvBiasResAddActivation_##dir##_##type, Test) { Run(); }            \
    INSTANTIATE_TEST_SUITE_P(Smoke,                                                 \
                             GPU_ConvBiasResAddActivation_##dir##_##type,           \
                             testing::Combine(testing::ValuesIn(ConvTestConfigs()), \
                                              testing::ValuesIn({1.0f, 2.5f}),      \
                                              testing::ValuesIn({1.0f, 2.0f}),      \
                                              testing::Values(miopenTensorNDHWC))); \
    TEST_F(GPU_ConvBiasResAddActivation_##dir##_##type, TestExceptions) { RunExceptionCheck(); }

DEFINE_GRAPH_API_CONV_BIAS_ACTIV_TEST(FP16, half_float::half, fwd);
DEFINE_GRAPH_API_CONV_BIAS_ACTIV_TEST(FP32, float, fwd);
DEFINE_GRAPH_API_CONV_BIAS_ACTIV_TEST(BFP16, bfloat16, fwd);
