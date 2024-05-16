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

#include "mha_helper.hpp"
#include "../verify.hpp"
#include "../get_handle.hpp"
#include "../tensor_holder.hpp"
#include "../workspace.hpp"

#include <gtest/gtest.h>

#include <miopen/graphapi/execution_plan.hpp>
#include <miopen/graphapi/matmul.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <miopen/graphapi/pointwise.hpp>
#include <miopen/graphapi/reduction.hpp>
#include <miopen/graphapi/rng.hpp>
#include <miopen/graphapi/util.hpp>
#include <miopen/graphapi/variant_pack.hpp>

#include <numeric>
#include <string>
#include <unordered_map>

namespace gr = miopen::graphapi;

namespace mha_graph_test {

class MhaFwdGraphTest : public testing::TestWithParam<std::tuple<int, int, int, int>>
{

    struct TensorData
    {
        gr::Tensor* mTensPtr;
        tensor<float> mCpuTensor;
        miopen::Allocator::ManageDataPtr mGpuBuf;

        explicit TensorData(gr::Tensor* tens_ptr) : mTensPtr(tens_ptr), mCpuTensor()
        {
            assert(tens_ptr);
            std::vector<size_t> dims;
            const auto& d = tens_ptr->getDimensions();
            std::copy(d.begin(), d.end(), dims.begin());
            mCpuTensor = tensor<float>{dims};
        }

        void init(tensor<float>&& tens_val)
        {
            mCpuTensor   = std::move(tens_val);
            auto& handle = get_handle();
            mGpuBuf      = handle.Write(mCpuTensor.data);
        }

        void init(float val)
        {
            mCpuTensor.generate([=](auto...) { return val; });
            auto& handle = get_handle();
            mGpuBuf      = handle.Write(mCpuTensor.data);
        }

        void copyBack()
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

    std::unique_ptr<gr::OpGraphBuilder> mGraphBuilder;
    gr::OpGraph mGraph;
    gr::AutoDeleteAllocator mAlloc;
    std::unordered_map<std::string, TensorData> mFilledTensors;

    template <bool IsVirt>
    gr::Tensor* makeTensor(std::string_view name, const std::vector<int64_t>& dims)
    {
        auto ptr = mAlloc.allocate(gr::makeTensor<IsVirt>(name, dims));
        if constexpr(!IsVirt)
        {
            mFilledTensors.emplace(std::string(name), TensorData(ptr));
        }
        return ptr;
    }

    auto* makePointWiseDesc(miopenPointwiseMode_t mode)
    {
        return mAlloc.allocate(gr::PointwiseBuilder{}
                                   .setMode(MIOPEN_POINTWISE_MUL)
                                   .setMathPrecision(miopenFloat)
                                   .build());
    }

    using TensorVec = std::vector<gr::Tensor*>;

    void addBinaryPointwiseNode(gr::Pointwise* pw,
                                const TensorVec& in_tensors,
                                const TensorVec& out_tensors)
    {

        assert(in_tensors.size() == 2);
        assert(out_tensors.size() == 1);

        mGraphBuilder->addNode(mAlloc.allocate(gr::OperationPointwiseBuilder{}
                                                   .setPointwise(pw)
                                                   .setX(in_tensors[0])
                                                   .setB(in_tensors[1])
                                                   .setY(out_tensors[0])
                                                   .build()));
    }

    void addUnaryPointwiseNode(gr::Pointwise* pw,
                               const TensorVec& in_tensors,
                               const TensorVec& out_tensors)
    {

        assert(in_tensors.size() == 1);
        assert(out_tensors.size() == 1);

        mGraphBuilder->addNode(mAlloc.allocate(gr::OperationPointwiseBuilder{}
                                                   .setPointwise(pw)
                                                   .setX(in_tensors[0])
                                                   .setY(out_tensors[0])
                                                   .build()));
    }

    void addReductionNode(miopenReduceTensorOp_t red_op,
                          const TensorVec& in_tensors,
                          const TensorVec& out_tensors)
    {

        auto* red_desc = mAlloc.allocate(
            gr::ReductionBuilder{}.setCompType(miopenFloat).setReductionOperator(red_op).build());

        assert(in_tensors.size() == 1);
        assert(out_tensors.size() == 1);

        mGraphBuilder->addNode(mAlloc.allocate(gr::OperationReductionBuilder{}
                                                   .setReduction(red_desc)
                                                   .setX(in_tensors[0])
                                                   .setY(out_tensors[0])
                                                   .build()));
    }

    void addNode(std::string_view name, const TensorVec& in_tensors, const TensorVec& out_tensors)
    {

        if(name == "OP_MATMUL")
        {
            assert(in_tensors.size() == 2);
            assert(out_tensors.size() == 1);

            auto* mm_desc =
                mAlloc.allocate(gr::MatmulBuilder().setComputeType(miopenFloat8).build());
            mGraphBuilder->addNode(mAlloc.allocate(gr::OperationMatmulBuilder{}
                                                       .setA(in_tensors[0])
                                                       .setB(in_tensors[1])
                                                       .setC(out_tensors[0])
                                                       .setMatmulDescriptor(mm_desc)
                                                       .build()));
        }
        else if(name == "OP_POINTWISE:MUL")
        {

            auto* pw = makePointWiseDesc(MIOPEN_POINTWISE_MUL);
            addBinaryPointwiseNode(pw, in_tensors, out_tensors);
        }
        else if(name == "OP_POINTWISE:SUB")
        {
            auto* pw = makePointWiseDesc(MIOPEN_POINTWISE_SUB);
            addBinaryPointwiseNode(pw, in_tensors, out_tensors);
        }
        else if(name == "OP_POINTWISE:EXP")
        {
            auto* pw = makePointWiseDesc(MIOPEN_POINTWISE_EXP);
            addUnaryPointwiseNode(pw, in_tensors, out_tensors);
        }
        else if(name == "OP_POINTWISE:RECIPROCAL")
        {
            auto* pw = makePointWiseDesc(MIOPEN_POINTWISE_RECIPROCAL);
            addUnaryPointwiseNode(pw, in_tensors, out_tensors);
        }
        else if(name == "OP_REDUCTION:MAX")
        {
            addReductionNode(MIOPEN_REDUCE_TENSOR_MAX, in_tensors, out_tensors);
        }
        else if(name == "OP_REDUCTION:SUM")
        {
            addReductionNode(MIOPEN_REDUCE_TENSOR_MAX, in_tensors, out_tensors);
        }
        else if(name == "OP_RNG")
        {
            constexpr double BERNOULLI = 0.5;
            auto* rng_desc             = mAlloc.allocate(gr::RngBuilder{}
                                                 .setDistribution(MIOPEN_RNG_DISTRIBUTION_BERNOULLI)
                                                 .setBernoulliProb(BERNOULLI)
                                                 .build());

            assert(in_tensors.size() == 2); // first is seed tensor, second is offset
            assert(out_tensors.size() == 1);

            mGraphBuilder->addNode(mAlloc.allocate(gr::OperationRngBuilder{}
                                                       .setRng(rng_desc)
                                                       .setSeed(in_tensors[0])
                                                       .setOffset(in_tensors[1])
                                                       .setOutput(out_tensors[0])
                                                       .build()));
        }
        else
        {
            std::cerr << "Unknown graph node type" << std::endl;
            std::abort();
        }
    }

    void createMhaGraph(int64_t n, int64_t h, int64_t s, int64_t d)
    {

        mGraphBuilder = std::make_unique<gr::OpGraphBuilder>();

        std::vector<int64_t> nhsd  = {n, h, s, d};
        std::vector<int64_t> nhss  = {n, h, s, s};
        std::vector<int64_t> nhs1  = {n, h, s, 1};
        std::vector<int64_t> all1s = {1, 1, 1, 1};

#define MAKE_TENSOR(name, dims, isVirt) auto* name = makeTensor<isVirt>(#name, dims)

        MAKE_TENSOR(Q, nhsd, false);
        MAKE_TENSOR(K, nhsd, false);
        MAKE_TENSOR(V, nhsd, false);

        MAKE_TENSOR(T_MM_0, nhss, true);
        addNode("OP_MATMUL", {Q, K}, {T_MM_0});

        MAKE_TENSOR(T_SCL_0, nhss, true);
        MAKE_TENSOR(ATN_SCL, all1s, false);

        addNode("OP_POINTWISE:MUL", {T_MM_0, ATN_SCL}, {T_SCL_0});

        MAKE_TENSOR(T_SCL_1, nhss, true);
        MAKE_TENSOR(DSCL_Q, all1s, false);

        addNode("OP_POINTWISE:MUL", {T_SCL_0, DSCL_Q}, {T_SCL_1});

        MAKE_TENSOR(T_SCL_2, nhss, true);
        MAKE_TENSOR(DSCL_K, all1s, false);

        addNode("OP_POINTWISE:MUL", {T_SCL_1, DSCL_K}, {T_SCL_2});

        MAKE_TENSOR(M, nhs1, false);
        addNode("OP_REDUCTION:MAX", {T_SCL_2}, {M});

        MAKE_TENSOR(T_SUB, nhss, true);
        addNode("OP_POINTWISE:SUB", {T_SCL_2, M}, {T_SUB});

        MAKE_TENSOR(T_EXP, nhss, true);
        addNode("OP_POINTWISE:EXP", {T_SUB}, {T_EXP});

        MAKE_TENSOR(T_SUM, nhs1, true);
        addNode("OP_REDUCTION:SUM", {T_EXP}, {T_SUM});

        MAKE_TENSOR(Z_INV, nhs1, false);
        addNode("OP_POINTWISE:RECIPROCAL", {T_SUM}, {Z_INV});

        MAKE_TENSOR(T_MUL_0, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_EXP, Z_INV}, {T_MUL_0});

        MAKE_TENSOR(AMAX_S, all1s, false);
        addNode("OP_REDUCTION:MAX", {T_MUL_0}, {AMAX_S});

        MAKE_TENSOR(RND_SD, all1s, false);
        MAKE_TENSOR(RND_OFF, all1s, false);

        MAKE_TENSOR(T_RND, nhss, true);
        addNode("OP_RNG", {RND_SD, RND_OFF}, {T_RND});

        MAKE_TENSOR(T_MUL_1, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_MUL_0, T_RND}, {T_MUL_1});

        MAKE_TENSOR(RND_PRB, all1s, false); // TODO(Amber): revisit
        MAKE_TENSOR(T_SCL_3, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_MUL_1, RND_PRB}, {T_SCL_3});

        MAKE_TENSOR(T_SCL_4, nhss, true);
        MAKE_TENSOR(SCL_S, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_3, SCL_S}, {T_SCL_4});

        MAKE_TENSOR(T_MM_1, nhsd, true);
        addNode("OP_MATMUL", {T_SCL_4, V}, {T_MM_1});

        MAKE_TENSOR(T_SCL_5, nhsd, true);
        MAKE_TENSOR(DSCL_S, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_MM_1, DSCL_S}, {T_SCL_5});

        MAKE_TENSOR(T_SCL_6, nhsd, true);
        MAKE_TENSOR(DSCL_V, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_5, DSCL_V}, {T_SCL_6});

        MAKE_TENSOR(T_SCL_7, nhsd, true);
        MAKE_TENSOR(SCL_O, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_6, SCL_O}, {T_SCL_7});

        MAKE_TENSOR(AMAX_O, all1s, false);
        addNode("OP_REDUCTION:MAX", {T_SCL_6}, {AMAX_O});

        mGraph = std::move(*mGraphBuilder).build();
        mGraphBuilder.reset(nullptr);

#undef MAKE_TENSOR
    }

    gr::VariantPack makeMhaVariantPack(void* wspace_ptr)
    {

        std::vector<int64_t> tens_ids;
        std::vector<void*> gpu_ptrs;

        for(const auto& [k, v] : mFilledTensors)
        {
            tens_ids.emplace_back(v.mTensPtr->getId());
            assert(v.mGpuBuf.get());
            gpu_ptrs.emplace_back(v.mGpuBuf.get());
        }

        return gr::VariantPackBuilder()
            .setTensorIds(tens_ids)
            .setDataPointers(gpu_ptrs)
            .setWorkspace(wspace_ptr)
            .build();
    }

    void executeMhaGraph()
    {
        // TODO(amber): should this be a vector of pointers
        std::vector<gr::Engine> engines = gr::findEngines(&mGraph);

        auto engine_cfg = gr::EngineCfgBuilder().setEngine(engines[0]).build();

        auto& handle = get_handle();
        auto h       = static_cast<miopenHandle_t>(&handle);
        auto plan    = gr::ExecutionPlanBuilder().setEngineCfg(engine_cfg).setHandle(h).build();

        Workspace ws(plan.getWorkspaceSize());

        auto variant_pack = makeMhaVariantPack(ws.ptr());

        plan.execute(variant_pack);
    }

    void initInputs(size_t n, size_t h, size_t s, size_t d)
    {
        using namespace test::cpu;

        ScaledTensor Q = GenScaledTensor(n, h, s, d);
        ScaledTensor K = GenScaledTensor(n, h, s, d);
        ScaledTensor V = GenScaledTensor(n, h, s, d);

        for(auto& [k, v] : mFilledTensors)
        {
            if(k == "Q")
            {
                v.init(std::move(Q.mTensor));
            }
            else if(k == "DSCL_Q")
            {
                v.init(Q.mDescale);
            }
            else if(k == "K")
            {
                v.init(std::move(K.mTensor));
            }
            else if(k == "DSCL_K")
            {
                v.init(K.mDescale);
            }
            else if(k == "V")
            {
                v.init(std::move(V.mTensor));
            }
            else if(k == "DSCL_V")
            {
                v.init(V.mDescale);
            }
            else if(k == "SCL_O" || k == "SCL_S" || k == "DSCL_S" || k == "ATN_SCL")
            {
                v.init(1.0f);
            }
            else if(k == "RND_PRB" || k == "RND_SD" || k == "RND_OFF")
            {
                v.init(0.0f);
            }
            else
            {
                FAIL() << "Uninitialized input: " << k;
            }
        }
    }

    void runCPUverify(size_t n, size_t h, size_t s, size_t d)
    {

        auto softmax_ref  = tensor<float>{n, h, s, s};
        auto oDesc_ref    = tensor<float>{n, h, s, d};
        auto mDesc_ref    = tensor<float>{n, h, s, 1};
        auto zInvDesc_ref = tensor<float>{n, h, s, 1};
        float amaxS_ref   = 0;
        float amaxO_ref   = 0;

        auto lookup = [this](const std::string& k) -> TensorData& {
            auto it = mFilledTensors.find(k);
            assert(it != mFilledTensors.cend());
            return it->second;
        };

        test::cpu::MultiHeadAttentionfp8(lookup("Q").mCpuTensor,
                                         lookup("K").mCpuTensor,
                                         lookup("V").mCpuTensor,
                                         softmax_ref,
                                         mDesc_ref,
                                         zInvDesc_ref,
                                         lookup("DSCL_Q").mCpuTensor[0],
                                         lookup("DSCL_K").mCpuTensor[0],
                                         lookup("DSCL_V").mCpuTensor[0],
                                         lookup("DSCL_S").mCpuTensor[0],
                                         lookup("SCL_S").mCpuTensor[0],
                                         lookup("SCL_O").mCpuTensor[0],
                                         lookup("RND_PRB").mCpuTensor[0],
                                         static_cast<uint64_t>(lookup("RND_SD").mCpuTensor[0]),
                                         static_cast<uint64_t>(lookup("RND_OFF").mCpuTensor[0]),
                                         amaxS_ref,
                                         amaxO_ref,
                                         oDesc_ref);

        auto GetResult = [&](const std::string& t_name) {
            auto it = mFilledTensors.find(t_name);
            assert(it != mFilledTensors.cend());
            auto& v = it->second;
            v.copyBack();
            return v.mCpuTensor;
        };

        const double error_threshold = 5e-6;

        const auto& resAmaxS = GetResult("AMAX_S");
        auto amaxS_abs_diff  = std::abs(amaxS_ref - resAmaxS[0]);
        EXPECT_LT(amaxS_abs_diff, error_threshold)
            << " ref: " << amaxS_ref << " result: " << resAmaxS[0];

        const auto& resAmaxO = GetResult("AMAX_O");
        auto amaxO_abs_diff  = std::abs(amaxO_ref - resAmaxO[0]);
        EXPECT_LT(amaxO_abs_diff, error_threshold)
            << " ref: " << amaxO_ref << " result: " << resAmaxO[0];

        double M_error = miopen::rms_range(mDesc_ref, GetResult("M"));
        EXPECT_LT(M_error, error_threshold);

        double ZInv_error = miopen::rms_range(zInvDesc_ref, GetResult("Z_INV"));
        EXPECT_LT(ZInv_error, error_threshold);

        double O_error = miopen::rms_range(oDesc_ref, GetResult("O"));
        EXPECT_LT(O_error, error_threshold);
    }

public:
    void Run()
    {
        auto [n, h, s, d] = GetParam();
        createMhaGraph(n, h, s, d);
        initInputs(n, h, s, d);
        executeMhaGraph();
        runCPUverify(n, h, s, d);
    }
};

} // end namespace mha_graph_test

using namespace mha_graph_test;

TEST_P(MhaFwdGraphTest, MhaFwdGraph) { Run(); }

INSTANTIATE_TEST_SUITE_P(MhaGraphFwdSuite,
                         MhaFwdGraphTest,
                         testing::Combine(testing::ValuesIn({2}), // n
                                          testing::ValuesIn({4}), // s
                                          testing::ValuesIn({8}), // h
                                          testing::ValuesIn({16}) // d
                                          ));
