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
#include "gtest_common.hpp"
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
#include <miopen/graphapi/reshape.hpp>
#include <miopen/graphapi/rng.hpp>
#include <miopen/graphapi/util.hpp>
#include <miopen/graphapi/variant_pack.hpp>

#include <numeric>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace gr = miopen::graphapi;

namespace mha_graph_test {
class MhaGraphTestBase
    : public testing::TestWithParam<std::tuple<size_t, size_t, size_t, size_t, float>>
{

protected:
    struct TensorData
    {
        using TensFlt = tensor<float>;
        using TensI64 = tensor<int64_t>;

        gr::Tensor* mTensPtr;
        std::variant<TensFlt, TensI64> mCpuTensor;
        miopen::Allocator::ManageDataPtr mGpuBuf;

        explicit TensorData(gr::Tensor* tens_ptr) : mTensPtr(tens_ptr), mCpuTensor()
        {
            assert(mTensPtr);
            const auto& d = mTensPtr->GetLengths();
            std::vector<size_t> dims(d.begin(), d.end());
            if(auto dt = mTensPtr->GetType(); dt == miopenFloat)
            {
                mCpuTensor = TensFlt{dims};
            }
            else if(dt == miopenInt64)
            {
                mCpuTensor = TensI64{dims};
            }
            else
            {
                MIOPEN_FRIENDLY_FAIL("Unsupported data type for tensor" << dt);
            }
        }

        void init(tensor<float>&& tens_val)
        {
            auto& handle = get_handle();
            assert(mTensPtr->GetType() == miopenFloat);
            auto& ct = std::get<TensFlt>(mCpuTensor);
            ct       = std::move(tens_val);
            mGpuBuf  = handle.Write(ct.data);
        }

        template <typename T>
        void init(T val)
        {
            auto& handle = get_handle();
            std::visit(
                [&](auto&& ct) {
                    ct.generate([=](auto...) { return val; });
                    mGpuBuf = handle.Write(ct.data);
                },
                mCpuTensor);
        }

        void copyBack()
        {
            auto& handle = get_handle();
            std::visit([&](auto&& ct) { handle.ReadToVec(mGpuBuf, ct.data); }, mCpuTensor);
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
    float mAttentionScale = 1.0f;
    float mProbDropout    = 0.0f;
    double mErrorThresh   = 5e-5;

    virtual void createMhaGraph(size_t n, size_t h, size_t s, size_t d) = 0;
    virtual void initInputs(size_t n, size_t h, size_t s, size_t d)     = 0;
    virtual void runCpuVerify(size_t n, size_t h, size_t s, size_t d)   = 0;

    virtual ~MhaGraphTestBase() = default;

    auto* makePointWiseDesc(miopenPointwiseMode_t mode)
    {
        return mAlloc.allocate(
            gr::PointwiseBuilder{}.setMode(mode).setMathPrecision(miopenFloat).build());
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

    void addUnaryPointwiseNode(gr::Pointwise* pw,
                               const TensorVec& in_tensors,
                               const TensorVec& out_tensors,
                               float alpha1)
    {

        assert(in_tensors.size() == 1);
        assert(out_tensors.size() == 1);

        mGraphBuilder->addNode(mAlloc.allocate(gr::OperationPointwiseBuilder{}
                                                   .setPointwise(pw)
                                                   .setX(in_tensors[0])
                                                   .setY(out_tensors[0])
                                                   .setAlpha1(alpha1)
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
        else if(name == "OP_REDUCTION:ADD")
        {
            addReductionNode(MIOPEN_REDUCE_TENSOR_ADD, in_tensors, out_tensors);
        }
        else if(name == "OP_RNG")
        {
            auto* rng_desc = mAlloc.allocate(gr::RngBuilder{}
                                                 .setDistribution(MIOPEN_RNG_DISTRIBUTION_BERNOULLI)
                                                 .setBernoulliProb(mProbDropout)
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
        else if(name == "OP_RESHAPE")
        {
            assert(in_tensors.size() == 1);
            assert(out_tensors.size() == 1);
            mGraphBuilder->addNode(mAlloc.allocate(
                gr::OperationReshapeBuilder{}.setX(in_tensors[0]).setY(out_tensors[0]).build()));
        }
        else
        {
            FAIL() << "Unknown graph node type: " << name;
        }
    }

    template <bool IsVirt>
    gr::Tensor*
    makeTensor(std::string_view name, miopenDataType_t dt, const std::vector<size_t>& dims)
    {
        auto ptr = mAlloc.allocate(gr::makeTensor<IsVirt>(name, dt, dims));
        if constexpr(!IsVirt)
        {
            auto [it, inserted] = mFilledTensors.try_emplace(std::string(name), TensorData(ptr));
            if(!inserted)
            {
                MIOPEN_FRIENDLY_FAIL("Duplicate tensor name");
            }
        }
        return ptr;
    }

#define MAKE_TENSOR_F(name, dims, isVirt) auto* name = makeTensor<isVirt>(#name, miopenFloat, dims)
#define MAKE_TENSOR_I(name, dims, isVirt) auto* name = makeTensor<isVirt>(#name, miopenInt64, dims)

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

    /// \todo remove virtual once backward mha is ready to execute
    virtual void executeMhaGraph()
    {
        auto& handle = get_handle();
        mGraphBuilder->setHandle(static_cast<miopenHandle_t>(&handle));
        mGraph = std::move(*mGraphBuilder).build();
        mGraphBuilder.reset(nullptr);

        // TODO(amber): should this be a vector of pointers
        std::vector<gr::Engine> engines = gr::findEngines(&mGraph);

        auto engine_cfg = gr::EngineCfgBuilder().setEngine(engines[0]).build();

        auto h = static_cast<miopenHandle_t>(&handle);

        auto plan = gr::ExecutionPlanBuilder().setEngineCfg(engine_cfg).setHandle(h).build();

        // Serialize and deserialize the plan to test JSON attribute
        plan =
            gr::ExecutionPlanBuilder().setJsonRepresentation(plan.getJsonRepresentation()).build();

        Workspace ws(plan.getWorkspaceSize());

        auto variant_pack = makeMhaVariantPack(ws.ptr());

        plan.execute(h, variant_pack);
    }

    struct CpuMhaFwdOut
    {
        tensor<float> mSoftMax;
        tensor<float> mO;
        tensor<float> mM;
        tensor<float> mZinv;
        float mAmaxS = 0;
        float mAmaxO = 0;

        CpuMhaFwdOut(size_t n, size_t h, size_t s, size_t d)
            : mSoftMax(n, h, s, s), mO(n, h, s, d), mM(n, h, s, 1), mZinv(n, h, s, 1)
        {
        }
    };

    TensorData& lookup(const std::string& k)
    {
        auto it = mFilledTensors.find(k);
        assert(it != mFilledTensors.cend());
        return it->second;
    }
    auto lookup_f(const std::string& k)
    {
        return std::get<TensorData::TensFlt>(lookup(k).mCpuTensor);
    };

    auto lookup_i(const std::string& k)
    {
        return std::get<TensorData::TensI64>(lookup(k).mCpuTensor);
    };

    CpuMhaFwdOut runCpuMhaFWd(size_t n, size_t h, size_t s, size_t d)
    {

        CpuMhaFwdOut out(n, h, s, d);

        test::cpu::MultiHeadAttentionForwardfp8(lookup_f("Q"),
                                                lookup_f("K"),
                                                lookup_f("V"),
                                                out.mSoftMax,
                                                out.mM,
                                                out.mZinv,
                                                lookup_f("DSCL_Q")[0],
                                                lookup_f("DSCL_K")[0],
                                                lookup_f("DSCL_V")[0],
                                                lookup_f("DSCL_S")[0],
                                                lookup_f("SCL_S")[0],
                                                lookup_f("SCL_O")[0],
                                                lookup_f("RND_PRB")[0],
                                                lookup_i("RND_SD")[0],
                                                lookup_i("RND_OFF")[0],
                                                out.mAmaxS,
                                                out.mAmaxO,
                                                out.mO);

        return out;
    }

    TensorData::TensFlt& GetResult(const std::string& t_name)
    {
        auto it = mFilledTensors.find(t_name);
        if(it == mFilledTensors.cend())
        {
            MIOPEN_FRIENDLY_FAIL("Tensor not found in the map: " << t_name);
        }
        auto& v = it->second;
        v.copyBack();
        return std::get<TensorData::TensFlt>(v.mCpuTensor);
    }

    void checkAmax(const std::string& t_name, const float refAmax)
    {
        const auto& resAmax = GetResult(t_name);
        auto abs_diff       = std::abs(refAmax - resAmax[0]);
        ASSERT_LT(abs_diff, mErrorThresh) << " ref: " << refAmax << " result: " << resAmax[0];
    }

    void checkTensor(const std::string& t_name, const tensor<float>& ref_tens)
    {
        double rms = miopen::rms_range(ref_tens, GetResult(t_name));
        ASSERT_LT(rms, mErrorThresh);
    }

public:
    enum class MhaDir
    {
        Fwd,
        Bwd
    };

    void Run(MhaDir direction)
    {
        auto [n, h, s, d, p] = GetParam();
        std::cout << "n:" << n << ", h:" << h << ", s:" << s << ", d:" << d << ", p:" << p
                  << std::endl;
        mProbDropout = p;

        auto& handle = get_handle();
        if(direction == MhaDir::Fwd && (p > 0.0f) && (s % handle.GetWavefrontWidth() != 0))
        {
            GTEST_SKIP()
                << "CPU Fwd pass with Dropout currently supprorts only fully occupied warps";
        }

        if(direction == MhaDir::Bwd && p > 0.0f)
        {
            GTEST_SKIP() << "CPU backward pass with Dropout is not supported currently";
        }

        createMhaGraph(n, h, s, d);
        initInputs(n, h, s, d);
        executeMhaGraph();
        runCpuVerify(n, h, s, d);
    }
};

} // end namespace mha_graph_test
