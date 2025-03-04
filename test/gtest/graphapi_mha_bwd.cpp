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

#include "graphapi_mha_cpp_common.hpp"

namespace mha_graph_test {

class GPU_MhaBwdGraphTest_FP32 : public MhaGraphTestBase
{

    tensor<float> mSoftMax;

public:
    void createMhaGraph(size_t n, size_t h, size_t s, size_t d) override
    {
        mGraphBuilder = std::make_unique<gr::OpGraphBuilder>();

        std::vector<size_t> nhsd  = {n, h, s, d};
        std::vector<size_t> nhds  = {n, h, d, s};
        std::vector<size_t> nhss  = {n, h, s, s};
        std::vector<size_t> nhs1  = {n, h, s, 1};
        std::vector<size_t> all1s = {1, 1, 1, 1};

        MAKE_TENSOR_F(Q, nhsd, false);
        MAKE_TENSOR_F(K, nhsd, false);
        MAKE_TENSOR_F(V, nhsd, false);
        MAKE_TENSOR_F(dO, nhsd, false);
        MAKE_TENSOR_F(O, nhsd, false);

        MAKE_TENSOR_F(K_T, nhds, true);
        MAKE_TENSOR_F(T_MM_0, nhss, true);

        addNode("OP_RESHAPE", {K}, {K_T});
        addNode("OP_MATMUL", {Q, K_T}, {T_MM_0});

        MAKE_TENSOR_F(T_SCL_0, nhss, true);
        // MAKE_TENSOR_F(ATN_SCL, all1s, false);
        // addNode("OP_POINTWISE:MUL", {T_MM_0, ATN_SCL}, {T_SCL_0});
        // NOTE(Amber): Replacing the code above with a hacky solution to pass
        // attention scale as a scalar param instead of a tensor. This is because
        // Find 2.0 for MHA expects it as a scalar param set on MHA Problem
        // Descriptor.
        auto* pw_id = makePointWiseDesc(MIOPEN_POINTWISE_IDENTITY);
        addUnaryPointwiseNode(pw_id, {T_MM_0}, {T_SCL_0}, mAttentionScale);

        MAKE_TENSOR_F(T_SCL_1, nhss, true);
        MAKE_TENSOR_F(DSCL_Q, all1s, false);

        addNode("OP_POINTWISE:MUL", {T_SCL_0, DSCL_Q}, {T_SCL_1});

        MAKE_TENSOR_F(T_SCL_2, nhss, true);
        MAKE_TENSOR_F(DSCL_K, all1s, false);

        addNode("OP_POINTWISE:MUL", {T_SCL_1, DSCL_K}, {T_SCL_2});

        MAKE_TENSOR_F(M, nhs1, false);
        MAKE_TENSOR_F(T_SUB_0, nhss, true);
        addNode("OP_POINTWISE:SUB", {T_SCL_2, M}, {T_SUB_0});

        MAKE_TENSOR_F(T_EXP, nhss, true);
        addNode("OP_POINTWISE:EXP", {T_SUB_0}, {T_EXP});

        MAKE_TENSOR_F(Z_INV, nhs1, false);
        MAKE_TENSOR_F(T_MUL_0, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_EXP, Z_INV}, {T_MUL_0});

        MAKE_TENSOR_I(RND_SD, all1s, false);
        MAKE_TENSOR_I(RND_OFF, all1s, false);

        MAKE_TENSOR_F(T_RND, nhss, true);
        addNode("OP_RNG", {RND_SD, RND_OFF}, {T_RND});

        MAKE_TENSOR_F(T_MUL_1, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_MUL_0, T_RND}, {T_MUL_1});

        MAKE_TENSOR_F(RND_PRB, all1s, false); // TODO(Amber): revisit
        MAKE_TENSOR_F(T_SCL_3, nhss, true);
        // T_SCL_3 feeds into the middle column of the graph
        addNode("OP_POINTWISE:MUL", {T_MUL_1, RND_PRB}, {T_SCL_3});

        MAKE_TENSOR_F(T_SCL_4, nhss, true);
        MAKE_TENSOR_F(SCL_S, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_3, SCL_S}, {T_SCL_4});

        MAKE_TENSOR_F(SCL_4T, nhss, true);
        addNode("OP_RESHAPE", {T_SCL_4}, {SCL_4T});

        MAKE_TENSOR_F(T_MM_1, nhsd, true);
        addNode("OP_MATMUL", {SCL_4T, dO}, {T_MM_1});

        MAKE_TENSOR_F(T_SCL_5, nhsd, true);
        MAKE_TENSOR_F(DSCL_S, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_MM_1, DSCL_S}, {T_SCL_5});

        MAKE_TENSOR_F(T_SCL_6, nhsd, true);
        MAKE_TENSOR_F(DSCL_dO, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_5, DSCL_dO}, {T_SCL_6});

        MAKE_TENSOR_F(dV, nhsd, false);
        MAKE_TENSOR_F(SCL_dV, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_6, SCL_dV}, {dV});

        MAKE_TENSOR_F(AMax_dV, all1s, false);
        addNode("OP_REDUCTION:MAX", {T_SCL_6}, {AMax_dV});

        ////////////////// Center-top //////////////////////////////////

        MAKE_TENSOR_F(V_T, nhds, true);
        addNode("OP_RESHAPE", {V}, {V_T});

        MAKE_TENSOR_F(T_MM_2, nhss, true);
        addNode("OP_MATMUL", {dO, V_T}, {T_MM_2});

        MAKE_TENSOR_F(T_SCL_7, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_MM_2, DSCL_dO}, {T_SCL_7});

        MAKE_TENSOR_F(T_SCL_8, nhss, true);
        MAKE_TENSOR_F(DSCL_V, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_7, DSCL_V}, {T_SCL_8});

        MAKE_TENSOR_F(T_SCL_9, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_SCL_8, T_RND}, {T_SCL_9});

        MAKE_TENSOR_F(T_SCL_10, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_SCL_9, RND_PRB}, {T_SCL_10});

        ////////////////// Right-top //////////////////////////////////
        MAKE_TENSOR_F(T_SCL_11, nhsd, true);
        addNode("OP_POINTWISE:MUL", {dO, DSCL_dO}, {T_SCL_11});

        MAKE_TENSOR_F(DSCL_O, all1s, false);
        MAKE_TENSOR_F(T_SCL_12, nhsd, true);
        addNode("OP_POINTWISE:MUL", {O, DSCL_O}, {T_SCL_12});

        MAKE_TENSOR_F(T_MUL_2, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_SCL_11, T_SCL_12}, {T_MUL_2});

        MAKE_TENSOR_F(T_SCL_13, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_MUL_2, RND_PRB}, {T_SCL_13});

        MAKE_TENSOR_F(T_SUM_0, nhs1, true);
        addNode("OP_REDUCTION:ADD", {T_SCL_13}, {T_SUM_0});

        ////////////////// Center Part //////////////////////////////////
        MAKE_TENSOR_F(T_SUB_1, nhss, true);
        addNode("OP_POINTWISE:SUB", {T_SCL_10, T_SUM_0}, {T_SUB_1});

        MAKE_TENSOR_F(T_SCL_14, nhss, true);
        addUnaryPointwiseNode(pw_id, {T_SUB_1}, {T_SCL_14}, mAttentionScale);

        MAKE_TENSOR_F(T_MUL_3, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_SCL_14, T_SCL_3}, {T_MUL_3});

        MAKE_TENSOR_F(T_SCL_15, nhss, true);
        MAKE_TENSOR_F(SCL_dS, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_MUL_3, SCL_dS}, {T_SCL_15});

        MAKE_TENSOR_F(AMax_dS, all1s, false);
        addNode("OP_REDUCTION:MAX", {T_MUL_3}, {AMax_dS});

        ////////////////// Center Bottom //////////////////////////////////

        MAKE_TENSOR_F(T_MM_3, nhsd, true);
        addNode("OP_MATMUL", {T_SCL_15, K}, {T_MM_3});

        MAKE_TENSOR_F(T_SCL_16, nhsd, true);
        MAKE_TENSOR_F(DSCL_dS, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_MM_3, DSCL_dS}, {T_SCL_16});

        MAKE_TENSOR_F(T_SCL_17, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_SCL_16, DSCL_K}, {T_SCL_17});

        MAKE_TENSOR_F(dQ, nhsd, false);
        MAKE_TENSOR_F(SCL_dQ, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_17, SCL_dQ}, {dQ});

        MAKE_TENSOR_F(AMax_dQ, all1s, false);
        addNode("OP_REDUCTION:MAX", {T_SCL_17}, {AMax_dQ});
        ////////////////// Right Bottom //////////////////////////////////

        MAKE_TENSOR_F(SCL_15T, nhss, true);
        addNode("OP_RESHAPE", {T_SCL_15}, {SCL_15T});

        MAKE_TENSOR_F(T_MM_4, nhsd, true);
        addNode("OP_MATMUL", {SCL_15T, Q}, {T_MM_4});

        MAKE_TENSOR_F(T_SCL_18, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_MM_4, DSCL_dS}, {T_SCL_18});

        MAKE_TENSOR_F(T_SCL_19, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_SCL_18, DSCL_Q}, {T_SCL_19});

        MAKE_TENSOR_F(dK, nhsd, false);
        MAKE_TENSOR_F(SCL_dK, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_19, SCL_dK}, {dK});

        MAKE_TENSOR_F(AMax_dK, all1s, false);
        addNode("OP_REDUCTION:MAX", {T_SCL_19}, {AMax_dK});

        // create tensor SCL_O for running cpu fwd mha
        MAKE_TENSOR_F(SCL_O, all1s, false);
        std::ignore = SCL_O;
    }

    void initInputs(size_t n, size_t h, size_t s, size_t d) override
    {
        using namespace test::cpu;

        auto Q  = GenScaledTensorBackward<float>(n, h, s, d);
        auto K  = GenScaledTensorBackward<float>(n, h, s, d);
        auto V  = GenScaledTensorBackward<float>(n, h, s, d);
        auto dO = GenScaledTensorBackward<float>(n, h, s, d);

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
            else if(k == "dO")
            {
                v.init(std::move(dO.mTensor));
            }
            else if(k == "DSCL_dO")
            {
                v.init(dO.mDescale);
            }
            else if(k == "SCL_O" || k == "SCL_S" || k == "DSCL_O" || k == "DSCL_S" ||
                    k == "ATN_SCL" || k == "SCL_dS" || k == "DSCL_dS" || k == "SCL_dV" ||
                    k == "SCL_dQ" || k == "SCL_dK")
            {
                v.init(1.0f);
            }
            else if(k == "RND_PRB")
            {
                v.init(mProbDropout);
            }
            else if(k == "RND_SD" || k == "RND_OFF")
            {
                v.init(0ll);
            }
            else if(k == "dV" || k == "dQ" || k == "dK" || k == "AMax_dV" || k == "AMax_dQ" ||
                    k == "AMax_dK" || k == "AMax_dS")
            {
                // these are outputs
                v.init(0.0f);
            }
            else if(k == "M" || k == "Z_INV" || k == "O")
            {
                // init later below
            }
            else
            {
                FAIL() << "Uninitialized input or output: " << k;
            }
        }

        CpuMhaFwdOut out = runCpuMhaFWd(n, h, s, d);

        lookup("M").init(std::move(out.mM));
        lookup("Z_INV").init(std::move(out.mZinv));
        lookup("O").init(std::move(out.mO));

        // softmax needed for calling cpu backward mha
        mSoftMax = std::move(out.mSoftMax);

        // Remove "SCL_O" here so that it doesn't pollute the variant pack used for
        // execution
        size_t cnt = mFilledTensors.erase("SCL_O");
        ASSERT_EQ(cnt, 1);
    }

    void runCpuVerify(size_t n, size_t h, size_t s, size_t d) override
    {
        /// \todo remove virtual once backward mha is ready to execute
        static constexpr bool disable_verification = true; // TODO

        auto dQ_ref = tensor<float>{n, h, s, d};
        auto dK_ref = tensor<float>{n, h, s, d};
        auto dV_ref = tensor<float>{n, h, s, d};

        float amax_dS_ref = 0;
        float amax_dQ_ref = 0;
        float amax_dK_ref = 0;
        float amax_dV_ref = 0;

        test::cpu::MultiHeadAttentionBackwardDataf8(lookup_f("Q"),
                                                    lookup_f("K"),
                                                    lookup_f("V"),
                                                    lookup_f("O"),
                                                    lookup_f("dO"),
                                                    mSoftMax,
                                                    lookup_f("DSCL_Q")[0],
                                                    lookup_f("DSCL_K")[0],
                                                    lookup_f("DSCL_V")[0],
                                                    lookup_f("SCL_dQ")[0],
                                                    lookup_f("SCL_dK")[0],
                                                    lookup_f("SCL_dV")[0],
                                                    lookup_f("SCL_S")[0],
                                                    lookup_f("DSCL_S")[0],
                                                    lookup_f("SCL_dS")[0],
                                                    lookup_f("DSCL_dS")[0],
                                                    lookup_f("DSCL_O")[0],
                                                    lookup_f("DSCL_dO")[0],
                                                    amax_dS_ref,
                                                    amax_dQ_ref,
                                                    amax_dK_ref,
                                                    amax_dV_ref,
                                                    dQ_ref,
                                                    dK_ref,
                                                    dV_ref);

        /// \todo remove once backward mha is ready to execute
        if(disable_verification)
            return;

        checkAmax("AMax_dS", amax_dS_ref);
        checkAmax("AMax_dQ", amax_dQ_ref);
        checkAmax("AMax_dK", amax_dK_ref);
        checkAmax("AMax_dV", amax_dV_ref);

        checkTensor("dQ", dQ_ref);
        checkTensor("dK", dK_ref);
        checkTensor("dV", dV_ref);
    }
};

} // end namespace mha_graph_test
  //
using namespace mha_graph_test;

TEST_P(GPU_MhaBwdGraphTest_FP32, MhaBwdGraph) { Run(MhaDir::Bwd); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_MhaBwdGraphTest_FP32,
                         testing::Combine(testing::ValuesIn(std::vector<std::size_t>{2}),     // n
                                          testing::ValuesIn(std::vector<std::size_t>{8}),     // h
                                          testing::ValuesIn(std::vector<std::size_t>{4, 64}), // s
                                          testing::ValuesIn(std::vector<std::size_t>{16}),    // d
                                          testing::ValuesIn({0.0f, 0.5f}) // mProbDropout
                                          ));
