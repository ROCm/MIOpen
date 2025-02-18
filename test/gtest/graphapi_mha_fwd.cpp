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

class GPU_MhaFwdGraph_FP32 : public MhaGraphTestBase
{
protected:
    void createMhaGraph(size_t n, size_t h, size_t s, size_t d) override
    {

        mGraphBuilder = std::make_unique<gr::OpGraphBuilder>();

        std::vector<size_t> nhsd  = {n, h, s, d};
        std::vector<size_t> nhss  = {n, h, s, s};
        std::vector<size_t> nhs1  = {n, h, s, 1};
        std::vector<size_t> all1s = {1, 1, 1, 1};

        MAKE_TENSOR_F(Q, nhsd, false);
        MAKE_TENSOR_F(K, nhsd, false);
        MAKE_TENSOR_F(V, nhsd, false);

        MAKE_TENSOR_F(T_MM_0, nhss, true);
        addNode("OP_MATMUL", {Q, K}, {T_MM_0});

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
        addNode("OP_REDUCTION:MAX", {T_SCL_2}, {M});

        MAKE_TENSOR_F(T_SUB, nhss, true);
        addNode("OP_POINTWISE:SUB", {T_SCL_2, M}, {T_SUB});

        MAKE_TENSOR_F(T_EXP, nhss, true);
        addNode("OP_POINTWISE:EXP", {T_SUB}, {T_EXP});

        MAKE_TENSOR_F(T_SUM, nhs1, true);
        addNode("OP_REDUCTION:ADD", {T_EXP}, {T_SUM});

        MAKE_TENSOR_F(Z_INV, nhs1, false);
        addNode("OP_POINTWISE:RECIPROCAL", {T_SUM}, {Z_INV});

        MAKE_TENSOR_F(T_MUL_0, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_EXP, Z_INV}, {T_MUL_0});

        MAKE_TENSOR_F(AMAX_S, all1s, false);
        addNode("OP_REDUCTION:MAX", {T_MUL_0}, {AMAX_S});

        MAKE_TENSOR_I(RND_SD, all1s, false);
        MAKE_TENSOR_I(RND_OFF, all1s, false);

        MAKE_TENSOR_F(T_RND, nhss, true);
        addNode("OP_RNG", {RND_SD, RND_OFF}, {T_RND});

        MAKE_TENSOR_F(T_MUL_1, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_MUL_0, T_RND}, {T_MUL_1});

        MAKE_TENSOR_F(RND_PRB, all1s, false); // TODO(Amber): revisit
        MAKE_TENSOR_F(T_SCL_3, nhss, true);
        addNode("OP_POINTWISE:MUL", {T_MUL_1, RND_PRB}, {T_SCL_3});

        MAKE_TENSOR_F(T_SCL_4, nhss, true);
        MAKE_TENSOR_F(SCL_S, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_3, SCL_S}, {T_SCL_4});

        MAKE_TENSOR_F(T_MM_1, nhsd, true);
        addNode("OP_MATMUL", {T_SCL_4, V}, {T_MM_1});

        MAKE_TENSOR_F(T_SCL_5, nhsd, true);
        MAKE_TENSOR_F(DSCL_S, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_MM_1, DSCL_S}, {T_SCL_5});

        MAKE_TENSOR_F(T_SCL_6, nhsd, true);
        MAKE_TENSOR_F(DSCL_V, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_5, DSCL_V}, {T_SCL_6});

        MAKE_TENSOR_F(O, nhsd, false);
        MAKE_TENSOR_F(SCL_O, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_6, SCL_O}, {O});

        MAKE_TENSOR_F(AMAX_O, all1s, false);
        addNode("OP_REDUCTION:MAX", {T_SCL_6}, {AMAX_O});
    }

    void initInputs(size_t n, size_t h, size_t s, size_t d) override
    {
        using namespace test::cpu;

        auto Q = GenScaledTensor<float>(n, h, s, d);
        auto K = GenScaledTensor<float>(n, h, s, d);
        auto V = GenScaledTensor<float>(n, h, s, d);

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
            else if(k == "RND_PRB")
            {
                v.init(mProbDropout);
            }
            else if(k == "RND_SD" || k == "RND_OFF")
            {
                v.init(0ll);
            }
            else if(k == "M" || k == "O" || k == "Z_INV" || k == "AMAX_O" || k == "AMAX_S")
            {
                // these are outputs
                v.init(0.0f);
            }
            else
            {
                FAIL() << "Uninitialized input or output: " << k;
            }
        }
    }

    void runCpuVerify(size_t n, size_t h, size_t s, size_t d) override
    {

        CpuMhaFwdOut out = runCpuMhaFWd(n, h, s, d);

        checkAmax("AMAX_S", out.mAmaxS);
        checkAmax("AMAX_O", out.mAmaxO);

        checkTensor("M", out.mM);
        checkTensor("Z_INV", out.mZinv);
        checkTensor("O", out.mO);
    }
};

} // end namespace mha_graph_test

using namespace mha_graph_test;

TEST_P(GPU_MhaFwdGraph_FP32, MhaFwdGraph) { Run(MhaDir::Fwd); }

INSTANTIATE_TEST_SUITE_P(Unit,
                         GPU_MhaFwdGraph_FP32,
                         testing::Combine(testing::ValuesIn(std::vector<std::size_t>{2}),     // n
                                          testing::ValuesIn(std::vector<std::size_t>{8}),     // h
                                          testing::ValuesIn(std::vector<std::size_t>{4, 64}), // s
                                          testing::ValuesIn(std::vector<std::size_t>{16}),    // d
                                          testing::ValuesIn({0.0f, 0.5f}) // mProbDropout
                                          ));
