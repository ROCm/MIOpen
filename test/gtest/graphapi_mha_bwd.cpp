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



namespace mha_graph_test {

class MhaFwdGraphTest : public MhaGraphTestBase
{

    void createMhaGraph(int64_t n, int64_t h, int64_t s, int64_t d)
    {
        mGraphBuilder = std::make_unique<gr::OpGraphBuilder>();

        std::vector<int64_t> nhsd  = {n, h, s, d};
        std::vector<int64_t> nhss  = {n, h, s, s};
        std::vector<int64_t> nhs1  = {n, h, s, 1};
        std::vector<int64_t> all1s = {1, 1, 1, 1};

        MAKE_TENSOR_F(Q, nhsd, false);
        MAKE_TENSOR_F(K, nhsd, false);
        MAKE_TENSOR_F(dO, nhsd, false);

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
        MAKE_TENSOR_F(T_SUB_0, nhss, true);
        addNode("OP_POINTWISE:SUB", {T_SCL_2, M}, {T_SUB_0});

        MAKE_TENSOR_F(T_EXP, nhss, true);
        addNode("OP_POINTWISE:EXP", {T_SUB_0}, {T_EXP});

        MAKE_TENSOR_F(Z_INV, nhs1, false);
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

        // NOTE(Amber): omitting the Reshape transpose node here
        MAKE_TENSOR_F(T_MM_1, nhsd, true);
        addNode("OP_MATMUL", {T_SCL_4, dO}, {T_MM_1});

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

        MAKE_TENSOR_F(T_MM_2, nhss, true);
        addNode("OP_MATMUL", {dO, V}, {T_MM_2});

        MAKE_TENSOR_F(T_SCL_7, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_MM_2, DSCL_dO}, {T_SCL_7});
        
        MAKE_TENSOR_F(T_SCL_8, nhsd, true);
        MAKE_TENSOR_F(DSCL_V, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_7, DSCL_V}, {T_SCL_8});

        MAKE_TENSOR_F(T_SCL_9, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_SCL_8, T_RND}, {T_SCL_9});

        MAKE_TENSOR_F(T_SCL_10, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_SCL_9, RND_PRB}, {T_SCL_10});

        ////////////////// Right-top //////////////////////////////////
        MAKE_TENSOR_F(T_SCL_11, nhsd, true);
        addNode("OP_POINTWISE:MUL", {dO, DSCL_dO}, {T_SCL_11});

        MAKE_TENSOR_F(T_SCL_12, nhsd, true);
        addNode("OP_POINTWISE:MUL", {O, DSCL_O}, {T_SCL_12});

        MAKE_TENSOR_F(T_MUL_2, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_SCL_11, T_SCL_12}, {T_MUL_2});

        MAKE_TENSOR_F(T_SCL_13, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_MUL_2, RND_PRB}, {T_SCL_13});

        MAKE_TENSOR_F(T_SUM_0, nhsd, true);
        addNode("OP_REDUCTION:ADD", {T_SCL_13}, {T_SUM_0});

        
        ////////////////// Center Part //////////////////////////////////
        MAKE_TENSOR_F(T_SUB_1, nhsd, true);
        addNode("OP_POINTWISE:SUB", {T_SCL_10, T_SUM_0}, {T_SUB_1});
        
        MAKE_TENSOR_F(T_SCL_14, nhsd, true);
        addUnaryPointwiseNode(pw_id, {T_SUB_1}, {T_SCL_14}, mAttentionScale);

        MAKE_TENSOR_F(T_MUL_3, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_SCL_14, T_SCL_3}, {T_MUL_3});

        MAKE_TENSOR_F(T_SCL_15, nhsd, true);
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

        //XXX(Amber): Reshape transpose node goes here
        MAKE_TENSOR_F(T_MM_4, nhsd, true);
        addNode("OP_MATMUL", {T_SCL_15, Q}, {T_MM_4});

        MAKE_TENSOR_F(T_SCL_18, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_MM_4, DSCL_dS}, {T_SCL_18});

        MAKE_TENSOR_F(T_SCL_19, nhsd, true);
        addNode("OP_POINTWISE:MUL", {T_SCL_18, DSCL_Q}, {T_SCL_19});

        MAKE_TENSOR_F(dQ, nhsd, false);
        MAKE_TENSOR_F(SCL_dK, all1s, false);
        addNode("OP_POINTWISE:MUL", {T_SCL_19, SCL_dK}, {dK});

        MAKE_TENSOR_F(AMax_dK, all1s, false);
        addNode("OP_REDUCTION:MAX", {T_SCL_19}, {AMax_dK});
    }

};

}// end namespace mha_graph_test
