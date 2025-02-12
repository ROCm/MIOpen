/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "graphapi_opgraph_common.hpp"

TEST(CPU_GraphAPI_NONE, BuildDiamond)
{
    using namespace graphapi_opgraph_tests;

    /*
     *       |
     *       | t_in
     *       v
     *      Top
     * t_a /   \ t_b
     *    /     \
     *   v       v
     *  Left    Right
     *    \      /
     * t_c \    / t_d
     *      v  v
     *     Bottom
     *       |
     *       |t_out
     *       v
     */

    gr::PatternGraphGenerator gen;
    using DummyNode = gr::PatternGraphGenerator::DummyNode;

    auto t_in  = gen.makeDummyTensor("t_in");
    auto t_out = gen.makeDummyTensor("t_out");

    auto t_a = gen.makeDummyTensor("t_a");
    auto t_b = gen.makeDummyTensor("t_b");
    auto t_c = gen.makeDummyTensor("t_c");
    auto t_d = gen.makeDummyTensor("t_d");

    miopenHandle_t handle = nullptr;
    miopenStatus_t status = miopenCreate(&handle);

    ASSERT_EQ(status, miopenStatusSuccess) << "Handle wasn't created";
    ASSERT_NE(handle, nullptr) << "miopenCreate() created a null handle";

    gr::OpGraphBuilder graph_builder;

    graph_builder.setHandle(handle);

    DummyNode top{"top", {t_in}, {t_a, t_b}};
    DummyNode left{"left", {t_a}, {t_c}};
    DummyNode right{"right", {t_b}, {t_d}};
    DummyNode bottom{"bottom", {t_c, t_d}, {t_out}};

    graph_builder.addNode(&top);
    graph_builder.addNode(&left);
    graph_builder.addNode(&right);
    graph_builder.addNode(&bottom);

    gr::OpGraph graph = std::move(graph_builder).build();

    ASSERT_TRUE(graph.hasNode(&top));
    ASSERT_TRUE(graph.hasNode(&left));
    ASSERT_TRUE(graph.hasNode(&right));
    ASSERT_TRUE(graph.hasNode(&bottom));

    ASSERT_TRUE(graph.hasEdge(&top, t_a, &left));
    ASSERT_TRUE(graph.hasEdge(&top, t_b, &right));
    ASSERT_TRUE(graph.hasEdge(&left, t_c, &bottom));
    ASSERT_TRUE(graph.hasEdge(&right, t_d, &bottom));

    ASSERT_TRUE(graph.numNodes() == 4);
    ASSERT_TRUE(graph.numEdges() == 4);
    ASSERT_TRUE(graph.hasEdgeToSink(&bottom, t_out));
    ASSERT_TRUE(graph.hasEdgeFromSource(&top, t_in));

    miopenDestroy(handle);
}
