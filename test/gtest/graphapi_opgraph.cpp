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
#include <gtest/gtest.h>

#include <miopen/miopen.h>

#include <miopen/graphapi/opgraph.h>

namespace gr = miopen::graphapi;

template <bool IsVirtual>
gr::Tensor makeDummyTensor(std::string_view name)
{
    assert(name.size() <= sizeof(int64_t));
    int64_t id = 0;
    std::copy(name.begin(), name.end(), reinterpret_cast<char*>(&id));

    return gr::TensorBuilder{}
        .setDataType(miopenFloat)
        .setDim({1})
        .setStride({1})
        .setId(id)
        .setVirtual(IsVirtual)
        .build();
}

struct DummyNode : public gr::OpNode
{
    std::string mName;
    std::vector<gr::Tensor*> mInTensors;
    std::vector<gr::Tensor*> mOutTensors;

    DummyNode(const char* name,
              std::initializer_list<gr::Tensor*> ins,
              std::initializer_list<gr::Tensor*> outs)
        : mName(name), mInTensors(ins), mOutTensors(outs)
    {
    }

    const std::string& signName() const final { return mName; }

    std::vector<gr::Tensor*> getInTensors() const final { return mInTensors; }

    std::vector<gr::Tensor*> getOutTensors() const final { return mOutTensors; }
};

TEST(GraphAPI, BuildDiamond)
{

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

    auto t_in  = makeDummyTensor<false>("t_in");
    auto t_out = makeDummyTensor<false>("t_out");

    auto t_a = makeDummyTensor<true>("t_a");
    auto t_b = makeDummyTensor<true>("t_b");
    auto t_c = makeDummyTensor<true>("t_c");
    auto t_d = makeDummyTensor<true>("t_d");

    gr::OpGraphBuilder graph_builder;

    DummyNode top{"top", {&t_in}, {&t_a, &t_b}};
    DummyNode left{"left", {&t_a}, {&t_c}};
    DummyNode right{"right", {&t_b}, {&t_d}};
    DummyNode bottom{"bottom", {&t_c, &t_d}, {&t_out}};

    graph_builder.addNode(&top);
    graph_builder.addNode(&left);
    graph_builder.addNode(&right);
    graph_builder.addNode(&bottom);

    gr::OpGraph graph = std::move(graph_builder).build();

    ASSERT_TRUE(graph.hasNode(&top));
    ASSERT_TRUE(graph.hasNode(&left));
    ASSERT_TRUE(graph.hasNode(&right));
    ASSERT_TRUE(graph.hasNode(&bottom));

    ASSERT_TRUE(graph.hasEdge(&top, &t_a, &left));
    ASSERT_TRUE(graph.hasEdge(&top, &t_b, &right));
    ASSERT_TRUE(graph.hasEdge(&left, &t_c, &bottom));
    ASSERT_TRUE(graph.hasEdge(&right, &t_d, &bottom));
}
