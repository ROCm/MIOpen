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

struct DummyNode: public gr::OpNode {
  std::string name;
  // TODO: add in and out tensors

  DummyNode(

  virtual const std::string& SignName() const override { return name; }
};

TEST(GraphAPI, BuildDiamond) {

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

  gr::OpGraphBuilder graph_builder;

  DummyNode top{"top"};
  DummyNode left{"left"};
  DummyNode right{"right"};
  DummyNode bottom{"bottom"};


  graph_builder.addNode(&top);
  graph_builder.addNode(&left);
  graph_builder.addNode(&right);
  graph_builder.addNode(&bottom);

  gr::OpGraph graph = std::move(graph_builder).build();

  ASSERT(graph.hasNode(&top));
  ASSERT(graph.hasNode(&left));
  ASSERT(graph.hasNode(&right));
  ASSERT(graph.hasNode(&bottom));

  ASSERT(graph.hasEdge(&top, &t_a, &left));
  ASSERT(graph.hasEdge(&top, &t_b, &right));
  ASSERT(graph.hasEdge(&left, &t_c, &bottom));
  ASSERT(graph.hasEdge(&right, &t_d, &bottom));

}

