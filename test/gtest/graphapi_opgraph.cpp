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

TEST(GraphAPI, BuildDiamond)
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

    DiamondGraphHolder d{};

    ASSERT_TRUE(d.graph.hasNode(&d.top));
    ASSERT_TRUE(d.graph.hasNode(&d.left));
    ASSERT_TRUE(d.graph.hasNode(&d.right));
    ASSERT_TRUE(d.graph.hasNode(&d.bottom));

    ASSERT_TRUE(d.graph.hasEdge(&d.top, &d.t_a, &d.left));
    ASSERT_TRUE(d.graph.hasEdge(&d.top, &d.t_b, &d.right));
    ASSERT_TRUE(d.graph.hasEdge(&d.left, &d.t_c, &d.bottom));
    ASSERT_TRUE(d.graph.hasEdge(&d.right, &d.t_d, &d.bottom));

    ASSERT_TRUE(d.graph.numNodes() == 4);
    ASSERT_TRUE(d.graph.numEdges() == 4);
    ASSERT_TRUE(d.graph.hasEdgeToSink(&d.bottom, &d.t_out));
    ASSERT_TRUE(d.graph.hasEdgeFromSource(&d.top, &d.t_in));
}
