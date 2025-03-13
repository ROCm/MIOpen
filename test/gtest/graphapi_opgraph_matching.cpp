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

TEST(CPU_GraphMatchingAPI_NONE, DiamondGraphMatch)
{
    using namespace graphapi_opgraph_tests;

    auto dg1 = makeDiamondGraph();

    ASSERT_TRUE(gr::isIsomorphic(dg1->graph(), dg1->graph()));

    {
        // create identical copy
        auto dg2 = makeDiamondGraph();

        ASSERT_TRUE(gr::isIsomorphic(dg1->graph(), dg2->graph()));
    }

    {
        // create a mirror copy
        auto dg3 = gr::PatternGraphGenerator::Make({{"top", {"t_in"}, {"t_a", "t_b"}},
                                                    {"left", {"t_b"}, {"t_d"}},
                                                    {"right", {"t_a"}, {"t_c"}},
                                                    {"bottom", {"t_c", "t_d"}, {"t_out"}}});

        ASSERT_TRUE(gr::isIsomorphic(dg1->graph(), dg3->graph()));
    }

    {
        // remove one of the edges to bottom
        auto dg4 = gr::PatternGraphGenerator::Make({{"top", {"t_in"}, {"t_a", "t_b"}},
                                                    {"left", {"t_b"}, {"t_d"}},
                                                    {"right", {"t_a"}, {"t_c"}},
                                                    {"bottom", {"t_c"}, {"t_out"}}});
        ASSERT_FALSE(gr::isIsomorphic(dg1->graph(), dg4->graph()));
    }

    {
        // remove one of the edges out of top
        auto dg5 = gr::PatternGraphGenerator::Make({{"top", {"t_in"}, {"t_a"}},
                                                    {"left", {"t_b"}, {"t_d"}},
                                                    {"right", {"t_a"}, {"t_c"}},
                                                    {"bottom", {"t_c", "t_d"}, {"t_out"}}});
        ASSERT_FALSE(gr::isIsomorphic(dg1->graph(), dg5->graph()));
    }
}
