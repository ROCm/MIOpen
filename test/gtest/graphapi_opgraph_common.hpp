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
#pragma once

#include <gtest/gtest.h>

#include <miopen/miopen.h>
#include <miopen/errors.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <miopen/graphapi/util.hpp>

#include <unordered_map>

namespace graphapi_opgraph_tests {

namespace gr = miopen::graphapi;

inline std::unique_ptr<gr::PatternGraphGenerator> makeDiamondGraph()
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

    return gr::PatternGraphGenerator::Make({{"top", {"t_in"}, {"t_a", "t_b"}},
                                            {"left", {"t_a"}, {"t_c"}},
                                            {"right", {"t_b"}, {"t_d"}},
                                            {"bottom", {"t_c", "t_d"}, {"t_out"}}});
}

} // end namespace graphapi_opgraph_tests
