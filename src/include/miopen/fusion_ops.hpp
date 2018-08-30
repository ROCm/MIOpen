/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#include <string>
#include <unordered_map>
#include <vector>

namespace miopen {

// Supported operators
enum miopenFusionOp_t
{
    miopenFusionOpConvForward        = 0,
    miopenFusionOpActivForward       = 1,
    miopenFusionOpBatchNormInference = 2,
    miopenFusionOpBiasForward        = 3,
};

enum MDGraph_op_t
{
    OpEqual,
    OpNotEqual,
    OpAny,
};

struct EdgeOp
{
    template <class U, class V>
    EdgeOp(U v, V r = true, MDGraph_op_t o = OpAny)
        : val(std::to_string(v)), result(std::to_string(r)), op(o){};
    std::string val    = "";
    std::string result = "";
    MDGraph_op_t op    = OpAny;
};

using FusionMDGraph_Edge_Map     = std::unordered_map<std::string, EdgeOp>;
using FusionMDGraph_Edge_Map_Vec = std::vector<FusionMDGraph_Edge_Map>;

} // namespace miopen
