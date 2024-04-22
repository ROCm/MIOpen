
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

#include <miopen/errors.hpp>
#include <miopen/graphapi/opgraph.hpp>

namespace miopen {
namespace graphapi {

GraphPattern::~GraphPattern() = default;

class MHA_FP8_Pattern : public GraphPattern
{
    static const OpGraph& getPatternGraph()
    {
        static auto graph_gen = DummyOpGraphGenerator::Make({
            {"OP_MATMUL", {"Q", "K"}, {"T_BMM_0"}},
            {"OP_POINTWISE:SCALE", {"T_BMM_0", "ATTN_S"}, {"PW_S_0"}},
            {"OP_POINTWISE:SCALE", {"PW_S_0", "DSCL_Q"}, {"PW_S_1"}},
            {"OP_POINTWISE:SCALE", {"PW_S_1", "DSCL_K"}, {"PW_S_2"}},
            {"OP_REDUCTION:MAX", {"PW_S_2"}, {"M"}},
            {"OP_POINTWISE:SUB", {"PW_S_2", "M"}, {"T_SUB"}},
            {"OP_POINTWISE:EXP", {"T_SUB"}, {"T_EXP"}},
            {"OP_REDUCTION:SUM", {"T_SUB"}, {"T_SUM"}},
            {"OP_POINTWISE:RECIPROCAL", {"T_SUB"}, {"Z_INV"}},
            {"OP_POINTWISE:MUL", {"Z_INV", "T_EXP"}, {"T_MUL_0"}},
            {"OP_REDUCTION:MAX", {"T_MUL_0"}, {"AMAX_S"}},
            {"OP_RNG", {"SEED", "OFFSET"}, {"T_RND"}},
            {"OP_POINTWISE:MUL", {"T_RND", "T_MUL_0"}, {"T_MUL_1"}},
            {"OP_POINTWISE:SCALE", {"T_MUL_1", "I_PROB"}, {"PW_S_3"}},
            {"OP_POINTWISE:SCALE", {"PW_S_3", "SCL_S"}, {"PW_S_4"}},
            {"OP_POINTWISE:SCALE", {"PW_S_3", "SCL_S"}, {"PW_S_4"}},
            {"OP_MATMUL", {"PW_S_4", "V"}, {"T_BMM_1"}},
            {"OP_POINTWISE:SCALE", {"T_BMM_1", "DSCL_S"}, {"PW_S_5"}},
            {"OP_POINTWISE:SCALE", {"PW_S_5", "DSCL_V"}, {"PW_S_6"}},
            {"OP_POINTWISE:SCALE", {"PW_S_6", "SCL_O"}, {"O"}},
            {"OP_REDUCTION:MAX", {"PW_S_6"}, {"AMAX_O"}},

        });
    }

public:
    static std::unique_ptr<GraphPattern> Make() { return std::make_unique<MHA_FP8_Pattern>(); }
};

class FwdConvResAddBiasActPattern : public GraphPattern
{
public:
    static std::unique_ptr<GraphPattern> Make()
    {
        return std::make_unique<FwdConvResAddBiasActPattern>();
    }
};

std::vector<GraphEngine> findSolution(const OpGraph& graph)
{

    std::vector<std::unique_ptr<GraphPattern>> patterns = {MHA_FP8_Pattern::Make(),
                                                           FwdConvResAddBiasActPattern::Make()};

    bool found = false;

    std::vector<GraphEngine> ret;
    for(const auto& p : patterns)
    {
        if(p->matches(graph))
        {
            found = true;
            ret   = p->getEngines(graph);
            break;
        }
    }

    if(!found)
    {
        MIOPEN_THROW("Solution not found");
    }

    return ret;
}

} // end namespace graphapi
} // end namespace miopen
