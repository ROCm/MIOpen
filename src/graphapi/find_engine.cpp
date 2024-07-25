
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
#include <miopen/miopen.h>
#include <miopen/graphapi/engine.hpp>
#include <miopen/graphapi/matmul.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <miopen/graphapi/pointwise.hpp>
#include <miopen/graphapi/reduction.hpp>
#include <miopen/graphapi/rng.hpp>
#include <miopen/graphapi/util.hpp>
#include <miopen/graphapi/variant_pack.hpp>

namespace miopen {
namespace graphapi {

GraphPatternMatcher::~GraphPatternMatcher() = default;

class MHA_Fwd_F8_Pattern : public GraphPatternMatcher
{
    static const OpGraph& getPatternGraph()
    {
        static auto graph_gen = PatternGraphGenerator::Make({
            {"OP_MATMUL", {"Q", "K"}, {"T_BMM_0"}},
            // {"OP_POINTWISE:MUL", {"T_BMM_0", "ATTN_S"}, {"PW_S_0"}},
            // replacing the MUL node above with IDENTITY node below for now
            // because Find 2.0 mha descriptor expects attention scale as a scalar
            // parameter on host side
            {"OP_POINTWISE:IDENTITY", {"T_BMM_0"}, {"PW_S_0"}},
            {"OP_POINTWISE:MUL", {"PW_S_0", "DSCL_Q"}, {"PW_S_1"}},
            {"OP_POINTWISE:MUL", {"PW_S_1", "DSCL_K"}, {"PW_S_2"}},
            {"OP_REDUCTION:MAX", {"PW_S_2"}, {"M"}},
            {"OP_POINTWISE:SUB", {"PW_S_2", "M"}, {"T_SUB"}},
            {"OP_POINTWISE:EXP", {"T_SUB"}, {"T_EXP"}},
            {"OP_REDUCTION:ADD", {"T_EXP"}, {"T_SUM"}},
            {"OP_POINTWISE:RECIPROCAL", {"T_SUM"}, {"Z_INV"}},
            {"OP_POINTWISE:MUL", {"Z_INV", "T_EXP"}, {"T_MUL_0"}},
            {"OP_REDUCTION:MAX", {"T_MUL_0"}, {"AMAX_S"}},
            {"OP_RNG", {"SEED", "OFFSET"}, {"T_RND"}},
            {"OP_POINTWISE:MUL", {"T_RND", "T_MUL_0"}, {"T_MUL_1"}},
            {"OP_POINTWISE:MUL", {"T_MUL_1", "I_PROB"}, {"PW_S_3"}},
            {"OP_POINTWISE:MUL", {"PW_S_3", "SCL_S"}, {"PW_S_4"}},
            {"OP_MATMUL", {"PW_S_4", "V"}, {"T_BMM_1"}},
            {"OP_POINTWISE:MUL", {"T_BMM_1", "DSCL_S"}, {"PW_S_5"}},
            {"OP_POINTWISE:MUL", {"PW_S_5", "DSCL_V"}, {"PW_S_6"}},
            {"OP_POINTWISE:MUL", {"PW_S_6", "SCL_O"}, {"O"}},
            {"OP_REDUCTION:MAX", {"PW_S_6"}, {"AMAX_O"}},

        });

        return graph_gen->graph();
    }

    std::shared_ptr<TensorInfoMap> extractFind20Tensors(const OpGraph& graph,
                                                        float* attn_scale) const
    {

        assert(attn_scale);
        std::vector<std::size_t> all1s = {
            std::size_t{1}, std::size_t{1}, std::size_t{1}, std::size_t{1}};

        auto tensor_map = std::make_shared<TensorInfoMap>();

        auto add_mapping = [&](miopenTensorArgumentId_t enum_id, Tensor* tens_ptr) {
            assert(tens_ptr);
            assert(enum_id != miopenTensorArgumentIdInvalid);

            /*
             * leaving to check that correct tensors are picked
            MIOPEN_LOG_W("Tensor enum id: " << tensorEnumIdToStr(enum_id)
                                            << " tensor unique id as str: "
                                            << tensorIdAsStr(tens_ptr));
            */

            tensor_map->try_emplace(tens_ptr->getId(), TensorInfo(enum_id, tens_ptr));
        };

        for(auto [neigh, tens_ptr] : graph.getOutEdges(graph.getSourceNode()))
        {

            if(neigh->signName() == "OP_MATMUL")
            {
                auto* matmul = dynamic_cast<OperationMatmul*>(neigh);
                assert(matmul);

                if(auto* pw_prev = graph.findInNeighByName(matmul, "OP_POINTWISE:MUL");
                   pw_prev == nullptr)
                {
                    // this is the first matmul node
                    add_mapping(miopenTensorMhaQ, matmul->getA());
                    add_mapping(miopenTensorMhaK, matmul->getB());
                    /// \todo dim check on Q and K --amberhassaan May, 2024

                    /// \note  old code assuming attn_scale applies to pw_mult
                    // auto* pw_0 = dynamic_cast<OperationPointwise*>(
                    // graph.findOutNeighByName(matmul, "OP_POINTWISE:MUL"));
                    auto* pw_0 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(matmul, "OP_POINTWISE:IDENTITY"));
                    assert(pw_0);

                    float alpha1 = std::get<float>(pw_0->getAlpha1());
                    *attn_scale  = alpha1;
                    /// \note old code assuming attn_scale applies to pw_mult
                    // auto* attn_scl = pw_0->getB();
                    // assert(attn_scl->GetLengths() == all1s);
                    // add_mapping(miopenTensorMhaAttnScale, attn_scl);

                    auto* pw_1 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(pw_0, "OP_POINTWISE:MUL"));
                    assert(pw_1);
                    auto dscl_q = pw_1->getB();
                    assert(dscl_q->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaDescaleQ, dscl_q);

                    auto* pw_2 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(pw_1, "OP_POINTWISE:MUL"));
                    assert(pw_2);
                    auto* dscl_k = pw_2->getB();
                    assert(dscl_k->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaDescaleK, dscl_k);

                    auto* red = dynamic_cast<OperationReduction*>(
                        graph.findOutNeighByName(pw_2, "OP_REDUCTION:MAX"));
                    assert(red);
                    auto* m = red->getY();
                    assert(m->GetLengths()[3] == 1LL);
                    add_mapping(miopenTensorMhaM, m);
                }
                else
                {
                    // this is the second matmul node
                    add_mapping(miopenTensorMhaV, matmul->getB());

                    auto* pw_prev_cast = dynamic_cast<OperationPointwise*>(pw_prev);
                    assert(pw_prev_cast);
                    auto* scl_s = pw_prev_cast->getB();
                    assert(scl_s->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaScaleS, scl_s);

                    auto* pw_0 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(matmul, "OP_POINTWISE:MUL"));
                    assert(pw_0);

                    auto* dscl_s = pw_0->getB();
                    assert(dscl_s->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaDescaleS, dscl_s);

                    auto* pw_1 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(pw_0, "OP_POINTWISE:MUL"));
                    assert(pw_1);
                    auto* dscl_v = pw_1->getB();
                    assert(dscl_v->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaDescaleV, dscl_v);

                    auto* red = dynamic_cast<OperationReduction*>(
                        graph.findOutNeighByName(pw_1, "OP_REDUCTION:MAX"));
                    assert(red);
                    auto* amax_o = red->getY();
                    assert(amax_o->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaAmaxO, amax_o);

                    auto* pw_2 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(pw_1, "OP_POINTWISE:MUL"));
                    assert(pw_2);
                    auto* scl_o = pw_2->getB();
                    assert(scl_o->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaScaleO, scl_o);

                    auto* o = pw_2->getY();
                    add_mapping(miopenTensorMhaO, o);
                }
            }
            else if(neigh->signName() == "OP_RNG")
            {
                auto* rng = dynamic_cast<OperationRng*>(neigh);
                assert(rng);
                add_mapping(miopenTensorMhaDropoutSeed, std::get<Tensor*>(rng->getSeed()));
                add_mapping(miopenTensorMhaDropoutOffset, rng->getOffset());

                auto* pw_mult_0 = dynamic_cast<OperationPointwise*>(
                    graph.findOutNeighByName(rng, "OP_POINTWISE:MUL"));
                assert(pw_mult_0);

                auto* pw_mult_1 = dynamic_cast<OperationPointwise*>(
                    graph.findOutNeighByName(pw_mult_0, "OP_POINTWISE:MUL"));
                assert(pw_mult_1);

                auto* prob = pw_mult_1->getB();
                add_mapping(miopenTensorMhaDropoutProbability, prob);
            }
        }

        { // discovering Z_INV and AMAX_S tensors
            auto* exp_node = graph.findNodeByName("OP_POINTWISE:EXP");
            assert(exp_node);

            // get exp_node's neighbor that is a Pointwise mult
            auto* pw_mult = dynamic_cast<OperationPointwise*>(
                graph.findOutNeighByName(exp_node, "OP_POINTWISE:MUL"));
            assert(pw_mult);

            auto* red = dynamic_cast<OperationReduction*>(
                graph.findOutNeighByName(pw_mult, "OP_REDUCTION:MAX"));
            assert(red);

            add_mapping(miopenTensorMhaAmaxS, red->getY());

            auto* inv_node =
                dynamic_cast<OperationPointwise*>(graph.findNodeByName("OP_POINTWISE:RECIPROCAL"));
            add_mapping(miopenTensorMhaZInv, inv_node->getY());
        }

        return tensor_map;
    }

public:
    static std::unique_ptr<GraphPatternMatcher> Make()
    {
        return std::make_unique<MHA_Fwd_F8_Pattern>();
    }

    std::string_view name() const final
    {
        static const char* n = "mha_fwd_f8";
        return n;
    }

    bool matches(const OpGraph* graph_ptr) const final
    {
        assert(graph_ptr);
        return isIsomorphic(*graph_ptr, getPatternGraph());
    }

    std::vector<Engine> getEngines(OpGraph* graph_ptr) const override
    {

        assert(graph_ptr);
        assert(matches(graph_ptr));
        auto& graph = *graph_ptr;

        miopenProblem_t mha_prob;
        MhaDescriptor mha_desc;

        float attn_scale                          = std::numeric_limits<float>::quiet_NaN();
        std::shared_ptr<TensorInfoMap> tensor_map = extractFind20Tensors(graph, &attn_scale);
        assert(attn_scale != std::numeric_limits<float>::quiet_NaN());

        mha_desc.SetParams(attn_scale);

        auto s = miopenCreateMhaProblem(&mha_prob, &mha_desc, miopenProblemDirectionForward);
        MIOPEN_THROW_IF(s != miopenStatusSuccess, "failed while creating problem for mha fwd");

        for(auto& [k, v] : *tensor_map)
        {
            s = miopenSetProblemTensorDescriptor(mha_prob, v.mEnumId, v.mGraphTensor);
            MIOPEN_THROW_IF(s != miopenStatusSuccess,
                            "failed while setting tensor descriptor for mha fwd");
        }

        std::vector<miopenSolution_t> solutions(10);
        size_t num_found = 0;
        s                = miopenFindSolutions(
            graph.getHandle(), mha_prob, nullptr, solutions.data(), &num_found, solutions.size());
        MIOPEN_THROW_IF(s != miopenStatusSuccess, "failed while finding solutions for mha fwd");

        solutions.resize(num_found);

        std::vector<Engine> engines;

        size_t i = 0;
        for(const auto& sol : solutions)
        {
            std::shared_ptr<GraphPatternExecutor> exec = GraphExecutorFind20::make(sol, tensor_map);

            engines.emplace_back(
                EngineBuilder().setGraph(graph_ptr).setExecutor(exec).setGlobalIndex(i).build());
            ++i;
        }

        return engines;
    }
};

/*
class FwdConvResAddBiasActPattern : public GraphPattern
{
public:
    static std::unique_ptr<GraphPattern> Make()
    {
        return std::make_unique<FwdConvResAddBiasActPattern>();
    }
};
*/

std::vector<Engine> findEngines(OpGraph* graph)
{
    assert(graph);

    std::vector<std::unique_ptr<GraphPatternMatcher>> patterns;
    patterns.emplace_back(MHA_Fwd_F8_Pattern::Make());

    for(const auto& p : patterns)
    {
        if(p->matches(graph))
        {
            MIOPEN_LOG_I2("Matched against pattern: " << p->name());
            return p->getEngines(graph);
        }
    }

    return {};
}

} // end namespace graphapi
} // end namespace miopen
