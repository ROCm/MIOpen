
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
#include <miopen/graphapi/engine.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <miopen/graphapi/util.hpp>

namespace miopen {
namespace graphapi {

GraphPattern::~GraphPattern() = default;

class MHA_FP8_Pattern : public GraphPattern
{
    static const OpGraph& getPatternGraph()
    {
        static auto graph_gen = PatternGraphGenerator::Make({
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

        return graph_gen->graph();
    }

    std::shared_ptr<TensorInfoMap> extractFind20Tensors(OpGraph* graph) {

      std::vector<int64_t> all1s = {1ll, 1ll, 1ll, 1ll};

      auto tensor_map = std::make_shared<TensorInfoMap>();

      auto add_mapping = [&] (miopenTensorArgumentId_t enum_id, Tensor* tens_ptr) {
        assert(tens_ptr);
        assert(enum_id != miopenTensorArgumentIdInvalid);

        tensor_map->emplace(tens_ptr->getId(), enum_id, tens_ptr);
      };

      for (auto [neigh, tens_ptr] : graph->getOutEdges(graph->getSourceNode())) {

        if (neigh->signName() == "OP_MATMUL") {
          auto* matmul = dynamic_cast<OperationMatmul*>(neigh);
          assert(matmul);


          if (auto* pw_prev = graph->findInNeighByName(matmul, "OP_POINTWISE:MUL"); pw_prev  == nullptr) {
            // this is the first matmul node
            add_mapping(miopenTensorMhaQ, matmul->getA());
            add_mapping(miopenTensorMhaK, matmul->getB());
            // TODO: dim check on Q and K

            auto* pw_0 = dynamic_cast<OperationPointwise*>(graph->findOutNeighByName(matmul, "OP_POINTWISE:MUL"));
            assert(pw_0);

            auto* attn_scl = pw_0->getB();
            assert(attn_scl->getDims() == all1s);
            add_mapping(miopenTensorMhaAttnScale, attn_scl);

            auto* pw_1 = dynamic_cast<OperationPointwise*>(graph->findOutNeighByName(pw_0, "OP_POINTWISE:MUL"));
            assert(pw_1);
            auto dscl_q = pw_1->getB();
            assert(dscl_q->getDims() == all1s);
            add_mapping(miopenTensorMhaDescaleQ, dscl_q);

            auto* pw_2 = dynamic_cast<OperationPointwise*>(graph->findOutNeighByName(pw_1, "OP_POINTWISE:MUL"));
            assert(pw_2);
            auto* dscl_k = pw_2->getB();
            assert(dscl_k->getDims() == all1s);
            add_mapping(miopenTensorMhaDescaleK, dscl_k);

            auto* red = dynamic_cast<OperationReduction*>(graph->findOutNeighByName(pw_2, "OP_REDUCTION:MAX"));
            assert(red);
            auto* m = red->getY();
            assert(m->getDims()[2] == 1ll);
            add_mapping(miopenTensorMhaM, m);

          } else {
            // this is the second matmul node
            add_mapping(miopenTensorMhaV, matmul->getB());

            auto* scl_s = pw_prev->getB();
            assert(scl_s->getDims() == all1s);
            add_mapping(miopenTensorMhaScaleS, scl_s);

            auto* pw_0 = dynamic_cast<OperationPointwise*>(graph->findOutNeighByName(matmul, "OP_POINTWISE:MUL"));
            assert(pw_0);

            auto* dscl_s = pw_0->getB();
            assert(dscl_s->getDims() == all1s);
            add_mapping(miopenTensorMhaDescaleS, dscl_s);

            auto* pw_1 = dynamic_cast<OperationPointwise*>(graph->findOutNeighByName(pw_0, "OP_POINTWISE:MUL"));
            assert(pw_1);
            auto* dscl_v = pw_1->getB();
            assert(dscl_v->getDims() == all1s);
            add_mapping(miopenTensorMhaDescaleV, dscl_v);

            auto* red = dynamic_cast<OperationReduction*>(graph->findOutNeighByName(pw_1, "OP_REDUCTION:MAX"));
            assert(red);
            auto* amax_o = red->getY();
            assert(m->getDims()[2] == 1ll);
            add_mapping(miopenTensorMhaAmaxO, amax_o);

            auto* pw_2 = dynamic_cast<OperationPointwise*>(graph->findOutNeighByName(pw_1, "OP_POINTWISE:MUL"));
            assert(pw_2);
            auto* scl_o = pw_2->getB();
            assert(scl_o->getDims() == all1s);
            add_mapping(miopenTensorMhaScaleO, scl_o);

            auto* o = pw_2->getY();
            assert(o->getDims() == all1s);
            add_mapping(miopenTensorMhaO, o);
          }

        } else if (neigh->signName() == "OP_RNG") {
          auto* rng = dynamic_cast<OperationRng*>(neigh);
          assert(rng);
          add_mapping(miopenTensorMhaDropoutSeed, std::get<Tensor*>(rng->getSeed()));
          add_mapping(miopenTensorMhaDropoutOffset, rng->getOffset());
          add_mapping(miopenTensorMhaDropoutProbability, rng->getRng()->getBernoulliProb());

        }
      }

      { // discovering Z_INV and AMAX_S tensors
        auto* exp_node = graph->findNodeByName("OP_POINTWISE:EXP");
        assert(exp_node);

        // get exp_node's neighbor that is a Pointwise mult
        auto* pw_mult = dynamic_cast<OperationPointwise*>(graph->findOutNeighByName(exp_node, "OP_POINTWISE:MUL"));
        assert(pw_mult);

        auto* red = dynamic_cast<OperationReduction*>(graph->findOutNeighByName(pw_mult, "OP_REDUCTION:MAX"));
        assert(red);

        add_mapping(miopenTensorMhaAmaxS, red->getY());

        auto* inv_node = dynamic_cast<OperationPointwise*>(graph->findNodeByName("OP_POINTWISE:RECIPROCAL"));
        add_mapping(miopenTensorMhaZInv, inv_node->getY());

      }


      return tensor_map;
    }

public:
    static std::unique_ptr<GraphPatternMatcher> Make() { return std::make_unique<MHA_FP8_Pattern>(); }

    bool matches(const OpGraph* graph) const final  {
      assert(graph);
      return isIsomorphic(*graph, getPatternGraph());
    }

    std::vector<Engine> getEngines(OpGraph* graph) const override {

      assert(graph);
      assert(matches(graph));


      miopenProblem_t mha_prob;
      MhaDescriptor mha_desc;

      float scale = 1.0; // TODO: get attention scale

      mha_desc.SetParams(scale);

      auto s =miopenCreateMhaProblem(&mha_prob, &mha_desc, miopenProblemDirectionForward);
      MIOPEN_THROW_IF(s != miopenStatusSuccess, "failed while creating problem for mha fwd");

      std::shared_ptr<TensorInfoMap> tensor_map = extractFind20Tensors(*graph);

      for (auto& [k, v] : *tensor_map) {
        s = miopenSetProblemTensorDescriptor(mha_prob, v.mEnumId, &(v.mTensDesc));
        MIOPEN_THROW_IF(s != miopenStatusSuccess, "failed while setting tensor descriptor for mha fwd");
      }

      std::vector<miopenSolution_t> solutions(10);
      size_t num_found = 0; 
      s = miopenFindSolutions(graph->getHandle(), mha_prob, solutions.data(), &num_found, solutions.size());
      MIOPEN_THROW_IF(s != miopenStatusSuccess, "failed while finding solutions for mha fwd");

      solutions.resize(num_found);

      std::vector<Engine> engines;

      size_t i = 0;
      for(const auto& sol : solutions) {
        auto exec = GraphExecutorFind20::make(sol, tensor_map);

        engines.emplace_back(EngineBuilder()
            .setGraph(graph)
            .setExecutor(sol)
            .setGlobalIndex(i)
            .build());
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

std::vector<Engine> findEngines(const OpGraph& graph)
{

    std::vector<std::unique_ptr<GraphPattern>> patterns; 
    patterns.emplace_back(MHA_FP8_Pattern::Make());

    for(const auto& p : patterns)
    {
        if(p->matches(graph))
        {
            found = true;
            return p->getEngines(graph);
        }
    }

    MIOPEN_THROW("Solution not found");
    return {};
}

} // end namespace graphapi
} // end namespace miopen
