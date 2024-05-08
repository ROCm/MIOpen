/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "test.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"

#include <gtest/gtest.h>

#include <miopen/graphapi/matmul.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <miopen/graphapi/pointwise.hpp>
#include <miopen/graphapi/reduction.hpp>
#include <miopen/graphapi/rng.hpp>
#include <miopen/graphapi/util.hpp>

#include <numeric>
#include <string>

namespace gr = miopen::graphapi;


namespace mha_graph_test {

class MhaFwdGraphTest: public testing::TestWithParam<std::tuple<int, int, int, int>> {

  std::unique_ptr<gr::OpGraphBuilder> mGraphBuilder;
  gr::OpGraph mGraph;
  gr::AutoDeleteAllocator mAlloc;


  gr::Tensor* makeTensor(std::string_view name, const std::vector<int64_t>& dims) {
    return mAlloc.allocate(gr::makeTensor<true>(name, dims));
  }

  auto* makePointWiseDesc(miopenPointwiseMode_t mode) {
      return mAlloc.allocate(gr::PointwiseBuilder{}
          .setMode(MIOPEN_POINTWISE_MUL)
          .setMathPrecision(miopenFloat)
          .build());
  }

  using TensorVec = std::vector<gr::Tensor*>;

  void addBinaryPointwiseNode( gr::Pointwise* pw,
      const TensorVec& in_tensors, 
      const TensorVec&  out_tensors) {

      assert(in_tensors.size() == 2);
      assert(out_tensors.size() == 1);

      mGraphBuilder->addNode(mAlloc.allocate(gr::OperationPointwiseBuilder{}
            .setPointwise(pw)
            .setX(in_tensors[0])
            .setB(in_tensors[1])
            .setY(out_tensors[0])
            .build()));

  }

  void addUnaryPointwiseNode( gr::Pointwise* pw,
      const TensorVec&  in_tensors, 
      const TensorVec&  out_tensors) {

      assert(in_tensors.size() == 1);
      assert(out_tensors.size() == 1);

      mGraphBuilder->addNode(mAlloc.allocate(gr::OperationPointwiseBuilder{}
            .setPointwise(pw)
            .setX(in_tensors[0])
            .setY(out_tensors[0])
            .build()));

  }

  void addReductionNode(miopenReduceTensorOp_t red_op,
      const TensorVec&  in_tensors, 
      const TensorVec&  out_tensors) {

      auto* red_desc = mAlloc.allocate(gr::ReductionBuilder{}
        .setCompType(miopenFloat)
        .setReductionOperator(red_op)
        .build());
      
      assert(in_tensors.size() == 1);
      assert(out_tensors.size() == 1);

      mGraphBuilder->addNode(mAlloc.allocate(gr::OperationReductionBuilder{}
            .setReduction(red_desc)
            .setX(in_tensors[0])
            .setY(out_tensors[0])
            .build()));
  }

  void addNode(std::string_view name, 
      const TensorVec&  in_tensors, 
      const TensorVec&  out_tensors) {
    
    if (name == "OP_MATMUL") {
      assert(in_tensors.size() == 2);
      assert(out_tensors.size() == 1);

      auto* mm_desc = mAlloc.allocate(gr::MatmulBuilder().setComputeType(miopenFloat8).build());
      mGraphBuilder->addNode(mAlloc.allocate(gr::OperationMatmulBuilder{}
        .setA(in_tensors[0])
        .setB(in_tensors[1])
        .setC(out_tensors[0])
        .setMatmulDescriptor(mm_desc)
        .build()));

    } else if (name == "OP_POINTWISE:MUL") {

      auto* pw = makePointWiseDesc(MIOPEN_POINTWISE_MUL);
      addBinaryPointwiseNode(pw, in_tensors, out_tensors);

    } else if (name == "OP_POINTWISE:SUB") {
      auto* pw = makePointWiseDesc(MIOPEN_POINTWISE_SUB);
      addBinaryPointwiseNode(pw, in_tensors, out_tensors);

    } else if (name == "OP_POINTWISE:EXP") {
      auto* pw = makePointWiseDesc(MIOPEN_POINTWISE_EXP);
      addUnaryPointwiseNode(pw, in_tensors, out_tensors);

    } else if (name == "OP_POINTWISE:RECIPROCAL") {
      auto* pw = makePointWiseDesc(MIOPEN_POINTWISE_RECIPROCAL);
      addUnaryPointwiseNode(pw, in_tensors, out_tensors);

    } else if (name == "OP_REDUCTION:MAX") {
      addReductionNode(MIOPEN_REDUCE_TENSOR_MAX, in_tensors, out_tensors);

    } else if (name == "OP_REDUCTION:SUM") {
      addReductionNode(MIOPEN_REDUCE_TENSOR_MAX, in_tensors, out_tensors);

    } else if (name == "OP_RNG") {
      constexpr double BERNOULLI = 0.5; 
      auto* rng_desc = mAlloc.allocate(gr::RngBuilder{}
          .setDistribution(MIOPEN_RNG_DISTRIBUTION_BERNOULLI)
          .setBernoulliProb(BERNOULLI)
          .build());

      assert(in_tensors.size() == 2);// first is seed tensor, second is offset
      assert(out_tensors.size() == 1);

      mGraphBuilder->addNode(mAlloc.allocate(gr::OperationRngBuilder{}
            .setRng(rng_desc)
            .setSeed(in_tensors[0])
            .setOffset(in_tensors[1])
            .setOutput(out_tensors[0])
            .build()));


    } else {
      std::cerr << "Unknown graph node type" << std::endl;
      std::abort();
    }

  }


  void createMhaGraph(int64_t n, int64_t h, int64_t s, int64_t d) {

    mGraphBuilder = std::make_unique<gr::OpGraphBuilder>();

    std::vector<int64_t> nhsd = {n, h, s, d};
    std::vector<int64_t> nhss = {n, h, s, s};
    std::vector<int64_t> nhs1 = {n, h, s, 1};
    std::vector<int64_t> all1s = {1, 1, 1, 1};

#define MAKE_TENSOR(name, dims) auto* name = makeTensor(#name, dims)

    // auto Q = makeTensor("Q", nhsd, false);
    // auto K = makeTensor("K", nhsd, false);
    // auto V = makeTensor("V", nhsd, false);
    MAKE_TENSOR(Q, nhsd);
    MAKE_TENSOR(K, nhsd);
    MAKE_TENSOR(V, nhsd);

    MAKE_TENSOR(T_MM_0, nhss);
    addNode("OP_MATMUL", {Q, K}, {T_MM_0});

    MAKE_TENSOR(T_SCL_0, nhss);
    MAKE_TENSOR(ATN_SCL, all1s);

    addNode("OP_POINTWISE:MUL", {T_MM_0, ATN_SCL}, {T_SCL_0});

    MAKE_TENSOR(T_SCL_1, nhss);
    MAKE_TENSOR(DSCL_Q, all1s);

    addNode("OP_POINTWISE:MUL", {T_SCL_0, DSCL_Q}, {T_SCL_1});

    MAKE_TENSOR(T_SCL_2, nhss);
    MAKE_TENSOR(DSCL_K, all1s);

    addNode("OP_POINTWISE:MUL", {T_SCL_1, DSCL_K}, {T_SCL_2});

    MAKE_TENSOR(M, nhs1);
    addNode("OP_REDUCTION:MAX", {T_SCL_2}, {M});

    MAKE_TENSOR(T_SUB, nhss);
    addNode("OP_POINTWISE:SUB", {T_SCL_2, M}, {T_SUB});

    MAKE_TENSOR(T_EXP, nhss);
    addNode("OP_POINTWISE:EXP", {T_SUB}, {T_EXP});

    MAKE_TENSOR(T_SUM, nhs1);
    addNode("OP_REDUCTION:SUM", {T_EXP}, {T_SUM});

    MAKE_TENSOR(Z_INV, nhs1);
    addNode("OP_POINTWISE:RECIPROCAL", {T_SUM}, {Z_INV});

    MAKE_TENSOR(T_MUL_0, nhss);
    addNode("OP_POINTWISE:MUL", {T_EXP, Z_INV}, {T_MUL_0});

    MAKE_TENSOR(AMAX_S, all1s);
    addNode("OP_REDUCTION:MAX", {T_MUL_0}, {AMAX_S});

    MAKE_TENSOR(RND_SD, all1s);
    MAKE_TENSOR(RND_OFF, all1s);

    MAKE_TENSOR(T_RND, nhss);
    addNode("OP_RNG", {RND_SD, RND_OFF}, {T_RND});

    MAKE_TENSOR(T_MUL_1, nhss);
    addNode("OP_POINTWISE:MUL", {T_MUL_0, T_RND}, {T_MUL_1});

    MAKE_TENSOR(RND_PRB, all1s); // TODO(Amber): revisit
    MAKE_TENSOR(T_SCL_3, nhss);
    addNode("OP_POINTWISE:MUL", {T_MUL_1, RND_PRB}, {T_SCL_3});

    MAKE_TENSOR(T_SCL_4, nhss);
    MAKE_TENSOR(SCL_S, all1s);
    addNode("OP_POINTWISE:MUL", {T_SCL_3, SCL_S}, {T_SCL_4});

    MAKE_TENSOR(T_MM_1, nhsd);
    addNode("OP_MATMUL", {T_SCL_4, V}, {T_MM_1});

    MAKE_TENSOR(T_SCL_5, nhsd);
    MAKE_TENSOR(DSCL_S, all1s);
    addNode("OP_POINTWISE:MUL", {T_MM_1, DSCL_S}, {T_SCL_5});

    MAKE_TENSOR(T_SCL_6, nhsd);
    MAKE_TENSOR(DSCL_V, all1s);
    addNode("OP_POINTWISE:MUL", {T_SCL_5, DSCL_V}, {T_SCL_6});

    MAKE_TENSOR(T_SCL_7, nhsd);
    MAKE_TENSOR(SCL_O, all1s);
    addNode("OP_POINTWISE:MUL", {T_SCL_6, SCL_O}, {T_SCL_7});

    MAKE_TENSOR(AMAX_O, all1s);
    addNode("OP_REDUCTION:MAX", {T_SCL_6}, {AMAX_O});

    mGraph = std::move(*mGraphBuilder).build(); 
    mGraphBuilder.reset(nullptr);

#undef MAKE_TENSOR

  }

  gr::OpNode* findNodeByName(const std::string& name) const {

    for (auto* n : mGraph->getNodes()) {
      if (n->signName() == name) {
        return n;
      }
    }
    return nullptr;
  }

  gr::OpNode* findOutNeighByName(gr::OpNode* node, const std::string& neigh_name) const {
    for (auto [m,t] : mGraph->getOutEdges(node)) {
      if (m->signName() == neigh_name) {
        return m;
      }
    }
    return nullptr;
  }

  gr::OpNode* findInNeighByName(gr::OpNode* node, const std::string& neigh_name) const {
    for (auto [m,t] : mGraph->getInEdges(node)) {
      if (m->signName() == neigh_name) {
        return m;
      }
    }
    return nullptr;
  }

  void extractFind20Tensors() {

    std::vector<int64_t> all1s = {1ll, 1ll, 1ll, 1ll};

    std::unordered_map<miopenTensorArgumentId_t, gr::Tensor*> tensor_map;

    for (auto [neigh, tens_ptr] : mGraph.GetOutEdges(mGraph.GetSourceNode())) {

      if (neigh->signName() == "OP_MATMUL") {
        auto* matmul = dynamic_cast<gr::OperationMatmul*>(neigh);
        assert(matmul);


        if (auto* pw_prev = findInNeighByName(matmul, "OP_POINTWISE:MUL"); pw_prev  == nullptr) {
          // this is the first matmul node
          tensor_map[miopenTensorMhaQ] = matmul->getA();
          tensor_map[miopenTensorMhaK] = matmul->getB();
          // TODO: dim check on Q and K

          auto* pw_0 = dynamic_cast<gr::OperationPointwise*>(findOutNeighByName(matmul, "OP_POINTWISE:MUL"));
          assert(pw_0);

          auto* attn_scl = pw_0->getB();
          assert(attn_scl->getDims() == all1s);
          tensor_map[miopenTensorMhaAttnScale] = attn_scl;

          auto* pw_1 = dynamic_cast<gr::OperationPointwise*>(findOutNeighByName(pw_0, "OP_POINTWISE:MUL"));
          assert(pw_1);
          auto dscl_q = pw_1->getB();
          assert(dscl_q->getDims() == all1s);
          tensor_map[miopenTensorMhaDescaleQ] = dscl_q;

          auto* pw_2 = dynamic_cast<gr::OperationPointwise*>(findOutNeighByName(pw_1, "OP_POINTWISE:MUL"));
          assert(pw_2);
          auto* dscl_k = pw_2->getB();
          assert(dscl_k->getDims() == all1s);
          tensor_map[miopenTensorMhaDescaleK] = dscl_k;

          auto* red = dynamic_cast<gr::OperationReduction*>(findOutNeighByName(pw_2, "OP_REDUCTION:MAX"));
          assert(red);
          auto* m = red->getY();
          assert(m->getDims()[2] == 1ll);
          tensor_map[miopenTensorMhaM] = m;

        } else {
          // this is the second matmul node
          tensor_map[miopenTensorMhaV] = matmul->getB();

          auto* scl_s = pw_prev->getB();
          assert(scl_s->getDims() == all1s);
          tensor_map[miopenTensorMhaScaleS] = scl_s;

          auto* pw_0 = dynamic_cast<gr::OperationPointwise*>(findOutNeighByName(matmul, "OP_POINTWISE:MUL"));
          assert(pw_0);

          auto* dscl_s = pw_0->getB();
          assert(dscl_s->getDims() == all1s);
          tensor_map[miopenTensorMhaDescaleS] = dscl_s;

          auto* pw_1 = dynamic_cast<gr::OperationPointwise*>(findOutNeighByName(pw_0, "OP_POINTWISE:MUL"));
          assert(pw_1);
          auto* dscl_v = pw_1->getB();
          assert(dscl_v->getDims() == all1s);
          tensor_map[miopenTensorMhaDescaleV] = dscl_v;

          auto* red = dynamic_cast<gr::OperationReduction*>(findOutNeighByName(pw_1, "OP_REDUCTION:MAX"));
          assert(red);
          auto* amax_o = red->getY();
          assert(m->getDims()[2] == 1ll);
          tensor_map[miopenTensorMhaAmaxO] = amax_o;

          auto* pw_2 = dynamic_cast<gr::OperationPointwise*>(findOutNeighByName(pw_1, "OP_POINTWISE:MUL"));
          assert(pw_2);
          auto* scl_o = pw_2->getB();
          assert(scl_o->getDims() == all1s);
          tensor_map[miopenTensorMhaScaleO] = scl_o;

          auto* o = pw_2->getY();
          assert(o->getDims() == all1s);
          tensor_map[miopenTensorMhaO] = o;
        }

      } else if (neigh->signName() == "OP_RNG") {
        auto* rng = dynamic_cast<gr::OperationRng*>(neigh);
        assert(rng);
        tensor_map[miopenTensorMhaDropoutSeed] = std::get<gr::Tensor*>(rng->getSeed());
        tensor_map[miopenTensorMhaDropoutOffset] = rng->getOffset();
        tensor_map[miopenTensorMhaDropoutProbability] = rng->getRng()->getBernoulliProb();

      }
    }

    { // discovering Z_INV and AMAX_S tensors
      auto* exp_node = findNodeByName("OP_POINTWISE:EXP");
      assert(exp_node);

      // get exp_node's neighbor that is a Pointwise mult
      auto* pw_mult = dynamic_cast<gr::OperationPointwise*>(findOutNeighByName(exp_node, "OP_POINTWISE:MUL"));
      assert(pw_mult);

      auto* red = dynamic_cast<gr::OperationReduction*>(findOutNeighByName(pw_mult, "OP_REDUCTION:MAX"));
      assert(red);

      tensor_map[miopenTensorMhaAmaxS] = red->getY();

      auto* inv_node = dynamic_cast<gr::OperationPointwise*>(findNodeByName("OP_POINTWISE:RECIPROCAL"));
      tensor_map[miopenTensorMhaZInv] = inv_node->getY();

    }

  }

public:
  void Run() {
    auto [n, h, s, d] = GetParam();
    createMhaGraph(n, h, s, d);
  }

};

} // end namespace mha_graph_test
 

using namespace mha_graph_test;

TEST_P(MhaFwdGraphTest, MhaFwdGraph) {
  Run();
}


INSTANTIATE_TEST_SUITE_P(MhaGraphFwdSuite, MhaFwdGraphTest, 
    testing::Combine(
      testing::ValuesIn({2}),  // n
      testing::ValuesIn({4}),  // s
      testing::ValuesIn({8}),  // h
      testing::ValuesIn({16}) // d
    ));
