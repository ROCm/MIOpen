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

#include <miopen/graphapi/opgraph.hpp>
#include <miopen/graphapi/pointwise.hpp>
#include <miopen/graphapi/reduction.hpp>

#include <numeric>
#include <string>

namespace gr = miopen::graphapi;


namespace mha_graph_test {


struct Deleter {
  using Fn = std::function<void()>;
  const Fn emptyFn = [](){};
  Fn mFn = emptyFn;

  template <typename T>
  explicit Deleter(T* ptr) {
    mFn = [ptr] () { delete ptr; };
  }

  Deleter(const Deleter&) = delete;
  Deleter& operator = (const Deleter&) = delete;

  Deleter(Deleter&& that) noexcept : mFn(std::move(that.mFn)) {
    that.mFn = emptyFn;
  }

  Deleter& operator = (Deleter&& that) noexcept { 
    this->mFn(); // destruct self.
    this->mFn = std::move(that.mFn);
    that.mFn = emptyFn;
    return *this;
  }

  ~Deleter() {
    mFn();
  }
};



class MhaGraphTest: public testing::TestWithParam<std::array<int64_t, 4>> {

  std::vector<Deleter> mPtrsToFree;
  std::unique_ptr<gr::OpGraphBuilder> mGraphBuilder;
  gr::OpGraph mGraph;

  template <typename T>
  T* alloc(T&& val) {
    T* ret = new T(std::forward<T>(val));
    mPtrsToFree.emplace_back(ret);
    return ret;
  }

  template <typename Vec>
  gr::Tensor* makeTensor(std::string_view name, const Vec& dims, bool isVirtual=true) {
    int64_t id = 0;
    MIOPEN_THROW_IF(name.size() > sizeof(id), "tensor name exceeds 8 chars");
    std::copy_n(name.begin(), std::min(sizeof(id), name.size()), reinterpret_cast<char*>(id));


    Vec strides(dims);
    using T = typename Vec::value_type;
    std::exclusive_scan(dims.begin(), dims.end(), strides.begin(), T{1ll}, std::multiplies<T>{});


    return alloc(gr::TensorBuilder{}
    .setDataType(miopenFloat)
      .setDim(dims)
      .setStride(strides)
      .setId(id)
      .setVirtual(isVirtual)
      .build());
  }

  auto* makePointWiseDesc(miopenPointwiseMode_t mode) {
      return alloc(gr::PointwiseBuilder{}
          .setMode(MIOPEN_POINTWISE_MUL)
          .setMathPrecision(miopenFloat)
          .build());
  }

  void addBinaryPointwiseNode( gr::Pointwise* pw,
      std::initializer_list<gr::Tensor*> in_tensors, 
      std::initializer_list<gr::Tensor*> out_tensors) {

      assert(in_tensors.size() == 2);
      assert(out_tensors.size() == 1);

      mGraphBuilder->addNode(alloc(gr::OperationPointwiseBuilder{}
            .setPointwise(pw)
            .setX(in_tensors[0])
            .setB(in_tensors[1])
            .setY(out_tensors[0])
            .build()));

  }

  void addUnaryPointwiseNode( gr::Pointwise* pw,
      std::initializer_list<gr::Tensor*> in_tensors, 
      std::initializer_list<gr::Tensor*> out_tensors) {

      assert(in_tensors.size() == 1);
      assert(out_tensors.size() == 1);

      mGraphBuilder->addNode(alloc(gr::OperationPointwiseBuilder{}
            .setPointwise(pw)
            .setX(in_tensors[0])
            .setY(out_tensors[0])
            .build()));

  }

  void addReductionNode(miopenReduceTensorOp_t red_op,
      std::initializer_list<gr::Tensor*> in_tensors, 
      std::initializer_list<gr::Tensor*> out_tensors) {

      auto* red_desc = alloc(gr::ReductionBuilder{}
        .setCompType(miopenFloat)
        .setReductionOperator(red_op)
        .build());
      
      assert(in_tensors.size() == 1);
      assert(out_tensors.size() == 1);

      mGraphBuilder->addNode(alloc(gr::OperationReductionBuilder{}
            .setReduction(red_desc)
            .setX(in_tensors[0])
            .setY(out_tensors[0])
            .build()));
  }

  void addNode(std::string_view name, 
      std::initializer_list<gr::Tensor*> in_tensors, 
      std::initializer_list<gr::Tensor*> out_tensors) {
    
    if (name == "OP_MATMUL") {
      assert(in_tensors.size() == 2);
      assert(out_tensors.size() == 1);

      auto* mm_desc = alloc(gr::MatmulBuilder().setComputeType(miopenFloat8).build());
      mGraphBuilder->addNode(alloc(gr::OperationMatmulBuilder{}
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

    } else if (name == "OP_POINTWISE:RECIPROCAL) {
      auto* pw = makePointWiseDesc(MIOPEN_POINTWISE_RECIPROCAL);
      addUnaryPointwiseNode(pw, in_tensors, out_tensors);

    } else if (name == "OP_REDUCTION:MAX") {
      addReductionNode(MIOPEN_REDUCE_TENSOR_MAX, in_tensors, out_tensors);

    } else if (name == "OP_REDUCTION:SUM") {
      addReductionNode(MIOPEN_REDUCE_TENSOR_MAX, in_tensors, out_tensors);

    } else if (name == "OP_RNG") {
      constexpr double BERNOULLI = 0.5; 
      auto* rng_desc = alloc(gr::RngBuilder{}
          .setDistribution(MIOPEN_RNG_DISTRIBUTION_BERNOULLI)
          .setBernoulliProb(BERNOULLI)
          .build());

      assert(in_tensors.size() == 2);// first is seed tensor, second is offset
      assert(out_tensors.size() == 1);

      mGraphBuilder->addNode(alloc(gr::OperationRngBuilder{}
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

    auto Q = makeTensor("Q", nhsd, false);
    auto K = makeTensor("K", nhsd, false);
    auto V = makeTensor("V", nhsd, false);

#define MAKE_TENSOR(name, dims) auto* name = makeTensor(#name, dims)

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
};

} // end namespace mha_graph_test
