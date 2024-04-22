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


    auto t = gr::TensorBuilder{}
    .setDataType(miopenFloat)
      .setDim(dims)
      .setStride(strides)
      .setId(id)
      .setVirtual(isVirtual)
      .build();

    return alloc(std::move(t));
  }


  gr::OpNode* makeNode(std::string_view name, 
      std::initializer_list<gr::Tensor*> in_tensors, 
      std::initializer_list<gr::Tensor*> out_tensors) {
    
    if (name == "OP_MATMUL") {
      assert(in_tensors.size() == 2);
      assert(out_tensors.size() == 1);

      auto mm_desc = 
      auto node = gr::OperationMatmulBuilder{}
        .setA(in_tensors[0])
        .setB(in_tensors[1])
        .setC(out_tensors[0])
        .setMatmulDescriptor(

    }

  }


  void createMhaGraph(int64_t n, int64_t h, int64_t s, int64_t d) {

    std::vector<int64_t> nhsd = {n, h, s, d};
    std::vector<int64_t> nhss = {n, h, s, s};
    std::vector<int64_t> nhs1 = {n, h, s, 1};
    std::vector<int64_t> all1s = {1, 1, 1, 1};

    OpGraphBuilder graph_builder;

    auto Q = makeTensor("Q", nhsd, false);
    auto K = makeTensor("K", nhsd, false);
    auto V = makeTensor("V", nhsd, false);

#define MAKE_TENSOR(name, dims) auto* name = makeTensor(#name, dims)

    MAKE_TENSOR(T_MM_0, nhss);
    auto mm_node_0 = makeNode("OP_MATMUL", {Q, K}, {T_MM_0});
    graph_builder.addNode(mm_node_0);

    MAKE_TENSOR(T_SCL_0, nhss);
    MAKE_TENSOR(ATN_SCL, all1s);

    auto scale_node_0 = makeNode("OP_POINTWISE:MUL", {T_MM_0, ATN_SCL}, {T_SCL_0});
    graph_builder.addNode(scale_node_0);

    MAKE_TENSOR(T_SCL_1, nhss);
    MAKE_TENSOR(DSCL_Q, all1s);

    auto scale_node_1 = makeNode("OP_POINTWISE:MUL", {T_SCL_0, DSCL_Q}, {T_SCL_1});
    graph_builder.addNode(scale_node_1);

    MAKE_TENSOR(T_SCL_2, nhss);
    MAKE_TENSOR(DSCL_K, all1s);

    auto scale_node_2 = makeNode("OP_POINTWISE:MUL", {T_SCL_1, DSCL_K}, {T_SCL_2});
    graph_builder.addNode(scale_node_2);

    MAKE_TENSOR(M, nhs1);
    auto reduce_node_0 = makeNode("OP_REDUCTION:MAX", {T_SCL_2}, {M});
    graph_builder.addNode(reduce_node_0);

    MAKE_TENSOR(T_SUB, nhss);
    auto sub_node = makeNode("OP_POINTWISE:SUB", {T_SCL_2, M}, {T_SUB});
    graph_builder.addNode(sub_node);

    MAKE_TENSOR(T_EXP, nhss);
    auto exp_node = makeNode("OP_POINTWISE:EXP", {T_SUB}, {T_EXP});
    graph_builder.addNode(sub_node);

    MAKE_TENSOR(T_SUM, nhs1);
    auto reduce_node_1 = makeNode("OP_REDUCTION:SUM", {T_EXP}, {T_SUM});
    graph_builder.addNode(reduce_node_1);

    MAKE_TENSOR(Z_INV, nhs1);
    auto inv_node = makeNode("OP_POINTWISE:RECIPROCAL", {T_SUM}, {Z_INV});
    graph_builder.addNode(inv_node);

    MAKE_TENSOR(T_MUL_0, nhss);
    auto mul_node_0 = makeNode("OP_POINTWISE:MUL", {T_EXP, Z_INV}, {T_MUL_0});
    graph_builder.addNode(mul_node_0);

    MAKE_TENSOR(AMAX_S, all1s);
    auto reduce_node_2 = makeNode("OP_REDUCTION:MAX", {T_MUL_0}, {AMAX_S});
    graph_builder.addNode(reduce_node_2);

    MAKE_TENSOR(RND_SD, all1s);
    MAKE_TENSOR(RND_OFF, all1s);
    MAKE_TENSOR(RND_PRB, all1s);

    MAKE_TENSOR(T_RND, nhss);
    auto mul_node_1 = makeNode("OP_POINTWISE:MUL", {T_MUL_0, T_RND}, {T_MUL_1});
    graph_builder.addNode(mul_node_1);


    MAKE_TENSOR(T_SCL_3, nhss);
    auto scale_node_3 = makeNode("OP_POINTWISE:MUL", {T_MUL_1, RND_PRB}, {T_SCL_3});
    graph_builder.addNode(scale_node_3);

    MAKE_TENSOR(T_SCL_4, nhss);
    MAKE_TENSOR(SCL_S, all1s);
    auto scale_node_4 = makeNode("OP_POINTWISE:MUL", {T_SCL_3, SCL_S}, {T_SCL_4});
    graph_builder.addNode(scale_node_4);

    MAKE_TENSOR(T_MM_1, nhsd);
    auto mm_node_1 = makeNode("OP_MATMUL", {T_SCL_4, V}, {T_MM_1});
    graph_builder.addNode(mm_node_1);

    MAKE_TENSOR(T_SCL_5, nhsd);
    MAKE_TENSOR(DSCL_S, all1s);
    auto scale_node_5 = makeNode("OP_POINTWISE:MUL", {T_MM_1, DSCL_S}, {T_SCL_5});
    graph_builder.addNode(scale_node_5);

    MAKE_TENSOR(T_SCL_6, nhsd);
    MAKE_TENSOR(DSCL_V, all1s);
    auto scale_node_6 = makeNode("OP_POINTWISE:MUL", {T_SCL_5, DSCL_V}, {T_SCL_6});
    graph_builder.addNode(scale_node_6);

    MAKE_TENSOR(T_SCL_7, nhsd);
    MAKE_TENSOR(SCL_O, all1s);
    auto scale_node_7 = makeNode("OP_POINTWISE:MUL", {T_SCL_6, SCL_O}, {T_SCL_7});
    graph_builder.addNode(scale_node_7);

    MAKE_TENSOR(AMAX_O, all1s);
    auto reduce_node_3 = makeNode("OP_REDUCTION:MAX", {T_SCL_6}, {AMAX_O});
    graph_builder.addNode(reduce_node_2);


    auto graph = std::move(graph_builder).build(); 

#undef MAKE_TENSOR

    return graph; // FIXME(amber): all local pointers will be invalidated upon return

  }
};

} // end namespace mha_graph_test
