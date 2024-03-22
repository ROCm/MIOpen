#pragma once

#include <miopen/graphapi/graphapi_tensor.hpp>

#include <algorithm>
#include <vector>
#include <unordered_map>

#include <cassert>

namespace miopen {

namespace graphapi {

namespace internal {
  template <typename C>
  bool contains(const C& container, const typename C::value_type& val) noexcept {
    return std::find(container.cbegin(), container.cend(), val) != container.cend();
  }
} // end namespace internal


class OpGraphBuilder;
class OpGraph;

struct OpNode {

  using Edge = std::pair<OpNode*, Tensor*>;

  friend class OpGraphBuilder;
  friend class OpGraph;

protected:

  virtual const std::string& SignName() const = 0;

  virtual std::vector<Tensor*> getInTensors() const = 0;

  virtual std::vector<Tensor*> getOutTensors() const = 0;

  /*

  */

  const auto& iterateInEdges() const { return mInEdges; }

  const auto& iterateOutEdges() const { return mOutEdges; }

  bool hasInEdge(OpNode* src, Tensor* tens_ptr) const { 
    Edge e{src, tens_ptr};
    return internal::contains(mInEdges, e);
  }

  bool hasOutEdge(OpNode* dst, Tensor* tens_ptr) const { 
    Edge e{dst, tens_ptr};
    return internal::contains(mOutEdges, e);
  }

  void addOutEdge(OpNode* dst, Tensor* tens_ptr) {
    assert(dst);
    Edge e{dst, tens_ptr};
    if (!hasOutEdge(e)) {
      mOutEdges.emplace_back(e);
    }
  }

  void addInEdge(OpNode* src, Tensor* tens_ptr) {
    assert(src);
    Edge e{src, tens_ptr};
    if (!hasInEdge(e)) {
      mInEdges.emplace_back(e);
    }
  }

private:

  // NOTE(Amber): We could implement the iterate{In,Out}Tensors, 
  // has{In,Out}Tensor has pure virtual functions that derived classes must
  // override and thus avoid duplicating tensor pointers in this class by
  // eliminating the vectors below.

  /*
  std::vector<Tensor*> mInTensors;
  std::vector<Tensor*> mOutTensors;
  */

  std::vector<Edge> mInEdges;
  std::vector<Edge> mOutEdges;

};

using OpEdge = OpNode::Edge;

class SourceOpNode: public OpNode {
protected:
    
  friend class OpGraph;

  const std::string& SignName() const final { 
    static const std::string s =  "INTERNAL::SRC"; 
    return s;
  }

  std::vector<Tensor*> getInTensors() const final {
    return {};
  }

  std::vector<Tensor*> getOutTensors() const final {
    return mOutTensors;
  }

  bool hasOutTensor(Tensor* tensor) const {
    return internal::contains(mOutTensors, tensor);
  }

  void addOutTensor(Tensor* tens_ptr) {
    assert(!hasOutTensor(tens_ptr));
    mOutTensors.emplace_back(tens_ptr);
  }

private:

  std::vector<Tensor*> mOutTensors;
};

class SinkOpNode: public OpNode {
protected:

  friend class OpGraph;

  const std::string& SignName() const final { 
    static const std::string s =  "INTERNAL::SINK"; 
    return s;
  }

  std::vector<Tensor*> getInTensors() const final {
    return mInTensors;
  }

  std::vector<Tensor*> getOutTensors() const final {
    return {};
  }

  bool hasInTensor(Tensor* tensor) const {
    return internal::contains(mInTensors, tensor);
  }

  void addInTensor(Tensor* tens_ptr) {
    assert(!hasInTensor(tens_ptr));
    mInTensors.emplace_back(tens_ptr);
  }
  
  std::vector<Tensor*> mInTensors;

};


class OpGraph {

public:

  bool hasNode(OpNode* n) const {
    return internal::contains(mNodes, n);
  }

  bool hasEdge(OpNode* src, Tensor* tens_ptr, OpNode* dst) const {
    assert(src);
    assert(dst);
    return src->hasOutEdge(dst, tens_ptr) && dst->hasInEdge(src, tens_ptr);
  }

private:

  friend class OpGraphBuilder;

  void addNodes(std::vector<OpNode*>&& nodes) {
    mNodes = std::move(nodes);
  }

  void addEdge(OpNode* src, Tensor* tens_ptr, OpNode* dst) {
    assert(src);
    assert(dst);
    src->addOutEdge(dst, tens_ptr);
    dst->addInEdge(src, tens_ptr);
  }

  void addEdgeFromSrc(OpNode* dst, Tensor* tens_ptr) {
    mSrcNode.addOutTensor(tens_ptr);
    addEdge(&mSrcNode, tens_ptr, dst);
  }

  void addEdgeToSink(OpNode* src, Tensor* tens_ptr) {
    mSinkNode.addInTensor(tens_ptr);
    addEdge(src, tens_ptr, &mSinkNode);
  }

  SourceOpNode mSrcNode{};
  SinkOpNode mSinkNode{};
  std::vector<OpNode*> mNodes{};

};


class OpGraphBuilder {

public:

  bool hasNode(OpNode* node) const { 
    return internal::contains(mNodes, node);
  }

  void addNode(OpNode* node) {
    assert(!hasNode(node));
    mNodes.emplace_back(node);
  }

  struct EdgeInfo {
    OpNode* mSrc = nullptr;
    std::vector<OpNode*> mDests{};
  };

  OpGraph build() {

    OpGraph graph;

    // key = tensor ptr, value = vec. of dest nodes
    std::unordered_map<Tensor*, EdgeInfo> e_map;

    for (OpNode* n: mNodes) {

      for (Tensor* i: n->getInTensors()) {
        auto [iter, _ignore] = e_map.emplace(i); // add empty EdgeInfo if not present

        iter->second.mDests.emplace_back(n);

      }

      for (Tensor* o: n->getOutTensors()) {
        auto [iter, _ignore] = e_map.emplace(o); // add empty EdgeInfo if not present
                                               
        assert(iter->second.mSrc == nullptr);
        iter->second.mSrc = n;
      }
    }

    graph.addNodes(std::move(mNodes));


    for (const auto& [tens_ptr, edge_info]: e_map) {

      if (edge_info.mSrc != nullptr && !edge_info.mDests.empty()) {
        for (const auto& d : edge_info.mDests) {
          graph.addEdge(edge_info.mSrc, tens_ptr, d);
        }
      } else if (edge_info.mSrc == nullptr) {

        // tens_ptr is a non-virtual input tensor
        assert(!tens_ptr->isVirtual()); // tensor pointer can't be virtual
        assert(!edge_info.mDests.empty());

        // NOTE(Amber): we may take out this step if we decide not to add a source
        // and sink in the graph
        for (const auto& d : edge_info.mDests) {
          graph.addEdgeFromSrc(d, tens_ptr);
        }
      } else if (edge_info.mDests.empty()) {
        // tens_ptr is a non-virtual output tensor
        assert(!tens_ptr->isVirtual());
        assert(edge_info.mSrc != nullptr);
        graph.addEdgeToSink(edge_info.mSrc, tens_ptr);
      } else {
        // both can't be true at the same time
        assert(!(edge_info.mSrc == nullptr && edge_info.mDests.empty())); 
      }
    }

    return graph;

  }

private:
  std::vector<OpNode*> mNodes;

};


} // end namespace graphapi

} // end namespace miopen
