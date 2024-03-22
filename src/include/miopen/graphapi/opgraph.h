#pragma once

#include <algorithm>
#include <vector>
#include <unordered_map>

#include <cassert>

namespace miopen {

namespace graphapi {

class TensorDescriptorEx; // TODO(Amber): may need to remove



namespace internal {
  template <typename C>
  bool contains(const C& container, const typename C::value_type& val) noexcept {
    return std::find(container.cbegin(), container.cend(), val) != container.cend();
  }
} // end namespace internal


class OpGraphBuilder;
class OpGraph;

struct OpNode {

  using Edge = std::pair<OpNode*, TensorDescriptorEx*>;

  friend class OpGraphBuilder;
  friend class OpGraph;

protected:

  virtual const std::string& SignName() const = 0;

  const auto& iterateInTensors() const { return mInTensors; }

  const auto& iterateOutTensors() const { return mOutTensors; }

  const auto& iterateInEdges() const { return mInEdges; }

  const auto& iterateOutEdges() const { return mOutEdges; }
  
  void addInTensor(TensorDescriptorEx* tens_ptr) {
    assert(!hasInTensor(tens_ptr));
    mInTensors.emplace_back(tens_ptr);
  }
  
  void addOutTensor(TensorDescriptorEx* tens_ptr) {
    assert(!hasOutTensor(tens_ptr));
    mOutTensors.emplace_back(tens_ptr);
  }

  bool hasInTensor(TensorDescriptorEx* tensor) const {
    return internal::contains(mInTensors, tensor);
  }

  bool hasOutTensor(TensorDescriptorEx* tensor) const {
    return internal::contains(mOutTensors, tensor);
  }

  bool hasInEdge(OpNode* src, TensorDescriptorEx* tens_ptr) const { 
    Edge e{src, tens_ptr};
    return internal::contains(mInEdges, e);
  }

  bool hasOutEdge(OpNode* dst, TensorDescriptorEx* tens_ptr) const { 
    Edge e{dst, tens_ptr};
    return internal::contains(mOutEdges, e);
  }

  void addOutEdge(OpNode* dst, TensorDescriptorEx* tens_ptr) {
    assert(dst);
    Edge e{dst, tens_ptr};
    if (!hasOutEdge(e)) {
      mOutEdges.emplace_back(e);
    }
  }

  void addInEdge(OpNode* src, TensorDescriptorEx* tens_ptr) {
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

  std::vector<TensorDescriptorEx*> mInTensors;
  std::vector<TensorDescriptorEx*> mOutTensors;

  std::vector<Edge> mInEdges;
  std::vector<Edge> mOutEdges;

};

using OpEdge = OpNode::Edge;

class SourceOpNode: public OpNode {
};

class SinkOpNode: public OpNode {
};


class OpGraph {

public:

  bool hasNode(OpNode* n) const {
    return std::find(nodes.cbegin(), nodes.cend(), n) != nodes.cend();
  }

  bool hasEdge(OpNode* src, TensorDescriptorEx* tens_ptr, OpNode* dst) const {
    assert(src);
    assert(dst);
    return src->hasOutEdge(dst, tens_ptr) && dst->hasInEdge(src, tens_ptr);
  }

private:

  friend class OpGraphBuilder;

  void addNodes(std::vector<OpNode*>&& nodes) {
    mNodes = std::move(nodes);
  }

  void addEdge(OpNode* src, TensorDescriptorEx* tens_ptr, OpNode* dst) {
    assert(src);
    assert(dst);
    src->addOutEdge(dst, tens_ptr);
    dst->addInEdge(src, tens_ptr);
  }

  void addEdgeFromSrc(OpNode* dst, TensorDescriptorEx* tens_ptr) {
    addEdge(&mSrcNode, tens_ptr, dst);
  }

  void addEdgeToSink(OpNode* src, TensorDescriptorEx* tens_ptr) {
    addEdge(src, tens_ptr, &mSinkNode);
  }

  SourceOpNode mSrcNode{};
  SinkOpNode mSinkNode{};
  std::vector<OpNode*> mNodes{};

};


class OpGraphBuilder {

public:

  bool hasNode(OpNode* node) const { 
    return std::find(mNodes.cbegin(), mNodes.cend(), node) != mNodes.cend();
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
    std::unordered_map<TensorDescriptorEx*, EdgeInfo> e_map;

    for (OpNode* n: mNodes) {

      for (TensorDescriptorEx* i: mNodes->iterateInTensors()) {
        auto [iter, _ignore] = e_map.emplace(i); // add empty EdgeInfo if not present

        iter->second.mDests.emplace_back(n);

      }

      for (TensorDescriptorEx* o: mNodes->iterateOutTensors()) {
        auto [iter, _ignore] = e_map.emplace(i); // add empty EdgeInfo if not present
                                               
        assert(iter->second.mSrc == nullptr);
        iter->second.mSrc = n;
      }
    }

    graph.addNodes(mNodes);


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
