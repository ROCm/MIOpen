#pragma once

#include <algorithm>
#include <vector>
#include <unordered_map>

namespace miopen {

namespace graphapi {

class TensorDescriptorEx; // TODO(Amber): may need to remove



namespace internal {
  template <typename C>
  bool contains(const C& container, const typename C::value_type& val) noexcept {
    return std::find(container.cbegin(), container.cend(), val) != container.cend();
  }
}

class OpNode {

protected:
  void addInTensor(const TensorDescriptorEx* tensor);

public:
  bool hasInTensor(const TensorDescriptorEx* tensor) {
    return internal::contains(mInputTensors, tensor);
  }

  bool hasOutTensor(const TensorDescriptorEx* tensor) {
    return internal::contains(mOutputTensors, tensor);
  }

  bool hasOutNeighbor(OpNode* dst) const { 
    return internal::contains(mOutNeighbors, dst);
  }

  bool hasInNeighbor(OpNode* src) const { 
    return internal::contains(mInNeighbors, src);
  }

  const auto& iterateInTensors() const { return mInputTensors; }

  const auto& iterateOutTensors() const { return mOutputTensors; }

  const auto& iterateInNeighbors() const { return mInNeighbors; }

  const auto& iterateOutNeighbors() const { return mOutNeighbors; }

  
  void addOutNeighbor(OpNode* dst) {
    assert(dst);
    if (!hasOutNeighbor(dst)) {
      mOutNeighbors.emplace_back(dst);
    }
  }

  void addInNeighbor(OpNode* src) {
    assert(src);
    if (!hasInNeighbor(src)) {
      mInNeighbors.emplace_back(src);
    }
  }

private:


  std::vector<TensorDescriptorEx*> mInputTensors;
  std::vector<TensorDescriptorEx*> mOutputTensors;

  std::vector<OpNode*> mInNeighbors;
  std::vector<OpNode*> mOutNeighbors;

};

class OpGraphBuilder;

class OpGraph {

public:

  bool hasNode(OpNode* n) const {
    return std::find(nodes.cbegin(), nodes.cend(), n) != nodes.cend();
  }

private:

  friend class OpGraphBuilder;

  void addNode(OpNode* n) {
    assert(!hasNode(n));
    nodes.emplace_back(n);
  }

  void addEdge(OpNode* src, OpNode* dst) {
    assert(src);
    assert(dst);
    src->addOutNeighbor(dst);
    dst->addInNeighbor(src);
  }


  std::vector<OpNode*> nodes;

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

      graph.addNode(n);

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


    for (const auto& [tens_ptr, edge_info]: e_map) {


      if (edge_info.mSrc != nullptr && !edge_info.mDests.empty()) {
        for (const auto& d : edge_info.mDests) {
          graph.addEdge(edge_info.mSrc, d);
        }
      } else if (edge_info.mSrc == nullptr) {

        // tens_ptr is a non-virtual input tensor
        assert(!tens_ptr->isVirtual()); // tensor pointer can't be virtual
        assert(!edge_info.mDests.empty());

        // NOTE(Amber): we may take out this step if we decide not to add a source
        // and sink in the graph
        for (const auto& d : edge_info.mDests) {
          graph.addEdgeFromSrc(d);
        }
      } else if (edge_info.mDests.empty()) {
        // tens_ptr is a non-virtual output tensor
        assert(!tens_ptr->isVirtual());
        assert(edge_info.mSrc != nullptr);
        graph.addEdgeToSink(edge_info.mSrc);
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




void ConstructGraph



} // end namespace graphapi


} // end namespace miopen
