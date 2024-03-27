#pragma once

#include <miopen/graphapi/graphapi_tensor.hpp>

#include <algorithm>
#include <deque>
#include <vector>
#include <unordered_map>

#include <cassert>

namespace miopen {

namespace graphapi {

namespace internal {
template <typename C>
bool contains(const C& container, const typename C::value_type& val) noexcept
{
    return std::find(container.cbegin(), container.cend(), val) != container.cend();
}
} // end namespace internal

class OpGraphBuilder;
class OpGraph;

struct OpNode
{

    using Edge = std::pair<OpNode*, Tensor*>;

    friend class OpGraphBuilder;
    friend class OpGraph;

protected:
    virtual const std::string& signName() const = 0;

    virtual std::vector<Tensor*> getInTensors() const = 0;

    virtual std::vector<Tensor*> getOutTensors() const = 0;

    const auto& iterateInEdges() const { return mInEdges; }

    const auto& iterateOutEdges() const { return mOutEdges; }

    bool hasInEdge(OpNode* src, Tensor* tens_ptr) const
    {
        Edge e{src, tens_ptr};
        return internal::contains(mInEdges, e);
    }

    bool hasOutEdge(OpNode* dst, Tensor* tens_ptr) const
    {
        Edge e{dst, tens_ptr};
        return internal::contains(mOutEdges, e);
    }

    void addOutEdge(OpNode* dst, Tensor* tens_ptr)
    {
        assert(dst);
        if(!hasOutEdge(dst, tens_ptr))
        {
            mOutEdges.emplace_back(dst, tens_ptr);
        }
    }

    void addInEdge(OpNode* src, Tensor* tens_ptr)
    {
        assert(src);
        if(!hasInEdge(src, tens_ptr))
        {
            mInEdges.emplace_back(src, tens_ptr);
        }
    }

    size_t getInDegree() const { return mInEdges.size(); }
    size_t getOutDegree() const { return mOutEdges.size(); }

private:
    std::vector<Edge> mInEdges;
    std::vector<Edge> mOutEdges;
};

using OpEdge = OpNode::Edge;

class SourceOpNode : public OpNode
{
protected:
    friend class OpGraph;

    const std::string& signName() const final
    {
        static const std::string s = "INTERNAL::SRC";
        return s;
    }

    std::vector<Tensor*> getInTensors() const final { return {}; }

    std::vector<Tensor*> getOutTensors() const final { return mOutTensors; }

    bool hasOutTensor(Tensor* tensor) const { return internal::contains(mOutTensors, tensor); }

    void addOutTensor(Tensor* tens_ptr)
    {
        assert(!hasOutTensor(tens_ptr));
        mOutTensors.emplace_back(tens_ptr);
    }

private:
    std::vector<Tensor*> mOutTensors;
};

class SinkOpNode : public OpNode
{
protected:
    friend class OpGraph;

    const std::string& signName() const final
    {
        static const std::string s = "INTERNAL::SINK";
        return s;
    }

    std::vector<Tensor*> getInTensors() const final { return mInTensors; }

    std::vector<Tensor*> getOutTensors() const final { return {}; }

    bool hasInTensor(Tensor* tensor) const { return internal::contains(mInTensors, tensor); }

    void addInTensor(Tensor* tens_ptr)
    {
        assert(!hasInTensor(tens_ptr));
        mInTensors.emplace_back(tens_ptr);
    }

    std::vector<Tensor*> mInTensors;
};

using Path = std::vector<OpNode*>;
using VecOfPaths = std::vector<Path>;

class OpGraph
{

public:
    bool hasNode(OpNode* n) const { return internal::contains(mNodes, n); }

    bool hasEdge(OpNode* src, Tensor* tens_ptr, OpNode* dst) const
    {
        assert(src);
        assert(dst);
        return src->hasOutEdge(dst, tens_ptr) && dst->hasInEdge(src, tens_ptr);
    }

    size_t numNodes() const { 
      return mNodes.size();
    }

    size_t numEdges() const { 
      size_t ret = 0;
      for (OpNode* n: mNodes) {
        ret += n->getOutDegree();
      }
      // ignore the edges that lead to mSinkNode
      assert(ret >= mSinkNode.getInDegree());
      ret -= mSinkNode.getInDegree();

      return ret;
    }

    std::vector<std::pair<size_t, size_t>> getInOutDegrees() const { 
      std::vector<std::pair<size_t, size_t>> ret;
      for (OpNode* n: mNodes) {
        ret.emplace_back(n->getInDegree(), n->getOutDegree());
      }
      return ret;
    }

    VecOfPaths getAllPaths() const {
      // TODO(Amber): does not check for cycles. Use DFS to first check for cycles
      // at construction time perhaps. 
      VecOfPaths all_paths;

      std::deque<Path> paths_to_explore = {mSrcNode};

      while (!paths_to_explore.empty()) {
        Path path = paths_to_explore.front();
        paths_to_explore.pop_front();

        OpNode* last_node = path.back();
        if (last_node->iterateOutEdges().empty()) {
          // all paths should terminate at the sink
          assert(last_node == &mSinkNode);
          all_paths.emplace_back(std::move(path));
        } else {
          for (auto& [dst, tens_ptr]: last_node->iterateOutEdges()) {
            Path newPath{path};
            newPath.emplace_back(dst);
            paths_to_explore.emplace_back(std::move(newPath));
          }
        }
      } // end while

      return all_paths;
    }

private:
    friend class OpGraphBuilder;

    void addNodes(std::vector<OpNode*>&& nodes) { mNodes = std::move(nodes); }

    void addEdge(OpNode* src, Tensor* tens_ptr, OpNode* dst)
    {
        assert(src);
        assert(dst);
        src->addOutEdge(dst, tens_ptr);
        dst->addInEdge(src, tens_ptr);
    }

    void addEdgeFromSrc(OpNode* dst, Tensor* tens_ptr)
    {
        mSrcNode.addOutTensor(tens_ptr);
        addEdge(&mSrcNode, tens_ptr, dst);
    }

    void addEdgeToSink(OpNode* src, Tensor* tens_ptr)
    {
        mSinkNode.addInTensor(tens_ptr);
        addEdge(src, tens_ptr, &mSinkNode);
    }

    SourceOpNode mSrcNode{};
    SinkOpNode mSinkNode{};
    std::vector<OpNode*> mNodes{};
};

class OpGraphBuilder
{

public:
    bool hasNode(OpNode* node) const { return internal::contains(mNodes, node); }

    void addNode(OpNode* node)
    {
        assert(!hasNode(node));
        mNodes.emplace_back(node);
    }

    struct EdgeInfo
    {
        OpNode* mSrc = nullptr;
        std::vector<OpNode*> mDests{};
    };

    OpGraph build()
    {

        OpGraph graph;

        // key = tensor ptr, value = vec. of dest nodes
        std::unordered_map<Tensor*, EdgeInfo> e_map;

        for(OpNode* n : mNodes)
        {

            for(Tensor* i : n->getInTensors())
            {
                auto [iter, _ignore] =
                    e_map.emplace(i, EdgeInfo{}); // add empty EdgeInfo if not present

                iter->second.mDests.emplace_back(n);
            }

            for(Tensor* o : n->getOutTensors())
            {
                auto [iter, _ignore] =
                    e_map.emplace(o, EdgeInfo{}); // add empty EdgeInfo if not present

                assert(iter->second.mSrc == nullptr);
                iter->second.mSrc = n;
            }
        }

        graph.addNodes(std::move(mNodes));

        for(const auto& [tens_ptr, edge_info] : e_map)
        {

            if(edge_info.mSrc != nullptr && !edge_info.mDests.empty())
            {
                for(const auto& d : edge_info.mDests)
                {
                    graph.addEdge(edge_info.mSrc, tens_ptr, d);
                }
            }
            else if(edge_info.mSrc == nullptr)
            {

                // tens_ptr is a non-virtual input tensor
                assert(!tens_ptr->isVirtual()); // tensor pointer can't be virtual
                assert(!edge_info.mDests.empty());

                // NOTE(Amber): we may take out this step if we decide not to add a source
                // and sink in the graph
                for(const auto& d : edge_info.mDests)
                {
                    graph.addEdgeFromSrc(d, tens_ptr);
                }
            }
            else if(edge_info.mDests.empty())
            {
                // tens_ptr is a non-virtual output tensor
                assert(!tens_ptr->isVirtual());
                assert(edge_info.mSrc != nullptr);
                graph.addEdgeToSink(edge_info.mSrc, tens_ptr);
            }
            else
            {
                // both can't be true at the same time
                assert(!(edge_info.mSrc == nullptr && edge_info.mDests.empty()));
            }
        }

        return graph;
    }

private:
    std::vector<OpNode*> mNodes;
};


inline bool DegreeEqualityTest(const OpGraph& left, const OpGraph& right) {
  auto l_degs = left.getInOutDegrees();
  auto r_degs = right.getInOutDegrees();

  auto sort_deg_vec = [] (auto& deg_vec) {

      std::sort(deg_vec.begin(), deg_vec.end(), 
          [] (const auto& left, const auto& right) {
            if (left.first == right.first) {
              return left.second < right.second;
            }
            return left.first < right.first;
          });

  };
  sort_deg_vec(l_degs);
  sort_deg_vec(r_degs);
  return l_degs == r_degs;
}

inline bool PathEqualityTest(const OpGraph& left, const OpGraph& right) {

  using MapSizeToPathVec = std::unordered_map<size_t, VecOfPaths>;

  auto group_by_size = [] (VecOfPaths&& all_paths) {
    MapSizeToPathVec paths_by_size;

    for (auto& p: all_paths) {
      auto [it, _ignore]  = paths_by_size.emplace(p.size(), VecOfPaths{});
      it->second.emplace_back(std::move(p));
    }

    return paths_by_size;
  };

  MapSizeToPathVec l_paths_by_sz{};
  auto r_paths_by_sz = l_paths_by_sz;

  {
    auto l_paths = left.getAllPaths();
    auto r_paths = right.getAllPaths();

    if (l_paths.size() != r_paths.size()) {
      return false;
    }

    auto sum_paths = [] (const VecOfPaths all_paths) {
      size_t ret = 0;
      for (const auto& p: all_paths) {
        ret += p.size();
      }
      return ret;
    };

    if (sum_paths(l_paths) != sum_paths(r_paths)) {
      return false;
    }

    l_paths_by_sz = group_by_size(std::move(l_paths));
    r_paths_by_sz = group_by_size(std::move(r_paths));
  }

  auto get_keys = [] (const MapSizeToPathVec& paths_by_size) {
    std::vector<size_t> keys{};
    for (const auto& [k, v]: paths_by_size) {
      keys.emplace_back(k);
    }
    return keys;
  };

  auto l_keys = get_keys(l_paths_by_sz);
  auto r_keys = get_keys(r_paths_by_sz);

  if (l_keys != r_keys) {
    return false;
  }

  auto check_equal_path_vecs = [](VecOfPaths& left, const VecOfPaths& right) {
    if (left.size() != right.size()) {
      return false;
    }

    std::sort(left.begin(), left.end());
    std::sort(right.begin(), right.end());

    return left == right;
  }

  for (size_t k: l_keys) {
    if (!check_equal_path_vecs(l_paths_by_sz[k], r_paths_by_sz[k])) {
      return false;
    }
  }

  return true;

}


inline bool isIsomorphic(const OpGraph& left, const OpGraph& right) {
  if (left.numNodes() != right.numNodes()) {
    return false;
  }

  if (!DegreeEqualityTest(left, right)) {
    return false;
  }
}




} // end namespace graphapi

} // end namespace miopen
