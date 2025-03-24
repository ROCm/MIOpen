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
#pragma once

#include <miopen/graphapi/tensor.hpp>
#include <miopen/graphapi/engine.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <cassert>

namespace miopen {
namespace graphapi {

namespace internal {
template <typename C, typename T>
bool contains(const C& container, const T& val) noexcept
{
    return std::find(container.cbegin(), container.cend(), val) != container.cend();
}

template <typename R, int threshold = 500>
bool noRepetitions(const R& r)
{
    auto begin = r.cbegin();
    auto end   = r.cend();

    if(std::distance(begin, end) < threshold)
    {
        // time = O(n^2) mem = O(1)
        bool result = true;
        while(result && begin != end)
        {
            const auto& val = *begin++;
            result          = std::find(begin, end, val) == end;
        }
        return result;
    }
    else
    {
        // time = O(n) mem = O(n)
        std::unordered_set<std::remove_cv_t<std::remove_reference_t<decltype(*begin)>>> seen;
        for(; begin != end; ++begin)
        {
            const auto& val = *begin;
            if(seen.find(val) != seen.end())
            {
                return false;
            }
            seen.insert(val);
        }
        return true;
    }
}
} // end namespace internal

class OpGraphBuilder;
class OpGraph;

class MIOPEN_INTERNALS_EXPORT OpNode
{
public:
    using Edge = std::pair<OpNode*, Tensor*>;
    virtual ~OpNode();

    virtual const std::string& signName() const = 0;

private:
    std::vector<Edge> mInEdges;
    std::vector<Edge> mOutEdges;

    friend class OpGraphBuilder;
    friend class OpGraph;

protected:
    static Edge makeEdge(OpNode* n, Tensor* t) { return Edge{n, t}; }
    static Edge makeEdge(const OpNode* n, const Tensor* t)
    {
        return Edge{
            const_cast<OpNode*>(n), // NOLINT (cppcoreguidelines-pro-type-const-cast)
            const_cast<Tensor*>(t)  // NOLINT (cppcoreguidelines-pro-type-const-cast)
        };
    }
    virtual std::vector<Tensor*> getInTensors() const = 0;

    virtual std::vector<Tensor*> getOutTensors() const = 0;

    const auto& getInEdges() const { return mInEdges; }

    const auto& getOutEdges() const { return mOutEdges; }

    bool hasInEdge(const OpNode* src, const Tensor* tens_ptr) const
    {
        assert(src);
        assert(tens_ptr);
        auto e = makeEdge(src, tens_ptr);
        return internal::contains(mInEdges, e);
    }

    bool hasOutEdge(const OpNode* dst, const Tensor* tens_ptr) const
    {
        assert(dst);
        assert(tens_ptr);
        auto e = makeEdge(dst, tens_ptr);
        return internal::contains(mOutEdges, e);
    }

    void addOutEdge(OpNode* dst, Tensor* tens_ptr)
    {
        assert(dst);
        assert(tens_ptr);
        if(!hasOutEdge(dst, tens_ptr))
        {
            mOutEdges.emplace_back(dst, tens_ptr);
        }
    }

    void addInEdge(OpNode* src, Tensor* tens_ptr)
    {
        assert(src);
        assert(tens_ptr);
        if(!hasInEdge(src, tens_ptr))
        {
            mInEdges.emplace_back(src, tens_ptr);
        }
    }

    size_t getInDegree() const { return mInEdges.size(); }
    size_t getOutDegree() const { return mOutEdges.size(); }
};

using Edge = OpNode::Edge;

class MIOPEN_INTERNALS_EXPORT SourceOpNode : public OpNode
{
protected:
    std::vector<Tensor*> mOutTensors;
    friend class OpGraph;

    const std::string& signName() const final
    {
        static const std::string s = "INTERNAL::SRC";
        return s;
    }

    std::vector<Tensor*> getInTensors() const final { return {}; }

    std::vector<Tensor*> getOutTensors() const final { return mOutTensors; }

    bool hasOutTensor(const Tensor* tensor) const
    {
        return internal::contains(mOutTensors, tensor);
    }

    void addOutTensor(Tensor* tens_ptr)
    {
        if(!hasOutTensor(tens_ptr))
        {
            mOutTensors.emplace_back(tens_ptr);
        }
    }
};

class MIOPEN_INTERNALS_EXPORT SinkOpNode : public OpNode
{
protected:
    std::vector<Tensor*> mInTensors;
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
        if(!hasInTensor(tens_ptr))
        {
            mInTensors.emplace_back(tens_ptr);
        }
    }
};

using Path       = std::vector<const OpNode*>;
using VecOfPaths = std::vector<Path>;

class Engine;

class MIOPEN_INTERNALS_EXPORT OpGraph
{
    // NOTE: mSrcNode and mSinkNode need to reside on the heap because the graph may move
    // to a new memory location after building, while the nodes maintain address
    // of SourceOpNode and SinkOpNode in their in and out edge lists
    std::unique_ptr<SourceOpNode> mSrcNode = std::make_unique<SourceOpNode>();
    std::unique_ptr<SinkOpNode> mSinkNode  = std::make_unique<SinkOpNode>();
    std::vector<OpNode*> mNodes{};

    // Descriptor related members
    miopenHandle_t mHandle = nullptr;
    std::vector<Engine> mEngines{};

public:
    OpGraph(const OpGraph&) = delete;
    OpGraph& operator=(const OpGraph&) = delete;

    OpGraph()          = default;
    OpGraph(OpGraph&&) = default;
    OpGraph& operator=(OpGraph&&) = default;
    ~OpGraph()                    = default;

    SourceOpNode* getSourceNode() const noexcept { return mSrcNode.get(); }

    SinkOpNode* getSinkNode() const noexcept { return mSinkNode.get(); }

    bool hasNode(const OpNode* n) const { return internal::contains(mNodes, n); }

    bool hasEdge(const OpNode* src, const Tensor* tens_ptr, const OpNode* dst) const
    {
        assert(src);
        assert(dst);
        return src->hasOutEdge(dst, tens_ptr) && dst->hasInEdge(src, tens_ptr);
    }

    size_t numNodes() const { return mNodes.size(); }

    size_t numEdges() const
    {
        size_t ret = 0;
        for(OpNode* n : mNodes)
        {
            ret += n->getOutDegree();
        }
        // ignore the edges that lead to mSinkNode
        assert(ret >= mSinkNode->getInDegree());
        ret -= mSinkNode->getInDegree();

        return ret;
    }

    const std::vector<OpNode*>& getNodes() const noexcept { return mNodes; }

    const std::vector<Edge>& getOutEdges(const OpNode* n) const noexcept
    {
        assert(n);
        return n->getOutEdges();
    }

    const std::vector<Edge>& getInEdges(const OpNode* n) const noexcept
    {
        assert(n);
        return n->getInEdges();
    }

    OpNode* findNodeByName(const std::string& name) const noexcept
    {
        for(auto* n : getNodes())
        {
            if(n->signName() == name)
            {
                return n;
            }
        }
        return nullptr;
    }

    OpNode* findOutNeighByName(const OpNode* node, const std::string& neigh_name) const noexcept
    {
        assert(node);
        for(auto [m, t] : getOutEdges(node))
        {
            std::ignore = t;
            if(m->signName() == neigh_name)
            {
                return m;
            }
        }
        return nullptr;
    }

    OpNode* findInNeighByName(const OpNode* node, const std::string& neigh_name) const
    {
        assert(node);
        for(auto [m, t] : getInEdges(node))
        {
            std::ignore = t;
            if(m->signName() == neigh_name)
            {
                return m;
            }
        }
        return nullptr;
    }

    std::vector<std::string> getNodeNames() const
    {
        std::vector<std::string> names(mNodes.size());
        for(size_t i = 0; i < mNodes.size(); ++i)
        {
            names[i] = mNodes[i]->signName();
        }
        return names;
    }

    std::vector<std::pair<size_t, size_t>> getInOutDegrees() const
    {
        std::vector<std::pair<size_t, size_t>> ret(mNodes.size());
        for(size_t i = 0; i < mNodes.size(); ++i)
        {
            ret[i] = {mNodes[i]->getInDegree(), mNodes[i]->getOutDegree()};
        }
        return ret;
    }

    VecOfPaths getAllPaths() const;

    // NOTE: for testing only. May remove in the future
    bool hasEdgeFromSource(OpNode* dst, Tensor* tens_ptr) const
    {
        return hasEdge(mSrcNode.get(), tens_ptr, dst);
    }

    // NOTE: for testing only. May remove in the future
    bool hasEdgeToSink(OpNode* src, Tensor* tens_ptr) const
    {
        return hasEdge(src, tens_ptr, mSinkNode.get());
    }

    miopenHandle_t getHandle() const noexcept { return mHandle; }
    const std::vector<Engine>& getEngines() const noexcept { return mEngines; }

    void initEngines(); /// \todo make private. Called in finalize, but also
                        /// from C++ tests --amberhassaan May, 2024

private:
    friend class OpGraphBuilder;

    void initNodes(std::vector<OpNode*>&& nodes) { mNodes = std::move(nodes); }

    void addEdge(OpNode* src, Tensor* tens_ptr, OpNode* dst)
    {
        assert(src);
        assert(dst);
        src->addOutEdge(dst, tens_ptr);
        dst->addInEdge(src, tens_ptr);
    }

    void addEdgeFromSrc(OpNode* dst, Tensor* tens_ptr)
    {
        mSrcNode->addOutTensor(tens_ptr);
        addEdge(mSrcNode.get(), tens_ptr, dst);
    }

    void addEdgeToSink(OpNode* src, Tensor* tens_ptr)
    {
        mSinkNode->addInTensor(tens_ptr);
        addEdge(src, tens_ptr, mSinkNode.get());
    }
};

class MIOPEN_INTERNALS_EXPORT OpGraphBuilder
{
private:
    std::vector<OpNode*> mNodes;
    miopenHandle_t mHandle = nullptr;

public:
    void setHandle(miopenHandle_t handle) { mHandle = checkPtr(handle); }
    miopenHandle_t getHandle() const noexcept { return mHandle; }

    bool hasNode(OpNode* node) const { return internal::contains(mNodes, node); }

    void addNode(OpNode* node)
    {
        assert(!hasNode(node));
        mNodes.emplace_back(node);
    }

    void setNodes(const std::vector<OpNode*>& nodes)
    {
        assert(internal::noRepetitions(nodes));
        mNodes = nodes;
    }

    void setNodes(std::vector<OpNode*>&& nodes)
    {
        assert(internal::noRepetitions(nodes));
        mNodes = std::move(nodes);
    }

    struct EdgeInfo
    {
        OpNode* mSrc = nullptr;
        std::vector<OpNode*> mDests{};
    };

    // r-value method that consumes *this
    OpGraph build() &&;
};

MIOPEN_INTERNALS_EXPORT bool isIsomorphic(const OpGraph& left, const OpGraph& right);

MIOPEN_INTERNALS_EXPORT std::string pathToStr(const Path& path);

class MIOPEN_INTERNALS_EXPORT BackendOperationGraphDescriptor : public BackendDescriptor
{
private:
    OpGraphBuilder mBuilder;
    OpGraph mOpGraph;
    std::vector<miopenBackendDescriptor_t> mOps; // to return them in getAttribute

public:
    void setAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t elementCount,
                      void* arrayOfElements) override;
    void finalize() override;
    void getAttribute(miopenBackendAttributeName_t attributeName,
                      miopenBackendAttributeType_t attributeType,
                      int64_t requestedElementCount,
                      int64_t* elementCount,
                      void* arrayOfElements) override;

    const OpGraph* getOperationGraph() const noexcept { return &mOpGraph; }
    OpGraph* getOperationGraph() noexcept { return &mOpGraph; }
};

} // end namespace graphapi
} // end namespace miopen
