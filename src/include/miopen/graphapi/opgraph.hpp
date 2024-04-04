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
#pragma once

#include <miopen/graphapi/tensor.hpp>

#include <algorithm>
#include <vector>

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

class OpNode
{
public:
    using Edge = std::pair<OpNode*, Tensor*>;
    virtual ~OpNode();

private:
    std::vector<Edge> mInEdges;
    std::vector<Edge> mOutEdges;

    friend class OpGraphBuilder;
    friend class OpGraph;

protected:
    virtual const std::string& signName() const = 0;

    virtual std::vector<Tensor*> getInTensors() const = 0;

    virtual std::vector<Tensor*> getOutTensors() const = 0;

    const auto& getInEdges() const { return mInEdges; }

    const auto& getOutEdges() const { return mOutEdges; }

    bool hasInEdge(OpNode* src, Tensor* tens_ptr) const
    {
        assert(src);
        assert(tens_ptr);
        Edge e{src, tens_ptr};
        return internal::contains(mInEdges, e);
    }

    bool hasOutEdge(OpNode* dst, Tensor* tens_ptr) const
    {
        assert(dst);
        assert(tens_ptr);
        Edge e{dst, tens_ptr};
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
};

using OpEdge = OpNode::Edge;

class SourceOpNode : public OpNode
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

    bool hasOutTensor(Tensor* tensor) const { return internal::contains(mOutTensors, tensor); }

    void addOutTensor(Tensor* tens_ptr)
    {
        assert(!hasOutTensor(tens_ptr));
        mOutTensors.emplace_back(tens_ptr);
    }
};

class SinkOpNode : public OpNode
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
        assert(!hasInTensor(tens_ptr));
        mInTensors.emplace_back(tens_ptr);
    }
};

class OpGraph
{
    SourceOpNode mSrcNode{};
    SinkOpNode mSinkNode{};
    std::vector<OpNode*> mNodes{};

public:
    bool hasNode(OpNode* n) const { return internal::contains(mNodes, n); }

    bool hasEdge(OpNode* src, Tensor* tens_ptr, OpNode* dst) const
    {
        assert(src);
        assert(dst);
        return src->hasOutEdge(dst, tens_ptr) && dst->hasInEdge(src, tens_ptr);
    }

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
        mSrcNode.addOutTensor(tens_ptr);
        addEdge(&mSrcNode, tens_ptr, dst);
    }

    void addEdgeToSink(OpNode* src, Tensor* tens_ptr)
    {
        mSinkNode.addInTensor(tens_ptr);
        addEdge(src, tens_ptr, &mSinkNode);
    }
};

class OpGraphBuilder
{
private:
    std::vector<OpNode*> mNodes;

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

    // r-value method that consumes *this
    OpGraph build() &&;
};

} // end namespace graphapi
} // end namespace miopen
