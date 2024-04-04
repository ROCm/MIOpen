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

#include <miopen/graphapi/opgraph.hpp>

#include <unordered_map>

namespace miopen {
namespace graphapi {

OpNode::~OpNode() = default;

OpGraph OpGraphBuilder::build() &&
{

    OpGraph graph;

    // key = tensor ptr, value = vec. of dest nodes
    /// \todo might eventually move this state to the graph class during build()
    std::unordered_map<Tensor*, EdgeInfo> e_map;

    for(OpNode* n : mNodes)
    {

        for(Tensor* i : n->getInTensors())
        {
            auto [iter, _ignore] =
                e_map.try_emplace(i, EdgeInfo{}); // add empty EdgeInfo if not present

            iter->second.mDests.emplace_back(n);
        }

        for(Tensor* o : n->getOutTensors())
        {
            auto [iter, _ignore] =
                e_map.try_emplace(o, EdgeInfo{}); // add empty EdgeInfo if not present

            assert(iter->second.mSrc == nullptr);
            iter->second.mSrc = n;
        }
    }

    graph.initNodes(std::move(mNodes));

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
            // tens_ptr is an output tensor (virtual or non-virtual)
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

} // end namespace graphapi
} // end namespace miopen
