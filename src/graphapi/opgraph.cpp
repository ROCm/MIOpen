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

#include <miopen/errors.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <miopen/graphapi/engine.hpp>

#include <deque>
#include <unordered_map>

namespace miopen {
namespace graphapi {

OpNode::~OpNode() = default;

OpGraph OpGraphBuilder::build() &&
{
    if(mNodes.empty())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    OpGraph graph;

    graph.mHandle = mHandle;

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

            MIOPEN_THROW_IF(iter->second.mSrc != nullptr, "Output tensor with two source op nodes");
            iter->second.mSrc = n;
        }
    }

    graph.initNodes(std::move(mNodes));

    for(const auto& [tens_ptr, edge_info] : e_map)
    {
        MIOPEN_THROW_IF(edge_info.mSrc == nullptr && edge_info.mDests.empty(),
                        "Invalid state with null src node and empty dest nodes");

        if(edge_info.mSrc != nullptr && !edge_info.mDests.empty())
        {
            for(const auto& d : edge_info.mDests)
            {
                graph.addEdge(edge_info.mSrc, tens_ptr, d);
            }
        }
        else if(edge_info.mSrc == nullptr)
        {
            for(const auto& d : edge_info.mDests)
            {
                graph.addEdgeFromSrc(d, tens_ptr);
            }
        }
        else if(edge_info.mDests.empty())
        {
            graph.addEdgeToSink(edge_info.mSrc, tens_ptr);
        }
    }

    return graph;
}

void OpGraph::initEngines()
{
    // cache the engines in the graph.
    // NOTE(amber): this may be expensive and there may be benefit in delaying
    // findEngines to when the user calls it instead of calling it at graph build
    // time, but cudnn graph API has semantics that suggest that graph knows its
    // engines or engine count at least.
    //
    /// \todo findEngines takes pointer to the graph and uses it to construct
    // engines. This pointer  may become invalid when the graph object is moved. Fix
    // by using shared_ptr or not storing graph inside engine
    // --amberhassaan May, 2024
    mEngines = findEngines(this);
}

VecOfPaths OpGraph::getAllPaths() const
{
    /// \todo does not check for cycles. Use DFS to first check for cycles
    /// at construction time perhaps. --amberhassaan May, 2024
    VecOfPaths all_paths;

    std::deque<Path> paths_to_explore;
    paths_to_explore.emplace_back(Path{mSrcNode.get()});

    while(!paths_to_explore.empty())
    {
        Path path = paths_to_explore.front();
        paths_to_explore.pop_front();

        assert(!path.empty());
        const OpNode* last_node = path.back();
        assert(last_node);
        if(last_node->getOutEdges().empty())
        {
            // all paths should terminate at the sink
            assert(last_node == mSinkNode.get());
            all_paths.emplace_back(std::move(path));
        }
        else
        {
            for(const auto& [dst, tens_ptr] : last_node->getOutEdges())
            {
                Path newPath{path};
                newPath.emplace_back(dst);
                paths_to_explore.emplace_back(std::move(newPath));
            }
        }
    } // end while

    return all_paths;
}

std::string pathToStr(const Path& path)
{
    std::ostringstream oss;
    for(const OpNode* n : path)
    {
        oss << n->signName() << ",";
    }
    return oss.str();
}

namespace internal {

using MapSizeToPathVec = std::unordered_map<size_t, VecOfPaths>;

bool checkSameNodesByName(const OpGraph& left, const OpGraph& right)
{
    auto l_names = left.getNodeNames();
    auto r_names = right.getNodeNames();
    if(l_names.size() != r_names.size())
    {
        return false;
    }

    std::sort(l_names.begin(), l_names.end());
    std::sort(r_names.begin(), r_names.end());

    return l_names == r_names;
}

bool checkSameDegreeVecs(const OpGraph& left, const OpGraph& right)
{
    auto l_degs = left.getInOutDegrees();
    auto r_degs = right.getInOutDegrees();

    std::sort(l_degs.begin(), l_degs.end());
    std::sort(r_degs.begin(), r_degs.end());
    return l_degs == r_degs;
}

auto groupBySize(VecOfPaths&& all_paths)
{
    MapSizeToPathVec paths_by_size;

    for(auto& p : all_paths)
    {
        auto [it, _ignore] = paths_by_size.emplace(p.size(), VecOfPaths{});
        it->second.emplace_back(std::move(p));
    }

    return paths_by_size;
}

bool checkSamePathVecs(const VecOfPaths& left, const VecOfPaths& right)
{
    if(left.size() != right.size())
    {
        return false;
    }

    using VecOfStr = std::vector<std::string>;

    auto pathvec_to_strvec = [](const VecOfPaths& pathvec) {
        VecOfStr ret;
        for(const Path& path : pathvec)
        {
            ret.emplace_back(pathToStr(path));
        }
        return ret;
    };

    VecOfStr l_paths_as_str = pathvec_to_strvec(left);
    VecOfStr r_paths_as_str = pathvec_to_strvec(right);

    std::sort(l_paths_as_str.begin(), l_paths_as_str.end());
    std::sort(r_paths_as_str.begin(), r_paths_as_str.end());

    return l_paths_as_str == r_paths_as_str;
}

bool checkSamePaths(const OpGraph& left, const OpGraph& right)
{

    auto l_paths = left.getAllPaths();
    auto r_paths = right.getAllPaths();

    if(l_paths.size() != r_paths.size())
    {
        return false;
    }

    auto l_paths_by_sz = groupBySize(std::move(l_paths));
    auto r_paths_by_sz = groupBySize(std::move(r_paths));

    auto get_keys = [](const MapSizeToPathVec& paths_by_size) {
        std::vector<size_t> keys{};
        for(const auto& [k, v] : paths_by_size)
        {
            keys.emplace_back(k);
        }
        return keys;
    };

    auto l_keys = get_keys(l_paths_by_sz);
    auto r_keys = get_keys(r_paths_by_sz);

    if(l_keys != r_keys)
    {
        return false;
    }

    for(size_t k : l_keys)
    {
        if(!checkSamePathVecs(l_paths_by_sz[k], r_paths_by_sz[k]))
        {
            return false;
        }
    }

    return true;
}

} // end namespace internal

bool isIsomorphic(const OpGraph& left, const OpGraph& right)
{
    if(left.numNodes() != right.numNodes())
    {
        MIOPEN_LOG_I2("test failed due to num nodes being different");
        return false;
    }

    if(left.numEdges() != right.numEdges())
    {
        MIOPEN_LOG_I2("test failed due to num edges being different");
        return false;
    }

    if(!internal::checkSameNodesByName(left, right))
    {
        MIOPEN_LOG_I2("test failed due to node names being different");
        return false;
    }

    if(!internal::checkSameDegreeVecs(left, right))
    {
        MIOPEN_LOG_I2("test failed due to node degrees being different");
        return false;
    }

    if(!internal::checkSamePaths(left, right))
    {
        MIOPEN_LOG_I2("test failed due to paths being different");
        return false;
    }

    return true;
}

void BackendOperationGraphDescriptor::setAttribute(miopenBackendAttributeName_t attributeName,
                                                   miopenBackendAttributeType_t attributeType,
                                                   int64_t elementCount,
                                                   void* arrayOfElements)
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    switch(attributeName)
    {
    case MIOPEN_ATTR_OPERATIONGRAPH_HANDLE:
        if(attributeType == MIOPEN_TYPE_HANDLE && elementCount == 1)
        {
            mBuilder.setHandle(*static_cast<miopenHandle_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATIONGRAPH_OPS:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && elementCount > 0)
        {
            std::vector<miopenBackendDescriptor_t> descriptors;
            descriptors.reserve(elementCount);
            std::vector<OpNode*> nodes;
            nodes.reserve(elementCount);

            // for_each_n is not available on RHEL/SLES, see issue #2973
            std::for_each(static_cast<miopenBackendDescriptor_t*>(arrayOfElements),
                          static_cast<miopenBackendDescriptor_t*>(arrayOfElements) + elementCount,
                          [&descriptors, &nodes](miopenBackendDescriptor_t apiDescriptor) {
                              BackendDescriptor& backendDescriptor = deref(apiDescriptor);
                              if(backendDescriptor.isFinalized())
                              {
                                  descriptors.push_back(apiDescriptor);
                                  nodes.push_back(backendDescriptor.getOperation());
                              }
                              else
                              {
                                  MIOPEN_THROW(miopenStatusBadParm, "descriptor not finalized");
                              }
                          });

            if(!internal::noRepetitions(nodes))
            {
                MIOPEN_THROW(miopenStatusBadParm, "Repeated node pointer found");
            }

            mBuilder.setNodes(std::move(nodes));
            mOps = std::move(descriptors);
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm, "Invalid attribute type or count");
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

void BackendOperationGraphDescriptor::finalize()
{
    if(mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }
    if(mBuilder.getHandle() == nullptr) // this is not checked by build() so far but API requires
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    mOpGraph = std::move(mBuilder).build();
    mOpGraph.initEngines();
    mFinalized = true;
}

void BackendOperationGraphDescriptor::getAttribute(miopenBackendAttributeName_t attributeName,
                                                   miopenBackendAttributeType_t attributeType,
                                                   int64_t requestedElementCount,
                                                   int64_t* elementCount,
                                                   void* arrayOfElements)
{
    if(!mFinalized)
    {
        MIOPEN_THROW(miopenStatusNotInitialized);
    }

    switch(attributeName)
    {
    case MIOPEN_ATTR_OPERATIONGRAPH_HANDLE:
        if(attributeType == MIOPEN_TYPE_HANDLE && requestedElementCount == 1)
        {
            *elementCount                                  = 1;
            *static_cast<miopenHandle_t*>(arrayOfElements) = mOpGraph.getHandle();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATIONGRAPH_OPS:
        if(attributeType == MIOPEN_TYPE_BACKEND_DESCRIPTOR && requestedElementCount >= 0)
        {
            *elementCount = mOps.size();
            std::copy_n(mOps.cbegin(),
                        minimum(*elementCount, requestedElementCount),
                        static_cast<miopenBackendDescriptor_t*>(arrayOfElements));
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    case MIOPEN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT:
        if(attributeType == MIOPEN_TYPE_INT64 && requestedElementCount == 1)
        {
            *elementCount                           = 1;
            *static_cast<int64_t*>(arrayOfElements) = mOpGraph.getEngines().size();
        }
        else
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        break;

    default: MIOPEN_THROW(miopenStatusBadParm);
    }
}

} // end namespace graphapi
} // end namespace miopen
