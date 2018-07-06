#pragma once
#include <miopen/miopen.h>
#include <miopen/fusion.hpp>

#include <boost/functional/hash.hpp>

namespace miopen {

using FusionMDGraph_Edge_Map = std::map<std::string, std::string>;

struct MDGraph_vertex
{
    MDGraph_vertex(miopenFusionOp_t o) : op(o){};
    miopenFusionOp_t op;
    bool is_leaf = false;
    std::map<std::string, std::string> vertex_data;
    size_t map_hash = 0;

    MDGraph_vertex(const MDGraph_vertex& other) = delete;
#if 0
    bool operator==(const MDGraph_vertex& other) const
    {
        return (op==other.op) && (is_leaf == other.is_leaf) && (vertex_data == other.vertex_data);
    };
#endif
};

using MDGraph_vertex_ptr = std::shared_ptr<MDGraph_vertex>;

struct FusionMDGraph
{
    void Init();
    void Reset();
    bool Advance(std::vector<std::shared_ptr<FusionOpDescriptor>> ops);
    void AddVeretx(MDGraph_vertex& vertex);
    void AddEdge(MDGraph_vertex_ptr& src, MDGraph_vertex_ptr& dst, FusionMDGraph_Edge_Map& map);

    template <class T, class U>
    bool CmpOpKey(T&& edge_val, U&& op_val) const;

    protected:
    MDGraph_vertex_ptr root       = nullptr;
    MDGraph_vertex_ptr cur_vertex = nullptr;
    std::unordered_map<MDGraph_vertex_ptr,
                       std::unordered_map<MDGraph_vertex_ptr, FusionMDGraph_Edge_Map>>
        edge_list;
};

} // namespace miopen
