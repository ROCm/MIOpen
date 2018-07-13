#pragma once
#include <miopen/miopen.h>
#include <miopen/fusion.hpp>

#include <boost/functional/hash.hpp>

namespace miopen {

using FusionMDGraph_Edge_Map = std::unordered_map<std::string, std::vector<std::string>>;

struct MDGraph_vertex
{
    static int running_id;
    MDGraph_vertex(miopenFusionOp_t o,
                   std::string program_name = "",
                   std::string kernel_name  = "",
                   std::string algo_name    = "",
                   bool _is_leaf            = false);
    miopenFusionOp_t op;
    bool is_leaf = false;
    std::map<std::string, std::string> vertex_data;
    size_t map_hash = 0;
    int id;

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
    static void Init(FusionMDGraph& g, miopenFusionOp_t op);
    static void InitConv(FusionMDGraph& g);
    static void InitBN(FusionMDGraph& g);
    void Reset();
    bool Advance(std::shared_ptr<FusionOpDescriptor> op);
    void AddVeretx(MDGraph_vertex& vertex);
    void AddEdge(MDGraph_vertex_ptr src, MDGraph_vertex_ptr dst, FusionMDGraph_Edge_Map& map);

    template <class T, class U>
    bool CmpOpKey(T&& edge_val, U&& op_val) const;
    MDGraph_vertex_ptr GetCurVertex();
    std::string GetProgramName();
    std::string GetKernelName();
    std::string GetAlgoName();

    protected:
    std::vector<std::pair<MDGraph_vertex_ptr, int>> cur_vertex = {{nullptr, 0}};
    std::unordered_map<MDGraph_vertex_ptr,
                       std::unordered_map<MDGraph_vertex_ptr, FusionMDGraph_Edge_Map>>
        edge_list;
};

} // namespace miopen
