#include <miopen/md_graph.hpp>

namespace miopen {

void FusionMDGraph::Init()
{
#if 0
    MDGraph_vertex conv_v(miopenFusionOpConvForward);
    MDGraph_vertex bias_v(miopenFusionOpBiasForward);
    MDGraph_vertex activ_v(miopenFusionOpActivForward);

#endif
    auto conv_v  = std::make_shared<MDGraph_vertex>(miopenFusionOpConvForward);
    auto bias_v  = std::make_shared<MDGraph_vertex>(miopenFusionOpBiasForward);
    auto activ_v = std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward);
    // populate the graph
    FusionMDGraph_Edge_Map map_asm_conv = {{"key", "0,0,1,1,0,0"}};
    FusionMDGraph_Edge_Map empty_map    = {};
    AddEdge(conv_v, bias_v, map_asm_conv);
    AddEdge(bias_v, activ_v, empty_map);
}
void FusionMDGraph::AddEdge(MDGraph_vertex_ptr& src,
                            MDGraph_vertex_ptr& dst,
                            FusionMDGraph_Edge_Map& map)
{
    // Is this the first edge to be inserted
    if(root == nullptr)
    {
        root = src;
    }
    if(map.empty())
    {
        edge_list[src][dst]["key"] = "";
    }
    else
    {
        edge_list[src][dst] = map;
        if(map.count("key") == 0)
        {
            edge_list[src][dst]["key"] = "";
        }
    }
}

template <class T, class U>
bool FusionMDGraph::CmpOpKey(T&& edge_val, U&& op_val) const
{
    // if the edge has no value set, anything matches
    if(edge_val.empty())
        return true;
    else
        return edge_val == op_val;
}

bool FusionMDGraph::Advance(std::vector<std::shared_ptr<FusionOpDescriptor>> ops)
{
    size_t start_idx = 0;
    if(cur_vertex == nullptr)
    {
        if(ops[0]->kind() == root->op)
        {
            cur_vertex = root;
            start_idx  = 1;
        }
        else
        {
            MIOPEN_THROW("Operation not supported");
        }
    }

    for(auto idx = start_idx; idx < ops.size(); idx++)
    {
        auto op = ops[idx];
        // get the children of the cur_vertex
        auto& ch = edge_list[cur_vertex];
        // if op is in the children and the edge key satisfies update cur_vertex
        for(auto ch_it = ch.begin(); ch_it != ch.end(); ch_it++)
        {
            if(ch_it->first->op == op->kind() /* &&  */)
            {
                if(CmpOpKey(ch_it->second["key"], op->MDGraphKey()))
                {
                    cur_vertex = ch_it->first;
                }
                else
                {
                    // MIOPEN_THROW("Operator Config not supported");
                    return false;
                }
            }
            else
            {
                MIOPEN_THROW("Unsupported Operator");
            }
        }
    }
    return true;
}

void FusionMDGraph::Reset() { cur_vertex = nullptr; }

} // namespace miopen
