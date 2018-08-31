/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#include <miopen/md_graph.hpp>
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_FUSED_WINOGRAD)

namespace miopen {

int MDGraph_vertex::running_id = 0;

MDGraph_vertex::MDGraph_vertex(miopenFusionOp_t o,
                               std::string program_name,
                               std::string kernel_name,
                               std::string algo_name,
                               bool _is_leaf)
    : op(o), is_leaf(_is_leaf), id(MDGraph_vertex::running_id)
{
    MDGraph_vertex::running_id++;
    vertex_data["program"]   = program_name;
    vertex_data["kernel"]    = kernel_name;
    vertex_data["algorithm"] = algo_name;
}

MDGraph_vertex_ptr FusionMDGraph::GetCurVertex()
{
    auto& cur_map           = cur_vertex[0].second;
    int weight              = std::stoi(cur_map["weight"]);
    MDGraph_vertex_ptr& ptr = cur_vertex[0].first;

    for(auto& cur : cur_vertex)
    {
        if(std::stoi(cur.second["weight"]) > weight)
        {
            weight = std::stoi(cur.second["weight"]);
            ptr    = cur.first;
        }
    }

    return ptr;
}

std::string FusionMDGraph::GetProgramName()
{
    auto ptr = GetCurVertex();

    if(ptr != nullptr)
    {
        return ptr->vertex_data["program"];
    }
    else
    {
        MIOPEN_THROW("Invalid FusionPlan");
    }
}

std::string FusionMDGraph::GetKernelName()
{
    auto ptr = GetCurVertex();
    if(ptr != nullptr)
    {
        return ptr->vertex_data["kernel"];
    }
    else
    {
        MIOPEN_THROW("Invalid FusionPlan");
    }
}

std::string FusionMDGraph::GetAlgoName()
{
    auto ptr = GetCurVertex();
    if(ptr != nullptr)
    {
        return ptr->vertex_data["algorithm"];
    }
    else
    {
        MIOPEN_THROW("Invalid FusionPlan");
    }
}

void FusionMDGraph::Init(FusionMDGraph& g, miopenFusionOp_t op)
{
    switch(op)
    {
    case miopenFusionOpBatchNormInference: InitBN(g); break;
    case miopenFusionOpActivForward:
    case miopenFusionOpBiasForward:
    case miopenFusionOpConvForward:
        MIOPEN_THROW(
            "Operators Conv, Activ and Bias are not supported as first ops in a Fusion Plan (yet)");
    }
}

FusionMDGraph_Edge_Map FusionMDGraph::EmptyEdgeMap(int weight /* = 0 */,
                                                   MDGraph_op_t op /* = OpAny */)
{
    return {{"weight", EdgeOp(weight, true, op)}};
}

void FusionMDGraph::InitBN(FusionMDGraph& g)
{
    FusionMDGraph_Edge_Map empty_map = FusionMDGraph::EmptyEdgeMap();

    {
        auto bn_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormInference,
                                                     "MIOpenBatchNormActivInfer.cl",
                                                     "MIOpenBatchNormActivInferPerActEst",
                                                     "MIOpenBatchNormActivInferPerActEst");
        FusionMDGraph_Edge_Map edg_activ =
            BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNPerActivation);
        edg_activ.insert(empty_map.begin(), empty_map.end()); // add the weight etc.

        g.AddEdge(nullptr, bn_v, edg_activ);
        auto activ_v = std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
                                                        "MIOpenBatchNormActivInfer.cl",
                                                        "MIOpenBatchNormActivInferPerActEst",
                                                        "MIOpenBatchNormActivInferPerActEst");
        g.AddEdge(bn_v, activ_v, empty_map);
    }
    {
        auto bn_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormInference,
                                                     "MIOpenBatchNormActivInfer.cl",
                                                     "MIOpenBatchNormActivInferSpatialEst",
                                                     "MIOpenBatchNormActivInferSpatialEst");
        FusionMDGraph_Edge_Map edg_spatial =
            BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNSpatial);
        edg_spatial.insert(empty_map.begin(), empty_map.end());
        g.AddEdge(nullptr, bn_v, edg_spatial);
        auto activ_v = std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
                                                        "MIOpenBatchNormActivInfer.cl",
                                                        "MIOpenBatchNormActivInferSpatialEst",
                                                        "MIOpenBatchNormActivInferSpatialEst");
        g.AddEdge(bn_v, activ_v, empty_map);
    }
}

void FusionMDGraph::AddEdge(MDGraph_vertex_ptr src,
                            MDGraph_vertex_ptr dst,
                            FusionMDGraph_Edge_Map& map)
{
    if(edge_list[src][dst].empty())
    {
        edge_list[src][dst] = {map};
    }
    else
    {
        edge_list[src][dst].emplace_back(map);
    }
}

bool FusionMDGraph::ExecEdgeOp(const EdgeOp& edg_op, const EdgeOp& op_val)
{
    switch(edg_op.op)
    {
    case OpEqual: { return edg_op.val == op_val.val;
    }
    case OpNotEqual: { return edg_op.val != op_val.val;
    }
    case OpAny: { return true;
    }
    }
    return false;
}

bool FusionMDGraph::CmpOpKey(const FusionMDGraph_Edge_Map& edge_val,
                             const FusionMDGraph_Edge_Map& op_val) const
{
    for(auto& kv : edge_val)
    {
        if(op_val.count(kv.first) == 1)
        {
            if(!FusionMDGraph::ExecEdgeOp(kv.second, op_val.at(kv.first)))
                return false;
        }
    }
    return true;
}

bool FusionMDGraph::Advance(std::shared_ptr<FusionOpDescriptor> op)
{

    std::vector<std::pair<MDGraph_vertex_ptr, cur_vertex_map>> new_list;
    // get the children of the cur_vertex
    for(auto& kinder : cur_vertex)
    {
        MDGraph_vertex_ptr& cur_vertex_ptr = kinder.first;
        auto& cur_map                      = kinder.second;
        int weight                         = std::stoi(cur_map["weight"]);

        auto& ch = edge_list[cur_vertex_ptr];
        // if op is in the children and the edge key satisfies update cur_vertex
        for(auto& ch_it : ch)
        {
            if(ch_it.first->op == op->kind())
            {
                for(auto& edg_map : ch_it.second)
                {
                    if(CmpOpKey(edg_map, op->MDGraphKey()))
                    {
                        weight += std::stoi(edg_map.at("weight").val);
                        cur_map["weight"] = std::to_string(weight);
                        cur_map["algo"]   = "";
                        new_list.emplace_back(ch_it.first, cur_map);
                    }
                }
            }
        }
    }
    cur_vertex = new_list;

    return (!cur_vertex.empty());
}

void FusionMDGraph::Reset()
{
    cur_vertex.clear();
    cur_vertex_map empty_map = {{"weight", "0"}};
    cur_vertex.emplace_back(nullptr, empty_map);
}

} // namespace miopen
