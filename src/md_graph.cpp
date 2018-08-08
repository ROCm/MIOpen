#include <miopen/md_graph.hpp>

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
    int weight              = cur_vertex[0].second;
    MDGraph_vertex_ptr& ptr = cur_vertex[0].first;

    for(auto& cur : cur_vertex)
    {
        if(cur.second > weight)
        {
            weight = cur.second;
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
    case miopenFusionOpConvForward: { InitConv(g);
    }
    break;
    case miopenFusionOpBatchNormInference: { InitBN(g);
    }
    break;
    case miopenFusionOpActivForward:
    case miopenFusionOpBiasForward:
        MIOPEN_THROW("Operators Activ and Bias are not supported as first ops in a Fusion Plan");
    }
}

void FusionMDGraph::InitBN(FusionMDGraph& g)
{
    FusionMDGraph_Edge_Map empty_map = {{"key", {}}, {"weight", {"0"}}};

    {
        auto bn_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormInference,
                                                     "MIOpenBatchNormActivInfer.cl",
                                                     "MIOpenBatchNormActivInferPerActEst",
                                                     "MIOpenBatchNormActivInferPerActEst");
        FusionMDGraph_Edge_Map edg_activ = {
            {"key", {BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNPerActivation)}},
            {"weight", {"0"}}};
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
        FusionMDGraph_Edge_Map edg_spatial = {
            {"key", {BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNSpatial)}},
            {"weight", {"0"}}};
        g.AddEdge(nullptr, bn_v, edg_spatial);
        auto activ_v = std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
                                                        "MIOpenBatchNormActivInfer.cl",
                                                        "MIOpenBatchNormActivInferSpatialEst",
                                                        "MIOpenBatchNormActivInferSpatialEst");
        g.AddEdge(bn_v, activ_v, empty_map);
    }
}

void FusionMDGraph::InitConv(FusionMDGraph& g)
{
    std::map<std::string, int> defaults = {{"mode", miopenConvolution},
                                           {"paddingMode", miopenPaddingDefault},
                                           {"pad_h", 0},
                                           {"pad_w", 0},
                                           {"u", 1},
                                           {"v", 1},
                                           {"dilation_h", 1},
                                           {"dilation_w", 1}};
    FusionMDGraph_Edge_Map empty_map = {{"key", {}}, {"weight", {"0"}}};
    // first path (asm kernel)
    { // Conv -> Bias -> Activ // Conv -> Activ
        auto conv_v = std::make_shared<MDGraph_vertex>(miopenFusionOpConvForward,
                                                       "conv1x1u_bias_activ.s",
                                                       "gcnAsmConv1x1U",
                                                       "miopenConvolutionDirectBiasActivAsm");
        auto bias_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBiasForward,
                                                       "conv1x1u_bias_activ.s",
                                                       "gcnAsmConv1x1U",
                                                       "miopenConvolutionDirectBiasActivAsm");
        auto activ_v = std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
                                                        "conv1x1u_bias_activ.s",
                                                        "gcnAsmConv1x1U",
                                                        "miopenConvolutionDirectBiasActivAsm",
                                                        true);
        // populate the graph
        auto key = ConvForwardOpDescriptor::MDGraphKey(
            defaults, {0, 0, 1, 1}, miopenConvolutionFwdAlgoDirect);
        FusionMDGraph_Edge_Map map_asm_conv = {{"key", {key}}, {"weight", {"1"}}};

        g.AddEdge(nullptr, conv_v, map_asm_conv);
        g.AddEdge(conv_v, bias_v, empty_map);
        g.AddEdge(bias_v, activ_v, empty_map);

        g.AddEdge(conv_v, activ_v, empty_map);
    }

    // second path (ocl kernel)
    {
        auto conv_v = std::make_shared<MDGraph_vertex>(miopenFusionOpConvForward,
                                                       "MIOpenConvDirBatchNormActiv.cl",
                                                       "MIOpenConvUniBatchNormActiv",
                                                       "miopenConvolutionDirectBiasActiv");

        // from ConvolutionDescriptor::IsDirectSupported
        std::vector<size_t> lens = {1, 3, 5, 7, 9, 11};
        for(auto len : lens)
        {
            auto cb_key = ConvForwardOpDescriptor::MDGraphKey(
                defaults, {0, 0, len, len}, miopenConvolutionFwdAlgoDirect);
            FusionMDGraph_Edge_Map map_conv_bias = {{"key", {cb_key}}, {"weight", {"0"}}};
            g.AddEdge(nullptr, conv_v, map_conv_bias);
        }

        { // Conv -> Bias

            auto bias_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBiasForward,
                                                           "MIOpenConvDirBatchNormActiv.cl",
                                                           "MIOpenConvUniBatchNormActiv",
                                                           "miopenConvolutionDirectBiasActiv");

            g.AddEdge(conv_v, bias_v, empty_map);
            { // Conv -> Bias -> Activ // Conv -> Activ
                auto activ_v = std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
                                                                "MIOpenConvDirBatchNormActiv.cl",
                                                                "MIOpenConvUniBatchNormActiv",
                                                                "miopenConvolutionDirectBiasActiv",
                                                                true);
                g.AddEdge(bias_v, activ_v, empty_map);

                g.AddEdge(conv_v, activ_v, empty_map);
            }

            { // Conv -> Bias -> BatchNorm -> Activ
                auto bn_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormInference,
                                                             "MIOpenConvDirBatchNormActiv.cl",
                                                             "MIOpenConvUniBatchNormActiv",
                                                             "miopenConvDirectBatchNormBiasActiv");
                FusionMDGraph_Edge_Map edg_activ = {
                    {"key",
                     {BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNPerActivation)}},
                    {"weight", {"0"}}};
                FusionMDGraph_Edge_Map edg_spatial = {
                    {"key", {BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNSpatial)}},
                    {"weight", {"0"}}};

                g.AddEdge(bias_v, bn_v, edg_activ);
                g.AddEdge(bias_v, bn_v, edg_spatial);

                auto activ_v =
                    std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
                                                     "MIOpenConvDirBatchNormActiv.cl",
                                                     "MIOpenConvUniBatchNormActiv",
                                                     "miopenConvDirectBatchNormBiasActiv");
                g.AddEdge(bn_v, activ_v, empty_map);
            }
        }

        { // Conv -> BN
            auto bn_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormInference,
                                                         "MIOpenConvDirBatchNormActiv.cl",
                                                         "MIOpenConvUniBatchNormActiv",
                                                         "miopenConvDirectBatchNormBiasActiv");
            FusionMDGraph_Edge_Map edg_activ = {
                {"key", {BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNPerActivation)}},
                {"weight", {"0"}}};
            FusionMDGraph_Edge_Map edg_spatial = {
                {"key", {BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNSpatial)}},
                {"weight", {"0"}}};
            g.AddEdge(conv_v, bn_v, edg_activ);
            g.AddEdge(conv_v, bn_v, edg_spatial);

            auto activ_v = std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
                                                            "MIOpenConvDirBatchNormActiv.cl",
                                                            "MIOpenConvUniBatchNormActiv",
                                                            "miopenConvDirectBatchNormBiasActiv");
            g.AddEdge(bn_v, activ_v, empty_map);
        }
    }
}

void FusionMDGraph::AddEdge(MDGraph_vertex_ptr src,
                            MDGraph_vertex_ptr dst,
                            FusionMDGraph_Edge_Map& map)
{
    if(map.empty())
    {
        edge_list[src][dst]["key"] = {""};
    }
    else
    {
        auto& old_map = edge_list[src][dst];
        for(auto& it : map)
        {
            if(old_map.count(it.first) == 0)
            {
                old_map[it.first] = {it.second};
            }
            else
            {
                old_map[it.first].insert(
                    old_map[it.first].end(), it.second.begin(), it.second.end());
            }
        }
        if(old_map.count("key") == 0)
        {
            edge_list[src][dst]["key"] = {""};
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
    {
        auto it = std::find(edge_val.begin(), edge_val.end(), op_val);
        return (it != edge_val.end());
    }
}

bool FusionMDGraph::Advance(std::shared_ptr<FusionOpDescriptor> op)
{
#if 0
    if(cur_vertex == nullptr)
    {
        if(ops[0]->kind() == root->op)
        {
            cur_vertex = root;
            start_idx = 1;
        }
        else
        {
            MIOPEN_THROW("Operation not supported");
        }
    }
#endif

    std::vector<std::pair<MDGraph_vertex_ptr, int>> new_list;
    // get the children of the cur_vertex
    // unsigned int idx_cur = 0;
    for(auto& kinder : cur_vertex)
    {

        MDGraph_vertex_ptr& cur_vertex_ptr = kinder.first;  // cur_vertex[idx_cur].first;
        int& weight                        = kinder.second; // cur_vertex[idx_cur].second;

        auto& ch = edge_list[cur_vertex_ptr];
        // if op is in the children and the edge key satisfies update cur_vertex
        for(auto& ch_it : ch)
        {
            if(ch_it.first->op == op->kind())
            {
                if(CmpOpKey(ch_it.second["key"], op->MDGraphKey()))
                {
                    weight += std::stoi(ch_it.second["weight"][0]);
                    new_list.emplace_back(std::pair<MDGraph_vertex_ptr, int>(ch_it.first, weight));
                }
            }
        }
        // idx_cur++;
    }
    cur_vertex = new_list;
    return (!cur_vertex.empty());
}

void FusionMDGraph::Reset() { cur_vertex = {{nullptr, 0}}; }

} // namespace miopen
