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

std::vector<miopenConvFwdAlgorithm_t> FusionMDGraph::GetConvAlgos()
{
    std::vector<miopenConvFwdAlgorithm_t> ret(conv_algo_set.begin(), conv_algo_set.end());
    return ret;
}

bool FusionMDGraph::SetConvAlgo(miopenConvFwdAlgorithm_t algo)
{
    // Make sure algo is in the current paths being tracked
    if(conv_algo_set.empty())
        MIOPEN_THROW("No algorithm supported by current fusion plan");

    if(conv_algo_set.find(algo) == conv_algo_set.end())
    {
        MIOPEN_THROW("Current fusion plan does not support the algorithm requested");
    }
    std::vector<std::pair<MDGraph_vertex_ptr, cur_vertex_map>> new_list;

    for(auto& kinder : cur_vertex)
    {
        MDGraph_vertex_ptr& cur_vertex_ptr = kinder.first;
        auto& cur_map                      = kinder.second;
        if(cur_map.find("algo") != cur_map.end())
        {
            miopenConvFwdAlgorithm_t a =
                static_cast<miopenConvFwdAlgorithm_t>(std::stoi(cur_map["algo"]));
            if(a == algo)
            {
                new_list.push_back(
                    std::pair<MDGraph_vertex_ptr, cur_vertex_map>(cur_vertex_ptr, cur_map));
            }
        }
        else
            MIOPEN_THROW("Current fusion plan does not support the algorithm requested");
    }

    cur_vertex = new_list;
    if(new_list.empty())
        return false;
    else
        return true;
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
        MIOPEN_THROW(
            "Operators Activ and Bias are not supported as first ops in a Fusion Plan (yet)");
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
        auto key = ConvForwardOpDescriptor::MDGraphKey(defaults, {0, 0, 1, 1});
        FusionMDGraph_Edge_Map map_asm_conv = {
            {"key", {key}},
            {"weight", {"1"}},
            {"algo", {std::to_string(miopenConvolutionFwdAlgoDirect)}}};

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
            auto cb_key = ConvForwardOpDescriptor::MDGraphKey(defaults, {0, 0, len, len});
            FusionMDGraph_Edge_Map map_conv_bias = {
                {"key", {cb_key}},
                {"weight", {"0"}},
                {"algo", {std::to_string(miopenConvolutionFwdAlgoDirect)}}};
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
        for(auto it = map.begin(); it != map.end(); it++)
        {
            if(old_map.count(it->first) == 0)
            {
                old_map[it->first] = {it->second};
            }
            else
            {
                old_map[it->first].insert(
                    old_map[it->first].end(), it->second.begin(), it->second.end());
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
        if(it != edge_val.end())
            return true;
        else
            return false;
    }
}

bool FusionMDGraph::Advance(std::shared_ptr<FusionOpDescriptor> op)
{

    std::vector<std::pair<MDGraph_vertex_ptr, cur_vertex_map>> new_list;
    std::set<miopenConvFwdAlgorithm_t> new_set;
    // get the children of the cur_vertex
    for(auto& kinder : cur_vertex) //  idx_cur = 0; idx_cur < cur_vertex.size(); idx_cur++)
    {
        MDGraph_vertex_ptr& cur_vertex_ptr = kinder.first;
        auto& cur_map                      = kinder.second;
        int weight                         = std::stoi(cur_map["weight"]);

        auto& ch = edge_list[cur_vertex_ptr];
        // if op is in the children and the edge key satisfies update cur_vertex
        for(auto ch_it = ch.begin(); ch_it != ch.end(); ch_it++)
        {
            std::set<miopenConvFwdAlgorithm_t> cur_path_set;
            if(ch_it->first->op == op->kind())
            {
                if(CmpOpKey(ch_it->second["key"], op->MDGraphKey()))
                {
                    weight += std::stoi(ch_it->second["weight"][0]);
                    cur_map["weight"] = std::to_string(weight);
                    // Update the algo set
                    if(op->kind() == miopenFusionOpConvForward)
                    {
                        for(auto s_algo : ch_it->second["algo"])
                        {
                            miopenConvFwdAlgorithm_t algo =
                                static_cast<miopenConvFwdAlgorithm_t>(std::stoi(s_algo));
                            cur_path_set.insert(algo);
                        }
                        new_set.insert(cur_path_set.begin(), cur_path_set.end());
                        assert(cur_path_set.size() == 1);
                        cur_map["algo"] =
                            std::to_string(*cur_path_set.begin()); // there should be only one algo
                    }
                    else
                    {
                        cur_map["algo"] = "";
                    }
                    new_list.push_back(
                        std::pair<MDGraph_vertex_ptr, cur_vertex_map>(ch_it->first, cur_map));
                }
            }
        }
    }
    cur_vertex = new_list;
    if(op->kind() == miopenFusionOpConvForward) // TODO: Or any other convolution
    {
        conv_algo_set = new_set;
    }
    else
    {
        conv_algo_set.clear();
    }

    if(cur_vertex.size() == 0)
        return false;
    else
        return true;
}

void FusionMDGraph::Reset()
{
    cur_vertex.clear();
    cur_vertex_map empty_map = {{"weight", "0"}};
    cur_vertex.push_back(std::pair<MDGraph_vertex_ptr, cur_vertex_map>(nullptr, empty_map));
}

} // namespace miopen
