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
#include <miopen/solver.hpp>
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

std::ostream& operator<<(std::ostream& stream, const MDGraph_vertex& v)
{
    MIOPEN_LOG_ENUM(stream,
                    v.op,
                    miopenFusionOpConvForward,
                    miopenFusionOpActivForward,
                    miopenFusionOpBatchNormInference,
                    miopenFusionOpBiasForward);
    stream << " program: " << v.vertex_data.at("program")
           << " kernel: " << v.vertex_data.at("kernel")
           << " algorithm : " << v.vertex_data.at("algorithm");
    return stream;
}

MDGraph_vertex_ptr FusionMDGraph::GetCurVertex()
{
    auto& cur_map           = cur_vertex[0].second;
    int weight              = boost::any_cast<int>(cur_map["weight"]);
    MDGraph_vertex_ptr& ptr = cur_vertex[0].first;

    for(auto& cur : cur_vertex)
    {
        if(boost::any_cast<int>(cur.second["weight"]) > weight)
        {
            weight = boost::any_cast<int>(cur.second["weight"]);
            ptr    = cur.first;
        }
    }

    return ptr;
}
std::vector<solver::AnySolver> FusionMDGraph::GetSolvers()
{
    // sort according to the edge weight
    std::sort(cur_vertex.begin(),
              cur_vertex.end(),
              [&](const std::pair<MDGraph_vertex_ptr, cur_vertex_map>& a,
                  const std::pair<MDGraph_vertex_ptr, cur_vertex_map>& b) {
                  return boost::any_cast<int>(a.second.at("weight")) >
                         boost::any_cast<int>(b.second.at("weight"));
              });

    // return a vector of just the solvers
    std::vector<solver::AnySolver> res;
    for(auto& cur : cur_vertex)
    {
        if(cur.second.find("solver") != cur.second.end())
        {
            res.push_back(boost::any_cast<solver::AnySolver>(cur.second.at("solver")));
        }
    }
    return res;
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
        MIOPEN_LOG_I2("Invalid FusionPlan");
        MIOPEN_THROW(miopenStatusBadParm);
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
        MIOPEN_LOG_I2("Invalid FusionPlan");
        MIOPEN_THROW(miopenStatusBadParm);
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
        MIOPEN_LOG_I2("Invalid FusionPlan");
        MIOPEN_THROW(miopenStatusBadParm);
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
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "Either the last added convolution operator does not "
                     "support the requested algorithm or the last added "
                     "opeartor is not convolution");
    }

    if(conv_algo_set.find(algo) == conv_algo_set.end())
    {
        MIOPEN_THROW(miopenStatusBadParm,
                     "The last convolution operator does not support the requested algorithm");
    }
    std::vector<std::pair<MDGraph_vertex_ptr, cur_vertex_map>> new_list;

    for(auto& kinder : cur_vertex)
    {
        MDGraph_vertex_ptr& cur_vertex_ptr = kinder.first;
        auto& cur_map                      = kinder.second;
        if(cur_map.find("algo") != cur_map.end())
        {
            miopenConvFwdAlgorithm_t a = boost::any_cast<miopenConvFwdAlgorithm_t>(cur_map["algo"]);
            if(a == algo)
            {
                new_list.emplace_back(cur_vertex_ptr, cur_map);
            }
        }
        else
        {
            MIOPEN_LOG_I("Current fusion plan does not support the algorithm requested");
            MIOPEN_THROW(miopenStatusBadParm);
        }
    }

    cur_vertex = new_list;

    return (!new_list.empty());
}

void FusionMDGraph::Init(FusionMDGraph& g, miopenFusionOp_t op)
{
    switch(op)
    {
    case miopenFusionOpConvForward: InitConv(g); break;
    case miopenFusionOpBatchNormInference: InitBN(g); break;
    case miopenFusionOpActivForward:
    case miopenFusionOpBiasForward:
        MIOPEN_THROW(
            miopenStatusNotImplemented,
            "Operators Activ and Bias are not supported as first ops in a Fusion Plan (yet)");
    }
}

FusionMDGraph_Edge_Map FusionMDGraph::EmptyEdgeMap(int weight /* = 0 */,
                                                   MDGraph_op_t op /* = OpAny */)
{
    return {{"weight", {EdgeOp(weight, true, op)}}};
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

void FusionMDGraph::InitConv(FusionMDGraph& g)
{

    FusionMDGraph_Edge_Map empty_map = FusionMDGraph::EmptyEdgeMap();
    if(!miopen::IsDisabled(MIOPEN_DEBUG_AMD_FUSED_WINOGRAD{}))
    { /// Fused Winograd.
        static const std::string program("conv_3x3_wheel_alpha_v9_2_7_GFX*_md10.so");
        static const std::string kernel("sp3AsmConvRxSU_CBA");
        static const std::string algo("miopenConvolutionWinogradBiasActiv");
        auto vc =
            std::make_shared<MDGraph_vertex>(miopenFusionOpConvForward, program, kernel, algo);
        vc->solver = solver::ConvBinWinogradRxS{};
        /// \todo Only 0x0 padding for now. 9_2_7 supports asymmetric padding, from 0 to 2^16.
        /// \todo Winograd supports wide range of RxS. 3x3 only for now.
        auto map_wino_conv = ConvForwardOpDescriptor::MDGraphKey(miopenConvolution,
                                                                 miopenPaddingDefault,
                                                                 /*pad_h*/ 0,
                                                                 /*pad_w*/ 0,
                                                                 /* u */ 1,
                                                                 /* v */ 1,
                                                                 /*dilation_h*/ 1,
                                                                 /*dilation_w*/ 1,
                                                                 /*k any*/ 0,
                                                                 /*c any*/ 0,
                                                                 /* x */ 3,
                                                                 /* y */ 3);
        map_wino_conv["c"].push_back(EdgeOp(0, true, OpNotEqual));
        map_wino_conv["c"].push_back(EdgeOp(2, 0, OpModulo));
        map_wino_conv["c"].push_back(EdgeOp(18, true, OpGTE));
        map_emplace(map_wino_conv, "weight", EdgeOp(10, true, OpAny));
        map_emplace(map_wino_conv, "algo", EdgeOp(miopenConvolutionFwdAlgoWinograd, true, OpAny));
        map_emplace(map_wino_conv, "precision", EdgeOp(miopenFloat, true, OpEqual));

        g.AddEdge(nullptr, vc, map_wino_conv);

        /// C>B>A| (4)
        auto vb =
            std::make_shared<MDGraph_vertex>(miopenFusionOpBiasForward, program, kernel, algo);
        g.AddEdge(vc, vb, empty_map);

        auto va_leaf = std::make_shared<MDGraph_vertex>(
            miopenFusionOpActivForward, program, kernel, algo, true);

        FusionMDGraph_Edge_Map edg_activ_relu =
            ActivFusionOpDescriptor::MDGraphKey(miopenActivationRELU);
        map_emplace(edg_activ_relu, "weight", EdgeOp(0, true, OpAny));
        map_emplace(edg_activ_relu, "precision", EdgeOp(miopenFloat, true, OpEqual));

        FusionMDGraph_Edge_Map edg_activ_leaky_relu =
            ActivFusionOpDescriptor::MDGraphKey(miopenActivationLEAKYRELU);
        map_emplace(edg_activ_leaky_relu, "weight", EdgeOp(0, true, OpAny));
        map_emplace(edg_activ_leaky_relu, "precision", EdgeOp(miopenFloat, true, OpEqual));

        g.AddEdge(vb, va_leaf, edg_activ_relu);
        g.AddEdge(vb, va_leaf, edg_activ_leaky_relu);

        /// C>A| (5)
        g.AddEdge(vc, va_leaf, edg_activ_relu);
        g.AddEdge(vc, va_leaf, edg_activ_leaky_relu);

        /// \FIXME Bug: In spite of C>B| topology is disabled below, it is selected anyway for
        /// Winograd. Possible reason is presence of C>B>A| configuration, which is somehow matches
        /// C>B| fused configuration. Fortunately, it is supported.
        ///
        /// C>B| (6)
        /// \todo Shader supports this config, but it is not required for now.
        /// auto vb_leaf = std::make_shared<MDGraph_vertex>(miopenFusionOpBiasForward,  program,
        /// kernel, algo, true);
        /// g.AddEdge(vc, vb_leaf, edg_activ_relu);
        /// g.AddEdge(vc, vb_leaf, edg_activ_leaky_relu);
    }

    // first path (asm kernel)
    { // Conv -> Bias -> Activ // Conv -> Activ
        auto conv_v = std::make_shared<MDGraph_vertex>(miopenFusionOpConvForward,
                                                       "conv1x1u_bias_activ.s",
                                                       "gcnAsmConv1x1U",
                                                       "miopenConvolutionDirectBiasActivAsm");
        conv_v->solver = solver::ConvActivAsm1x1U{};

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
        auto map_asm_conv = ConvForwardOpDescriptor::MDGraphKey(miopenConvolution,
                                                                miopenPaddingDefault,
                                                                /*pad_h*/ 0,
                                                                /*pad_w*/ 0,
                                                                /* u */ 1,
                                                                /* v */ 1,
                                                                /*dilation_h*/ 1,
                                                                /*dilation_w*/ 1,
                                                                /*k any*/ 0,
                                                                /*c any*/ 0,
                                                                /* x */ 1,
                                                                /* y */ 1);
        map_emplace(map_asm_conv, "weight", EdgeOp(1, true, OpAny));
        map_emplace(map_asm_conv, "algo", EdgeOp(miopenConvolutionFwdAlgoDirect, true, OpAny));
        map_emplace(map_asm_conv, "precision", EdgeOp(miopenFloat, true, OpEqual));

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

        conv_v->solver = solver::ConvOclDirectFwdFused{};

        // from ConvolutionDescriptor::IsDirectSupported
        std::vector<size_t> lens = {1, 3, 5, 7, 9, 11};
        for(auto len : lens)
        {
            auto map_conv_bias = ConvForwardOpDescriptor::MDGraphKey(miopenConvolution,
                                                                     miopenPaddingDefault,
                                                                     /*pad_h*/ 0,
                                                                     /*pad_w*/ 0,
                                                                     /* u */ 1,
                                                                     /* v */ 1,
                                                                     /*dilation_h*/ 1,
                                                                     /*dilation_w*/ 1,
                                                                     /*k any*/ 0,
                                                                     /*c any*/ 0,
                                                                     /* x */ len,
                                                                     /* y */ len);
            map_emplace(map_conv_bias, "weight", EdgeOp(0, true, OpAny));
            map_emplace(map_conv_bias, "algo", EdgeOp(miopenConvolutionFwdAlgoDirect, true, OpAny));

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
                auto edg_activ =
                    BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNPerActivation);
                map_emplace(edg_activ, "weight", EdgeOp(0, true, OpAny));

                auto edg_spatial =
                    BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNSpatial);
                map_emplace(edg_spatial, "weight", EdgeOp(0, true, OpAny));

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
            auto edg_activ =
                BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNPerActivation);
            map_emplace(edg_activ, "weight", EdgeOp(0, true, OpAny));

            auto edg_spatial = BatchNormInferenceFusionOpDescriptor::MDGraphKey(miopenBNSpatial);
            map_emplace(edg_spatial, "weight", EdgeOp(0, true, OpAny));

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
    if(edge_list[src][dst].empty())
    {
        edge_list[src][dst] = {map};
    }
    else
    {
        edge_list[src][dst].emplace_back(map);
    }
}

bool FusionMDGraph::ExecOpEqual(const EdgeOp& edg_op, const EdgeOp& op_val)
{
    if(edg_op.val.type() != op_val.val.type())
        return false;
    if(edg_op.val.type() == typeid(std::string))
        return boost::any_cast<std::string>(edg_op.val) == boost::any_cast<std::string>(op_val.val);
    else if(edg_op.val.type() == typeid(int))
        return boost::any_cast<int>(edg_op.val) == boost::any_cast<int>(op_val.val);
    else if(edg_op.val.type() == typeid(miopenConvolutionMode_t))
        return boost::any_cast<miopenConvolutionMode_t>(edg_op.val) ==
               boost::any_cast<miopenConvolutionMode_t>(op_val.val);
    else if(edg_op.val.type() == typeid(miopenPaddingMode_t))
        return boost::any_cast<miopenPaddingMode_t>(edg_op.val) ==
               boost::any_cast<miopenPaddingMode_t>(op_val.val);
    else if(edg_op.val.type() == typeid(size_t))
        return boost::any_cast<size_t>(edg_op.val) == boost::any_cast<size_t>(op_val.val);
    else if(edg_op.val.type() == typeid(miopenBatchNormMode_t))
        return boost::any_cast<miopenBatchNormMode_t>(edg_op.val) ==
               boost::any_cast<miopenBatchNormMode_t>(op_val.val);
    else if(edg_op.val.type() == typeid(miopenActivationMode_t))
        return boost::any_cast<miopenActivationMode_t>(edg_op.val) ==
               boost::any_cast<miopenActivationMode_t>(op_val.val);
    else if(edg_op.val.type() == typeid(miopenDataType_t))
        return boost::any_cast<miopenDataType_t>(edg_op.val) ==
               boost::any_cast<miopenDataType_t>(op_val.val);
    else
    {
        MIOPEN_LOG_I("Unsupported Graph Edge Operation");
        MIOPEN_THROW(miopenStatusNotImplemented);
    }
}

bool FusionMDGraph::ExecOpModulo(const EdgeOp& edg_op, const EdgeOp& op_val)
{
    if(!(edg_op.val.type() == typeid(int) && op_val.val.type() == typeid(int) &&
         edg_op.result.type() == typeid(int)))
    {
        MIOPEN_LOG_I("Invalid operand types for Edge Op OpModulo");
        MIOPEN_THROW(miopenStatusBadParm);
    }

    return (boost::any_cast<int>(op_val.val) % boost::any_cast<int>(edg_op.val)) ==
           boost::any_cast<int>(edg_op.result);
}

bool FusionMDGraph::ExecOpGTE(const EdgeOp& edg_op, const EdgeOp& op_val)
{
    if(!(edg_op.val.type() == typeid(int) && op_val.val.type() == typeid(int)))
    {
        MIOPEN_LOG_I("Invalid operand types for Edge Op OpGTE (>=)");
        MIOPEN_THROW(miopenStatusBadParm);
    }
    return (boost::any_cast<int>(op_val.val) >= boost::any_cast<int>(edg_op.val));
}
bool FusionMDGraph::ExecEdgeOp(const EdgeOp& edg_op, const EdgeOp& op_val)
{
    switch(edg_op.op)
    {
    case OpEqual: { return FusionMDGraph::ExecOpEqual(edg_op, op_val);
    }
    case OpNotEqual: { return !(FusionMDGraph::ExecOpEqual(edg_op, op_val));
    }
    case OpAny: { return true;
    }
    case OpModulo: { return FusionMDGraph::ExecOpModulo(edg_op, op_val);
    }
    case OpGTE: { return FusionMDGraph::ExecOpGTE(edg_op, op_val);
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
            if(op_val.at(kv.first).size() > 1)
            {
                MIOPEN_LOG_I("The operator attribute vector length cannot be greater than 1");
                MIOPEN_THROW(miopenStatusInternalError);
            }
            for(auto& edg_ops : kv.second)
            {
                if(!FusionMDGraph::ExecEdgeOp(edg_ops, op_val.at(kv.first).at(0)))
                {
                    MIOPEN_LOG_I2("Edge Op :" << edg_ops << " Op Val: " << op_val.at(kv.first).at(0)
                                              << " Edge Op for key: "
                                              << kv.first
                                              << " Failed");
                    return false;
                }
            }
            MIOPEN_LOG_I("Edge Op for key: " << kv.first << " Successfull");
        }
        else
        {
            MIOPEN_LOG_I("Key: " << kv.first << " NOT found");
        }
    }
    return true;
}

bool FusionMDGraph::Advance(std::shared_ptr<FusionOpDescriptor> op)
{
    MIOPEN_LOG_I("Adding Op: " << *op);
    std::vector<std::pair<MDGraph_vertex_ptr, cur_vertex_map>> new_list;
    std::set<miopenConvFwdAlgorithm_t> new_set;
    // iterate over the list of current vertices
    for(auto& kinder : cur_vertex)
    {
        MDGraph_vertex_ptr& cur_vertex_ptr = kinder.first;
        if(cur_vertex_ptr == nullptr)
        {
            MIOPEN_LOG_I("Current vertex: nullptr");
        }
        else
        {
            MIOPEN_LOG_I("Current vertex: " << *cur_vertex_ptr);
        }
        auto cur_map = kinder.second;
        // get the children of the cur_vertex
        auto& ch = edge_list[cur_vertex_ptr];
        // if op is in the children and the edge key satisfies update cur_vertex
        for(auto& ch_it : ch)
        {
            MIOPEN_LOG_I("Child: " << *ch_it.first);
            std::set<miopenConvFwdAlgorithm_t> cur_path_set;
            if(ch_it.first->op == op->kind())
            {
                for(auto& edg_map : ch_it.second)
                {
                    int weight = boost::any_cast<int>(cur_map["weight"]);
                    if(CmpOpKey(edg_map, op->MDGraphKey()))
                    {
                        MIOPEN_LOG_I("Key Match Successfull");
                        weight += boost::any_cast<int>(edg_map.at("weight").at(0).val);
                        cur_map["weight"] = weight;

                        // Update the algo set
                        if(op->kind() == miopenFusionOpConvForward)
                        {
                            miopenConvFwdAlgorithm_t algo =
                                boost::any_cast<miopenConvFwdAlgorithm_t>(
                                    edg_map.at("algo").at(0).val);
                            MIOPEN_LOG_I("Operator Matched: Convolution: Algo: " +
                                         std::to_string(algo));
                            cur_path_set.insert(algo);

                            new_set.insert(cur_path_set.begin(), cur_path_set.end());
                            assert(cur_path_set.size() == 1);
                            cur_map["algo"] =
                                *cur_path_set.begin(); // there should be only one algo
                            cur_map.erase("solver");
                            if(!ch_it.first->solver.IsEmpty())
                            {
                                cur_map.insert(std::pair<std::string, solver::AnySolver>(
                                    "solver", ch_it.first->solver));
                            }
                        }
                        else
                        {
                            MIOPEN_LOG_I("Operator Matched: " + std::to_string(op->kind()));
                            cur_map.erase("algo");
                        }
                        new_list.emplace_back(ch_it.first, cur_map);
                    }
                    else
                    {
                        MIOPEN_LOG_I("Key Map Match failed");
                    }
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

    return (!cur_vertex.empty());
}

void FusionMDGraph::Reset()
{
    cur_vertex.clear();
    cur_vertex_map empty_map = {{"weight", 0}};
    cur_vertex.emplace_back(nullptr, empty_map);
}

} // namespace miopen
