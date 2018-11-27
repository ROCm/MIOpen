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
#include <miopen/mdg_expr.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_FUSED_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_KERNELS)

namespace miopen {

int MDGraph_vertex::running_id = 1;

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

MDGraph_vertex_ptr FusionMDGraph::GetCurVertex(Handle& handle)
{
    int weight             = -1;
    MDGraph_vertex_ptr ptr = nullptr;
    auto cur_arch          = handle.GetDeviceName();

    for(auto& cur : cur_vertex)
    {
        auto it =
            std::find(cur.first->supported_arch.begin(), cur.first->supported_arch.end(), cur_arch);
        // Empty inidicates any arch is supported (say OpenCL kernels)
        bool arch_sup =
            cur.first->supported_arch.empty() || (it != cur.first->supported_arch.end());
        if((boost::any_cast<int>(cur.second["weight"]) > weight) && arch_sup)
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

std::string FusionMDGraph::GetProgramName(Handle& handle)
{
    auto ptr = GetCurVertex(handle);

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

std::string FusionMDGraph::GetKernelName(Handle& handle)
{
    auto ptr = GetCurVertex(handle);
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

std::string FusionMDGraph::GetAlgoName(Handle& handle)
{
    auto ptr = GetCurVertex(handle);
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

std::vector<DefaultKernelArg> FusionMDGraph::GetKernelArgs(Handle& handle)
{
    auto ptr = GetCurVertex(handle);
    if(ptr != nullptr)
    {
        return ptr->default_args;
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
        auto& cur_map = kinder.second;
        if(cur_map.find("algo") != cur_map.end())
        {
            miopenConvFwdAlgorithm_t a = boost::any_cast<miopenConvFwdAlgorithm_t>(cur_map["algo"]);
            if(a == algo)
            {
                new_list.emplace_back(kinder.first, cur_map);
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
    case miopenFusionOpBatchNormFwdTrain: InitBNFwd(g); break;
    case miopenFusionOpBatchNormBwdTrain: InitBNBwd(g); break;
    case miopenFusionOpActivForward:
    case miopenFusionOpActivBackward:
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

static std::vector<DefaultKernelArg> BNFwdArgs(miopenBatchNormMode_t mode)
{
    if(mode == miopenBNPerActivation)
    {
        return {
            DefaultKernelArg("activAlpha", OpArg, OpKernelArg(static_cast<int>(0)), 1),
            DefaultKernelArg("activBeta", OpArg, OpKernelArg(static_cast<int>(0)), 1),
            DefaultKernelArg("activGamma", OpArg, OpKernelArg(static_cast<int>(0)), 1),
            DefaultKernelArg("epsilon", OpArg, OpKernelArg(static_cast<double>(0.0)), 0),
            DefaultKernelArg("expAvgFactor", OpArg, OpKernelArg(static_cast<double>(0.0)), 0),
            DefaultKernelArg("input", InputTensor, OpKernelArg(nullptr)),
            DefaultKernelArg("output", OutputTensor, OpKernelArg(nullptr)),
            DefaultKernelArg("bnBias", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("bnScale", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("runningMean", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("runningVariance", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("savedInvVariance", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("savedMean", OpArg, OpKernelArg(nullptr), 0),
        };
    }
    else if(mode == miopenBNSpatial)
    {
        return {
            DefaultKernelArg("iNHW", OpAttr, OpKernelArg(static_cast<float>(0)), 0),
            DefaultKernelArg("activAlpha", OpArg, OpKernelArg(static_cast<float>(0)), 1),
            DefaultKernelArg("activBeta", OpArg, OpKernelArg(static_cast<float>(0)), 1),
            DefaultKernelArg("activGamma", OpArg, OpKernelArg(static_cast<float>(0)), 1),
            DefaultKernelArg("epsilon", OpArg, OpKernelArg(static_cast<double>(0.0)), 0),
            DefaultKernelArg("expAvgFactor", OpArg, OpKernelArg(static_cast<double>(0.0)), 0),
            DefaultKernelArg("input", InputTensor, OpKernelArg(nullptr)),
            DefaultKernelArg("output", OutputTensor, OpKernelArg(nullptr)),
            DefaultKernelArg("bnBias", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("bnScale", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("runningMean", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("runningVariance", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("savedInvVariance", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("savedMean", OpArg, OpKernelArg(nullptr), 0),
        };
    }
    else
    {
        MIOPEN_THROW("Unknown batch norm mode");
    }
}

void FusionMDGraph::InitBNFwd(FusionMDGraph& g)
{
    FusionMDGraph_Edge_Map empty_map = FusionMDGraph::EmptyEdgeMap();
    // Batch Norm + Activation Fwd Training
    {
        auto bn_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormFwdTrain,
                                                     "MIOpenBatchNormActivFwdTrainPerAct.cl",
                                                     "MIOpenBatchNormActivFwdTrainPerActivation",
                                                     "MIOpenBatchNormActivFwdTrainPerActivation");
        bn_v->default_args = BNFwdArgs(miopenBNPerActivation);
        FusionMDGraph_Edge_Map edg_activ =
            BatchNormFwdTrainFusionOpDescriptor::MDGraphKey(miopenBNPerActivation);
        edg_activ.insert(empty_map.begin(), empty_map.end()); // add the weight etc.

        g.AddEdge(nullptr, bn_v, edg_activ);
        auto activ_v =
            std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
                                             "MIOpenBatchNormActivFwdTrainPerAct.cl",
                                             "MIOpenBatchNormActivFwdTrainPerActivation",
                                             "MIOpenBatchNormActivFwdTrainPerActivation");
        activ_v->default_args = BNFwdArgs(miopenBNPerActivation);
        g.AddEdge(bn_v, activ_v, empty_map);
    }

    {
        auto bn_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormFwdTrain,
                                                     "MIOpenBatchNormActivFwdTrainSpatial.cl",
                                                     "MIOpenBatchNormActivFwdTrainSpatial",
                                                     "MIOpenBatchNormActivFwdTrainSpatial");
        bn_v->default_args = BNFwdArgs(miopenBNSpatial);
        FusionMDGraph_Edge_Map edg_spatial =
            BatchNormFwdTrainFusionOpDescriptor::MDGraphKey(miopenBNSpatial);
        edg_spatial.insert(empty_map.begin(), empty_map.end());
        g.AddEdge(nullptr, bn_v, edg_spatial);
        auto activ_v = std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
                                                        "MIOpenBatchNormActivFwdTrainSpatial.cl",
                                                        "MIOpenBatchNormActivFwdTrainSpatial",
                                                        "MIOpenBatchNormActivFwdTrainSpatial");
        activ_v->default_args = BNFwdArgs(miopenBNSpatial);
        g.AddEdge(bn_v, activ_v, empty_map);
    }
}

static std::vector<DefaultKernelArg> BNBwdArgs(miopenBatchNormMode_t mode)
{
    float f_zero = 0;
    if(mode == miopenBNPerActivation)
    {
        return {
            DefaultKernelArg("x", OpArg, OpKernelArg(nullptr)),
            DefaultKernelArg("y", OpArg, OpKernelArg(nullptr), 1), // probably from activation bwd
            DefaultKernelArg("input", InputTensor, OpKernelArg(nullptr)),
            DefaultKernelArg("output", OutputTensor, OpKernelArg(nullptr)),
            DefaultKernelArg("activDiffScale", OpArg, OpKernelArg(f_zero), 1),
            DefaultKernelArg("activGamma", OpArg, OpKernelArg(f_zero), 1),
            DefaultKernelArg("activBeta", OpArg, OpKernelArg(f_zero), 1),
            DefaultKernelArg("activAlpha", OpArg, OpKernelArg(f_zero), 1),
            DefaultKernelArg("bnScale", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("bnBias", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("resBnScaleDiff", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("resBnBiasDiff", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("savedMean", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("savedInvVariance", OpArg, OpKernelArg(nullptr), 0),
        };
    }
    else if(mode == miopenBNSpatial)
    {
        return {
            DefaultKernelArg("x", OpArg, OpKernelArg(nullptr)),
            DefaultKernelArg("y", OpArg, OpKernelArg(nullptr), 1), // probably from activation bwd
            DefaultKernelArg("input", InputTensor, OpKernelArg(nullptr)),
            DefaultKernelArg("output", OutputTensor, OpKernelArg(nullptr)),
            DefaultKernelArg("activDiffScale", OpArg, OpKernelArg(f_zero), 1),
            DefaultKernelArg("activGamma", OpArg, OpKernelArg(f_zero), 1),
            DefaultKernelArg("activBeta", OpArg, OpKernelArg(f_zero), 1),
            DefaultKernelArg("activAlpha", OpArg, OpKernelArg(f_zero), 1),
            DefaultKernelArg("bnScale", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("bnBias", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("resBnScaleDiff", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("resBnBiasDiff", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("savedMean", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("savedInvVariance", OpArg, OpKernelArg(nullptr), 0),
            DefaultKernelArg("iNHW", OpAttr, OpKernelArg(f_zero), 0),
        };
    }
    else
    {
        MIOPEN_THROW("Unknown batch norm mode");
    }
}

void FusionMDGraph::InitBNBwd(FusionMDGraph& g)
{
    FusionMDGraph_Edge_Map empty_map = FusionMDGraph::EmptyEdgeMap();
    // Batch Norm + Activation Backwards Training
    {
        auto bn_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormBwdTrain,
                                                     "MIOpenBatchNormActivBwdPerAct.cl",
                                                     "MIOpenBatchNormActivBwdPerActivation",
                                                     "MIOpenBatchNormActivBwdPerActivation");
        bn_v->default_args = BNBwdArgs(miopenBNPerActivation);
        FusionMDGraph_Edge_Map edg_activ =
            BatchNormBwdTrainFusionOpDescriptor::MDGraphKey(miopenBNPerActivation);
        edg_activ.insert(empty_map.begin(), empty_map.end()); // add the weight etc.

        g.AddEdge(nullptr, bn_v, edg_activ);
        auto activ_v = std::make_shared<MDGraph_vertex>(miopenFusionOpActivBackward,
                                                        "MIOpenBatchNormActivBwdPerAct.cl",
                                                        "MIOpenBatchNormActivBwdPerActivation",
                                                        "MIOpenBatchNormActivBwdPerActivation");
        activ_v->default_args = BNBwdArgs(miopenBNPerActivation);
        g.AddEdge(bn_v, activ_v, empty_map);
    }

    {
        auto bn_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormBwdTrain,
                                                     "MIOpenBatchNormActivBwdSpatial.cl",
                                                     "MIOpenBatchNormActivBwdSpatial",
                                                     "MIOpenBatchNormActivBwdSpatial");
        bn_v->default_args = BNBwdArgs(miopenBNSpatial);
        FusionMDGraph_Edge_Map edg_spatial =
            BatchNormBwdTrainFusionOpDescriptor::MDGraphKey(miopenBNSpatial);
        edg_spatial.insert(empty_map.begin(), empty_map.end());
        g.AddEdge(nullptr, bn_v, edg_spatial);
        auto activ_v = std::make_shared<MDGraph_vertex>(miopenFusionOpActivBackward,
                                                        "MIOpenBatchNormActivBwdSpatial.cl",
                                                        "MIOpenBatchNormActivBwdSpatial",
                                                        "MIOpenBatchNormActivBwdSpatial");
        activ_v->default_args = BNBwdArgs(miopenBNSpatial);
        g.AddEdge(bn_v, activ_v, empty_map);
    }
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

static std::vector<DefaultKernelArg> WinogradNodeArgs()
{
    auto zero_int = OpKernelArg(static_cast<int>(0));
    return {
        DefaultKernelArg("iN", InputTensorDesc, zero_int),
        DefaultKernelArg("iC", InputTensorDesc, zero_int),
        DefaultKernelArg("iH", InputTensorDesc, zero_int),
        DefaultKernelArg("iW", InputTensorDesc, zero_int),
        DefaultKernelArg("oK", OutputTensorDesc, zero_int),
        DefaultKernelArg("devCUs", DevAttribute, zero_int),
        DefaultKernelArg("flags", Other, zero_int),
        DefaultKernelArg("reserved", Other, zero_int),
        DefaultKernelArg("input", InputTensor, OpKernelArg(nullptr)),
        DefaultKernelArg("weights", OpArg, OpKernelArg(nullptr), 0),
        DefaultKernelArg("output", OutputTensor, OpKernelArg(nullptr)),
        DefaultKernelArg("return_addr", Other, OpKernelArg(nullptr)),
        DefaultKernelArg("x", OpAttr, zero_int, 0),
        DefaultKernelArg("y", OpAttr, zero_int, 0),
        DefaultKernelArg("pad_h", OpAttr, zero_int, 0),
        DefaultKernelArg("pad_w", OpAttr, zero_int, 0),
        DefaultKernelArg("oH", OutputTensorDesc, zero_int),
        DefaultKernelArg("oW", OutputTensorDesc, zero_int),
        DefaultKernelArg(
            "bias", Other, OpKernelArg(nullptr)), // only valid when bias is in the plan
        DefaultKernelArg(
            "activAlpha",
            Other,
            OpKernelArg(static_cast<float>(0.0))) // Op[2] only for leaky relu otherwise 0
    };
}

void FusionMDGraph::InitConv(FusionMDGraph& g)
{
    const auto common_constr = {
        EdgeOp(std::string("u == v"), true, OpEqual),
        EdgeOp(std::string("dilation_h == 1"), true, OpEqual),
        EdgeOp(std::string("dilation_w == 1"), true, OpEqual),

        EdgeOp(std::string("c * x * y <= (2^28)"), true, OpEqual),
        EdgeOp(std::string("k * x * y <= (2^28)"), true, OpEqual),
        EdgeOp(std::string("k * oH * oW <= (2^28)"), true, OpEqual),
        EdgeOp(std::string("c * iH * iW <= (2^28)"), true, OpEqual),
        EdgeOp(std::string("x <= (2^16)"), true, OpEqual),
        EdgeOp(std::string("y <= (2^16)"), true, OpEqual),
        EdgeOp(std::string("pad_h <= (2^16)"), true, OpEqual),
        EdgeOp(std::string("pad_w <= (2^16)"), true, OpEqual),
        EdgeOp(std::string("oH <= (2^16)"), true, OpEqual),
        EdgeOp(std::string("oW <= (2^16)"), true, OpEqual),
        EdgeOp(std::string("iH <= (2^16)"), true, OpEqual),
        EdgeOp(std::string("iW <= (2^16)"), true, OpEqual),
        EdgeOp(std::string("c <= (2^16)"), true, OpEqual),
        EdgeOp(std::string("k <= (2^16)"), true, OpEqual),
        EdgeOp(std::string("iN <= (2^16)"), true, OpEqual),
        EdgeOp(std::string("((padded_x / 3) * (padded_y / 3) * c ) >= 18"), true, OpEqual),
    };
    FusionMDGraph_Edge_Map empty_map = FusionMDGraph::EmptyEdgeMap();

    if(!miopen::IsDisabled(MIOPEN_DEBUG_AMD_FUSED_WINOGRAD{}))
    {
        /// Fused Winograd.
        static const std::string program("conv_3x3_wheel_alpha_v9_2_7_GFX*_md10.so");
        static const std::string kernel("sp3AsmConvRxSU_CBA");
        static const std::string algo("miopenConvolutionWinogradBiasActiv");
        auto vc_s1 =
            std::make_shared<MDGraph_vertex>(miopenFusionOpConvForward, program, kernel, algo);
        vc_s1->solver         = solver::ConvBinWinogradRxS{};
        vc_s1->default_args   = WinogradNodeArgs();
        vc_s1->supported_arch = {"gfx803", "gfx900", "gfx906"};

        FusionMDGraph_Edge_Map map_wino_conv_s1;
        map_wino_conv_s1["constraints"] = {
            EdgeOp(std::string("u == 1"), true, OpEqual),
            EdgeOp(std::string("y <= 3"), true, OpEval),
            EdgeOp(std::string("padded_y === 3"), true, OpEval),
            EdgeOp(std::string("padded_x === (x ~ 3)"), true, OpEval),
            EdgeOp(std::string("(c % 2) == 0"), true, OpAny),
        };
        map_emplace(map_wino_conv_s1, "weight", EdgeOp(5, true, OpAny));
        map_emplace(
            map_wino_conv_s1, "algo", EdgeOp(miopenConvolutionFwdAlgoWinograd, true, OpAny));
        map_emplace(map_wino_conv_s1, "precision", EdgeOp(miopenFloat, true, OpEqual));
        map_wino_conv_s1["constraints"].insert(
            map_wino_conv_s1["constraints"].end(), common_constr.begin(), common_constr.end());
        g.AddEdge(nullptr, vc_s1, map_wino_conv_s1);

        FusionMDGraph_Edge_Map map_wino_conv_s1_xgt3;
        map_wino_conv_s1_xgt3["constraints"] = {
            EdgeOp(std::string("u == 1"), true, OpEqual),
            EdgeOp(std::string("y > 3"), true, OpEval),
            EdgeOp(std::string("padded_y === (y ~ 6)"), true, OpEval),
            EdgeOp(std::string("padded_x === (x ~ 3)"), true, OpEval),
        };
        map_emplace(map_wino_conv_s1_xgt3, "weight", EdgeOp(5, true, OpAny));
        map_emplace(
            map_wino_conv_s1_xgt3, "algo", EdgeOp(miopenConvolutionFwdAlgoWinograd, true, OpAny));
        map_emplace(map_wino_conv_s1_xgt3, "precision", EdgeOp(miopenFloat, true, OpEqual));
        map_wino_conv_s1_xgt3["constraints"].insert(
            map_wino_conv_s1_xgt3["constraints"].end(), common_constr.begin(), common_constr.end());
        g.AddEdge(nullptr, vc_s1, map_wino_conv_s1_xgt3);

        // add 3x3 with higher priority since its the fastest case
        FusionMDGraph_Edge_Map map_wino_conv_xe3;
        map_wino_conv_xe3["constraints"] = {
            EdgeOp(std::string("u == 1"), true, OpEqual),
            EdgeOp(std::string("(y == 3) & (x == 3)"), true, OpEval),
            EdgeOp(std::string("padded_y === 3"), true, OpEval),
            EdgeOp(std::string("padded_x === (x ~ 3)"), true, OpEval),
            EdgeOp(std::string("(c % 2) == 0"), true, OpAny),
        };
        map_emplace(map_wino_conv_xe3, "weight", EdgeOp(100, true, OpAny));
        map_emplace(
            map_wino_conv_xe3, "algo", EdgeOp(miopenConvolutionFwdAlgoWinograd, true, OpAny));
        map_emplace(map_wino_conv_xe3, "precision", EdgeOp(miopenFloat, true, OpEqual));
        map_wino_conv_xe3["constraints"].insert(
            map_wino_conv_xe3["constraints"].end(), common_constr.begin(), common_constr.end());
        g.AddEdge(nullptr, vc_s1, map_wino_conv_xe3);

        /// C>B>A| (4)
        auto vb =
            std::make_shared<MDGraph_vertex>(miopenFusionOpBiasForward, program, kernel, algo);
        vb->default_args                = WinogradNodeArgs();
        vb->default_args[6].default_val = OpKernelArg(1 << 7);
        // set the bias parameters
        vb->default_args[18].type   = OpArg;
        vb->default_args[18].op_idx = 1;
        g.AddEdge(vc_s1, vb, empty_map);

        auto vba_leaf = std::make_shared<MDGraph_vertex>(
            miopenFusionOpActivForward, program, kernel, algo, true);
        vba_leaf->default_args                = WinogradNodeArgs();
        vba_leaf->default_args[6].default_val = OpKernelArg((1 << 7) + (1 << 8));
        // set the bias parameters
        vba_leaf->default_args[18].type   = OpArg;
        vba_leaf->default_args[18].op_idx = 1;

        vba_leaf->default_args[19].type   = OpArg;
        vba_leaf->default_args[19].op_idx = 2;

        FusionMDGraph_Edge_Map edg_activ_relu =
            ActivFwdFusionOpDescriptor::MDGraphKey(miopenActivationRELU);
        map_emplace(edg_activ_relu, "weight", EdgeOp(0, true, OpAny));
        map_emplace(edg_activ_relu, "precision", EdgeOp(miopenFloat, true, OpEqual));

        FusionMDGraph_Edge_Map edg_activ_leaky_relu =
            ActivFwdFusionOpDescriptor::MDGraphKey(miopenActivationLEAKYRELU);
        map_emplace(edg_activ_leaky_relu, "weight", EdgeOp(0, true, OpAny));
        map_emplace(edg_activ_leaky_relu, "precision", EdgeOp(miopenFloat, true, OpEqual));

        g.AddEdge(vb, vba_leaf, edg_activ_relu);
        g.AddEdge(vb, vba_leaf, edg_activ_leaky_relu);

        /// C>A| (5)
        auto va_leaf = std::make_shared<MDGraph_vertex>(
            miopenFusionOpActivForward, program, kernel, algo, true);
        va_leaf->default_args                = WinogradNodeArgs();
        va_leaf->default_args[6].default_val = OpKernelArg((1 << 8));
        va_leaf->default_args[19].type       = OpArg;
        va_leaf->default_args[19].op_idx     = 1;

        g.AddEdge(vc_s1, va_leaf, edg_activ_relu);
        g.AddEdge(vc_s1, va_leaf, edg_activ_leaky_relu);

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

        // Stride 2
        {

            auto add_meta_wino = [&](FusionMDGraph_Edge_Map& m, int weight) {
                map_emplace(m, "weight", EdgeOp(weight, true, OpEqual));
                map_emplace(m, "algo", EdgeOp(miopenConvolutionFwdAlgoWinograd, true, OpAny));
                map_emplace(m, "precision", EdgeOp(miopenFloat, true, OpEqual));
                m["constraints"].insert(
                    m["constraints"].end(), common_constr.begin(), common_constr.end());
            };

            static const std::string program_s2(
                "conv_3x3_wheel_alpha_v9_2_7_stride_2_dec_GFX*_md10.so");

            auto vc_s2 = std::make_shared<MDGraph_vertex>(
                miopenFusionOpConvForward, program_s2, kernel, algo);
            vc_s2->solver         = solver::ConvBinWinogradRxS{};
            vc_s2->default_args   = WinogradNodeArgs();
            vc_s2->supported_arch = {"gfx803", "gfx900", "gfx906"};

            FusionMDGraph_Edge_Map map_wino_conv_s2;
            map_wino_conv_s2["constraints"] = {
                EdgeOp(std::string("u == 2"), true, OpEqual),
                EdgeOp(std::string("padded_y === (y ~ 6)"), true, OpEqual),
                EdgeOp(std::string("(x % 6) == 1"), true, OpEqual),
                EdgeOp(std::string("padded_x === (x ~ 3)"), true, OpEqual),
            };

            add_meta_wino(map_wino_conv_s2, 5);
            g.AddEdge(nullptr, vc_s2, map_wino_conv_s2);

            FusionMDGraph_Edge_Map map_wino_conv_s2_modd;
            map_wino_conv_s2_modd["constraints"] = {
                EdgeOp(std::string("u == 2"), true, OpEqual),
                EdgeOp(std::string("padded_y === (y ~ 6)"), true, OpEqual),
                EdgeOp(std::string("(x % 6) != 1"), true, OpEqual),
                EdgeOp(std::string("padded_x === (x ~ 6)"), true, OpEqual),
            };

            add_meta_wino(map_wino_conv_s2_modd, 5);
            g.AddEdge(nullptr, vc_s2, map_wino_conv_s2_modd);

            // high priority edge for 3x3 kernels
            FusionMDGraph_Edge_Map map_wino_conv_s2_modd_xe3;
            map_wino_conv_s2_modd_xe3["constraints"] = {
                EdgeOp(std::string("u == 2"), true, OpEqual),
                EdgeOp(std::string("(x == 3) & (y == 3)"), true, OpEqual),
                EdgeOp(std::string("padded_y === (y ~ 6)"), true, OpEqual),
                EdgeOp(std::string("(x % 6) != 1"), true, OpEqual),
                EdgeOp(std::string("padded_x === (x ~ 6)"), true, OpEqual),
            };
            add_meta_wino(map_wino_conv_s2_modd_xe3, 100);
            g.AddEdge(nullptr, vc_s2, map_wino_conv_s2_modd_xe3);

            auto bias_s2 = std::make_shared<MDGraph_vertex>(
                miopenFusionOpBiasForward, program_s2, kernel, algo);
            bias_s2->default_args                = WinogradNodeArgs();
            bias_s2->default_args[6].default_val = OpKernelArg(1 << 7);
            // set the bias parameters
            bias_s2->default_args[18].type   = OpArg;
            bias_s2->default_args[18].op_idx = 1;
            g.AddEdge(vc_s2, bias_s2, empty_map);

            auto vba_leaf_s2 = std::make_shared<MDGraph_vertex>(
                miopenFusionOpActivForward, program_s2, kernel, algo, true);
            vba_leaf_s2->default_args                = WinogradNodeArgs();
            vba_leaf_s2->default_args[6].default_val = OpKernelArg((1 << 7) + (1 << 8));
            // set the bias parameters
            vba_leaf_s2->default_args[18].type   = OpArg;
            vba_leaf_s2->default_args[18].op_idx = 1;

            vba_leaf_s2->default_args[19].type   = OpArg;
            vba_leaf_s2->default_args[19].op_idx = 2;

            FusionMDGraph_Edge_Map edg_activ_relu_s2 =
                ActivFwdFusionOpDescriptor::MDGraphKey(miopenActivationRELU);
            map_emplace(edg_activ_relu_s2, "weight", EdgeOp(0, true, OpAny));
            map_emplace(edg_activ_relu_s2, "precision", EdgeOp(miopenFloat, true, OpEqual));

            FusionMDGraph_Edge_Map edg_activ_leaky_relu_s2 =
                ActivFwdFusionOpDescriptor::MDGraphKey(miopenActivationLEAKYRELU);
            map_emplace(edg_activ_leaky_relu_s2, "weight", EdgeOp(0, true, OpAny));
            map_emplace(edg_activ_leaky_relu_s2, "precision", EdgeOp(miopenFloat, true, OpEqual));

            g.AddEdge(bias_s2, vba_leaf_s2, edg_activ_relu_s2);
            g.AddEdge(bias_s2, vba_leaf_s2, edg_activ_leaky_relu_s2);

            /// C>A| (5)
            auto va_leaf_s2 = std::make_shared<MDGraph_vertex>(
                miopenFusionOpActivForward, program_s2, kernel, algo, true);
            va_leaf_s2->default_args                = WinogradNodeArgs();
            va_leaf_s2->default_args[6].default_val = OpKernelArg((1 << 8));
            va_leaf_s2->default_args[19].type       = OpArg;
            va_leaf_s2->default_args[19].op_idx     = 1;

            g.AddEdge(vc_s2, va_leaf_s2, edg_activ_relu_s2);
            g.AddEdge(vc_s2, va_leaf_s2, edg_activ_leaky_relu_s2);
        }
    }

    // first path (asm kernel)
    if(!miopen::IsDisabled(MIOPEN_DEBUG_GCN_ASM_KERNELS{}))
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
        map_emplace(map_asm_conv, "weight", EdgeOp(50, true, OpAny));
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
            if(len != 1)
            {
                map_conv_bias["pad_h"].clear();
                map_conv_bias["pad_h"].push_back(EdgeOp(2, true, OpLTE));
                map_conv_bias["pad_w"].clear();
                map_conv_bias["pad_w"].push_back(EdgeOp(2, true, OpLTE));

                map_conv_bias["u"].clear();
                map_conv_bias["u"].push_back(EdgeOp(1, true, OpGTE));
                map_conv_bias["u"].push_back(EdgeOp(2, true, OpLTE));
                map_conv_bias["v"].clear();
                map_conv_bias["v"].push_back(EdgeOp(1, true, OpGTE));
                map_conv_bias["v"].push_back(EdgeOp(2, true, OpLTE));
            }
            else
            {
                // 1x1 convolutions are only supported for single precision
                map_emplace(map_conv_bias, "precision", EdgeOp(miopenFloat, true, OpEqual));
            }
            map_emplace(map_conv_bias, "weight", EdgeOp(10, true, OpAny));
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
        }
    }

    // third path (ocl kernel no padding support for batch norm)
    {
        auto conv_v = std::make_shared<MDGraph_vertex>(miopenFusionOpConvForward,
                                                       "MIOpenConvDirBatchNormActiv.cl",
                                                       "MIOpenConvUniBatchNormActiv",
                                                       "miopenConvolutionDirectBiasActiv");

        conv_v->solver = solver::ConvOclDirectFwdFused{};

        // from ConvolutionDescriptor::IsDirectSupported
        std::vector<size_t> lens = {3, 5, 7, 9, 11};
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
            map_conv_bias["pad_h"].clear();
            map_conv_bias["pad_h"].push_back(EdgeOp(2, true, OpLTE));
            map_conv_bias["pad_w"].clear();
            map_conv_bias["pad_w"].push_back(EdgeOp(2, true, OpLTE));

            map_conv_bias["u"].clear();
            map_conv_bias["u"].push_back(EdgeOp(1, true, OpGTE));
            map_conv_bias["u"].push_back(EdgeOp(2, true, OpLTE));
            map_conv_bias["v"].clear();
            map_conv_bias["v"].push_back(EdgeOp(1, true, OpGTE));
            map_conv_bias["v"].push_back(EdgeOp(2, true, OpLTE));
            map_emplace(map_conv_bias, "weight", EdgeOp(10, true, OpAny));
            map_emplace(map_conv_bias, "algo", EdgeOp(miopenConvolutionFwdAlgoDirect, true, OpAny));

            g.AddEdge(nullptr, conv_v, map_conv_bias);
        }

        { // Conv -> Bias

            auto bias_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBiasForward,
                                                           "MIOpenConvDirBatchNormActiv.cl",
                                                           "MIOpenConvUniBatchNormActiv",
                                                           "miopenConvolutionDirectBiasActiv");
            g.AddEdge(conv_v, bias_v, empty_map);
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

bool FusionMDGraph::ExecOpLTE(const EdgeOp& edg_op, const EdgeOp& op_val)
{
    if(!(edg_op.val.type() == typeid(int) && op_val.val.type() == typeid(int)))
    {
        MIOPEN_LOG_I("Invalid operand types for Edge Op OpLTE (<=)");
        MIOPEN_THROW(miopenStatusBadParm);
    }
    return (boost::any_cast<int>(op_val.val) <= boost::any_cast<int>(edg_op.val));
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
    case OpLTE: { return FusionMDGraph::ExecOpLTE(edg_op, op_val);
    }
    case OpAdd:
    case OpSub:
    case OpMul:
    case OpDiv:
    case OpPow:
    case OpAnd:
    case OpOr:
    case OpCeil:
    case OpAssign:
    case OpGT:
    case OpLT:
    case OpEval: { assert(false);
    };
    }
    return false;
}

bool FusionMDGraph::CmpOpKey(const FusionMDGraph_Edge_Map& edge_val,
                             const std::shared_ptr<FusionOpDescriptor>& op,
                             std::function<bool(const std::string& sym, int& val)> attr_fun) const
{
    for(auto& kv : edge_val)
    {
        if(kv.first == "constraints")
        {
            tree_visit v(attr_fun);
            for(auto& edg_op : kv.second)
            {
                assert(edg_op.val.type() == typeid(std::string));
                using It = std::string::const_iterator;
                auto exp = boost::any_cast<std::string>(edg_op.val);
                It f(exp.begin()), l(exp.end());
                MDGExprParser p;
                boost::spirit::utree e;
                auto parse_success =
                    boost::spirit::qi::phrase_parse(f, l, p, boost::spirit::ascii::space, e);
                if(!parse_success)
                {
                    MIOPEN_LOG_I2("Remaining unparsed: " << std::string(exp.begin(), exp.end()));
                    MIOPEN_THROW(miopenStatusInternalError,
                                 "Unable to parse graph constraint expression");
                }
                visit_res r = boost::spirit::utree::visit(e, v);
                v.tabl.insert(r.tabl.begin(), r.tabl.end());
                if(r.b_res)
                {
                    MIOPEN_LOG_I2("Constraint satisfied: " + exp);
                }
                else
                {
                    MIOPEN_LOG_I("Condition unsuccessful while matching graph: " + exp);
                    return false;
                }
            }
        }
        else
        {
            auto op_val = op->MDGraphKey();
            if(op_val.count(kv.first) == 1)
            {
                auto edg_op_it =
                    std::find_if(kv.second.begin(), kv.second.end(), [&](auto&& edg_ops) {
                        return !FusionMDGraph::ExecEdgeOp(edg_ops, op_val.at(kv.first).at(0));
                    });
                if(edg_op_it == kv.second.end())
                {
                    MIOPEN_LOG_I2("Edge Op for key: " << kv.first << " Successfull");
                }
                else
                {
                    MIOPEN_LOG_I2("Edge Op :" << *edg_op_it << " Op Val: "
                                              << op_val.at(kv.first).at(0)
                                              << " Edge Op for key: "
                                              << kv.first
                                              << " Failed");
                    return false;
                }
            }
            else
            {
                MIOPEN_LOG_I2("Key: " << kv.first << " NOT found");
            }
        }
    }
    return true;
}

bool FusionMDGraph::Advance(std::shared_ptr<FusionOpDescriptor> op,
                            std::function<bool(const std::string& sym, int& val)> attr_fun)
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
            MIOPEN_LOG_I2("Current vertex: nullptr");
        }
        else
        {
            MIOPEN_LOG_I2("Current vertex: " << *cur_vertex_ptr);
        }
        // get the children of the cur_vertex
        auto& ch = edge_list[cur_vertex_ptr];
        // if op is in the children and the edge key satisfies update cur_vertex
        for(auto& ch_it : ch)
        {
            auto cur_map = kinder.second;
            MIOPEN_LOG_I2("Current path weight: " << boost::any_cast<int>(cur_map["weight"]));
            MIOPEN_LOG_I2("Child: " << *ch_it.first);
            std::set<miopenConvFwdAlgorithm_t> cur_path_set;
            if(ch_it.first->op == op->kind())
            {
                for(auto& edg_map : ch_it.second)
                {
                    int weight = boost::any_cast<int>(cur_map["weight"]);
                    if(CmpOpKey(edg_map, op, attr_fun))
                    {
                        MIOPEN_LOG_I2("Key Match Successfull");
                        weight += boost::any_cast<int>(edg_map.at("weight").at(0).val);
                        cur_map["weight"] = weight;

                        // Update the algo set
                        if(op->kind() == miopenFusionOpConvForward)
                        {
                            miopenConvFwdAlgorithm_t algo =
                                boost::any_cast<miopenConvFwdAlgorithm_t>(
                                    edg_map.at("algo").at(0).val);
                            MIOPEN_LOG_I2("Operator Matched: Convolution: Algo: " +
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
                            MIOPEN_LOG_I2("Operator Matched: " + std::to_string(op->kind()));
                            cur_map.erase("algo");
                        }
                        new_list.emplace_back(ch_it.first, cur_map);
                    }
                    else
                    {
                        MIOPEN_LOG_I2("Key Map Match failed");
                    }
                }
            }
            MIOPEN_LOG_I2("Current path final weight: " << boost::any_cast<int>(cur_map["weight"]));
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

// guard for debug only
#define MIOPEN_ENUM_STR(x) std::pair<decltype(x), std::string>(x, #x)
#define MIOPEN_ENUM_ARR(...) make_array(MIOPEN_PP_TRANSFORM_ARGS(MIOPEN_ENUM_STR, __VA_ARGS__))

template <class U>
std::unordered_map<int, std::string> enum_map(U lst)
{
    std::unordered_map<int, std::string> m;
    for(auto& kinder : lst)
    {
        m.emplace(kinder.first, kinder.second);
    }
    return m;
}

std::string edge_op_str(const MDGraph_op_t o)
{
    switch(o)
    {
    case OpEqual: return " == ";
    case OpNotEqual: return " != ";
    case OpAny: return " : ";
    case OpModulo: return " % ";
    case OpGTE: return " >= ";
    case OpLTE: return " <= ";
    case OpEval: return " eval ";
    case OpAdd: return " + ";
    case OpSub: return " - ";
    case OpMul: return " * ";
    case OpDiv: return " / ";
    case OpPow: return " ^ ";
    case OpAnd: return " && ";
    case OpOr: return " || ";
    case OpCeil: return " ceil ";
    case OpAssign: return " = ";
    case OpGT: return " > ";
    case OpLT: return " < ";
    }
    MIOPEN_THROW("Invalid Operation");
}

std::string any_string(const boost::any& a)
{
    if(a.type() == typeid(std::string))
        return boost::any_cast<std::string>(a);
    else if(a.type() == typeid(int))
        return std::to_string(boost::any_cast<int>(a));
    else if(a.type() == typeid(miopenConvolutionMode_t))
        return std::to_string(boost::any_cast<miopenConvolutionMode_t>(a));
    else if(a.type() == typeid(miopenPaddingMode_t))
        return std::to_string(boost::any_cast<miopenPaddingMode_t>(a));
    else if(a.type() == typeid(size_t))
        return std::to_string(boost::any_cast<size_t>(a));
    else if(a.type() == typeid(miopenBatchNormMode_t))
        return std::to_string(boost::any_cast<miopenBatchNormMode_t>(a));
    else if(a.type() == typeid(miopenActivationMode_t))
        return std::to_string(boost::any_cast<miopenActivationMode_t>(a));
    else if(a.type() == typeid(miopenDataType_t))
        return std::to_string(boost::any_cast<miopenDataType_t>(a));
    else
        return ""; // assert(false);
}

void FusionMDGraph::WriteToFile(std::string filename)
{
    const auto op_enum = enum_map(MIOPEN_ENUM_ARR(miopenFusionOpConvForward,
                                                  miopenFusionOpActivForward,
                                                  miopenFusionOpBatchNormInference,
                                                  miopenFusionOpBiasForward));

    if(filename.empty())
    {
        filename = "/tmp/mdgraph.dot";
    }
    std::set<MDGraph_vertex_ptr> nodes;
    std::ofstream dot_file;
    std::stringstream dot_graph;
    dot_file.open(filename);

    for(auto& edge : edge_list)
    {
        nodes.insert(edge.first);
        for(auto& edge2 : edge.second)
        {
            nodes.insert(edge2.first);
        }
    }

    dot_graph << "digraph { " << std::endl;
    for(auto& node : nodes)
    {
        if(node == nullptr)
        {
            dot_graph << "0 [label=root];" << std::endl;
        }
        else
        {
            dot_graph << node->id << " [ label=\"" << op_enum.at(node->op) << ":"
                      << node->vertex_data.at("kernel") << ":" << node->id << "\"];" << std::endl;
        }
    }

    int src_id, dst_id;

    for(auto& edge : edge_list)
    {
        if(edge.first != nullptr)
            src_id = edge.first->id;
        else
            src_id = 0;
        for(auto& edge2 : edge.second)
        {
            if(edge2.first != nullptr)
                dst_id = edge2.first->id;
            else
                dst_id = 0;
            for(auto& edg_map : edge2.second)
            {
                std::stringstream edge_label;
                for(auto& edg_ops : edg_map)
                {
                    for(auto& e : edg_ops.second)
                    {
                        if(e.op != OpAny) // skip metadata and dont cares
                            edge_label << edg_ops.first << edge_op_str(e.op) << any_string(e.val)
                                       << "\\n";
                    }
                }
                dot_graph << src_id << "->" << dst_id << "[label=\"" << edge_label.str() << "\"];"
                          << std::endl;
            }
        }
    }

    dot_graph << "}" << std::endl;
    dot_file << dot_graph.str();
}

} // namespace miopen
