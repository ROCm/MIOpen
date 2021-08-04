/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#if MIOPEN_ENABLE_SQLITE
#include <miopen/sqlite_db.hpp>
#endif
#include <miopen/db.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_FUSED_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_KERNELS)

namespace miopen {

int MDGraph_vertex::running_id = 1; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)

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

MDGraph_vertex_ptr FusionMDGraph::GetCurVertex(const Handle& handle)
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

std::string FusionMDGraph::GetProgramName(const Handle& handle)
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

std::string FusionMDGraph::GetKernelName(const Handle& handle)
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

std::string FusionMDGraph::GetAlgoName(const Handle& handle)
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

std::vector<DefaultKernelArg> FusionMDGraph::GetKernelArgs(const Handle& handle)
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

std::vector<miopenConvFwdAlgorithm_t> FusionMDGraph::GetConvAlgos() const
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
    FusionMDGraph_Edge_Map empty_map;
    empty_map["constraints"] = {"weight === 0"};
    // Batch Norm + Activation Fwd Training
    {
        auto bn_v          = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormFwdTrain,
                                                     "MIOpenBatchNormActivFwdTrainPerAct.cl",
                                                     "MIOpenBatchNormActivFwdTrainPerActivation",
                                                     "MIOpenBatchNormActivFwdTrainPerActivation");
        bn_v->default_args = BNFwdArgs(miopenBNPerActivation);
        FusionMDGraph_Edge_Map edg_activ;
        edg_activ["constraints"] = {"weight === 0", "bn_mode == miopenBNPerActivation"};

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
        auto bn_v          = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormFwdTrain,
                                                     "MIOpenBatchNormActivFwdTrainSpatial.cl",
                                                     "MIOpenBatchNormActivFwdTrainSpatial",
                                                     "MIOpenBatchNormActivFwdTrainSpatial");
        bn_v->default_args = BNFwdArgs(miopenBNSpatial);
        FusionMDGraph_Edge_Map edg_spatial;
        edg_spatial["constraints"] = {"weight === 0", "bn_mode == miopenBNSpatial"};
        g.AddEdge(nullptr, bn_v, edg_spatial);
        auto activ_v          = std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
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
    FusionMDGraph_Edge_Map empty_map;
    empty_map["constraints"] = {"weight === 0"};
    // Batch Norm + Activation Backwards Training
    {
        auto bn_v          = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormBwdTrain,
                                                     "MIOpenBatchNormActivBwdPerAct.cl",
                                                     "MIOpenBatchNormActivBwdPerActivation",
                                                     "MIOpenBatchNormActivBwdPerActivation");
        bn_v->default_args = BNBwdArgs(miopenBNPerActivation);
        FusionMDGraph_Edge_Map edg_activ;
        edg_activ["constraints"] = {"weight === 0", "bn_mode == miopenBNPerActivation"};

        g.AddEdge(nullptr, bn_v, edg_activ);
        auto activ_v          = std::make_shared<MDGraph_vertex>(miopenFusionOpActivBackward,
                                                        "MIOpenBatchNormActivBwdPerAct.cl",
                                                        "MIOpenBatchNormActivBwdPerActivation",
                                                        "MIOpenBatchNormActivBwdPerActivation");
        activ_v->default_args = BNBwdArgs(miopenBNPerActivation);
        g.AddEdge(bn_v, activ_v, empty_map);
    }

    {
        auto bn_v          = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormBwdTrain,
                                                     "MIOpenBatchNormActivBwdSpatial.cl",
                                                     "MIOpenBatchNormActivBwdSpatial",
                                                     "MIOpenBatchNormActivBwdSpatial");
        bn_v->default_args = BNBwdArgs(miopenBNSpatial);
        FusionMDGraph_Edge_Map edg_spatial;
        edg_spatial["constraints"] = {"weight === 0", "bn_mode == miopenBNSpatial"};
        g.AddEdge(nullptr, bn_v, edg_spatial);
        auto activ_v          = std::make_shared<MDGraph_vertex>(miopenFusionOpActivBackward,
                                                        "MIOpenBatchNormActivBwdSpatial.cl",
                                                        "MIOpenBatchNormActivBwdSpatial",
                                                        "MIOpenBatchNormActivBwdSpatial");
        activ_v->default_args = BNBwdArgs(miopenBNSpatial);
        g.AddEdge(bn_v, activ_v, empty_map);
    }
}

void FusionMDGraph::InitBN(FusionMDGraph& g)
{
    FusionMDGraph_Edge_Map empty_map;
    empty_map["constraints"] = {"weight === 0"};

    {
        auto bn_v = std::make_shared<MDGraph_vertex>(miopenFusionOpBatchNormInference,
                                                     "MIOpenBatchNormActivInfer.cl",
                                                     "MIOpenBatchNormActivInferPerActEst",
                                                     "MIOpenBatchNormActivInferPerActEst");
        FusionMDGraph_Edge_Map edg_activ;
        edg_activ["constraints"] = {"bn_mode == miopenBNPerActivation", "weight === 0"};

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
        FusionMDGraph_Edge_Map edg_spatial;
        edg_spatial["constraints"] = {"bn_mode == miopenBNSpatial", "weight === 0"};
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

static std::vector<DefaultKernelArg> WinogradV21NodeArgs()
{
    auto zero_int                             = OpKernelArg(static_cast<int>(0));
    auto zero_uint64                          = OpKernelArg(static_cast<uint64_t>(0));
    auto nodeArgs                             = WinogradNodeArgs();
    std::vector<DefaultKernelArg> v21NodeArgs = {
        DefaultKernelArg("reserved2", Other, zero_int),
        DefaultKernelArg("d_offset", Other, zero_uint64),
        DefaultKernelArg("f_offset", Other, zero_uint64),
        DefaultKernelArg("o_offset", Other, zero_uint64),
        DefaultKernelArg("b_offset", Other, zero_uint64),
        DefaultKernelArg("d_byte_stride_nk", InputTensorDesc, zero_int),
        DefaultKernelArg("d_byte_stride_c", InputTensorDesc, zero_int),
        DefaultKernelArg("d_byte_stride_h", InputTensorDesc, zero_int),
        DefaultKernelArg("d_byte_stride_w", InputTensorDesc, zero_int),
        DefaultKernelArg("f_byte_stride_nk", OpAttr, zero_int),
        DefaultKernelArg("f_byte_stride_c", OpAttr, zero_int),
        DefaultKernelArg("f_byte_stride_h", OpAttr, zero_int),
        DefaultKernelArg("f_byte_stride_w", OpAttr, zero_int),
        DefaultKernelArg("o_byte_stride_nk", OutputTensorDesc, zero_int),
        DefaultKernelArg("o_byte_stride_c", OutputTensorDesc, zero_int),
        DefaultKernelArg("o_byte_stride_h", OutputTensorDesc, zero_int),
        DefaultKernelArg("o_byte_stride_w", OutputTensorDesc, zero_int),
        DefaultKernelArg("group_count", OpAttr, zero_int),
        DefaultKernelArg("d_byte_stride_g", Other, zero_int),
        DefaultKernelArg("f_byte_stride_g", Other, zero_int),
        DefaultKernelArg("o_byte_stride_g", Other, zero_int),
    };
    nodeArgs.insert(nodeArgs.end(), v21NodeArgs.begin(), v21NodeArgs.end());
    return nodeArgs;
}

void FusionMDGraph::InitConv(FusionMDGraph& g)
{
    FusionMDGraph_Edge_Map empty_map;
    empty_map["constraints"] = {"weight === 0"};

    if(!miopen::IsDisabled(MIOPEN_DEBUG_AMD_FUSED_WINOGRAD{}) &&
       !miopen::IsDisabled(MIOPEN_DEBUG_GCN_ASM_KERNELS{}))
    {
        static const std::string algo("miopenConvolutionWinogradBiasActiv");
        // clang-format off
        const auto common_constr = {
            "algo === miopenConvolutionFwdAlgoWinograd",
            "precision == miopenFloat", "stride_h == stride_w",
            "dilation_h == 1",          "dilation_w == 1",
            "c * x * y <= (2^28)",      "k * x * y <= (2^28)",
            "k * oH * oW <= (2^28)",    "c * iH * iW <= (2^28)",
            "x <= (2^16)",              "y <= (2^16)",
            "pad_h <= (2^16)",          "pad_w <= (2^16)",
            "oH <= (2^16)",             "oW <= (2^16)",
            "iH <= (2^16)",             "iW <= (2^16)",
            "c <= (2^16)",              "k <= (2^16)",
            "iN <= (2^16)",             "group_count == 1",
        };
        // clang-format on

        auto add_relu = [&](const std::string& program,
                            const std::string& kernel,
                            MDGraph_vertex_ptr vc,
                            std::function<std::vector<DefaultKernelArg>(void)> nodeArgs,
                            const std::vector<std::string> supported_arch) {
            /// C>B>A| (4)
            auto bias =
                std::make_shared<MDGraph_vertex>(miopenFusionOpBiasForward, program, kernel, algo);
            bias->default_args                = nodeArgs();
            bias->default_args[6].default_val = OpKernelArg(1 << 7);
            // set the bias parameters
            bias->default_args[18].type   = OpArg;
            bias->default_args[18].op_idx = 1;
            bias->supported_arch          = supported_arch;
            g.AddEdge(vc, bias, empty_map);

            auto vba_leaf = std::make_shared<MDGraph_vertex>(
                miopenFusionOpActivForward, program, kernel, algo, true);
            vba_leaf->default_args                = nodeArgs();
            vba_leaf->default_args[6].default_val = OpKernelArg((1 << 7) + (1 << 8));
            // set the bias parameters
            vba_leaf->default_args[18].type   = OpArg;
            vba_leaf->default_args[18].op_idx = 1;

            vba_leaf->default_args[19].type   = OpArg;
            vba_leaf->default_args[19].op_idx = 2;
            vba_leaf->supported_arch          = supported_arch;

            FusionMDGraph_Edge_Map edg_activ_relu;
            edg_activ_relu["constraints"] = {"activ_mode == miopenActivationRELU", "weight === 0"};
            g.AddEdge(bias, vba_leaf, edg_activ_relu);

            FusionMDGraph_Edge_Map edg_activ_leaky_relu;
            edg_activ_leaky_relu["constraints"] = {"activ_mode == miopenActivationLEAKYRELU",
                                                   "weight === 0"};

            g.AddEdge(bias, vba_leaf, edg_activ_leaky_relu);

            /// C>A| (5)
            auto va_leaf = std::make_shared<MDGraph_vertex>(
                miopenFusionOpActivForward, program, kernel, algo, true);
            va_leaf->default_args                = nodeArgs();
            va_leaf->default_args[6].default_val = OpKernelArg((1 << 8));
            va_leaf->default_args[19].type       = OpArg;
            va_leaf->default_args[19].op_idx     = 1;
            va_leaf->supported_arch              = supported_arch;

            g.AddEdge(vc, va_leaf, edg_activ_relu);
            g.AddEdge(vc, va_leaf, edg_activ_leaky_relu);

            /// \FIXME Bug: In spite of C>B| topology is disabled below, it is selected anyway for
            /// Winograd. Possible reason is presence of C>B>A| configuration, which is somehow
            /// matches
            /// C>B| fused configuration. Fortunately, it is supported.
            ///
            /// C>B| (6)
            /// \todo Shader supports this config, but it is not required for now.
            /// auto vb_leaf = std::make_shared<MDGraph_vertex>(miopenFusionOpBiasForward,  program,
            /// kernel, algo, true);
            /// g.AddEdge(vc, vb_leaf, edg_activ_relu);
            /// g.AddEdge(vc, vb_leaf, edg_activ_leaky_relu);
        };

        // Fused Winograd v9_2_7
        {
            auto add_meta_wino = [&](FusionMDGraph_Edge_Map& m, int weight) {
                m["constraints"].emplace_back("weight === " + std::to_string(weight));
                m["constraints"].emplace_back("((padded_x / 3) * (padded_y / 3) * c ) >= 18");
                m["constraints"].insert(
                    m["constraints"].end(), common_constr.begin(), common_constr.end());
            };

            static const std::string program("conv_3x3_wheel_alpha_v9_2_7.s");
            static const std::string kernel("miopenSp3AsmConvRxSU_CBA");
            static const std::vector<std::string> supported_arch = {"gfx803"};

            auto vc_s1 =
                std::make_shared<MDGraph_vertex>(miopenFusionOpConvForward, program, kernel, algo);
            vc_s1->solver         = solver::ConvBinWinogradRxSFused{};
            vc_s1->default_args   = WinogradNodeArgs();
            vc_s1->supported_arch = supported_arch;

            FusionMDGraph_Edge_Map map_wino_conv_s1;
            map_wino_conv_s1["constraints"] = {"stride_h == 1",
                                               "y <= 3",
                                               "padded_y === 3",
                                               "padded_x === (x ~ 3)",
                                               "(c % 2) == 0"};
            add_meta_wino(map_wino_conv_s1, 5);
            g.AddEdge(nullptr, vc_s1, map_wino_conv_s1);

            FusionMDGraph_Edge_Map map_wino_conv_s1_xgt3;
            map_wino_conv_s1_xgt3["constraints"] = {
                "stride_h == 1", "y > 3", "padded_y === (y ~ 6)", "padded_x === (x ~ 3)"};
            add_meta_wino(map_wino_conv_s1_xgt3, 5);
            g.AddEdge(nullptr, vc_s1, map_wino_conv_s1_xgt3);

            // add 3x3 with higher priority since its the fastest case
            FusionMDGraph_Edge_Map map_wino_conv_xe3;
            map_wino_conv_xe3["constraints"] = {"stride_h == 1",
                                                "(y == 3) & (x == 3)",
                                                "padded_y === 3",
                                                "padded_x === (x ~ 3)",
                                                "(c % 2) == 0"};
            add_meta_wino(map_wino_conv_xe3, 100);
            g.AddEdge(nullptr, vc_s1, map_wino_conv_xe3);

            /// C>B>A| (4)
            add_relu(program, kernel, vc_s1, WinogradNodeArgs, supported_arch);

            // Stride 2
            {
                static const std::string program_s2("conv_3x3_wheel_alpha_v9_2_7_stride_2_dec.s");

                auto vc_s2 = std::make_shared<MDGraph_vertex>(
                    miopenFusionOpConvForward, program_s2, kernel, algo);
                vc_s2->solver         = solver::ConvBinWinogradRxSFused{};
                vc_s2->default_args   = WinogradNodeArgs();
                vc_s2->supported_arch = supported_arch;

                FusionMDGraph_Edge_Map map_wino_conv_s2;
                map_wino_conv_s2["constraints"] = {"stride_h == 2",
                                                   "padded_y === (y ~ 6)",
                                                   "(x % 6) == 1",
                                                   "padded_x === (x ~ 3)"};
                add_meta_wino(map_wino_conv_s2, 5);
                g.AddEdge(nullptr, vc_s2, map_wino_conv_s2);

                FusionMDGraph_Edge_Map map_wino_conv_s2_modd;
                map_wino_conv_s2_modd["constraints"] = {"stride_h == 2",
                                                        "padded_y === (y ~ 6)",
                                                        "(x % 6) != 1",
                                                        "padded_x === (x ~ 6)"};
                add_meta_wino(map_wino_conv_s2_modd, 5);
                g.AddEdge(nullptr, vc_s2, map_wino_conv_s2_modd);

                // high priority edge for 3x3 kernels
                FusionMDGraph_Edge_Map map_wino_conv_s2_modd_xe3;
                map_wino_conv_s2_modd_xe3["constraints"] = {"stride_h == 2",
                                                            "(x == 3) & (y == 3)",
                                                            "padded_y === (y ~ 6)",
                                                            "(x % 6) != 1",
                                                            "padded_x === (x ~ 6)"};
                add_meta_wino(map_wino_conv_s2_modd_xe3, 100);
                g.AddEdge(nullptr, vc_s2, map_wino_conv_s2_modd_xe3);

                add_relu(program_s2, kernel, vc_s2, WinogradNodeArgs, supported_arch);
            }
        }

        // Fused Winograd v21_1_2
        {
            auto add_meta_wino = [&](FusionMDGraph_Edge_Map& m, int weight) {
                m["constraints"].emplace_back("weight === " + std::to_string(weight));
                m["constraints"].emplace_back("oH * oW <= (2^23)");
                m["constraints"].insert(
                    m["constraints"].end(), common_constr.begin(), common_constr.end());
            };

            auto add_v21_wino = [&](const std::string family,
                                    const std::vector<std::string> supported_arch,
                                    const int stride) {
                const auto kernel_postfix = "_fp32_stride" + std::to_string(stride);
                const auto kernel_file    = "Conv_Winograd_v21_1_2" + kernel_postfix + ".s";
                const auto kernel_name    = "miopenSp3AsmConv_v21_1_2_" + family + kernel_postfix;

                auto vc = std::make_shared<MDGraph_vertex>(
                    miopenFusionOpConvForward, kernel_file, kernel_name, algo);
                vc->solver         = solver::ConvBinWinogradRxSf2x3g1Fused{};
                vc->default_args   = WinogradV21NodeArgs();
                vc->supported_arch = supported_arch;

                const auto stride_constr = "stride_h == " + std::to_string(stride);

                FusionMDGraph_Edge_Map map_wino_conv;
                map_wino_conv["constraints"] = {stride_constr};
                add_meta_wino(map_wino_conv, 5);
                g.AddEdge(nullptr, vc, map_wino_conv);

                // add 3x3 with higher priority since its the fastest case
                FusionMDGraph_Edge_Map map_wino_conv_xe3;
                map_wino_conv_xe3["constraints"] = {stride_constr, "(y == 3) & (x == 3)"};
                add_meta_wino(map_wino_conv_xe3, 100);
                g.AddEdge(nullptr, vc, map_wino_conv_xe3);

                add_relu(kernel_file, kernel_name, vc, WinogradV21NodeArgs, supported_arch);
            };

            add_v21_wino("gfx9", {"gfx900", "gfx906", "gfx908"}, 1);
            add_v21_wino("gfx9", {"gfx900", "gfx906", "gfx908"}, 2);
            add_v21_wino("gfx10", {"gfx1011", "gfx1012", "gfx1030"}, 1);
            add_v21_wino("gfx10", {"gfx1011", "gfx1012", "gfx1030"}, 2);
        }
    }

    // first path (asm kernel)
    if(!miopen::IsDisabled(MIOPEN_DEBUG_GCN_ASM_KERNELS{}))
    { // Conv -> Bias -> Activ // Conv -> Activ
        // single precision
        {
            const std::vector<std::string> supported_arch = {
                "gfx803", "gfx900", "gfx906", "gfx908"};
            auto conv_v            = std::make_shared<MDGraph_vertex>(miopenFusionOpConvForward,
                                                           "conv1x1u_bias_activ.s",
                                                           "miopenGcnAsmConv1x1U",
                                                           "miopenConvolutionDirectBiasActivAsm");
            conv_v->solver         = solver::ConvBiasActivAsm1x1U{};
            conv_v->supported_arch = supported_arch;

            auto bias_v            = std::make_shared<MDGraph_vertex>(miopenFusionOpBiasForward,
                                                           "conv1x1u_bias_activ.s",
                                                           "miopenGcnAsmConv1x1U",
                                                           "miopenConvolutionDirectBiasActivAsm");
            bias_v->supported_arch = supported_arch;

            auto activ_v            = std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
                                                            "conv1x1u_bias_activ.s",
                                                            "miopenGcnAsmConv1x1U",
                                                            "miopenConvolutionDirectBiasActivAsm",
                                                            true);
            activ_v->supported_arch = supported_arch;
            FusionMDGraph_Edge_Map map_asm_conv;

            map_asm_conv["constraints"] = {"group_count == 1",
                                           "pad_h == 0",
                                           "pad_w == 0",
                                           "stride_h == 1",
                                           "stride_w == 1",
                                           "dilation_h == 1",
                                           "dilation_w == 1",
                                           "x == 1",
                                           "y == 1",
                                           "c < (2^16)",
                                           "k < (2^16)",
                                           "iN < (2^16)",
                                           "(c * iH * iW * 4) < (2^24)",
                                           "(k * oH * oW * 4) < (2^24)",
                                           "(iN * c * iH * iW) < (2^29)",
                                           "(iN * k * oH * oW) < (2^29)",
                                           "(c * k) < (2^29)",
                                           "precision == miopenFloat",
                                           "weight === 50",
                                           "algo === miopenConvolutionFwdAlgoDirect"};

            g.AddEdge(nullptr, conv_v, map_asm_conv);
            g.AddEdge(conv_v, bias_v, empty_map);
            g.AddEdge(bias_v, activ_v, empty_map);

            g.AddEdge(conv_v, activ_v, empty_map);
        }
        // half precision
        {
            const std::vector<std::string> supported_arch = {"gfx900", "gfx906", "gfx908"};
            auto conv_v            = std::make_shared<MDGraph_vertex>(miopenFusionOpConvForward,
                                                           "conv1x1u_bias_activ.s",
                                                           "miopenGcnAsmConv1x1U",
                                                           "miopenConvolutionDirectBiasActivAsm");
            conv_v->solver         = solver::ConvBiasActivAsm1x1U{};
            conv_v->supported_arch = supported_arch;

            auto bias_v            = std::make_shared<MDGraph_vertex>(miopenFusionOpBiasForward,
                                                           "conv1x1u_bias_activ.s",
                                                           "miopenGcnAsmConv1x1U",
                                                           "miopenConvolutionDirectBiasActivAsm");
            bias_v->supported_arch = supported_arch;

            auto activ_v            = std::make_shared<MDGraph_vertex>(miopenFusionOpActivForward,
                                                            "conv1x1u_bias_activ.s",
                                                            "miopenGcnAsmConv1x1U",
                                                            "miopenConvolutionDirectBiasActivAsm",
                                                            true);
            activ_v->supported_arch = supported_arch;
            FusionMDGraph_Edge_Map map_asm_conv;

            map_asm_conv["constraints"] = {
                "group_count == 1",
                "pad_h == 0",
                "pad_w == 0",
                "stride_h == 1",
                "stride_w == 1",
                "dilation_h == 1",
                "dilation_w == 1",
                "x == 1",
                "y == 1",
                "c < (2^16)",
                "k < (2^16)",
                "iN < (2^16)",
                "k >= 4",
                "(oH * oW) >= 2", // (4 / elements_in_dword); elements_in_dword = 2
                "(c * iH * iW * 4) < (2^24)",
                "(k * oH * oW * 4) < (2^24)",
                "(iN * c * iH * iW) < (2^29)",
                "(iN * k * oH * oW) < (2^29)",
                "(c * k) < (2^29)",
                "precision == miopenHalf",
                "(c % 2) == 0",
                "(k % 2) == 0",
                "weight === 50",
                "algo === miopenConvolutionFwdAlgoDirect"};

            g.AddEdge(nullptr, conv_v, map_asm_conv);
            g.AddEdge(conv_v, bias_v, empty_map);
            g.AddEdge(bias_v, activ_v, empty_map);

            g.AddEdge(conv_v, activ_v, empty_map);
        }
    }

    // second path (ocl kernel)
    {
        auto conv_v = std::make_shared<MDGraph_vertex>(miopenFusionOpConvForward,
                                                       "MIOpenConvDirBatchNormActiv.cl",
                                                       "MIOpenConvUniBatchNormActiv",
                                                       "miopenConvolutionDirectBiasActiv");

        conv_v->solver = solver::ConvOclDirectFwdFused{};

        std::vector<size_t> lens = {3, 5, 7, 9, 11};
        for(auto len : lens)
        {
            FusionMDGraph_Edge_Map map_conv_bias;
            map_conv_bias["constraints"] = {"group_count == 1",
                                            "dilation_h == 1",
                                            "dilation_w == 1",
                                            "x == " + std::to_string(len),
                                            "y == " + std::to_string(len),
                                            "precision == miopenFloat",
                                            "weight === 10",
                                            "algo === miopenConvolutionFwdAlgoDirect"};
            if(len != 1)
            {
                map_conv_bias["constraints"].emplace_back("pad_h <= 2");
                map_conv_bias["constraints"].emplace_back("pad_w <= 2");
                map_conv_bias["constraints"].emplace_back("((stride_h == 1) | (stride_h == 2))");
                map_conv_bias["constraints"].emplace_back("((stride_w == 1) | (stride_w == 2))");
            }
            else
            {
                // 1x1 convolutions are only supported for single precision
                map_conv_bias["constraints"].emplace_back("precision == miopenFloat");
                map_conv_bias["constraints"].emplace_back("pad_h == 0");
                map_conv_bias["constraints"].emplace_back("pad_w == 0");
                map_conv_bias["constraints"].emplace_back("stride_h == 1");
                map_conv_bias["constraints"].emplace_back("stride_w == 1");
            }

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
            FusionMDGraph_Edge_Map map_conv_bias;
            map_conv_bias["constraints"] = {"group_count == 1",
                                            "dilation_h == 1",
                                            "dilation_w == 1",
                                            "x == " + std::to_string(len),
                                            "y == " + std::to_string(len),
                                            "precision == miopenFloat",
                                            "weight === 10",
                                            "algo === miopenConvolutionFwdAlgoDirect",
                                            "pad_h <= 2",
                                            "pad_w <= 2",
                                            "((stride_h == 1) | (stride_h == 2))",
                                            "((stride_w == 1) | (stride_w == 2))"};

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
                FusionMDGraph_Edge_Map edg_activ;
                edg_activ["constraints"] = {"weight === 0", "bn_mode == miopenBNPerActivation"};
                g.AddEdge(bias_v, bn_v, edg_activ);

                FusionMDGraph_Edge_Map edg_spatial;
                edg_spatial["constraints"] = {
                    "weight === 0",
                    "bn_mode == miopenBNSpatial",
                };

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
            FusionMDGraph_Edge_Map edg_activ;
            edg_activ["constraints"] = {"bn_mode == miopenBNPerActivation", "weight === 0"};
            g.AddEdge(conv_v, bn_v, edg_activ);

            FusionMDGraph_Edge_Map edg_spatial;
            edg_activ["constraints"] = {"bn_mode == miopenBNSpatial", "weight === 0"};
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

bool FusionMDGraph::CmpOpKey(const FusionMDGraph_Edge_Map& edge_val,
                             std::function<bool(const std::string& sym, int& val)> attr_fun,
                             std::unordered_map<std::string, int>& syms) const
{
    for(auto& kv : edge_val)
    {
        if(kv.first == "constraints")
        {
            tree_visit v(attr_fun);
            for(auto& edg_op : kv.second)
            {
                using It = std::string::const_iterator;
                It f(edg_op.begin()), l(edg_op.end());
                MDGExprParser p;
                boost::spirit::utree e;
                auto parse_success =
                    boost::spirit::qi::phrase_parse(f, l, p, boost::spirit::ascii::space, e);
                if(!parse_success)
                {
                    MIOPEN_LOG_I2(
                        "Remaining unparsed: " << std::string(edg_op.begin(), edg_op.end()));
                    MIOPEN_THROW(miopenStatusInternalError,
                                 "Unable to parse graph constraint expression");
                }
                visit_res r = boost::spirit::utree::visit(e, v);
                v.tabl.insert(r.tabl.begin(), r.tabl.end());
                syms = v.tabl;
                if(r.b_res)
                {
                    MIOPEN_LOG_I2("Constraint satisfied: " + edg_op);
                }
                else
                {
                    MIOPEN_LOG_I("Condition unsuccessful while matching graph: " + edg_op);
                    return false;
                }
            }
        }
        else
        {
            assert(false);
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
                    std::unordered_map<std::string, int> syms;
                    if(CmpOpKey(edg_map, attr_fun, syms))
                    {
                        MIOPEN_LOG_I2("Key Match Successfull");
                        if(syms.count("weight") != 0)
                        {
                            weight += syms.at("weight");
                        }
                        else
                        {
                            MIOPEN_LOG_I2("Weight not found, assuming zero");
                        }
                        cur_map["weight"] = weight;

                        // Update the algo set
                        if(op->kind() == miopenFusionOpConvForward)
                        {
                            if(syms.count("algo") != 0)
                            {
                                auto algo = static_cast<miopenConvFwdAlgorithm_t>(syms.at("algo"));
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
                                MIOPEN_THROW(miopenStatusInternalError,
                                             "algo is not provided for "
                                             "a convolution oeprator in "
                                             "the metadata graph");
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
                        MIOPEN_LOG_I2("Key Map Match unsuccessful");
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
    // sort according to the edge weight
    std::sort(cur_vertex.begin(),
              cur_vertex.end(),
              [&](const std::pair<MDGraph_vertex_ptr, cur_vertex_map>& a,
                  const std::pair<MDGraph_vertex_ptr, cur_vertex_map>& b) {
                  return boost::any_cast<int>(a.second.at("weight")) >
                         boost::any_cast<int>(b.second.at("weight"));
              });

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
                        edge_label << e << "\\n";
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
