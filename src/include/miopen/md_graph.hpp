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

#ifndef MIOPEN_GUARD_MLOPEN_MD_GRAPH_HPP
#define MIOPEN_GUARD_MLOPEN_MD_GRAPH_HPP

#include <miopen/miopen.h>
#include <miopen/fusion_ops.hpp>
#include <miopen/fusion.hpp>
#include <miopen/any_solver.hpp>

#include <unordered_map>

namespace miopen {

enum ArgOwnerType
{
    Other,
    InputTensor,
    OutputTensor,
    DevAttribute,
    OpArg,
    OpAttr,
    InputTensorDesc,
    OutputTensorDesc
};

struct DefaultKernelArg
{
    DefaultKernelArg(std::string k, ArgOwnerType t, OpKernelArg v, int idx = 0)
        : type(t), default_val(v), op_idx(idx), key(k){};
    ArgOwnerType type;
    OpKernelArg default_val;
    int op_idx;
    std::string key;
};

struct MDGraph_vertex
{
    static int running_id; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)
    MDGraph_vertex(miopenFusionOp_t o,
                   std::string program_name = "",
                   std::string kernel_name  = "",
                   std::string algo_name    = "",
                   bool _is_leaf            = false);
    miopenFusionOp_t op;
    bool is_leaf = false;
    std::map<std::string, std::string> vertex_data;
    std::vector<std::string> supported_arch;
    boost::optional<bool> supported_xnack = boost::none;
    size_t map_hash                       = 0;
    int id;

    MDGraph_vertex(const MDGraph_vertex& other) = delete;
    std::string& operator[](const std::string& x) { return vertex_data[x]; }
    std::vector<DefaultKernelArg> default_args;

    solver::AnySolver solver;
    friend std::ostream& operator<<(std::ostream& stream, const MDGraph_vertex& v);
};

using MDGraph_vertex_ptr = std::shared_ptr<MDGraph_vertex>;
using cur_vertex_map     = std::unordered_map<std::string, boost::any>;

struct FusionMDGraph
{
    FusionMDGraph() { Reset(); }
    static void Init(FusionMDGraph& g, miopenFusionOp_t op);
    static void InitConv(FusionMDGraph& g);
    static void InitBN(FusionMDGraph& g);
    static void InitBNFwd(FusionMDGraph& g);
    static void InitBNBwd(FusionMDGraph& g);
    void Reset();
    bool Advance(std::shared_ptr<FusionOpDescriptor> op,
                 std::function<bool(const std::string& sym, int& val)> attr_fun);
    void AddEdge(MDGraph_vertex_ptr src, MDGraph_vertex_ptr dst, FusionMDGraph_Edge_Map& map);

    bool CmpOpKey(const FusionMDGraph_Edge_Map& edge_val,
                  std::function<bool(const std::string& sym, int& val)> attr_fun,
                  std::unordered_map<std::string, int>& syms) const;
    MDGraph_vertex_ptr GetCurVertex(const Handle& handle);
    std::string GetProgramName(const Handle& handle);
    std::string GetKernelName(const Handle& handle);
    std::string GetAlgoName(const Handle& handle);
    std::vector<DefaultKernelArg> GetKernelArgs(const Handle& handle);
    std::vector<miopenConvFwdAlgorithm_t> GetConvAlgos() const;
    bool SetConvAlgo(miopenConvFwdAlgorithm_t algo);
    std::vector<solver::AnySolver> GetSolvers();
    void WriteToFile(std::string filename = "");

    std::vector<std::pair<MDGraph_vertex_ptr, cur_vertex_map>> cur_vertex;
    std::set<miopenConvFwdAlgorithm_t> conv_algo_set;

    std::unordered_map<MDGraph_vertex_ptr,
                       std::unordered_map<MDGraph_vertex_ptr, FusionMDGraph_Edge_Map_Vec>>
        edge_list;
};

} // namespace miopen

#endif
