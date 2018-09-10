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

#pragma once
#include <miopen/miopen.h>
#include <miopen/fusion_ops.hpp>
#include <miopen/fusion.hpp>

#include <boost/functional/hash.hpp>
#include <unordered_map>

namespace miopen {

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
    std::string& operator[](std::string& x) { return vertex_data[x]; }

    solver::AnySolver solver;
    friend std::ostream& operator<<(std::ostream& stream, const MDGraph_vertex& v);
};

using MDGraph_vertex_ptr = std::shared_ptr<MDGraph_vertex>;

struct FusionMDGraph
{
    using cur_vertex_map = std::unordered_map<std::string, boost::any>;
    FusionMDGraph() { Reset(); }
    static void Init(FusionMDGraph& g, miopenFusionOp_t op);
    static void InitConv(FusionMDGraph& g);
    static void InitBN(FusionMDGraph& g);
    void Reset();
    bool Advance(std::shared_ptr<FusionOpDescriptor> op);
    void AddEdge(MDGraph_vertex_ptr src, MDGraph_vertex_ptr dst, FusionMDGraph_Edge_Map& map);

    bool CmpOpKey(const FusionMDGraph_Edge_Map& edge_val,
                  const FusionMDGraph_Edge_Map& op_val) const;
    MDGraph_vertex_ptr GetCurVertex();
    std::string GetProgramName();
    std::string GetKernelName();
    std::string GetAlgoName();
    std::vector<miopenConvFwdAlgorithm_t> GetConvAlgos();
    bool SetConvAlgo(miopenConvFwdAlgorithm_t algo);
    static FusionMDGraph_Edge_Map EmptyEdgeMap(int weight = 0, MDGraph_op_t op = OpAny);
    static bool ExecEdgeOp(const EdgeOp& edg_op, const EdgeOp& op_val);
    static bool ExecOpEqual(const EdgeOp& edg_op, const EdgeOp& op_val);
    static bool ExecOpModulo(const EdgeOp& edg_op, const EdgeOp& op_val);
    static bool ExecOpGTE(const EdgeOp& edg_op, const EdgeOp& op_val);
    std::vector<solver::AnySolver> GetSolvers();

    protected:
    std::vector<std::pair<MDGraph_vertex_ptr, cur_vertex_map>> cur_vertex;
    std::set<miopenConvFwdAlgorithm_t> conv_algo_set;

    std::unordered_map<MDGraph_vertex_ptr,
                       std::unordered_map<MDGraph_vertex_ptr, FusionMDGraph_Edge_Map_Vec>>
        edge_list;
};

} // namespace miopen
