/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <cstddef>
#include "miopen/solver.hpp"
#include "miopen/handle.hpp"
#include <miopen/generic_search.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include "implicitgemm_util.hpp"
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/tensor_ops.hpp>

namespace miopen {
namespace solver {

static inline std::vector<TunableImplicitGemmGTCDynamic_t>&
GetImplicitGemmWrwGTCDynamicXdlopsKernelList()
{
    // retrieve dynamic igemm wrw pass's possible kernel name
    // clang-format off
    static std::vector<TunableImplicitGemmGTCDynamic_t> kernel_param_list {
        { "wrw", "fp32",   4,   0, 256, 128,  16,  64,  32,   1,   1,   2,   2,   {1,   4,   4,   1},   {1,   4,   1,  64},   {1,   4,   2,   1},   {1,   4,   1,  64},   0},
		{ "wrw", "fp32",   4,   0, 256, 128,  16,  64,  32,   1,   1,   2,   2,   {1,   4,   4,   1},   {1,   4,   1,  64},   {1,   4,   2,   1},   {1,   4,   1,  64},   1},
		{ "wrw", "fp32",   4,   0, 256, 128,   8,  64,  32,   1,   1,   2,   2,   {1,   4,   2,   1},   {1,   2,   1, 128},   {1,   4,   1,   1},   {1,   2,   1, 128},   0},
		{ "wrw", "fp32",   4,   0, 256, 128,   8,  64,  32,   1,   1,   2,   2,   {1,   4,   2,   1},   {1,   2,   1, 128},   {1,   4,   1,   1},   {1,   2,   1, 128},   1},
		{ "wrw", "fp32",   1,   1, 256, 128,  16,  64,  32,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1, 256, 128,  16,  64,  32,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1, 256, 128,   8,  64,  32,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
		{ "wrw", "fp32",   1,   1, 256, 128,   8,  64,  32,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   1},
		{ "wrw", "fp32",   4,   0, 256,  64,  16,  64,  16,   1,   1,   2,   2,   {1,   4,   4,   1},   {1,   4,   1,  64},   {1,   4,   1,   1},   {1,   4,   1,  64},   0},
		{ "wrw", "fp32",   4,   0, 256,  64,  16,  64,  16,   1,   1,   2,   2,   {1,   4,   4,   1},   {1,   4,   1,  64},   {1,   4,   1,   1},   {1,   4,   1,  64},   1},
		{ "wrw", "fp32",   1,   1, 256,  64,  16,  64,  16,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1, 256,  64,  16,  64,  16,   1,   1,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1, 256,  64,   8,  64,  16,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
		{ "wrw", "fp32",   1,   1, 256,  64,   8,  64,  16,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   1},
		{ "wrw", "fp32",   1,   1, 256,  64,   4,  64,  16,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   4,   1,  64},   {1,   1,   1,   1},   {1,   4,   1,  64},   0},
		{ "wrw", "fp32",   1,   1, 256,  64,   4,  64,  16,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   4,   1,  64},   {1,   1,   1,   1},   {1,   4,   1,  64},   1},
		{ "wrw", "fp32",   1,   1, 256,  32,  16,  64,   4,   1,   2,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1, 256,  32,  16,  64,   4,   1,   2,   2,   2,   {1,   1,  16,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1, 256,  32,   8,  64,   4,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
		{ "wrw", "fp32",   1,   1, 256,  32,   8,  64,   4,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
		{ "wrw", "fp32",   4,   0, 128, 128,  16,  32,  32,   1,   1,   2,   2,   {1,   4,   2,   1},   {1,   4,   1,  64},   {1,   4,   2,   1},   {1,   4,   1,  64},   0},
		{ "wrw", "fp32",   4,   0, 128, 128,  16,  32,  32,   1,   1,   2,   2,   {1,   4,   2,   1},   {1,   4,   1,  64},   {1,   4,   2,   1},   {1,   4,   1,  64},   1},
		{ "wrw", "fp32",   1,   1, 128, 128,  16,  32,  32,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1, 128, 128,  16,  32,  32,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1, 128, 128,   8,  32,  32,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
		{ "wrw", "fp32",   1,   1, 128, 128,   8,  32,  32,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   1},
		{ "wrw", "fp32",   4,   0, 128,  64,  16,  32,   8,   1,   2,   2,   2,   {1,   4,   2,   1},   {1,   4,   1,  64},   {1,   4,   1,   1},   {1,   4,   1,  64},   0},
		{ "wrw", "fp32",   4,   0, 128,  64,  16,  32,   8,   1,   2,   2,   2,   {1,   4,   2,   1},   {1,   4,   1,  64},   {1,   4,   1,   1},   {1,   4,   1,  64},   1},
		{ "wrw", "fp32",   1,   1, 128,  64,  16,  32,   8,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1, 128,  64,  16,  32,   8,   1,   2,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1, 128,  64,   8,  32,   8,   1,   2,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
		{ "wrw", "fp32",   1,   1, 128,  64,   8,  32,   8,   1,   2,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   1},
		{ "wrw", "fp32",   1,   1, 128,  32,  16,  32,   8,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1, 128,  32,  16,  32,   8,   1,   1,   2,   2,   {1,   1,   8,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1, 128,  32,   8,  32,   8,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
		{ "wrw", "fp32",   1,   1, 128,  32,   8,  32,   8,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
		{ "wrw", "fp32",   1,   1,  64, 256,  16,  16,  64,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,  16,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1,  64, 256,  16,  16,  64,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,  16,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1,  64, 256,   8,  16,  64,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   8,   1},   {1,   8,   1,  32},   0},
		{ "wrw", "fp32",   1,   1,  64, 256,   8,  16,  64,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   8,   1},   {1,   8,   1,  32},   1},
		{ "wrw", "fp32",   1,   1,  64, 128,  16,   8,  32,   2,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1,  64, 128,  16,   8,  32,   2,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   8,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1,  64, 128,   8,   8,  32,   2,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   0},
		{ "wrw", "fp32",   1,   1,  64, 128,   8,   8,  32,   2,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   4,   1},   {1,   8,   1,  32},   1},
		{ "wrw", "fp32",   1,   1,  64,  64,  16,  16,  16,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1,  64,  64,  16,  16,  16,   1,   1,   2,   2,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   4,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1,  64,  64,   8,  16,  16,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   0},
		{ "wrw", "fp32",   1,   1,  64,  64,   8,  16,  16,   1,   1,   2,   2,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   2,   1},   {1,   8,   1,  32},   1},
		{ "wrw", "fp32",   1,   1,  64,  32,  16,  32,   8,   1,   2,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1,  64,  32,  16,  32,   8,   1,   2,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1,  64,  32,   8,  32,   8,   1,   2,   1,   1,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
		{ "wrw", "fp32",   1,   1,  64,  32,   8,  32,   8,   1,   2,   1,   1,   {1,   1,   2,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
		{ "wrw", "fp32",   1,   1,  64,  16,  16,  64,   4,   1,   1,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   1,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1,  64,  16,  16,  64,   4,   1,   1,   1,   1,   {1,   1,   4,   1},   {1,  16,   1,  16},   {1,   1,   1,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1,   4,  64,  16,   4,  64,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   4},   {1,   1,  16,   1},   {1,  16,   1,   4},   1},
		{ "wrw", "fp32",   1,   1,   4,  64,  16,   4,  64,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,  16,   1,   4},   {1,   1,  16,   1},   {1,  16,   1,   4},   0},
		{ "wrw", "fp32",   1,   1,  32,  32,  16,  16,  16,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   1},
		{ "wrw", "fp32",   1,   1,  32,  32,  16,  16,  16,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,  16,   1,  16},   {1,   1,   2,   1},   {1,  16,   1,  16},   0},
		{ "wrw", "fp32",   1,   1,  32,  32,   8,  16,  16,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   1},
		{ "wrw", "fp32",   1,   1,  32,  32,   8,  16,  16,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  32},   {1,   1,   1,   1},   {1,   8,   1,  32},   0},
		{ "wrw", "fp32",   1,   1,  16,  32,  16,   8,  32,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,  16,   1,   8},   {1,   1,   4,   1},   {1,  16,   1,   8},   0},
		{ "wrw", "fp32",   1,   1,  16,  32,  16,   8,  32,   1,   1,   1,   1,   {1,   1,   2,   1},   {1,  16,   1,   8},   {1,   1,   4,   1},   {1,  16,   1,   8},   1},
		{ "wrw", "fp32",   1,   1,  16,  32,   8,   8,  32,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  16},   {1,   1,   2,   1},   {1,   8,   1,  16},   0},
		{ "wrw", "fp32",   1,   1,  16,  32,   8,   8,  32,   1,   1,   1,   1,   {1,   1,   1,   1},   {1,   8,   1,  16},   {1,   1,   2,   1},   {1,   8,   1,  16},   1},
        // clang-format on
    };
    return kernel_param_list;
}

static inline int GetImplicitGemmWrwGTCDynamicXdlopsGemmkSplits(
    const conv::ProblemDescription& conv_problem, const int GemmKPerBlock, const int grid_size)
{
    int n                   = conv_problem.GetInBatchSize();
    int ho                  = conv_problem.GetInHeight();
    int wo                  = conv_problem.GetInWidth();
    int gemm_k_global_split = 0;

    int max_grid_size = 1200;

    int n_per_group;
    for(int i = 0; i < 8; i++)
    {
        if((grid_size << i) > max_grid_size)
        {
            break;
        }
        if(0 == n % (1 << i))
        {
            n_per_group = n >> i;
            if(0 == ((n_per_group * ho * wo) % GemmKPerBlock))
                gemm_k_global_split = i;
            else
                break;
        }
        else
            break;
    }

    return gemm_k_global_split;
}

// tuple<log2_gemm_k_global_split, grid_size>
static inline std::tuple<int, int> get_grid_size(const ConvolutionContext& ctx,
                                                 const TunableImplicitGemmGTCDynamic_t* tunable)
{
    int k = ctx.n_inputs;
    int c = ctx.n_outputs;
    int y = ctx.kernel_size_h;
    int x = ctx.kernel_size_w;

    int gemm_m_per_block         = tunable->gemm_m_per_block;
    int gemm_n_per_block         = tunable->gemm_n_per_block;
    int gemm_k_per_block         = tunable->gemm_k_per_block;
    int gemm_k_global_split      = tunable->gemm_k_global_split;
    int log2_gemm_k_global_split = 0;

    int gemm_m = k;
    int gemm_n = c * y * x;

    // assume that gemm m/n can be divided with no remainder by gemm m/n per block
    int grid_size = (gemm_m / gemm_m_per_block) * (gemm_n / gemm_n_per_block);

    if(gemm_k_global_split == 1)
        log2_gemm_k_global_split = GetImplicitGemmWrwGTCDynamicXdlopsGemmkSplits(
            ctx.conv_problem, gemm_k_per_block, grid_size);
    else
        log2_gemm_k_global_split = 0;

    int num_of_gemm = 1 << log2_gemm_k_global_split;
    grid_size *= num_of_gemm;
    return std::make_tuple(log2_gemm_k_global_split, grid_size);
}

static inline int find_tunable(const std::vector<TunableImplicitGemmGTCDynamic_t> tunables,
                               const int gemm_m_per_block,
                               const int gemm_n_per_block,
                               const int gemm_k_per_block,
                               const int gemm_k_global_split,
                               const int nxb,
                               const int nxe)
{
    int i;
    for(i = 0; i < tunables.size(); i++)
    {
        if((tunables[i].gemm_m_per_block == gemm_m_per_block) &&
           (tunables[i].gemm_n_per_block == gemm_n_per_block) &&
           (tunables[i].gemm_k_per_block == gemm_k_per_block) &&
           (tunables[i].gemm_k_global_split == gemm_k_global_split) && (tunables[i].nxb == nxb) &&
           (tunables[i].nxe == nxe))
        {
            break;
        }
    }
    return i;
}

static inline int if_gemm_k_global_split(const ConvolutionContext& ctx,
                                         const int gemm_m_per_block,
                                         const int gemm_n_per_block,
                                         const int gemm_k_per_block)
{
    int gemm_k_global_split = 0;
    int n                   = ctx.batch_sz;
    int k                   = ctx.n_inputs;
    int c                   = ctx.n_outputs;
    int ho                  = ctx.in_height;
    int wo                  = ctx.in_width;
    int y                   = ctx.kernel_size_h;
    int x                   = ctx.kernel_size_w;

    int gemm_m = k;
    int gemm_n = c * y * x;

    int max_grid_size = 1200;

    int grid_size;
    // assume that gemm m/n can be divided with no remainder by gemm m/n per block
    grid_size = (gemm_m / gemm_m_per_block) * (gemm_n / gemm_n_per_block);
    if((n % 2 == 0) && (grid_size < max_grid_size) && ((n >> 1) * ho * wo % gemm_k_per_block == 0))
    {
        gemm_k_global_split = 1;
    }
    else
    {
        gemm_k_global_split = 0;
    }
    return gemm_k_global_split;
}

static inline float CallImplicitGemmWrwDynamic(const miopen::Handle& handle,
                                               const conv::ProblemDescription& conv_problem,
                                               ConstData_t src,
                                               ConstData_t dst,
                                               Data_t wei,
                                               const std::vector<KernelInvoke>& kernels,
                                               const int log2_gemm_k_global_splits)
{
    float elapsed = 0.0f;

    auto kernel = kernels[0];
    // clang-format off
    int hi           = conv_problem.GetOutHeight();
    int wi           = conv_problem.GetOutWidth();
    int n            = conv_problem.GetInBatchSize();
    int k            = conv_problem.GetInChannels();
    int c            = conv_problem.GetOutChannels();
    int ho           = conv_problem.GetInHeight();
    int wo           = conv_problem.GetInWidth();
    int stride_h     = conv_problem.GetInHeight() > 1 ? conv_problem.GetKernelStrideH() : 1;
    int stride_w     = conv_problem.GetInWidth() > 1 ? conv_problem.GetKernelStrideW() : 1;
    int dilation_h   = conv_problem.GetWeightsHeight() > 1? conv_problem.GetDilationH() : 1;
    int dilation_w   = conv_problem.GetWeightsWidth() > 1? conv_problem.GetDilationW() : 1;
    int pad_h        = conv_problem.GetPadH();
    int pad_w        = conv_problem.GetPadW();
    int y            = conv_problem.GetWeightsHeight();
    int x            = conv_problem.GetWeightsWidth();
    
    //std::cout << "nchiwi: " << n << " " << c  << " " << hi << " " << wi << std::endl;
    //std::cout << "nkhowo: " << n << " " << k  << " " << ho << " " << wo << std::endl;
    //std::cout << "kcyx: " << k << " " << c  << " " << y << " " << x << std::endl;
    
    MIOPEN_LOG_I2(kernel.GetName() << " with groups for reduction: " << (1 << log2_gemm_k_global_splits));

    // clang-format on
    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(src);
    opArgs.emplace_back(wei);
    opArgs.emplace_back(dst);
    opArgs.emplace_back(hi);
    opArgs.emplace_back(wi);
    opArgs.emplace_back(n);
    opArgs.emplace_back(k);
    opArgs.emplace_back(c);
    opArgs.emplace_back(ho);
    opArgs.emplace_back(wo);
    opArgs.emplace_back(stride_h);
    opArgs.emplace_back(stride_w);
    opArgs.emplace_back(dilation_h);
    opArgs.emplace_back(dilation_w);
    opArgs.emplace_back(pad_h);
    opArgs.emplace_back(pad_w);
    opArgs.emplace_back(y);
    opArgs.emplace_back(x);
    opArgs.emplace_back(log2_gemm_k_global_splits);
    kernel(opArgs);

    if(handle.IsProfilingEnabled())
        elapsed = handle.GetKernelTime();

    return elapsed;
}

// find wrw dynamic kernel by a simple algo
// check wether this kernel can be applicable
static inline std::tuple<bool, int>
FindImplicitGemmWrwGTCDynamicXdlopsKernel(const ConvolutionContext& ctx)
{
    int n               = ctx.batch_sz;
    int k               = ctx.n_inputs;
    int c               = ctx.n_outputs;
    int ho              = ctx.in_height;
    int wo              = ctx.in_width;
    int y               = ctx.kernel_size_h;
    int x               = ctx.kernel_size_w;
    const auto stride_h = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx);
    const auto stride_w = ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx);

    int gemm_n = c * y * x;
    int gemm_m = k;
    int gemm_k = n * ho * wo;

    int gemm_m_per_block;
    int gemm_n_per_block;
    int gemm_k_per_block    = 0;
    int gemm_k_global_split = 0;

    int grid_size;
    int block_size;
    int nxb = 1;
    int nxe = 1;

    int sel_index = -1;

    std::vector<TunableImplicitGemmGTCDynamic_t> tunables =
        GetImplicitGemmWrwGTCDynamicXdlopsKernelList();

    std::string selected_kernel = std::string("NONE");

    /* applicable table (except 128x128 case):
    gemm_m/gemmn        256 64  32  16  4
                --------------------------
                256 |   0  |1  |0  |0  |0
                64  |   1  |1  |0  |0  |1
                32  |   1  |1  |1  |1  |0
                16  |   0  |1  |0  |0  |0

    */
    int i, j;
    int max_grid_size  = 0;
    int cur_grid_size  = 0;
    int num_cu         = 120;
    int max_block_size = 0;

    // i=log2(gemm_m_per_block*gemm_n_per_block)  to find largest kernel
    // switch l and r to get differnet kernel size like 256*64 or 64*256
    for(i = 15; i > 7; i--)
    {
        int r, l;
        r = (i + 1) >> 1;
        l = i - r;
        while(l > 1 && r < 9)
        {
            for(int swap = 0; swap < 2; swap++)
            {
                if(swap == 0)
                {
                    gemm_m_per_block = 1 << r;
                    gemm_n_per_block = 1 << l;
                }
                else
                {
                    gemm_m_per_block = 1 << l;
                    gemm_n_per_block = 1 << r;
                }

                if(gemm_m % gemm_m_per_block != 0 || gemm_n % gemm_n_per_block != 0)
                    continue;
                for(j = 4; j > 1; j--)
                {
                    gemm_k_per_block = 1 << j;
                    if(gemm_k % gemm_k_per_block != 0)
                        continue;
                    gemm_k_global_split = if_gemm_k_global_split(
                        ctx, gemm_m_per_block, gemm_n_per_block, gemm_k_per_block);

                    nxb               = 1;
                    nxe               = 1;
                    int tunable_index = -1;

                    if((x * y * stride_h * stride_w == 1) && (ho * wo % 4 == 0))
                    {
                        nxb           = 4;
                        nxe           = 0;
                        tunable_index = find_tunable(tunables,
                                                     gemm_m_per_block,
                                                     gemm_n_per_block,
                                                     gemm_k_per_block,
                                                     gemm_k_global_split,
                                                     nxb,
                                                     nxe);
                        if(tunable_index < 0 || tunable_index >= tunables.size())
                        {
                            nxb = 1;
                            nxe = 1;

                            // std::cout << gemm_m_per_block << ", " << gemm_n_per_block << ", " <<
                            // gemm_k_per_block << std::endl;

                            tunable_index = find_tunable(tunables,
                                                         gemm_m_per_block,
                                                         gemm_n_per_block,
                                                         gemm_k_per_block,
                                                         gemm_k_global_split,
                                                         nxb,
                                                         nxe);
                        }
                    }
                    else
                    {
                        tunable_index = find_tunable(tunables,
                                                     gemm_m_per_block,
                                                     gemm_n_per_block,
                                                     gemm_k_per_block,
                                                     gemm_k_global_split,
                                                     nxb,
                                                     nxe);
                    }

                    if(tunable_index < 0 || tunable_index >= tunables.size())
                        continue;

                    int log2_gemm_k_global_splits = 0;
                    grid_size = (gemm_m / gemm_m_per_block) * (gemm_n / gemm_n_per_block);
                    for(int gs = 0; gs < 8; gs++)
                    {
                        if((grid_size << gs) > 1200)
                            break;

                        if((n % (1 << gs)) != 0)
                        {
                            break;
                        }

                        if((n >> gs) * ho * wo % gemm_k_per_block != 0)
                        {
                            break;
                        }
                        log2_gemm_k_global_splits = gs;
                    }

                    if(gemm_k_global_split == 0)
                        log2_gemm_k_global_splits = 0;

                    // std::cout << tunable_index << std::endl;

                    block_size = tunables[tunable_index].GetBlockSize();

                    cur_grid_size = grid_size << log2_gemm_k_global_splits;

                    if(block_size >= max_block_size && cur_grid_size > max_grid_size)
                    {
                        max_block_size = block_size;
                        max_grid_size  = cur_grid_size;
                        sel_index      = tunable_index;
                    }

                    if(max_grid_size > num_cu * 2)
                        break;
                }
                if(max_grid_size > num_cu * 2)
                    break;
            }
            if(max_grid_size > num_cu * 2)
                break;

            r++;
            l--;
        }
        if(max_grid_size > num_cu)
            break;
    }
    // std::cout << "sel_index:" << sel_index << std::endl;
    bool is_valid = !(sel_index < 0 || sel_index >= tunables.size());

    return std::make_tuple(is_valid, sel_index);
}

bool ConvAsmImplicitGemmGTCDynamicWrwXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(device_name, "gfx908")))
        return false;

    if(!IsApplicableXdlops(ctx))
        return false;

    if(!ctx.direction.IsBackwardWrW())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.IsFp32())
        return false;

    if(!ctx.rmv.IsV3())
        return false;

    if(ctx.group_counts != 1)
        return false;

    bool is_valid;
    std::tie(is_valid, std::ignore) = FindImplicitGemmWrwGTCDynamicXdlopsKernel(ctx);

    return is_valid;
}

ConvSolution
ConvAsmImplicitGemmGTCDynamicWrwXdlops::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution result;

    KernelInfo kernel;
    std::ostringstream options;

    std::vector<TunableImplicitGemmGTCDynamic_t> kernel_configs =
        GetImplicitGemmWrwGTCDynamicXdlopsKernelList();

    int block_size;
    int grid_size;
    std::string kernel_name;
    bool is_valid    = false;
    int kernel_index = -1;
    std::tie(is_valid, kernel_index) = FindImplicitGemmWrwGTCDynamicXdlopsKernel(ctx);

    if(!is_valid)
        MIOPEN_THROW("this kernel should not run with igemm dynamic!");

    kernel_name = kernel_configs[kernel_index].GetKernelName();
    block_size  = kernel_configs[kernel_index].GetBlockSize();

    int log2_gemm_k_global_splits = 0;

    std::tie(log2_gemm_k_global_splits, grid_size) =
        get_grid_size(ctx, &kernel_configs[kernel_index]);

    // std::cout << "tuple=" << grid_size << " " << log2_gemm_k_global_splits << std::endl;

    result.workspce_sz = 0;

    kernel.kernel_file = "igemm_wrw_gtc_gfx908.s";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    /* Note here, for API like hipHccModuleLaunchKernel(), hipExtModuleLaunchKernel()
    * grid dims is in unit of work item.
    * But for api like hipModuleLaunchKernel(), grid dim is in unit of block.
    */
    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);

    kernel.comp_options = options.str();

    MIOPEN_LOG_I2(kernel.kernel_file + ":" + kernel.kernel_name);

    result.construction_params.push_back(kernel);

    const auto& conv_problem = ctx.conv_problem;

    result.invoker_factory = [conv_problem,
                              log2_gemm_k_global_splits](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            decltype(auto) data_ctx = primitive_parameters.CastTo<conv::WrWInvokeParams>();
            const auto& tensors     = data_ctx.tensors;
            std::vector<KernelInvoke> ks;
            std::transform(kernels.begin(),
                           kernels.end(),
                           std::back_inserter(ks),
                           [&](const Kernel& k_wrw) { return handle.Run(k_wrw); });
            float elapsed = 0;
            float zero    = 0.f;

            SetTensor(handle, tensors.dwDesc, tensors.dw, &zero);
            if(handle.IsProfilingEnabled())
                elapsed += handle.GetKernelTime();

            elapsed += CallImplicitGemmWrwDynamic(handle,
                                                  conv_problem,
                                                  tensors.x,
                                                  tensors.dy,
                                                  tensors.dw,
                                                  ks,
                                                  log2_gemm_k_global_splits);
            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed);
            }
        };
    };

    return result;
}

} // namespace solver
} // namespace miopen
