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
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/solver.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/tensor.hpp>
#include <miopen/idiv.hpp>
#include <boost/any.hpp>

// clang-format off
constexpr int start_r2c_s   = 0;
constexpr int start_c2r_s   = start_r2c_s+8;
constexpr int start_r2c_x   = start_c2r_s+22;
constexpr int start_r2c_xg  = start_r2c_x+6;
constexpr int start_c2r_x   = start_r2c_xg+8;
constexpr int start_r2c     = start_c2r_x+4;
constexpr int start_r2c_pad = start_r2c+96;
constexpr int start_c2r     = start_r2c_pad+48;

static const char* g_knames[] =
{
    "sfft4x4_r2c_perm_s3x3",
    "sfft4x4_r2c_perm_s5x5",
    "sfft4x4_r2c_qerm_s3x3",
    "sfft4x4_r2c_qerm_s5x5",
    "sfft5x5_r2c_perm_s3x3",
    "sfft5x5_r2c_perm_s5x5",
    "sfft5x5_r2c_qerm_s3x3",
    "sfft5x5_r2c_qerm_s5x5",

    "sfft4x4_c2r_g3x1",
    "sfft4x4_c2r_g5x1",
    "sfft4x4_c2r_g7x1",
    "sfft4x4_c2r_g9x1",
    "sfft4x4_c2r_g1x3",
    "sfft4x4_c2r_g1x5",
    "sfft4x4_c2r_g1x7",
    "sfft4x4_c2r_g1x9",
    "sfft4x4_c2r_g3x3",
    "sfft4x4_c2r_g5x5",
    "sfft4x4_c2r_g7x7",
    "sfft5x5_c2r_g3x1",
    "sfft5x5_c2r_g5x1",
    "sfft5x5_c2r_g7x1",
    "sfft5x5_c2r_g9x1",
    "sfft5x5_c2r_g1x3",
    "sfft5x5_c2r_g1x5",
    "sfft5x5_c2r_g1x7",
    "sfft5x5_c2r_g1x9",
    "sfft5x5_c2r_g3x3",
    "sfft5x5_c2r_g5x5",
    "sfft5x5_c2r_g7x7",

    "sfft4x4_r2c_grid_perm"    ,
    "sfft4x4_r2c_grid_perm_ex" ,
    "sfft4x4_r2c_grid_perm_pad",
    "sfft5x5_r2c_grid_perm"    ,
    "sfft5x5_r2c_grid_perm_ex" ,
    "sfft5x5_r2c_grid_perm_pad",

    "sfft4x4_r2c_grad"    ,
    "sfft4x4_r2c_grad_ex" ,
    "sfft4x4_r2c_grad_pad",
    "sfft4x4_r2c_grad_nov",
    "sfft5x5_r2c_grad"    ,
    "sfft5x5_r2c_grad_ex" ,
    "sfft5x5_r2c_grad_pad",
    "sfft5x5_r2c_grad_nov",

    "sfft4x4_c2r_grid_perm"     ,
    "sfft4x4_c2r_grid_perm_relu",
    "sfft5x5_c2r_grid_perm"     ,
    "sfft5x5_c2r_grid_perm_relu",

    "sfft4x4_r2c_perm_v01",
    "sfft4x4_r2c_perm_v02",
    "sfft4x4_r2c_perm_v03",
    "sfft4x4_r2c_perm_v04",
    "sfft4x4_r2c_perm_v05",
    "sfft4x4_r2c_perm_v06",
    "sfft4x4_r2c_perm_v07",
    "sfft4x4_r2c_perm_v08",
    "sfft4x4_r2c_perm_v09",
    "sfft4x4_r2c_perm_v10",
    "sfft4x4_r2c_perm_v11",
    "sfft4x4_r2c_perm_v12",
    "sfft4x4_r2c_perm_v13",
    "sfft4x4_r2c_perm_v14",
    "sfft4x4_r2c_perm_v15",
    "sfft4x4_r2c_perm_v16",
    "sfft5x5_r2c_perm_v01",
    "sfft5x5_r2c_perm_v02",
    "sfft5x5_r2c_perm_v03",
    "sfft5x5_r2c_perm_v04",
    "sfft5x5_r2c_perm_v05",
    "sfft5x5_r2c_perm_v06",
    "sfft5x5_r2c_perm_v07",
    "sfft5x5_r2c_perm_v08",
    "sfft5x5_r2c_perm_v09",
    "sfft5x5_r2c_perm_v10",
    "sfft5x5_r2c_perm_v11",
    "sfft5x5_r2c_perm_v12",
    "sfft5x5_r2c_perm_v13",
    "sfft5x5_r2c_perm_v14",
    "sfft5x5_r2c_perm_v15",
    "sfft5x5_r2c_perm_v16",
    "sfft5x5_r2c_perm_v17",
    "sfft5x5_r2c_perm_v18",
    "sfft5x5_r2c_perm_v19",
    "sfft5x5_r2c_perm_v20",
    "sfft5x5_r2c_perm_v21",
    "sfft5x5_r2c_perm_v22",
    "sfft5x5_r2c_perm_v23",
    "sfft5x5_r2c_perm_v24",
    "sfft5x5_r2c_perm_v25",
    "sfft5x5_r2c_perm_v26",
    "sfft5x5_r2c_perm_v27",
    "sfft5x5_r2c_perm_v28",
    "sfft5x5_r2c_perm_v29",
    "sfft5x5_r2c_perm_v30",
    "sfft5x5_r2c_perm_v31",
    "sfft5x5_r2c_perm_v32",

    "sfft4x4_r2c_qerm_v01",
    "sfft4x4_r2c_qerm_v02",
    "sfft4x4_r2c_qerm_v03",
    "sfft4x4_r2c_qerm_v04",
    "sfft4x4_r2c_qerm_v05",
    "sfft4x4_r2c_qerm_v06",
    "sfft4x4_r2c_qerm_v07",
    "sfft4x4_r2c_qerm_v08",
    "sfft4x4_r2c_qerm_v09",
    "sfft4x4_r2c_qerm_v10",
    "sfft4x4_r2c_qerm_v11",
    "sfft4x4_r2c_qerm_v12",
    "sfft4x4_r2c_qerm_v13",
    "sfft4x4_r2c_qerm_v14",
    "sfft4x4_r2c_qerm_v15",
    "sfft4x4_r2c_qerm_v16",
    "sfft5x5_r2c_qerm_v01",
    "sfft5x5_r2c_qerm_v02",
    "sfft5x5_r2c_qerm_v03",
    "sfft5x5_r2c_qerm_v04",
    "sfft5x5_r2c_qerm_v05",
    "sfft5x5_r2c_qerm_v06",
    "sfft5x5_r2c_qerm_v07",
    "sfft5x5_r2c_qerm_v08",
    "sfft5x5_r2c_qerm_v09",
    "sfft5x5_r2c_qerm_v10",
    "sfft5x5_r2c_qerm_v11",
    "sfft5x5_r2c_qerm_v12",
    "sfft5x5_r2c_qerm_v13",
    "sfft5x5_r2c_qerm_v14",
    "sfft5x5_r2c_qerm_v15",
    "sfft5x5_r2c_qerm_v16",
    "sfft5x5_r2c_qerm_v17",
    "sfft5x5_r2c_qerm_v18",
    "sfft5x5_r2c_qerm_v19",
    "sfft5x5_r2c_qerm_v20",
    "sfft5x5_r2c_qerm_v21",
    "sfft5x5_r2c_qerm_v22",
    "sfft5x5_r2c_qerm_v23",
    "sfft5x5_r2c_qerm_v24",
    "sfft5x5_r2c_qerm_v25",
    "sfft5x5_r2c_qerm_v26",
    "sfft5x5_r2c_qerm_v27",
    "sfft5x5_r2c_qerm_v28",
    "sfft5x5_r2c_qerm_v29",
    "sfft5x5_r2c_qerm_v30",
    "sfft5x5_r2c_qerm_v31",
    "sfft5x5_r2c_qerm_v32",

    "sfft4x4_r2c_perm_p00",
    "sfft4x4_r2c_perm_p01",
    "sfft4x4_r2c_perm_p02",
    "sfft4x4_r2c_perm_p03",
    "sfft4x4_r2c_perm_p04",
    "sfft4x4_r2c_perm_p05",
    "sfft4x4_r2c_perm_p06",
    "sfft4x4_r2c_perm_p07",
    "sfft4x4_r2c_perm_p08",
    "sfft4x4_r2c_perm_p09",
    "sfft4x4_r2c_perm_p10",
    "sfft4x4_r2c_perm_p11",
    "sfft4x4_r2c_perm_p12",
    "sfft4x4_r2c_perm_p13",
    "sfft4x4_r2c_perm_p14",
    "sfft4x4_r2c_perm_p15",
    "sfft5x5_r2c_perm_p00",
    "sfft5x5_r2c_perm_p01",
    "sfft5x5_r2c_perm_p02",
    "sfft5x5_r2c_perm_p03",
    "sfft5x5_r2c_perm_p04",
    "sfft5x5_r2c_perm_p05",
    "sfft5x5_r2c_perm_p06",
    "sfft5x5_r2c_perm_p07",
    "sfft5x5_r2c_perm_p08",
    "sfft5x5_r2c_perm_p09",
    "sfft5x5_r2c_perm_p10",
    "sfft5x5_r2c_perm_p11",
    "sfft5x5_r2c_perm_p12",
    "sfft5x5_r2c_perm_p13",
    "sfft5x5_r2c_perm_p14",
    "sfft5x5_r2c_perm_p15",
    "sfft5x5_r2c_perm_p16",
    "sfft5x5_r2c_perm_p17",
    "sfft5x5_r2c_perm_p18",
    "sfft5x5_r2c_perm_p19",
    "sfft5x5_r2c_perm_p20",
    "sfft5x5_r2c_perm_p21",
    "sfft5x5_r2c_perm_p22",
    "sfft5x5_r2c_perm_p23",
    "sfft5x5_r2c_perm_p24",
    "sfft5x5_r2c_perm_p25",
    "sfft5x5_r2c_perm_p26",
    "sfft5x5_r2c_perm_p27",
    "sfft5x5_r2c_perm_p28",
    "sfft5x5_r2c_perm_p29",
    "sfft5x5_r2c_perm_p30",
    "sfft5x5_r2c_perm_p31",

    "sfft4x4_c2r_perm_v01",
    "sfft4x4_c2r_perm_v02",
    "sfft4x4_c2r_perm_v03",
    "sfft4x4_c2r_perm_v04",
    "sfft4x4_c2r_perm_v05",
    "sfft4x4_c2r_perm_v06",
    "sfft4x4_c2r_perm_v07",
    "sfft4x4_c2r_perm_v08",
    "sfft4x4_c2r_perm_v09",
    "sfft4x4_c2r_perm_v10",
    "sfft4x4_c2r_perm_v11",
    "sfft4x4_c2r_perm_v12",
    "sfft4x4_c2r_perm_v13",
    "sfft4x4_c2r_perm_v14",
    "sfft4x4_c2r_perm_v15",
    "sfft4x4_c2r_perm_v16",
    "sfft4x4_c2r_perm_relu_v01",
    "sfft4x4_c2r_perm_relu_v02",
    "sfft4x4_c2r_perm_relu_v03",
    "sfft4x4_c2r_perm_relu_v04",
    "sfft4x4_c2r_perm_relu_v05",
    "sfft4x4_c2r_perm_relu_v06",
    "sfft4x4_c2r_perm_relu_v07",
    "sfft4x4_c2r_perm_relu_v08",
    "sfft4x4_c2r_perm_relu_v09",
    "sfft4x4_c2r_perm_relu_v10",
    "sfft4x4_c2r_perm_relu_v11",
    "sfft4x4_c2r_perm_relu_v12",
    "sfft4x4_c2r_perm_relu_v13",
    "sfft4x4_c2r_perm_relu_v14",
    "sfft4x4_c2r_perm_relu_v15",
    "sfft4x4_c2r_perm_relu_v16",
    "sfft5x5_c2r_perm_v01",
    "sfft5x5_c2r_perm_v02",
    "sfft5x5_c2r_perm_v03",
    "sfft5x5_c2r_perm_v04",
    "sfft5x5_c2r_perm_v05",
    "sfft5x5_c2r_perm_v06",
    "sfft5x5_c2r_perm_v07",
    "sfft5x5_c2r_perm_v08",
    "sfft5x5_c2r_perm_v09",
    "sfft5x5_c2r_perm_v10",
    "sfft5x5_c2r_perm_v11",
    "sfft5x5_c2r_perm_v12",
    "sfft5x5_c2r_perm_v13",
    "sfft5x5_c2r_perm_v14",
    "sfft5x5_c2r_perm_v15",
    "sfft5x5_c2r_perm_v16",
    "sfft5x5_c2r_perm_v17",
    "sfft5x5_c2r_perm_v18",
    "sfft5x5_c2r_perm_v19",
    "sfft5x5_c2r_perm_v20",
    "sfft5x5_c2r_perm_v21",
    "sfft5x5_c2r_perm_v22",
    "sfft5x5_c2r_perm_v23",
    "sfft5x5_c2r_perm_v24",
    "sfft5x5_c2r_perm_v25",
    "sfft5x5_c2r_perm_v26",
    "sfft5x5_c2r_perm_v27",
    "sfft5x5_c2r_perm_v28",
    "sfft5x5_c2r_perm_v29",
    "sfft5x5_c2r_perm_v30",
    "sfft5x5_c2r_perm_v31",
    "sfft5x5_c2r_perm_v32",
    "sfft5x5_c2r_perm_relu_v01",
    "sfft5x5_c2r_perm_relu_v02",
    "sfft5x5_c2r_perm_relu_v03",
    "sfft5x5_c2r_perm_relu_v04",
    "sfft5x5_c2r_perm_relu_v05",
    "sfft5x5_c2r_perm_relu_v06",
    "sfft5x5_c2r_perm_relu_v07",
    "sfft5x5_c2r_perm_relu_v08",
    "sfft5x5_c2r_perm_relu_v09",
    "sfft5x5_c2r_perm_relu_v10",
    "sfft5x5_c2r_perm_relu_v11",
    "sfft5x5_c2r_perm_relu_v12",
    "sfft5x5_c2r_perm_relu_v13",
    "sfft5x5_c2r_perm_relu_v14",
    "sfft5x5_c2r_perm_relu_v15",
    "sfft5x5_c2r_perm_relu_v16",
    "sfft5x5_c2r_perm_relu_v17",
    "sfft5x5_c2r_perm_relu_v18",
    "sfft5x5_c2r_perm_relu_v19",
    "sfft5x5_c2r_perm_relu_v20",
    "sfft5x5_c2r_perm_relu_v21",
    "sfft5x5_c2r_perm_relu_v22",
    "sfft5x5_c2r_perm_relu_v23",
    "sfft5x5_c2r_perm_relu_v24",
    "sfft5x5_c2r_perm_relu_v25",
    "sfft5x5_c2r_perm_relu_v26",
    "sfft5x5_c2r_perm_relu_v27",
    "sfft5x5_c2r_perm_relu_v28",
    "sfft5x5_c2r_perm_relu_v29",
    "sfft5x5_c2r_perm_relu_v30",
    "sfft5x5_c2r_perm_relu_v31",
    "sfft5x5_c2r_perm_relu_v32"
};
// clang-format on

// return with a cgemm_id to select cgemm tile: scgemm5x4, scgemm5x5, scgemm5x6, scgemm6x4,
// scgemm6x5, scgemm6x6
static uint32_t choose_cgemm_id(uint32_t m, uint32_t n)
{
    uint32_t mi = (m + 31u) >> 5;
    uint32_t ni = (n + 15u) >> 4;
    return ((1 ^ (mi & 1)) * 3 + ((ni & 3) == 0 ? 2 : (1 ^ (ni & 1))));
}

// round up the num to be multiple of the mul
static uint32_t round_up(uint32_t num, uint mul) { return (((num) + (mul)) & (~(mul))); }

static uint32_t choose_optimal_cell_id(uint32_t anx, uint32_t any, uint32_t bnx, uint32_t bny)
{
    uint32_t id = 1, n = 0x7fffffff;
    for(int i = 1; i >= 0; --i)
    {
        uint32_t cell_size = 1 << (4 + i);
        uint32_t tile_x    = cell_size - bnx + 1;
        uint32_t tile_y    = cell_size - bny + 1;
        uint32_t grid_x    = (anx + tile_x - bnx) / tile_x;
        uint32_t grid_y    = (any + tile_y - bny) / tile_y;
        uint32_t size      = grid_x * grid_y * cell_size * ((cell_size >> 1) + 1);
        if(size < n)
        {
            id = i;
            n  = size;
        }
    }
    return id;
}
enum cellfft_dir
{
    cellFFTFwdConv = 0, // convolution forward path
    cellFFTBwdConv = 1, // convolution backward data path
    cellFFTWrWConv = 2, // convolution backward weight path
};

namespace miopen {
namespace solver {
struct cellfft_param_t
{
    magic_t xmag;
    magic_t ymag;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t tile_x;
    uint32_t tile_y;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t abks;
    uint32_t bbks;
    uint32_t cbks;
    uint32_t aldx;
    uint32_t aldy;
    uint32_t bldx;
    uint32_t bldy;
    uint32_t cldx;
    uint32_t cldy;
    uint32_t anx;
    uint32_t any;
    uint32_t bnx;
    uint32_t bny;
    uint32_t cnx;
    uint32_t cny;
    uint32_t pad_l;
    uint32_t pad_r;
    uint32_t pad_t;
    uint32_t pad_b;
    uint32_t nbanks;
    uint32_t id;
    cellfft_dir dir;
};
static size_t get_auxbuf_size(const ConvolutionContext& ctx)
{
    uint32_t bs  = ctx.batch_sz;
    uint32_t inc = ctx.n_inputs;
    uint32_t onc = ctx.n_outputs;
    uint32_t anx = ctx.in_width;
    uint32_t any = ctx.in_height;
    uint32_t fnx = ctx.kernel_size_w;
    uint32_t fny = ctx.kernel_size_h;
    uint32_t pu  = ctx.pad_w;
    uint32_t pv  = ctx.pad_h;
    if(!ctx.direction.IsForward())
    {
        pu = ctx.GetBackwardPadW();
        pv = ctx.GetBackwardPadH();
    }
    uint32_t pnx    = anx + (pu << 1);
    uint32_t pny    = any + (pv << 1);
    uint32_t id     = choose_optimal_cell_id(pnx, pny, fnx, fny);
    uint32_t cell   = 1 << (4 + id);
    uint32_t nbanks = cell * ((cell >> 1) + 1);
    uint32_t tile_x = cell - fnx + 1;
    uint32_t tile_y = cell - fny + 1;
    uint32_t grid_x = (pnx + tile_x - fnx) / tile_x;
    uint32_t grid_y = (pny + tile_y - fny) / tile_y;
    grid_x          = grid_x == 0 ? 1 : grid_x;
    grid_y          = grid_y == 0 ? 1 : grid_y;
    uint32_t m      = bs * grid_x * grid_y;
    uint32_t n      = onc;
    uint32_t k      = inc;
    uint32_t ek     = round_up(k, 7);
    uint32_t lda    = round_up(m, 31) >> 5;
    uint32_t ldb    = round_up(n, 31) >> 5;
    lda             = (lda + (1 ^ (lda & 1))) << 5;
    ldb             = (ldb + (1 ^ (ldb & 1))) << 5;
    uint64_t abks   = lda * ek + 16;
    uint64_t bbks   = ldb * ek + 16;
    uint64_t cbks   = lda * n + 16;
    return ((abks + bbks + cbks) * (nbanks << 3));
}
static size_t get_auxbuf_size_grad(const ConvolutionContext& ctx)
{
    uint32_t bs  = ctx.batch_sz;
    uint32_t pnc = ctx.n_outputs;
    uint32_t qnc = ctx.n_inputs;
    uint32_t cnx = ctx.kernel_size_w;
    uint32_t cny = ctx.kernel_size_h;
    uint32_t anx = ctx.out_width;
    uint32_t any = ctx.out_height;
    uint32_t pu  = ctx.pad_w;
    uint32_t pv  = ctx.pad_h;
    uint32_t pnx = anx;
    uint32_t pny = any;
    if((pu | pv) != 0)
    {
        pnx += pu << 1;
        pny += pv << 1;
    }
    uint32_t id     = choose_optimal_cell_id(pnx, pny, cnx, cny);
    uint32_t cell   = 1 << (4 + id);
    uint32_t nbanks = cell * ((cell >> 1) + 1);
    uint32_t tile_x = cell - cnx + 1;
    uint32_t tile_y = cell - cny + 1;
    uint32_t grid_x = (pnx + tile_x - cnx) / tile_x;
    uint32_t grid_y = (pny + tile_y - cny) / tile_y;
    grid_x          = grid_x == 0 ? 1 : grid_x;
    grid_y          = grid_y == 0 ? 1 : grid_y;
    uint32_t k      = bs * grid_x * grid_y;
    uint32_t ek     = round_up(k, 7);
    uint32_t lda    = round_up(pnc, 31) >> 5;
    uint32_t ldb    = round_up(qnc, 31) >> 5;
    lda             = (lda + (1 ^ (lda & 1))) << 5;
    ldb             = (ldb + (1 ^ (ldb & 1))) << 5;
    uint64_t abks   = lda * ek + 16;
    uint64_t bbks   = ldb * ek + 16;
    uint64_t cbks   = lda * qnc + 16;
    return ((abks + bbks + cbks) * (nbanks << 3));
}
static void build_cellfft_params(cellfft_param_t& p, const ConvolutionContext& ctx)
{
    uint32_t bs  = ctx.batch_sz;
    uint32_t inc = ctx.n_inputs;
    uint32_t onc = ctx.n_outputs;
    uint32_t pu  = ctx.pad_w;
    uint32_t pv  = ctx.pad_h;
    p.anx        = ctx.in_width;
    p.any        = ctx.in_height;
    p.bnx        = ctx.kernel_size_w;
    p.bny        = ctx.kernel_size_h;
    p.cnx        = ctx.out_width;
    p.cny        = ctx.out_height;
    if(ctx.direction.IsForward())
    {
        p.dir = cellFFTFwdConv;
    }
    else
    {
        p.dir = cellFFTBwdConv;
    }
    if(p.dir == cellFFTBwdConv)
    {
        pu = ctx.GetBackwardPadW();
        pv = ctx.GetBackwardPadH();
    }
    p.pad_l       = pu;
    p.pad_r       = pu;
    p.pad_t       = pv;
    p.pad_b       = pv;
    uint32_t pnx  = p.anx + (pu << 1);
    uint32_t pny  = p.any + (pv << 1);
    p.id          = choose_optimal_cell_id(pnx, pny, p.bnx, p.bny);
    uint32_t cell = 1 << (4 + p.id);
    p.nbanks      = cell * ((cell >> 1) + 1);
    p.tile_x      = cell - p.bnx + 1;
    p.tile_y      = cell - p.bny + 1;
    p.grid_x      = (pnx + p.tile_x - p.bnx) / p.tile_x;
    p.grid_y      = (pny + p.tile_y - p.bny) / p.tile_y;
    p.grid_x      = p.grid_x == 0 ? 1 : p.grid_x;
    p.grid_y      = p.grid_y == 0 ? 1 : p.grid_y;
    p.m           = bs * p.grid_x * p.grid_y;
    p.n           = onc;
    p.k           = inc;
    uint32_t ek   = round_up(p.k, 7);
    p.lda         = round_up(p.m, 31) >> 5;
    p.ldb         = round_up(p.n, 31) >> 5;
    p.lda         = (p.lda + (1 ^ (p.lda & 1))) << 5;
    p.ldb         = (p.ldb + (1 ^ (p.ldb & 1))) << 5;
    p.abks        = p.lda * ek + 16;
    p.bbks        = p.ldb * ek + 16;
    p.cbks        = p.lda * p.n + 16;
    p.aldy        = p.anx * p.any;
    p.cldy        = p.cnx * p.cny;
    p.bldy        = p.bnx * p.bny;
    p.aldx        = inc * p.aldy;
    p.cldx        = onc * p.cldy;
    p.bldx        = (p.dir == cellFFTFwdConv ? inc : 1) * p.bldy;
    p.bldy        = (p.dir == cellFFTFwdConv ? 1 : onc) * p.bldy;
    if((p.grid_x | p.grid_y) != 1)
    {
        uint32_t pm   = round_up(p.m, 15);
        uint32_t reso = p.grid_x * p.grid_y;
        p.xmag        = idiv_magic(pm, reso);
        p.ymag        = idiv_magic(reso, p.grid_x);
    }
}
static void build_cellfft_params_grad(cellfft_param_t& p, const ConvolutionContext& ctx)
{
    uint32_t bs   = ctx.batch_sz;
    uint32_t pnc  = ctx.n_outputs;
    uint32_t qnc  = ctx.n_inputs;
    uint32_t pu   = ctx.pad_w;
    uint32_t pv   = ctx.pad_h;
    p.anx         = ctx.out_width;
    p.any         = ctx.out_height;
    p.bnx         = ctx.in_width;
    p.bny         = ctx.in_height;
    p.cnx         = ctx.kernel_size_w;
    p.cny         = ctx.kernel_size_h;
    uint32_t pnx  = p.anx + (pu << 1);
    uint32_t pny  = p.any + (pv << 1);
    p.dir         = cellFFTWrWConv;
    p.pad_l       = pu;
    p.pad_r       = pu;
    p.pad_t       = pv;
    p.pad_b       = pv;
    p.id          = choose_optimal_cell_id(pnx, pny, p.cnx, p.cny);
    uint32_t cell = 1 << (4 + p.id);
    p.nbanks      = cell * ((cell >> 1) + 1);
    p.tile_x      = cell - p.cnx + 1;
    p.tile_y      = cell - p.cny + 1;
    p.grid_x      = (pnx + p.tile_x - p.cnx) / p.tile_x;
    p.grid_y      = (pny + p.tile_y - p.cny) / p.tile_y;
    p.grid_x      = p.grid_x == 0 ? 1 : p.grid_x;
    p.grid_y      = p.grid_y == 0 ? 1 : p.grid_y;
    p.m           = pnc;
    p.n           = qnc;
    p.k           = bs * p.grid_x * p.grid_y;
    uint32_t ek   = round_up(p.k, 7);
    p.lda         = round_up(p.m, 31) >> 5;
    p.ldb         = round_up(p.n, 31) >> 5;
    p.lda         = (p.lda + (1 ^ (p.lda & 1))) << 5;
    p.ldb         = (p.ldb + (1 ^ (p.ldb & 1))) << 5;
    p.abks        = p.lda * ek + 16;
    p.bbks        = p.ldb * ek + 16;
    p.cbks        = p.lda * p.n + 16;
    p.aldx        = p.anx * p.any;
    p.bldx        = p.bnx * p.bny;
    p.cldx        = p.cnx * p.cny;
    p.aldy        = p.m * p.aldx;
    p.bldy        = p.n * p.bldx;
    p.cldy        = p.m * p.cldx;
    if((p.grid_x | p.grid_y) != 1)
    {
        uint32_t pk   = round_up(p.k, 15);
        uint32_t reso = p.grid_x * p.grid_y;
        p.xmag        = idiv_magic(pk, reso);
        p.ymag        = idiv_magic(reso, p.grid_x);
    }
}
static void lk_cgemm(const Handle& h,
                     const Kernel& kern,
                     const cellfft_param_t& p,
                     void* c,
                     void* a,
                     void* b,
                     float alpha)
{
    // coef=alpha*(1.0/(cell_size*cell_size)), cell_size=16 or 32
    float coef = alpha * (p.id == 0 ? 0.00390625f : 0.0009765625f);
    h.Run(kern)(c, p.lda, p.cbks, a, b, p.lda, p.ldb, p.m, p.n, p.k, p.abks, p.bbks, coef);
}
static void lk_fft2d_r2c_perm_a(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    h.Run(kern)(dst,
                p.lda,
                p.abks,
                src,
                (p.dir != cellFFTWrWConv ? 0 : 0x80000000) | p.m,
                p.anx,
                p.aldx,
                p.aldy);
}
static void lk_fft2d_r2c_perm_b(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    h.Run(kern)(dst,
                p.ldb,
                p.bbks,
                src,
                (p.dir == cellFFTFwdConv ? 0 : 0x80000000) | p.n,
                p.bnx,
                p.bldx,
                p.bldy);
}
static void lk_fft2d_r2c_perm_pad(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    h.Run(kern)(dst,
                p.lda,
                p.abks,
                src,
                p.m,
                (p.dir != cellFFTWrWConv ? 0 : 0x80000000) | p.pad_l,
                p.aldx,
                p.aldy,
                p.anx,
                p.any);
}
static void lk_fft2d_r2c_perm_s(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    uint32_t ldr = p.bnx * p.bny * (p.dir == cellFFTFwdConv ? p.k : p.n);
    h.Run(kern)(dst, p.ldb, p.bbks, src, p.n, ldr);
}
static void lk_fft2d_r2c_grid_perm(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    uint32_t grid = (p.grid_y << 16) | p.grid_x;
    uint32_t tile = (p.tile_y << 16) | p.tile_x;
    h.Run(kern)(
        dst, p.lda, p.abks, src, p.m, p.anx, p.aldx, p.aldy, grid, tile, p.xmag, p.ymag, p.any);
}
static void lk_fft2d_r2c_grid_perm_pad(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    uint32_t grid = (p.grid_y << 16) | p.grid_x;
    uint32_t tile = (p.tile_y << 16) | p.tile_x;
    uint32_t pad  = (p.pad_t << 16) | p.pad_l;
    h.Run(kern)(dst,
                p.lda,
                p.abks,
                src,
                p.m,
                p.anx,
                p.aldx,
                p.aldy,
                grid,
                tile,
                p.xmag,
                p.ymag,
                p.any,
                pad);
}
static void lk_fft2d_r2c_grid_perm_nov(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    uint32_t grid = (p.grid_y << 16) | p.grid_x;
    uint32_t tile = (p.tile_y << 16) | p.tile_x;
    h.Run(kern)(
        dst, p.ldb, p.bbks, src, p.n, p.bnx, p.bldx, p.bldy, grid, tile, p.xmag, p.ymag, p.bny);
}
static void lk_fft2d_c2r_perm(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src)
{
    h.Run(kern)(dst, p.cldx, p.cldy, src, p.lda, p.cbks, p.m, p.cnx);
}
static void lk_fft2d_c2r_grid_perm(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src)
{
    uint32_t grid = (p.grid_y << 16) | p.grid_x;
    uint32_t tile = (p.tile_y << 16) | p.tile_x;
    h.Run(kern)(
        dst, p.cldx, p.cldy, src, p.lda, p.cbks, p.xmag, p.ymag, grid, tile, p.cnx, p.cny, p.m);
}
static void lk_fft2d_c2r_grad_perm(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src)
{
    h.Run(kern)(dst, p.cldx, p.cldy, src, p.lda, p.cbks, p.m, p.cnx);
}
static void lk_fft2d_c2r_grad_perm_s(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src)
{
    h.Run(kern)(dst, p.cnx * p.cny * p.m, p.m, src, p.lda, p.cbks);
}

static void cgemm(const Handle& h,
                  const Kernel& kern,
                  const cellfft_param_t& p,
                  void* c,
                  void* a,
                  void* b,
                  float alpha)
{
    lk_cgemm(h, kern, p, c, a, b, alpha);
}
static void fft2d_r2c_a(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    if((p.pad_l | p.pad_t) != 0)
    {
        lk_fft2d_r2c_perm_pad(h, kern, p, dst, src);
    }
    else
    {
        lk_fft2d_r2c_perm_a(h, kern, p, dst, src);
    }
}
static void fft2d_r2c_b(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    if((p.bnx == p.bny) && ((p.bnx == 3) || (p.bnx == 5)))
    {
        lk_fft2d_r2c_perm_s(h, kern, p, dst, src);
    }
    else
    {
        lk_fft2d_r2c_perm_b(h, kern, p, dst, src);
    }
}
static void fft2d_r2c_grid_a(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    if((p.pad_l | p.pad_t) != 0)
    {
        lk_fft2d_r2c_grid_perm_pad(h, kern, p, dst, src);
    }
    else
    {
        lk_fft2d_r2c_grid_perm(h, kern, p, dst, src);
    }
}
static void fft2d_r2c_grid_b(
    const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    lk_fft2d_r2c_grid_perm_nov(h, kern, p, dst, src);
}
static void
fft2d_c2r(const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src)
{
    lk_fft2d_c2r_perm(h, kern, p, dst, src);
}
static void
fft2d_c2r_grid(const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src)
{
    lk_fft2d_c2r_grid_perm(h, kern, p, dst, src);
}
static void
fft2d_c2r_grad(const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src)
{
    bool cc0 = (p.cnx == p.cny) && ((p.cnx == 3) || (p.cnx == 5) || (p.cnx == 7));
    bool cc1 = (p.cnx == 1) && ((p.cny & 0x1) != 0 && (p.cny > 1) && (p.cny <= 9));
    bool cc2 = (p.cny == 1) && ((p.cnx & 0x1) != 0 && (p.cnx > 1) && (p.cnx <= 9));
    if(cc0 || cc1 || cc2)
    {
        lk_fft2d_c2r_grad_perm_s(h, kern, p, dst, src);
    }
    else
    {
        lk_fft2d_c2r_grad_perm(h, kern, p, dst, src);
    }
}
static void
dtr(const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    if((p.grid_x | p.grid_y) > 1)
    {
        fft2d_r2c_grid_a(h, kern, p, dst, src);
    }
    else
    {
        fft2d_r2c_a(h, kern, p, dst, src);
    }
}
static void
ftr(const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, const void* src)
{
    if((p.dir == cellFFTWrWConv) && ((p.grid_x | p.grid_y) > 1))
    {
        fft2d_r2c_grid_b(h, kern, p, dst, src);
    }
    else
    {
        fft2d_r2c_b(h, kern, p, dst, src);
    }
}
static void otr(const Handle& h, const Kernel& kern, const cellfft_param_t& p, void* dst, void* src)
{
    if(p.dir != cellFFTWrWConv)
    {
        if((p.grid_x | p.grid_y) > 1)
        {
            fft2d_c2r_grid(h, kern, p, dst, src);
        }
        else
        {
            fft2d_c2r(h, kern, p, dst, src);
        }
    }
    else
    {
        fft2d_c2r_grad(h, kern, p, dst, src);
    }
}
InvokerFactory MakeCellfftInvokerFactory(const cellfft_param_t& conv_params, float alpha)
{
    return [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& h, const AnyInvokeParams& prim_params) {
            const size_t abks    = static_cast<size_t>(conv_params.abks);
            const size_t bbks    = static_cast<size_t>(conv_params.bbks);
            const size_t cbks    = static_cast<size_t>(conv_params.cbks);
            const size_t nbks    = static_cast<size_t>(conv_params.nbanks) << 3;
            const size_t auxsize = nbks * (abks + bbks + cbks);
            const auto& params   = prim_params.CastTo<conv::DataInvokeParams>();
            if(params.workSpace == nullptr || params.workSpaceSize < auxsize)
                MIOPEN_THROW("Workspace is not enough for cellfft");
            const auto& tensors = params.tensors;
            auto& auxbuf        = params.workSpace;
            const void* src     = tensors.in;
            const void* fil     = tensors.w;
            void* dst           = tensors.out;
            uint8_t* a          = reinterpret_cast<uint8_t*>(auxbuf);
            uint8_t* b          = a + nbks * abks;
            uint8_t* c          = b + nbks * bbks;
            float elapsed       = 0.f;
            dtr(h, kernels[1], conv_params, a, src);
            if(h.IsProfilingEnabled())
            {
                elapsed += h.GetKernelTime();
            }
            ftr(h, kernels[2], conv_params, b, fil);
            if(h.IsProfilingEnabled())
            {
                elapsed += h.GetKernelTime();
            }
            cgemm(h, kernels[0], conv_params, c, a, b, alpha);
            if(h.IsProfilingEnabled())
            {
                elapsed += h.GetKernelTime();
            }
            otr(h, kernels[3], conv_params, dst, c);
            if(h.IsProfilingEnabled())
            {
                elapsed += h.GetKernelTime();
                h.ResetKernelTime();
                h.AccumKernelTime(elapsed);
            }
        };
    };
}
InvokerFactory MakeCellfftInvokerFactoryGrad(const cellfft_param_t& conv_params, float alpha)
{
    return [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& h, const AnyInvokeParams& prim_params) {
            const size_t abks    = static_cast<size_t>(conv_params.abks);
            const size_t bbks    = static_cast<size_t>(conv_params.bbks);
            const size_t cbks    = static_cast<size_t>(conv_params.cbks);
            const size_t nbks    = static_cast<size_t>(conv_params.nbanks) << 3;
            const size_t auxsize = nbks * (abks + bbks + cbks);
            const auto& params   = prim_params.CastTo<conv::WrWInvokeParams>();
            if(params.workSpace == nullptr || params.workSpaceSize < auxsize)
                MIOPEN_THROW("Workspace is not enough for cellfft");
            const auto& tensors = params.tensors;
            auto& auxbuf        = params.workSpace;
            const void* pin     = tensors.x;
            const void* qin     = tensors.dy;
            void* dst           = tensors.dw;
            uint8_t* a          = reinterpret_cast<uint8_t*>(auxbuf);
            uint8_t* b          = a + nbks * abks;
            uint8_t* c          = b + nbks * bbks;
            float elapsed       = 0.f;
            dtr(h, kernels[1], conv_params, a, pin);
            if(h.IsProfilingEnabled())
            {
                elapsed += h.GetKernelTime();
            }
            ftr(h, kernels[2], conv_params, b, qin);
            if(h.IsProfilingEnabled())
            {
                elapsed += h.GetKernelTime();
            }
            cgemm(h, kernels[0], conv_params, c, a, b, alpha);
            if(h.IsProfilingEnabled())
            {
                elapsed += h.GetKernelTime();
            }
            otr(h, kernels[3], conv_params, dst, c);
            if(h.IsProfilingEnabled())
            {
                elapsed += h.GetKernelTime();
                h.ResetKernelTime();
                h.AccumKernelTime(elapsed);
            }
        };
    };
}
static KernelInfo
get_kinfo_cgemm(const ConvolutionContext& ctx, const cellfft_param_t& p, const std::string& fname)
{
    static const uint32_t blk[]    = {64, 64, 128, 64, 128, 256};
    static const char* knames[][2] = {{"scgemm5x4", "scgemm5x4_ck"},
                                      {"scgemm5x5", "scgemm5x5_ck"},
                                      {"scgemm5x6", "scgemm5x6_ck"},
                                      {"scgemm6x4", "scgemm6x4_ck"},
                                      {"scgemm6x5", "scgemm6x5_ck"},
                                      {"scgemm6x6", "scgemm6x6_ck"}};
    const uint32_t tile_id = choose_cgemm_id(p.m, p.n);
    const uint32_t shx     = tile_id < 3 ? 5 : 6;
    const uint32_t shy     = (0x654654 >> (tile_id << 2)) & 0xf;
    const uint32_t shz     = tile_id == 0 ? 1 : 0;
    const uint32_t gdx     = (p.m + (1 << shx) - 1) >> shx;
    const uint32_t gdy     = (p.n + (1 << shy) - 1) >> shy;
    const uint32_t gdz     = p.nbanks >> shz;
    const uint32_t kid     = (p.k & 7) != 0 ? 1 : 0;
    const std::vector<size_t> block{blk[tile_id], 1, 1};
    const std::vector<size_t> grid{gdx * blk[tile_id], gdy, gdz};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, fname, knames[tile_id][kid]};
}
static KernelInfo
get_kinfo_r2c_a(const ConvolutionContext& ctx, const cellfft_param_t& p, const std::string& fname)
{
    uint32_t kid = start_r2c + (p.id << 4) + p.any - 1;
    if((p.pad_l | p.pad_t) != 0)
    {
        kid = start_r2c_pad + (p.id << 4) + p.pad_t;
    }
    const uint32_t r = (p.m + 15) >> 4;
    const size_t bdx = p.id == 0 ? 256 : 512;
    const size_t gdx = p.dir != cellFFTWrWConv ? p.k : r;
    const size_t gdy = p.dir != cellFFTWrWConv ? r : p.k;
    const std::vector<size_t> block{bdx, 1, 1};
    const std::vector<size_t> grid{gdx * bdx, gdy, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, fname, g_knames[kid]};
}
static KernelInfo
get_kinfo_r2c_b(const ConvolutionContext& ctx, const cellfft_param_t& p, const std::string& fname)
{
    uint32_t kid = start_r2c + (p.dir != cellFFTBwdConv ? 0 : 48) + (p.id << 4) + p.bny - 1;
    if((p.bnx == p.bny) && ((p.bnx == 3) || (p.bnx == 5)))
    {
        kid = start_r2c_s + ((p.id << 2) | ((p.dir & cellFFTBwdConv) << 1) | (p.bnx == 3 ? 0 : 1));
    }
    const uint32_t r = (p.n + 15) >> 4;
    const size_t bdx = p.id == 0 ? 256 : 512;
    const size_t gdx = p.dir == cellFFTFwdConv ? p.k : r;
    const size_t gdy = p.dir == cellFFTFwdConv ? r : p.k;
    const std::vector<size_t> block{bdx, 1, 1};
    const std::vector<size_t> grid{gdx * bdx, gdy, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, fname, g_knames[kid]};
}
static KernelInfo
get_kinfo_r2c_x(const ConvolutionContext& ctx, const cellfft_param_t& p, const std::string& fname)
{
    const uint32_t nx  = p.tile_x * p.grid_x + p.bnx - 1;
    const uint32_t ny  = p.tile_y * p.grid_y + p.bny - 1;
    const uint32_t ex  = ((nx != p.anx) || (ny != p.any)) ? 1 : 0;
    const uint32_t kid = start_r2c_x + p.id * 3 + ((p.pad_l | p.pad_t) != 0 ? 2 : ex);
    const size_t bdx   = p.id == 0 ? 256 : 512;
    const size_t gdx   = (p.m + 15) >> 4;
    const std::vector<size_t> block{bdx, 1, 1};
    const std::vector<size_t> grid{gdx * bdx, p.k, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, fname, g_knames[kid]};
}
static KernelInfo
get_kinfo_r2c_xga(const ConvolutionContext& ctx, const cellfft_param_t& p, const std::string& fname)
{
    const uint32_t nx  = p.tile_x * p.grid_x + p.bnx - 1;
    const uint32_t ny  = p.tile_y * p.grid_y + p.bny - 1;
    const uint32_t ex  = ((nx != p.anx) || (ny != p.any)) ? 1 : 0;
    const uint32_t kid = start_r2c_xg + (p.id << 2) + ((p.pad_l | p.pad_t) != 0 ? 2 : ex);
    const size_t bdx   = p.id == 0 ? 256 : 512;
    const size_t gdx   = (p.m + 15) >> 4;
    const std::vector<size_t> block{bdx, 1, 1};
    const std::vector<size_t> grid{gdx * bdx, p.k, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, fname, g_knames[kid]};
}
static KernelInfo
get_kinfo_r2c_xgb(const ConvolutionContext& ctx, const cellfft_param_t& p, const std::string& fname)
{
    const uint32_t kid = start_r2c_xg + ((p.id << 2) | 3);
    const size_t bdx   = p.id == 0 ? 256 : 512;
    const size_t gdx   = (p.n + 15) >> 4;
    const std::vector<size_t> block{bdx, 1, 1};
    const std::vector<size_t> grid{gdx * bdx, p.k, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, fname, g_knames[kid]};
}
static KernelInfo get_kinfo_c2r(const ConvolutionContext& ctx,
                                const cellfft_param_t& p,
                                const std::string& fname,
                                uint32_t relu)
{
    const uint32_t shx = 4 - p.id;
    const uint32_t kid = start_c2r + (p.id << 5) + (relu << (p.id + 4)) + p.cny - 1;
    const size_t gdx   = (p.m + (1 << shx) - 1) >> shx;
    const std::vector<size_t> block{256, 1, 1};
    const std::vector<size_t> grid{gdx << 8, p.n, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, fname, g_knames[kid]};
}
static KernelInfo get_kinfo_c2r_x(const ConvolutionContext& ctx,
                                  const cellfft_param_t& p,
                                  const std::string& fname,
                                  uint32_t relu)
{
    const uint32_t shx = 4 - p.id;
    const uint32_t kid = start_c2r_x + ((p.id << 1) | relu);
    const size_t gdx   = (p.m + (1 << shx) - 1) >> shx;
    const std::vector<size_t> block{256, 1, 1};
    const std::vector<size_t> grid{gdx << 8, p.n, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, fname, g_knames[kid]};
}
static KernelInfo
get_kinfo_c2r_g(const ConvolutionContext& ctx, const cellfft_param_t& p, const std::string& fname)
{
    uint32_t nmax = p.cnx > p.cny ? p.cnx : p.cny;
    uint32_t nmin = p.cnx > p.cny ? p.cny : p.cnx;
    uint32_t shx  = 4 - p.id;
    uint32_t kid  = start_c2r + (p.id << 5) + p.cny - 1;
    bool cc0      = (p.cnx == p.cny) && ((p.cnx == 3) || (p.cnx == 5) || (p.cnx == 7));
    bool cc1      = (nmin == 1) && ((nmax & 1) != 0) && (nmax > 1) && (nmax <= 9);
    if(cc0 || cc1)
    {
        kid = start_c2r_s + 11 * p.id + (cc0 ? 8 : (p.cnx > p.cny ? 0 : 4)) + (nmax >> 1) - 1;
    }
    size_t gdx = (p.m + (1 << shx) - 1) >> shx;
    const std::vector<size_t> block{256, 1, 1};
    const std::vector<size_t> grid{gdx << 8, p.n, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, fname, g_knames[kid]};
}
static void
fill_kernels_info(ConvSolution& sol, const ConvolutionContext& ctx, const cellfft_param_t& p)
{
    const std::string fname = "cellfft_" + ctx.GetStream().GetDeviceName() + ".s";
    sol.construction_params.push_back(get_kinfo_cgemm(ctx, p, fname));
    if(p.dir != cellFFTWrWConv)
    {
        if((p.grid_x | p.grid_y) > 1)
        {
            sol.construction_params.push_back(get_kinfo_r2c_x(ctx, p, fname));
            sol.construction_params.push_back(get_kinfo_r2c_b(ctx, p, fname));
            sol.construction_params.push_back(get_kinfo_c2r_x(ctx, p, fname, 0));
        }
        else
        {
            sol.construction_params.push_back(get_kinfo_r2c_a(ctx, p, fname));
            sol.construction_params.push_back(get_kinfo_r2c_b(ctx, p, fname));
            sol.construction_params.push_back(get_kinfo_c2r(ctx, p, fname, 0));
        }
    }
    else
    {
        if((p.grid_x | p.grid_y) > 1)
        {
            sol.construction_params.push_back(get_kinfo_r2c_xga(ctx, p, fname));
            sol.construction_params.push_back(get_kinfo_r2c_xgb(ctx, p, fname));
        }
        else
        {
            sol.construction_params.push_back(get_kinfo_r2c_a(ctx, p, fname));
            sol.construction_params.push_back(get_kinfo_r2c_b(ctx, p, fname));
        }
        sol.construction_params.push_back(get_kinfo_c2r_g(ctx, p, fname));
    }
}
bool ConvCellfft::IsApplicable(const ConvolutionContext& ctx) const
{
#if MIOPEN_BACKEND_OPENCL
    (void)ctx;
    return false;
#else
    const auto name = ctx.GetStream().GetDeviceName();
    if(name != "gfx900" && name != "gfx906")
        return false;
    if(!ctx.IsLayoutDefault())
        return false;
    if((ctx.kernel_stride_w != 1) || (ctx.kernel_stride_h != 1) || (ctx.kernel_dilation_w != 1) ||
       (ctx.kernel_dilation_h != 1) || (ctx.group_counts != 1))
        return false;
    if(!ctx.direction.IsForward() && (ctx.GetBackwardPadW() < 0 || ctx.GetBackwardPadH() < 0))
        return false;
    // workaround when input is le 5x5, will select an optimized kernel for input
    // (like sfft4x4_r2c_perm_s5x5), which is not correct.
    if(ctx.direction.IsBackwardWrW() && (ctx.out_height <= 5 || ctx.out_width <= 5))
        return false;
    return (ctx.Is2d() && ctx.IsFp32() && (ctx.in_layout == "NCHW") && (ctx.bias == 0));
#endif
}
size_t ConvCellfft::GetWorkspaceSize(const ConvolutionContext& ctx) const
{
    if(!ctx.direction.IsBackwardWrW())
    {
        return get_auxbuf_size(ctx);
    }
    else
    {
        return get_auxbuf_size_grad(ctx);
    }
}
ConvSolution ConvCellfft::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution sol;
    cellfft_param_t params{};
    if(!ctx.direction.IsBackwardWrW())
    {
        build_cellfft_params(params, ctx);
    }
    else
    {
        build_cellfft_params_grad(params, ctx);
    }
    sol.workspce_sz = GetWorkspaceSize(ctx);
    fill_kernels_info(sol, ctx, params);
    if(!ctx.direction.IsBackwardWrW())
    {
        sol.invoker_factory = MakeCellfftInvokerFactory(params, 1.f);
    }
    else
    {
        sol.invoker_factory = MakeCellfftInvokerFactoryGrad(params, 1.f);
    }
    return sol;
}
} // namespace solver
} // namespace miopen
