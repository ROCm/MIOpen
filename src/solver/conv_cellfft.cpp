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
#include <sstream>
#include <miopen/config.h>
#include <miopen/kernel_info.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/solver.hpp>
#include <miopen/env.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/conv/tensors.hpp>
#include <boost/any.hpp>
#include <miopen/conv/invokers/cellfft.hpp>

// clang-format off
#define START_R2C_S   0
#define START_C2R_S   (START_R2C_S+8)
#define START_R2C_X   (START_C2R_S+22)
#define START_R2C_Xg  (START_R2C_X+6)
#define START_C2R_X   (START_R2C_Xg+8)
#define START_R2C     (START_C2R_X+4)
#define START_R2C_PAD (START_R2C+96)
#define START_C2R     (START_R2C_PAD+48)

static const char* g_knames[]=
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

static uint32_t choose_cgemm_id(uint32_t m, uint32_t n)
{
    uint32_t mi = (m + 31u) >> 5;
    uint32_t ni = (n + 15u) >> 4;
    return ((1 ^ (mi & 1)) * 3 + ((ni & 3) == 0 ? 2 : (1 ^ (ni & 1))));
}

namespace miopen {
namespace solver {
static KernelInfo get_kernel_cgemm(const ConvolutionContext& ctx,
                                   const cellfft_param_t& p,
                                   const std::string& file_name)
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
    const std::vector<size_t> block{blk[tile_id], 1, 1};
    const std::vector<size_t> grid{gdx * blk[tile_id], gdy, gdz};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{
        options.str(), block, grid, file_name, knames[tile_id][(p.k & 7) != 0 ? 1 : 0]};
}
static KernelInfo get_kernel_r2c_a(const ConvolutionContext& ctx,
                                   const cellfft_param_t& p,
                                   const std::string& file_name)
{
    uint32_t kid = START_R2C + (p.id << 4) + p.any - 1;
    if((p.pad_l | p.pad_t) != 0)
    {
        kid = START_R2C_PAD + (p.id << 4) + p.pad_t;
    }
    const uint32_t r = (p.m + 15) >> 4;
    const size_t bdx = p.id == 0 ? 256 : 512;
    const size_t gdx = p.dir != 2 ? p.k : r;
    const size_t gdy = p.dir != 2 ? r : p.k;
    const std::vector<size_t> block{bdx, 1, 1};
    const std::vector<size_t> grid{gdx * bdx, gdy, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, file_name, g_knames[kid]};
}
static KernelInfo get_kernel_r2c_b(const ConvolutionContext& ctx,
                                   const cellfft_param_t& p,
                                   const std::string& file_name)
{
    uint32_t kid = START_R2C + (p.dir != 1 ? 0 : 48) + (p.id << 4) + p.bny - 1;
    if((p.bnx == p.bny) && ((p.bnx == 3) || (p.bnx == 5)))
    {
        kid = START_R2C_S + ((p.id << 2) | ((p.dir & 1) << 1) | (p.bnx == 3 ? 0 : 1));
    }
    const uint32_t r = (p.n + 15) >> 4;
    const size_t bdx = p.id == 0 ? 256 : 512;
    const size_t gdx = p.dir == 0 ? p.k : r;
    const size_t gdy = p.dir == 0 ? r : p.k;
    const std::vector<size_t> block{bdx, 1, 1};
    const std::vector<size_t> grid{gdx * bdx, gdy, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, file_name, g_knames[kid]};
}
static KernelInfo get_kernel_r2c_grid(const ConvolutionContext& ctx,
                                      const cellfft_param_t& p,
                                      const std::string& file_name)
{
    const uint32_t nx  = p.tile_x * p.grid_x + p.bnx - 1;
    const uint32_t ny  = p.tile_y * p.grid_y + p.bny - 1;
    const uint32_t ex  = ((nx != p.anx) || (ny != p.any)) ? 1 : 0;
    const uint32_t kid = START_R2C_X + p.id * 3 + ((p.pad_l | p.pad_t) != 0 ? 2 : ex);
    const size_t bdx   = p.id == 0 ? 256 : 512;
    const size_t gdx   = (p.m + 15) >> 4;
    const std::vector<size_t> block{bdx, 1, 1};
    const std::vector<size_t> grid{gdx * bdx, p.k, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, file_name, g_knames[kid]};
}
static KernelInfo get_kernel_r2c_xgrad_a(const ConvolutionContext& ctx,
                                         const cellfft_param_t& p,
                                         const std::string& file_name)
{
    const uint32_t nx  = p.tile_x * p.grid_x + p.bnx - 1;
    const uint32_t ny  = p.tile_y * p.grid_y + p.bny - 1;
    const uint32_t ex  = ((nx != p.anx) || (ny != p.any)) ? 1 : 0;
    const uint32_t kid = START_R2C_Xg + (p.id << 2) + ((p.pad_l | p.pad_t) != 0 ? 2 : ex);
    const size_t bdx   = p.id == 0 ? 256 : 512;
    const size_t gdx   = (p.m + 15) >> 4;
    const std::vector<size_t> block{bdx, 1, 1};
    const std::vector<size_t> grid{gdx * bdx, p.k, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, file_name, g_knames[kid]};
}
static KernelInfo get_kernel_r2c_xgrad_b(const ConvolutionContext& ctx,
                                         const cellfft_param_t& p,
                                         const std::string& file_name)
{
    const uint32_t kid = START_R2C_Xg + ((p.id << 2) | 3);
    const size_t bdx   = p.id == 0 ? 256 : 512;
    const size_t gdx   = (p.n + 15) >> 4;
    const std::vector<size_t> block{bdx, 1, 1};
    const std::vector<size_t> grid{gdx * bdx, p.k, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, file_name, g_knames[kid]};
}
static KernelInfo get_kernel_c2r(const ConvolutionContext& ctx,
                                 const cellfft_param_t& p,
                                 const std::string& file_name,
                                 uint32_t relu)
{
    const uint32_t shx = 4 - p.id;
    const uint32_t kid = START_C2R + (p.id << 5) + (relu << (p.id + 4)) + p.cny - 1;
    const size_t gdx   = (p.m + (1 << shx) - 1) >> shx;
    const std::vector<size_t> block{256, 1, 1};
    const std::vector<size_t> grid{gdx << 8, p.n, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, file_name, g_knames[kid]};
}
static KernelInfo get_kernel_c2r_grid(const ConvolutionContext& ctx,
                                      const cellfft_param_t& p,
                                      const std::string& file_name,
                                      uint32_t relu)
{
    const uint32_t shx = 4 - p.id;
    const uint32_t kid = START_C2R_X + ((p.id << 1) | relu);
    const size_t gdx   = (p.m + (1 << shx) - 1) >> shx;
    const std::vector<size_t> block{256, 1, 1};
    const std::vector<size_t> grid{gdx << 8, p.n, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, file_name, g_knames[kid]};
}
static KernelInfo get_kernel_c2r_grad(const ConvolutionContext& ctx,
                                      const cellfft_param_t& p,
                                      const std::string& file_name)
{
    uint32_t nmax = p.cnx > p.cny ? p.cnx : p.cny;
    uint32_t nmin = p.cnx > p.cny ? p.cny : p.cnx;
    uint32_t shx  = 4 - p.id;
    uint32_t kid  = START_C2R + (p.id << 5) + p.cny - 1;
    bool cc0      = (p.cnx == p.cny) && ((p.cnx == 3) || (p.cnx == 5) || (p.cnx == 7));
    bool cc1      = (nmin == 1) && ((nmax & 1) != 0) && (nmax > 1) && (nmax <= 9);
    if(cc0 || cc1)
    {
        kid = START_C2R_S + 11 * p.id + (cc0 ? 8 : (p.cnx > p.cny ? 0 : 4)) + (nmax >> 1) - 1;
    }
    size_t gdx = (p.m + (1 << shx) - 1) >> shx;
    const std::vector<size_t> block{256, 1, 1};
    const std::vector<size_t> grid{gdx << 8, p.n, 1};
    std::ostringstream options;
    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);
    return KernelInfo{options.str(), block, grid, file_name, g_knames[kid]};
}
static void get_solution(const ConvolutionContext& ctx, ConvSolution& sol, const cellfft_param_t& p)
{
    const std::string file_name = "cellfft_" + ctx.GetStream().GetDeviceName() + ".s";
    sol.construction_params.push_back(get_kernel_cgemm(ctx, p, file_name));
    if(p.dir != 2)
    {
        if((p.grid_x | p.grid_y) > 1)
        {
            sol.construction_params.push_back(get_kernel_r2c_grid(ctx, p, file_name));
            sol.construction_params.push_back(get_kernel_r2c_b(ctx, p, file_name));
            sol.construction_params.push_back(get_kernel_c2r_grid(ctx, p, file_name, 0));
        }
        else
        {
            sol.construction_params.push_back(get_kernel_r2c_a(ctx, p, file_name));
            sol.construction_params.push_back(get_kernel_r2c_b(ctx, p, file_name));
            sol.construction_params.push_back(get_kernel_c2r(ctx, p, file_name, 0));
        }
    }
    else
    {
        if((p.grid_x | p.grid_y) > 1)
        {
            sol.construction_params.push_back(get_kernel_r2c_xgrad_a(ctx, p, file_name));
            sol.construction_params.push_back(get_kernel_r2c_xgrad_b(ctx, p, file_name));
        }
        else
        {
            sol.construction_params.push_back(get_kernel_r2c_a(ctx, p, file_name));
            sol.construction_params.push_back(get_kernel_r2c_b(ctx, p, file_name));
        }
        sol.construction_params.push_back(get_kernel_c2r_grad(ctx, p, file_name));
    }
}
bool ConvCellfft::IsApplicable(const ConvolutionContext& ctx) const
{
    if(MIOPEN_BACKEND_OPENCL)
        return false;
    const auto name = ctx.GetStream().GetDeviceName();
    if(name != "gfx900" && name != "gfx906")
        return false;
    if((ctx.kernel_stride_w | ctx.kernel_stride_h | ctx.kernel_dilation_w | ctx.kernel_dilation_h |
        ctx.group_counts) != 1)
        return false;
    if(!ctx.direction.IsForward())
    {
        int pu = ctx.kernel_size_w - ctx.pad_w - 1;
        int pv = ctx.kernel_size_h - ctx.pad_h - 1;
        if((pu < 0) || (pv < 0))
            return false;
    }
    return (ctx.Is2d() && ctx.IsFp32() && (ctx.in_layout == "NCHW") && (ctx.bias == 0));
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
    sol.workspce_sz = get_auxbuf_size(params);
    get_solution(ctx, sol, params);
    if(!ctx.direction.IsBackwardWrW())
    {
        sol.invoker_factory = conv::MakeCellfftInvokerFactory(params, 1.f);
    }
    else
    {
        sol.invoker_factory = conv::MakeCellfftInvokerFactoryGrad(params, 1.f);
    }
    return sol;
}
} // namespace solver
} // namespace miopen
