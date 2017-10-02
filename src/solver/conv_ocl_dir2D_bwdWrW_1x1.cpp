/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include "miopen/solver.hpp"

/*
int mlo_construct_BwdWrW2D::mloConstruct1x1Mmap()
{

    int ret = 0;
    size_t localMemSize = 64 * 1024;

    _hw_wave_sz = 64;
    _dev_local_mem_sz = localMemSize; // in bytes
                                      // major parameters

                                      // inpout are outputs
    int wei_cstride = _kernel_size0*_kernel_size1;
    int wei_bstride = _n_outputs*wei_cstride;


    // number  of batch iterations
    _n_stacks = 1;
    _n_stacks = std::min(_batch_sz, _n_stacks);
    // defines how to proceed : 1 grouop per batch or with a loop over all batches
    // loop over al batches make sense in 2 cases: a lot of small inputs/outputs or few batches
    // param
    int N_BATCH_LOOPS = (_n_inputs*_n_outputs <= 8 * 1024) ? 1 : _batch_sz / _n_stacks;
    int n_batch_blks = (_batch_sz + N_BATCH_LOOPS * _n_stacks - 1) / (N_BATCH_LOOPS * _n_stacks);

    _out_pix_tile0 = 1;
    _out_pix_tile1 = 1;
    _in_tile1 = 1;
    _in_tile0 = 1;

    // n of wvaefront in a group
    // param
    int n_waves = (_in_width <= 8) ? 1 : 4;
    int GRP_SZ = _hw_wave_sz * n_waves;
    // number of input maps per group

    int map_sz = _in_width*_in_height;
    // define a special size for a specific width as a devisor to avoid dealing with out of range
    // param
    int read_unit = (_in_width == 7 || _in_width == 14) ? 7 : (_in_width == 28) ? 14 : (((map_sz /
8) * 8) == map_sz) ? 8 : (((map_sz / 4) * 4) == map_sz) ? 4 : (((map_sz / 2) * 2) == map_sz) ? 2 :
1;

    int MAP_WK_SZ = ((map_sz + read_unit - 1) / read_unit);

    // to avoid exeeding the group size but trying to keep multiple of the same unit
    while (MAP_WK_SZ > GRP_SZ)
    {
        read_unit *= 2;
        MAP_WK_SZ = ((map_sz + read_unit - 1) / read_unit);
    }

    // this one is valid only till _FLOAT8
    // but it's not an error, the kernel does not use these types at all
    std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((read_unit));

    int POW2_MAP_WK_SZ = (1 << mloLg2(MAP_WK_SZ));
    // number of output maps fetched into LDS and to be shred with the input mapped kept in
registers
    // param
    int n_out_stacks = (_in_width == 28) ? 10 : ((_in_width == 7) || (_in_width == 14)) ? 8 :
(GRP_SZ / MAP_WK_SZ);
    int lcl_size_limit = (_in_width <= 8) ? n_out_stacks*MAP_WK_SZ*read_unit : _dev_local_mem_sz /
(2 * sizeof(float));

    // not to exeed local memory size
    while ((_in_width > 8) && n_out_stacks*MAP_WK_SZ*read_unit > lcl_size_limit)
    {
        n_out_stacks--;
    }

    // number input maps stacks.
    // n_in_stacks input map wil be written in teh local memory sequentially
    int n_in_stacks = (GRP_SZ / MAP_WK_SZ);

    n_out_stacks = std::min(_n_inputs, n_out_stacks);
    n_in_stacks = std::min(_n_outputs, n_in_stacks);

    // param
    // this is 1 currently
    _n_out_pix_tiles = std::min(1, (_n_inputs + n_out_stacks - 1) / n_out_stacks);

    // number of maps in a stack or number of input read blocks written into 1 wk-item (lane)
    // param
    _n_in_data_tiles = std::min(((_in_width == 28) ? 2 : 4), (_n_outputs + n_in_stacks - 1) /
n_in_stacks);
    // to be able to do an easy final transform and summation
    while ((_in_width > 8) && n_in_stacks*_n_in_data_tiles* n_out_stacks > GRP_SZ)
    {
        n_in_stacks--;
    }

    // total maps per group
    int total_out_maps = _n_out_pix_tiles * n_out_stacks;
    int total_in_maps = _n_in_data_tiles * n_in_stacks;

    _grp_tile0 = GRP_SZ;
    _grp_tile1 = 1;
    int grp_tile2 = 1;


    // utility parameters
    int n_ut_waves = 4;
    int UT_GRP_SZ0 = _hw_wave_sz * n_ut_waves;
    int ut_read_unit = ((wei_cstride / 4) * 4 == wei_cstride) ? 4 : ((wei_cstride / 2) * 2 ==
wei_cstride) ? 2 : 1;
    std::string UT_READ_TYPE = (ut_read_unit == 1) ? "_FLOAT" : "_FLOAT" +
std::to_string((ut_read_unit));


    // it's backward - inputs are outputs and vs versa
    _comp_options =
        std::string(" -DMLO_DIR_FORWARD=") + std::to_string(_direction)
        + std::string(" -DMLO_GRP_SZ=") + std::to_string(GRP_SZ)
        + std::string(" -DMLO_GRP_SZ0=") + std::to_string(_grp_tile0)
        + std::string(" -DMLO_GRP_SZ1=") + std::to_string(_grp_tile1)
        + std::string(" -DMLO_GRP_SZ2=") + std::to_string(grp_tile2)
        + std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(_kernel_size0)
        + std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(_kernel_size1)
        + std::string(" -DMLO_FILTER_PAD0=") + std::to_string(_pad0)
        + std::string(" -DMLO_FILTER_PAD1=") + std::to_string(_pad1)
        + std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string(_kernel_stride0)
        + std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string(_kernel_stride1)
        + std::string(" -DSTRIDE_W=") + std::to_string(_kernel_stride0)
        + std::string(" -DSTRIDE_H=") + std::to_string(_kernel_stride1)
        + std::string(" -DMLO_N_OUTPUTS=") + std::to_string(_n_inputs)
        + std::string(" -DMLO_N_INPUTS=") + std::to_string(_n_outputs)
        + std::string(" -DMLO_BATCH_SZ=") + std::to_string(_batch_sz)
        + std::string(" -DMLO_N_BATCH_LOOPS=") + std::to_string(N_BATCH_LOOPS)
        + std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(_in_batch_stride)
        + std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(_in_channel_stride)
        + std::string(" -DMLO_OUT_STRIDE=") + std::to_string(_in_stride)
        + std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(_out_batch_stride)
        + std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string(_out_channel_stride)
        + std::string(" -DMLO_IN_STRIDE=") + std::to_string(_out_stride)
        + std::string(" -DMLO_WEI_BATCH_STRIDE=") + std::to_string(wei_bstride)
        + std::string(" -DMLO_WEI_CHANNEL_STRIDE=") + std::to_string(wei_cstride)
        + std::string(" -DMLO_IN_WIDTH=") + std::to_string(_out_width)
        + std::string(" -DMLO_IN_HEIGHT=") + std::to_string(_out_height)
        + std::string(" -DMLO_OUT_WIDTH=") + std::to_string(_in_width)
        + std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(_in_height)
        + std::string(" -DMLO_IN_TILE1=") + std::to_string(_in_tile1)
        + std::string(" -DMLO_IN_TILE0=") + std::to_string(_in_tile0)
        + std::string(" -DMLO_N_LCL_BATCHS=") + std::to_string(_n_stacks) // # of diff stacks (part
of batch).
        + std::string(" -DMLO_N_LCL_OUT_MAPS=") + std::to_string(_n_out_pix_tiles)  // # output
pixel tiles per wk-item (ALU)
        + std::string(" -DMLO_N_LCL_IN_MAPS=") + std::to_string(_n_in_data_tiles) // total # of
blocks of different inputs in LDS
        + std::string(" -DMLO_OUT_TILE0=") + std::to_string(_out_pix_tile0)  // size of ouptput tile
per wk-item (ALU)
        + std::string(" -DMLO_OUT_TILE1=") + std::to_string(_out_pix_tile1)  //
        + std::string(" -DMLO_OUT_STACKS=") + std::to_string(n_out_stacks)
        + std::string(" -DMLO_IN_STACKS=") + std::to_string(n_in_stacks)
        + std::string(" -DMLO_N_WAVES=") + std::to_string(n_waves)
        + std::string(" -DMLO_MAP_WK_SZ=") + std::to_string(MAP_WK_SZ)
        + std::string(" -DMLO_POW2_MAP_WK_SZ=") + std::to_string(POW2_MAP_WK_SZ)
        + std::string(" -DMLO_LCL_MEM_SZ=") + std::to_string(lcl_size_limit)

        + std::string(" -DMLO_READ_TYPE=") + READ_TYPE
        + std::string(" -DMLO_READ_UNIT=") + std::to_string(read_unit)
        + std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(_hw_wave_sz)
        + std::string(" -DMLO_LG2_PHYS_WAVE_SZ=") + std::to_string(mloLg2(_hw_wave_sz))

        + std::string(" -DMLO_CONV_BIAS=") + std::to_string(_bias)

        + std::string(" -DMLO_UT_READ_TYPE=") + UT_READ_TYPE
        + std::string(" -DMLO_UT_READ_UNIT=") + std::to_string(ut_read_unit)

        + std::string(" -DMLO_UT_GRP_SZ0=") + std::to_string(UT_GRP_SZ0)

        //		+ std::string(" -limit-vector-registers=64 ")
        + getGeneralCompOptions()
        ;


    _mlo_kernels_info.clear();
    // wrt to W
    {
        _l_wk.clear();
        _l_wk.push_back(_grp_tile0);
        _l_wk.push_back(_grp_tile1);
        _l_wk.push_back(grp_tile2);
        // input is output

        size_t gbl_wk0 = GRP_SZ * ((_n_inputs + total_out_maps - 1) / total_out_maps);
        size_t gbl_wk1 = ((_n_outputs + total_in_maps - 1) / total_in_maps);
        size_t gbl_wk2 = n_batch_blks;


        _g_wk.clear();
        _g_wk.push_back(gbl_wk0);
        _g_wk.push_back(gbl_wk1);
        _g_wk.push_back(gbl_wk2);

        _kernel_file = "MIOpenConvBwdWrW1x1Mmap.cl";
        _kernel_name = "MIOpenCvBwdWrW";

        auto kern_info = std::make_tuple(_kernel_name, _kernel_file, _comp_options, _g_wk, _l_wk);
        _mlo_kernels_info.push_back(kern_info);

        _workspce_sz = 0;

    }

    // sum over batch
    if (n_batch_blks > 1)
    {


        std::string kernel_file = "MIOpenConvBwdWrW1x1Mmap.cl";
        std::string kernel_name = "MIOpenCvBwdWrW_rdc";

        std::vector<size_t> l_wk;
        l_wk.clear();
        l_wk.push_back(UT_GRP_SZ0);
        l_wk.push_back(1);
        l_wk.push_back(1);

        int gbl_ut_wk0 = wei_bstride * _n_inputs / ut_read_unit;

        std::vector<size_t> g_wk;
        g_wk.push_back(gbl_ut_wk0);
        g_wk.push_back(1);
        g_wk.push_back(1);
        auto kern_info = std::make_tuple(kernel_name, kernel_file, _comp_options, g_wk, l_wk);
        _mlo_kernels_info.push_back(kern_info);

        int data_len = (_out_data_type == "FP32" ? 4 : 8);
        _workspce_sz = wei_bstride * _n_inputs * n_batch_blks * data_len;
    }

    return(ret);
}
*/

namespace miopen {
namespace solver {

bool ConvOclBwdWrW1x1::IsApplicable(const ConvolutionContext& params) const
{
    return (params.kernel_size0 == 1) && (params.kernel_size1 == 1);
}

ConvSolution ConvOclBwdWrW1x1::GetSolution(const ConvolutionContext& params,
                                           const PerformanceConfig&) const
{
    ConvSolution result;
    if(params.n_passes)
    {
        result.passes = 1;
        return result;
    }
#if 0 // MD: Calls old 1x1 kernel (MIOpenConvBwdWrW1x1Mmap.cl) that has been optimized by Stas
        if (params.in_width == 14 && params.in_height == 14 && params.n_inputs == 192 && params.n_outputs == 512)
        {
            return(mloConstruct1x1Mmap());
        }
#endif
    // size_t localMemSize = 64 * 1024;

    const auto hw_wave_sz = 64;
    //_dev_local_mem_sz = localMemSize; // in bytes
    // major parameters

    // inpout are outputs
    int wei_cstride = params.kernel_size0 * params.kernel_size1;
    int wei_bstride = params.n_outputs * wei_cstride;

    // number  of batch iterations
    result.n_stacks = 1;
    result.n_stacks = std::min(params.batch_sz, result.n_stacks);
    // defines how to proceed : 1 grouop per batch or with a loop over all batches
    // loop over al batches make sense in 2 cases: a lot of small inputs/outputs or few batches
    // param
    int N_BATCH_LOOPS = 1; // (params.n_inputs*params.n_outputs <= 8 * 1024) ? 1 : params.batch_sz /
                           // result.n_stacks;
    int n_batch_blks = 1;  // (params.batch_sz + N_BATCH_LOOPS * result.n_stacks - 1) /
                           // (N_BATCH_LOOPS * result.n_stacks);

    result.out_pix_tile0 = 1;
    result.out_pix_tile1 = 1;
    result.in_tile1      = 1;
    result.in_tile0      = 1;

    int map_sz = params.in_width * params.in_height;
    // define a special size for a specific width as a devisor to avoid dealing with out of range
    // param
    int read_unit = (params.in_width < 8)
                        ? params.in_width
                        : (((map_sz / 7) * 7) == map_sz)
                              ? 7
                              : (((map_sz / 8) * 8) == map_sz)
                                    ? 8
                                    : (((map_sz / 5) * 5) == map_sz)
                                          ? 5
                                          : (((map_sz / 6) * 6) == map_sz) ? 6 : 4;

    if(params.in_width * params.in_height > 512)
    {
        read_unit = 4;
    }

    int MAP_WK_SZ  = ((map_sz + read_unit - 1) / read_unit);
    int N_PIXS_OFF = map_sz - (map_sz / read_unit) * read_unit;
    bool large_map = (MAP_WK_SZ > hw_wave_sz * 2);
    // not in use
    bool midsize_map = false; // (MAP_WK_SZ <= _hw_wave_sz * 2);

    // n of wavefronts in a group
    // param
    int n_waves = 4;
    int GRP_SZ  = hw_wave_sz * n_waves;

    // this one is valid only till _FLOAT8
    // but it's not an error, the kernel does not use these types at all
    std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((read_unit));

    int N_out_lcl   = 1;
    int out_lcl_blk = 8 / N_out_lcl;
    while(!large_map && out_lcl_blk > 0 && MAP_WK_SZ * read_unit * out_lcl_blk > n_waves * 1024)
    {
        N_out_lcl <<= 1;
        out_lcl_blk >>= 1;
    }
    // number of output maps
    result.n_out_pix_tiles = std::min(params.n_inputs, N_out_lcl * out_lcl_blk);
    out_lcl_blk =
        (result.n_out_pix_tiles < N_out_lcl * out_lcl_blk) ? result.n_out_pix_tiles : out_lcl_blk;
    N_out_lcl = (result.n_out_pix_tiles < N_out_lcl * out_lcl_blk)
                    ? result.n_out_pix_tiles / out_lcl_blk
                    : N_out_lcl;

    // total maps per group
    int total_out_maps = result.n_out_pix_tiles;

    int n_out_blocks = ((params.n_inputs + total_out_maps - 1) / total_out_maps);

    int N_OUT_MAPS_ALIGNED = (n_out_blocks * total_out_maps == params.n_inputs) ? 1 : 0;

    // number input maps
    // para

    // number of input maps per group
    // large map cover a group in full
    int N_MAPS_PER_GROUP =
        (!large_map) ? std::min(params.n_outputs, std::max(1, GRP_SZ / MAP_WK_SZ)) : 1;

    result.n_in_data_tiles = std::min(params.n_outputs / N_MAPS_PER_GROUP,
                                      ((read_unit > 4) ? 6 : (read_unit > 2) ? 8 : 10));

    result.n_in_data_tiles = (params.n_outputs >= result.n_in_data_tiles * N_MAPS_PER_GROUP)
                                 ? result.n_in_data_tiles
                                 : 1;

    int total_in_maps = result.n_in_data_tiles * N_MAPS_PER_GROUP;

    int n_in_blocks = ((params.n_outputs + total_in_maps - 1) / total_in_maps);

    int N_IN_MAPS_ALIGNED = (n_in_blocks * total_in_maps == params.n_outputs) ? 1 : 0;

    int lcl_comm_size = out_lcl_blk * MAP_WK_SZ * read_unit;
    // reduction loop step

    int accum_sz = result.n_in_data_tiles * result.n_out_pix_tiles;
    int REDUC_LOOP_STEP =
        (accum_sz < MAP_WK_SZ && accum_sz < 8) ? accum_sz : result.n_out_pix_tiles;
    // adjust reduction step
    while((REDUC_LOOP_STEP > MAP_WK_SZ ||
           (accum_sz / REDUC_LOOP_STEP) * REDUC_LOOP_STEP != accum_sz) &&
          REDUC_LOOP_STEP > 1)
    {
        REDUC_LOOP_STEP--;
    }

    // calculate log of summation loop
    int lg2_red_splits = 0;
    int range          = (!large_map) ? MAP_WK_SZ : GRP_SZ;

    for(; ((REDUC_LOOP_STEP << lg2_red_splits) <= range); ++lg2_red_splits)
        ;

    // more than 1 summation areas
    int first_round = 0;
    int can_divide  = 1;
    if(lg2_red_splits > 0)
    {
        // check if MAP_WK_SZ can be devided into that number at te firts round of summation
        int firsts_round_split = (1 << (lg2_red_splits - 1));
        first_round            = (range + firsts_round_split - 1) / firsts_round_split;
        can_divide             = ((first_round * firsts_round_split) == range) ? 1 : 0;
    }

    int lcl_red_size = GRP_SZ * std::max(REDUC_LOOP_STEP, (accum_sz / REDUC_LOOP_STEP));

    int lcl_size_limit =
        (!(large_map || midsize_map)) ? std::max(lcl_comm_size, lcl_red_size) : lcl_red_size;

    result.grp_tile0 = GRP_SZ;
    result.grp_tile1 = 1;
    int grp_tile2    = 1;

    // it's backward - inputs are outputs and vs versa
    const auto comp_options =
        std::string(" -DMLO_DIR_FORWARD=") + std::to_string(params.forward) +
        std::string(" -DMLO_GRP_SZ=") + std::to_string(GRP_SZ) + std::string(" -DMLO_GRP_SZ0=") +
        std::to_string(result.grp_tile0) + std::string(" -DMLO_GRP_SZ1=") +
        std::to_string(result.grp_tile1) + std::string(" -DMLO_GRP_SZ2=") +
        std::to_string(grp_tile2) + std::string(" -DMLO_FILTER_SIZE0=") +
        std::to_string(params.kernel_size0) + std::string(" -DMLO_FILTER_SIZE1=") +
        std::to_string(params.kernel_size1) + std::string(" -DMLO_FILTER_PAD0=") +
        std::to_string(params.pad0) + std::string(" -DMLO_FILTER_PAD1=") +
        std::to_string(params.pad1) + std::string(" -DMLO_FILTER_STRIDE0=") +
        std::to_string(params.kernel_stride0) + std::string(" -DMLO_FILTER_STRIDE1=") +
        std::to_string(params.kernel_stride1) + std::string(" -DSTRIDE_W=") +
        std::to_string(params.kernel_stride0) + std::string(" -DSTRIDE_H=") +
        std::to_string(params.kernel_stride1) + std::string(" -DMLO_N_OUTPUTS=") +
        std::to_string(params.n_inputs) + std::string(" -DMLO_N_INPUTS=") +
        std::to_string(params.n_outputs) + std::string(" -DMLO_BATCH_SZ=") +
        std::to_string(params.batch_sz) + std::string(" -DMLO_N_BATCH_LOOPS=") +
        std::to_string(N_BATCH_LOOPS) + std::string(" -DMLO_OUT_BATCH_STRIDE=") +
        std::to_string(params.in_batch_stride) + std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
        std::to_string(params.in_channel_stride) + std::string(" -DMLO_OUT_STRIDE=") +
        std::to_string(params.in_stride) + std::string(" -DMLO_IN_BATCH_STRIDE=") +
        std::to_string(params.out_batch_stride) + std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
        std::to_string(params.out_channel_stride) + std::string(" -DMLO_IN_STRIDE=") +
        std::to_string(params.out_stride) + std::string(" -DMLO_WEI_BATCH_STRIDE=") +
        std::to_string(wei_bstride) + std::string(" -DMLO_WEI_CHANNEL_STRIDE=") +
        std::to_string(wei_cstride) + std::string(" -DMLO_IN_WIDTH=") +
        std::to_string(params.out_width) + std::string(" -DMLO_IN_HEIGHT=") +
        std::to_string(params.out_height) + std::string(" -DMLO_OUT_WIDTH=") +
        std::to_string(params.in_width) + std::string(" -DMLO_OUT_HEIGHT=") +
        std::to_string(params.in_height) + std::string(" -DMLO_IN_TILE1=") +
        std::to_string(result.in_tile1) + std::string(" -DMLO_IN_TILE0=") +
        std::to_string(result.in_tile0) + std::string(" -DMLO_N_LCL_BATCHS=") +
        std::to_string(result.n_stacks) // # of diff stacks (part of batch).
        + std::string(" -DMLO_N_LCL_OUT_MAPS=") +
        std::to_string(result.n_out_pix_tiles) // # output pixel tiles per wk-item (ALU)
        + std::string(" -DMLO_N_LCL_IN_MAPS=") +
        std::to_string(result.n_in_data_tiles) // total # of blocks of different inputs in LDS
        + std::string(" -DMLO_OUT_TILE0=") +
        std::to_string(result.out_pix_tile0) // size of ouptput tile per wk-item (ALU)
        + std::string(" -DMLO_OUT_TILE1=") + std::to_string(result.out_pix_tile1) //
        + std::string(" -DMLO_N_WAVES=") + std::to_string(n_waves) +
        std::string(" -DMLO_MAP_WK_SZ=") + std::to_string(MAP_WK_SZ) +
        std::string(" -DMLO_N_PIXS_OFF=") + std::to_string(N_PIXS_OFF) +
        std::string(" -DMLO_LCL_MEM_SZ=") + std::to_string(lcl_size_limit) +
        std::string(" -DMLO_N_OUT_MAPS_ALIGNED=") + std::to_string(N_OUT_MAPS_ALIGNED) +
        std::string(" -DMLO_N_IN_MAPS_ALIGNED=") + std::to_string(N_IN_MAPS_ALIGNED) +
        std::string(" -DMLO_REDUC_LOOP_STEP=") + std::to_string(REDUC_LOOP_STEP) +
        std::string(" -DMLO_N_MAPS_PER_GROUP=") + std::to_string(N_MAPS_PER_GROUP) +
        std::string(" -DMLO_N_LCL_OUT=") + std::to_string(N_out_lcl) +
        std::string(" -DMLO_OUT_LCL_BLK=") + std::to_string(out_lcl_blk) +
        std::string(" -DMLO_FIRST_ROUND=") + std::to_string(first_round) +
        std::string(" -DMLO_FIRST_CAN_DIVIDE=") + std::to_string(can_divide) +
        std::string(" -DMLO_LG2_REDUC_ROUNDS=") + std::to_string(lg2_red_splits)

        + std::string(" -DMLO_READ_TYPE=") + READ_TYPE + std::string(" -DMLO_READ_UNIT=") +
        std::to_string(read_unit) + std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(hw_wave_sz) +
        std::string(" -DMLO_LG2_PHYS_WAVE_SZ=") + std::to_string(mloLg2(hw_wave_sz))

        //		+ std::string(" -limit-vector-registers=64 ")
        + params.general_compile_options;

    // wrt to W
    {
        KernelInfo kernel;

        kernel.l_wk.push_back(result.grp_tile0);
        kernel.l_wk.push_back(result.grp_tile1);
        kernel.l_wk.push_back(grp_tile2);
        // input is output

        size_t gbl_wk0 = GRP_SZ * n_out_blocks;
        size_t gbl_wk1 = n_in_blocks;
        size_t gbl_wk2 = (!large_map) ? n_batch_blks : 1;

        kernel.g_wk.push_back(gbl_wk0);
        kernel.g_wk.push_back(gbl_wk1);
        kernel.g_wk.push_back(gbl_wk2);

        kernel.kernel_file  = "MIOpenConvBwdWrW1x1.cl";
        kernel.kernel_name  = (large_map) ? "MLOpenCvBwdWrWLmap" : "MIOpenCvBwdWrWSmap";
        kernel.comp_options = comp_options;

        result.construction_params.push_back(kernel);
        result.workspce_sz = 0;
    }
    return result;
}
} // namespace solver
} // namespace miopen
