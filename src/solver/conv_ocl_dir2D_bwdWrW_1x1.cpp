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

#define TWO_PASSES 1

namespace miopen {
namespace solver {

bool ConvOclBwdWrW1x1::IsApplicable(const ConvolutionContext& params) const
{
    bool result = (params.kernel_size0 == 1) && (params.kernel_size1 == 1);

    // Does not support C, K != 16X yet  Still in to-do-list
    if(/*(params.kernel_stride0 > 1 || params.kernel_stride1 > 1) &&*/
       ((params.n_inputs & 0xF) > 0 || (params.n_outputs & 0xF) > 0))
        result = false;

    return result;
}

ConvSolution ConvOclBwdWrW1x1::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
#if TWO_PASSES
    if((params.batch_sz >= 16 || 2 * params.n_outputs > params.n_inputs) && params.pad1 == 0 &&
       params.pad0 == 0 && (params.kernel_stride0 > 1 || params.kernel_stride1 > 1))
    {

        result.passes = 2;
    }
    else
#endif
    {
        result.passes = 1;
    }

    // FIX ME! FIX ME! FIX ME! Does not support C, K != 16X yet
    // NON-Stride/PAD mode NON-16X will be supported by MIOpenConvBwdWrW1x1.CL
    if((params.n_inputs & 0xF) == 0 && (params.n_outputs & 0xF) == 0)
    {
        // params.n_inputs==> C
        // params.n_outputs==>K
        // Jian: following kernel uses C as input, K as output, different from original definition
        // FIX ME! FIX ME! FIX ME!
        // JIANYANG: not know the meaning of following ==>
        result.n_stacks      = 1;
        result.n_stacks      = std::min(params.batch_sz, result.n_stacks);
        result.out_pix_tile0 = 1;
        result.out_pix_tile1 = 1;
        result.in_tile1      = 1;
        result.in_tile0      = 1;
        // JIANYANG: not know the meaning of above <==

        // 8/16/64
        int n_lcl_in_maps = 8;

        /*if(4 *((params.n_outputs+63)/64) * ((params.n_inputs+63)/64) >=512)
        {
                n_lcl_in_maps =64;
        }
        else
        */
        if(4 * ((params.n_outputs + 15) / 16) * ((params.n_inputs + 15) / 16) >= 512)
        {
            n_lcl_in_maps = 16;
        }

        // 8/16/64
        int n_lcl_out_maps = n_lcl_in_maps;

        int n_grp_size0 = 64;

        int n_out_blocks = ((params.n_inputs + n_lcl_out_maps - 1) / n_lcl_out_maps);
        int n_in_blocks  = ((params.n_outputs + n_lcl_in_maps - 1) / n_lcl_in_maps);
        int total_waves  = n_in_blocks * n_out_blocks;

        result.n_out_pix_tiles = n_lcl_out_maps;
        result.n_in_data_tiles = n_lcl_in_maps;

        if(total_waves < 512) // force 64 threads to see what happened
        {
            n_grp_size0 = 256;
        }

        int n_load_dwords_per_map_once = 64;
        if(n_lcl_out_maps == 16 || n_lcl_out_maps == 64)
            n_load_dwords_per_map_once = 16;

        result.grp_tile0 = n_grp_size0;
        result.grp_tile1 = 1;

        // workload and Kernel name

        /*#if 0//nef ML_OPEN_RUNNING
        // W 28 x H 28 x C 512 x K 256 X N 16
        //#define MLO_GRP_SZ
        #define MLO_GRP_SZ0 256
        #define MLO_GRP_SZ1  1
        #define MLO_GRP_SZ2  1
        #define MLO_FILTER_SIZE0    1
        #define MLO_FILTER_SIZE1    1
        #define MLO_FILTER_PAD0     0
        #define MLO_FILTER_PAD1     0
        #define MLO_FILTER_STRIDE0  2
        #define MLO_FILTER_STRIDE1  2
        #define STRIDE_W            1
        #define STRIDE_H            1
        #define MLO_N_OUTPUTS       256
        #define MLO_N_INPUTS        512
        #define MLO_BATCH_SZ        16
        #define MLO_IN_WIDTH            28
        #define MLO_IN_HEIGHT           28
        #define MLO_OUT_WIDTH           14
        #define MLO_OUT_HEIGHT          14
        //64x64 16x16 ==> 16, 8x8 ==> 64
        #define MLO_N_LOAD_DWORDS_PER_MAP_ONCE 64
        #define MLO_N_LCL_IN_MAPS        8
        #define MLO_N_LCL_OUT_MAPS       8

        #define MLO_READ_UNIT          4

        #define MLO_OUT_BATCH_STRIDE   (MLO_OUT_WIDTH*MLO_OUT_HEIGHT*MLO_N_OUTPUTS)
        #define MLO_OUT_CHANNEL_STRIDE (MLO_OUT_WIDTH*MLO_OUT_WIDTH)

        #define MLO_IN_BATCH_STRIDE    (MLO_IN_WIDTH*MLO_IN_HEIGHT* MLO_N_INPUTS)
        #define MLO_IN_CHANNEL_STRIDE  (MLO_IN_WIDTH*MLO_IN_HEIGHT)
        #define MLO_WEI_BATCH_STRIDE   (MLO_N_INPUTS*MLO_N_OUTPUTS)
        #define MLO_WEI_CHANNEL_STRIDE (1*1*MLO_N_INPUTS)
        #define MLO_MAX_LOADS     ((MLO_OUT_CHANNEL_STRIDE / MLO_READ_UNIT) * MLO_BATCH_SZ)

        #define MLO_ACCUM_SZ      ( MLO_N_LCL_IN_MAPS * MLO_N_LCL_OUT_MAPS)
        #define MLO_OUT_READ_SZ    (N_LCL_OUT_MAPS * MLO_READ_UNIT)
        #define MLO_IN_READ_SZ     (MLO_N_LCL_IN_MAPS * MLO_READ_UNIT)

        #define MLO_OUT_CHANNEL_READ_SZ (MLO_OUT_CHANNEL_STRIDE/MLO_READ_UNIT)
        #define MLO_N_IN_TILE_BLOCK  4
        #endif*/

        int read_unit = 4;
        // subsampled input
        int in_width  = (result.passes > 1) ? params.in_width : params.out_width;
        int in_height = (result.passes > 1) ? params.in_height : params.out_height;
        int in_stride = (result.passes > 1) ? params.in_stride : params.out_stride;
        int in_channel_stride =
            (result.passes > 1) ? in_stride * in_height : params.out_channel_stride;
        int in_batch_stride =
            (result.passes > 1) ? in_channel_stride * params.n_outputs : params.out_batch_stride;
        int out_batch_stride   = params.in_batch_stride;
        int out_channel_stride = params.in_channel_stride;
        int out_stride         = params.in_stride;
        int wei_batch_stride =
            params.n_inputs * params.n_outputs * params.kernel_size0 * params.kernel_size1;
        int wei_channel_stride     = params.n_outputs * params.kernel_size0 * params.kernel_size1;
        int max_loads_per_readunit = (out_channel_stride / read_unit) * params.batch_sz;

        // limited shape size shows better performance with ead_uint == 3
        /*
        if( (out_channel_stride % 3) == 1)
        {
                read_unit              = 3;
                max_loads_per_readunit = (out_channel_stride / read_unit) * params.batch_sz;
        }
        */

        int out_pad_min_x  = 0;
        int out_pad_min_y  = 0;
        int out_pad_width  = params.in_width;
        int out_pad_height = params.in_height;

        int in_pad_min_x = 0;
        int in_pad_min_y = 0;

        if(params.pad0 > 0)
        {
            in_pad_min_x = params.kernel_stride0 - (params.pad0 % params.kernel_stride0);
            // In case PAD == STRIDE
            in_pad_min_x = in_pad_min_x % params.kernel_stride0;

            out_pad_min_x = (params.pad0 + params.kernel_stride0 - 1) / params.kernel_stride0;
            out_pad_width = (params.out_width - in_pad_min_x + params.kernel_stride0 - 1) /
                            params.kernel_stride0;
        }
        if(params.pad1 > 0)
        {
            in_pad_min_y = params.kernel_stride1 - (params.pad1 % params.kernel_stride1);
            // In case PAD == STRIDE
            in_pad_min_y = in_pad_min_y % params.kernel_stride1;

            out_pad_min_y  = (params.pad1 + params.kernel_stride1 - 1) / params.kernel_stride1;
            out_pad_height = (params.out_height - in_pad_min_y + params.kernel_stride1 - 1) /
                             params.kernel_stride1;
        }

        if(params.pad0 > 0 || params.pad1 > 0 ||
           (result.passes == 1 && (params.kernel_stride0 > 1 || params.kernel_stride1 > 1)))
        {
            read_unit = (out_pad_width % 4 == 0) ? 4 : (out_pad_width % 3 == 0)
                                                           ? 3
                                                           : (out_pad_width % 2 == 0) ? 2 : 1;
            // read_unit = (out_pad_width % 7 == 0) ? 7 : (out_pad_width % 5 == 0) ? 5 :
            // (out_pad_width % 4 == 0) ? 4 : (out_pad_width % 3 == 0) ? 3 : (out_pad_width % 2
            // == 0) ? 2 : 1;
            max_loads_per_readunit = (out_pad_width / read_unit) * out_pad_height * params.batch_sz;
        }

        int kernel_stride0 = params.kernel_stride0;
        int kernel_stride1 = params.kernel_stride1;

        if(result.passes > 1 && params.pad1 == 0 && params.pad0 == 0 &&
           (params.kernel_stride0 > 1 || params.kernel_stride1 > 1))
        {
            kernel_stride0 = 1;
            kernel_stride1 = 1;
        }

        int out_read_sz         = n_lcl_out_maps * read_unit;
        int in_read_sz          = n_lcl_in_maps * read_unit;
        int out_channel_read_sz = out_channel_stride / read_unit;
        int n_in_tile_block     = 8;
        int n_lcl_out_map_once  = 8;
        int n_lcl_in_map_once   = 8;
        int accum_sz            = n_lcl_out_map_once * n_lcl_in_map_once;

        int write_unit = (out_pad_width % 4 == 0) ? 4 : (out_pad_width % 3 == 0)
                                                            ? 3
                                                            : (out_pad_width % 2 == 0) ? 2 : 1;
        int n_grp0_size0 = 256;
        // real input strides
        int in0_stride         = params.out_stride;
        int in0_channel_stride = params.out_channel_stride;
        int in0_batch_stride   = params.out_batch_stride;
        int kernel0_stride0    = params.kernel_stride0;
        int kernel0_stride1    = params.kernel_stride1;

        if(params.n_passes)
        {

            return result;
        }

        const auto comp_options =
            std::string(" -DMLO_GRP_SZ0=") + std::to_string(n_grp_size0) +
            std::string(" -DMLO_GRP_SZ1=1 ") + std::string(" -DMLO_GRP_SZ2=1 ") +
            std::string(" -DMLO_GRP0_SZ0=") + std::to_string(n_grp0_size0) +
            std::string(" -DMLO_GRP0_SZ1=1 ") + std::string(" -DMLO_GRP0_SZ2=1 ") +
            std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(params.kernel_size0) +
            std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(params.kernel_size1) +
            std::string(" -DMLO_FILTER_PAD0=") + std::to_string(params.pad0) +
            std::string(" -DMLO_FILTER_PAD1=") + std::to_string(params.pad1) +
            std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string(kernel_stride0) +
            std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string(kernel_stride1) +
            std::string(" -DMLO_FILTER0_STRIDE0=") + std::to_string(kernel0_stride0) +
            std::string(" -DMLO_FILTER0_STRIDE1=") + std::to_string(kernel0_stride1) +
            std::string(" -DMLO_N_OUTPUTS=") + std::to_string(params.n_inputs) +
            std::string(" -DMLO_N_INPUTS=") + std::to_string(params.n_outputs) +
            std::string(" -DMLO_BATCH_SZ=") + std::to_string(params.batch_sz) +
            std::string(" -DMLO_IN_WIDTH=") + std::to_string(in_width) +
            std::string(" -DMLO_IN_HEIGHT=") + std::to_string(in_height) +
            std::string(" -DMLO_OUT_WIDTH=") + std::to_string(params.in_width) +
            std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(params.in_height) +
            std::string(" -DMLO_N_LOAD_DWORDS_PER_MAP_ONCE=") +
            std::to_string(n_load_dwords_per_map_once) + std::string(" -DMLO_N_LCL_IN_MAPS=") +
            std::to_string(n_lcl_in_maps) + std::string(" -DMLO_N_LCL_OUT_MAPS=") +
            std::to_string(n_lcl_out_maps) + std::string(" -DMLO_READ_UNIT=") +
            std::to_string(read_unit) + std::string(" -DMLO_WRITE_UNIT=") +
            std::to_string(write_unit) + std::string(" -DMLO_OUT_BATCH_STRIDE=") +
            std::to_string(out_batch_stride) + std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
            std::to_string(out_channel_stride) + std::string(" -DMLO_OUT_STRIDE=") +
            std::to_string(out_stride) + std::string(" -DMLO_IN_BATCH_STRIDE=") +
            std::to_string(in_batch_stride) + std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
            std::to_string(in_channel_stride) + std::string(" -DMLO_IN_STRIDE=") +
            std::to_string(in_stride) + std::string(" -DMLO_IN0_BATCH_STRIDE=") +
            std::to_string(in0_batch_stride) + std::string(" -DMLO_IN0_CHANNEL_STRIDE=") +
            std::to_string(in0_channel_stride) + std::string(" -DMLO_IN0_STRIDE=") +
            std::to_string(in0_stride) + std::string(" -DMLO_WEI_BATCH_STRIDE=") +
            std::to_string(wei_batch_stride) + std::string(" -DMLO_WEI_CHANNEL_STRIDE=") +
            std::to_string(wei_channel_stride) + std::string(" -DMLO_MAX_LOADS=") +
            std::to_string(max_loads_per_readunit) + std::string(" -DMLO_ACCUM_SZ=") +
            std::to_string(accum_sz) + std::string(" -DMLO_OUT_READ_SZ=") +
            std::to_string(out_read_sz) + std::string(" -DMLO_IN_READ_SZ=") +
            std::to_string(in_read_sz) + std::string(" -DMLO_OUT_CHANNEL_READ_SZ=") +
            std::to_string(out_channel_read_sz) + std::string(" -DMLO_N_IN_TILE_BLOCK=") +
            std::to_string(n_in_tile_block) + std::string(" -DMLO_N_LCL_OUT_MAPS_ONCE=") +
            std::to_string(n_lcl_out_map_once) + std::string(" -DMLO_N_LCL_IN_MAPS_ONCE=") +
            std::to_string(n_lcl_in_map_once) + std::string(" -DMLO_IN_PAD_MIN_X=") +
            std::to_string(in_pad_min_x) + std::string(" -DMLO_IN_PAD_MIN_Y=") +
            std::to_string(in_pad_min_y) + std::string(" -DMLO_OUT_PAD_MIN_X=") +
            std::to_string(out_pad_min_x) + std::string(" -DMLO_OUT_PAD_MIN_Y=") +
            std::to_string(out_pad_min_y) + std::string(" -DMLO_OUT_PAD_WIDTH=") +
            std::to_string(out_pad_width) + std::string(" -DMLO_OUT_PAD_HEIGHT=") +
            std::to_string(out_pad_height) + std::string(" -DMLO_TWO_PASSES=") +
            std::to_string((result.passes == 1) ? 0 : 1) + params.general_compile_options;

        result.workspce_sz = 0;

        if(result.passes > 1 && params.pad1 == 0 && params.pad0 == 0 &&
           (params.kernel_stride0 > 1 || params.kernel_stride1 > 1))
        {
            KernelInfo kernel;

            kernel.l_wk.push_back(n_grp0_size0);
            kernel.l_wk.push_back(1);
            kernel.l_wk.push_back(1);
            // output is number of subsampled input maps
            size_t gbl_wk0 = (in_batch_stride / write_unit);
            size_t gbl_wk1 = params.batch_sz;
            size_t gbl_wk2 = 1;

            kernel.g_wk.push_back(gbl_wk0);
            kernel.g_wk.push_back(gbl_wk1);
            kernel.g_wk.push_back(gbl_wk2);

            kernel.kernel_file = "MIOpenConvBwdWrW1x1_PAD_read4.cl";

            kernel.kernel_name = "MIOpenSubsample";

            kernel.comp_options = comp_options;

            result.construction_params.push_back(kernel);

            result.workspce_sz = in_batch_stride * params.batch_sz * sizeof(float);
        }

        {
            // std::cout << comp_options << std::endl;
            int grp_tile2 = 1;
            KernelInfo kernel;

            kernel.l_wk.push_back(result.grp_tile0);
            kernel.l_wk.push_back(result.grp_tile1);
            kernel.l_wk.push_back(grp_tile2);
            // input is output

            // Traverse Smaller Batch_stride first
            size_t gbl_wk0 = n_grp_size0 * n_out_blocks;
            size_t gbl_wk1 = n_in_blocks;
            size_t gbl_wk2 = 1;

            if(in_batch_stride < out_batch_stride)
            {
                gbl_wk0 = n_grp_size0 * n_in_blocks;
                gbl_wk1 = n_out_blocks;
                gbl_wk2 = 1;
            }

            kernel.g_wk.push_back(gbl_wk0);
            kernel.g_wk.push_back(gbl_wk1);
            kernel.g_wk.push_back(gbl_wk2);

            kernel.kernel_file = "MIOpenConvBwdWrW1x1_PAD_read4.cl";

            kernel.kernel_name = "MIOpenCvBwdWrW_8x8map";
            if(n_lcl_in_maps == 16)
            {
                kernel.kernel_name = "MIOpenCvBwdWrW_16x16map";
            }
            if(n_lcl_in_maps == 8)
            {
                kernel.kernel_name = "MIOpenCvBwdWrW_8x8map";
            }

            // std::cout << kernel.kernel_name << std::endl;

            kernel.comp_options = comp_options;

            result.construction_params.push_back(kernel);
        }

        return result;
    }

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

    if(!params.direction.IsBackwardWrW())
        MIOPEN_THROW("!params.direction.IsBackwardWrW()");
    // it's backward - inputs are outputs and vs versa
    const auto comp_options =
        std::string(" -DMLO_DIR_FORWARD=0") + std::string(" -DMLO_GRP_SZ=") +
        std::to_string(GRP_SZ) + std::string(" -DMLO_GRP_SZ0=") + std::to_string(result.grp_tile0) +
        std::string(" -DMLO_GRP_SZ1=") + std::to_string(result.grp_tile1) +
        std::string(" -DMLO_GRP_SZ2=") + std::to_string(grp_tile2) +
        std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(params.kernel_size0) +
        std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(params.kernel_size1) +
        std::string(" -DMLO_FILTER_PAD0=") + std::to_string(params.pad0) +
        std::string(" -DMLO_FILTER_PAD1=") + std::to_string(params.pad1) +
        std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string(params.kernel_stride0) +
        std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string(params.kernel_stride1) +
        std::string(" -DSTRIDE_W=") + std::to_string(params.kernel_stride0) +
        std::string(" -DSTRIDE_H=") + std::to_string(params.kernel_stride1) +
        std::string(" -DMLO_N_OUTPUTS=") + std::to_string(params.n_inputs) +
        std::string(" -DMLO_N_INPUTS=") + std::to_string(params.n_outputs) +
        std::string(" -DMLO_BATCH_SZ=") + std::to_string(params.batch_sz) +
        std::string(" -DMLO_N_BATCH_LOOPS=") + std::to_string(N_BATCH_LOOPS) +
        std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(params.in_batch_stride) +
        std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(params.in_channel_stride) +
        std::string(" -DMLO_OUT_STRIDE=") + std::to_string(params.in_stride) +
        std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(params.out_batch_stride) +
        std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string(params.out_channel_stride) +
        std::string(" -DMLO_IN_STRIDE=") + std::to_string(params.out_stride) +
        std::string(" -DMLO_WEI_BATCH_STRIDE=") + std::to_string(wei_bstride) +
        std::string(" -DMLO_WEI_CHANNEL_STRIDE=") + std::to_string(wei_cstride) +
        std::string(" -DMLO_IN_WIDTH=") + std::to_string(params.out_width) +
        std::string(" -DMLO_IN_HEIGHT=") + std::to_string(params.out_height) +
        std::string(" -DMLO_OUT_WIDTH=") + std::to_string(params.in_width) +
        std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(params.in_height) +
        std::string(" -DMLO_IN_TILE1=") + std::to_string(result.in_tile1) +
        std::string(" -DMLO_IN_TILE0=") + std::to_string(result.in_tile0) +
        std::string(" -DMLO_N_LCL_BATCHS=") +
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

        //      + std::string(" -limit-vector-registers=64 ")
        + params.general_compile_options;

    // wrt to W
    {
        int n_batch_blks = 1; // (params.batch_sz + N_BATCH_LOOPS * result.n_stacks - 1) /
                              // (N_BATCH_LOOPS * result.n_stacks);
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
