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

namespace miopen {
namespace solver {

bool ConvOclBwdWrW53::IsApplicable(const ConvolutionContext& params) const
{
    return ((params.kernel_size0 >= 2 || params.kernel_size1 >= 2) &&
            (params.kernel_stride1 == 1 && params.kernel_stride0 == 1));
}

ConvSolution ConvOclBwdWrW53::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;

    size_t localMemSize = 64 * 1024;

    const auto hw_wave_sz       = 64;
    const auto dev_local_mem_sz = localMemSize; // in bytes
                                                // major parameters

    // inpout are outputs
    int wei_cstride = params.kernel_size0 * params.kernel_size1;
    int wei_bstride = params.n_outputs * wei_cstride;

    static const int read_unit = 4;
    static const std::string READ_TYPE =
        (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((read_unit));

    // number  of batch iterations
    result.n_stacks = 1;
    result.n_stacks = std::min(params.batch_sz, result.n_stacks);
    // defines how to proceed : 1 grouop per batch or with a loop over all batches
    // loop over al batches make sense in 2 cases: a lot of small inputs/outputs or few batches
    // param
    int N_BATCH_LOOPS = (params.n_inputs * params.n_outputs <= 8 * 1024)
                            ? 1
                            : (params.batch_sz <= 16 || params.in_width <= 32)
                                  ? (params.batch_sz / result.n_stacks)
                                  : 4;
    int n_batch_blks =
        (params.batch_sz + N_BATCH_LOOPS * result.n_stacks - 1) / (N_BATCH_LOOPS * result.n_stacks);
    if(params.n_passes)
    {
        result.passes = (n_batch_blks > 1) ? 2 : 1;
        return result;
    }

    result.out_pix_tile0 = params.kernel_size0;
    result.out_pix_tile1 = params.kernel_size1;
    result.in_tile1      = 1;

    // span size
    // param
    result.in_tile0 = ((result.out_pix_tile0 * result.out_pix_tile1) <= 16 && (params.in_width > 8))
                          ? 4
                          : ((params.in_width / 3) * 3 == params.in_width) ? 3 : 2;
    int n_spans = (params.in_width + result.in_tile0 - 1) / result.in_tile0;

    // n of wavefronts per group
    // param
    int n_waves = ((result.out_pix_tile0 * result.out_pix_tile1) <= 16 && (params.in_width > 8))
                      ? 4
                      : (params.in_width <= 16) ? 1 : 2;
    int GRP_SZ = hw_wave_sz * n_waves;

    result.n_out_pix_tiles = 1;
    int n_out_stacks       = std::min(params.n_inputs, std::max(1, GRP_SZ / n_spans));
    // number of input maps per group

    result.n_in_data_tiles =
        (params.in_width <= 32 && (result.out_pix_tile0 * result.out_pix_tile1) <= 16) ? 4 : 1;

    result.n_in_data_tiles = std::min(result.n_in_data_tiles, params.n_outputs);
    // calculate number of input scans in the input block
    // max LDS size is 8K
    int in_lcl_width =
        ((params.in_width + read_unit - 1) / read_unit) * read_unit + 2 * params.pad0;
    // number of input map blocks being process at once
    // param
    int in_n_vert_reads = (params.out_height > 32 && params.out_width <= 64 &&
                           (result.out_pix_tile0 * result.out_pix_tile1) <= 16)
                              ? (params.out_height + 1) / 2
                              : params.out_height;
    while(in_lcl_width * in_n_vert_reads * result.n_in_data_tiles >
          (dev_local_mem_sz / (2 * sizeof(float))))
    {
        in_n_vert_reads = (in_n_vert_reads + 1) / 2;
        if(in_n_vert_reads < 2 && result.n_in_data_tiles >= 2)
        {
            in_n_vert_reads = params.in_height;
            result.n_in_data_tiles /= 2;
        }
        else if(in_n_vert_reads < 2)
        {
            std::cout << "CONFIG ERROR: not enough local memory for the configuration\n";
            return ConvSolution(static_cast<miopenStatus_t>(-1));
        }
    }
    int in_n_vert_read_loops = (params.in_height + in_n_vert_reads - 1) / in_n_vert_reads;

    int ALIGNED_OUT_SCAN_LN = ((params.in_width + read_unit - 1) / read_unit); // image aligned scan

    // select output mapping
    int total_out_maps = result.n_out_pix_tiles * n_out_stacks;

    total_out_maps = (total_out_maps > params.n_inputs) ? params.n_inputs : total_out_maps;

    result.grp_tile0 = GRP_SZ;
    result.grp_tile1 = 1;
    int grp_tile2    = 1;

    // utility parameters
    int n_ut_waves = 4;
    int UT_GRP_SZ0 = hw_wave_sz * n_ut_waves;
    int ut_read_unit =
        ((wei_cstride / 4) * 4 == wei_cstride) ? 4 : ((wei_cstride / 2) * 2 == wei_cstride) ? 2 : 1;
    std::string UT_READ_TYPE =
        (ut_read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((ut_read_unit));

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
        + std::string(" -DMLO_OUT_STACKS=") + std::to_string(n_out_stacks) +
        std::string(" -DMLO_N_WAVES=") + std::to_string(n_waves) +
        std::string(" -DMLO_READ_TYPE=") + READ_TYPE + std::string(" -DMLO_READ_UNIT=") +
        std::to_string(read_unit) + std::string(" -DMLO_ALIGNED_OUT_SCAN_LN=") +
        std::to_string(ALIGNED_OUT_SCAN_LN) // image aligned scan
        + std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(hw_wave_sz) +
        std::string(" -DMLO_LG2_PHYS_WAVE_SZ=") + std::to_string(mloLg2(hw_wave_sz)) +
        std::string(" -DMLO_IN_EXTENT1=") + std::to_string(in_n_vert_reads) +
        std::string(" -DMLO_IN_N_VERT_LOOPS=") + std::to_string(in_n_vert_read_loops)

        + std::string(" -DMLO_CONV_BIAS=") + std::to_string(params.bias)

        + std::string(" -DMLO_UT_READ_TYPE=") + UT_READ_TYPE + std::string(" -DMLO_UT_READ_UNIT=") +
        std::to_string(ut_read_unit)

        + std::string(" -DMLO_UT_GRP_SZ0=") + std::to_string(UT_GRP_SZ0)

        //		+ std::string(" -limit-vector-registers=64 ")
        + params.general_compile_options;

    // wrt to W
    {
        KernelInfo kernel;

        kernel.l_wk.push_back(result.grp_tile0);
        kernel.l_wk.push_back(result.grp_tile1);
        kernel.l_wk.push_back(grp_tile2);
        // input is output

        size_t gbl_wk0 =
            GRP_SZ * ((params.n_outputs + result.n_in_data_tiles - 1) / result.n_in_data_tiles);
        size_t gbl_wk1 = ((params.n_inputs + total_out_maps - 1) / total_out_maps);
        size_t gbl_wk2 = n_batch_blks;

        kernel.g_wk.push_back(gbl_wk0);
        kernel.g_wk.push_back(gbl_wk1);
        kernel.g_wk.push_back(gbl_wk2);

        kernel.kernel_file =
            (params.kernel_size0 == 5 && params.kernel_size1 == 5 && in_n_vert_read_loops == 1)
                ? "MIOpenConvBwdWrW_LxG_5x5.cl"
                : "MIOpenConvBwdWrW_LxG_P53.cl";
        kernel.kernel_name  = "MIOpenCvBwdWrW";
        kernel.comp_options = comp_options;

        result.construction_params.push_back(kernel);
        result.workspce_sz = 0;
    }

    // sum over batch
    if(n_batch_blks > 1)
    {
        KernelInfo kernel;

        kernel.kernel_file =
            (params.kernel_size0 == 5 && params.kernel_size1 == 5 && in_n_vert_read_loops == 1)
                ? "MIOpenConvBwdWrW_LxG_5x5.cl"
                : "MIOpenConvBwdWrW_LxG_P53.cl";
        kernel.kernel_name  = "MIOpenCvBwdWrW_rdc";
        kernel.comp_options = comp_options;

        kernel.l_wk.push_back(UT_GRP_SZ0);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);

        int gbl_ut_wk0 = wei_bstride * params.n_inputs / ut_read_unit;

        kernel.g_wk.push_back(gbl_ut_wk0);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        int data_len = (params.out_data_type == "FP32" ? 4 : 8);

        result.construction_params.push_back(kernel);
        result.workspce_sz = wei_bstride * params.n_inputs * n_batch_blks * data_len;
    }
    return result;
}
} // namespace solver
} // namespace miopen
