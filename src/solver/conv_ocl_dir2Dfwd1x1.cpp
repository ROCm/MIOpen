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
#include "miopen/handle.hpp"

namespace miopen {
namespace solver {

bool ConvOclDirectFwd1x1::IsApplicable(const ConvolutionContext& params) const
{
    return params.kernel_size0 == 1 && params.kernel_size1 == 1;
}

ConvSolution
ConvOclDirectFwd1x1::GetSolution(const ConvolutionContext& params,
                                 const PerformanceConfig& exhaustive_search_result) const
{
    ConvSolution result;
    const auto& searched_params =
        dynamic_cast<const PerformanceConfigImpl&>(exhaustive_search_result);
    searched_params.CopyTo(result);

    if(params.n_outputs % 16 == 0 &&
       params.n_inputs % 16 == 0)
    {
        int version = result.out_pix_tile1;

        if(version == 1)
        {

            uint N_LCL_IN_MAPS = result.n_in_data_tiles;

            int N_LCL_OUT_MAPS = result.n_out_pix_tiles;
            // 0 or 1
            uint CHEAT_SHADER_COMPILER = result.out_pix_tile0;

            int BATCHSIZE = params.batch_sz;
            int W         = params.in_width;
            int H         = params.in_height;
            int C         = params.n_inputs;
            int K         = params.n_outputs;
            int W_out     = params.out_width;
            int H_out     = params.out_height;

            N_LCL_OUT_MAPS = std::min(N_LCL_OUT_MAPS, K);
            if(N_LCL_OUT_MAPS > 32 && (K % N_LCL_OUT_MAPS) != 0)
            {
                N_LCL_OUT_MAPS = 32;
            }
            if(N_LCL_OUT_MAPS > 16 && (K % N_LCL_OUT_MAPS) != 0)
            {
                N_LCL_OUT_MAPS = 16;
            }

            result.n_out_pix_tiles = N_LCL_OUT_MAPS;

            if(N_LCL_IN_MAPS < C && N_LCL_IN_MAPS > 0 && (N_LCL_IN_MAPS % 8) == 0)
            {
                // Pass will do nothing
            }
            else
            {
                N_LCL_IN_MAPS = C;
            }

            result.n_in_data_tiles = N_LCL_IN_MAPS;

            /*
            #define H  28
            #define W 28
            #define C 192
            #define K 64

            #define MLO_IN_HEIGHT              H
            #define MLO_IN_WIDTH               W
            #define MLO_N_INPUTS               C

            //128 or MLO_N_INPUTS
            #define MLO_N_LCL_IN_MAPS          128

            #define MLO_N_OUTPUTS              K

            #define H_out                      H
            #define W_out					   W
            */
            //#define  MLO_N_IN_GROUPS             (( MLO_N_INPUTS + MLO_N_LCL_IN_MAPS - 1) /
            // MLO_N_LCL_IN_MAPS)
            //#define MLO_CLOOP0                   (MLO_N_LCL_IN_MAPS/MLO_N_LCL_IN_MAPS_ONCE )
            //#define  MLO_CLOOP2                  ((MLO_N_INPUTS -
            // MLO_N_LCL_IN_MAPS*(MLO_N_IN_GROUPS-1)) / MLO_N_LCL_IN_MAPS_ONCE )
            //#define MLO_N_LCL_OUT_MAPS           16

            uint N_IN_GROUPS        = (C + N_LCL_IN_MAPS - 1) / N_LCL_IN_MAPS;
            uint N_LCL_IN_MAPS_ONCE = 8;

            if(params.kernel_stride0 > 1 || params.kernel_stride1 > 1)
                N_LCL_IN_MAPS_ONCE = 4;

            uint CLOOP0 = N_LCL_IN_MAPS / N_LCL_IN_MAPS_ONCE;
            uint CLOOP2 = (C - N_LCL_IN_MAPS * (N_IN_GROUPS - 1)) / N_LCL_IN_MAPS_ONCE;

            KernelInfo kernel;

            kernel.comp_options =
                std::string(" -DMLO_N_LCL_IN_MAPS_ONCE=") + std::to_string(N_LCL_IN_MAPS_ONCE) +
                std::string(" -DBATCHSIZE=") + std::to_string(BATCHSIZE) + std::string(" -DH=") +
                std::to_string(H) + std::string(" -DW=") + std::to_string(W) +
                std::string(" -DC=") + std::to_string(C) + std::string(" -DK=") +
                std::to_string(K) + std::string(" -DMLO_N_LCL_IN_MAPS=") +
                std::to_string(N_LCL_IN_MAPS) + std::string(" -DMLO_N_INPUTS=") +
                std::to_string(C) + std::string(" -DMLO_N_OUTPUTS=") + std::to_string(K) +
                std::string(" -DH_out=") + std::to_string(H_out) + std::string(" -DW_out=") +
                std::to_string(W_out) + std::string(" -DMLO_N_IN_GROUPS=") +
                std::to_string(N_IN_GROUPS) + std::string(" -DMLO_CLOOP0=") +
                std::to_string(CLOOP0) + std::string(" -DMLO_CLOOP2=") + std::to_string(CLOOP2) +
                std::string(" -DMLO_N_LCL_OUT_MAPS=") + std::to_string(N_LCL_OUT_MAPS) +
                std::string(" -DMLO_CHEAT_SHADER_COMPILER=") +
                std::to_string(CHEAT_SHADER_COMPILER) +
                std::string(
                    " -DMLopen_RUNNING=1") + // to disable macro defines for CodeXL Shader Analyzer
                params.general_compile_options;

            kernel.comp_options = std::string(" -DMLO_FILTER_STRIDE0=") +
                                  std::to_string(params.kernel_stride0) +
                                  std::string(" -DMLO_FILTER_STRIDE1=") +
                                  std::to_string(params.kernel_stride1) + kernel.comp_options;

            // std::cout << "compile options:\n"<< _comp_options << std::endl;

            // 1x1_Stride: FIX ME!!! NO padding support
            if(params.kernel_stride0 > 1 || params.kernel_stride1 > 1)
            {
                int FIXED_WORKGROUP_SIZE = 64;

                size_t N_OUT_GROUPS = (K / N_LCL_OUT_MAPS);

                size_t local_wk1 = 1;
                kernel.l_wk.push_back(FIXED_WORKGROUP_SIZE);
                kernel.l_wk.push_back(local_wk1);
                kernel.l_wk.push_back(1);

                size_t imagesizeAlign = ((params.out_width * params.out_height * params.batch_sz +
                                          FIXED_WORKGROUP_SIZE - 1) /
                                         FIXED_WORKGROUP_SIZE) *
                                        FIXED_WORKGROUP_SIZE;

                size_t gbl_wk0 = imagesizeAlign * N_IN_GROUPS * N_OUT_GROUPS;
                size_t gbl_wk1 = local_wk1;
                size_t gbl_wk2 = 1;

                kernel.g_wk.push_back(gbl_wk0);
                kernel.g_wk.push_back(gbl_wk1);
                kernel.g_wk.push_back(gbl_wk2);

                kernel.kernel_file = "MIOpenConv1x1J1_stride.cl";
                kernel.kernel_name = "MIOpenConv1x1";
                result.construction_params.push_back(kernel);
            }
            else
            {
                int FIXED_WORKGROUP_SIZE = 64;

                kernel.l_wk.push_back(FIXED_WORKGROUP_SIZE);
                kernel.l_wk.push_back(1);
                kernel.l_wk.push_back(1);

                size_t imagesizeAlign = ((params.in_width * params.in_height * params.batch_sz +
                                          FIXED_WORKGROUP_SIZE - 1) /
                                         FIXED_WORKGROUP_SIZE) *
                                        FIXED_WORKGROUP_SIZE;
                size_t N_OUT_GROUPS = (K / N_LCL_OUT_MAPS);

                size_t gbl_wk0 = imagesizeAlign * N_IN_GROUPS * N_OUT_GROUPS;

                size_t gbl_wk1 = 1;
                ;
                size_t gbl_wk2 = 1;

                kernel.g_wk.push_back(gbl_wk0);
                kernel.g_wk.push_back(gbl_wk1);
                kernel.g_wk.push_back(gbl_wk2);

                kernel.kernel_file = "MIOpenConv1x1J1.cl";
                kernel.kernel_name = "MIOpenConv1x1";
                result.construction_params.push_back(kernel);
            }
            // std::cout << _kernel_file << std::endl;
        }
        else
        {

            // parameters
            //	int i_sz = params.in_width * params.in_height;
            //	_out_pix_tile0 = (i_sz & 1) ? 1 : 2;
            int read_unit = result.out_pix_tile0;
            //	_n_out_pix_tiles = 16;
            //	_n_in_data_tiles = 4;
            //	_grp_tile0 = 64;

            int wei_cstride = params.kernel_size0 * params.kernel_size1;
            // backward: inputs are forward outputs
            int wei_bstride = (params.forward ? params.n_inputs : params.n_outputs) * wei_cstride;

            std::string READ_TYPE =
                (read_unit == 1) ? "_FLOAT"
                                 : "_FLOAT" + std::to_string(static_cast<long long>(read_unit));

            int OUT_WIDTH4 = params.out_width;
            int MAP_SZ4    = (OUT_WIDTH4 * params.out_height + read_unit - 1) / (read_unit);
            // stride > 1 and/or apdding
            if(params.pad0 > 0 || params.kernel_stride0 > 1 || params.pad1 > 0 ||
               params.kernel_stride1 > 1)
            {
                int step   = (params.forward) ? read_unit : read_unit * params.kernel_stride0;
                OUT_WIDTH4 = (params.out_width + step - 1) / (step);
                int OUT_HEIGHT4 =
                    (params.forward)
                        ? params.out_height
                        : (params.out_height + params.kernel_stride1 - 1) / params.kernel_stride1;
                MAP_SZ4 = (OUT_WIDTH4 * OUT_HEIGHT4);
            }

            int VERT_ALIGNED  = 1;
            int HORIZ_ALIGNED = 1;
            if(!params.forward)
            {
                VERT_ALIGNED =
                    (params.out_height / params.kernel_stride1 == params.in_height) ? 1 : 0;
                HORIZ_ALIGNED =
                    (params.out_width / params.kernel_stride0 == params.in_width) ? 1 : 0;
            }

            int GRP_SZ = result.grp_tile0;

            // number of inputs inside wk-items
            result.n_in_data_tiles = std::min(params.n_inputs, result.n_in_data_tiles);

            int CLOOP0 = (params.n_inputs + result.n_in_data_tiles - 1) / result.n_in_data_tiles;

            // number of outputs inside wk_item
            result.n_out_pix_tiles = std::min(params.n_outputs, result.n_out_pix_tiles);

            KernelInfo kernel;

            kernel.comp_options =
                std::string(" -DMLO_DIR_FORWARD=") + std::to_string(params.forward) +
                std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(params.kernel_size0) +
                std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(params.kernel_size1) +
                std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string(params.kernel_stride0) +
                std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string(params.kernel_stride1) +
                std::string(" -DMLO_FILTER_PAD0=") + std::to_string(params.pad0) +
                std::string(" -DMLO_FILTER_PAD1=") + std::to_string(params.pad1) +
                std::string(" -DMLO_IN_WIDTH=") + std::to_string(params.in_width) +
                std::string(" -DMLO_IN_HEIGHT=") + std::to_string(params.in_height) +
                std::string(" -DMLO_OUT_WIDTH=") + std::to_string(params.out_width) +
                std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(params.out_height) +
                std::string(" -DMLO_N_OUTPUTS=") + std::to_string(params.n_outputs) +
                std::string(" -DMLO_N_INPUTS=") + std::to_string(params.n_inputs) +
                std::string(" -DMLO_BATCH_SZ=") + std::to_string(params.batch_sz) +
                std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(params.out_batch_stride) +
                std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
                std::to_string(params.out_channel_stride) + std::string(" -DMLO_OUT_STRIDE=") +
                std::to_string(params.out_stride) + std::string(" -DMLO_IN_BATCH_STRIDE=") +
                std::to_string(params.in_batch_stride) + std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
                std::to_string(params.in_channel_stride) + std::string(" -DMLO_IN_STRIDE=") +
                std::to_string(params.in_stride) + std::string(" -DMLO_WEI_BSTRIDE=") +
                std::to_string(wei_bstride) + std::string(" -DMLO_WEI_CHANNEL_STRIDE=") +
                std::to_string(wei_cstride) +
                // algorithm parameters
                std::string(" -DMLO_GRP_SZ0=") + std::to_string(GRP_SZ) +
                std::string(" -DMLO_GRP_SZ1=") + std::to_string(1) +
                std::string(" -DMLO_GRP_SZ2=") + std::to_string(1) +

                std::string(" -DMLO_MAP_SZ4=") + std::to_string(MAP_SZ4) +
                std::string(" -DMLO_OUT_WIDTH4=") + std::to_string(OUT_WIDTH4) +
                std::string(" -DMLO_VERT_ALIGNED=") + std::to_string(VERT_ALIGNED) +
                std::string(" -DMLO_HORIZ_ALIGNED=") + std::to_string(HORIZ_ALIGNED) +

                std::string(" -DMLO_N_LCL_BATCHS=") +
                std::to_string(result.n_stacks) + // # of diff stacks (part of batch).
                std::string(" -DMLO_N_LCL_OUT_MAPS=") +
                std::to_string(result.n_out_pix_tiles) + // # output pixel tiles per wk-item (ALU)
                std::string(" -DMLO_N_LCL_IN_MAPS=") +
                std::to_string(
                    result.n_in_data_tiles) + // total # of blocks of different inputs in LDS
                std::string(" -DMLO_CONV_BIAS=") +
                std::to_string(params.bias) +

                std::string(" -DMLO_READ_UNIT=") + std::to_string(read_unit) +
                std::string(" -DMLO_CLOOP0=") + std::to_string(CLOOP0) +

                params.general_compile_options;

            kernel.l_wk.push_back(result.grp_tile0);
            kernel.l_wk.push_back(1);
            kernel.l_wk.push_back(1);

            size_t gbl_wk0 = params.batch_sz * MAP_SZ4;

            size_t gbl_wk1 =
                (params.n_outputs + result.n_out_pix_tiles - 1) / result.n_out_pix_tiles;
            size_t gbl_wk2 = 1;

            kernel.g_wk.push_back(gbl_wk0);
            kernel.g_wk.push_back(gbl_wk1);
            kernel.g_wk.push_back(gbl_wk2);

            kernel.kernel_file = "MIOpenConv1x1S.cl";
            kernel.kernel_name = (params.pad0 == 0 && params.kernel_stride0 == 1)
                                     ? "MIOpenConv1x1"
                                     : "MIOpenConv1x1pquv";
            result.construction_params.push_back(kernel);
        }
    }
    else
    {

        // size_t localMemSize = params.GetStream().GetLocalMemorySize();

        // _hw_wave_sz = 64;
        // _dev_local_mem_sz = localMemSize; // in bytes

        result.in_tile0      = 4;
        result.in_tile1      = 1;
        result.out_pix_tile0 = 4;
        result.out_pix_tile1 = 1;

        int wei_cstride = params.kernel_size0 * params.kernel_size1;
        // backward: inputs are forward outputs
        int wei_bstride       = (params.forward ? params.n_inputs : params.n_outputs) * wei_cstride;
        int read_unit         = 4;
        std::string READ_TYPE = (read_unit == 1)
                                    ? "_FLOAT"
                                    : "_FLOAT" + std::to_string(static_cast<long long>(read_unit));

        // currently always 1
        int N4S = 1;

        int MAP_SZ4 =
            (params.in_width * params.in_height + N4S * read_unit - 1) / (N4S * read_unit);

        int DIVBY4 = (MAP_SZ4 * read_unit == params.in_width * params.in_height) ? 1 : 0;

        int C1x1_PIXLEFT =
            (DIVBY4 == 1) ? 0 : params.in_width * params.in_height - (MAP_SZ4 - 1) * read_unit;

        bool small_map      = false;
        int GRP_SZ          = result.grp_tile0;
        int N_MAPS_PERGROUP = 1;
        // exchange step is a number of partial sums that can be exchanged in the kernel in one pass
        // it's used for small maps at the end of the kerenl to reduce partial sums
        // the number is kept in and passed through _n_in_data_tiles (with abused semantics).
        int exchange_step = 6;
        if(MAP_SZ4 <= GRP_SZ / 2)
        {
            N_MAPS_PERGROUP        = GRP_SZ / MAP_SZ4;
            exchange_step          = result.n_in_data_tiles;
            result.n_in_data_tiles = 1;
            small_map              = true;
        }

        // number of inputs inside wk-items
        result.n_in_data_tiles = std::min(params.n_inputs, result.n_in_data_tiles);
        // scale input by n of map per wk_item
        int n_input_scaled =
            (params.n_inputs + result.n_in_data_tiles - 1) / result.n_in_data_tiles;

        // number of outputs inside wk_item
        result.n_out_pix_tiles = std::min(params.n_outputs, result.n_out_pix_tiles);

        if(small_map)
        {
            exchange_step =
                std::min(std::min(exchange_step, result.n_out_pix_tiles), N_MAPS_PERGROUP);
            result.n_out_pix_tiles = (result.n_out_pix_tiles / exchange_step) * exchange_step;
        }
        // n of input map per group
        N_MAPS_PERGROUP = std::min(N_MAPS_PERGROUP, n_input_scaled);
        // number of input loops
        //   int n_in_loop = (n_input_scaled + N_MAPS_PERGROUP - 1) / N_MAPS_PERGROUP;

        // number of batches inside wk_item
        result.n_stacks = std::min(params.batch_sz, result.n_stacks);

        int n_out_tiles_pergroup = result.n_out_pix_tiles * result.n_stacks;

        int batch_aligned  = 0;
        int output_aligned = 0;
        if((params.batch_sz / result.n_stacks) * result.n_stacks == params.batch_sz)
        {
            batch_aligned = 1;
        }
        if((params.n_outputs / result.n_out_pix_tiles) * result.n_out_pix_tiles == params.n_outputs)
        {
            output_aligned = 1;
        }

        KernelInfo kernel;

        kernel.comp_options =
            std::string(" -DMLO_DIR_FORWARD=") +
            std::to_string(static_cast<long long>(params.forward)) +
            std::string(" -DMLO_FILTER_PAD1=") +
            std::to_string(static_cast<long long>(params.pad1)) + std::string(" -DMLO_N_OUTPUTS=") +
            std::to_string(static_cast<long long>(params.n_outputs)) +
            std::string(" -DMLO_N_INPUTS=") +
            std::to_string(static_cast<long long>(params.n_inputs)) +
            std::string(" -DMLO_BATCH_SZ=") +
            std::to_string(static_cast<long long>(params.batch_sz)) +
            std::string(" -DMLO_OUT_BATCH_STRIDE=") +
            std::to_string(static_cast<long long>(params.out_batch_stride)) +
            std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
            std::to_string(static_cast<long long>(params.out_channel_stride)) +
            std::string(" -DMLO_OUT_STRIDE=") +
            std::to_string(static_cast<long long>(params.out_stride)) +
            std::string(" -DMLO_IN_BATCH_STRIDE=") +
            std::to_string(static_cast<long long>(params.in_batch_stride)) +
            std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
            std::to_string(static_cast<long long>(params.in_channel_stride)) +
            std::string(" -DMLO_IN_STRIDE=") +
            std::to_string(static_cast<long long>(params.in_stride)) +
            std::string(" -DMLO_WEI_BSTRIDE=") +
            std::to_string(static_cast<long long>(wei_bstride)) +
            std::string(" -DMLO_WEI_CHANNEL_STRIDE=") +
            std::to_string(static_cast<long long>(wei_cstride))
            // algorithm parameters
            + std::string(" -DMLO_GRP_SZ0=") + std::to_string(static_cast<long long>(GRP_SZ)) +
            std::string(" -DMLO_GRP_SZ1=") + std::to_string(1) + std::string(" -DMLO_GRP_SZ2=") +
            std::to_string(1) +

            std::string(" -DMLO_MAP_SZ4=") + std::to_string(static_cast<long long>(MAP_SZ4)) +
            std::string(" -DMLO_C1x1_PIXLEFT=") +
            std::to_string(static_cast<long long>(C1x1_PIXLEFT)) + std::string(" -DMLO_DIVBY4=") +
            std::to_string(static_cast<long long>(DIVBY4)) +
            // std::string(" -DMLO_IN_LOOP=") + std::to_string(static_cast<long long>(n_in_loop)) +
            std::string(" -DMLO_N_LCL_BATCHS=") +
            std::to_string(
                static_cast<long long>(result.n_stacks)) // # of diff stacks (part of batch).
            + std::string(" -DMLO_N_LCL_OUT_MAPS=") +
            std::to_string(static_cast<long long>(
                result.n_out_pix_tiles)) // # output pixel tiles per wk-item (ALU)
            + std::string(" -DMLO_N_OUT_TILES_PERGROUP=") +
            std::to_string(static_cast<long long>(n_out_tiles_pergroup)) +
            std::string(" -DMLO_N_LCL_IN_MAPS=") +
            std::to_string(static_cast<long long>(
                result.n_in_data_tiles)) // total # of blocks of different inputs in LDS
            + std::string(" -DMLO_N_MAPS_PERGROUP=") +
            std::to_string(static_cast<long long>(
                N_MAPS_PERGROUP)) // total # of blocks of different inputs in LDS
            + std::string(" -DMLO_CONV_BIAS=") +
            std::to_string(static_cast<long long>(params.bias)) +
            std::string(" -DMLO_BATCH_ALIGNED=") +
            std::to_string(static_cast<long long>(batch_aligned)) +
            std::string(" -DMLO_OUTPUTS_ALIGNED=") +
            std::to_string(static_cast<long long>(output_aligned)) +
            std::string(" -DMLO_EXCHANGE_STEP=") +
            std::to_string(static_cast<long long>(exchange_step)) +
            std::string(" -DMLO_READ_TYPE=") + READ_TYPE + std::string(" -DMLO_READ_UNIT=") +
            std::to_string(static_cast<long long>(read_unit)) + params.general_compile_options;

        kernel.l_wk.push_back(result.grp_tile0);
        kernel.l_wk.push_back(result.grp_tile1);
        kernel.l_wk.push_back(1);

        size_t gbl_wk0 = (GRP_SZ < MAP_SZ4) ? ((MAP_SZ4 + GRP_SZ - 1) / GRP_SZ) * GRP_SZ : GRP_SZ;

        size_t gbl_wk1 = (params.n_outputs + result.n_out_pix_tiles - 1) / result.n_out_pix_tiles;
        size_t gbl_wk2 = (params.batch_sz + result.n_stacks - 1) / result.n_stacks;

        kernel.g_wk.push_back(gbl_wk0);
        kernel.g_wk.push_back(gbl_wk1);
        kernel.g_wk.push_back(gbl_wk2);

        kernel.kernel_file = "MIOpenConv1x1.cl";
        kernel.kernel_name = "MIOpenConv1x1";
        result.construction_params.push_back(kernel);

        // see above comment
        if(small_map)
        {
            result.n_in_data_tiles = exchange_step;
        }
    }
    return result;
}
} // namespace solver
} // namespace miopen
