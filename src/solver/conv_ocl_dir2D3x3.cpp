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

bool ConvOclDirectFwd3x3::IsApplicable(const ConvolutionContext& params) const
{
    return (params.kernel_size0 == 3 && params.kernel_size1 == 3 && params.pad1 == 1 &&
            params.pad0 == 1 && params.kernel_stride0 == 1 && params.kernel_stride1 == 1 &&
            params.direction.IsForward()) &&
           (params.out_width == 512 || params.out_width == 64 || params.out_width == 128 ||
            params.out_width == 256);
}

ConvSolution ConvOclDirectFwd3x3::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
    // size_t localMemSize = params.stream.GetLocalMemorySize();
    auto hw_wave_sz = 64;
    // auto dev_local_mem_sz = localMemSize; // in bytes
    int n_waves = 4;

    int wei_cstride = params.kernel_size0 * params.kernel_size1;
    int wei_bstride = params.n_inputs * wei_cstride;

    result.out_pix_tile0   = 4;
    result.out_pix_tile1   = 2;
    result.n_stacks        = 1;
    result.n_out_pix_tiles = 4;
    int read_unit          = result.out_pix_tile0;
    //	std::string READ_TYPE = (read_unit == 1) ? "_FLOAT" : "_FLOAT" +
    // std::to_string(static_cast<long long>(read_unit));
    // MD: read_unit is never == 1
    std::string READ_TYPE = "_FLOAT" + std::to_string(static_cast<long long>(read_unit));

    int GRP_SZ = hw_wave_sz * n_waves;

    int ALU_EXTENT_X     = (params.out_width + read_unit - 1) / read_unit;
    auto LG2ALU_EXTENT_X = static_cast<int>(std::ceil(std::log(ALU_EXTENT_X) / std::log(2)));
    int ALU_EXTENT_Y     = (GRP_SZ >> LG2ALU_EXTENT_X);
    auto LG2ALU_EXTENT_Y = static_cast<int>(std::ceil(std::log(ALU_EXTENT_Y) / std::log(2)));

    // the wave is logical is a unit of shareing weights in SGPRs
    // it cannot be less than HW_WAVE_SIZE = 64
    // it cannot be larger than the group size.

    int LG2_WAVE_SZ0    = std::ceil(std::log(ALU_EXTENT_X) / std::log(2));
    int logical_wave_sz = std::max(1, ALU_EXTENT_X / hw_wave_sz) * hw_wave_sz;
    if(logical_wave_sz > GRP_SZ)
    {
        std::cout << "Conv3x3 conf error\n";
        return ConvSolution(static_cast<miopenStatus_t>(-1));
    }
    int logical_n_waves = std::max(1, GRP_SZ / logical_wave_sz);
    int LG2_WAVE_SZ     = std::ceil(std::log(logical_wave_sz) / std::log(2));
    int WAVE_SZ1        = (logical_wave_sz >> LG2_WAVE_SZ0);
    int lg2_n_waves     = std::ceil(std::log(logical_n_waves) / std::log(2));
    int N_WAVES_MASK    = (1 << lg2_n_waves) - 1;

    int OUT_EXTENT1 = result.out_pix_tile1 * WAVE_SZ1;
    int OUT_EXTENT0 = (result.out_pix_tile0 << LG2_WAVE_SZ0);

    int total_out_maps = result.n_out_pix_tiles * logical_n_waves;

    // number of batches inside wk_item
    result.n_stacks = std::min(params.batch_sz, result.n_stacks);

    int N_HEIGHT_EXTENTS = (params.out_height + OUT_EXTENT1 - 1) / OUT_EXTENT1;
    int N_WIDTH_EXTENTS  = (params.out_width + OUT_EXTENT0 - 1) / OUT_EXTENT0;
    int N_GROUPS_PER_MAP = N_HEIGHT_EXTENTS * N_WIDTH_EXTENTS;

    result.grp_tile0 = GRP_SZ;
    result.grp_tile1 = 1;
    int grp_tile2    = 1;
    // auto in_tile0 = OUT_EXTENT0;
    // auto in_tile1 = OUT_EXTENT1;
    auto n_in_data_tiles = 1;

    //	_gen_comp_options += std::string(" -limit-vector-registers=64 ");

    KernelInfo construction_parameters;

    construction_parameters.comp_options =
        std::string(" -DMLO_DIR_FORWARD=") + (params.direction.IsForward() ? "1" : "0") +
        std::string(" -DMLO_GRP_SZ=") + std::to_string(static_cast<long long>(GRP_SZ)) +
        std::string(" -DMLO_GRP_SZ0=") + std::to_string(static_cast<long long>(result.grp_tile0)) +
        std::string(" -DMLO_GRP_SZ1=") + std::to_string(static_cast<long long>(result.grp_tile1)) +
        std::string(" -DMLO_GRP_SZ2=") + std::to_string(static_cast<long long>(grp_tile2)) +
        std::string(" -DMLO_FILTER_SIZE0=") +
        std::to_string(static_cast<long long>(params.kernel_size0)) +
        std::string(" -DMLO_FILTER_SIZE1=") +
        std::to_string(static_cast<long long>(params.kernel_size1)) +
        std::string(" -DMLO_FILTER_PAD0=") + std::to_string(static_cast<long long>(params.pad0)) +
        std::string(" -DMLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(params.pad1)) +
        std::string(" -DMLO_N_OUTPUTS=") +
        std::to_string(static_cast<long long>(params.n_outputs)) + std::string(" -DMLO_N_INPUTS=") +
        std::to_string(static_cast<long long>(params.n_inputs)) + std::string(" -DMLO_BATCH_SZ=") +
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
        std::string(" -DMLO_WEI_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(wei_bstride)) +
        std::string(" -DMLO_WEI_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(wei_cstride)) + std::string(" -DMLO_IN_WIDTH=") +
        std::to_string(static_cast<long long>(params.in_width)) + std::string(" -DMLO_IN_HEIGHT=") +
        std::to_string(static_cast<long long>(params.in_height)) +
        std::string(" -DMLO_N_LCL_BATCHS=") +
        std::to_string(static_cast<long long>(result.n_stacks)) // # of diff stacks (part of batch).
        + std::string(" -DMLO_N_LCL_OUT_MAPS=") +
        std::to_string(static_cast<long long>(
            result.n_out_pix_tiles)) // # output pixel tiles per wk-item (ALU)
        + std::string(" -DMLO_N_LCL_IN_MAPS=") +
        std::to_string(
            static_cast<long long>(n_in_data_tiles)) // total # of blocks of different inputs in LDS
        + std::string(" -DMLO_OUT_TILE0=") +
        std::to_string(
            static_cast<long long>(result.out_pix_tile0)) // size of ouptput tile per wk-item (ALU))
        + std::string(" -DMLO_OUT_TILE1=") +
        std::to_string(static_cast<long long>(result.out_pix_tile1)) //
        + std::string(" -DMLO_ALU_EXTENT_X=") +
        std::to_string(static_cast<long long>(ALU_EXTENT_X)) +
        std::string(" -DMLO_LG2ALU_EXTENT_X=") +
        std::to_string(static_cast<long long>(LG2ALU_EXTENT_X)) +
        std::string(" -DMLO_ALU_EXTENT_Y=") + std::to_string(static_cast<long long>(ALU_EXTENT_Y)) +
        std::string(" -DMLO_LG2ALU_EXTENT_Y=") +
        std::to_string(static_cast<long long>(LG2ALU_EXTENT_Y)) +
        std::string(" -DMLO_OUT_EXTENT1=") + std::to_string(static_cast<long long>(OUT_EXTENT1)) +
        std::string(" -DMLO_OUT_EXTENT0=") + std::to_string(static_cast<long long>(OUT_EXTENT0)) +
        std::string(" -DMLO_N_WAVES=") + std::to_string(static_cast<long long>(logical_n_waves)) +
        std::string(" -DMLO_N_WAVES_MASK=") + std::to_string(static_cast<long long>(N_WAVES_MASK)) +
        std::string(" -DMLO_LG2_WAVE_SZ=") + std::to_string(static_cast<long long>(LG2_WAVE_SZ)) +
        std::string(" -DMLO_LG2_WAVE_SZ0=") + std::to_string(static_cast<long long>(LG2_WAVE_SZ0)) +
        std::string(" -DMLO_READ_TYPE=") + READ_TYPE + std::string(" -DMLO_READ_UNIT=") +
        std::to_string(static_cast<long long>(read_unit)) + std::string(" -DMLO_CONV_BIAS=") +
        std::to_string(static_cast<long long>(params.bias)) + params.general_compile_options;

    construction_parameters.l_wk.push_back(result.grp_tile0);
    construction_parameters.l_wk.push_back(result.grp_tile1);
    construction_parameters.l_wk.push_back(grp_tile2);

    size_t gbl_wk0 = N_GROUPS_PER_MAP;

    size_t gbl_wk1 = (params.n_outputs + total_out_maps - 1) / total_out_maps;
    size_t gbl_wk2 = (params.batch_sz + result.n_stacks - 1) / result.n_stacks;

    construction_parameters.g_wk.push_back(gbl_wk0 * result.grp_tile0);
    construction_parameters.g_wk.push_back(gbl_wk1);
    construction_parameters.g_wk.push_back(gbl_wk2);

    construction_parameters.kernel_file = "MIOpenConvD3x3.cl";
    construction_parameters.kernel_name = "MIOpenCvD3x3_WSR0";

    result.construction_params.push_back(construction_parameters);
    return result;
}
} // namespace solver
} // namespace miopen
