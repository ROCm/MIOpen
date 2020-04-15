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

#include <miopen/solver.hpp>
#include <miopen/handle.hpp>
#include <miopen/env.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>
#include <miopen/conv/data_invoke_params.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD11X11)

namespace miopen {
namespace solver {

bool ConvOclDirectFwd11x11::IsApplicable(const ConvolutionContext& params) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD11X11{}))
        return false;
    if(!params.use_opencl_convolutions)
        return false;
    if(!params.Is2d())
        return false;
    if(!(params.IsFp32() || params.IsFp16() || params.IsBfp16()))
        return false;

    return params.direction.IsForward() && params.group_counts == 1 &&
           params.kernel_dilation_h == 1 && params.kernel_dilation_w == 1 &&
           params.kernel_size_h == 11 && params.kernel_size_w == 11 &&
           params.kernel_stride_h == 4 && params.kernel_stride_w == 4;
}

ConvSolution ConvOclDirectFwd11x11::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
    const bool is_forward = params.direction.IsForward();
    // size_t localMemSize = 64 * 1024;
    auto hw_wave_sz = 64;
    // auto dev_local_mem_sz = localMemSize; // in bytes
    // major parameters
    int LG2_WAVE_SZ = mloLg2(hw_wave_sz);
    int wei_cstride = params.kernel_size_w * params.kernel_size_h;
    int wei_bstride = (is_forward ? params.n_inputs : params.n_outputs) * wei_cstride;

    // number  of batch iterations
    result.n_stacks = 1;
    result.n_stacks = std::min(params.batch_sz, result.n_stacks);
    // defines how to proceed : 1 grouop per batch or with a loop over all batches
    // loop over al batches make sense in 2 cases: a lot of small inputs/outputs or few batches
    // param
    int N_BATCH_LOOPS = 1; // (_n_inputs*_n_outputs <= 8 * 1024) ? 1 : _batch_sz / _n_stacks;
    int n_batch_blks =
        (params.batch_sz + N_BATCH_LOOPS * result.n_stacks - 1) / (N_BATCH_LOOPS * result.n_stacks);

    int N_FILTER_SPLITS0 =
        ((params.kernel_size_w + params.kernel_stride_w - 1) / params.kernel_stride_w);
    int N_FILTER_SPLITS1 =
        ((params.kernel_size_h + params.kernel_stride_h - 1) / params.kernel_stride_h);

    static const int data_multiplier0 =
// win runs Catalyst right now
#ifdef _WIN32
        1
#else
        2
#endif
        ;

    static const int data_multiplier1 = 1;

    result.out_pix_tile0 =
        (is_forward) ? N_FILTER_SPLITS0 : data_multiplier0 * params.kernel_stride_w;
    result.out_pix_tile1 = (is_forward) ? 1 : data_multiplier1 * params.kernel_stride_h;

    int in_pix_tile0 =
        (is_forward) ? 1 : (result.out_pix_tile0 / params.kernel_stride_w - 1) + N_FILTER_SPLITS0;
    int in_pix_tile1 =
        (is_forward) ? 1 : (result.out_pix_tile1 / params.kernel_stride_h - 1) + N_FILTER_SPLITS1;

    result.in_tile1 = 1;
    result.in_tile0 = 1;

    // n of wvaefront in a group
    // param
    int n_waves      = 4;
    int GRP_SZ       = hw_wave_sz * n_waves;
    int lg2_n_waves  = mloLg2(n_waves);
    int N_WAVES_MASK = (1 << lg2_n_waves) - 1;

    // number of input maps per group
    // processing arrangement
    // generate full output width
    // extent1 == MLO_GRP_SZ / MLO_PROCESING_WIDTH
    int PROCESING_WIDTH = ((params.out_width + result.out_pix_tile0 - 1) / result.out_pix_tile0);

    int OUT_EXTENT1 = std::min(params.out_height, (GRP_SZ / PROCESING_WIDTH));

    // define a special size for a specific width as a devisor to avoid dealing with out of range
    // param
    static const int read_unit =
        10; // (((_in_width / 8) * 8) == _in_width) ? 8 : (((_in_width / 4) * 4) ==
            // _in_width) ? 4 : (((_in_width / 2) * 2) == _in_width) ? 2 : 1;

    // this one is valid only till _FLOAT8
    // but it's not an error, the kernel does not use these types at all
    static const std::string READ_TYPE =
        (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((read_unit));

    // param
    int n_out_stacks = 1;

    // n_in_stacks input map will be written in the local memory.
    int n_in_stacks = 1;

    n_in_stacks  = std::min(params.n_inputs, n_in_stacks);
    n_out_stacks = std::min(params.n_outputs, n_out_stacks);

    // param
    // 6 get us the min
    // cppcheck-suppress knownConditionTrueFalse
    static const int backwards_min_output = (data_multiplier1 > 1 || data_multiplier0 > 1) ? 1 : 4;
    result.n_out_pix_tiles                = (is_forward)
                                 ? std::min(6, (params.n_outputs + n_out_stacks - 1) / n_out_stacks)
                                 : std::min(params.n_outputs, backwards_min_output);

    // number of maps in a stack or number of input read blocks written into 1 wk-item (lane)
    // param
    result.n_in_data_tiles = 1;

    // total maps per group
    int total_out_maps = result.n_out_pix_tiles * (is_forward ? n_out_stacks : 1);

    // n of mini tiles of the same output map in vertical dir per wk_item
    result.grp_tile0 = GRP_SZ;
    result.grp_tile1 = 1;
    int grp_tile2    = 1;

    // second pass if needed
    int n_extents           = ((params.out_height + OUT_EXTENT1 - 1) / OUT_EXTENT1);
    int n_output_map_blocks = ((params.n_outputs + total_out_maps - 1) / total_out_maps);
    int last_out_extent1 =
        params.out_height - (std::max(1, params.out_height / OUT_EXTENT1) * OUT_EXTENT1);
    last_out_extent1    = (last_out_extent1 < 0) ? 0 : last_out_extent1;
    int n_batches_pass2 = 1;
    bool second_pass    = false;
    if(is_forward && 0 < last_out_extent1 && last_out_extent1 <= OUT_EXTENT1 / 2)
    {
        n_extents       = std::max(1, params.out_height / OUT_EXTENT1);
        n_batches_pass2 = std::max(1, GRP_SZ / (PROCESING_WIDTH * last_out_extent1));
        second_pass     = true;
    }

    // calc bwd grid
    int n_out_pix_tiles1 =
        (params.out_height + result.out_pix_tile1 - 1 + 2 * params.pad_h) / result.out_pix_tile1;
    int n_out_pix_tiles0 =
        (params.out_width + result.out_pix_tile0 - 1 + 2 * params.pad_w) / result.out_pix_tile0;
    int n_out_pix_tiles = n_out_pix_tiles1 * n_out_pix_tiles0;

    // calculate lcl mem size for backward data
    int n_out_tiles_rows_pgrp =
        std::min(n_out_pix_tiles1, (GRP_SZ + n_out_pix_tiles0 - 1) / n_out_pix_tiles0);
    int n_out_tiles_cols_pgrp = std::min(GRP_SZ, n_out_pix_tiles0);
    int in_data1 = ((n_out_tiles_rows_pgrp * result.out_pix_tile1) / params.kernel_stride_h - 1) +
                   N_FILTER_SPLITS1 + 1;
    int in_data0 = ((n_out_tiles_cols_pgrp * result.out_pix_tile0) / params.kernel_stride_w - 1) +
                   N_FILTER_SPLITS0;

    int lcl_wei_sz = wei_cstride * result.n_out_pix_tiles;
#ifndef _WIN32
    int lcl_in_data_sz = in_data1 * in_data0 * result.n_in_data_tiles;
    int lcl_bwd_sz     = std::max(lcl_in_data_sz, lcl_wei_sz);
#else
    // win runs Catalyst right now

    int lcl_bwd_sz = lcl_wei_sz;
#endif

    // it's backward - inputs are outputs and vs versa
    const auto comp_options =
        std::string(" -DMLO_DIR_FORWARD=") + (is_forward ? "1" : "0") +
        std::string(" -DMLO_GRP_SZ=") + std::to_string(GRP_SZ) + std::string(" -DMLO_GRP_SZ0=") +
        std::to_string(result.grp_tile0) + std::string(" -DMLO_GRP_SZ1=") +
        std::to_string(result.grp_tile1) + std::string(" -DMLO_GRP_SZ2=") +
        std::to_string(grp_tile2) + std::string(" -DMLO_FILTER_SIZE0=") +
        std::to_string(params.kernel_size_w) + std::string(" -DMLO_FILTER_SIZE1=") +
        std::to_string(params.kernel_size_h) + std::string(" -DMLO_FILTER_PAD0=") +
        std::to_string(params.pad_w) + std::string(" -DMLO_FILTER_PAD1=") +
        std::to_string(params.pad_h) + std::string(" -DMLO_FILTER_STRIDE0=") +
        std::to_string(params.kernel_stride_w) + std::string(" -DMLO_FILTER_STRIDE1=") +
        std::to_string(params.kernel_stride_h) + std::string(" -DSTRIDE_W=") +
        std::to_string(params.kernel_stride_w) + std::string(" -DSTRIDE_H=") +
        std::to_string(params.kernel_stride_h) + std::string(" -DMLO_N_OUTPUTS=") +
        std::to_string(params.n_outputs) + std::string(" -DMLO_N_INPUTS=") +
        std::to_string(params.n_inputs) + std::string(" -DMLO_BATCH_SZ=") +
        std::to_string(params.batch_sz) + std::string(" -DMLO_N_BATCH_LOOPS=") +
        std::to_string(N_BATCH_LOOPS) + std::string(" -DMLO_OUT_BATCH_STRIDE=") +
        std::to_string(params.out_batch_stride) + std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
        std::to_string(params.out_channel_stride) + std::string(" -DMLO_OUT_STRIDE=") +
        std::to_string(params.out_stride) + std::string(" -DMLO_IN_BATCH_STRIDE=") +
        std::to_string(params.in_batch_stride) + std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
        std::to_string(params.in_channel_stride) + std::string(" -DMLO_IN_STRIDE=") +
        std::to_string(params.in_stride) + std::string(" -DMLO_WEI_BATCH_STRIDE=") +
        std::to_string(wei_bstride) + std::string(" -DMLO_WEI_CHANNEL_STRIDE=") +
        std::to_string(wei_cstride) + std::string(" -DMLO_IN_WIDTH=") +
        std::to_string(params.in_width) + std::string(" -DMLO_IN_HEIGHT=") +
        std::to_string(params.in_height) + std::string(" -DMLO_OUT_WIDTH=") +
        std::to_string(params.out_width) + std::string(" -DMLO_OUT_HEIGHT=") +
        std::to_string(params.out_height) + std::string(" -DMLO_IN_TILE1=") +
        std::to_string(result.in_tile1) + std::string(" -DMLO_IN_TILE0=") +
        std::to_string(result.in_tile0) + std::string(" -DMLO_N_LCL_BATCHS=") +
        std::to_string(result.n_stacks) // # of diff stacks (part of batch).
        + std::string(" -DMLO_N_LCL_OUT_MAPS=") +
        std::to_string(result.n_out_pix_tiles) // # output pixel tiles per wk-item (ALU)
        + std::string(" -DMLO_N_LCL_IN_MAPS=") +
        std::to_string(result.n_in_data_tiles) // total # of blocks of different inputs in LDS
        + std::string(" -DMLO_IN_PIX_TILE1=") +
        std::to_string(in_pix_tile1) // size of ouptput tile per wk-item (ALU)
        + std::string(" -DMLO_IN_PIX_TILE0=") + std::to_string(in_pix_tile0) //
        + std::string(" -DMLO_OUT_PIX_TILE1=") +
        std::to_string(result.out_pix_tile1) // size of ouptput tile per wk-item (ALU)
        + std::string(" -DMLO_OUT_PIX_TILE0=") + std::to_string(result.out_pix_tile0) //
        + std::string(" -DMLO_OUT_STACKS=") + std::to_string(n_out_stacks) +
        std::string(" -DMLO_IN_STACKS=") + std::to_string(n_in_stacks) +
        std::string(" -DMLO_N_WAVES=") + std::to_string(n_waves) +
        std::string(" -DMLO_N_FILTER_SPLITS0=") + std::to_string(N_FILTER_SPLITS0) +
        std::string(" -DMLO_N_FILTER_SPLITS1=") + std::to_string(N_FILTER_SPLITS1) +
        std::string(" -DMLO_PROCESSING_WIDTH=") + std::to_string(PROCESING_WIDTH) +
        std::string(" -DMLO_OUT_EXTENT1=") + std::to_string(OUT_EXTENT1) +
        std::string(" -DMLO_LAST_OUT_EXTENT1=") + std::to_string(last_out_extent1) +
        std::string(" -DMLO_N_LCL_BATCHS_PASS2=") + std::to_string(n_batches_pass2) +
        std::string(" -DMLO_TILE_REPLICATE0=") + std::to_string(data_multiplier0) +
        std::string(" -DMLO_TILE_REPLICATE1=") + std::to_string(data_multiplier1) +
        std::string(" -DMLO_LCL_BWD_MEM_SZ=") + std::to_string(lcl_bwd_sz) +
        std::string(" -DMLO_N_IN_BWD_HORIZ_READS=") + std::to_string(in_data0) +
        std::string(" -DMLO_N_IN_BWD_VERT_READS=") + std::to_string(in_data1)

        + std::string(" -DMLO_READ_TYPE=") + READ_TYPE + std::string(" -DMLO_READ_UNIT=") +
        std::to_string(read_unit) + std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(hw_wave_sz) +
        std::string(" -DMLO_LG2_WAVE_SZ=") + std::to_string(LG2_WAVE_SZ) +
        std::string(" -DMLO_N_WAVES_MASK=") + std::to_string(static_cast<long long>(N_WAVES_MASK))

        + std::string(" -DMLO_CONV_BIAS=") + std::to_string(params.bias)

        + std::string(" -cl-denorms-are-zero  ") + params.general_compile_options;

    // 1st pass
    {
        KernelInfo construction_parameters;

        construction_parameters.l_wk.push_back(result.grp_tile0);
        construction_parameters.l_wk.push_back(result.grp_tile1);
        construction_parameters.l_wk.push_back(grp_tile2);
        // input is output

        size_t gbl_wk0 =
            is_forward ? GRP_SZ * n_extents : ((n_out_pix_tiles + GRP_SZ - 1) / GRP_SZ) * GRP_SZ;
        size_t gbl_wk1 = n_output_map_blocks;
        size_t gbl_wk2 = n_batch_blks;

        construction_parameters.g_wk.push_back(gbl_wk0);
        construction_parameters.g_wk.push_back(gbl_wk1);
        construction_parameters.g_wk.push_back(gbl_wk2);

        construction_parameters.kernel_file  = "MIOpenConvFwd_LxL_11.cl";
        construction_parameters.kernel_name  = is_forward ? "MIOpenCvFwd11x11" : "MIOpenCvBwd11x11";
        construction_parameters.comp_options = comp_options;

        result.construction_params.push_back(construction_parameters);
    }
    // 2nd  pass
    if(second_pass)
    {
        KernelInfo construction_parameters;

        construction_parameters.kernel_file  = "MIOpenConvFwd_LxL_11.cl";
        construction_parameters.kernel_name  = "MIOpenCvFwd11x11_2";
        construction_parameters.comp_options = comp_options;

        construction_parameters.l_wk.push_back(result.grp_tile0);
        construction_parameters.l_wk.push_back(result.grp_tile1);
        construction_parameters.l_wk.push_back(grp_tile2);
        // input is output

        size_t gbl_wk0 = GRP_SZ;
        size_t gbl_wk1 = n_output_map_blocks;
        n_batch_blks   = (params.batch_sz + n_batches_pass2 - 1) / n_batches_pass2;
        size_t gbl_wk2 = n_batch_blks;

        construction_parameters.g_wk.push_back(gbl_wk0);
        construction_parameters.g_wk.push_back(gbl_wk1);
        construction_parameters.g_wk.push_back(gbl_wk2);

        result.construction_params.push_back(construction_parameters);
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            if(kernels.size() != 2)
                MIOPEN_THROW("Two kernels were expected by solver");

            return [=](Handle& handle, const boost::any& primitive_parameters) {
                auto invoke_params  = boost::any_cast<conv::DataInvokeParams>(primitive_parameters);
                const auto& tensors = invoke_params.tensors;

                const auto first_pass_kernel  = handle.Run(kernels[0]);
                const auto second_pass_kernel = handle.Run(kernels[1]);

                float padding_val = 0;
                float elapsed     = 0;

                visit_float(tensors.inDesc.GetType(), [&](auto as_float) {
                    first_pass_kernel(tensors.in, tensors.w, tensors.out, as_float(padding_val));
                });

                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                visit_float(tensors.inDesc.GetType(), [&](auto as_float) {
                    second_pass_kernel(tensors.in, tensors.w, tensors.out, as_float(padding_val));
                });

                if(handle.IsProfilingEnabled())
                {
                    elapsed += handle.GetKernelTime();
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
    }
    else
    {
        result.invoker_factory = &conv::MakeGenericXWYPadInvoker;
    }
    return result;
}
} // namespace solver
} // namespace miopen
