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

#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/visit_float.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1)

#define TWO_PASSES 1

#define WORKAROUND_SWDEV_266868 1

namespace miopen {
namespace solver {

bool ConvOclBwdWrW1x1::IsApplicable(const ConvolutionContext& params) const
{
#if WORKAROUND_SWDEV_266868
    if(StartsWith(params.GetStream().GetDeviceName(), "gfx10"))
        if(!miopen::IsEnabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1{}))
            return false;
#endif
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1{}))
        return false;
    if(!params.use_opencl_convolutions)
        return false;
    if(!params.Is2d())
        return false;
    if(!params.direction.IsBackwardWrW())
        return false;
    if(params.IsAsymmetricPadH() || params.IsAsymmetricPadW())
        return false;
    if(!(params.IsFp32() || params.IsFp16() || params.IsBfp16()))
        return false;
    if(!params.IsLayoutDefault())
    {
        return false;
    }

    bool result =
        (params.kernel_size_w == 1 && params.kernel_size_h == 1 && params.kernel_dilation_w == 1 &&
         params.kernel_dilation_h == 1 && params.group_counts == 1);

    // Does not support strides > 1 if not multiple of 16
    if((params.n_inputs & 0xF) > 0 || (params.n_outputs & 0xF) > 0)
        result = false;

    return result;
}

static inline int GetNPasses(const ConvolutionContext& params)
{
    const int n_passes =
#if TWO_PASSES
        ((params.batch_sz >= 16 || 2 * params.n_outputs > params.n_inputs) && params.pad_h == 0 &&
         params.pad_w == 0 && (params.kernel_stride_w > 1 || params.kernel_stride_h > 1))
            ? 2
            :
#endif
            1;
    return n_passes;
}

size_t ConvOclBwdWrW1x1::GetWorkspaceSize(const ConvolutionContext& params) const
{
    const int n_passes = GetNPasses(params);
    if(((params.n_inputs & 0xF) == 0 && (params.n_outputs & 0xF) == 0) &&
       (n_passes > 1 && params.pad_h == 0 && params.pad_w == 0 &&
        (params.kernel_stride_w > 1 || params.kernel_stride_h > 1)))
    {
        const auto in_channel_stride = params.in_stride * params.in_height;
        const auto in_batch_stride   = in_channel_stride * params.n_outputs;
        return in_batch_stride * params.batch_sz * GetTypeSize(params.out_data_type);
    }
    else
        return 0;
}

ConvSolution ConvOclBwdWrW1x1::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;
    const int n_passes = GetNPasses(params);

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
        int in_width          = (n_passes > 1) ? params.in_width : params.out_width;
        int in_height         = (n_passes > 1) ? params.in_height : params.out_height;
        int in_stride         = (n_passes > 1) ? params.in_stride : params.out_stride;
        int in_channel_stride = (n_passes > 1) ? in_stride * in_height : params.out_channel_stride;
        int in_batch_stride =
            (n_passes > 1) ? in_channel_stride * params.n_outputs : params.out_batch_stride;
        int out_batch_stride   = params.in_batch_stride;
        int out_channel_stride = params.in_channel_stride;
        int out_stride         = params.in_stride;
        int wei_batch_stride =
            params.n_inputs * params.n_outputs * params.kernel_size_w * params.kernel_size_h;
        int wei_channel_stride     = params.n_outputs * params.kernel_size_w * params.kernel_size_h;
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

        if(params.pad_w > 0)
        {
            in_pad_min_x = params.kernel_stride_w - (params.pad_w % params.kernel_stride_w);
            // In case PAD == STRIDE
            in_pad_min_x = in_pad_min_x % params.kernel_stride_w;

            out_pad_min_x = (params.pad_w + params.kernel_stride_w - 1) / params.kernel_stride_w;
            out_pad_width = (params.out_width - in_pad_min_x + params.kernel_stride_w - 1) /
                            params.kernel_stride_w;
        }
        if(params.pad_h > 0)
        {
            in_pad_min_y = params.kernel_stride_h - (params.pad_h % params.kernel_stride_h);
            // In case PAD == STRIDE
            in_pad_min_y = in_pad_min_y % params.kernel_stride_h;

            out_pad_min_y  = (params.pad_h + params.kernel_stride_h - 1) / params.kernel_stride_h;
            out_pad_height = (params.out_height - in_pad_min_y + params.kernel_stride_h - 1) /
                             params.kernel_stride_h;
        }

        if(params.pad_w > 0 || params.pad_h > 0 ||
           (n_passes == 1 && (params.kernel_stride_w > 1 || params.kernel_stride_h > 1)))
        {
            read_unit = (out_pad_width % 4 == 0)
                            ? 4
                            : (out_pad_width % 3 == 0) ? 3 : (out_pad_width % 2 == 0) ? 2 : 1;
            // read_unit = (out_pad_width % 7 == 0) ? 7 : (out_pad_width % 5 == 0) ? 5 :
            // (out_pad_width % 4 == 0) ? 4 : (out_pad_width % 3 == 0) ? 3 : (out_pad_width % 2
            // == 0) ? 2 : 1;
            max_loads_per_readunit = (out_pad_width / read_unit) * out_pad_height * params.batch_sz;
        }

        int kernel_stride_w = params.kernel_stride_w;
        int kernel_stride_h = params.kernel_stride_h;

        if(n_passes > 1 && params.pad_h == 0 && params.pad_w == 0 &&
           (params.kernel_stride_w > 1 || params.kernel_stride_h > 1))
        {
            kernel_stride_w = 1;
            kernel_stride_h = 1;
        }

        int out_read_sz         = n_lcl_out_maps * read_unit;
        int in_read_sz          = n_lcl_in_maps * read_unit;
        int out_channel_read_sz = out_channel_stride / read_unit;
        int n_in_tile_block     = 8;
        int n_lcl_out_map_once  = 8;
        int n_lcl_in_map_once   = 8;
        int accum_sz            = n_lcl_out_map_once * n_lcl_in_map_once;

        int write_unit = (out_pad_width % 4 == 0)
                             ? 4
                             : (out_pad_width % 3 == 0) ? 3 : (out_pad_width % 2 == 0) ? 2 : 1;
        int n_grp0_size0 = 256;
        // real input strides
        int in0_stride         = params.out_stride;
        int in0_channel_stride = params.out_channel_stride;
        int in0_batch_stride   = params.out_batch_stride;
        int kernel0_stride0    = params.kernel_stride_w;
        int kernel0_stride1    = params.kernel_stride_h;

        const auto comp_options =
            std::string(" -DMLO_GRP_SZ0=") + std::to_string(n_grp_size0) +
            std::string(" -DMLO_GRP_SZ1=1 ") + std::string(" -DMLO_GRP_SZ2=1 ") +
            std::string(" -DMLO_GRP0_SZ0=") + std::to_string(n_grp0_size0) +
            std::string(" -DMLO_GRP0_SZ1=1 ") + std::string(" -DMLO_GRP0_SZ2=1 ") +
            std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(params.kernel_size_w) +
            std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(params.kernel_size_h) +
            std::string(" -DMLO_FILTER_PAD0=") + std::to_string(params.pad_w) +
            std::string(" -DMLO_FILTER_PAD1=") + std::to_string(params.pad_h) +
            std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string(kernel_stride_w) +
            std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string(kernel_stride_h) +
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
            std::to_string((n_passes == 1) ? 0 : 1) + params.general_compile_options;

        if(n_passes > 1 && params.pad_h == 0 && params.pad_w == 0 &&
           (params.kernel_stride_w > 1 || params.kernel_stride_h > 1))
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

            kernel.kernel_file = "MIOpenUtilKernels3.cl";

            kernel.kernel_name = "SubSample";

            kernel.comp_options = comp_options;

            result.construction_params.push_back(kernel);
        }

        const auto ws_sz   = GetWorkspaceSize(params);
        result.workspce_sz = ws_sz;

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

        if(n_passes == 2)
        {
            result.invoker_factory = [ws_sz](const std::vector<Kernel>& kernels) {
                return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                    const auto ss_kernel      = handle.Run(kernels[0]);
                    const auto main_kernel    = handle.Run(kernels[1]);
                    const auto& invoke_params = primitive_params.CastTo<conv::WrWInvokeParams>();

                    if(invoke_params.workSpaceSize < ws_sz)
                        MIOPEN_THROW("Not enough workspace for ConvOclBwdWrW1x1");

                    const auto& tensors    = invoke_params.tensors;
                    const auto& workSpace  = invoke_params.workSpace;
                    const auto padding_val = 0.f;
                    auto elapsed           = 0.f;

                    ss_kernel(tensors.x, workSpace);
                    if(handle.IsProfilingEnabled())
                        elapsed += handle.GetKernelTime();
                    visit_float(tensors.dyDesc.GetType(), [&](auto as_float) {
                        main_kernel(tensors.dy, workSpace, tensors.dw, as_float(padding_val));
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
        else if(n_passes == 1)
        {
            result.invoker_factory = [](const std::vector<Kernel>& kernels) {
                return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
                    const auto k              = handle.Run(kernels[0]);
                    const auto& invoke_params = primitive_params.CastTo<conv::WrWInvokeParams>();
                    const auto& tensors       = invoke_params.tensors;
                    const auto padding_val    = 0.f;

                    visit_float(tensors.dyDesc.GetType(), [&](auto as_float) {
                        k(tensors.dy, tensors.x, tensors.dw, as_float(padding_val));
                    });
                };
            };
        }
    }
    return result;
}
} // namespace solver
} // namespace miopen
