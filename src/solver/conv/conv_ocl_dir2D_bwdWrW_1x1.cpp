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

#include <miopen/conv/solvers.hpp>

#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/visit_float.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1)

#define TWO_PASSES 1

#define WORKAROUND_SWDEV_266868 1

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvOclBwdWrW1x1::IsApplicable(const ExecutionContext& ctx,
                                    const ProblemDescription& problem) const
{
#if WORKAROUND_SWDEV_266868
    if(StartsWith(ctx.GetStream().GetDeviceName(), "gfx10") ||
       StartsWith(ctx.GetStream().GetDeviceName(), "gfx11"))
    {
        if(!env::enabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1))
            return false;
    }
#endif
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW1X1))
        return false;
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;
    if(!ctx.use_opencl_convolutions)
        return false;
    if(!problem.Is2d())
        return false;
    if(!problem.IsDirectionBackwardWrW())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsAsymmetricPadH() || problem.IsAsymmetricPadW())
        return false;
    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16()))
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    if(problem.IsTensorsCasted())
        return false;

    bool result = (problem.GetWeightsWidth() == 1 && problem.GetWeightsHeight() == 1 &&
                   problem.GetDilationW() == 1 && problem.GetDilationH() == 1 &&
                   problem.GetGroupCount() == 1);

    // Does not support strides > 1 if not multiple of 16
    if((problem.GetInChannels() & 0xF) > 0 || (problem.GetOutChannels() & 0xF) > 0)
        result = false;

    return result;
}

static inline int GetNPasses(const ProblemDescription& problem)
{
    const int n_passes =
#if TWO_PASSES
        ((problem.GetBatchSize() >= 16 || 2 * problem.GetOutChannels() > problem.GetInChannels()) &&
         problem.GetPadH() == 0 && problem.GetPadW() == 0 &&
         (problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1))
            ? 2
            :
#endif
            1;
    return n_passes;
}

size_t ConvOclBwdWrW1x1::GetWorkspaceSize(const ExecutionContext&,
                                          const ProblemDescription& problem) const
{
    const int n_passes = GetNPasses(problem);
    if(((problem.GetInChannels() & 0xF) == 0 && (problem.GetOutChannels() & 0xF) == 0) &&
       (n_passes > 1 && problem.GetPadH() == 0 && problem.GetPadW() == 0 &&
        (problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1)))
    {
        const auto in_channel_stride = problem.GetInStrideH() * problem.GetInHeight();
        const auto in_batch_stride   = in_channel_stride * problem.GetOutChannels();
        return GetTypeSize(problem.GetOutDataType()) * in_batch_stride * problem.GetBatchSize();
    }
    else
        return 0;
}

ConvSolution ConvOclBwdWrW1x1::GetSolution(const ExecutionContext& ctx,
                                           const ProblemDescription& problem) const
{
    ConvSolution result;
    const int n_passes = GetNPasses(problem);

    // FIX ME! FIX ME! FIX ME! Does not support C, K != 16X yet
    // NON-Stride/PAD mode NON-16X will be supported by MIOpenConvBwdWrW1x1.CL
    if((problem.GetInChannels() & 0xF) == 0 && (problem.GetOutChannels() & 0xF) == 0)
    {
        // problem.GetInChannels()==> C
        // problem.GetOutChannels()==>K
        // Jian: following kernel uses C as input, K as output, different from original definition
        // FIX ME! FIX ME! FIX ME!
        // JIANYANG: not know the meaning of following ==>
        result.n_stacks      = std::min(problem.GetBatchSize(), static_cast<std::size_t>(1));
        result.out_pix_tile0 = 1;
        result.out_pix_tile1 = 1;
        result.in_tile1      = 1;
        result.in_tile0      = 1;
        // JIANYANG: not know the meaning of above <==

        // 8/16/64
        int n_lcl_in_maps = 8;

        /*if(4 *((problem.GetOutChannels()+63)/64) * ((problem.GetInChannels()+63)/64) >=512)
        {
                n_lcl_in_maps =64;
        }
        else
        */
        if(4 * ((problem.GetOutChannels() + 15) / 16) * ((problem.GetInChannels() + 15) / 16) >=
           512)
        {
            n_lcl_in_maps = 16;
        }

        // 8/16/64
        int n_lcl_out_maps = n_lcl_in_maps;

        int n_grp_size0 = 64;

        int n_out_blocks = ((problem.GetInChannels() + n_lcl_out_maps - 1) / n_lcl_out_maps);
        int n_in_blocks  = ((problem.GetOutChannels() + n_lcl_in_maps - 1) / n_lcl_in_maps);
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
        int in_width          = (n_passes > 1) ? problem.GetInWidth() : problem.GetOutWidth();
        std::size_t in_height = (n_passes > 1) ? problem.GetInHeight() : problem.GetOutHeight();
        std::size_t in_stride = (n_passes > 1) ? problem.GetInStrideH() : problem.GetOutStrideH();
        int in_channel_stride =
            (n_passes > 1) ? in_stride * in_height : problem.GetOutChannelStride();
        int in_batch_stride    = (n_passes > 1) ? in_channel_stride * problem.GetOutChannels()
                                                : problem.GetOutBatchStride();
        int out_batch_stride   = problem.GetInBatchStride();
        int out_channel_stride = problem.GetInChannelStride();
        int out_stride         = problem.GetInStrideH();
        int wei_batch_stride   = problem.GetInChannels() * problem.GetOutChannels() *
                               problem.GetWeightsWidth() * problem.GetWeightsHeight();
        int wei_channel_stride =
            problem.GetOutChannels() * problem.GetWeightsWidth() * problem.GetWeightsHeight();
        int max_loads_per_readunit = (out_channel_stride / read_unit) * problem.GetBatchSize();

        // limited shape size shows better performance with ead_uint == 3
        /*
        if( (out_channel_stride % 3) == 1)
        {
                read_unit              = 3;
                max_loads_per_readunit = (out_channel_stride / read_unit) * problem.GetBatchSize();
        }
        */

        int out_pad_min_x  = 0;
        int out_pad_min_y  = 0;
        int out_pad_width  = problem.GetInWidth();
        int out_pad_height = problem.GetInHeight();

        int in_pad_min_x = 0;
        int in_pad_min_y = 0;

        if(problem.GetPadW() > 0)
        {
            in_pad_min_x =
                problem.GetKernelStrideW() - (problem.GetPadW() % problem.GetKernelStrideW());
            // In case PAD == STRIDE
            in_pad_min_x = in_pad_min_x % problem.GetKernelStrideW();

            out_pad_min_x =
                (problem.GetPadW() + problem.GetKernelStrideW() - 1) / problem.GetKernelStrideW();
            out_pad_width = (static_cast<int>(problem.GetOutWidth()) - in_pad_min_x +
                             problem.GetKernelStrideW() - 1) /
                            problem.GetKernelStrideW();
        }
        if(problem.GetPadH() > 0)
        {
            in_pad_min_y =
                problem.GetKernelStrideH() - (problem.GetPadH() % problem.GetKernelStrideH());
            // In case PAD == STRIDE
            in_pad_min_y = in_pad_min_y % problem.GetKernelStrideH();

            out_pad_min_y =
                (problem.GetPadH() + problem.GetKernelStrideH() - 1) / problem.GetKernelStrideH();
            out_pad_height = (static_cast<int>(problem.GetOutHeight()) - in_pad_min_y +
                              problem.GetKernelStrideH() - 1) /
                             problem.GetKernelStrideH();
        }

        if(problem.GetPadW() > 0 || problem.GetPadH() > 0 ||
           (n_passes == 1 && (problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1)))
        {
            read_unit = (out_pad_width % 4 == 0)   ? 4
                        : (out_pad_width % 3 == 0) ? 3
                        : (out_pad_width % 2 == 0) ? 2
                                                   : 1;
            // read_unit = (out_pad_width % 7 == 0) ? 7 : (out_pad_width % 5 == 0) ? 5 :
            // (out_pad_width % 4 == 0) ? 4 : (out_pad_width % 3 == 0) ? 3 : (out_pad_width % 2
            // == 0) ? 2 : 1;
            max_loads_per_readunit = (out_pad_width / read_unit) * out_pad_height *
                                     static_cast<int>(problem.GetBatchSize());
        }

        int kernel_stride_w = problem.GetKernelStrideW();
        int kernel_stride_h = problem.GetKernelStrideH();

        if(n_passes > 1 && problem.GetPadH() == 0 && problem.GetPadW() == 0 &&
           (problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1))
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

        int write_unit   = (out_pad_width % 4 == 0)   ? 4
                           : (out_pad_width % 3 == 0) ? 3
                           : (out_pad_width % 2 == 0) ? 2
                                                      : 1;
        int n_grp0_size0 = 256;
        // real input strides
        int in0_stride         = problem.GetOutStrideH();
        int in0_channel_stride = problem.GetOutChannelStride();
        int in0_batch_stride   = problem.GetOutBatchStride();
        int kernel0_stride0    = problem.GetKernelStrideW();
        int kernel0_stride1    = problem.GetKernelStrideH();

        const auto comp_options =
            std::string(" -DMLO_GRP_SZ0=") + std::to_string(n_grp_size0) +
            std::string(" -DMLO_GRP_SZ1=1 ") + std::string(" -DMLO_GRP_SZ2=1 ") +
            std::string(" -DMLO_GRP0_SZ0=") + std::to_string(n_grp0_size0) +
            std::string(" -DMLO_GRP0_SZ1=1 ") + std::string(" -DMLO_GRP0_SZ2=1 ") +
            std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(problem.GetWeightsWidth()) +
            std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(problem.GetWeightsHeight()) +
            std::string(" -DMLO_FILTER_PAD0=") + std::to_string(problem.GetPadW()) +
            std::string(" -DMLO_FILTER_PAD1=") + std::to_string(problem.GetPadH()) +
            std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string(kernel_stride_w) +
            std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string(kernel_stride_h) +
            std::string(" -DMLO_FILTER0_STRIDE0=") + std::to_string(kernel0_stride0) +
            std::string(" -DMLO_FILTER0_STRIDE1=") + std::to_string(kernel0_stride1) +
            std::string(" -DMLO_N_OUTPUTS=") + std::to_string(problem.GetInChannels()) +
            std::string(" -DMLO_N_INPUTS=") + std::to_string(problem.GetOutChannels()) +
            std::string(" -DMLO_BATCH_SZ=") + std::to_string(problem.GetBatchSize()) +
            std::string(" -DMLO_IN_WIDTH=") + std::to_string(in_width) +
            std::string(" -DMLO_IN_HEIGHT=") + std::to_string(in_height) +
            std::string(" -DMLO_OUT_WIDTH=") + std::to_string(problem.GetInWidth()) +
            std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(problem.GetInHeight()) +
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
            std::to_string((n_passes == 1) ? 0 : 1) + ctx.general_compile_options;

        if(n_passes > 1 && problem.GetPadH() == 0 && problem.GetPadW() == 0 &&
           (problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1))
        {
            KernelInfo kernel;

            kernel.l_wk.push_back(n_grp0_size0);
            kernel.l_wk.push_back(1);
            kernel.l_wk.push_back(1);
            // output is number of subsampled input maps
            size_t gbl_wk0 = (in_batch_stride / write_unit);
            size_t gbl_wk1 = problem.GetBatchSize();
            size_t gbl_wk2 = 1;

            kernel.g_wk.push_back(gbl_wk0);
            kernel.g_wk.push_back(gbl_wk1);
            kernel.g_wk.push_back(gbl_wk2);

            kernel.kernel_file = "MIOpenUtilKernels3.cl";

            kernel.kernel_name = "SubSample";

            kernel.comp_options = comp_options;

            result.construction_params.push_back(kernel);
        }

        const auto ws_sz    = GetWorkspaceSize(ctx, problem);
        result.workspace_sz = ws_sz;

        {
            // std::cout << comp_options << std::endl;
            int grp_tile2 = 1;
            KernelInfo kernel;

            kernel.l_wk.push_back(result.grp_tile0);
            kernel.l_wk.push_back(result.grp_tile1);
            kernel.l_wk.push_back(grp_tile2);
            // input is output

            // Traverse Smaller Batch_stride first
            size_t gbl_wk0 = static_cast<std::size_t>(n_grp_size0) * n_out_blocks;
            size_t gbl_wk1 = n_in_blocks;
            size_t gbl_wk2 = 1;

            if(in_batch_stride < out_batch_stride)
            {
                gbl_wk0 = static_cast<std::size_t>(n_grp_size0) * n_in_blocks;
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
                    const auto ss_kernel   = handle.Run(kernels[0]);
                    const auto main_kernel = handle.Run(kernels[1]);
                    const auto& invoke_params =
                        primitive_params.CastTo<miopen::conv::WrWInvokeParams>();

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
                    const auto k = handle.Run(kernels[0]);
                    const auto& invoke_params =
                        primitive_params.CastTo<miopen::conv::WrWInvokeParams>();
                    const auto& tensors    = invoke_params.tensors;
                    const auto padding_val = 0.f;

                    visit_float(tensors.dyDesc.GetType(), [&](auto as_float) {
                        k(tensors.dy, tensors.x, tensors.dw, as_float(padding_val));
                    });
                };
            };
        }
    }
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
