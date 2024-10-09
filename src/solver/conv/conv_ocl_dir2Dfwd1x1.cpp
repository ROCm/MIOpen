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

#include <miopen/handle.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>
#include <miopen/stringutils.hpp>

#define WORKAROUND_SWDEV_271887 1

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvOclDirectFwd1x1::IsApplicable(const ExecutionContext& ctx,
                                       const ProblemDescription& problem) const
{
#if WORKAROUND_SWDEV_271887
    if(StartsWith(ctx.GetStream().GetDeviceName(), "gfx10") ||
       StartsWith(ctx.GetStream().GetDeviceName(), "gfx11"))
    {
        if(!env::enabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1))
            return false;
    }
#endif
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1))
        return false;
    if(!ctx.use_opencl_convolutions)
        return false;
    if(!problem.Is2d())
        return false;
    if(!(problem.IsDirectionForward() || problem.IsDirectionBackwardData()))
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsAsymmetricPadH() || problem.IsAsymmetricPadW())
        return false;
    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16()))
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.IsLayoutDefault())
        return false;

    return problem.GetDilationW() == 1 && problem.GetDilationH() == 1 &&
           problem.GetWeightsWidth() == 1 && problem.GetWeightsHeight() == 1 &&
           problem.GetGroupCount() == 1 &&
           // TODO: update 1x1 fwd kernel to support padding
           problem.GetPadW() == 0 && problem.GetPadH() == 0;
}

ConvSolution ConvOclDirectFwd1x1::GetSolution(const ExecutionContext& ctx,
                                              const ProblemDescription& problem,
                                              const LegacyPerformanceConfig& config) const
{
    ConvSolution result;
    config.CopyTo(result);

    //   if(problem.GetOutChannels() % 4 == 0 && problem.GetInChannels() % 4 == 0)
    {
        //        int version = result.out_pix_tile1;

        if((problem.IsDirectionForward() && problem.GetInChannels() % 16 == 0 &&
            problem.GetOutChannels() % 16 == 0) &&
           (problem.GetInDataType() == miopenFloat))
        {

            int N_LCL_IN_MAPS = result.n_in_data_tiles;

            int N_LCL_OUT_MAPS = result.n_out_pix_tiles;
            // 0 or 1
            int CHEAT_SHADER_COMPILER = result.out_pix_tile0;

            int BATCHSIZE = problem.GetBatchSize();
            int W         = problem.GetInWidth();
            int H         = problem.GetInHeight();
            int C         = problem.GetInChannels();
            int K         = problem.GetOutChannels();
            int W_out     = problem.GetOutWidth();
            int H_out     = problem.GetOutHeight();

            N_LCL_IN_MAPS  = std::min(N_LCL_IN_MAPS, C);
            N_LCL_OUT_MAPS = std::min(N_LCL_OUT_MAPS, K);

            while((K % N_LCL_OUT_MAPS) != 0 && N_LCL_OUT_MAPS > 16)
            {
                N_LCL_OUT_MAPS /= 2;
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

            int N_IN_GROUPS        = (C + N_LCL_IN_MAPS - 1) / N_LCL_IN_MAPS;
            int N_LCL_IN_MAPS_ONCE = 8;

            if(problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1)
                N_LCL_IN_MAPS_ONCE = 4;

            int CLOOP0 = N_LCL_IN_MAPS / N_LCL_IN_MAPS_ONCE;
            int CLOOP2 = (C - N_LCL_IN_MAPS * (N_IN_GROUPS - 1)) / N_LCL_IN_MAPS_ONCE;

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
                ctx.general_compile_options;

            kernel.comp_options = std::string(" -DMLO_FILTER_STRIDE0=") +
                                  std::to_string(problem.GetKernelStrideW()) +
                                  std::string(" -DMLO_FILTER_STRIDE1=") +
                                  std::to_string(problem.GetKernelStrideH()) + kernel.comp_options;

            // std::cout << "compile options:\n"<< _comp_options << std::endl;

            // 1x1_Stride: FIX ME!!! NO padding support
            if(problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1)
            {
                int FIXED_WORKGROUP_SIZE = 64;

                size_t N_OUT_GROUPS = (K / N_LCL_OUT_MAPS);

                size_t local_wk1 = 1;
                kernel.l_wk.push_back(FIXED_WORKGROUP_SIZE);
                kernel.l_wk.push_back(local_wk1);
                kernel.l_wk.push_back(1);

                size_t imagesizeAlign =
                    ((problem.GetOutWidth() * problem.GetOutHeight() * problem.GetBatchSize() +
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

                size_t imagesizeAlign =
                    ((problem.GetInWidth() * problem.GetInHeight() * problem.GetBatchSize() +
                      FIXED_WORKGROUP_SIZE - 1) /
                     FIXED_WORKGROUP_SIZE) *
                    FIXED_WORKGROUP_SIZE;
                size_t N_OUT_GROUPS = (K / N_LCL_OUT_MAPS);

                size_t gbl_wk0 = imagesizeAlign * N_IN_GROUPS * N_OUT_GROUPS;

                size_t gbl_wk1 = 1;

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
            //	int i_sz = problem.GetInWidth() * problem.GetInHeight();
            //	_out_pix_tile0 = (i_sz & 1) ? 1 : 2;
            result.out_pix_tile0 =
                std::min(static_cast<int>(problem.GetOutWidth()), result.out_pix_tile0);
            result.out_pix_tile1 =
                std::min(static_cast<int>(problem.GetOutHeight()), result.out_pix_tile1);
            if(!problem.IsDirectionForward())
            {
                while(problem.GetOutWidth() % result.out_pix_tile0 != 0 && result.out_pix_tile0 > 1)
                {
                    result.out_pix_tile0 /= 2;
                }
            }

            int read_unit = result.out_pix_tile0;
            while(problem.GetInWidth() % read_unit != 0 && read_unit > 1)
            {
                read_unit /= 2;
            }

            // problem.GetOutWidth()
            //	_n_out_pix_tiles = 16;
            //	_n_in_data_tiles = 4;
            //	_grp_tile0 = 64;

            int wei_cstride = problem.GetWeightsWidth() * problem.GetWeightsHeight();
            // backward: inputs are forward outputs
            const bool is_forward = problem.IsDirectionForward();
            int wei_bstride =
                (is_forward ? problem.GetInChannels() : problem.GetOutChannels()) * wei_cstride;

            int OUT_WIDTH4 = problem.GetOutWidth();
            int MAP_SZ4    = (OUT_WIDTH4 * problem.GetOutHeight() + read_unit - 1) / (read_unit);
            // stride > 1 and/or apdding
            if(problem.GetPadW() > 0 || problem.GetKernelStrideW() > 1 || problem.GetPadH() > 0 ||
               problem.GetKernelStrideH() > 1)
            {
                int step        = is_forward ? read_unit : read_unit * problem.GetKernelStrideW();
                OUT_WIDTH4      = (problem.GetOutWidth() + step - 1) / (step);
                int OUT_HEIGHT4 = is_forward
                                      ? problem.GetOutHeight()
                                      : (problem.GetOutHeight() + problem.GetKernelStrideH() - 1) /
                                            problem.GetKernelStrideH();
                MAP_SZ4         = (OUT_WIDTH4 * OUT_HEIGHT4);
            }

            int VERT_ALIGNED  = 1;
            int HORIZ_ALIGNED = 1;
            if(!is_forward)
            {
                VERT_ALIGNED =
                    (problem.GetOutHeight() / problem.GetKernelStrideH() == problem.GetInHeight())
                        ? 1
                        : 0;
                HORIZ_ALIGNED =
                    (problem.GetOutWidth() / problem.GetKernelStrideW() == problem.GetInWidth())
                        ? 1
                        : 0;
            }

            int GRP_SZ = result.grp_tile0;

            // number of inputs inside wk-items
            result.n_in_data_tiles =
                std::min(static_cast<int>(problem.GetInChannels()), result.n_in_data_tiles);
            while(problem.GetInChannels() % result.n_in_data_tiles != 0 &&
                  result.n_in_data_tiles > 1)
            {
                result.n_in_data_tiles /= 2;
            }

            int CLOOP0 =
                (problem.GetInChannels() + result.n_in_data_tiles - 1) / result.n_in_data_tiles;

            // number of outputs inside wk_item
            result.n_out_pix_tiles =
                std::min(static_cast<int>(problem.GetOutChannels()), result.n_out_pix_tiles);
            while(problem.GetOutChannels() % result.n_out_pix_tiles != 0 &&
                  result.n_out_pix_tiles > 1)
            {
                result.n_out_pix_tiles /= 2;
            }

            KernelInfo kernel;

            kernel.comp_options =
                std::string(" -DMLO_DIR_FORWARD=") + (is_forward ? "1" : "0") +
                std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(problem.GetWeightsWidth()) +
                std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(problem.GetWeightsHeight()) +
                std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string(problem.GetKernelStrideW()) +
                std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string(problem.GetKernelStrideH()) +
                std::string(" -DMLO_FILTER_PAD0=") + std::to_string(problem.GetPadW()) +
                std::string(" -DMLO_FILTER_PAD1=") + std::to_string(problem.GetPadH()) +
                std::string(" -DMLO_IN_WIDTH=") + std::to_string(problem.GetInWidth()) +
                std::string(" -DMLO_IN_HEIGHT=") + std::to_string(problem.GetInHeight()) +
                std::string(" -DMLO_OUT_WIDTH=") + std::to_string(problem.GetOutWidth()) +
                std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(problem.GetOutHeight()) +
                std::string(" -DMLO_N_OUTPUTS=") + std::to_string(problem.GetOutChannels()) +
                std::string(" -DMLO_N_INPUTS=") + std::to_string(problem.GetInChannels()) +
                std::string(" -DMLO_BATCH_SZ=") + std::to_string(problem.GetBatchSize()) +
                std::string(" -DMLO_OUT_BATCH_STRIDE=") +
                std::to_string(problem.GetOutBatchStride()) +
                std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
                std::to_string(problem.GetOutChannelStride()) + std::string(" -DMLO_OUT_STRIDE=") +
                std::to_string(problem.GetOutStrideH()) + std::string(" -DMLO_IN_BATCH_STRIDE=") +
                std::to_string(problem.GetInBatchStride()) +
                std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
                std::to_string(problem.GetInChannelStride()) + std::string(" -DMLO_IN_STRIDE=") +
                std::to_string(problem.GetInStrideH()) + std::string(" -DMLO_WEI_BSTRIDE=") +
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
                std::to_string(problem.GetBias()) +

                std::string(" -DMLO_READ_UNIT=") + std::to_string(read_unit) +
                std::string(" -DMLO_CLOOP0=") + std::to_string(CLOOP0) +

                ctx.general_compile_options;

            kernel.l_wk.push_back(result.grp_tile0);
            kernel.l_wk.push_back(1);
            kernel.l_wk.push_back(1);

            size_t gbl_wk0 = problem.GetBatchSize() * MAP_SZ4;

            size_t gbl_wk1 =
                (problem.GetOutChannels() + result.n_out_pix_tiles - 1) / result.n_out_pix_tiles;
            size_t gbl_wk2 = 1;

            kernel.g_wk.push_back(gbl_wk0);
            kernel.g_wk.push_back(gbl_wk1);
            kernel.g_wk.push_back(gbl_wk2);

            kernel.kernel_file = "MIOpenConv1x1S.cl";
            kernel.kernel_name =
                (problem.GetKernelStrideW() == 1 && problem.GetKernelStrideH() == 1)
                    ? "MIOpenConv1x1"
                    : "MIOpenConv1x1pquv";
            result.construction_params.push_back(kernel);
        }
    }

    result.invoker_factory = &miopen::conv::MakeGenericXWYPadInvoker;
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
