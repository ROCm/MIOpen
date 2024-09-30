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
#include <miopen/handle.hpp>
#include <miopen/env.hpp>
#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWDGEN)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvOclDirectFwdGen::IsApplicable(const ExecutionContext& ctx,
                                       const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWDGEN))
        return false;
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;
    if(!ctx.use_opencl_convolutions)
        return false;
    if(!problem.Is2d())
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
    if(problem.GetGroupCount() > 1)
        return false;

    return problem.IsDirectionForward()                                                 //
           && problem.GetKernelStrideW() == problem.GetKernelStrideH()                  //
           && problem.GetPadW() == problem.GetPadH()                                    //
           && problem.GetDilationW() == 1                                               //
           && problem.GetDilationH() == 1                                               //
           && (problem.GetWeightsWidth() > 11                                           //
               || problem.GetWeightsHeight() > 11                                       //
               || (!(problem.GetWeightsWidth() == 1 && problem.GetWeightsHeight() == 1) //
                   && (problem.GetKernelStrideW() > 1 || problem.GetKernelStrideH() > 1)));
}

ConvSolution ConvOclDirectFwdGen::GetSolution(const ExecutionContext& ctx,
                                              const ProblemDescription& problem) const
{
    int n_in_stacks = 0;
    if(problem.GetWeightsHeight() == 3 && problem.GetWeightsWidth() == 3)
    {
        // n of input batches
        n_in_stacks = ((problem.GetBatchSize() / 4) * 4 == problem.GetBatchSize())   ? 4
                      : ((problem.GetBatchSize() / 2) * 2 == problem.GetBatchSize()) ? 2
                                                                                     : 1;
    }
    else
    {
        // n of input batches
        n_in_stacks = ((problem.GetBatchSize() / 2) * 2 == problem.GetBatchSize()) ? 2 : 1;
    }
    int n_proc_supertiles = n_in_stacks; // n of prosessing groups
    auto lg2n_proc_supertiles =
        static_cast<int>(std::ceil(std::log(n_proc_supertiles) / std::log(2)));
    int n_out_stacks      = 1; // n of output sets
    int n_proc_supertile0 = ((n_in_stacks > 1) ? 32 : 16) /
                            problem.GetKernelStrideW(); // n  processor in process supertile
    int n_proc_supertile1 =
        ((n_in_stacks > 1 && (problem.GetWeightsHeight() >= 11 || problem.GetWeightsWidth() >= 11))
             ? 32
             : 16) /
        n_in_stacks;
    auto lg2n_proc_supertile1 =
        static_cast<int>(std::ceil(std::log(n_proc_supertile1) / std::log(2)));
    int ocl_group_sz0 = n_proc_supertile0;
    int ocl_group_sz1 = n_proc_supertile1 * n_proc_supertiles;
    int ocl_group_sz2 = 1;
    int gbl0          = 0;
    int gbl1          = 0;
    int gbl2          = 0;

    int n_ins0 = 1; // number of inputs each a from different stack along dim 0
    int n_ins1 = 1; // number of inputs each a from different stack along dim 1

    int n_outs          = (problem.GetInWidth() >= 384 ||
                  (problem.GetWeightsWidth() >= 11 && problem.GetKernelStrideW() >= 4))
                              ? 16
                              : 32; // n outputs per a single input: major parameter
    int n_out_pix_horiz = (problem.GetInWidth() < 320 ||
                           (problem.GetWeightsWidth() >= 11 && problem.GetKernelStrideW() >= 4))
                              ? 1
                              : 2; // n of output px horix per wk-item: major parameter
    int n_out_pix_vert  = 1;       // n of output px horix per wk-item: major parameter

    int n_in_pix_horiz = n_out_pix_horiz; // n of input pix per wk_item
    int n_in_pix_vert  = n_out_pix_vert;  // n of input pix per wk_item
    int n_v_proc0      = (problem.GetOutWidth() + n_out_pix_horiz - 1) / n_out_pix_horiz;
    int n_v_proc1      = (problem.GetOutHeight() + n_out_pix_vert - 1) / n_out_pix_vert;

    int big = 1;

    int n_procs0 = n_proc_supertile0 / n_ins0;
    int n_procs1 = n_proc_supertile1 / n_ins1;

    int in_sz0 = (n_procs0 * n_out_pix_horiz - 1) * problem.GetKernelStrideW() +
                 1 /* + kernel_size_w - 2 * pad_w*/;
    int in_sz1 = (n_procs1 * n_out_pix_vert - 1) * problem.GetKernelStrideH() +
                 1 /* + kernel_size_h - 2 * pad_h*/;

    int n_ins = n_ins0 * n_ins1; // number of inputs each a from different stack

    n_outs = std::min(n_outs, static_cast<int>(problem.GetOutChannels()));
    n_ins  = std::min(n_ins, static_cast<int>(problem.GetBatchSize()));

    n_out_stacks = (static_cast<std::size_t>(n_outs) * n_out_stacks <= problem.GetOutChannels())
                       ? n_out_stacks
                       : 1;
    n_in_stacks =
        (static_cast<std::size_t>(n_ins) * n_in_stacks <= problem.GetBatchSize()) ? n_in_stacks : 1;
    int total_ins  = n_ins * n_in_stacks;
    int total_outs = n_outs * n_out_stacks;

    int n_out_blocks   = ((problem.GetOutChannels() + total_outs - 1) / total_outs);
    int n_stack_blocks = ((problem.GetBatchSize() + total_ins - 1) / total_ins);

    int batch_aligned = 0;
#if 1
    if((problem.GetBatchSize() / n_stack_blocks) * n_stack_blocks == problem.GetBatchSize())
    {
        batch_aligned = 1;
    }
#endif
    int out_aligned = 0;
#if 1
    if((problem.GetOutChannels() / total_outs) * total_outs == problem.GetOutChannels())
    {
        out_aligned = 1;
    }
#endif

    // global work size
    gbl0 = n_ins0 * ((n_v_proc0 + n_procs0 - 1) / (n_procs0)) * n_procs0;
    gbl1 = n_ins1 * ((n_v_proc1 + n_procs1 - 1) / (n_procs1)) * n_procs1 * n_proc_supertiles;
    gbl2 = n_out_blocks * n_stack_blocks;

    int aligned_out = 1;

    if(gbl0 != n_ins0 * n_v_proc0 || gbl1 != n_ins1 * n_v_proc1)
    {
        aligned_out = 0;
    }

    int bias = problem.GetBias();
    KernelInfo construction_params;

    construction_params.comp_options =
        std::string("-DMLO_GRP_SZ=") +
        std::to_string(static_cast<long long>(ocl_group_sz0) * ocl_group_sz1 * ocl_group_sz2) +
        std::string(" -DMLO_GRP_SZ0=") + std::to_string(static_cast<long long>(ocl_group_sz0)) +
        std::string(" -DMLO_GRP_SZ1=") + std::to_string(static_cast<long long>(ocl_group_sz1)) +
        std::string(" -DMLO_GRP_SZ2=") + std::to_string(static_cast<long long>(ocl_group_sz2)) +
        std::string(" -DMLO_LCL_N_IN_CHNLS=") + std::to_string(static_cast<long long>(n_ins)) +
        std::string(" -DMLO_LCL_N_OUT_CHNLS=") + std::to_string(static_cast<long long>(n_outs)) +
        std::string(" -DMLO_OUT_STACKS=") + std::to_string(static_cast<long long>(n_out_stacks)) +
        std::string(" -DMLO_IN_STACKS=") + std::to_string(static_cast<long long>(n_in_stacks)) +
        std::string(" -DMLO_BATCH_SZ=") +
        std::to_string(static_cast<long long>(problem.GetBatchSize())) +
        std::string(" -DMLO_FLTR_SZ0=") +
        std::to_string(static_cast<long long>(problem.GetWeightsWidth())) +
        std::string(" -DMLO_FLTR_PAD_SZ0=") +
        std::to_string(static_cast<long long>(problem.GetPadW())) +
        std::string(" -DMLO_FLTR_STRIDE0=") +
        std::to_string(static_cast<long long>(problem.GetKernelStrideW())) +
        std::string(" -DMLO_FLTR_SZ1=") +
        std::to_string(static_cast<long long>(problem.GetWeightsHeight())) +
        std::string(" -DMLO_FLTR_PAD_SZ1=") +
        std::to_string(static_cast<long long>(problem.GetPadH())) +
        std::string(" -DMLO_FLTR_STRIDE1=") +
        std::to_string(static_cast<long long>(problem.GetKernelStrideH())) +
        std::string(" -DMLO_N_OUT_CHNLS=") +
        std::to_string(
            static_cast<long long>(problem.GetOutChannels())) // total number of output channels
        + std::string(" -DMLO_OUT_WIDTH=") +
        std::to_string(static_cast<long long>(problem.GetOutWidth())) +
        std::string(" -DMLO_OUT_HEIGHT=") +
        std::to_string(static_cast<long long>(problem.GetOutHeight())) +
        std::string(" -DMLO_OUT_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetOutStrideH())) +
        std::string(" -DMLO_OUT_CHNL_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetOutChannelStride())) +
        std::string(" -DMLO_OUT_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetOutBatchStride())) +
        std::string(" -DMLO_N_OUT_PIX_SZ0=") +
        std::to_string(static_cast<long long>(n_out_pix_horiz)) +
        std::string(" -DMLO_N_OUT_PIX_SZ1=") +
        std::to_string(static_cast<long long>(n_out_pix_vert)) + std::string(" -DMLO_N_IN_CHNLS=") +
        std::to_string(static_cast<long long>(problem.GetInChannels())) +
        std::string(" -DMLO_IN_WIDTH=") +
        std::to_string(static_cast<long long>(problem.GetInWidth())) +
        std::string(" -DMLO_IN_HEIGHT=") +
        std::to_string(static_cast<long long>(problem.GetInHeight())) +
        std::string(" -DMLO_IN_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetInStrideH())) +
        std::string(" -DMLO_IN_CHNL_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetInChannelStride())) +
        std::string(" -DMLO_IN_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetInBatchStride())) +
        std::string(" -DMLO_N_IN_PIX_SZ0=") +
        std::to_string(
            static_cast<long long>(n_in_pix_horiz)) // size of output processing group in 0 dim
        + std::string(" -DMLO_N_IN_PIX_SZ1=") +
        std::to_string(
            static_cast<long long>(n_in_pix_vert)) // size of output processing group in 1 dim
        + std::string(" -DMLO_WEI_SZ=") +
        std::to_string(static_cast<long long>(problem.GetOutChannels()) * problem.GetInChannels() *
                       problem.GetWeightsWidth() * problem.GetWeightsHeight()) +
        std::string(" -DMLO_WEIGHTS_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetInChannels()) * problem.GetWeightsWidth() *
                       problem.GetWeightsHeight()) //	weights stride
        + std::string(" -DMLO_N_STACKS=") +
        std::to_string(static_cast<long long>(n_stack_blocks)) // n of separate data stacks
        + std::string(" -DMLO_N_PROCS0=") +
        std::to_string(static_cast<long long>(n_procs0)) // n of processors per stack
        + std::string(" -DMLO_N_PROCS1=") +
        std::to_string(static_cast<long long>(n_procs1)) // n of processors per stack
        + std::string(" -DMLO_ALIGNED=") +
        std::to_string(static_cast<long long>(aligned_out)) //	dimesions aligned
        + std::string(" -DMLO_BATCH_ALIGNED=") +
        std::to_string(static_cast<long long>(batch_aligned)) // batch is multiple of n_ins
        + std::string(" -DMLO_OUT_ALINED=") +
        std::to_string(static_cast<long long>(out_aligned)) // outputs is multiple of n_outs
        + std::string(" -DMLO_IN_SZ0=") +
        std::to_string(static_cast<long long>(in_sz0)) // horizontal read dim 0
        + std::string(" -DMLO_IN_SZ1=") +
        std::to_string(static_cast<long long>(in_sz1)) // vertical read dim 1
        + std::string(" -DMLO_LG2N_PROC_TILES=") +
        std::to_string(static_cast<long long>(lg2n_proc_supertiles)) +
        std::string(" -DMLO_LG2N_PROC_TILE1=") +
        std::to_string(static_cast<long long>(lg2n_proc_supertile1)) + std::string(" -DMLO_BIG=") +
        std::to_string(static_cast<long long>(big)) //	resolution > 32 x 32
        + std::string(" -DMLO_CONV_BIAS=") +
        std::to_string(static_cast<long long>(bias))

        //		+ std::string(" -limit-vector-registers=64 ")

        + ctx.general_compile_options;

    construction_params.kernel_file = "MIOpenConvDirGenFwd.cl";
    construction_params.kernel_name = (n_proc_supertiles == 1) ? "MIOpenCDFGen" : "MIOpenCDFGen4";

    construction_params.l_wk.push_back(ocl_group_sz0);
    construction_params.l_wk.push_back(ocl_group_sz1);
    construction_params.l_wk.push_back(ocl_group_sz2);

    construction_params.g_wk.push_back(gbl0);
    construction_params.g_wk.push_back(gbl1);
    construction_params.g_wk.push_back(gbl2);

    ConvSolution result;
    result.construction_params.push_back(construction_params);
    result.invoker_factory = &miopen::conv::MakeGenericXWYPadInvoker;
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
