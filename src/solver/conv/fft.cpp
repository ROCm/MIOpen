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

#include <miopen/algorithm.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/convolution_fft.hpp>
#include <miopen/env.hpp>
#include <miopen/tensor.hpp>

#include <boost/any.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_FFT)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

static void cgemm_grid(size_t* global_work_size,
                       size_t* local_work_size,
                       int cgemm_choice,
                       const int N,
                       const int out_c,
                       const int out_n)
{
    unsigned int threadTile[2];
    unsigned int groupSize[2];

    // grid for cgemm
    if(cgemm_choice == 1)
    {
        threadTile[0] = 4;
        threadTile[1] = 4;

        groupSize[0] = 16;
        groupSize[1] = 16;

        local_work_size[0] = 16;
        local_work_size[1] = 16;
    }
    else if(cgemm_choice == 2)
    {
        threadTile[0] = 4;
        threadTile[1] = 4;

        groupSize[0] = 4;
        groupSize[1] = 4;

        local_work_size[0] = 64;
        local_work_size[1] = 1;
    }
    else
    {
        threadTile[0] = 2;
        threadTile[1] = 2;

        groupSize[0] = 4;
        groupSize[1] = 4;

        local_work_size[0] = 64;
        local_work_size[1] = 1;
    }

    global_work_size[2] = 1;
    global_work_size[2] *= N;

    unsigned int sizeOfC0         = out_c;
    unsigned int sizeOfC1         = out_n;
    auto macroTile0               = static_cast<unsigned int>(groupSize[0] * threadTile[0]);
    auto macroTile1               = static_cast<unsigned int>(groupSize[1] * threadTile[1]);
    unsigned int totalWorkGroups0 = sizeOfC0 / macroTile0;
    unsigned int totalWorkGroups1 = sizeOfC1 / macroTile1;
    // b/c single kernel, add extra work-group here if edge needed
    if(totalWorkGroups0 * macroTile0 < sizeOfC0)
    {
        totalWorkGroups0++;
    }
    if(totalWorkGroups1 * macroTile1 < sizeOfC1)
    {
        totalWorkGroups1++;
    }
    global_work_size[0] = totalWorkGroups0 * local_work_size[0];
    global_work_size[1] = totalWorkGroups1 * local_work_size[1];
}

bool fft::IsApplicable(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    std::ignore = ctx;

    // disable running any FFT based convolutions by checking this env variable
    if(problem.IsDirectionBackwardWrW() || !problem.IsFp32())
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    if(!problem.AllTensorsDimsFitIntoInt())
        return false;

    const auto is_fwd    = problem.IsDirectionForward();
    decltype(auto) conv  = problem.GetConv();
    decltype(auto) xDesc = is_fwd ? problem.GetIn() : problem.GetOut();
    decltype(auto) yDesc = is_fwd ? problem.GetOut() : problem.GetIn();
    decltype(auto) wDesc = problem.GetWeights();

    if(conv.GetSpatialDimension() != 2 || conv.group_count != 1 ||
       !miopen::all_of(conv.GetConvDilations(), [](auto v) { return v == 1; }))
        return false;

    int in_n, in_c, in_h, in_w;
    int out_n, out_c, out_h, out_w;
    int wei_k, wei_c, wei_h, wei_w;
    std::tie(in_n, in_c, in_h, in_w)     = miopen::tien<4>(xDesc.GetLengths());
    std::tie(out_n, out_c, out_h, out_w) = miopen::tien<4>(yDesc.GetLengths());
    std::tie(wei_k, wei_c, wei_h, wei_w) = miopen::tien<4>(wDesc.GetLengths());

    // FFT convolutions only works for specific config(s)
    // coverage to expand gradually

    if((in_n < 1) || (in_n > 512) || (wei_k < 1) || (wei_k > 512) || ((in_c * in_n) % 16 != 0) ||
       (wei_c * wei_k) % 16 != 0 || (out_c * out_n) % 16 != 0)
        return false;

    if((std::tie(in_h, in_w) != std::make_tuple(28, 28)) &&
       (std::tie(in_h, in_w) != std::make_tuple(27, 27)) &&
       (std::tie(in_h, in_w) != std::make_tuple(14, 14)) &&
       (std::tie(in_h, in_w) != std::make_tuple(7, 7)))
        return false;

    const auto cparam = std::make_tuple(conv.GetConvPads()[0],
                                        conv.GetConvPads()[1],
                                        conv.GetConvStrides()[0],
                                        conv.GetConvStrides()[1]);

    return std::tie(wei_h, wei_w) == std::make_tuple(5, 5) && cparam == std::make_tuple(2, 2, 1, 1);
}

size_t fft::GetWorkspaceSize(const ExecutionContext&, const ProblemDescription& problem) const
{
    const auto fwd       = problem.IsDirectionForward();
    decltype(auto) xDesc = fwd ? problem.GetIn() : problem.GetOut();
    decltype(auto) yDesc = fwd ? problem.GetOut() : problem.GetIn();
    decltype(auto) wDesc = problem.GetWeights();

    int in_n, in_c, in_h, in_w;
    std::tie(in_n, in_c, in_h, in_w) = miopen::tien<4>(xDesc.GetLengths());

    int out_n, out_c, out_h, out_w;
    std::tie(out_n, out_c, out_h, out_w) = miopen::tien<4>(yDesc.GetLengths());

    int wei_k, wei_c, wei_h, wei_w;
    std::tie(wei_k, wei_c, wei_h, wei_w) = miopen::tien<4>(wDesc.GetLengths());

    // FFT convolutions only works for specific config(s)
    // coverage to expand gradually

    const int N       = FFTConvParams::TileSize(in_h, in_w);
    const int Padding = FFTConvParams::TransposePadding;

    int temp_size = 0;

    if(fwd)
    {
        int temp_size1 = (in_c * in_n + Padding) + (wei_k * wei_c + Padding);
        int temp_size2 = (out_n * out_c + Padding);
        temp_size      = std::max(temp_size1, temp_size2);
    }
    else
    {
        int temp_size1 = (out_n * out_c + Padding) + (wei_k * wei_c + Padding);
        int temp_size2 = (in_c * in_n + Padding);
        temp_size      = std::max(temp_size1, temp_size2);
    }

    return sizeof(float) * 2 * 2 * N * temp_size;
}

ConvSolution fft::GetSolution(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    std::ignore = ctx;

    int in_n  = problem.GetBatchSize();
    int in_c  = problem.GetInChannels();
    int in_h  = problem.GetInHeight();
    int in_w  = problem.GetInWidth();
    int out_n = problem.GetBatchSize();
    int out_c = problem.GetOutChannels();

    const int N          = FFTConvParams::TileSize(in_h, in_w);
    const int NumKernels = FFTConvParams::NumKernels;

    size_t global_work_size[NumKernels][3];
    size_t local_work_size[NumKernels][3];

    for(int ik = 0; ik < NumKernels; ik++)
    {
        global_work_size[ik][0] = local_work_size[ik][0] = 1;
        global_work_size[ik][1] = local_work_size[ik][1] = 1;
        global_work_size[ik][2] = local_work_size[ik][2] = 1;
    }

    // grid for FFT kernels
    if((in_h == 7) && (in_w == 7))
    {
        local_work_size[0][0]  = 192;
        global_work_size[0][0] = ((in_c * out_n) / 16) * local_work_size[0][0];

        local_work_size[1][0]  = 192;
        global_work_size[1][0] = ((in_c * out_c) / 16) * local_work_size[1][0];

        local_work_size[6][0]  = 192;
        global_work_size[6][0] = ((out_n * out_c) / 16) * local_work_size[6][0];
    }
    else if((in_h == 14) && (in_w == 14))
    {
        local_work_size[0][0]  = 128;
        global_work_size[0][0] = ((in_c * out_n) / 4) * local_work_size[0][0];

        local_work_size[1][0]  = 128;
        global_work_size[1][0] = ((in_c * out_c) / 4) * local_work_size[1][0];

        local_work_size[6][0]  = 128;
        global_work_size[6][0] = ((out_n * out_c) / 4) * local_work_size[6][0];
    }
    else
    {
        local_work_size[0][0]  = 64;
        global_work_size[0][0] = local_work_size[0][0] * in_c * out_n;

        local_work_size[1][0]  = 64;
        global_work_size[1][0] = local_work_size[1][0] * in_c * out_c;

        local_work_size[6][0]  = 64;
        global_work_size[6][0] = local_work_size[6][0] * out_n * out_c;
    }

    // decide tranpose kernel options based on params
    int in_tranpose_choice = 0;
    int wt_tranpose_choice = 0;
    int ot_tranpose_choice = 0;

    // grid for transpose kernels
    if((in_h == 7) && (in_w == 7))
    {
        local_work_size[5][0] = 256;
        global_work_size[5][0] =
            static_cast<size_t>((1 + N / 16) * (out_n * out_c / 16)) * local_work_size[5][0];
    }
    else if((in_h == 14) && (in_w == 14))
    {
        local_work_size[2][0] = 256;
        global_work_size[2][0] =
            static_cast<size_t>((1 + N / 16) * (in_c * out_n / 16)) * local_work_size[2][0];

        local_work_size[3][0] = 256;
        global_work_size[3][0] =
            static_cast<size_t>((1 + N / 16) * (in_c * out_c / 16)) * local_work_size[3][0];

        local_work_size[5][0] = 256;
        global_work_size[5][0] =
            static_cast<size_t>((1 + N / 16) * (out_n * out_c / 16)) * local_work_size[5][0];
    }
    else
    {
        if((in_n * in_c >= 64) && ((in_n * in_c) % 32 == 0))
            in_tranpose_choice = 1;
        if((out_c * in_c >= 64) && ((out_c * in_c) % 32 == 0))
            wt_tranpose_choice = 1;
        if((out_n * out_c >= 64) && ((out_n * out_c) % 32 == 0))
            ot_tranpose_choice = 1;

        int in_tranpose_bwidth = in_tranpose_choice != 0 ? 32 : 16;
        int wt_tranpose_bwidth = wt_tranpose_choice != 0 ? 32 : 16;
        int ot_tranpose_bwidth = ot_tranpose_choice != 0 ? 32 : 16;

        local_work_size[2][0] = 256;
        global_work_size[2][0] =
            static_cast<size_t>((N / in_tranpose_bwidth) * (in_c * out_n / in_tranpose_bwidth)) *
            local_work_size[2][0];

        local_work_size[3][0] = 256;
        global_work_size[3][0] =
            static_cast<size_t>((N / wt_tranpose_bwidth) * (in_c * out_c / wt_tranpose_bwidth)) *
            local_work_size[3][0];

        local_work_size[5][0] = 256;
        global_work_size[5][0] =
            static_cast<size_t>((N / ot_tranpose_bwidth) * (out_n * out_c / ot_tranpose_bwidth)) *
            local_work_size[5][0];
    }

    // cgemm kernel options
    int cgemm_choice = 0;

    if(((in_h == 28) && (in_w == 28)) || ((in_h == 14) && (in_w == 14)) ||
       ((in_h == 7) && (in_w == 7)))
    {
        cgemm_choice = 2;
    }
    else if((in_h == 27) && (in_w == 27))
    {
        cgemm_choice = 1;
    }

    if((in_n < 16) || (in_c < 16) || (out_c < 16))
        cgemm_choice = 0;

    cgemm_grid(global_work_size[4], local_work_size[4], cgemm_choice, N, out_c, out_n);

    std::string parms;

    if(in_tranpose_choice == 0)
        parms += " -DCFF_TRANSP_IN_MOD16=1";
    if(wt_tranpose_choice == 0)
        parms += " -DCFF_TRANSP_WT_MOD16=1";
    if(ot_tranpose_choice == 0)
        parms += " -DCFF_TRANSP_OT_MOD16=1";

    switch(cgemm_choice)
    {
    case 1: parms += " -DCFF_CGEMM_CHOICE_1=1"; break;
    case 2: parms += " -DCFF_CGEMM_CHOICE_2=1"; break;
    default: parms += " -DCFF_CGEMM_CHOICE_0=1"; break;
    }

    if((in_h == 28) && (in_w == 28))
    {
        parms += " -DCFF_IMG_SZ_28_28";
    }
    else if((in_h == 27) && (in_w == 27))
    {
        parms += " -DCFF_IMG_SZ_27_27";
    }
    else if((in_h == 14) && (in_w == 14))
    {
        parms += " -DCFF_IMG_SZ_14_14";
    }
    else if((in_h == 7) && (in_w == 7))
    {
        parms += " -DCFF_IMG_SZ_7_7";
    }

    const auto workSpaceSize = GetWorkspaceSize(ctx, problem);

    parms += " -DCFF_IMG_H=";
    parms += std::to_string(in_h);
    parms += " -DCFF_IMG_W=";
    parms += std::to_string(in_w);
    parms += " -DCFF_BATCH=";
    parms += std::to_string(in_n);
    parms += " -DCFF_NFILTER=";
    parms += std::to_string(out_c);
    parms += " -DCFF_CHANNELS=";
    parms += std::to_string(in_c);
    parms += " -DCFF_HALFW=";
    parms += std::to_string(workSpaceSize / (sizeof(float) * 2 * 2));

    if(!problem.IsDirectionForward())
    {
        parms += " -DCFF_BACKWARD";
    }

    const std::string algorithm    = "miopenConvolutionFwdAlgoFFT";
    const std::string program_name = "MIOpenConvFFT.cl";

    auto sol         = ConvSolution{miopenStatusSuccess};
    sol.workspace_sz = workSpaceSize;

    // skip front transposes for 7x7
    const auto skip_front_transposes = (in_h == 7) && (in_w == 7);

    for(int ik = 0; ik < NumKernels; ik++)
    {
        if(skip_front_transposes && ((ik == 2) || (ik == 3)))
            continue;

        const auto kernel_name = [=]() {
            switch(ik)
            {
            case 0: return "MIOpenConvFFT_fwd_in";
            case 1: return "MIOpenConvFFT_fwd_we";
            case 2: return "MIOpenConvFFT_transpose_in";
            case 3: return "MIOpenConvFFT_transpose_we";
            case 4: return "MIOpenConvFFT_cgemm";
            case 5: return "MIOpenConvFFT_transpose_out";
            case 6: return "MIOpenConvFFT_inv_out";
            default: assert(false); return "";
            }
        }();

        std::vector<size_t> vld(3);
        std::vector<size_t> vgd(3);

        vld[0] = local_work_size[ik][0];
        vld[1] = local_work_size[ik][1];
        vld[2] = local_work_size[ik][2];

        vgd[0] = global_work_size[ik][0];
        vgd[1] = global_work_size[ik][1];
        vgd[2] = global_work_size[ik][2];

        auto kernel         = KernelInfo{};
        kernel.kernel_file  = program_name;
        kernel.kernel_name  = kernel_name;
        kernel.comp_options = parms;
        kernel.g_wk         = vgd;
        kernel.l_wk         = vld;
        sol.construction_params.push_back(kernel);
    }

    sol.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        int halfw = static_cast<int>(workSpaceSize) / (2 * 2 * static_cast<int>(sizeof(float)));
        const int padding = FFTConvParams::TransposePadding;

        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& params  = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors = params.tensors;

            if(params.workSpaceSize < workSpaceSize)
            {
                MIOPEN_THROW("Not enough workspace for FFT: expected " +
                             std::to_string(workSpaceSize) + ", got " +
                             std::to_string(params.workSpaceSize));
            }

            float time_fft = 0;
            int kernel_id  = 0;
            for(int ik = 0; ik < NumKernels; ik++)
            {
                if(skip_front_transposes && ((ik == 2) || (ik == 3)))
                    continue;

                const auto& k = handle.Run(kernels[kernel_id++]);

                switch(ik)
                {
                case 0: k(tensors.in, params.workSpace); break;
                case 1: k(tensors.w, params.workSpace); break;
                case 4: {
                    k(params.workSpace,
                      0,
                      halfw + N * (in_n * in_c + padding),
                      halfw + 0,
                      out_c,
                      out_n * out_c + padding,
                      in_c,
                      in_c * out_c + padding,
                      in_c,
                      in_n * in_c + padding,
                      out_c,
                      in_n,
                      N,
                      in_c);
                    break;
                }
                case 6: k(params.workSpace, tensors.out); break;
                case 2:
                case 3:
                case 5: k(params.workSpace); break;
                default: assert(false);
                }

                if(handle.IsProfilingEnabled())
                    time_fft += handle.GetKernelTime();
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(time_fft);
            }
        };
    };

    return sol;
}

} // namespace conv
} // namespace solver
} // namespace miopen
