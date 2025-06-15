/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <miopen/solver/conv_direct_naive_conv.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/env.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/datatype.hpp>
#include <ostream>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <miopen/solver/ck_utility_common.hpp>
#include <ck/utility/data_type.hpp>
#endif

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DEPTH_WISE_CONV_WRW)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

struct PerfArgs
{
    ck::index_t block_size;
    ck::index_t tile_w;
    ck::index_t tile_h;
    ck::index_t filter_size;
    ck::index_t dilation_w;
    ck::index_t dilation_h;
    ck::index_t stride_w;
    ck::index_t stride_h;
    ck::index_t pad_w;
    ck::index_t pad_h;
    ck::index_t n_batch;
    ck::index_t num_wave_per_tile;
    ck::index_t in_scalar_per_vector;
    ck::index_t out_scalar_per_vector;
    ck::index_t dst_scalar_per_vector;
    ck::index_t w_split;
    bool require_padding;
};

//                                    b_s, t_w, t_h, f_s, d_w, d_h, s_w, s_h, p_w, p_h, n_b, n_w, i_s, o_s, d_s, w_s, r_p
static const std::vector<PerfArgs> perf_arg = {{256, 28,  28,  5,   1,   1,   1,   1,   2,   2,   2,   1,   2,   2,   2,   1,   false}, 
                                               {256, 14,  14,  5,   1,   1,   1,   1,   2,   2,   8,   1,   2,   2,   8,   1,   false}};

struct CKArgs
{
    CKArgs(const ProblemDescription& problem)
    {
        hi      = ProblemInterpreter::GetInputHeightHi(problem);
        wi      = ProblemInterpreter::GetInputWidthWi(problem);
        n       = ProblemInterpreter::GetBatchN(problem);
        k       = ProblemInterpreter::GetOutputChannelK(problem);
        c       = ProblemInterpreter::GetInputChannelC(problem);
        ho      = ProblemInterpreter::GetOutputHeightHo(problem);
        wo      = ProblemInterpreter::GetOutputWidthWo(problem);
        sy      = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
        sx      = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
        dy      = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
        dx      = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);
        py      = ProblemInterpreter::GetInputLeftPadH(problem);
        px      = ProblemInterpreter::GetInputLeftPadW(problem);
        fy      = ProblemInterpreter::GetFilterHeightY(problem);
        fx      = ProblemInterpreter::GetFilterWidthX(problem);
        g       = ProblemInterpreter::GetGroupCountG(problem);
        c_per_g = c / g;
        k_per_g = k / g;

        in_lengths  = {g,                           n,                 c_per_g, hi,           wi     };
        in_strides  = {n * hi * wi * c_per_g,       hi * wi * c_per_g, 1,       wi * c_per_g, c_per_g};
        out_lengths = {g,                           n,                 k_per_g, ho,           wo     };
        out_strides = {n * ho * wo * k_per_g,       ho * wo * k_per_g, 1,       wo * k_per_g, k_per_g};
        wei_lengths = {g,                           k_per_g,           c_per_g, fy,           fx     };
        wei_strides = {k_per_g * fy * fx * c_per_g, fy * fx * c_per_g, 1,       fx * c_per_g, c_per_g};

        padding  = {py, px};
        stride   = {sy, sx};
        dilation = {dy, dx};
    }

    // size_t GetParamHash() const
    // {
    //     size_t seed = 0;
    //     // Combine hashes of each parameter  
    //     hash_combine(seed, hash_array(input_lengths));
    //     hash_combine(seed, hash_array(in_strides));
    //     hash_combine(seed, hash_array(out_lens));
    //     hash_combine(seed, hash_array(out_strides));
    //     hash_combine(seed, hash_array(wei_lens));
    //     hash_combine(seed, hash_array(wei_strides));
    //     hash_combine(seed, hash_array(bias_lens));
    //     hash_combine(seed, hash_array(bias_strides));
    //     hash_combine(seed, hash_array(filter_stride));
    //     hash_combine(seed, hash_array(filter_dilation));
    //     hash_combine(seed, hash_array(lPadding));
    //     hash_combine(seed, hash_array(rPadding));

    //     std::array<ck::index_t, 5> others = {C1, K1, Di, Do, Z };
    //     hash_combine(seed, hash_array(others));

    //     return seed;
    // }

    CKArgs(const CKArgs&) = default;
    CKArgs(CKArgs&&)      = default;
    CKArgs& operator=(const CKArgs&) = default;
    ~CKArgs()                        = default;

    // std::size_t GetFlops() const
    // {
    //     // 2 * G * N * K * C * <output spatial lengths product> * <filter spatial lengths product>
    //     return static_cast<std::size_t>(2) * G * N * K * C *
    //         std::accumulate(std::next(std::begin(out_lens), 3),
    //                         std::end(out_lens),
    //                         static_cast<std::size_t>(1), std::multiplies<>()) *
    //         std::accumulate(std::next(std::begin(wei_lens), 3),
    //                         std::end(wei_lens),
    //                         static_cast<std::size_t>(1), std::multiplies<>());
    // }

        int hi;
        int wi;
        int n;
        int k;
        int c;
        int ho;
        int wo;
        int sy;
        int sx;
        int dy;
        int dx;
        int py;
        int px;
        int fy;
        int fx;
        int g;
        int c_per_g;
        int k_per_g;

        std::array<ck::index_t, 5> in_lengths;
        std::array<ck::index_t, 5> in_strides;
        std::array<ck::index_t, 5> out_lengths;
        std::array<ck::index_t, 5> out_strides;
        std::array<ck::index_t, 5> wei_lengths;
        std::array<ck::index_t, 5> wei_strides;

        std::array<ck::index_t, 2> padding;
        std::array<ck::index_t, 2> stride;
        std::array<ck::index_t, 2> dilation;
};

static bool IsSupportedArgument(const PerfArgs& arg,
                                const CKArgs& ck_arg,
                                const ck::index_t k_batch)
{
    constexpr ck::index_t spatial_offset = 3;
    constexpr ck::index_t wave_size = 64;
    // In
    const ck::index_t hi        = ck_arg.in_lengths[spatial_offset + 0];
    const ck::index_t wi        = ck_arg.in_lengths[spatial_offset + 1];
    const ck::index_t wi_stride = ck_arg.in_strides[spatial_offset + 1];
    const ck::index_t n         = ck_arg.in_lengths[1];
    // Out
    const ck::index_t wo        = ck_arg.out_lengths[spatial_offset + 1];
    const ck::index_t wo_stride = ck_arg.out_strides[spatial_offset + 1];
    // Wei
    const ck::index_t filter_y = ck_arg.wei_lengths[spatial_offset + 0];
    const ck::index_t filter_x = ck_arg.wei_lengths[spatial_offset + 1];
    const ck::index_t filter_k = ck_arg.wei_lengths[1];
    const ck::index_t filter_c = ck_arg.wei_lengths[2];

    const ck::index_t tile_h     = arg.tile_h;
    const ck::index_t tile_w     = arg.tile_w;
    const ck::index_t pad_h      = arg.pad_h;
    const ck::index_t pad_w      = arg.pad_w;
    const ck::index_t stride_h   = arg.stride_h;
    const ck::index_t stride_w   = arg.stride_w;
    const ck::index_t dilation_h = arg.dilation_h;
    const ck::index_t dilation_w = arg.dilation_w;

    const ck::index_t num_tile_per_block = arg.block_size / wave_size / arg.num_wave_per_tile;

    if(filter_k != 1 || filter_c != 1)
    {
        return false;
    }
    if(n % (k_batch * arg.n_batch * num_tile_per_block) != 0)
    {
        return false;
    }
    if (arg.require_padding == false)
    {
        if(hi != tile_h || wi != tile_w)
        {
            return false;
        }
    }
    if(filter_y != arg.filter_size || filter_x != arg.filter_size)
    {
        return false;
    }
    if(pad_h != ck_arg.padding[0] || pad_w != ck_arg.padding[1])
    {
        return false;
    }
    if(stride_h != ck_arg.stride[0] || stride_w != ck_arg.stride[1])
    {
        return false;
    }
    if(dilation_h != ck_arg.dilation[0] ||
        dilation_w != ck_arg.dilation[1])
    {
        return false;
    }
    if(arg.in_scalar_per_vector > 1)
    {
        if(wi % arg.in_scalar_per_vector != 0)
        {
            return false;
        }
        if(wi_stride != 1)
        {
            return false;
        }
    }
    if(arg.out_scalar_per_vector > 1)
    {
        if(wo % arg.out_scalar_per_vector != 0)
        {
            return false;
        }
        if(wo_stride != 1)
        {
            return false;
        }
    }
    return true;
}

static bool GetSupportedSolutionCount(const ProblemDescription& problem)
{
    const auto& ck_args = CKArgs{problem};
    uint solutionCount = 0;

    for (int i = 0; i < perf_arg.size(); i++)
    {
        bool is_supported = IsSupportedArgument(perf_arg[i],
                                                ck_args,
                                                1); // splitk
        if (is_supported)
        {
            solutionCount++;
        }
    }

    return solutionCount;
}

size_t ConvDepthWiseConvWrw::GetWorkspaceSize(const ExecutionContext&,
                                              const ProblemDescription& problem) const
{
    // Since the best split_k is not known here, always treating it > 1.
    auto ck_args = CKArgs{problem};

    return ck::math::integer_least_multiple(
                    sizeof(float) * ck_args.wei_lengths[0] * ck_args.wei_lengths[1] *
                    ck_args.wei_lengths[2] * ck_args.wei_lengths[3] *
                    ck_args.wei_lengths[4],
                    128);
}

bool ConvDepthWiseConvWrw::IsApplicable(const ExecutionContext& ctx,
                                        const ProblemDescription& problem) const
{
    if(!miopen::debug::AlwaysEnableConvDirectNaive)
    {
        if(env::disabled(MIOPEN_DEBUG_CONV_DEPTH_WISE_CONV_WRW))
            return false;
        if(!ctx.use_hip_kernels)
            return false;
    }

    if(!ConvDirectNaiveConvIsApplicableByKernelType(ctx, problem))
        return false;

    if(!problem.IsLayoutDefault() && !problem.IsLayoutNHWC())
        return false;

    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16() || problem.IsFp8() ||
         problem.IsBfp8()))
        return false;

    if(!problem.IsDirectionBackwardWrW())
        return false;
    if(!problem.AllTensorsLengthsFitIntoInt())
        return false;
    if(problem.IsTensorsCasted())
    {
        auto test_cast = [&](const TensorDescriptor& desc) {
            if(desc.GetCastType())
            {
                const auto cast_type = *desc.GetCastType();
                if(cast_type == miopenFloat8_fnuz || cast_type == miopenBFloat8_fnuz)
                    return false;
            }
            // all tested tensors must have cast type set
            return true;
        };
        if(test_cast(problem.GetIn()))
            return false;
        if(test_cast(problem.GetOut()))
            return false;
    }

    if (GetSupportedSolutionCount(problem) == 0)
        return false;

    return true;
}

ConvSolution ConvDepthWiseConvWrw::GetSolution(const ExecutionContext& ctx,
                                               const ProblemDescription& problem) const
{
    ConvSolution result;

    if(problem.Is2d())
    {
        auto ck_args = CKArgs{problem};

        // TODO: choose the best perf_arg !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        uint best_perf_arg_index = 0;

        // TODO: choose the best split_k !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        uint split_k = 2;

        std::array<ck::index_t, 5> acc_strides = {};

        if (split_k > 1)
        {
            ck_args.in_lengths[1] /= split_k;
            ck_args.out_lengths[1] /= split_k;

            acc_strides[4] = 1;
            acc_strides[3] = ck_args.wei_lengths[4];
            acc_strides[2] = acc_strides[3] * ck_args.wei_lengths[3];
            acc_strides[1] = acc_strides[2] * ck_args.wei_lengths[2];
            acc_strides[0] = acc_strides[1] * ck_args.wei_lengths[1];
        }

        size_t block_size = perf_arg[best_perf_arg_index].block_size;
        size_t grid_size  = static_cast<size_t>(ck_args.in_lengths[0]);

        KernelInfo kernel0_info, kernel1_info;

        kernel0_info.kernel_file = "device_grouped_conv_bwd_weight_dl_v4.cpp";
        // "depth_wise_conv_wrw.cpp";
        // "naive_conv.cpp";
        // "depth_wise_conv_wrw.cpp";
        kernel0_info.kernel_name = "kernel_grouped_conv_bwd_weight_dl_v4_run";
        // ConvDirectNaiveConvKernelName(problem);
        kernel0_info.g_wk.clear();

        kernel0_info.g_wk.push_back(grid_size);
        kernel0_info.g_wk.push_back(split_k);
        kernel0_info.g_wk.push_back(1);
        kernel0_info.l_wk.clear();
        kernel0_info.l_wk.push_back(block_size);
        kernel0_info.l_wk.push_back(1);
        kernel0_info.l_wk.push_back(1);

        kernel0_info.comp_options = ck_utility::get_ck_common_compiler_flag(ctx.GetStream())
            + ctx.general_compile_options
            + " -DCK_PARAM_BLOCKSIZE=" + std::to_string(perf_arg[best_perf_arg_index].block_size)
            + " -DCK_PARAM_TILE_W=" + std::to_string(perf_arg[best_perf_arg_index].tile_w)
            + " -DCK_PARAM_TILE_H=" + std::to_string(perf_arg[best_perf_arg_index].tile_h)
            + " -DCK_PARAM_FILTERSIZE=" + std::to_string(perf_arg[best_perf_arg_index].filter_size)
            + " -DCK_PARAM_PROBLEM_CONV_DILATION_W=" + std::to_string(ck_args.dx)
            + " -DCK_PARAM_PROBLEM_CONV_DILATION_H=" + std::to_string(ck_args.dy)
            + " -DCK_PARAM_PROBLEM_CONV_STRIDE_W=" + std::to_string(ck_args.sx)
            + " -DCK_PARAM_PROBLEM_CONV_STRIDE_H=" + std::to_string(ck_args.sy)
            + " -DCK_PARAM_PROBLEM_CONV_PAD_W=" + std::to_string(ck_args.px)
            + " -DCK_PARAM_PROBLEM_CONV_PAD_H=" + std::to_string(ck_args.py)
            + " -DCK_PARAM_NBATCH=" + std::to_string(perf_arg[best_perf_arg_index].n_batch)
            + " -DCK_PARAM_NUMWAVEPERTILE=" + std::to_string(perf_arg[best_perf_arg_index].num_wave_per_tile)
            + " -DCK_PARAM_INSCALARPERVECTOR=" + std::to_string(perf_arg[best_perf_arg_index].in_scalar_per_vector)
            + " -DCK_PARAM_OUTSCALARPERVECTOR=" + std::to_string(perf_arg[best_perf_arg_index].out_scalar_per_vector)
            + " -DCK_PARAM_DSTSCALARPERVECTOR=" + std::to_string(perf_arg[best_perf_arg_index].dst_scalar_per_vector)
            + " -DCK_PARAM_REQUIREPADDING=" + std::to_string(perf_arg[best_perf_arg_index].require_padding)
            + " -DCK_PARAM_WSPLIT=" + std::to_string(perf_arg[best_perf_arg_index].w_split)
            ;
        // ConvDirectNaiveConvCompileOption(ctx, problem);

        kernel1_info.kernel_file = "device_grouped_conv_bwd_weight_dl_v4.cpp";
        kernel1_info.kernel_name = "kernel_grouped_conv_bwd_weight_elementwise_run";

        kernel1_info.comp_options = kernel0_info.comp_options;
        kernel1_info.l_wk = {perf_arg[best_perf_arg_index].filter_size * perf_arg[best_perf_arg_index].filter_size, 1, 1};
        kernel1_info.g_wk = {ck_args.in_lengths[0], 1, 1};

        result.workspace_sz = GetWorkspaceSize(ctx, problem);

        result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                decltype(auto) data_ctx = primitive_parameters.CastTo<miopen::conv::WrWInvokeParams>();
                const auto& tensors     = data_ctx.tensors;
                float elapsed           = 0;
                {
                    if (split_k > 1)
                    {
                        hipMemsetAsync(data_ctx.workSpace, 0, data_ctx.workSpaceSize, handle.GetStream());
                    }
                    handle.Run(kernels[0])(tensors.x,
                                           split_k > 1 ? nullptr : tensors.dw,
                                           tensors.dy,
                                           split_k > 1 ? data_ctx.workSpace : nullptr,
                                           ck_args.in_lengths,
                                           ck_args.in_strides,
                                           ck_args.wei_lengths,
                                           split_k > 1 ? acc_strides : ck_args.wei_strides,
                                           ck_args.out_lengths,
                                           ck_args.out_strides,
                                           split_k > 1);
                }
                if(handle.IsProfilingEnabled())
                {
                    elapsed += handle.GetKernelTime();
                }

                if (split_k > 1)
                {
                    handle.Run(kernels[1])(tensors.dw,
                                           data_ctx.workSpace,
                                           ck_args.wei_strides,
                                           acc_strides);
                }

                if(handle.IsProfilingEnabled())
                {
                    if (split_k > 1)
                    {
                        elapsed += handle.GetKernelTime();
                    }
                    
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };

        result.construction_params.push_back(kernel0_info);
        if (split_k > 1)
        {
            result.construction_params.push_back(kernel1_info);
        }
    }
    else
    {
        // result = conv_internal::GetConv3DWRWSolution(ctx, problem);
    }
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
