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

#include <limits>
#include <cassert>
#include <miopen/solver.hpp>
#include <miopen/env.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/tensor.hpp>
#include <miopen/gemm_v2.hpp>

#include <boost/any.hpp>

#if(MIOPEN_BACKEND_HIP && MIOPEN_USE_ROCBLAS)
#include <miopen/conv/data_invoke_params.hpp>
#define WORKAROUND_SWDEV_203031 1 // See also issues #2075, #2067
#endif

namespace miopen {
namespace solver {
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F2X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F4X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F6X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_CONV_PRECISE_ROCBLAS_TIMING)

// Introduces a number of shader-specific aliases (names) in the current scope at zero cost.
// These names represent shader parameters, e.g. shader C is batch_size etc and useful for
// programming.
#define DEFINE_GETXFORMHWSIZE(params)                                                        \
    const auto                                                                               \
        wino_xform_h =                                                                       \
            solver::ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>:: \
                GetSolverWinoXformHWSize(),                                                  \
        wino_xform_w =                                                                       \
            solver::ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>:: \
                GetSolverWinoXformHWSize();

#define DEFINE_SHADER_ALIASES(params)                      \
    const auto group_cnt = (params).group_counts;          \
    const auto& N        = (params).batch_sz;              \
    const auto& K        = (params).n_outputs / group_cnt; \
    const auto& C        = (params).n_inputs / group_cnt;  \
    const auto& R        = (params).kernel_size_h;         \
    const auto& S        = (params).kernel_size_w;         \
    const auto& H        = (params).in_height;             \
    const auto& W        = (params).in_width;              \
    const auto& out_H    = (params).out_height;            \
    const auto& out_W    = (params).out_width;

#if(MIOPEN_BACKEND_HIP && MIOPEN_USE_ROCBLAS)
#define GENERATE_MAIN_OPTIONS(options)                                                             \
    GenerateClangDefsym((options), "acc_type", 1);                                                 \
    GenerateClangDefsym((options), "buf_type", (params.IsFp32() ? 1 : (params.IsFp16() ? 2 : 3))); \
    GenerateClangDefsym((options), "ROCM_METADATA_VERSION", 5);                                    \
    GenerateClangDefsym((options), "xformx_o_size", WinoDataW);                                    \
    GenerateClangDefsym((options), "xformy_o_size", WinoDataH);                                    \
    GenerateClangDefsym((options), "xformx_d_size", wino_xform_w);                                 \
    GenerateClangDefsym((options), "xformy_d_size", wino_xform_h);                                 \
    GenerateClangDefsym((options), "xformx_f_size", WinoFilterW);                                  \
    GenerateClangDefsym((options), "xformy_f_size", WinoFilterH);                                  \
    GenerateClangDefsym((options), "fdilation_w", params.kernel_stride_w);                         \
    GenerateClangDefsym((options), "fdilation_h", params.kernel_stride_h);

static inline size_t Ceil(const size_t v, const size_t m)
{
    assert(m > 0);
    return (v + m - 1) / m;
}
#endif

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>
GetWinoBuffer(const ConvolutionContext& params, const ConvWinoBuffType buff_type)
{
    DEFINE_GETXFORMHWSIZE(params)
    DEFINE_SHADER_ALIASES(params)

    WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW> Transform_info(
        N,
        K,
        C,
        group_cnt,
        out_H,
        out_W,
        R,
        S,
        (MemLayout_t::GCNHW),
        (ConvWinoXformType::N_GXhXw_C_Th_Tw),
        GetTypeSize(params.in_data_type),
        buff_type,
        wino_xform_h,
        wino_xform_w);

    (void)H;
    (void)W;
    return Transform_info;
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
bool ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(
    const ConvolutionContext& params) const
{
// HIP backend required for sending ptr (buffer + offset)
// ROCBLAS for GEMM step

#if(MIOPEN_BACKEND_HIP && MIOPEN_USE_ROCBLAS)
    static const int wino_data_tile   = std::max(WinoDataH, WinoDataW);
    static const int wino_filter_tile = std::max(WinoFilterH, WinoFilterW);

    const std::string name = params.GetStream().GetDeviceName();

    if(wino_data_tile == 6 && wino_filter_tile == 3)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F6X3{}))
            return false;
    if(wino_data_tile == 5 && wino_filter_tile == 3)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F5X3{}))
            return false;
    if(wino_data_tile == 4 && wino_filter_tile == 3)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F4X3{}))
            return false;
    if(wino_data_tile == 3 && wino_filter_tile == 3)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3{}))
            return false;
    if(wino_data_tile == 2 && wino_filter_tile == 3)
        if(miopen::IsDisabled(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F2X3{}))
            return false;
    if(!params.use_asm_kernels)
        return false;
    if(!params.rmv.IsV3())
        return false;
    if(!params.Is2d())
        return false;
    if(params.direction.IsBackwardWrW())
        return false;
    if(!params.IsFp32())
        return false;

    if(!(StartsWith(name, "gfx9")))
        return false;

    {
        std::size_t limit = miopen::Value(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_WORKSPACE_MAX{});
#if WORKAROUND_SWDEV_203031
        if(limit == 0)
        {
            if(name == "gfx900" ||
               (name == "gfx906" && params.GetStream().GetMaxComputeUnits() <= 60))
                limit = 2000000000ULL; // ~1.862 GiB
            else
                limit = std::numeric_limits<std::size_t>::max();
        }
#else
        if(limit == 0)
            limit = std::numeric_limits<std::size_t>::max();
#endif
        if(limit != std::numeric_limits<std::size_t>::max())
        {
            const auto required = GetWorkspaceSize(params);
            MIOPEN_LOG_I2("Workspace required: " << required << ", limit: " << limit);
            if(required > limit)
                return false;
        }
    }

    // int offset for Workspace buffers.
    if(((GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(params,
                                                                       ConvWinoBuffType::Input))
                .buff_info.total_byte_size /
            GetTypeSize(params.in_data_type) +
        (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(params,
                                                                       ConvWinoBuffType::Output))
                .buff_info.total_byte_size /
            GetTypeSize(params.in_data_type)) >= (1LL << 31))
    {
        return false;
    }

    if(!params.IsLayoutDefault())
    {
        return false;
    }

    {
        unsigned int const waves_in_group = 512 / wave_size;
        unsigned int const tiles_per_wave = 8;
        auto const tiles_per_group        = waves_in_group * tiles_per_wave / 2;
        auto const n_groups               = params.GetStream().GetMaxComputeUnits();
        auto const tiles_step             = tiles_per_group * n_groups;
        if(tiles_step >= std::pow(2, 16))
            return false;
    }
    {
        unsigned long long const f_tiles_hw =
            Ceil(params.kernel_size_w, WinoFilterW) * Ceil(params.kernel_size_h, WinoFilterH);
        unsigned long long const d_tiles_hw =
            Ceil(params.out_width, WinoDataW) * Ceil(params.out_height, WinoDataH);
        unsigned long long const G_N_C_xTile =
            params.n_inputs * params.group_counts * params.batch_sz * d_tiles_hw;
        unsigned long long const G_K_C_xTile =
            params.n_inputs * params.group_counts * params.n_outputs * f_tiles_hw;
        unsigned long long const G_K_N_xTile =
            params.batch_sz * params.group_counts * params.n_outputs * d_tiles_hw;
        if(G_N_C_xTile >= std::pow(2, 24) || G_K_C_xTile >= std::pow(2, 24) ||
           G_K_N_xTile >= std::pow(2, 24))
            return false;
    }

    // clang-format off
    bool ok = (
           (params.kernel_size_w == WinoFilterW
               && params.kernel_size_h == WinoFilterH)
        && (params.kernel_stride_w == 1)
        && params.kernel_stride_h == params.kernel_stride_w
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        && params.batch_sz < std::pow(2, 16)
        && params.n_inputs < std::pow(2, 16)
        && params.n_outputs < std::pow(2, 16)
        && params.out_height < std::pow(2, 16)
        && params.out_width < std::pow(2, 16)
        && params.bias == 0
        && params.in_layout == "NCHW");
    // clang-format on
    return ok;
#else
    (void)params;
    return false;
#endif
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
size_t ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetWorkspaceSize(
    const ConvolutionContext& params) const
{
    return (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(params,
                                                                          ConvWinoBuffType::Input))
               .buff_info.total_byte_size +
           (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(params,
                                                                          ConvWinoBuffType::Output))
               .buff_info.total_byte_size +
           (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(params,
                                                                          ConvWinoBuffType::Weight))
               .buff_info.total_byte_size;
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
ConvSolution ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetSolution(
    const ConvolutionContext& params) const
{

    ConvSolution result;
    result.workspce_sz = GetWorkspaceSize(params);
#if(MIOPEN_BACKEND_HIP && MIOPEN_USE_ROCBLAS)

    const int n_groups = params.GetStream().GetMaxComputeUnits();
    DEFINE_GETXFORMHWSIZE(params)
    const std::vector<size_t> l_wk{512, 1, 1};
    const size_t g_wk_0 = n_groups * l_wk[0];
    const std::vector<size_t> g_wk{g_wk_0, 1, 1};

    std::ostringstream options_in;
    GENERATE_MAIN_OPTIONS(options_in)
    GenerateClangDefsym(options_in, "xform_mirror", 0);

    std::ostringstream options_filter;
    GENERATE_MAIN_OPTIONS(options_filter)
    GenerateClangDefsym(options_filter, "xform_mirror", params.direction.IsBackwardData());

    std::ostringstream options_out;
    GENERATE_MAIN_OPTIONS(options_out)
    GenerateClangDefsym(options_out, "xform_mirror", 0);

    KernelInfo InTransform{
        options_in.str(),
        l_wk,
        g_wk,
        ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetSolverFileNames(
            0),
        ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverKernelNames(0),
    };

    KernelInfo FilterTransform{
        options_filter.str(),
        l_wk,
        g_wk,
        ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetSolverFileNames(
            1),
        ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverKernelNames(1),
    };

    KernelInfo OutTransform{
        options_out.str(),
        l_wk,
        g_wk,
        ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetSolverFileNames(
            2),
        ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
            GetSolverKernelNames(2),
    };

    result.construction_params.push_back(InTransform);
    result.construction_params.push_back(FilterTransform);
    result.construction_params.push_back(OutTransform);

    const int pad_H = params.direction.IsForward() ? params.pad_h : params.GetBackwardPadH();
    const int pad_W = params.direction.IsForward() ? params.pad_w : params.GetBackwardPadW();
    DEFINE_SHADER_ALIASES(params)
    BuffInfo in_buff(GetGroupConvLayout(GetMemLayout_t(params.in_layout), true),
                     N,
                     C,
                     H,
                     W,
                     group_cnt,
                     GetTypeSize(params.in_data_type)),
        // cppcheck-suppress unreadVariable
        out_buff(GetGroupConvLayout(GetMemLayout_t(params.out_layout), true),
                 N,
                 K,
                 out_H,
                 out_W,
                 group_cnt,
                 GetTypeSize(params.out_data_type)),
        // cppcheck-suppress unreadVariable
        weights_buff(GetGroupConvLayout(params.direction.IsForward()
                                            ? (MemLayout_t::NCHW)
                                            : GetSwappedNCLayout(MemLayout_t::NCHW),
                                        false),
                     K,
                     C,
                     R,
                     S,
                     group_cnt,
                     GetTypeSize(params.weights_data_type));

    auto wino_in = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
        params, ConvWinoBuffType::Input);
    auto wino_out = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
        params, ConvWinoBuffType::Output);
    auto wino_wei = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
        params, ConvWinoBuffType::Weight);

    int reserved       = 0;
    void* reserved_ptr = nullptr;
    int unused         = 0;

    size_t wino_in_offset = 0, wino_out_offset = wino_in.buff_info.total_byte_size,
           wino_wei_offset = wino_out_offset + wino_out.buff_info.total_byte_size;

    // GEMM
    int m = K, k = C,
        n   = wino_in.buff_info.size.nk * wino_in.buff_info.size.w * wino_in.buff_info.size.h;
    int lda = m, ldb = n, ldc = n;
    int batch_count       = wino_xform_h * wino_xform_w * group_cnt;
    long long int strideA = m * k * 1LL, strideB = k * n * 1LL, strideC = m * n * 1LL;
    float alpha = 1., beta = 0.0;
    const bool isColMajor = false, transA = true, transB = false;
    // clang-format off
    GemmDescriptor wino_gemm_desc{isColMajor,transA,transB,m,n,k,
        lda,ldb,ldc,batch_count,strideA,strideB,
        strideC,alpha,beta,params.in_data_type};

    result.invoker_factory = [=](std::vector<Kernel> kernels) {
        return [=](const Handle& handle, const boost::any& ctx) {
            const auto data_ctx = boost::any_cast<conv::DataInvokeParams>(ctx);
            const auto tensors  = data_ctx.tensors;
            Data_t workSpace = data_ctx.workSpace;
            float total_time    = 0;
            
            for(int i = 0, cur=0; i < 4; i++)
            {
                std::string kernel_name ;
                if(i == 2) // GEMM
                {
                        CallGemmStridedBatched(handle,
                            wino_gemm_desc,
                            workSpace,
                            static_cast<int>(wino_wei_offset / GetTypeSize(params.in_data_type)),
                            workSpace,
                            static_cast<int>(wino_in_offset / GetTypeSize(params.in_data_type)),
                            workSpace,
                            static_cast<int>(wino_out_offset / GetTypeSize(params.in_data_type)),
                            nullptr,
                            false,
                            GemmBackend_t::rocblas);
                    kernel_name = "WRW_WINO_GEMM: ";
                }
                else
                {
                    const auto kernel        = handle.Run(kernels[cur++]);
                    const BuffInfo* d_buf    = nullptr;
                    const BuffInfo* o_buf    = nullptr;
                    Data_t buff_out_addr     = nullptr;

                    auto const_buff_in_adr  = tensors.in;
                    auto buff_in_adr        = workSpace;
                    bool const_input        = false;
                    size_t buff_in_addr_offset = 0, buff_out_addr_offset = 0;
                    kernel_name = kernel.GetName();

                    if(i==0) // Input
                    {// Transform
                        d_buf               = &in_buff;
                        o_buf               = &(wino_in.buff_info);
                        const_buff_in_adr   = tensors.in;
                        buff_in_addr_offset = 0;
                        buff_out_addr        = workSpace;
                        buff_out_addr_offset = wino_in_offset;
                        const_input         = true;
                    }
                    else if(i==1) // filter
                    {// Transform
                        d_buf                = &weights_buff;
                        o_buf                = &(wino_wei.buff_info);
                        const_buff_in_adr    = tensors.w;
                        buff_in_addr_offset = 0;
                        buff_out_addr         = workSpace;
                        buff_out_addr_offset = wino_wei_offset;
                        const_input          = true;
                    }
                    else if (i==3)
                    { //Output
                        d_buf               = &(wino_out.buff_info);
                        o_buf               = &(out_buff);
                        buff_in_adr         = workSpace;
                        buff_in_addr_offset = wino_out_offset;
                        buff_out_addr        = tensors.out;
                        buff_out_addr_offset = 0;
                        const_input          = false;
                    }

                    const auto input_ptr = static_cast<const void*>(
                        static_cast<const char*>(const_input ? const_buff_in_adr : buff_in_adr) +
                        buff_in_addr_offset);
                    const auto output_ptr =
                        static_cast<void*>(static_cast<char*>(buff_out_addr) + buff_out_addr_offset);
                    // clang-format off
                    MIOPEN_LOG_I2(" N=" << N << " G=" << group_cnt << " C=" << C << " H=" << H << " W=" << W << " K=" << K
                        << " n_groups=" << n_groups << " R=" << R << " S=" << S
                        << " pad_H=" << pad_H << " pad_W=" << pad_W << " out_H=" << out_H << " out_W=" << out_W
                        << " d_buf.byte_stride.nk=" << d_buf->byte_stride.nk << " d_buf->.byte_stride.c=" << d_buf->byte_stride.c
                        << " d_buf->.byte_stride.h=" << d_buf->byte_stride.h << " d_buf->.byte_stride.w=" << d_buf->byte_stride.w
                        << " o_buf->byte_stride.nk=" << o_buf->byte_stride.nk << " o_buf->byte_stride.c=" << o_buf->byte_stride.c
                        << " o_buf.byte_stride.h="  << o_buf->byte_stride.h <<  " o_buf->byte_stride.w=" << o_buf->byte_stride.w
                        << " d_buf->.byte_stride.g=" << d_buf->byte_stride.g  << " o_buf->byte_stride.g="  << o_buf->byte_stride.g);
                    // clang-format on
                    kernel(N,
                           C,
                           H,
                           W,
                           K,
                           n_groups,
                           unused,
                           reserved,
                           input_ptr,
                           reserved_ptr,
                           output_ptr,
                           reserved_ptr, // Unused return_addr.
                           R,
                           S,
                           pad_H, // Like Fwd wino.
                           pad_W,
                           out_H,
                           out_W,
                           reserved_ptr, // Unused bias_addr.
                           reserved,     // Unused relu_alpha.
                           d_buf->byte_stride.nk,
                           d_buf->byte_stride.c,
                           d_buf->byte_stride.h,
                           d_buf->byte_stride.w,
                           unused,
                           unused,
                           unused,
                           unused,
                           o_buf->byte_stride.nk,
                           o_buf->byte_stride.c,
                           o_buf->byte_stride.h,
                           o_buf->byte_stride.w,
                           group_cnt,
                           d_buf->byte_stride.g,
                           unused,
                           o_buf->byte_stride.g);
                }
                if(handle.IsProfilingEnabled())
                {
                    float cur_time = handle.GetKernelTime();
                    MIOPEN_LOG_I2(kernel_name << ": " << cur_time);

                    if(i < 3)
                        total_time += cur_time;
                    else
                        handle.AccumKernelTime(total_time);
                }
            }
        };
    };
    return result;
#else
    (void)params;
    MIOPEN_THROW(miopenStatusBadParm, "ConvMPBidirectWinograd Unsupported ");
#endif
}
template struct ConvMPBidirectWinograd<2, 3>;
template struct ConvMPBidirectWinograd<3, 3>;
template struct ConvMPBidirectWinograd<4, 3>;
template struct ConvMPBidirectWinograd<5, 3>;
template struct ConvMPBidirectWinograd<6, 3>;

} // namespace solver
} // namespace miopen
