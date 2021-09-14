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
#include <miopen/stringutils.hpp>
#include <miopen/miopen.h>
#include <miopen/generic_search.hpp>
#include <miopen/conv/invokers/impl_gemm.hpp>

#include <boost/any.hpp>
#include <miopen/conv/data_invoke_params.hpp>

#if MIOPEN_BACKEND_HIP

#define WORKAROUND_SWDEV_203031 1 // See also issues #2075, #2067
#define WORKAROUND_ISSUE_1146 1   // check asm solver applicability for gfx90a
#endif

#define WORKAROUND_SWDEV_257202 1 // For SSD convergence issue.

#if WORKAROUND_SWDEV_257202
// Workaround, solver disabled by default.
#define IS_DISABLED(expr) !miopen::IsEnabled(expr)
#else
// Normal behavior (solver enabled by default).
#define IS_DISABLED(expr) miopen::IsDisabled(expr)
#endif

namespace miopen {
namespace solver {
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F2X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F3X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F4X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F5X3)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F6X3)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM)

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

#define DEFINE_SHADER_ALIASES(params)                       \
    const auto& group_cnt = (params).group_counts;          \
    const auto& N         = (params).batch_sz;              \
    const int K           = (params).n_outputs / group_cnt; \
    const int C           = (params).n_inputs / group_cnt;  \
    const auto& R         = (params).kernel_size_h;         \
    const auto& S         = (params).kernel_size_w;         \
    const auto& H         = (params).in_height;             \
    const auto& W         = (params).in_width;              \
    const auto& out_H     = (params).out_height;            \
    const auto& out_W     = (params).out_width;

#if MIOPEN_BACKEND_HIP
#define GENERATE_MAIN_OPTIONS(options)                                     \
    GenerateClangDefsym((options), "acc_type", 1);                         \
    GenerateClangDefsym((options), "ROCM_METADATA_VERSION", 5);            \
    GenerateClangDefsym((options), "xformx_o_size", WinoDataW);            \
    GenerateClangDefsym((options), "xformy_o_size", WinoDataH);            \
    GenerateClangDefsym((options), "xformx_d_size", wino_xform_w);         \
    GenerateClangDefsym((options), "xformy_d_size", wino_xform_h);         \
    GenerateClangDefsym((options), "xformx_f_size", WinoFilterW);          \
    GenerateClangDefsym((options), "xformy_f_size", WinoFilterH);          \
    GenerateClangDefsym((options), "fdilation_w", params.kernel_stride_w); \
    GenerateClangDefsym((options), "fdilation_h", params.kernel_stride_h);

struct WinoOffsets
{
    const size_t in, out, wei;
    WinoOffsets(size_t in_size, size_t out_size) : in(0), out(in_size), wei(in_size + out_size) {}
};
#endif

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>
GetWinoBuffer(const ConvolutionContext& params,
              const ConvWinoBuffType buff_type,
              const miopenDataType_t transform_data_type)
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
        GetTypeSize(transform_data_type),
        buff_type,
        wino_xform_h,
        wino_xform_w);

    (void)H;
    (void)W;
    return Transform_info;
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
inline bool IsApplicableGEMM(const ConvolutionContext& params)
{
#if(MIOPEN_BACKEND_HIP && (MIOPEN_USE_ROCBLAS || MIOPEN_USE_MIOPENTENSILE))

    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? params.in_data_type
            : miopenFloat;

    // int offset for Workspace buffers.
    return !(((GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
                   params, ConvWinoBuffType::Input, transform_data_type))
                      .buff_info.total_byte_size /
                  GetTypeSize(transform_data_type) +
              (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
                   params, ConvWinoBuffType::Output, transform_data_type))
                      .buff_info.total_byte_size /
                  GetTypeSize(transform_data_type)) >= (1LL << 31));
#else
    (void)params;
    return false;
#endif
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
inline bool IsApplicableTransform(const ConvolutionContext& params)
{
#if MIOPEN_BACKEND_HIP
    if(!params.use_asm_kernels)
        return false;
    if(!params.rmv.IsV3())
        return false;
    if(!params.Is2d())
        return false;
    if(!(params.direction.IsForward() || params.direction.IsBackwardData()))
        return false;
    if(!(params.IsFp32() || params.IsFp16()))
        return false;

    const auto target = params.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    const std::string name = params.GetStream().GetDeviceName();
    if(!StartsWith(name, "gfx9"))
        return false;
#if WORKAROUND_ISSUE_1146
    if(name == "gfx90a")
        return false;
#endif

    {
        std::size_t limit = miopen::Value(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_WORKSPACE_MAX{});
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
            const auto required =
                ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>{}
                    .GetWorkspaceSize(params);
            MIOPEN_LOG_I2("Workspace required: " << required << ", limit: " << limit);
            if(required > limit)
                return false;
        }
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
    DEFINE_SHADER_ALIASES(params)
    {
        const miopenDataType_t transform_data_type =
            miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
                ? params.in_data_type
                : miopenFloat;

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
            wei_buff(GetGroupConvLayout(params.direction.IsForward()
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
            params, ConvWinoBuffType::Input, transform_data_type);
        auto wino_out = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            params, ConvWinoBuffType::Output, transform_data_type);
        auto wino_wei = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            params, ConvWinoBuffType::Weight, transform_data_type);

        if(in_buff.total_byte_size > std::pow(2, 31) ||
           wei_buff.total_byte_size > std::pow(2, 31) ||
           out_buff.total_byte_size > std::pow(2, 31) ||
           wino_in.buff_info.total_byte_size > std::pow(2, 31) ||
           wino_out.buff_info.total_byte_size > std::pow(2, 31) ||
           wino_wei.buff_info.total_byte_size > std::pow(2, 31))
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
        && N < std::pow(2, 16)
        && C < std::pow(2, 16)
        && K < std::pow(2, 16)
        && out_H < std::pow(2, 16)
        && out_W < std::pow(2, 16)
        && group_cnt < std::pow(2, 16)
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
bool ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(
    const ConvolutionContext& params) const
{
    // HIP backend required for sending ptr (buffer + offset)
    // ROCBLAS for GEMM step

    if(!params.IsLayoutDefault())
    {
        return false;
    }

    if(!IsApplicableGEMM<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(params))
        return false;

    static const int wino_data_tile   = std::max(WinoDataH, WinoDataW);
    static const int wino_filter_tile = std::max(WinoFilterH, WinoFilterW);

    if(wino_data_tile == 6 && wino_filter_tile == 3)
        if(IS_DISABLED(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3{}))
            return false;
    if(wino_data_tile == 5 && wino_filter_tile == 3)
        if(IS_DISABLED(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3{}))
            return false;
    if(wino_data_tile == 4 && wino_filter_tile == 3)
        if(IS_DISABLED(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3{}))
            return false;
    if(wino_data_tile == 3 && wino_filter_tile == 3)
        if(IS_DISABLED(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3{}))
            return false;
    if(wino_data_tile == 2 && wino_filter_tile == 3)
        if(IS_DISABLED(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3{}))
            return false;

    return IsApplicableTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(params);
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
size_t ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetWorkspaceSize(
    const ConvolutionContext& params) const
{
    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? params.in_data_type
            : miopenFloat;

    return (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
                params, ConvWinoBuffType::Input, transform_data_type))
               .buff_info.total_byte_size +
           (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
                params, ConvWinoBuffType::Output, transform_data_type))
               .buff_info.total_byte_size +
           (GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
                params, ConvWinoBuffType::Weight, transform_data_type))
               .buff_info.total_byte_size;
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
InvokerFactory MakeWinogradInvokerFactory(const ConvolutionContext& params,
                                          InvokerFactory xdlops_factory = InvokerFactory(),
                                          bool isXdlops                 = false)
{
#if MIOPEN_BACKEND_HIP
    const int pad_H    = params.direction.IsForward() ? params.pad_h : params.GetBackwardPadH();
    const int pad_W    = params.direction.IsForward() ? params.pad_w : params.GetBackwardPadW();
    const int n_groups = params.GetStream().GetMaxComputeUnits();
    DEFINE_SHADER_ALIASES(params)
    DEFINE_GETXFORMHWSIZE(params)
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

    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? params.in_data_type
            : miopenFloat;
    auto wino_in = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
        params, ConvWinoBuffType::Input, transform_data_type);
    auto wino_out = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
        params, ConvWinoBuffType::Output, transform_data_type);
    auto wino_wei = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
        params, ConvWinoBuffType::Weight, transform_data_type);

    int reserved       = 0;
    void* reserved_ptr = nullptr;
    int unused         = 0;
    const WinoOffsets transform_offset(wino_in.buff_info.total_byte_size,
                                       wino_out.buff_info.total_byte_size);

    InvokerFactory gemm_conv_factory;
    std::string gemm_conv_kernel_name;

    auto zeroDesc = TensorDescriptor();
    if(isXdlops)
    {
        gemm_conv_kernel_name = "XDLOPS_CONV: ";
        gemm_conv_factory     = xdlops_factory;
    }
    else
    {
#if MIOPEN_USE_ROCBLAS || MIOPEN_USE_MIOPENTENSILE
        // GEMM
        gemm_conv_kernel_name = "WRW_WINO_GEMM: ";

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
            strideC,alpha,beta,transform_data_type};
// clang-format on
#else
        (void)wino_xform_w;
        (void)wino_xform_h;
#endif

        gemm_conv_factory = [=](const std::vector<Kernel>&) {
            return [=](const Handle& handle, const AnyInvokeParams& ctx) {
#if MIOPEN_USE_ROCBLAS || MIOPEN_USE_MIOPENTENSILE
                const auto& data_ctx = ctx.CastTo<conv::DataInvokeParams>();
                Data_t workSpace     = data_ctx.workSpace;
                CallGemmStridedBatched(
                    handle,
                    wino_gemm_desc,
                    workSpace,
                    static_cast<int>(transform_offset.wei / wino_wei.buff_info.element_size),
                    workSpace,
                    static_cast<int>(transform_offset.in / wino_in.buff_info.element_size),
                    workSpace,
                    static_cast<int>(transform_offset.out / wino_out.buff_info.element_size),
                    nullptr,
                    GemmBackend_t::rocblas);
#else
                (void)handle;
                (void)ctx;
                MIOPEN_THROW(miopenStatusBadParm, "ConvMPBidirectWinograd is not supported ");
#endif
            };
        };
    }

    return [=](const std::vector<Kernel>& kernels) {
        const std::vector<Kernel> transform_kernels =
            std::vector<Kernel>{kernels[0], kernels[1], kernels[2]};

        const std::vector<Kernel> conv_kernels =
            isXdlops ? std::vector<Kernel>{kernels[3]} : std::vector<Kernel>{};

        auto gemm_conv_invoker = gemm_conv_factory(conv_kernels);

        return [=](const Handle& handle, const AnyInvokeParams& ctx) {
            const auto& data_ctx = ctx.CastTo<conv::DataInvokeParams>();
            const auto tensors   = data_ctx.tensors;
            Data_t workSpace     = data_ctx.workSpace;
            auto workSpaceSize   = data_ctx.workSpaceSize;
            float total_time     = 0;
            auto wino_in_ptr =
                static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.in);
            auto wino_w_ptr =
                static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.wei);
            auto wino_out_ptr =
                static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.out);

            for(int i = 0, cur = 0; i < 4; i++)
            {
                std::string kernel_name;
                if(i == 2) // GEMM
                {
                    // rocblas_gemm use workSpace pointer and constant offset
                    // xdlops_conv use tensors.in, tensors.w, tensors.out
                    ConvDataTensors xdlops_tensor = ConvDataTensors(ConvFwdTensors{
                        zeroDesc, wino_in_ptr, zeroDesc, wino_w_ptr, zeroDesc, wino_out_ptr});
                    const auto invoke_params =
                        conv::DataInvokeParams{xdlops_tensor, workSpace, workSpaceSize};

                    gemm_conv_invoker(handle, invoke_params);
                    kernel_name = gemm_conv_kernel_name;
                }
                else
                {
                    const auto kernel     = handle.Run(transform_kernels[cur++]);
                    const BuffInfo* d_buf = nullptr;
                    const BuffInfo* o_buf = nullptr;
                    void* buff_out_addr   = nullptr;

                    auto const_buff_in_adr = tensors.in;
                    auto buff_in_adr       = wino_out_ptr;
                    bool const_input       = false;
                    kernel_name            = kernel.GetName();

                    if(i == 0) // Input
                    {          // Transform
                        d_buf             = &in_buff;
                        o_buf             = &(wino_in.buff_info);
                        const_buff_in_adr = tensors.in;
                        buff_out_addr     = wino_in_ptr;
                        const_input       = true;
                    }
                    else if(i == 1) // filter
                    {               // Transform
                        d_buf             = &weights_buff;
                        o_buf             = &(wino_wei.buff_info);
                        const_buff_in_adr = tensors.w;
                        buff_out_addr     = wino_w_ptr;
                        const_input       = true;
                    }
                    else if(i == 3)
                    { // Output
                        d_buf         = &(wino_out.buff_info);
                        o_buf         = &(out_buff);
                        buff_in_adr   = wino_out_ptr;
                        buff_out_addr = tensors.out;
                        const_input   = false;
                    }

                    const auto input_ptr =
                        static_cast<const void*>(const_input ? const_buff_in_adr : buff_in_adr);
                    const auto output_ptr = buff_out_addr;
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
#else
    (void)params;
    (void)xdlops_factory;
    (void)isXdlops;
    MIOPEN_THROW(miopenStatusBadParm, "ConvMPBidirectWinograd is not supported ");
    return nullptr;
#endif
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
ConvSolution ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetSolution(
    const ConvolutionContext& params) const
{
    ConvSolution result;
    result.workspce_sz = GetWorkspaceSize(params);
#if MIOPEN_BACKEND_HIP

    const int n_groups = params.GetStream().GetMaxComputeUnits();
    DEFINE_GETXFORMHWSIZE(params)
    const std::vector<size_t> l_wk{512, 1, 1};
    const size_t g_wk_0 = n_groups * l_wk[0];
    const std::vector<size_t> g_wk{g_wk_0, 1, 1};
    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? params.in_data_type
            : miopenFloat;
    std::ostringstream options_in;
    GENERATE_MAIN_OPTIONS(options_in)
    GenerateClangDefsym(options_in, "xform_mirror", 0);
    GenerateClangDefsym(options_in, "in_type", (params.IsFp32() ? 1 : 2));
    GenerateClangDefsym(options_in, "out_type", (transform_data_type == miopenFloat ? 1 : 2));

    std::ostringstream options_filter;
    GENERATE_MAIN_OPTIONS(options_filter)
    GenerateClangDefsym(options_filter, "xform_mirror", params.direction.IsBackwardData());
    GenerateClangDefsym(options_filter, "in_type", (params.IsFp32() ? 1 : 2));
    GenerateClangDefsym(options_filter, "out_type", (transform_data_type == miopenFloat ? 1 : 2));

    std::ostringstream options_out;
    GENERATE_MAIN_OPTIONS(options_out)
    GenerateClangDefsym(options_out, "xform_mirror", 0);
    GenerateClangDefsym(options_out, "in_type", (transform_data_type == miopenFloat ? 1 : 2));
    GenerateClangDefsym(options_out, "out_type", (params.IsFp32() ? 1 : 2));

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

    result.invoker_factory =
        MakeWinogradInvokerFactory<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(params);

    return result;
#else
    (void)params;
    MIOPEN_THROW(miopenStatusBadParm, "ConvMPBidirectWinograd is not supported ");
#endif
}
template struct ConvMPBidirectWinograd<2, 3>;
template struct ConvMPBidirectWinograd<3, 3>;
template struct ConvMPBidirectWinograd<4, 3>;
template struct ConvMPBidirectWinograd<5, 3>;
template struct ConvMPBidirectWinograd<6, 3>;

// context transformation
// for winograd buffers calculation using xdlops_convolution
template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
ConvolutionContext ConvMPBidirectWinograd_xdlops<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::
    GetTransformedConvContext(const ConvolutionContext& ctx) const
{
    DEFINE_GETXFORMHWSIZE(ctx)
    int batch_count = wino_xform_h * wino_xform_w * ctx.group_counts;
    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? ctx.in_data_type
            : miopenFloat;

    WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>
        wino_in = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Input, transform_data_type),
        wino_out = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Output, transform_data_type),
        wino_wei = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Weight, transform_data_type);

    // GNCHW -> GCNHW
    TensorDescriptor in, wei, out;
    miopenSet4dTensorDescriptor(&in,
                                transform_data_type,
                                1,
                                wino_in.buff_info.size.c * batch_count,
                                1,
                                wino_in.buff_info.size.w * wino_in.buff_info.size.h *
                                    wino_in.buff_info.size.nk);

    miopenSet4dTensorDescriptor(&wei,
                                transform_data_type,
                                wino_wei.buff_info.size.nk * batch_count,
                                wino_wei.buff_info.size.c,
                                wino_wei.buff_info.size.h,
                                wino_wei.buff_info.size.w);

    miopenSet4dTensorDescriptor(&out,
                                transform_data_type,
                                1,
                                wino_out.buff_info.size.c * batch_count,
                                1,
                                wino_out.buff_info.size.w * wino_out.buff_info.size.h *
                                    wino_out.buff_info.size.nk);

    // default conv_desc.
    // pads{0,0}, stride{1,1}, dilation {1, 1}
    // trans_output_pads = {0, 0},  group_count = gem_batch_count
    ConvolutionDescriptor conv_desc({0, 0}, {1, 1}, {1, 1}, {0, 0}, batch_count);

    auto dir = conv::Direction::Forward;

    ConvolutionContext transformed_ctx(in, wei, out, conv_desc, dir, 0);
    transformed_ctx.ExecutionContext::operator=(ctx);

    transformed_ctx.SetupFloats();
    return transformed_ctx;
}

// must be same as invoke_params in Invoker
template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
conv::DataInvokeParams GetTransformedInvokeContext(const ConvolutionContext& ctx,
                                                   const AnyInvokeParams& invoke_ctx)
{
#if MIOPEN_BACKEND_HIP
    const miopenDataType_t transform_data_type =
        miopen::IsEnabled(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_EXPEREMENTAL_FP16_TRANSFORM{})
            ? ctx.in_data_type
            : miopenFloat;
    WinogradBufferInfo<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>
        wino_in = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Input, transform_data_type),
        wino_out = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Output, transform_data_type),
        wino_wei = GetWinoBuffer<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, ConvWinoBuffType::Weight, transform_data_type);

    // There are no unread variables, but the compiler still shows a warning.
    // cppcheck-suppress unreadVariable
    const WinoOffsets transform_offset(wino_in.buff_info.total_byte_size,
                                       wino_out.buff_info.total_byte_size);

    const auto& data_ctx = invoke_ctx.CastTo<conv::DataInvokeParams>();

    auto workSpace = data_ctx.workSpace;

    const auto wino_in_ptr =
        static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.in);
    const auto wino_w_ptr =
        static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.wei);
    const auto wino_out_ptr =
        static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_offset.out);

    const auto transform_workSpaceSize = wino_in.buff_info.total_byte_size +
                                         wino_wei.buff_info.total_byte_size +
                                         wino_out.buff_info.total_byte_size;

    const auto gemm_workSpaceSize = data_ctx.workSpaceSize - transform_workSpaceSize;
    const auto gemm_workSpace =
        static_cast<void*>(reinterpret_cast<char*>(workSpace) + transform_workSpaceSize);
    const auto zeroDesc           = TensorDescriptor();
    ConvDataTensors xdlops_tensor = ConvDataTensors(
        ConvFwdTensors{zeroDesc, wino_in_ptr, zeroDesc, wino_w_ptr, zeroDesc, wino_out_ptr});
    return conv::DataInvokeParams{xdlops_tensor, gemm_workSpace, gemm_workSpaceSize};
#else
    (void)invoke_ctx;
    (void)ctx;
    MIOPEN_THROW(miopenStatusBadParm, "ConvMPBidirectWinograd is not supported ");
#endif
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
bool ConvMPBidirectWinograd_xdlops<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::IsApplicable(
    const ConvolutionContext& ctx) const
{

    static const int wino_data_tile   = std::max(WinoDataH, WinoDataW);
    static const int wino_filter_tile = std::max(WinoFilterH, WinoFilterW);

    if(wino_data_tile == 6 && wino_filter_tile == 3)
        if(IS_DISABLED(MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F6X3{}))
            return false;
    if(wino_data_tile == 5 && wino_filter_tile == 3)
        if(IS_DISABLED(MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F5X3{}))
            return false;
    if(wino_data_tile == 4 && wino_filter_tile == 3)
        if(IS_DISABLED(MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F4X3{}))
            return false;
    if(wino_data_tile == 3 && wino_filter_tile == 3)
        if(IS_DISABLED(MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F3X3{}))
            return false;
    if(wino_data_tile == 2 && wino_filter_tile == 3)
        if(IS_DISABLED(MIOPEN_DEBUG_AMD_MP_BD_XDLOPS_WINOGRAD_F2X3{}))
            return false;

    return IsApplicableTransform<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(ctx) &&
           ConvHipImplicitGemmForwardV4R4Xdlops().IsApplicable(GetTransformedConvContext(ctx));
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
ConvSolution
ConvMPBidirectWinograd_xdlops<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::GetSolution(
    const ConvolutionContext& ctx,
    const PerformanceImplicitGemmForwardV4R4Xdlops& config,
    bool) const
{
    ConvSolution wino_transform =
        ConvMPBidirectWinograd<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>{}.GetSolution(ctx);

    const ConvolutionContext xdlops_conv_ctx = GetTransformedConvContext(ctx);

    ConvSolution xdlops_conv =
        ConvHipImplicitGemmForwardV4R4Xdlops{}.GetSolution(xdlops_conv_ctx, config);

    ConvSolution result;
    result.workspce_sz = wino_transform.workspce_sz + xdlops_conv.workspce_sz;

    assert(xdlops_conv.construction_params.size() == 1);

    // change transform layout
    // GCNHW -> GNCHW
    std::ostringstream additional_options_wei;
    GenerateClangDefsym(additional_options_wei, "swap_filter_layout_KC", 1);
    wino_transform.construction_params[1].comp_options += additional_options_wei.str();

    result.construction_params.push_back(wino_transform.construction_params[0]);
    result.construction_params.push_back(wino_transform.construction_params[1]);
    result.construction_params.push_back(wino_transform.construction_params[2]);
    result.construction_params.push_back(xdlops_conv.construction_params[0]);

    result.invoker_factory =
        MakeWinogradInvokerFactory<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(
            ctx, xdlops_conv.invoker_factory.value(), true);

    return result;
}

template <int WinoDataH, int WinoFilterH, int WinoDataW, int WinoFilterW>
PerformanceImplicitGemmForwardV4R4Xdlops
ConvMPBidirectWinograd_xdlops<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>::Search(
    const ConvolutionContext& ctx, const AnyInvokeParams& invoke_ctx) const
{
    const auto transformed_invoke_ctx =
        GetTransformedInvokeContext<WinoDataH, WinoFilterH, WinoDataW, WinoFilterW>(ctx,
                                                                                    invoke_ctx);
    return ConvHipImplicitGemmForwardV4R4Xdlops().Search(GetTransformedConvContext(ctx),
                                                         transformed_invoke_ctx);
}

template struct ConvMPBidirectWinograd_xdlops<2, 3>;
template struct ConvMPBidirectWinograd_xdlops<3, 3>;
template struct ConvMPBidirectWinograd_xdlops<4, 3>;
template struct ConvMPBidirectWinograd_xdlops<5, 3>;
template struct ConvMPBidirectWinograd_xdlops<6, 3>;

} // namespace solver
} // namespace miopen
