#ifndef CK_IMPLICITGEMM_UTIL_HPP_
#define CK_IMPLICITGEMM_UTIL_HPP_

#include <algorithm>
#include <miopen/env.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/mlo_internal.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS)
MIOPEN_DECLARE_ENV_VAR(
    MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE) // For internal debug purposes

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM)

namespace miopen {
namespace solver {

static inline std::size_t KernelFilterStrideH(const ConvolutionContext& c)
{
    if(c.direction.IsBackwardWrW())
        return c.kernel_dilation_h;
    else
        return c.kernel_stride_h;
}

static inline std::size_t KernelFilterStrideW(const ConvolutionContext& c)
{
    if(c.direction.IsBackwardWrW())
        return c.kernel_dilation_w;
    else
        return c.kernel_stride_w;
}

static inline std::size_t KernelFilterDilationH(const ConvolutionContext& c)
{
    if(c.direction.IsBackwardWrW())
        return c.kernel_stride_h;
    else
        return c.kernel_dilation_h;
}

static inline std::size_t KernelFilterDilationW(const ConvolutionContext& c)
{
    if(c.direction.IsBackwardWrW())
        return c.kernel_stride_w;
    else
        return c.kernel_dilation_w;
}

static inline std::size_t KernelOutputChannelK(const ConvolutionContext& c)
{
    if(c.direction.IsBackwardWrW())
        return c.n_inputs;
    else
        return c.n_outputs;
}

static inline std::size_t KernelInputChannelC(const ConvolutionContext& c)
{
    if(c.direction.IsBackwardWrW())
        return c.batch_sz;
    else
        return c.n_inputs / c.group_counts;
}

static inline std::size_t KernelBatchN(const ConvolutionContext& c)
{
    if(c.direction.IsBackwardWrW())
        return c.n_outputs / c.group_counts;
    else
        return c.batch_sz;
}

static inline std::size_t KernelOutputHeightHo(const ConvolutionContext& c)
{
    if(c.direction.IsForward())
        return c.out_height;
    else if(c.direction.IsBackwardWrW())
        return c.kernel_size_h;
    else
        return c.in_height;
}

static inline std::size_t KernelOutputWidthWo(const ConvolutionContext& c)
{
    if(c.direction.IsForward())
        return c.out_width;
    else if(c.direction.IsBackwardWrW())
        return c.kernel_size_w;
    else
        return c.in_width;
}

static inline std::size_t KernelFilterWidthX(const ConvolutionContext& c)
{
    if(c.direction.IsBackwardWrW())
        return c.in_width;
    else
        return c.kernel_size_w;
}

static inline std::size_t KernelFilterHeightY(const ConvolutionContext& c)
{
    if(c.direction.IsBackwardWrW())
        return c.in_height;
    else
        return c.kernel_size_h;
}

/// \todo move to separate header and use in other solvers.
template <int L, int H>
inline static bool IsTwoPower(const int v)
{
    static_assert(L <= H, "L <= H");
    if(((v - 1) & v) != 0)
        return false;
    return L <= v && v <= H;
}

template <int L, int H>
inline static bool NextTwoPower(int& v)
{
    static_assert((((L - 1) & L) == 0), "L is not power of 2");
    static_assert((((H - 1) & H) == 0), "H is not power of 2");
    assert((IsTwoPower<L, H>(v)));
    if(v == H)
    {
        v = L;
        return true;
    }
    v *= 2;
    return false;
}

static inline bool IsXdlopsSupport(const ConvolutionContext& c)
{
    if(miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE{}))
        return true;

    return StartsWith(c.GetStream().GetDeviceName(), "gfx908") &&
           // disable xdlops kernels by default due to possible failures:
           // 1) inline asm may crash
           // 2) llvm intrin may has incorrect results
           /// \todo enable xdlops kernels by default after llvm intrin fix (SWDEV-200782) in
           /// release
           ((miopen::HipGetHccVersion() >= external_tool_version_t{2, 10, 19392})
                ? !miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS{})
                : miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS{}));
}

inline static uint32_t GetReadWriteVectorSize(const int v)
{
    return v % 4 == 0 ? 4 : (v % 2 == 0 ? 2 : 1);
}

inline static uint32_t GetEPackLength(const ConvolutionContext& ctx, bool isXdlopsInvoked)
{
    // Based on data type, Es are packed
    int EPACK = 1;
    if(ctx.IsFp16()) // for fp16, either 2 or 4 Es could be packed
    {
        if(IsXdlopsSupport(ctx) && isXdlopsInvoked) // in xdlops, 4 fp16s are packed
            EPACK = 4;
        else // for fp16, either 2 or 4 Es could be packed in non-xdlops scenarios.
            // EPACK = (C * Y * X % 32) == 0 ? 4 : 2;
            EPACK = 2;
    }
    else if(ctx.IsBfp16()) // for bfp16, only 2 Es could be packed
    {
        EPACK = 2;
    }
    return EPACK;
}

template <class PerformanceImplicitGemm_t>
inline static auto GetPerformanceConfigBase(const ConvolutionContext& ctx)
{
    PerformanceImplicitGemm_t pp;
    pp.EuristicInit(ctx);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

static inline size_t ComputeLDSRequiredSize(const ConvolutionContext& ctx,
                                            const int BPerBlock,
                                            const int KPerBlock,
                                            const int EPerBlock,
                                            const unsigned int GemmDataPerReadA,
                                            const unsigned int GemmDataPerReadB,
                                            const unsigned int InBlockCopySubLengths_B,
                                            const unsigned int WeiBlockCopySubLengths_K,
                                            bool isXdlopsUsed)
{
    // Extend lds size by to take into account alignment
    // See max_algin code inside kernel_aglorithm files
    const std::size_t worst_case_alignment_adjustment =
        (ctx.IsBfp16() || ctx.IsFp16())
            ? std::max({GetReadWriteVectorSize(static_cast<int>(InBlockCopySubLengths_B)),
                        GetEPackLength(ctx, isXdlopsUsed)})
            : std::max({GetReadWriteVectorSize(static_cast<int>(WeiBlockCopySubLengths_K)),
                        GetReadWriteVectorSize(static_cast<int>(InBlockCopySubLengths_B)),
                        GemmDataPerReadA,
                        GemmDataPerReadB});

    // Multiplied worst_case_alignment_adjustment by 2 as
    // Both A and B matrix LDS size is increased.
    const std::size_t lds_size = (BPerBlock + KPerBlock) * EPerBlock * GetEPackLength(ctx, true) *
                                     GetTypeSize(ctx.in_data_type) * 2 +
                                 2 * worst_case_alignment_adjustment;

    return lds_size;
}

static inline int RunAndMeasureSolutionBase(miopen::Handle& profile_h,
                                            ConstData_t bot_buf,
                                            Data_t top_buf,
                                            ConstData_t wei_buf,
                                            const ConvolutionContext& ctx,
                                            const ConvSolution& solution,
                                            float& elapsed_time)
{
    KernelInfo k_info;

    k_info = solution.construction_params[0];

#ifdef NDEBUG
    try
#endif
    {
        elapsed_time = std::numeric_limits<float>::max();
        auto kernel  = profile_h.AddKernel("",
                                          "",
                                          k_info.kernel_file,
                                          k_info.kernel_name,
                                          k_info.l_wk,
                                          k_info.g_wk,
                                          k_info.comp_options);

        if(ctx.direction.IsBackwardWrW())
        {
            kernel(bot_buf, top_buf, wei_buf);
        }
        if(ctx.direction.IsBackwardData())
        {
            kernel(top_buf, wei_buf, bot_buf);
        }
        if(ctx.direction.IsForward())
        {
            kernel(bot_buf, wei_buf, top_buf);
        }

        elapsed_time = profile_h.GetKernelTime();
    }
#ifdef NDEBUG
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return -1;
    }
#endif
    return 0;
}

static inline bool use_amd_inline_asm(const ConvolutionContext& ctx)
{

    if(StartsWith(ctx.GetStream().GetDeviceName(), "gfx8"))
        return false;

    // disable fp16 inline asm for <= gfx900
    const auto device_name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(device_name, "gfx906") || StartsWith(device_name, "gfx908")) && ctx.IsFp16())
        return false;

    return !miopen::IsDisabled(MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM{});
}

} // namespace solver
} // namespace miopen
#endif
