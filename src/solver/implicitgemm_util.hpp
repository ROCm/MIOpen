#ifndef CK_IMPLICITGEMM_UTIL_HPP_
#define CK_IMPLICITGEMM_UTIL_HPP_

#include <miopen/env.hpp>
#include <miopen/hip_build_utils.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM)

namespace miopen {
namespace solver {

static inline int ImgHeight(const ConvolutionContext& c)
{
    return c.direction.IsForward() ? c.out_height : c.in_height;
}

static inline int ImgWidth(const ConvolutionContext& c)
{
    return c.direction.IsForward() ? c.out_width : c.in_width;
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

inline static int GetReadWriteVectorSize(const int v)
{
    return v % 4 == 0 ? 4 : (v % 2 == 0 ? 2 : 1);
}

inline static uint32_t GetEPackLength(const ConvolutionContext& ctx, bool isXdlopsInvoked)
{
    int C = ctx.n_inputs;
    int Y = ctx.kernel_size_h;
    int X = ctx.kernel_size_w;

    if(ctx.direction.IsBackwardWrW())
    {
        C = ctx.batch_sz;  // swapped
        Y = ctx.in_height; // swapped
        X = ctx.in_width;  // swapped
    }

    // Based on data type, Es are packed
    int EPACK = 1;
    if(ctx.IsFp16()) // for fp16, either 2 or 4 Es could be packed
    {
        if(IsXdlopsSupport(ctx) && isXdlopsInvoked) // in xdlops, 4 fp16s are packed
            EPACK = 4;
        else // for fp16, either 2 or 4 Es could be packed in non-xdlops scenarios.
            EPACK = (C * Y * X % 32) == 0 ? 4 : 2;
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
    bool amd_inline_asm = !miopen::IsDisabled(MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM{});

    if(StartsWith(ctx.GetStream().GetDeviceName(), "gfx8"))
        amd_inline_asm = false;

    if(!(StartsWith(ctx.GetStream().GetDeviceName(), "gfx906") ||
         StartsWith(ctx.GetStream().GetDeviceName(), "gfx908")) &&
       ctx.IsFp16())
        amd_inline_asm = false;

    return amd_inline_asm;
}

} // namespace solver
} // namespace miopen
#endif
