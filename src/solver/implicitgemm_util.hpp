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
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM)

#define WORKAROUND_SWDEV_200782 1

#define WORKAROUND_SWDEV_229277_227616_229195 1

namespace miopen {
namespace solver {

// greatest common divisor, aka highest common factor
template <typename T>
T gcd(T x, T y)
{
    if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x - y, y);
    }
    else
    {
        return gcd(x, y - x);
    }
}

template <typename T, typename... Ys>
T gcd(T x, Ys... ys)
{
    return gcd(x, gcd(ys...));
}

// least common multiple
template <typename T>
T lcm(T x, T y)
{
    if(x == 0 || y == 0)
    {
        return 0;
    }
    else
    {
        return (x * y) / gcd(x, y);
    }
}

template <typename T, typename... Ys>
T lcm(T x, Ys... ys)
{
    return lcm(x, lcm(ys...));
}

template <typename T>
T integer_divide_ceil(T x, T y)
{
    if(y == 0)
    {
        MIOPEN_THROW("divisor should not be 0");
    }

    return (x + y - 1) / y;
}

template <typename T>
T integer_least_multiple(T x, T y)
{
    return y * integer_divide_ceil(x, y);
}

// 1. get the original dimension of conv problem
//    (undo the dimeniosn swapping happened inside ConvolutionContext)
// 2. adjust right padding size to align with the way implicit GEMM deal with padding
struct ConvolutionContextInterpreter
{
    static auto GetGroupCountG(const ConvolutionContext& c) { return c.group_counts; }

    static auto GetBatchN(const ConvolutionContext& c) { return c.batch_sz; }

    static auto GetOutputChannelK(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.n_outputs;
        else
            return c.n_inputs;
    }

    static auto GetInputChannelC(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.n_inputs;
        else
            return c.n_outputs;
    }

    static auto GetInputDepthDi(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.in_depth;
        else
            return c.out_depth;
    }

    static auto GetInputHeightHi(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.in_height;
        else
            return c.out_height;
    }

    static auto GetInputWidthWi(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.in_width;
        else
            return c.out_width;
    }

    static auto GetOutputDepthDo(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.out_depth;
        else
            return c.in_depth;
    }

    static auto GetOutputHeightHo(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.out_height;
        else
            return c.in_height;
    }

    static auto GetOutputWidthWo(const ConvolutionContext& c)
    {
        if(c.direction.IsForward())
            return c.out_width;
        else
            return c.in_width;
    }

    static auto GetFilterDepthZ(const ConvolutionContext& c) { return c.kernel_size_d; }

    static auto GetFilterHeightY(const ConvolutionContext& c) { return c.kernel_size_h; }

    static auto GetFilterWidthX(const ConvolutionContext& c) { return c.kernel_size_w; }

    // adjust conv_stride_d to 1 if Do is 1
    static auto GetAdjustedConvolutionStrideD(const ConvolutionContext& c)
    {
        return GetOutputDepthDo(c) > 1 ? c.kernel_stride_d : 1;
    }

    // adjust conv_stride_h to 1 if Ho is 1
    static auto GetAdjustedConvolutionStrideH(const ConvolutionContext& c)
    {
        return GetOutputHeightHo(c) > 1 ? c.kernel_stride_h : 1;
    }

    // adjust conv_stride_w to 1 if Wo is 1
    static auto GetAdjustedConvolutionStrideW(const ConvolutionContext& c)
    {
        return GetOutputWidthWo(c) > 1 ? c.kernel_stride_w : 1;
    }

    // adjust conv_dilation_d to 1 if Z is 1
    static auto GetAdjustedConvolutionDilationD(const ConvolutionContext& c)
    {
        return GetFilterDepthZ(c) > 1 ? c.kernel_dilation_d : 1;
    }

    // adjust conv_dilation_h to 1 if Y is 1
    static auto GetAdjustedConvolutionDilationH(const ConvolutionContext& c)
    {
        return GetFilterHeightY(c) > 1 ? c.kernel_dilation_h : 1;
    }

    // adjust conv_dilation_w to 1 if X is 1
    static auto GetAdjustedConvolutionDilationW(const ConvolutionContext& c)
    {
        return GetFilterWidthX(c) > 1 ? c.kernel_dilation_w : 1;
    }

    static auto GetInputLeftPadD(const ConvolutionContext& c) { return c.pad_d; }

    static auto GetInputLeftPadH(const ConvolutionContext& c) { return c.pad_h; }

    static auto GetInputLeftPadW(const ConvolutionContext& c) { return c.pad_w; }

    // adjust right padding size to align with the way implicit GEMM deal with padding
    static auto GetAdjustedInputRightPadD(const ConvolutionContext& c)
    {
        int di              = GetInputDepthDi(c);
        int dout            = GetOutputDepthDo(c);
        int z               = GetFilterDepthZ(c);
        int conv_stride_d   = GetAdjustedConvolutionStrideD(c);
        int conv_dilation_d = GetAdjustedConvolutionDilationD(c);
        int in_left_pad_d   = GetInputLeftPadD(c);

        int di_padded = 1 + (z - 1) * conv_dilation_d + (dout - 1) * conv_stride_d;

        int in_right_pad_d =
            di_padded > (in_left_pad_d + di) ? di_padded - (in_left_pad_d + di) : 0;

        return in_right_pad_d;
    }

    // adjust right padding size to align with the way implicit GEMM deal with padding
    static auto GetAdjustedInputRightPadH(const ConvolutionContext& c)
    {
        int hi              = GetInputHeightHi(c);
        int ho              = GetOutputHeightHo(c);
        int y               = GetFilterHeightY(c);
        int conv_stride_h   = GetAdjustedConvolutionStrideH(c);
        int conv_dilation_h = GetAdjustedConvolutionDilationH(c);
        int in_left_pad_h   = GetInputLeftPadH(c);

        int hi_padded = 1 + (y - 1) * conv_dilation_h + (ho - 1) * conv_stride_h;

        int in_right_pad_h =
            hi_padded > (in_left_pad_h + hi) ? hi_padded - (in_left_pad_h + hi) : 0;

        return in_right_pad_h;
    }

    // adjust right padding size to align with the way implicit GEMM deal with padding
    static auto GetAdjustedInputRightPadW(const ConvolutionContext& c)
    {
        int wi              = GetInputWidthWi(c);
        int wo              = GetOutputWidthWo(c);
        int x               = GetFilterWidthX(c);
        int conv_stride_w   = GetAdjustedConvolutionStrideW(c);
        int conv_dilation_w = GetAdjustedConvolutionDilationW(c);
        int in_left_pad_w   = GetInputLeftPadW(c);

        int wi_padded = 1 + (x - 1) * conv_dilation_w + (wo - 1) * conv_stride_w;

        int in_right_pad_w =
            wi_padded > (in_left_pad_w + wi) ? wi_padded - (in_left_pad_w + wi) : 0;

        return in_right_pad_w;
    }
};

// these functions map the dimensions of a bwd-wrw problem into a fwd problem
// they are not supposed to be called by backward-data
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

    // disable xdlops kernels by default due to possible failures:
    // 1) inline asm may crash
    // 2) llvm intrin may has incorrect results
    return StartsWith(c.GetStream().GetDeviceName(), "gfx908") &&
#if WORKAROUND_SWDEV_200782
           /// \todo Remove workaround when we drop suport of HCC older than 2.10.19392.
           ((miopen::HipGetHccVersion() >= external_tool_version_t{2, 10, 19392})
                ? !miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS{})
                : miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS{}));
#else
           !miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS{});
#endif
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

static inline bool IsValidXdlopsGemm(const int GemmMPerBlock,
                                     const int GemmNPerBlock,
                                     const int GemmKPackedPerBlock, // packed
                                     const int GemmMPerWave,
                                     const int GemmNPerWave)
{
    // unsupported xdlops-gemm
    if(GemmMPerWave == 16 && GemmNPerWave == 32)
        return false;
    if(GemmMPerWave == 32 && GemmNPerWave == 16)
        return false;
    if(GemmMPerWave == 8 && GemmNPerWave != 64)
        return false;
    if(GemmMPerWave == 4 && GemmNPerWave != 64)
        return false;
    if(GemmMPerWave == 32 && GemmNPerWave == 32 && GemmKPackedPerBlock % 2 != 0)
        return false;
    if(GemmMPerWave == 16 && GemmNPerWave == 16 && GemmKPackedPerBlock % 4 != 0)
        return false;

    const auto WaveSize  = 64;
    const auto BlockSize = GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * WaveSize;

    // fail with blockSize >= 512
    /// \todo fix the issue with blockSize >= 512
    if(BlockSize < 64 || BlockSize > 256)
        return false;

    return (GemmMPerBlock % GemmMPerWave) == 0 && (GemmNPerBlock % GemmNPerWave) == 0;
}

static inline bool
IsValidGridGemmXdlops(const std::size_t GemmM, const std::size_t GemmN, const std::size_t GemmK)
{
    // unsupported xdlops-gemm
    if(GemmM % 16 != 0 && GemmN % 64 != 0)
        return false;

    const auto WaveSize = 64;

    return (GemmM * GemmN) % 256 == 0 && (GemmK * GemmM) % WaveSize == 0 &&
           (GemmK * GemmN) % WaveSize == 0 && GemmN % 16 == 0 && GemmM % 4 == 0 && GemmK % 4 == 0;
}

static inline bool IsApplicableXdlops(const ConvolutionContext& ctx)
{
    if(!IsXdlopsSupport(ctx))
        return false;

    std::size_t n  = ConvolutionContextInterpreter::GetBatchN(ctx);
    std::size_t k  = ConvolutionContextInterpreter::GetOutputChannelK(ctx) / ctx.group_counts;
    std::size_t c  = ConvolutionContextInterpreter::GetInputChannelC(ctx) / ctx.group_counts;
    std::size_t y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    std::size_t x  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
    std::size_t ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    std::size_t wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);

    std::size_t GemmM, GemmN, GemmK;
    // forward
    if(ctx.direction.IsForward())
    {
        // TBD/ Since bfp16/fp16 fwd kernel extracts epack from c*y*x,
        //      one could relax the following restriction for bfp16/fp16,
        //      allowing c=1 when y*x=epack.
        if(c % GetEPackLength(ctx, true) != 0)
            return false;
        const auto nonVectorizedC = c / GetEPackLength(ctx, true);
        GemmM                     = k;
        GemmN                     = static_cast<std::size_t>(n) * ho * wo;
        GemmK                     = static_cast<std::size_t>(nonVectorizedC) * y * x;
    }
    // backwardData
    else if(ctx.direction.IsBackwardData())
    {
        if(k % GetEPackLength(ctx, true) != 0)
            return false;
        const auto nonVectorizedK = k / GetEPackLength(ctx, true);
        GemmM                     = static_cast<std::size_t>(c) * y * x;
        GemmN                     = static_cast<std::size_t>(n) * ho * wo;
        GemmK                     = nonVectorizedK;
    }
    // backwardWeights
    else
    {
        if(n % GetEPackLength(ctx, true) != 0)
            return false;
        const auto nonVectorizedN = n / GetEPackLength(ctx, true);
        GemmM                     = k;
        GemmN                     = static_cast<std::size_t>(c) * y * x;
        GemmK                     = static_cast<std::size_t>(nonVectorizedN) * ho * wo;
    }

    return IsValidGridGemmXdlops(GemmM, GemmN, GemmK);
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
                                            const unsigned int EPACKSize)
{
    // Extend lds size by to take into account alignment
    // See max_algin code inside kernel_aglorithm files
    const std::size_t worst_case_alignment_adjustment =
        (ctx.IsBfp16() || ctx.IsFp16())
            ? std::max(
                  {GetReadWriteVectorSize(static_cast<int>(InBlockCopySubLengths_B)), EPACKSize})
            : std::max({GetReadWriteVectorSize(static_cast<int>(WeiBlockCopySubLengths_K)),
                        GetReadWriteVectorSize(static_cast<int>(InBlockCopySubLengths_B)),
                        GemmDataPerReadA,
                        GemmDataPerReadB});

    // Multiplied worst_case_alignment_adjustment by 2 as
    // Both A and B matrix LDS size is increased.
    const std::size_t lds_size =
        (BPerBlock + KPerBlock) * EPerBlock * EPACKSize * GetTypeSize(ctx.in_data_type) * 2 +
        2 * worst_case_alignment_adjustment;

    return lds_size;
}

template <typename BotBufType, typename TopBufType, typename WeiBufType>
static inline int RunAndMeasureSolutionBase(miopen::Handle& profile_h,
                                            BotBufType bot_buf,
                                            TopBufType top_buf,
                                            WeiBufType wei_buf,
                                            const ConvolutionContext& ctx,
                                            const ConvSolution& solution,
                                            float& elapsed_time)
{
#ifdef NDEBUG
    try
#endif
    {
        elapsed_time = float(0);

        for(auto& k_info : solution.construction_params)
        {
            auto kernel = profile_h.AddKernel("",
                                              "",
                                              k_info.kernel_file,
                                              k_info.kernel_name,
                                              k_info.l_wk,
                                              k_info.g_wk,
                                              k_info.comp_options);

            if(ctx.direction.IsBackwardWrW())
            {
                kernel(top_buf, bot_buf, wei_buf);
            }
            if(ctx.direction.IsBackwardData())
            {
                kernel(top_buf, wei_buf, bot_buf);
            }
            if(ctx.direction.IsForward())
            {
                kernel(bot_buf, wei_buf, top_buf);
            }

            elapsed_time += profile_h.GetKernelTime();
        }
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

static inline bool support_amd_buffer_atomic_add(const ConvolutionContext& ctx)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    return StartsWith(device_name, "gfx908") && ctx.IsFp32();
}

template <typename T>
int amd_buffer_load_max_length()
{
    if(std::is_same<float, T>())
    {
        return 4;
    }
    else
    {
        MIOPEN_LOG_I("not implemented");
        return 1;
    }
}

template <typename T>
int amd_buffer_store_max_length()
{
    if(std::is_same<float, T>())
    {
        return 4;
    }
    else
    {
        MIOPEN_LOG_I("not implemented");
        return 1;
    }
}

template <typename T>
int amd_lds_read_max_length()
{
    if(std::is_same<float, T>())
    {
        return 4;
    }
    else
    {
        MIOPEN_LOG_I("not implemented");
        return 1;
    }
}

template <typename T>
int amd_lds_write_max_length()
{
    if(std::is_same<float, T>())
    {
        return 4;
    }
    else
    {
        MIOPEN_LOG_I("not implemented");
        return 1;
    }
}

constexpr std::size_t get_lds_max_number_of_byte() { return 65536; }

} // namespace solver
} // namespace miopen
#endif
