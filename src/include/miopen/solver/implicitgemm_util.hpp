/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#ifndef GUARD_IMPLICITGEMM_UTIL_HPP_
#define GUARD_IMPLICITGEMM_UTIL_HPP_

#include <miopen/env.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/rocm_features.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <algorithm>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM)

#define WORKAROUND_SWDEV_229277_227616_229195 1
// workaround for unnecessary VGPA <--> AGRP data movement when using mfma LLVM intrinsic
#define WORKAROUND_SWDEV_229564 1
// workaround for buffer load/store fp16/bfp16 intrinsic bug
#define WORKAROUND_SWDEV_231101 1
// due to compiler bug, iGEMM xdlops kernels fail verification in some cases, if using "-O3" flag,
// (but will pass verification with "-O1" flag)
#define WORKAROUND_SWDEV_251757 1
// although gfx1030 supports buffer instructions,but it not work properly when we use the
// corresponding llvm intrinsic functions
// so we disable using those llvm intrinsic functions on gfx1030
#define WORKAROUND_MIOPEN_ISSUE_557 1

namespace miopen {

namespace solver {

// these functions map the dimensions of a bwd-wrw problem into a fwd problem
// they are not supposed to be called by backward-data
static inline std::size_t KernelFilterStrideH(const ProblemDescription& problem)
{
    if(problem.direction.IsBackwardWrW())
        return problem.kernel_dilation_h;
    else
        return problem.kernel_stride_h;
}

static inline std::size_t KernelFilterStrideW(const ProblemDescription& problem)
{
    if(problem.direction.IsBackwardWrW())
        return problem.kernel_dilation_w;
    else
        return problem.kernel_stride_w;
}

static inline std::size_t KernelFilterDilationH(const ProblemDescription& problem)
{
    if(problem.direction.IsBackwardWrW())
        return problem.kernel_stride_h;
    else
        return problem.kernel_dilation_h;
}

static inline std::size_t KernelFilterDilationW(const ProblemDescription& problem)
{
    if(problem.direction.IsBackwardWrW())
        return problem.kernel_stride_w;
    else
        return problem.kernel_dilation_w;
}

static inline std::size_t KernelOutputChannelK(const ProblemDescription& problem)
{
    if(problem.direction.IsBackwardWrW())
        return problem.n_inputs;
    else
        return problem.n_outputs;
}

static inline std::size_t KernelInputChannelC(const ProblemDescription& problem)
{
    if(problem.direction.IsBackwardWrW())
        return problem.batch_sz;
    else
        return problem.n_inputs / problem.group_counts;
}

static inline std::size_t KernelBatchN(const ProblemDescription& problem)
{
    if(problem.direction.IsBackwardWrW())
        return problem.n_outputs / problem.group_counts;
    else
        return problem.batch_sz;
}

static inline std::size_t KernelOutputHeightHo(const ProblemDescription& problem)
{
    if(problem.direction.IsForward())
        return problem.out_height;
    else if(problem.direction.IsBackwardWrW())
        return problem.kernel_size_h;
    else
        return problem.in_height;
}

static inline std::size_t KernelOutputWidthWo(const ProblemDescription& problem)
{
    if(problem.direction.IsForward())
        return problem.out_width;
    else if(problem.direction.IsBackwardWrW())
        return problem.kernel_size_w;
    else
        return problem.in_width;
}

static inline std::size_t KernelFilterWidthX(const ProblemDescription& problem)
{
    if(problem.direction.IsBackwardWrW())
        return problem.in_width;
    else
        return problem.kernel_size_w;
}

static inline std::size_t KernelFilterHeightY(const ProblemDescription& problem)
{
    if(problem.direction.IsBackwardWrW())
        return problem.in_height;
    else
        return problem.kernel_size_h;
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

template <int L, int H>
inline static bool PreviousTwoPower(int& v)
{
    static_assert((((L - 1) & L) == 0), "L is not power of 2");
    static_assert((((H - 1) & H) == 0), "H is not power of 2");
    assert((IsTwoPower<L, H>(v)));
    if(v == L)
    {
        v = H;
        return true;
    }
    v /= 2;
    return false;
}

template <bool L, bool H>
inline static bool NextFlag(bool& v)
{
    if(v == H)
    {
        v = L;
        return true;
    }
    v = H;
    return false;
}

static inline bool IsXdlopsSupport(const ExecutionContext& ctx)
{
    if(miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE{}))
        return true;

    // disable xdlops kernels by default due to possible failures:
    // 1) inline asm may crash
    // 2) llvm intrin may has incorrect results
    bool is_xdlops_supported = StartsWith(ctx.GetStream().GetDeviceName(), "gfx908") ||
                               StartsWith(ctx.GetStream().GetDeviceName(), "gfx90a");
    return is_xdlops_supported && !miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS{});
}

///\todo remove
inline static uint32_t GetReadWriteVectorSize(const int v)
{
    return v % 4 == 0 ? 4 : (v % 2 == 0 ? 2 : 1);
}

///\todo remove
inline static uint32_t
GetEPackLength(const ExecutionContext& ctx, const ProblemDescription& problem, bool isXdlopsInvoked)
{
    // Based on data type, Es are packed
    int EPACK = 1;
    if(problem.IsFp16()) // for fp16, either 2 or 4 Es could be packed
    {
        if(IsXdlopsSupport(ctx) && isXdlopsInvoked) // in xdlops, 4 fp16s are packed
            EPACK = 4;
        else // for fp16, either 2 or 4 Es could be packed in non-xdlops scenarios.
            // EPACK = (C * Y * X % 32) == 0 ? 4 : 2;
            EPACK = 2;
    }
    else if(problem.IsBfp16()) // for bfp16, only 2 Es could be packed
    {
        EPACK = 2;
    }
    return EPACK;
}

///\todo remove
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
    if(GemmMPerWave > 64 && GemmNPerWave < 64)
        return false;
    if(GemmNPerWave > 64 && GemmMPerWave < 64)
        return false;

    const auto WaveSize = 64;
    const auto BlockSize =
        (GemmNPerBlock * GemmMPerBlock) / (GemmMPerWave * GemmNPerWave) * WaveSize;

    if(BlockSize < 64 || BlockSize > 256)
        return false;

    return (GemmMPerBlock % GemmMPerWave) == 0 && (GemmNPerBlock % GemmNPerWave) == 0;
}

static inline bool IsIndexRangeLargeEnough(const ProblemDescription& problem)
{
    // composable kernel use int32_t for memory offset, which covers 2GB of memory maximum
    const std::size_t max_index_range = std::size_t(2) * 1024 * 1024 * 1024;

    return problem.bot_sz < max_index_range && problem.weights_sz < max_index_range &&
           problem.top_sz < max_index_range;
}

static inline bool IsValidBlockwiseGemmXdlops(const ProblemDescription& problem,
                                              const int GemmMPerBlock,
                                              const int GemmNPerBlock,
                                              const int GemmKPerBlock,
                                              const int GemmMPerWave,
                                              const int GemmNPerWave,
                                              const int GemmKPack)
{
#if WORKAROUND_SWDEV_251757
    if(problem.IsFp32() && GemmKPerBlock == 1 && GemmKPack == 8)
        return false;
#endif

    // check k
    if(problem.IsFp16() && GemmKPack % 4 != 0)
        return false;
    if(problem.IsBfp16() && GemmKPack % 2 != 0)
        return false;

    // check M, N and K
    std::vector<std::tuple<int, int, int>> validWaveGemmSize = {// std::make_tuple(128, 128, 1),
                                                                std::make_tuple(128, 64, 1),
                                                                // std::make_tuple(128, 32, 1),
                                                                // std::make_tuple(128, 16, 1),
                                                                std::make_tuple(64, 128, 1),
                                                                std::make_tuple(64, 64, 1),
                                                                std::make_tuple(64, 32, 1),
                                                                std::make_tuple(64, 16, 1),
                                                                // std::make_tuple(32, 128, 1),
                                                                std::make_tuple(32, 64, 1),
                                                                std::make_tuple(32, 32, 2),
                                                                // std::make_tuple(16, 128, 1),
                                                                std::make_tuple(16, 64, 1),
                                                                std::make_tuple(16, 16, 4),
                                                                // std::make_tuple(8, 128, 1),
                                                                std::make_tuple(8, 64, 1),
                                                                // std::make_tuple(4, 128, 1),
                                                                std::make_tuple(4, 64, 1)};

    if(!std::any_of(validWaveGemmSize.cbegin(),
                    validWaveGemmSize.cend(),
                    [GemmMPerWave, GemmNPerWave, GemmKPerBlock](const auto it) noexcept -> bool {
                        int validMPerWave, validNPerWave, validKPerWave;
                        std::tie(validMPerWave, validNPerWave, validKPerWave) = it;
                        return (GemmMPerWave == validMPerWave) && (GemmNPerWave == validNPerWave) &&
                               (GemmKPerBlock % validKPerWave == 0);
                    }))
        return false;

    const auto WaveSize = 64;
    const auto BlockSize =
        (GemmNPerBlock * GemmMPerBlock) / (GemmMPerWave * GemmNPerWave) * WaveSize;

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

///\todo remove
static inline bool IsApplicableXdlops(const ExecutionContext& ctx,
                                      const ProblemDescription& problem)
{
    if(!IsXdlopsSupport(ctx))
        return false;

    std::size_t n  = ProblemInterpreter::GetBatchN(problem);
    std::size_t k  = ProblemInterpreter::GetOutputChannelK(problem) / problem.group_counts;
    std::size_t c  = ProblemInterpreter::GetInputChannelC(problem) / problem.group_counts;
    std::size_t y  = ProblemInterpreter::GetFilterHeightY(problem);
    std::size_t x  = ProblemInterpreter::GetFilterWidthX(problem);
    std::size_t ho = ProblemInterpreter::GetOutputHeightHo(problem);
    std::size_t wo = ProblemInterpreter::GetOutputWidthWo(problem);

    std::size_t GemmM, GemmN, GemmK;
    // forward
    if(problem.direction.IsForward())
    {
        // TBD/ Since bfp16/fp16 fwd kernel extracts epack from c*y*x,
        //      one could relax the following restriction for bfp16/fp16,
        //      allowing c=1 when y*x=epack.
        if(c % GetEPackLength(ctx, problem, true) != 0)
            return false;
        const auto nonVectorizedC = c / GetEPackLength(ctx, problem, true);
        GemmM                     = k;
        GemmN                     = static_cast<std::size_t>(n) * ho * wo;
        GemmK                     = static_cast<std::size_t>(nonVectorizedC) * y * x;
    }
    // backwardData
    else if(problem.direction.IsBackwardData())
    {
        if(k % GetEPackLength(ctx, problem, true) != 0)
            return false;
        const auto nonVectorizedK = k / GetEPackLength(ctx, problem, true);
        GemmM                     = static_cast<std::size_t>(c) * y * x;
        GemmN                     = static_cast<std::size_t>(n) * ho * wo;
        GemmK                     = nonVectorizedK;
    }
    // backwardWeights
    else
    {
        if(n % GetEPackLength(ctx, problem, true) != 0)
            return false;
        const auto nonVectorizedN = n / GetEPackLength(ctx, problem, true);
        GemmM                     = k;
        GemmN                     = static_cast<std::size_t>(c) * y * x;
        GemmK                     = static_cast<std::size_t>(nonVectorizedN) * ho * wo;
    }

    return IsValidGridGemmXdlops(GemmM, GemmN, GemmK);
}

///\todo remove
template <class PerformanceImplicitGemm_t>
inline static auto GetPerformanceConfigBase(const ConvolutionContext& ctx,
                                            const ProblemDescription& problem)
{
    PerformanceImplicitGemm_t pp;
    pp.HeuristicInit(ctx, problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

///\todo remove
static inline size_t ComputeLDSRequiredSize(const ProblemDescription& problem,
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
        (problem.IsBfp16() || problem.IsFp16())
            ? std::max(
                  {GetReadWriteVectorSize(static_cast<int>(InBlockCopySubLengths_B)), EPACKSize})
            : std::max({GetReadWriteVectorSize(static_cast<int>(WeiBlockCopySubLengths_K)),
                        GetReadWriteVectorSize(static_cast<int>(InBlockCopySubLengths_B)),
                        GemmDataPerReadA,
                        GemmDataPerReadB});

    // Multiplied worst_case_alignment_adjustment by 2 as
    // Both A and B matrix LDS size is increased.
    const std::size_t lds_size = (static_cast<std::size_t>(BPerBlock) + KPerBlock) * EPerBlock *
                                     EPACKSize * GetTypeSize(problem.in_data_type) * 2 +
                                 2 * static_cast<std::size_t>(worst_case_alignment_adjustment);

    return lds_size;
}

static inline bool use_amd_inline_asm(const ConvolutionContext& ctx,
                                      const ProblemDescription& problem)
{

    if(StartsWith(ctx.GetStream().GetDeviceName(), "gfx8"))
        return false;

    // disable fp16 inline asm for <= gfx900
    const auto device_name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(device_name, "gfx906") || StartsWith(device_name, "gfx908")) &&
       problem.IsFp16())
        return false;

    return !miopen::IsDisabled(MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM{});
}

static inline bool is_use_amd_buffer_load_store(const ConvolutionContext& ctx)
{
#if WORKAROUND_MIOPEN_ISSUE_557
    const auto device_name = ctx.GetStream().GetDeviceName();
    return !StartsWith(device_name, "gfx103");
#else
    return true;
#endif
}

static inline bool is_use_v_fmac_f32(const ConvolutionContext& ctx)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    return StartsWith(device_name, "gfx103");
}

static inline bool support_amd_buffer_atomic_fadd(const std::string& device_name)
{
    return StartsWith(device_name, "gfx908");
}

template <typename T>
int amd_buffer_load_max_length()
{
    if(std::is_same<float, T>())
    {
        return 4;
    }
    else if(std::is_same<half_float::half, T>())
    {
        return 8;
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
    else if(std::is_same<half_float::half, T>())
    {
        return 8;
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
    else if(std::is_same<half_float::half, T>())
    {
        return 8;
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
    else if(std::is_same<half_float::half, T>())
    {
        return 8;
    }
    else
    {
        MIOPEN_LOG_I("not implemented");
        return 1;
    }
}

constexpr std::size_t get_lds_max_number_of_byte() { return 65536; }

static inline auto get_static_ck_common_compiler_flag(const ConvolutionContext& ctx)
{
    auto compiler_flag = std::string(" --std=c++14");

    // atomic-fadd
    compiler_flag += std::string(" -DCK_USE_AMD_BUFFER_ATOMIC_FADD=") +
                     (support_amd_buffer_atomic_fadd(ctx.GetStream().GetDeviceName()) ? '1' : '0');

    // LDS sync
    compiler_flag +=
        std::string(" -DCK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM=") +
        (miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM{})
             ? '0'
             : '1');

    // workaround
    compiler_flag +=
        std::string(" -DCK_WORKAROUND_SWDEV_229564=") + std::to_string(WORKAROUND_SWDEV_229564) +
        std::string(" -DCK_WORKAROUND_SWDEV_231101=") + std::to_string(WORKAROUND_SWDEV_231101);

    // enable or disable buffer load/store
    compiler_flag += std::string(" -DCK_USE_AMD_BUFFER_ADDRESSING=") +
                     (is_use_amd_buffer_load_store(ctx) ? '1' : '0');

    // use v_fmac_f32 or not
    compiler_flag +=
        std::string(" -DCK_USE_AMD_V_FMAC_F32=") + (is_use_v_fmac_f32(ctx) ? '1' : '0');

    return compiler_flag;
}

static inline bool IsComposableKernelSupportedHardware(const ConvolutionContext& c)
{
    return (StartsWith(c.GetStream().GetDeviceName(), "gfx803") &&
            c.GetStream().GetMaxComputeUnits() == 64) ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx900") ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx906") ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx908") ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx90a") ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx103");
}

// greatest common divisor, aka highest common factor
template <typename T>
T gcd(T x, T y)
{
    assert(!(x == 0 && y == 0));

    if(x < 0 || y < 0)
    {
        return gcd(abs(x), abs(y));
    }
    else if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x % y, y); // NOLINT
    }
    else
    {
        return gcd(x, y % x);
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

} // namespace solver
} // namespace miopen

#endif
