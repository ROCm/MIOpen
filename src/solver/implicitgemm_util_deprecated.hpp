#ifndef CK_IMPLICITGEMM_UTIL_DEPRECATED_HPP_
#define CK_IMPLICITGEMM_UTIL_DEPRECATED_HPP_

#include <algorithm>
#include <miopen/env.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/mlo_internal.hpp>
#include "implicitgemm_util.hpp"

namespace miopen {
namespace solver {

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

///\todo remove
template <class PerformanceImplicitGemm_t>
inline static auto GetPerformanceConfigBase(const ConvolutionContext& ctx)
{
    PerformanceImplicitGemm_t pp;
    pp.EuristicInit(ctx);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

///\todo remove
inline static uint32_t GetReadWriteVectorSize(const int v)
{
    return v % 4 == 0 ? 4 : (v % 2 == 0 ? 2 : 1);
}

///\todo remove
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

///\todo remove
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

///\todo remove
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

} // namespace solver
} // namespace miopen

#endif
