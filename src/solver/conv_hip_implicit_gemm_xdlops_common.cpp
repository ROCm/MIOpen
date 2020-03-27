/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#include "miopen/solver.hpp"
#include "miopen/handle.hpp"
#include "miopen/stringutils.hpp"
#include "implicitgemm_util.hpp"

namespace miopen {
namespace solver {

bool PerformanceImplicitGemmXdlops::IsValid(const ConvolutionContext& ctx) const
{
    const auto n  = ConvolutionContextInterpreter::GetBatchN(ctx);
    const auto k  = ConvolutionContextInterpreter::GetOutputChannelK(ctx) / ctx.group_counts;
    const auto c  = ConvolutionContextInterpreter::GetInputChannelC(ctx) / ctx.group_counts;
    const auto ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const auto wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const auto y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const auto x  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);

    std::size_t GemmM, GemmN, GemmK;
    // forward
    if(ctx.direction.IsForward())
    {
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

    const auto& GemmMPerBlock = KPerBlock;
    const auto& GemmNPerBlock = BPerBlock;
    const auto& GemmKPerBlock = EPerBlock;
    const auto& GemmKBlocks   = EBlocks;

    const auto& GemmBBlockCopyClusterLengths_GemmK = InBlockCopyClusterLengths_E;
    const auto& GemmBBlockCopyClusterLengths_GemmN = InBlockCopyClusterLengths_B;
    const auto& GemmABlockCopyClusterLengths_GemmK = WeiBlockCopyClusterLengths_E;
    const auto& GemmABlockCopyClusterLengths_GemmM = WeiBlockCopyClusterLengths_K;

    if(!(GemmKPerBlock % GemmBBlockCopyClusterLengths_GemmK == 0 &&
         GemmKPerBlock % GemmABlockCopyClusterLengths_GemmK == 0 &&
         GemmNPerBlock % GemmBBlockCopyClusterLengths_GemmN == 0 &&
         GemmMPerBlock % GemmABlockCopyClusterLengths_GemmM == 0))
        return false;

    if(!(ctx.direction.IsBackwardWrW()) && GemmKBlocks > 1)
        return false;

    if(!(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 &&
         GemmK % (GemmKPerBlock * GemmKBlocks) == 0))
        return false; // wrong! cannot divice N evenly among thread

    // unsupported xdlops-gemm
    if(GemmMPerWave == 16 && GemmNPerWave == 32)
        return false;
    if(GemmMPerWave == 32 && GemmNPerWave == 16)
        return false;
    if(GemmMPerWave == 8 && GemmNPerWave != 64)
        return false;
    if(GemmMPerWave == 4 && GemmNPerWave != 64)
        return false;

    const auto WaveSize  = 64;
    const auto BlockSize = GemmNPerBlock * GemmMPerBlock / (GemmMPerWave * GemmNPerWave) * WaveSize;

    // fail with blockSize >= 512
    /// \todo fix the issue with blockSize >= 512
    if(BlockSize < 64 || BlockSize > 256)
        return false;

    if(BlockSize != GemmBBlockCopyClusterLengths_GemmK * GemmBBlockCopyClusterLengths_GemmN)
        return false;

    if(BlockSize != GemmABlockCopyClusterLengths_GemmM * GemmABlockCopyClusterLengths_GemmK)
        return false;

    if((GemmMPerBlock % GemmMPerWave) != 0 || (GemmNPerBlock % GemmNPerWave) != 0)
        return false;

    const auto GemmBBlockCopyThreadSliceLengths_GemmN =
        GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;
    const auto GemmABlockCopyThreadSliceLengths_GemmM =
        GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;
    const auto lds_size = ComputeLDSRequiredSize(ctx,
                                                 GemmNPerBlock,
                                                 GemmMPerBlock,
                                                 GemmKPerBlock,
                                                 1,
                                                 1,
                                                 GemmBBlockCopyThreadSliceLengths_GemmN,
                                                 GemmABlockCopyThreadSliceLengths_GemmM,
                                                 true);
    return lds_size <= 64 * 1024;
}

PerformanceImplicitGemmXdlops::PerformanceImplicitGemmXdlops(bool spare)
{
    BPerBlock = spare ? 16 : 64;
    KPerBlock = spare ? 4 : 64;
    EPerBlock = spare ? 4 : 8;
    EBlocks   = 1;

    GemmMPerWave = spare ? 4 : 64;
    GemmNPerWave = spare ? 16 : 64;

    InBlockCopyClusterLengths_E = 4;
    InBlockCopyClusterLengths_B = 4;

    WeiBlockCopyClusterLengths_E = 2;
    WeiBlockCopyClusterLengths_K = 4;

    use_spare_set = spare;
}

PerformanceImplicitGemmXdlops::PerformanceImplicitGemmXdlops(int BPerBlock_,
                                                             int KPerBlock_,
                                                             int EPerBlock_,
                                                             int EBlocks_,
                                                             int GemmMPerWave_,
                                                             int GemmNPerWave_,
                                                             int InBlockCopyClusterLengths_E_,
                                                             int InBlockCopyClusterLengths_B_,
                                                             int WeiBlockCopyClusterLengths_E_,
                                                             int WeiBlockCopyClusterLengths_K_,
                                                             bool use_spare_set_)
    : BPerBlock(BPerBlock_),
      KPerBlock(KPerBlock_),
      EPerBlock(EPerBlock_),
      EBlocks(EBlocks_),
      GemmMPerWave(GemmMPerWave_),
      GemmNPerWave(GemmNPerWave_),
      InBlockCopyClusterLengths_E(InBlockCopyClusterLengths_E_),
      InBlockCopyClusterLengths_B(InBlockCopyClusterLengths_B_),
      WeiBlockCopyClusterLengths_E(WeiBlockCopyClusterLengths_E_),
      WeiBlockCopyClusterLengths_K(WeiBlockCopyClusterLengths_K_),
      use_spare_set(use_spare_set_)
{
}

bool PerformanceImplicitGemmXdlops::operator==(const PerformanceImplicitGemmXdlops& other) const
{
    // clang-format off
    return BPerBlock == other.BPerBlock
        && KPerBlock == other.KPerBlock
        && EPerBlock == other.EPerBlock
        && EBlocks == other.EBlocks
        && GemmMPerWave == other.GemmMPerWave
        && GemmNPerWave == other.GemmNPerWave
        && InBlockCopyClusterLengths_E == other.InBlockCopyClusterLengths_E
        && InBlockCopyClusterLengths_B == other.InBlockCopyClusterLengths_B
        && WeiBlockCopyClusterLengths_E == other.WeiBlockCopyClusterLengths_E
        && WeiBlockCopyClusterLengths_K == other.WeiBlockCopyClusterLengths_K
        && use_spare_set == other.use_spare_set;
    // clang-format on
}

bool PerformanceImplicitGemmXdlops::IsValidValue() const
{
    // clang-format off
    return IsTwoPower<16,128>(BPerBlock)
        && IsTwoPower<4,128>(KPerBlock)
        && IsTwoPower<4,32>(EPerBlock)
        && IsTwoPower<1,64>(EBlocks)
        && IsTwoPower<4,64>(GemmMPerWave)
        && IsTwoPower<16,64>(GemmNPerWave)
        && IsTwoPower<4,16>(InBlockCopyClusterLengths_E)
        && IsTwoPower<4,32>(InBlockCopyClusterLengths_B)
        && IsTwoPower<2,16>(WeiBlockCopyClusterLengths_E)
        && IsTwoPower<4,128>(WeiBlockCopyClusterLengths_K); // clang-format on
}

bool PerformanceImplicitGemmXdlops::SetNextValue()
{
    do
    {
        if(!use_spare_set)
        {
            if(!NextTwoPower<64, 128>(BPerBlock))
                break;
            if(!NextTwoPower<64, 128>(KPerBlock))
                break;
            if(!NextTwoPower<8, 32>(EPerBlock))
                break;
        }
        else
        {
            if(!NextTwoPower<16, 128>(BPerBlock))
                break;
            if(!NextTwoPower<4, 128>(KPerBlock))
                break;
            if(!NextTwoPower<4, 32>(EPerBlock))
                break;
            if(!NextTwoPower<4, 64>(GemmMPerWave))
                break;
            if(!NextTwoPower<16, 64>(GemmNPerWave))
                break;
        }
        if(!NextTwoPower<1, 64>(EBlocks))
            break;
        if(!NextTwoPower<4, 16>(InBlockCopyClusterLengths_E))
            break;
        if(!NextTwoPower<4, 32>(InBlockCopyClusterLengths_B))
            break;
        if(!NextTwoPower<2, 16>(WeiBlockCopyClusterLengths_E))
            break;
        if(!NextTwoPower<4, 128>(WeiBlockCopyClusterLengths_K))
            break;
        return false;
    } while(false);

    return true;
}

void PerformanceImplicitGemmXdlops::EuristicInit(const ConvolutionContext& ctx)
{
    PerformanceImplicitGemmXdlops tmp;
    tmp = {128, 128, 16, 1, 64, 64, 8, 32, 4, 64, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {64, 32, 4, 1, 32, 64, 4, 16, 2, 32, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {64, 32, 4, 1, 32, 64, 4, 16, 4, 16, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {32, 64, 4, 1, 64, 32, 4, 16, 4, 16, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {32, 32, 4, 1, 32, 32, 4, 16, 2, 32, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {64, 16, 4, 1, 16, 64, 4, 16, 4, 16, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {16, 64, 4, 1, 64, 16, 4, 16, 4, 16, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {16, 16, 4, 1, 16, 16, 4, 16, 4, 16, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {64, 4, 16, 1, 4, 64, 16, 4, 16, 4, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {64, 8, 8, 1, 8, 64, 4, 16, 8, 8, use_spare_set};
    if(!tmp.IsValid(ctx))
    {
        MIOPEN_LOG_E("All attempts failed");
        assert(false);
    }
    *this = tmp;
    MIOPEN_LOG_I(ToString());
}

std::string PerformanceImplicitGemmXdlops::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

} // namespace solver
} // namespace miopen
