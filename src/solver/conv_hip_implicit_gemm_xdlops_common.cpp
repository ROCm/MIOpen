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
    std::size_t N = KernelBatchN(ctx);
    std::size_t K = KernelOutputChannelK(ctx);
    std::size_t C = KernelInputChannelC(ctx);

    std::size_t Ho = KernelOutputHeightHo(ctx);
    std::size_t Wo = KernelOutputWidthWo(ctx);

    std::size_t Y = KernelFilterHeightY(ctx);
    std::size_t X = KernelFilterWidthX(ctx);

    const std::size_t B = N * Ho * Wo;

    const auto nonVectorizedC = C / GetEPackLength(ctx, true);
    const auto E              = static_cast<int>(nonVectorizedC) * Y * X;

    const auto KBlockWork = K / KPerBlock;
    if(KBlockWork % ctx.group_counts != 0)
        return false;

    if(!(EPerBlock % InBlockCopyClusterLengths_E == 0 &&
         EPerBlock % WeiBlockCopyClusterLengths_E == 0 &&
         BPerBlock % InBlockCopyClusterLengths_B == 0 &&
         KPerBlock % WeiBlockCopyClusterLengths_K == 0))
        return false;

    // divide block work by [K, B]
    if(!(K % KPerBlock == 0 && B % BPerBlock == 0 && E % EPerBlock == 0))
        return false; // wrong! cannot divice N evenly among thread

    if(ctx.direction.IsBackwardWrW())
    {
        if(!((X * Y) % (EPerBlock / WeiBlockCopyClusterLengths_E) == 0))
            return false;
    }

    // unsupported xdlops-gemm
    if(GemmMPerWave == 16 && GemmNPerWave == 32)
        return false;
    if(GemmMPerWave == 32 && GemmNPerWave == 16)
        return false;
    if(GemmMPerWave == 8 && GemmNPerWave != 64)
        return false;
    if(GemmMPerWave == 4 && GemmNPerWave != 64)
        return false;

    const int WaveSize  = 64;
    const int BlockSize = BPerBlock * KPerBlock / (GemmMPerWave * GemmNPerWave) * WaveSize;

    // fail with blockSize >= 512
    /// \todo fix the issue with blockSize >= 512
    if(BlockSize < 64 || BlockSize > 256)
        return false;

    if(BlockSize != InBlockCopyClusterLengths_E * InBlockCopyClusterLengths_B)
        return false;

    if(BlockSize != WeiBlockCopyClusterLengths_K * WeiBlockCopyClusterLengths_E)
        return false;

    if((KPerBlock % GemmMPerWave) != 0 || (BPerBlock % GemmNPerWave) != 0)
        return false;

    const int InBlockCopySubLengths_B  = BPerBlock / InBlockCopyClusterLengths_B;
    const int WeiBlockCopySubLengths_K = KPerBlock / WeiBlockCopyClusterLengths_K;
    const std::size_t lds_size         = ComputeLDSRequiredSize(ctx,
                                                        BPerBlock,
                                                        KPerBlock,
                                                        EPerBlock,
                                                        1,
                                                        1,
                                                        InBlockCopySubLengths_B,
                                                        WeiBlockCopySubLengths_K,
                                                        true);
    if(lds_size > 64 * 1024)
        return false;

    const int GemmMWaves = KPerBlock / GemmMPerWave;
    const int GemmNWaves = BPerBlock / GemmNPerWave;

    return (GemmMPerWave * GemmMWaves == KPerBlock && GemmNPerWave * GemmNWaves == BPerBlock);
}

PerformanceImplicitGemmXdlops::PerformanceImplicitGemmXdlops(bool spare)
{
    BPerBlock = spare ? 16 : 64;
    KPerBlock = spare ? 4 : 64;
    EPerBlock = spare ? 4 : 8;

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
    tmp = {128, 128, 16, 64, 64, 8, 32, 4, 64, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {64, 32, 4, 32, 64, 4, 16, 2, 32, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {64, 32, 4, 32, 64, 4, 16, 4, 16, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {32, 64, 4, 64, 32, 4, 16, 4, 16, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {32, 32, 4, 32, 32, 4, 16, 2, 32, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {64, 16, 4, 16, 64, 4, 16, 4, 16, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {16, 64, 4, 64, 16, 4, 16, 4, 16, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {16, 16, 4, 16, 16, 4, 16, 4, 16, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {64, 4, 16, 4, 64, 16, 4, 16, 4, use_spare_set};
    if(!tmp.IsValid(ctx))
        tmp = {64, 8, 8, 8, 64, 4, 16, 8, 8, use_spare_set};
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
