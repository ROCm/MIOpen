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
#include <miopen/solver.hpp>
#include <miopen/handle.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/implicitgemm_params.hpp>

#define WORKAROUND_ISSUE_659 1

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool PerformanceImplicitGemm::operator==(const PerformanceImplicitGemm& other) const
{
    // clang-format off
    return BPerBlock == other.BPerBlock
        && KPerBlock == other.KPerBlock
        && EPerBlock == other.EPerBlock
        && GemmNRepeat == other.GemmNRepeat
        && GemmMPerThreadSubC == other.GemmMPerThreadSubC
        && GemmNPerThreadSubC == other.GemmNPerThreadSubC
        && GemmMLevel0Cluster == other.GemmMLevel0Cluster
        && GemmNLevel0Cluster == other.GemmNLevel0Cluster
        && GemmMLevel1Cluster == other.GemmMLevel1Cluster
        && GemmNLevel1Cluster == other.GemmNLevel1Cluster
        && InBlockCopyClusterLengths_E == other.InBlockCopyClusterLengths_E
        && InBlockCopyClusterLengths_B == other.InBlockCopyClusterLengths_B
        && InBlockCopyClusterLengths_N1 == other.InBlockCopyClusterLengths_N1
        && InBlockCopyClusterLengths_N2 == other.InBlockCopyClusterLengths_N2
        && WeiBlockCopyClusterLengths_E == other.WeiBlockCopyClusterLengths_E
        && WeiBlockCopyClusterLengths_K == other.WeiBlockCopyClusterLengths_K
        && use_spare_set == other.use_spare_set;
    // clang-format on
}

bool PerformanceImplicitGemm::IsValid(const ExecutionContext& ctx,
                                      const ProblemDescription& problem) const
{
    std::size_t N = KernelBatchN(problem);
    std::size_t K = KernelOutputChannelK(problem);
    std::size_t C = KernelInputChannelC(problem);

    std::size_t Ho = KernelOutputHeightHo(problem);
    std::size_t Wo = KernelOutputWidthWo(problem);

    std::size_t Y = KernelFilterHeightY(problem);
    std::size_t X = KernelFilterWidthX(problem);

    const int N1 = GemmNRepeat;
    const int N2 = GemmNPerThreadSubC;
    if(N % static_cast<std::size_t>(N1 * N2) != 0)
        return false; // wrong! cannot divice N evenly among thread

    const auto N0 = N / static_cast<std::size_t>(N1 * N2);

    const auto B = N0 * Ho * Wo;

    const auto nonVectorizedC = C / GetEPackLength(ctx, problem, false);
    const auto E              = nonVectorizedC * Y * X;

    if(!(EPerBlock % InBlockCopyClusterLengths_E == 0 &&
         EPerBlock % WeiBlockCopyClusterLengths_E == 0 &&
         BPerBlock % InBlockCopyClusterLengths_B == 0 &&
         KPerBlock % WeiBlockCopyClusterLengths_K == 0 && N1 % InBlockCopyClusterLengths_N1 == 0 &&
         N2 % InBlockCopyClusterLengths_N2 == 0))
        return false;

    if(problem.IsDirectionBackwardWrW())
    {
        if(!((X * Y) % (EPerBlock / WeiBlockCopyClusterLengths_E) == 0))
            return false;
    }

    // divide block work by [K, B]
    if(!(K % KPerBlock == 0 && B % BPerBlock == 0 && E % EPerBlock == 0))
        return false; // wrong! cannot divice N evenly among thread

#if WORKAROUND_ISSUE_659
    if(E % static_cast<std::size_t>(2 * EPerBlock) != 0)
        return false;
#endif

    const auto KBlockWork = K / KPerBlock;
    if(KBlockWork % problem.GetGroupCount() != 0)
        return false;

    if((N1 * N2 * BPerBlock) % (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) != 0)
        return false;

    // fp16/bfp16: doesn't support asymmetric matrix mul
    if((problem.IsFp16() || problem.IsBfp16()) && GemmNPerThreadSubC != GemmMPerThreadSubC)
        return false;

    // fp16/bfp16: vector read of length 8 or greater is not supported
    // as vector_type<vector<half,4>, 2> is not working. So, restrict epack*gemmreada <= 4
    // and epack*gemmreadb <= 4
    // if((problem.IsFp16()  || problem.IsBfp16()) && ((GetEPackLength(ctx, problem,
    // false)*GemmNPerThreadSubC > 4)
    // ||
    //   (GetEPackLength(ctx, problem, false)*GemmMPerThreadSubC > 4)))
    //  return false;

    // sanity check
    if((KPerBlock % (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster)) != 0)
        return false;

    if(GemmNRepeat !=
       (N1 * N2 * BPerBlock) / (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster))
        return false;

    const int ThreadPerLevel1Cluster =
        GemmMLevel0Cluster * GemmNLevel0Cluster * GemmMLevel1Cluster * GemmNLevel1Cluster;

    const int block_size = ThreadPerLevel1Cluster;

    if(block_size < 64 || block_size > 512)
        return false;

    if(block_size != InBlockCopyClusterLengths_E * InBlockCopyClusterLengths_N1 *
                         InBlockCopyClusterLengths_B * InBlockCopyClusterLengths_N2)
        return false;

    if(block_size != WeiBlockCopyClusterLengths_K * WeiBlockCopyClusterLengths_E)
        return false;

    const int GemmMRepeat =
        KPerBlock / (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster);

    if(!(GemmMRepeat == 2 && GemmNRepeat == 2))
        return false;

    const int InBlockCopySubLengths_E  = EPerBlock / InBlockCopyClusterLengths_E;
    const int InBlockCopySubLengths_B  = BPerBlock / InBlockCopyClusterLengths_B;
    const int WeiBlockCopySubLengths_K = KPerBlock / WeiBlockCopyClusterLengths_K;

    const std::size_t lds_size = ComputeLDSRequiredSize(problem,
                                                        BPerBlock,
                                                        KPerBlock,
                                                        EPerBlock,
                                                        GemmMPerThreadSubC,
                                                        GemmNPerThreadSubC,
                                                        InBlockCopySubLengths_B,
                                                        WeiBlockCopySubLengths_K,
                                                        GetEPackLength(ctx, problem, false));

    if(lds_size > static_cast<std::size_t>(64) * 1024)
        return false;

    return (InBlockCopySubLengths_E == 1 && InBlockCopySubLengths_B == 1);
}

bool PerformanceImplicitGemmV4R1::IsValid(const ExecutionContext& ctx,
                                          const ProblemDescription& problem) const
{
    std::size_t N = KernelBatchN(problem);
    std::size_t K = KernelOutputChannelK(problem);
    std::size_t C = KernelInputChannelC(problem);

    std::size_t Ho = KernelOutputHeightHo(problem);
    std::size_t Wo = KernelOutputWidthWo(problem);

    std::size_t Y = KernelFilterHeightY(problem);
    std::size_t X = KernelFilterWidthX(problem);

    const int N1 = GemmNRepeat;
    const int N2 = GemmNPerThreadSubC;
    if(N % static_cast<std::size_t>(N1 * N2) != 0)
        return false; // wrong! cannot divice N evenly among thread

    const auto N0 = N / static_cast<std::size_t>(N1 * N2);

    const auto B = N0 * Ho * Wo;

    const auto nonVectorizedC = C / GetEPackLength(ctx, problem, false);
    const auto E              = nonVectorizedC * Y * X;

    if(!(EPerBlock % InBlockCopyClusterLengths_E == 0 &&
         EPerBlock % WeiBlockCopyClusterLengths_E == 0 &&
         BPerBlock % InBlockCopyClusterLengths_B == 0 &&
         KPerBlock % WeiBlockCopyClusterLengths_K == 0 && N1 % InBlockCopyClusterLengths_N1 == 0 &&
         N2 % InBlockCopyClusterLengths_N2 == 0))
        return false;

    // divide block work by [K, B]
    if(!(K % KPerBlock == 0 && B % BPerBlock == 0 && E % EPerBlock == 0))
        return false; // wrong! cannot divice N evenly among thread

    const auto KBlockWork = K / KPerBlock;
    if(KBlockWork % problem.GetGroupCount() != 0)
        return false;

    if((N1 * N2 * BPerBlock) % (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) != 0)
        return false;

    // fp16/bfp16: doesn't support asymmetric matrix mul
    if((problem.IsFp16() || problem.IsBfp16()) && GemmNPerThreadSubC != GemmMPerThreadSubC)
        return false;

    // fp16/bfp16: vector read of length 8 or greater is not supported
    // as vector_type<vector<half,4>, 2> is not working. So, restrict epack*gemmreada <= 4
    // and epack*gemmreadb <= 4
    // if((problem.IsFp16()  || problem.IsBfp16()) && ((GetEPackLength(ctx, problem,
    // false)*GemmNPerThreadSubC > 4)
    // ||
    //   (GetEPackLength(ctx, problem, false)*GemmMPerThreadSubC > 4)))
    //  return false;

    // sanity check
    if((KPerBlock % (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster)) != 0)
        return false;

    if(GemmNRepeat !=
       (N1 * N2 * BPerBlock) / (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster))
        return false;

    const int ThreadPerLevel1Cluster =
        GemmMLevel0Cluster * GemmNLevel0Cluster * GemmMLevel1Cluster * GemmNLevel1Cluster;

    const int block_size = ThreadPerLevel1Cluster;

    if(block_size < 64 || block_size > 512)
        return false;

    if(block_size != InBlockCopyClusterLengths_E * InBlockCopyClusterLengths_N1 *
                         InBlockCopyClusterLengths_B * InBlockCopyClusterLengths_N2)
        return false;

    if(block_size != WeiBlockCopyClusterLengths_K * WeiBlockCopyClusterLengths_E)
        return false;

    const int GemmMRepeat =
        KPerBlock / (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster);

    if(!(GemmMRepeat == 2 && GemmNRepeat == 2))
        return false;

    const int InBlockCopySubLengths_E  = EPerBlock / InBlockCopyClusterLengths_E;
    const int InBlockCopySubLengths_B  = BPerBlock / InBlockCopyClusterLengths_B;
    const int WeiBlockCopySubLengths_K = KPerBlock / WeiBlockCopyClusterLengths_K;

    const std::size_t lds_size = ComputeLDSRequiredSize(problem,
                                                        BPerBlock,
                                                        KPerBlock,
                                                        EPerBlock,
                                                        GemmMPerThreadSubC,
                                                        GemmNPerThreadSubC,
                                                        InBlockCopySubLengths_B,
                                                        WeiBlockCopySubLengths_K,
                                                        GetEPackLength(ctx, problem, false));

    if(lds_size > static_cast<std::size_t>(64) * 1024)
        return false;

    return (InBlockCopySubLengths_E == 1 && InBlockCopySubLengths_B == 1);
}

void PerformanceImplicitGemm::HeuristicInit(const ExecutionContext& ctx,
                                            const ProblemDescription& problem)
{
    // default
    {
        BPerBlock = 16;
        KPerBlock = 128;
        EPerBlock = 8;

        GemmNRepeat = 2;

        GemmMPerThreadSubC = 4;
        GemmNPerThreadSubC = 4;

        GemmMLevel0Cluster = 4;
        GemmNLevel0Cluster = 4;
        GemmMLevel1Cluster = 4;
        GemmNLevel1Cluster = 4;

        InBlockCopyClusterLengths_E  = 8;
        InBlockCopyClusterLengths_N1 = 2;
        InBlockCopyClusterLengths_B  = 16;
        InBlockCopyClusterLengths_N2 = 1;

        WeiBlockCopyClusterLengths_E = 2;
        WeiBlockCopyClusterLengths_K = 128;
    }

    if(!IsValid(ctx, problem))
    {
        BPerBlock = 8;
        KPerBlock = 128;
        EPerBlock = 8;

        GemmMLevel0Cluster = 4;
        GemmNLevel0Cluster = 4;
        GemmMLevel1Cluster = 4;
        GemmNLevel1Cluster = 2;

        InBlockCopyClusterLengths_E  = 8;
        InBlockCopyClusterLengths_N1 = 1;
        InBlockCopyClusterLengths_B  = 8;
        InBlockCopyClusterLengths_N2 = 2;

        WeiBlockCopyClusterLengths_E = 2;
        WeiBlockCopyClusterLengths_K = 64;
    }

    if(!IsValid(ctx, problem))
    {
        BPerBlock = 8;
        KPerBlock = 64;
        EPerBlock = 8;

        GemmMLevel0Cluster = 4;
        GemmNLevel0Cluster = 2;
        GemmMLevel1Cluster = 2;
        GemmNLevel1Cluster = 4;

        InBlockCopyClusterLengths_E  = 8;
        InBlockCopyClusterLengths_N1 = 1;
        InBlockCopyClusterLengths_B  = 8;
        InBlockCopyClusterLengths_N2 = 1;

        WeiBlockCopyClusterLengths_E = 4;
        WeiBlockCopyClusterLengths_K = 16;
    }

    if(!IsValid(ctx, problem))
    {
        BPerBlock = 16;
        KPerBlock = 32;
        EPerBlock = 4;

        GemmMLevel0Cluster = 1;
        GemmNLevel0Cluster = 4;
        GemmMLevel1Cluster = 4;
        GemmNLevel1Cluster = 4;

        InBlockCopyClusterLengths_E  = 4;
        InBlockCopyClusterLengths_N1 = 1;
        InBlockCopyClusterLengths_B  = 16;
        InBlockCopyClusterLengths_N2 = 1;
    }

    if(!IsValid(ctx, problem))
    {
        BPerBlock = 16;
        KPerBlock = 16;
        EPerBlock = 4;

        GemmMPerThreadSubC = 2;
        GemmNPerThreadSubC = 2;

        GemmMLevel0Cluster = 2;
        GemmNLevel0Cluster = 4;
        GemmMLevel1Cluster = 2;
        GemmNLevel1Cluster = 4;
    }

    if(!IsValid(ctx, problem))
    {
        BPerBlock = 8;
        KPerBlock = 32;
        EPerBlock = 4;

        GemmNRepeat = 2;

        GemmMPerThreadSubC = 2;
        GemmNPerThreadSubC = 2;

        GemmMLevel0Cluster = 2;
        GemmNLevel0Cluster = 4;
        GemmMLevel1Cluster = 4;
        GemmNLevel1Cluster = 2;

        InBlockCopyClusterLengths_E  = 4;
        InBlockCopyClusterLengths_N1 = 2;
        InBlockCopyClusterLengths_B  = 8;
        InBlockCopyClusterLengths_N2 = 1;

        WeiBlockCopyClusterLengths_E = 4;
        WeiBlockCopyClusterLengths_K = 16;
    }

    if(!IsValid(ctx, problem))
    {
        MIOPEN_LOG_E("All attempts failed");
        assert(false);
    }
    MIOPEN_LOG_I(ToString());
}

bool PerformanceImplicitGemm::IsValidValue() const
{
    // clang-format off
    return IsTwoPower<8,16>(BPerBlock)
        && IsTwoPower<16,128>(KPerBlock)
        && IsTwoPower<4,16>(EPerBlock)
        && GemmNRepeat == 2
        && IsTwoPower<2,4>(GemmMPerThreadSubC)
        && IsTwoPower<2,4>(GemmNPerThreadSubC)
        && IsTwoPower<1,4>(GemmMLevel0Cluster)
        && IsTwoPower<1,4>(GemmNLevel0Cluster)
        && IsTwoPower<1,4>(GemmMLevel1Cluster)
        && IsTwoPower<1,4>(GemmNLevel1Cluster)
        && IsTwoPower<4,16>(InBlockCopyClusterLengths_E)
        && IsTwoPower<8,16>(InBlockCopyClusterLengths_B)
        && IsTwoPower<1,2>(InBlockCopyClusterLengths_N1)
        && IsTwoPower<1,4>(InBlockCopyClusterLengths_N2)
        && IsTwoPower<1,4>(WeiBlockCopyClusterLengths_E)
        && IsTwoPower<16,128>(WeiBlockCopyClusterLengths_K); // clang-format on
}

bool PerformanceImplicitGemm::SetNextValue(const ProblemDescription&)
{
    // GemmNRepeat = 2 cosntant
    do
    {
        // BPerBlock == 16 constant for no-spare
        // GemmNLevel0Cluster = 4 constant for no-spare
        // GemmNLevel1Cluster = 4 constant for no-spare
        // InBlockCopyClusterLengths_B = 16 constant for no-spare
        if(!use_spare_set)
        {
            if(!NextTwoPower<2, 4>(GemmMLevel0Cluster))
                break;
            if(!NextTwoPower<2, 4>(GemmMLevel1Cluster))
                break;
        }
        else
        {
            if(!NextTwoPower<8, 16>(BPerBlock))
                break;
            if(!NextTwoPower<1, 4>(GemmNLevel0Cluster))
                break;
            if(!NextTwoPower<1, 4>(GemmNLevel1Cluster))
                break;
            if(!NextTwoPower<8, 16>(InBlockCopyClusterLengths_B))
                break;
            if(!NextTwoPower<1, 4>(GemmMLevel0Cluster))
                break;
            if(!NextTwoPower<1, 4>(GemmMLevel1Cluster))
                break;
        }

        if(!NextTwoPower<1, 4>(WeiBlockCopyClusterLengths_E))
            break;
        if(!NextTwoPower<16, 128>(WeiBlockCopyClusterLengths_K))
            break;
        if(!NextTwoPower<2, 4>(GemmMPerThreadSubC))
            break;
        if(!NextTwoPower<2, 4>(GemmNPerThreadSubC))
            break;
        if(!NextTwoPower<16, 128>(KPerBlock))
            break;
        if(!NextTwoPower<4, 16>(EPerBlock))
            break;
        if(!NextTwoPower<4, 16>(InBlockCopyClusterLengths_E))
            break;
        if(!NextTwoPower<1, 2>(InBlockCopyClusterLengths_N1))
            break;
        if(!NextTwoPower<1, 2>(InBlockCopyClusterLengths_N2))
            break;
        return false;
    } while(false);

    return true;
}

PerformanceImplicitGemm::PerformanceImplicitGemm(bool spare)
{
    BPerBlock = spare ? 8 : 16; // constant for no-spare
    KPerBlock = 16;
    EPerBlock = 4;

    GemmNRepeat = 2; // constant for all

    GemmMPerThreadSubC = 2;
    GemmNPerThreadSubC = 2;

    GemmMLevel0Cluster = spare ? 1 : 2;
    GemmNLevel0Cluster = spare ? 1 : 4; // constant for no-spare
    GemmMLevel1Cluster = spare ? 1 : 2;
    GemmNLevel1Cluster = spare ? 1 : 4; // constant for no-spare

    InBlockCopyClusterLengths_E  = 4;
    InBlockCopyClusterLengths_N1 = 1;
    InBlockCopyClusterLengths_B  = spare ? 8 : 16; // constant for no-spare
    InBlockCopyClusterLengths_N2 = 1;

    WeiBlockCopyClusterLengths_E = 1;
    WeiBlockCopyClusterLengths_K = 16;
    use_spare_set                = spare;
}

PerformanceImplicitGemm::PerformanceImplicitGemm(int BPerBlock_,
                                                 int KPerBlock_,
                                                 int EPerBlock_,
                                                 int GemmNRepeat_,
                                                 int GemmMPerThreadSubC_,
                                                 int GemmNPerThreadSubC_,
                                                 int GemmMLevel0Cluster_,
                                                 int GemmNLevel0Cluster_,
                                                 int GemmMLevel1Cluster_,
                                                 int GemmNLevel1Cluster_,
                                                 int InBlockCopyClusterLengths_E_,
                                                 int InBlockCopyClusterLengths_B_,
                                                 int InBlockCopyClusterLengths_N1_,
                                                 int InBlockCopyClusterLengths_N2_,
                                                 int WeiBlockCopyClusterLengths_E_,
                                                 int WeiBlockCopyClusterLengths_K_,
                                                 bool use_spare_set_)
    : BPerBlock(BPerBlock_),
      KPerBlock(KPerBlock_),
      EPerBlock(EPerBlock_),
      GemmNRepeat(GemmNRepeat_),
      GemmMPerThreadSubC(GemmMPerThreadSubC_),
      GemmNPerThreadSubC(GemmNPerThreadSubC_),
      GemmMLevel0Cluster(GemmMLevel0Cluster_),
      GemmNLevel0Cluster(GemmNLevel0Cluster_),
      GemmMLevel1Cluster(GemmMLevel1Cluster_),
      GemmNLevel1Cluster(GemmNLevel1Cluster_),
      InBlockCopyClusterLengths_E(InBlockCopyClusterLengths_E_),
      InBlockCopyClusterLengths_B(InBlockCopyClusterLengths_B_),
      InBlockCopyClusterLengths_N1(InBlockCopyClusterLengths_N1_),
      InBlockCopyClusterLengths_N2(InBlockCopyClusterLengths_N2_),
      WeiBlockCopyClusterLengths_E(WeiBlockCopyClusterLengths_E_),
      WeiBlockCopyClusterLengths_K(WeiBlockCopyClusterLengths_K_),
      use_spare_set(use_spare_set_)
{
}

} // namespace conv
} // namespace solver
} // namespace miopen
