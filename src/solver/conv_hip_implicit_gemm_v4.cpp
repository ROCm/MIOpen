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
#include <miopen/generic_search.hpp>
#include "miopen/stringutils.hpp"
#include "implicitgemm_util.hpp"
#include "miopen/implicitgemm_params.hpp"
#include "miopen/hip_build_utils.hpp"
#include "implicitgemm_util.hpp"

#define WORKAROUND_ISSUE_2174_2222_2224_2243 1

namespace miopen {
namespace solver {

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

bool PerformanceImplicitGemm::IsValid(const ConvolutionContext& ctx) const
{
    int N = ctx.batch_sz;
    int K = ctx.n_outputs;
    int C = ctx.n_inputs;

    int Ho = ImgHeight(ctx);
    int Wo = ImgWidth(ctx);

    int Y = ctx.kernel_size_h;
    int X = ctx.kernel_size_w;

    if(ctx.direction.IsBackwardWrW())
    {
        N  = ctx.n_outputs; // swapped
        K  = ctx.n_inputs;  // swapped
        C  = ctx.batch_sz;  // swapped
        Ho = ctx.kernel_size_h;
        Wo = ctx.kernel_size_w;
        Y  = ctx.in_height; // swapped
        X  = ctx.in_width;  // swapped
    }

    const int N1 = GemmNRepeat;
    const int N2 = GemmNPerThreadSubC;
    if(N % (N1 * N2) != 0)
        return false; // wrong! cannot divice N evenly among thread

    const int N0 = N / (N1 * N2);

    const int B = N0 * Ho * Wo;

    const auto nonVectorizedC = C / GetEPackLength(ctx, false);
    const auto E              = nonVectorizedC * Y * X;

    if(!(EPerBlock % InBlockCopyClusterLengths_E == 0 &&
         EPerBlock % WeiBlockCopyClusterLengths_E == 0 &&
         BPerBlock % InBlockCopyClusterLengths_B == 0 &&
         KPerBlock % WeiBlockCopyClusterLengths_K == 0 && N1 % InBlockCopyClusterLengths_N1 == 0 &&
         N2 % InBlockCopyClusterLengths_N2 == 0))
        return false;

    if(ctx.direction.IsBackwardWrW())
    {
        if(!((X * Y) % (EPerBlock / WeiBlockCopyClusterLengths_E) == 0))
            return false;
    }
    // divide block work by [K, B]
    if(!(K % KPerBlock == 0 && B % BPerBlock == 0 && E % (2 * EPerBlock) == 0))
        return false; // wrong! cannot divice N evenly among thread

    if((N1 * N2 * BPerBlock) % (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) != 0)
        return false;

    if(ctx.IsFp16() && GemmNPerThreadSubC != GemmMPerThreadSubC)
        return false;

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

    if(block_size !=
       InBlockCopyClusterLengths_E * InBlockCopyClusterLengths_N1 * InBlockCopyClusterLengths_B *
           InBlockCopyClusterLengths_N2)
        return false;

    if(block_size != WeiBlockCopyClusterLengths_K * WeiBlockCopyClusterLengths_E)
        return false;

    const int GemmMRepeat =
        KPerBlock / (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster);

    if(!(GemmMRepeat == 2 && GemmNRepeat == 2))
        return false;

    const int InBlockCopySubLengths_E = EPerBlock / InBlockCopyClusterLengths_E;
    const int InBlockCopySubLengths_B = BPerBlock / InBlockCopyClusterLengths_B;

    return (InBlockCopySubLengths_E == 1 && InBlockCopySubLengths_B == 1);
}

void PerformanceImplicitGemm::EuristicInit(const ConvolutionContext& config)
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

    if(!IsValid(config))
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

    if(!IsValid(config))
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

    if(!IsValid(config))
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

    if(!IsValid(config))
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

    if(!IsValid(config))
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

bool PerformanceImplicitGemm::SetNextValue()
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

std::string PerformanceImplicitGemm::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
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

bool ConvHipImplicitGemmV4_1x1::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!ctx.Is2d())
        return false;

    return ctx.IsFp32() && ctx.pad_h == 0 && ctx.pad_w == 0 && ctx.group_counts == 1 &&
           ctx.batch_sz % 8 == 0 && (ctx.batch_sz * ImgHeight(ctx) * ImgWidth(ctx)) % 64 == 0 &&
           ctx.n_outputs % 16 == 0 && ctx.kernel_size_h == 1 && ctx.kernel_size_w == 1 &&
           ctx.n_inputs % 8 == 0 && ctx.kernel_dilation_h == 1 && ctx.kernel_dilation_w == 1;
}

bool ConvHipImplicitGemmV4Fwd::IsApplicable(const ConvolutionContext& ctx) const
{
#if WORKAROUND_ISSUE_2174_2222_2224_2243
    if(miopen::HipGetHccVersion() >= external_tool_version_t{2, 6, 0})
    {
        if(!(ctx.kernel_dilation_h == 1 && ctx.kernel_dilation_w == 1))
            return false;
    }
#endif

    bool isTypeSupported = ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16();

    // channels is divided by epack to pack 2/4 fp16/bfp16
    if(ctx.n_inputs % GetEPackLength(ctx, false) != 0)
        return false;

    // For fp16, when E=c*x*y % 32 == 0, 4 channels are accumulated through dot4 (2 * dot2)
    // operation
    // For bfp16/fp16, when E=c*x*y % 16 == 0, 2 channels are accumulated through dot2 operation
    // For fp32, when E=c*x*y % 8 == 0, no dot2 operation exist.
    bool isEInMultiple = (ctx.IsFp16() || ctx.IsBfp16())
                             ? ((ctx.n_inputs * ctx.kernel_size_h * ctx.kernel_size_w) % 16 == 0)
                             : ((ctx.n_inputs * ctx.kernel_size_h * ctx.kernel_size_w) % 8 == 0);

    // padding support required for out_of_bound configs
    bool no_out_of_bound = (ctx.in_width >= ((ctx.kernel_size_w - 1) * ctx.kernel_dilation_w + 1) +
                                                (ctx.out_width - 1) * ctx.kernel_stride_w) &&
                           (ctx.in_height >= ((ctx.kernel_size_h - 1) * ctx.kernel_dilation_h + 1) +
                                                 (ctx.out_height - 1) * ctx.kernel_stride_h);

    return ctx.Is2d() && isTypeSupported && ctx.direction.IsForward() && ctx.pad_h == 0 &&
           ctx.pad_w == 0 && ctx.group_counts == 1 && ctx.batch_sz % 8 == 0 &&
           (ctx.batch_sz * ctx.out_height * ctx.out_width) % 64 == 0 && isEInMultiple &&
           ctx.n_outputs % 16 == 0 && no_out_of_bound;
}

bool ConvHipImplicitGemmV4WrW::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!ctx.direction.IsBackwardWrW())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;

#if WORKAROUND_ISSUE_2174_2222_2224_2243
    if(miopen::HipGetHccVersion() >= external_tool_version_t{2, 6, 0})
    {
        if(!(ctx.kernel_stride_w == 1 && ctx.kernel_stride_h == 1))
            return false;
    }
#endif

    // batch is divided by epack to pack 2/4 fp16/bfp16
    if(ctx.batch_sz % GetEPackLength(ctx, false) != 0)
        return false;

    bool isEInMultiple = (ctx.IsFp16() || ctx.IsBfp16())
                             ? ((ctx.batch_sz * ctx.in_height * ctx.in_width) % 16 == 0)
                             : ((ctx.batch_sz * ctx.in_height * ctx.in_width) % 8 == 0);

    return ctx.pad_h == 0 && ctx.pad_w == 0 && ctx.group_counts == 1 && ctx.n_outputs % 8 == 0 &&
           (ctx.n_outputs * ctx.kernel_size_h * ctx.kernel_size_w) % 64 == 0 && isEInMultiple &&
           ctx.n_inputs % 16 == 0;
}

PerformanceImplicitGemm
ConvHipImplicitGemmV4Fwd::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemm>(ctx);
}

PerformanceImplicitGemm
ConvHipImplicitGemmV4WrW::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemm>(ctx);
}

PerformanceImplicitGemm
ConvHipImplicitGemmV4_1x1::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemm>(ctx);
}

bool ConvHipImplicitGemmV4Fwd::IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                                        const PerformanceImplicitGemm& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

bool ConvHipImplicitGemmV4WrW::IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                                        const PerformanceImplicitGemm& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

bool ConvHipImplicitGemmV4_1x1::IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                                         const PerformanceImplicitGemm& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

using ImplicitGemmKernel_t = enum {
    ImplicitGemmV4     = 0,
    ImplicitGemmV4_1x1 = 1,
};

static inline ConvSolution GetSolutionBase(const ConvolutionContext& ctx,
                                           const PerformanceImplicitGemm& config,
                                           const ImplicitGemmKernel_t kernel,
                                           const int n,
                                           const int k,
                                           const int ho,
                                           const int wo)
{
    ConvSolution result;
    KernelInfo construction_parameters;

    const int N1 = config.GemmNRepeat;
    const int N2 = config.GemmNPerThreadSubC;

    std::size_t b = (static_cast<std::size_t>(n) * ho * wo) / (static_cast<std::size_t>(N1) * N2);

    std::size_t BPerBlock = config.BPerBlock;
    std::size_t KPerBlock = config.KPerBlock;
    std::size_t EPerBlock = config.EPerBlock;

    const int ThreadPerLevel1Cluster = config.GemmMLevel0Cluster * config.GemmNLevel0Cluster *
                                       config.GemmMLevel1Cluster * config.GemmNLevel1Cluster;

    const int block_size = ThreadPerLevel1Cluster;

    std::size_t grid_size = (b / BPerBlock) * (k / KPerBlock);

    std::size_t lkl_wk0 = block_size;
    std::size_t lkl_wk1 = 1;
    std::size_t lkl_wk2 = 1;

    construction_parameters.l_wk.push_back(lkl_wk0);
    construction_parameters.l_wk.push_back(lkl_wk1);
    construction_parameters.l_wk.push_back(lkl_wk2);

    std::size_t gbl_wk0 = lkl_wk0 * grid_size;
    std::size_t gbl_wk1 = 1;
    std::size_t gbl_wk2 = 1;

    construction_parameters.g_wk.push_back(gbl_wk0);
    construction_parameters.g_wk.push_back(gbl_wk1);
    construction_parameters.g_wk.push_back(gbl_wk2);

    if(kernel == ImplicitGemmV4)
    {
        construction_parameters.kernel_file =
            "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer.cpp";

        construction_parameters.kernel_name =
            "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer";
    }

    if(kernel == ImplicitGemmV4_1x1)
    {
        construction_parameters.kernel_file =
            "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer.cpp";

        construction_parameters.kernel_name =
            "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer";
    }

    std::size_t WeiBlockCopySubLengths_E = EPerBlock / config.WeiBlockCopyClusterLengths_E;
    std::size_t WeiBlockCopySubLengths_K = KPerBlock / config.WeiBlockCopyClusterLengths_K;

    int WeiBlockCopySrcDataPerRead_E  = GetReadWriteVectorSize(WeiBlockCopySubLengths_E);
    int WeiBlockCopyDstDataPerWrite_K = GetReadWriteVectorSize(WeiBlockCopySubLengths_K);

    std::size_t InBlockCopySubLengths_B  = BPerBlock / config.InBlockCopyClusterLengths_B;
    std::size_t InBlockCopySubLengths_N2 = N2 / config.InBlockCopyClusterLengths_N2;

    int InBlockCopySrcDataPerRead_B   = GetReadWriteVectorSize(InBlockCopySubLengths_B);
    int InBlockCopyDstDataPerWrite_N2 = GetReadWriteVectorSize(InBlockCopySubLengths_N2);

    InBlockCopySrcDataPerRead_B =
        ctx.kernel_size_w > 1
            ? std::min(InBlockCopySrcDataPerRead_B, GetReadWriteVectorSize(ctx.kernel_dilation_w))
            : InBlockCopySrcDataPerRead_B;

    InBlockCopySrcDataPerRead_B = ctx.kernel_stride_w > 1 ? 1 : InBlockCopySrcDataPerRead_B;

    WeiBlockCopySrcDataPerRead_E =
        ctx.direction.IsBackwardData() ? 1 : WeiBlockCopySrcDataPerRead_E;

    // TBD: Due to underlying bug, we need to restrict reading/writing only 1 fp16 value at a time
    if(ctx.IsFp16() || ctx.IsBfp16())
    {
        WeiBlockCopySrcDataPerRead_E  = 1;
        WeiBlockCopyDstDataPerWrite_K = 1;
        InBlockCopyDstDataPerWrite_N2 = 1;
        InBlockCopySrcDataPerRead_B   = 1;
    }

    if(ctx.direction.IsBackwardWrW())
    {
        // clang-format off
        construction_parameters.comp_options =
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ctx.n_inputs) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ctx.n_outputs) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ctx.out_height) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ctx.out_width) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ctx.in_height) +  // swapped
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ctx.in_width);
        // clang-format on
    }
    else
    {
        // clang-format off
        construction_parameters.comp_options =
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ctx.n_outputs) +
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ctx.n_inputs) +
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ctx.in_height) +
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ctx.in_width) +
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ctx.out_height) +
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ctx.out_width);
        // clang-format on
    }

    // clang-format off
    construction_parameters.comp_options +=
        std::string(" -std=c++14") +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=") + std::to_string(static_cast<std::size_t>(ctx.direction.IsForward())) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=") + std::to_string(static_cast<std::size_t>(ctx.direction.IsBackwardData())) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=") + std::to_string(static_cast<std::size_t>(ctx.direction.IsBackwardWrW())) +
        std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(ctx.batch_sz) +
        std::string(" -DCK_PARAM_PROBLEM_Y=") + std::to_string(ctx.kernel_size_h) +
        std::string(" -DCK_PARAM_PROBLEM_X=") + std::to_string(ctx.kernel_size_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(ctx.kernel_stride_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(ctx.kernel_stride_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_H=") + std::to_string(ctx.kernel_dilation_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_W=") + std::to_string(ctx.kernel_dilation_w) +
        std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
        std::string(" -DCK_PARAM_TUNABLE_B_PER_BLOCK=") + std::to_string(BPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_K_PER_BLOCK=") + std::to_string(KPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_E_PER_BLOCK=") + std::to_string(EPerBlock) +
        std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
        std::string(" -DCK_PARAM_GEMM_N_REPEAT=") + std::to_string(config.GemmNRepeat) +
        std::string(" -DCK_PARAM_GEMM_M_PER_THREAD_SUB_C=") + std::to_string(config.GemmMPerThreadSubC) +
        std::string(" -DCK_PARAM_GEMM_N_PER_THREAD_SUB_C=") + std::to_string(config.GemmNPerThreadSubC) +
        std::string(" -DCK_PARAM_GEMM_M_LEVEL0_CLUSTER=") + std::to_string(config.GemmMLevel0Cluster) +
        std::string(" -DCK_PARAM_GEMM_N_LEVEL0_CLUSTER=") + std::to_string(config.GemmNLevel0Cluster) +
        std::string(" -DCK_PARAM_GEMM_M_LEVEL1_CLUSTER=") + std::to_string(config.GemmMLevel1Cluster) +
        std::string(" -DCK_PARAM_GEMM_N_LEVEL1_CLUSTER=") + std::to_string(config.GemmNLevel1Cluster) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_E=") + std::to_string(config.InBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_N1=") + std::to_string(config.InBlockCopyClusterLengths_N1) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_B=") + std::to_string(config.InBlockCopyClusterLengths_B) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_N2=") + std::to_string(config.InBlockCopyClusterLengths_N2) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_SRC_DATA_PER_READ_B=") + std::to_string(InBlockCopySrcDataPerRead_B) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_DST_DATA_PER_WRITE_N2=") + std::to_string(InBlockCopyDstDataPerWrite_N2) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E=") + std::to_string(config.WeiBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K=") + std::to_string(config.WeiBlockCopyClusterLengths_K) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E=") + std::to_string(WeiBlockCopySrcDataPerRead_E) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_K=") + std::to_string(WeiBlockCopyDstDataPerWrite_K) +
        std::string(" -DCK_PARAM_EPACK_LENGTH=") + std::to_string(GetEPackLength(ctx, false)) +
        std::string(" -DCK_THREADWISE_GEMM_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx) ? '1' : '0') +
        std::string(" -D__HIP_PLATFORM_HCC__=1") +
        ctx.general_compile_options;
    // clang-format on

    result.construction_params.push_back(construction_parameters);
    return result;
}

ConvSolution ConvHipImplicitGemmV4Fwd::GetSolution(const ConvolutionContext& ctx,
                                                   const PerformanceImplicitGemm& config,
                                                   const bool) const
{
    return GetSolutionBase(
        ctx, config, ImplicitGemmV4, ctx.batch_sz, ctx.n_outputs, ctx.out_height, ctx.out_width);
}
ConvSolution ConvHipImplicitGemmV4WrW::GetSolution(const ConvolutionContext& ctx,
                                                   const PerformanceImplicitGemm& config,
                                                   const bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmV4,
                           ctx.n_outputs,
                           ctx.n_inputs,
                           ctx.kernel_size_h,
                           ctx.kernel_size_w);
}

ConvSolution ConvHipImplicitGemmV4_1x1::GetSolution(const ConvolutionContext& ctx,
                                                    const PerformanceImplicitGemm& config,
                                                    const bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmV4_1x1,
                           ctx.batch_sz,
                           ctx.n_outputs,
                           ImgHeight(ctx),
                           ImgWidth(ctx));
}

int ConvHipImplicitGemmV4Fwd::RunAndMeasureSolution(miopen::Handle& profile_h,
                                                    ConstData_t bot_buf,
                                                    Data_t top_buf,
                                                    ConstData_t wei_buf,
                                                    ConstData_t bias_buf,
                                                    const ConvolutionContext& ctx,
                                                    const ConvSolution& solution,
                                                    float& elapsed_time) const
{
    assert(bias_buf == nullptr);
    (void)bias_buf;

    return RunAndMeasureSolutionBase(
        profile_h, bot_buf, top_buf, wei_buf, ctx, solution, elapsed_time);
}

int ConvHipImplicitGemmV4WrW::RunAndMeasureSolution(miopen::Handle& profile_h,
                                                    ConstData_t bot_buf,
                                                    Data_t top_buf,
                                                    ConstData_t wei_buf,
                                                    ConstData_t bias_buf,
                                                    const ConvolutionContext& ctx,
                                                    const ConvSolution& solution,
                                                    float& elapsed_time) const
{
    assert(bias_buf == nullptr);
    (void)bias_buf;
    return RunAndMeasureSolutionBase(
        profile_h, bot_buf, top_buf, wei_buf, ctx, solution, elapsed_time);
}

int ConvHipImplicitGemmV4_1x1::RunAndMeasureSolution(miopen::Handle& profile_h,
                                                     ConstData_t bot_buf,
                                                     Data_t top_buf,
                                                     ConstData_t wei_buf,
                                                     ConstData_t bias_buf,
                                                     const ConvolutionContext& ctx,
                                                     const ConvSolution& solution,
                                                     float& elapsed_time) const
{
    assert(bias_buf == nullptr);
    (void)bias_buf;
    return RunAndMeasureSolutionBase(
        profile_h, bot_buf, top_buf, wei_buf, ctx, solution, elapsed_time);
}

PerformanceImplicitGemm ConvHipImplicitGemmV4Fwd::Search(const ConvolutionContext& context) const
{
    return GenericSearchFwd(*this, context);
}
PerformanceImplicitGemm ConvHipImplicitGemmV4WrW::Search(const ConvolutionContext& context) const
{
    return GenericSearchFwd(*this, context);
}

PerformanceImplicitGemm ConvHipImplicitGemmV4_1x1::Search(const ConvolutionContext& context) const
{
    return GenericSearchFwd(*this, context);
}

} // namespace solver
} // namespace miopen
