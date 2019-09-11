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

namespace miopen {
namespace solver {

bool ConvHipImplicitGemmV4Fwd::IsApplicable(const ConvolutionContext& ctx) const
{
    // disable IsFp16 due to NaN (issue #2071);
    ///\todo: 1) Fixed NaN issue in Fp16, 2) enable Fp16 and 3) add Fp16 tests
    bool isTypeSupported = ctx.IsFp32();

    // For fp16, when c*x*y % 64 == 0, 4 channels are accumulated through dot4 (2 * dot2) operation
    // when c*x*y % 32 == 0,  channels are accumulated through dot2 operation.
    bool isNumInputsInMultiple =
        ctx.IsFp16() ? ((ctx.n_inputs * ctx.kernel_size_h * ctx.kernel_size_w) % 32 == 0)
                     : ((ctx.n_inputs * ctx.kernel_size_h * ctx.kernel_size_w) % 8 == 0);

    // padding support required for out_of_bound configs
    bool no_out_of_bound = (ctx.in_width >= ((ctx.kernel_size_w - 1) * ctx.kernel_dilation_w + 1) +
                                                (ctx.out_width - 1) * ctx.kernel_stride_w) &&
                           (ctx.in_height >= ((ctx.kernel_size_h - 1) * ctx.kernel_dilation_h + 1) +
                                                 (ctx.out_height - 1) * ctx.kernel_stride_h);

    return ctx.Is2d() && isTypeSupported && ctx.direction.IsForward() && ctx.pad_h == 0 &&
           ctx.pad_w == 0 && ctx.group_counts == 1 && ctx.batch_sz % 8 == 0 &&
           (ctx.batch_sz * ctx.out_height * ctx.out_width) % 64 == 0 && isNumInputsInMultiple &&
           ctx.n_outputs % 16 == 0 && no_out_of_bound;
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

inline static int GetReadWriteVectorSize(const int v)
{
    return (v & 3) == 0 ? 4 : ((v & 1) == 0 ? 2 : 1);
}

inline bool PerformanceImplicitGemm::operator==(const PerformanceImplicitGemm& other) const
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
    const int N = ctx.batch_sz;
    const int K = ctx.n_outputs;
    const int C = ctx.n_inputs;

    const int Ho = ctx.out_height;
    const int Wo = ctx.out_width;

    const int Y = ctx.kernel_size_h;
    const int X = ctx.kernel_size_w;

    const int N1 = GemmNRepeat;
    const int N2 = GemmNPerThreadSubC;

    if(N % (N1 * N2) != 0)
        return false; // wrong! cannot divice N evenly among thread

    const int N0 = N / (N1 * N2);

    const int B = N0 * Ho * Wo;

    // Based on data type, Pack E so as to use dot2 operator
    int EPACK = 1;
    if(ctx.IsFp16())
        EPACK = 4;
    else if(ctx.IsBfp16())
        EPACK = 2;

    const int nonVectorizedC = C / EPACK;
    const int E              = nonVectorizedC * Y * X;

    if(!(EPerBlock % InBlockCopyClusterLengths_E == 0 &&
         EPerBlock % WeiBlockCopyClusterLengths_E == 0 &&
         BPerBlock % InBlockCopyClusterLengths_B == 0 &&
         KPerBlock % WeiBlockCopyClusterLengths_K == 0 && N1 % InBlockCopyClusterLengths_N1 == 0 &&
         N2 % InBlockCopyClusterLengths_N2 == 0))
        return false;

    // divide block work by [K, B]
    if(!(K % KPerBlock == 0 && B % BPerBlock == 0 && E % (2 * EPerBlock) == 0))
        return false; // wrong! cannot divice N evenly among thread

    if((N1 * N2 * BPerBlock) % (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) != 0)
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

    if(GemmMRepeat != 2 || GemmNRepeat != 2)
        return false;

    const int InBlockCopySubLengths_E = EPerBlock / InBlockCopyClusterLengths_E;
    const int InBlockCopySubLengths_B = BPerBlock / InBlockCopyClusterLengths_B;

    return (InBlockCopySubLengths_E == 1 && InBlockCopySubLengths_B == 1);
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

    GemmNRepeat = 2;

    do
    {
        if(!use_spare_set)
        {
            // use 8x8 thread-wise gemm as possible
            GemmMPerThreadSubC = 4;
            GemmNPerThreadSubC = 4;
            // use block_size = 256 as possible
            if(!NextTwoPower<2, 4>(WeiBlockCopyClusterLengths_E))
                break;
            WeiBlockCopyClusterLengths_K = 256 / WeiBlockCopyClusterLengths_E;
        }
        else
        {
            if(!NextTwoPower<2, 4>(GemmMPerThreadSubC))
                break;
            if(!NextTwoPower<2, 4>(GemmNPerThreadSubC))
                break;
            if(!NextTwoPower<1, 4>(WeiBlockCopyClusterLengths_E))
                break;
            if(!NextTwoPower<16, 128>(WeiBlockCopyClusterLengths_K))
                break;
        }

        if(!NextTwoPower<8, 16>(BPerBlock))
            break;
        if(!NextTwoPower<16, 128>(KPerBlock))
            break;
        if(!NextTwoPower<4, 16>(EPerBlock))
            break;
        if(!NextTwoPower<1, 4>(GemmMLevel0Cluster))
            break;
        if(!NextTwoPower<1, 4>(GemmNLevel0Cluster))
            break;
        if(!NextTwoPower<1, 4>(GemmMLevel1Cluster))
            break;
        if(!NextTwoPower<1, 4>(GemmNLevel1Cluster))
            break;
        if(!NextTwoPower<4, 16>(InBlockCopyClusterLengths_E))
            break;
        if(!NextTwoPower<8, 16>(InBlockCopyClusterLengths_B))
            break;
        if(!NextTwoPower<1, 2>(InBlockCopyClusterLengths_N1))
            break;
        if(!NextTwoPower<1, 4>(InBlockCopyClusterLengths_N2))
            break;
        return false;
    } while(false);

    return true;
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

std::string PerformanceImplicitGemm::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceImplicitGemm
ConvHipImplicitGemmV4Fwd::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceImplicitGemm pp;
    pp.EuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvHipImplicitGemmV4Fwd::IsValidPerformanceConfig(const ConvolutionContext& problem,
                                                        const PerformanceImplicitGemm& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(problem);
}

PerformanceImplicitGemm::PerformanceImplicitGemm(bool spare)
    : PerformanceImplicitGemm(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, spare)
{
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

ConvSolution ConvHipImplicitGemmV4Fwd::GetSolution(const ConvolutionContext& ctx,
                                                   const PerformanceImplicitGemm& config,
                                                   const bool) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    // const int GemmNPerThreadSubC = 4;

    const int N1 = config.GemmNRepeat;
    const int N2 = config.GemmNPerThreadSubC;

    std::size_t n  = ctx.batch_sz;
    std::size_t k  = ctx.n_outputs;
    std::size_t ho = ctx.out_height;
    std::size_t wo = ctx.out_width;

    std::size_t b = (n * ho * wo) / (N1 * N2);

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

    construction_parameters.kernel_file =
        "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer.cpp";

    construction_parameters.kernel_name =
        "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer";

    std::size_t WeiBlockCopySubLengths_E = EPerBlock / config.WeiBlockCopyClusterLengths_E;
    std::size_t WeiBlockCopySubLengths_K = KPerBlock / config.WeiBlockCopyClusterLengths_K;

    int WeiBlockCopySrcDataPerRead_E  = GetReadWriteVectorSize(WeiBlockCopySubLengths_E);
    int WeiBlockCopyDstDataPerWrite_K = GetReadWriteVectorSize(WeiBlockCopySubLengths_K);

    std::size_t InBlockCopySubLengths_B  = BPerBlock / config.InBlockCopyClusterLengths_B;
    std::size_t InBlockCopySubLengths_N2 = N2 / config.InBlockCopyClusterLengths_N2;

    int InBlockCopySrcDataPerRead_B   = GetReadWriteVectorSize(InBlockCopySubLengths_B);
    int InBlockCopyDstDataPerWrite_N2 = GetReadWriteVectorSize(InBlockCopySubLengths_N2);

    if(ctx.kernel_size_w > 1)
    {
        InBlockCopySrcDataPerRead_B =
            std::min(InBlockCopySrcDataPerRead_B, GetReadWriteVectorSize(ctx.kernel_dilation_w));
    }

    if(ctx.kernel_stride_w > 1)
    {
        InBlockCopySrcDataPerRead_B = 1;
    }

    // TBD: Due to underlying bug, we need to restrict reading/writing only 1 fp16 value at a time
    if(ctx.IsFp16())
    {
        WeiBlockCopySrcDataPerRead_E  = 1;
        WeiBlockCopyDstDataPerWrite_K = 1;
        InBlockCopyDstDataPerWrite_N2 = 1;
        InBlockCopySrcDataPerRead_B   = 1;
    }

    bool use_amd_inline_asm = true;
    if(StartsWith(ctx.GetStream().GetDeviceName(), "gfx8"))
        use_amd_inline_asm = false;

    // clang-format off
    construction_parameters.comp_options =
        std::string(" -std=c++14") +
        std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(ctx.batch_sz) +
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ctx.n_outputs) +
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ctx.n_inputs) +
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ctx.in_height) +
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ctx.in_width) +
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ctx.out_height) +
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ctx.out_width) +
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
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_SRC_DATE_PER_READ_E=") + std::to_string(WeiBlockCopySrcDataPerRead_E) + 
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_DST_DATE_PER_WRITE_K=") + std::to_string(WeiBlockCopyDstDataPerWrite_K) + 
        std::string(" -DCK_BLOCKWISE_GEMM_USE_AMD_INLINE_ASM=") + std::to_string(use_amd_inline_asm ? 1 : 0) +
        std::string(" -D__HIP_PLATFORM_HCC__=1") +
        ctx.general_compile_options;
    // clang-format on

    result.construction_params.push_back(construction_parameters);
    return result;
}

int ConvHipImplicitGemmV4Fwd::RunAndMeasureSolution(miopen::Handle& profile_h,
                                                    ConstData_t bot_ocl_buf,
                                                    Data_t top_ocl_buf,
                                                    ConstData_t wei_ocl_buf,
                                                    ConstData_t bias_ocl_buf,
                                                    const ConvolutionContext&,
                                                    const ConvSolution& solution,
                                                    float& elapsed_time) const
{
    assert(bias_ocl_buf == nullptr);
    (void)bias_ocl_buf;

    KernelInfo k_info;

    k_info = solution.construction_params[0];

#ifdef NDEBUG
    try
#endif
    {
        elapsed_time = std::numeric_limits<float>::max();
        // ConvolutionContext::general_compile_options is for OpenCL kernels
        // and thus not applicable for assembly.
        auto kernel = profile_h.AddKernel("",
                                          "",
                                          k_info.kernel_file,
                                          k_info.kernel_name,
                                          k_info.l_wk,
                                          k_info.g_wk,
                                          k_info.comp_options);

        kernel(bot_ocl_buf, wei_ocl_buf, top_ocl_buf);

        elapsed_time = profile_h.GetKernelTime();
    }
#ifdef NDEBUG
    catch(miopen::Exception&)
    {
        return -1;
    }
#endif
    return 0;
}

PerformanceImplicitGemm ConvHipImplicitGemmV4Fwd::Search(const ConvolutionContext& context) const
{
    return GenericSearchFwd(*this, context);
}

} // namespace solver
} // namespace miopen
