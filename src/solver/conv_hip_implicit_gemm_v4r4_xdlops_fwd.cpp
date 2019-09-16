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

bool ConvHipImplicitGemmV4R4FwdXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    bool use_xdlops = false;
    if(StartsWith(ctx.GetStream().GetDeviceName(), "gfx908"))
        use_xdlops = true;

    // padding support required for out_of_bound configs
    bool no_out_of_bound = (ctx.in_width >= ((ctx.kernel_size_w - 1) * ctx.kernel_dilation_w + 1) +
                                                (ctx.out_width - 1) * ctx.kernel_stride_w) &&
                           (ctx.in_height >= ((ctx.kernel_size_h - 1) * ctx.kernel_dilation_h + 1) +
                                                 (ctx.out_height - 1) * ctx.kernel_stride_h);

    return use_xdlops && no_out_of_bound && ctx.Is2d() && ctx.IsFp32() &&
           ctx.direction.IsForward() && ctx.pad_h == 0 && ctx.pad_w == 0 && ctx.group_counts == 1 &&
           (ctx.batch_sz * ctx.out_height * ctx.out_width) % 64 == 0 && ctx.n_outputs % 32 == 0 &&
           (ctx.n_inputs * ctx.kernel_size_h * ctx.kernel_size_w) % 8 == 0;
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

inline bool PerformanceImplicitGemmXdlops::
operator==(const PerformanceImplicitGemmXdlops& other) const
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

bool PerformanceImplicitGemmXdlops::IsValid(const ConvolutionContext& ctx) const
{
    const int N = ctx.batch_sz;
    const int K = ctx.n_outputs;
    const int C = ctx.n_inputs;

    const int Ho = ctx.out_height;
    const int Wo = ctx.out_width;

    const int Y = ctx.kernel_size_h;
    const int X = ctx.kernel_size_w;

    const int B = N * Ho * Wo;

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
         KPerBlock % WeiBlockCopyClusterLengths_K == 0))
        return false;

    // divide block work by [K, B]
    if(!(K % KPerBlock == 0 && B % BPerBlock == 0 && E % (2 * EPerBlock) == 0))
        return false; // wrong! cannot divice N evenly among thread

    std::size_t block_size = BPerBlock * KPerBlock / (GemmMPerWave * GemmNPerWave) * 64;

    if(block_size < 64 || block_size > 512)
        return false;

    if(block_size != InBlockCopyClusterLengths_E * InBlockCopyClusterLengths_B)
        return false;

    if(block_size != WeiBlockCopyClusterLengths_K * WeiBlockCopyClusterLengths_E)
        return false;

    const int GemmMWaves = KPerBlock / GemmMPerWave;
    const int GemmNWaves = BPerBlock / GemmNPerWave;

    if(!(GemmMPerWave == 32 || GemmMPerWave == 64))
        return false;

    if(GemmMPerWave * GemmMWaves != KPerBlock || GemmNPerWave * GemmNWaves != BPerBlock)
        return false;

    return true;
}

bool PerformanceImplicitGemmXdlops::IsValidValue() const
{
    // clang-format off
    return IsTwoPower<64,128>(BPerBlock)
        && IsTwoPower<32,128>(KPerBlock)
        && IsTwoPower<4,16>(EPerBlock)
        && IsTwoPower<32,64>(GemmMPerWave)
        && GemmNPerWave == 64
        && IsTwoPower<4,16>(InBlockCopyClusterLengths_E)
        && IsTwoPower<8,16>(InBlockCopyClusterLengths_B)
        && IsTwoPower<1,4>(WeiBlockCopyClusterLengths_E)
        && IsTwoPower<16,128>(WeiBlockCopyClusterLengths_K); // clang-format on
}

bool PerformanceImplicitGemmXdlops::SetNextValue()
{
    GemmNPerWave = 64;
    do
    {
        if(!NextTwoPower<64, 128>(BPerBlock))
            break;
        if(!NextTwoPower<32, 128>(KPerBlock))
            break;
        if(!NextTwoPower<4, 16>(EPerBlock))
            break;
        if(!NextTwoPower<32, 64>(GemmMPerWave))
            break;
        if(!NextTwoPower<4, 16>(InBlockCopyClusterLengths_E))
            break;
        if(!NextTwoPower<8, 16>(InBlockCopyClusterLengths_B))
            break;
        if(!NextTwoPower<1, 4>(WeiBlockCopyClusterLengths_E))
            break;
        if(!NextTwoPower<16, 128>(WeiBlockCopyClusterLengths_K))
            break;
        return false;
    } while(false);

    return true;
}

void PerformanceImplicitGemmXdlops::EuristicInit(const ConvolutionContext& config)
{
    // default
    {
        BPerBlock = 128;
        KPerBlock = 128;
        EPerBlock = 8;

        GemmMPerWave = 64;
        GemmNPerWave = 64;

        InBlockCopyClusterLengths_E = 8;
        InBlockCopyClusterLengths_B = 32;

        WeiBlockCopyClusterLengths_E = 2;
        WeiBlockCopyClusterLengths_K = 128;
    }

    // 64,32,8,32,64,8,8,4,16
    if(!IsValid(config))
    {
        BPerBlock = 64;
        KPerBlock = 32;
        EPerBlock = 8;

        GemmMPerWave = 32;
        GemmNPerWave = 64;

        InBlockCopyClusterLengths_E = 8;
        InBlockCopyClusterLengths_B = 8;

        WeiBlockCopyClusterLengths_E = 4;
        WeiBlockCopyClusterLengths_K = 16;
    }

    // 64,32,4,32,64,4,16,2,32
    if(!IsValid(config))
    {
        BPerBlock = 64;
        KPerBlock = 32;
        EPerBlock = 4;

        GemmMPerWave = 32;
        GemmNPerWave = 64;

        InBlockCopyClusterLengths_E = 4;
        InBlockCopyClusterLengths_B = 16;

        WeiBlockCopyClusterLengths_E = 2;
        WeiBlockCopyClusterLengths_K = 32;
    }

    // 64,32,4,32,64,4,16,4,16
    if(!IsValid(config))
    {
        BPerBlock = 64;
        KPerBlock = 32;
        EPerBlock = 4;

        GemmMPerWave = 32;
        GemmNPerWave = 64;

        InBlockCopyClusterLengths_E = 4;
        InBlockCopyClusterLengths_B = 16;

        WeiBlockCopyClusterLengths_E = 4;
        WeiBlockCopyClusterLengths_K = 16;
    }

    if(!IsValid(config))
    {
        MIOPEN_LOG_E("All attempts failed");
        assert(false);
    }
    MIOPEN_LOG_I(ToString());
}

std::string PerformanceImplicitGemmXdlops::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4FwdXdlops::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceImplicitGemmXdlops pp;
    pp.EuristicInit(params);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

bool ConvHipImplicitGemmV4R4FwdXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& problem, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(problem);
}

PerformanceImplicitGemmXdlops::PerformanceImplicitGemmXdlops(bool spare)
    : PerformanceImplicitGemmXdlops(1, 1, 1, 1, 1, 1, 1, 1, 1, spare)
{
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

ConvSolution ConvHipImplicitGemmV4R4FwdXdlops::GetSolution(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& config, bool) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    std::size_t n  = ctx.batch_sz;
    std::size_t k  = ctx.n_outputs;
    std::size_t ho = ctx.out_height;
    std::size_t wo = ctx.out_width;

    std::size_t b = (n * ho * wo);

    std::size_t BPerBlock = config.BPerBlock;
    std::size_t KPerBlock = config.KPerBlock;
    std::size_t EPerBlock = config.EPerBlock;

    std::size_t block_size =
        BPerBlock * KPerBlock / (config.GemmMPerWave * config.GemmNPerWave) * 64;

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
        "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_lds_double_buffer.cpp";

    construction_parameters.kernel_name =
        "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_lds_double_buffer";

    std::size_t WeiBlockCopySubLengths_E = EPerBlock / config.WeiBlockCopyClusterLengths_E;
    std::size_t WeiBlockCopySubLengths_K = KPerBlock / config.WeiBlockCopyClusterLengths_K;

    int WeiBlockCopySrcDataPerRead_E  = GetReadWriteVectorSize(WeiBlockCopySubLengths_E);
    int WeiBlockCopyDstDataPerWrite_K = GetReadWriteVectorSize(WeiBlockCopySubLengths_K);

    int OutThreadCopyDataPerAccess_B = 1;
    int InBlockCopyDataPerAccess_B   = 1;

    // TBD: Due to underlying bug, we need to restrict reading/writing only 1 fp16 value at a time
    if(ctx.IsFp16())
    {
        WeiBlockCopySrcDataPerRead_E  = 1;
        WeiBlockCopyDstDataPerWrite_K = 1;
    }

    bool use_xdlops = false;
    if(StartsWith(ctx.GetStream().GetDeviceName(), "gfx908"))
        use_xdlops = true;

    // clang-format off
    construction_parameters.comp_options =
        std::string(" -std=c++14 ") +
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
        std::string(" -DCK_PARAM_GEMM_M_PER_WAVE=") + std::to_string(config.GemmMPerWave) +
        std::string(" -DCK_PARAM_GEMM_N_PER_WAVE=") + std::to_string(config.GemmNPerWave) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_E=") + std::to_string(config.InBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_B=") + std::to_string(config.InBlockCopyClusterLengths_B) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E=") + std::to_string(config.WeiBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K=") + std::to_string(config.WeiBlockCopyClusterLengths_K) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_DATA_PER_ACCESS_B=") + std::to_string(InBlockCopyDataPerAccess_B) + 
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E=") + std::to_string(WeiBlockCopySrcDataPerRead_E) + 
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_K=") + std::to_string(WeiBlockCopyDstDataPerWrite_K) + 
        std::string(" -DCK_PARAM_OUT_THREAD_COPY_DATA_PER_ACCESS_B=") + std::to_string(OutThreadCopyDataPerAccess_B) + 
        std::string(" -DCK_ENABLE_XDLOPS=") + std::to_string(use_xdlops ? 1 : 0) +
        std::string(" -D__HIP_PLATFORM_HCC__=1") +
        ctx.general_compile_options;
    // clang-format on

    result.construction_params.push_back(construction_parameters);
    return result;
}

int ConvHipImplicitGemmV4R4FwdXdlops::RunAndMeasureSolution(miopen::Handle& profile_h,
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

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4FwdXdlops::Search(const ConvolutionContext& context) const
{
    return GenericSearchFwd(*this, context);
}

} // namespace solver
} // namespace miopen
