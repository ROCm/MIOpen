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
#include <miopen/env.hpp>

namespace miopen {
namespace solver {

// fail with vector load for some cases
/// \todo enable vector load after fix it
#define WORKAROUND_FAILED_VECTOR_LOAD 1

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM)

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

    const int B = N * Ho * Wo;

    const auto nonVectorizedC = C / GetEPackLength(ctx, true);
    const auto E              = static_cast<int>(nonVectorizedC) * Y * X;

    if(!(EPerBlock % InBlockCopyClusterLengths_E == 0 &&
         EPerBlock % WeiBlockCopyClusterLengths_E == 0 &&
         BPerBlock % InBlockCopyClusterLengths_B == 0 &&
         KPerBlock % WeiBlockCopyClusterLengths_K == 0))
        return false;

    // divide block work by [K, B]
    if(!(K % KPerBlock == 0 && B % BPerBlock == 0 && E % (2 * EPerBlock) == 0))
        return false; // wrong! cannot divice N evenly among thread

    if(ctx.direction.IsBackwardWrW())
    {
        if(!((X * Y) % (EPerBlock / WeiBlockCopyClusterLengths_E) == 0))
            return false;
    }

    const int WaveSize  = 64;
    const int BlockSize = BPerBlock * KPerBlock / (GemmMPerWave * GemmNPerWave) * WaveSize;

    // fail with blockSize >= 512
    /// \todo fix the issue with blockSize >= 512
    if(BlockSize < 64 || BlockSize > 256)
        return false;

    const std::size_t lds_size = (BPerBlock + KPerBlock) * EPerBlock * GetEPackLength(ctx, true) *
                                 GetTypeSize(ctx.in_data_type) * 2;

    if(lds_size > 64 * 1024)
        return false;

    if(BlockSize != InBlockCopyClusterLengths_E * InBlockCopyClusterLengths_B)
        return false;

    if(BlockSize != WeiBlockCopyClusterLengths_K * WeiBlockCopyClusterLengths_E)
        return false;

    if((KPerBlock % GemmMPerWave) != 0 || (BPerBlock % GemmNPerWave) != 0)
        return false;

    const int GemmMWaves = KPerBlock / GemmMPerWave;
    const int GemmNWaves = BPerBlock / GemmNPerWave;

    return (GemmMPerWave * GemmMWaves == KPerBlock && GemmNPerWave * GemmNWaves == BPerBlock);
}

bool PerformanceImplicitGemmXdlops::IsValidValue() const
{
    // clang-format off
    return IsTwoPower<32,128>(BPerBlock)
        && IsTwoPower<32,128>(KPerBlock)
        && IsTwoPower<4,32>(EPerBlock)
        && IsTwoPower<32,64>(GemmMPerWave)
        && IsTwoPower<32,64>(GemmNPerWave)
        && IsTwoPower<4,16>(InBlockCopyClusterLengths_E)
        && IsTwoPower<8,32>(InBlockCopyClusterLengths_B)
        && IsTwoPower<2,4>(WeiBlockCopyClusterLengths_E)
        && IsTwoPower<16,128>(WeiBlockCopyClusterLengths_K); // clang-format on
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
            if(!NextTwoPower<32, 128>(BPerBlock))
                break;
            if(!NextTwoPower<32, 128>(KPerBlock))
                break;
            if(!NextTwoPower<4, 32>(EPerBlock))
                break;
            if(!NextTwoPower<32, 64>(GemmMPerWave))
                break;
            if(!NextTwoPower<32, 64>(GemmNPerWave))
                break;
        }
        if(!NextTwoPower<4, 16>(InBlockCopyClusterLengths_E))
            break;
        if(!NextTwoPower<8, 32>(InBlockCopyClusterLengths_B))
            break;
        if(!NextTwoPower<2, 4>(WeiBlockCopyClusterLengths_E))
            break;
        if(!NextTwoPower<16, 128>(WeiBlockCopyClusterLengths_K))
            break;
        return false;
    } while(false);

    return true;
}

void PerformanceImplicitGemmXdlops::EuristicInit(const ConvolutionContext& ctx)
{
    // default:128,128,16,64,64,8,32,4,64
    {
        BPerBlock = 128;
        KPerBlock = 128;
        EPerBlock = 16;

        GemmMPerWave = 64;
        GemmNPerWave = 64;

        InBlockCopyClusterLengths_E = 8;
        InBlockCopyClusterLengths_B = 32;

        WeiBlockCopyClusterLengths_E = 4;
        WeiBlockCopyClusterLengths_K = 64;
    }

    // 64,32,4,32,64,4,16,2,32
    if(!IsValid(ctx))
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
    if(!IsValid(ctx))
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

    // 32,64,4,64,32,4,16,4,16
    if(!IsValid(ctx))
    {
        BPerBlock = 32;
        KPerBlock = 64;
        EPerBlock = 4;

        GemmMPerWave = 64;
        GemmNPerWave = 32;

        InBlockCopyClusterLengths_E = 4;
        InBlockCopyClusterLengths_B = 16;

        WeiBlockCopyClusterLengths_E = 4;
        WeiBlockCopyClusterLengths_K = 16;
    }

    // 32,32,4,32,32,4,16,2,32
    if(!IsValid(ctx))
    {
        BPerBlock = 32;
        KPerBlock = 32;
        EPerBlock = 4;

        GemmMPerWave = 32;
        GemmNPerWave = 32;

        InBlockCopyClusterLengths_E = 4;
        InBlockCopyClusterLengths_B = 16;

        WeiBlockCopyClusterLengths_E = 2;
        WeiBlockCopyClusterLengths_K = 32;
    }

    // 32,32,4,32,32,4,16,4,16
    if(!IsValid(ctx))
    {
        BPerBlock = 32;
        KPerBlock = 32;
        EPerBlock = 4;

        GemmMPerWave = 32;
        GemmNPerWave = 32;

        InBlockCopyClusterLengths_E = 4;
        InBlockCopyClusterLengths_B = 16;

        WeiBlockCopyClusterLengths_E = 4;
        WeiBlockCopyClusterLengths_K = 16;
    }

    if(!IsValid(ctx))
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
ConvHipImplicitGemmV4R4FwdXdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmXdlops>(ctx);
}

PerformanceImplicitGemmXdlops::PerformanceImplicitGemmXdlops(bool spare)
{
    BPerBlock = spare ? 32 : 64;
    KPerBlock = spare ? 32 : 64;
    EPerBlock = spare ? 4 : 8;

    GemmMPerWave = spare ? 32 : 64;
    GemmNPerWave = spare ? 32 : 64;

    InBlockCopyClusterLengths_E = 4;
    InBlockCopyClusterLengths_B = 8;

    WeiBlockCopyClusterLengths_E = 1;
    WeiBlockCopyClusterLengths_K = 16;

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

static inline ConvSolution GetSolutionBase(const ConvolutionContext& ctx,
                                           const PerformanceImplicitGemmXdlops& config,
                                           const ImplicitGemmXdlopsKernel kernel,
                                           const int n,
                                           const int k,
                                           const int ho,
                                           const int wo)
{
    ConvSolution result;
    KernelInfo construction_parameters;

    std::size_t b = (n * ho * wo);

    std::size_t BPerBlock = config.BPerBlock;
    std::size_t KPerBlock = config.KPerBlock;
    std::size_t EPerBlock = config.EPerBlock;

    const int WaveSize = 64;
    std::size_t block_size =
        BPerBlock * KPerBlock / (config.GemmMPerWave * config.GemmNPerWave) * WaveSize;

    std::size_t grid_size = (b / BPerBlock) * (k / KPerBlock);

    const int Y = ctx.kernel_size_h;
    const int X = ctx.kernel_size_w;
    const int C = ctx.n_inputs;
    const int E = C * Y * X;

    std::size_t global_load_size =
        (BPerBlock + KPerBlock) * E * grid_size * GetTypeSize(ctx.in_data_type);
    std::size_t global_store_size = k * b * GetTypeSize(ctx.in_data_type);
    std::size_t global_size       = global_load_size + global_store_size;

    MIOPEN_LOG_I2("global load/store size = "
                  << static_cast<float>(global_size) / 1024 / 1024 / 1024
                  << " GB.");

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

    if(kernel == ImplicitGemmXdlopsKernel::KernelFwdWrw)
    {
        construction_parameters.kernel_file =
            "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_lds_double_buffer.cpp";

        construction_parameters.kernel_name =
            "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_lds_double_buffer";
    }
    else
    {
        construction_parameters.kernel_file =
            "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kc1x1_nkhw_lds_double_buffer.cpp";

        construction_parameters.kernel_name =
            "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kc1x1_nkhw_lds_double_buffer";
    }

    std::size_t WeiBlockCopySubLengths_E = EPerBlock / config.WeiBlockCopyClusterLengths_E;
    std::size_t WeiBlockCopySubLengths_K = KPerBlock / config.WeiBlockCopyClusterLengths_K;

    int WeiBlockCopySrcDataPerRead_E  = GetReadWriteVectorSize(WeiBlockCopySubLengths_E);
    int WeiBlockCopyDstDataPerWrite_K = GetReadWriteVectorSize(WeiBlockCopySubLengths_K);

    int OutThreadCopyDataPerAccess_B = 1;

#if WORKAROUND_FAILED_VECTOR_LOAD
    int InBlockCopyDataPerAccess_B = 1;
#else
    std::size_t InBlockCopySubLengths_B = BPerBlock / config.InBlockCopyClusterLengths_B;
    int InBlockCopyDataPerAccess_B      = GetReadWriteVectorSize(InBlockCopySubLengths_B);
#endif

    WeiBlockCopySrcDataPerRead_E =
        ctx.direction.IsBackwardData() ? 1 : WeiBlockCopySrcDataPerRead_E;

    // TBD: Due to underlying bug, we need to restrict reading/writing only 1 fp16 value at a time
    if(ctx.IsFp16() || ctx.IsBfp16())
    {
        WeiBlockCopySrcDataPerRead_E  = 1;
        WeiBlockCopyDstDataPerWrite_K = 1;
    }

    const ImplicitGemmDirection direction =
        ctx.direction.IsForward()
            ? ImplicitGemmDirection::ForwardData
            : (ctx.direction.IsBackwardData() ? ImplicitGemmDirection::BackwardData
                                              : ImplicitGemmDirection::BackwardWeight);

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
        std::string(" -std=c++14 ") +
        std::string(" -DCK_PARAM_PROBLEM_DIRECTION=") + std::to_string(static_cast<int>(direction)) +
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
        std::string(" -DCK_PARAM_EPACK_LENGTH=") + std::to_string(GetEPackLength(ctx, true)) + 
        std::string(" -DCK_USE_AMD_XDLOPS=") + std::to_string(IsXdlopsSupport(ctx) ? 1 : 0) +
        std::string(" -DCK_USE_AMD_XDLOPS_INLINE_ASM=") + std::to_string(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM{}) ? 1 : 0) +
        std::string(" -D__HIP_PLATFORM_HCC__=1") +
        ctx.general_compile_options;
    // clang-format on

    result.construction_params.push_back(construction_parameters);
    return result;
}

ConvSolution ConvHipImplicitGemmV4R4FwdXdlops::GetSolution(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& config, bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmXdlopsKernel::KernelFwdWrw,
                           ctx.batch_sz,
                           ctx.n_outputs,
                           ctx.out_height,
                           ctx.out_width);
}

ConvSolution ConvHipImplicitGemmV4R4Xdlops_1x1::GetSolution(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& config, bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmXdlopsKernel::Kernel1x1,
                           ctx.batch_sz,
                           ctx.n_outputs,
                           ImgHeight(ctx),
                           ImgWidth(ctx));
}

ConvSolution ConvHipImplicitGemmV4R4WrWXdlops::GetSolution(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& config, bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmXdlopsKernel::KernelFwdWrw,
                           ctx.n_outputs,
                           ctx.n_inputs,
                           ctx.kernel_size_h,
                           ctx.kernel_size_w);
}

int ConvHipImplicitGemmV4R4FwdXdlops::RunAndMeasureSolution(miopen::Handle& profile_h,
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

int ConvHipImplicitGemmV4R4Xdlops_1x1::RunAndMeasureSolution(miopen::Handle& profile_h,
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

int ConvHipImplicitGemmV4R4WrWXdlops::RunAndMeasureSolution(miopen::Handle& profile_h,
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

bool ConvHipImplicitGemmV4R4FwdXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;

    if(!ctx.direction.IsForward())
        return false;

    if(!ctx.Is2d())
        return false;

    // channels is divided by epack to pack 2/4 fp16/bfp16
    if(ctx.n_inputs % GetEPackLength(ctx, true) != 0)
        return false;

    // For fp16, when c*x*y % 4 == 0, 4 channels are accumulated through dot4 (2 * dot2) operation
    const int MultipleOf = ctx.IsFp16() ? 32 : ctx.IsBfp16() ? 16 : 8;
    if((ctx.n_inputs * ctx.kernel_size_h * ctx.kernel_size_w) % MultipleOf != 0)
        return false;

    // padding support required for out_of_bound configs
    bool no_out_of_bound = (ctx.in_width >= ((ctx.kernel_size_w - 1) * ctx.kernel_dilation_w + 1) +
                                                (ctx.out_width - 1) * ctx.kernel_stride_w) &&
                           (ctx.in_height >= ((ctx.kernel_size_h - 1) * ctx.kernel_dilation_h + 1) +
                                                 (ctx.out_height - 1) * ctx.kernel_stride_h);

    return IsXdlopsSupport(ctx) && no_out_of_bound && ctx.pad_h == 0 && ctx.pad_w == 0 &&
           ctx.group_counts == 1 && ctx.n_outputs % 32 == 0 &&
           (ctx.batch_sz * ctx.out_height * ctx.out_width) % 32 == 0;
}

bool ConvHipImplicitGemmV4R4Xdlops_1x1::IsApplicable(const ConvolutionContext& ctx) const
{
    return IsXdlopsSupport(ctx) && ctx.Is2d() && ctx.IsFp32() && ctx.pad_h == 0 && ctx.pad_w == 0 &&
           ctx.group_counts == 1 && (ctx.batch_sz * ImgHeight(ctx) * ImgWidth(ctx)) % 32 == 0 &&
           ctx.n_outputs % 32 == 0 &&
           (ctx.n_inputs * ctx.kernel_size_h * ctx.kernel_size_w) % 8 == 0 &&
           ctx.kernel_size_h == 1 && ctx.kernel_size_w == 1;
}

bool ConvHipImplicitGemmV4R4WrWXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!ctx.direction.IsBackwardWrW())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;

    // channels is divided by epack to pack 2/4 fp16/bfp16
    if(ctx.batch_sz % GetEPackLength(ctx, true) != 0)
        return false;

    // For fp16, when c*x*y % 4 == 0, 4 channels are accumulated through dot4 (2 * dot2) operation
    const int MultipleOf = ctx.IsFp16() ? 32 : ctx.IsBfp16() ? 16 : 8;
    if((ctx.batch_sz * ctx.in_height * ctx.in_width) % MultipleOf != 0)
        return false;

    return IsXdlopsSupport(ctx) && ctx.pad_h == 0 && ctx.pad_w == 0 && ctx.group_counts == 1 &&
           ctx.n_outputs % 8 == 0 &&
           (ctx.n_outputs * ctx.kernel_size_h * ctx.kernel_size_w) % 32 == 0 &&
           ctx.n_inputs % 32 == 0;
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4Xdlops_1x1::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmXdlops>(ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4WrWXdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmXdlops>(ctx);
}

bool ConvHipImplicitGemmV4R4FwdXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

bool ConvHipImplicitGemmV4R4Xdlops_1x1::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

bool ConvHipImplicitGemmV4R4WrWXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4Xdlops_1x1::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchFwd(*this, ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4FwdXdlops::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchFwd(*this, ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4WrWXdlops::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchFwd(*this, ctx);
}

} // namespace solver
} // namespace miopen
