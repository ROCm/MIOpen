/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include "miopen/conv/solvers.hpp"
#include "miopen/handle.hpp"
#include <miopen/conv/invokers/impl_gemm_dynamic.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

struct TunableImplicitGemmV4R1Dynamic
{
    int BPerBlock;
    int KPerBlock;
    int EPerBlock;

    int GemmNRepeat;

    int GemmMPerThreadSubC;
    int GemmNPerThreadSubC;

    int GemmMLevel0Cluster;
    int GemmNLevel0Cluster;
    int GemmMLevel1Cluster;
    int GemmNLevel1Cluster;

    int InBlockCopyClusterLengths_E;
    int InBlockCopyClusterLengths_N1;
    int InBlockCopyClusterLengths_B;
    int InBlockCopyClusterLengths_N2;

    int WeiBlockCopyClusterLengths_E;
    int WeiBlockCopyClusterLengths_K;

    TunableImplicitGemmV4R1Dynamic(
        int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int);
    bool IsValid(const ExecutionContext&, const ProblemDescription&) const;
};

static inline const std::vector<TunableImplicitGemmV4R1Dynamic>&
GetImplicitGemmV4R1DynamicTunables()
{
    // clang-format off
    static const std::vector<TunableImplicitGemmV4R1Dynamic> tunables = {
        {  16, 128,  16,   2,   4,   4,   4,   4,   4,   4,  16,   1,  16,   1,   4,  64},
        {  16, 128,   8,   2,   4,   4,   4,   4,   4,   4,   8,   2,  16,   1,   2, 128},
        {   8, 128,   8,   2,   4,   4,   4,   4,   4,   2,   8,   1,   8,   2,   2,  64},
        {   8,  64,   8,   2,   4,   4,   4,   2,   2,   4,   8,   1,   8,   1,   4,  16},
        {  16,  32,   4,   2,   4,   4,   1,   4,   4,   4,   4,   1,  16,   1,   4,  16},
        {  16,  16,   4,   2,   2,   2,   2,   4,   2,   4,   4,   1,  16,   1,   4,  16},
        {   8,  32,   4,   2,   2,   2,   2,   4,   4,   2,   4,   2,   8,   1,   4,  16}
    };
    // clang-format on
    return tunables;
}

using AsmImplicitGemmKernelV4R1Fwd_t = enum {
    AsmImplicitGemmV4R1     = 0,
    AsmImplicitGemmV4R1_1x1 = 1,
};

static inline std::string
GetKernelNameImplicitGemmV4R1Dynamic(const TunableImplicitGemmV4R1Dynamic& config,
                                     AsmImplicitGemmKernelV4R1Fwd_t kernel_type)
{
    int GemmMRepeat = config.KPerBlock / (config.GemmMPerThreadSubC * config.GemmMLevel0Cluster *
                                          config.GemmMLevel1Cluster);
    int ThreadTileM = GemmMRepeat * config.GemmMPerThreadSubC;
    int ThreadTileN = config.GemmNRepeat * config.GemmNPerThreadSubC;
    std::string kernel_name_prefix;
    if(AsmImplicitGemmV4R1 == kernel_type)
        kernel_name_prefix = std::string("igemm_v4r1_dynamic_");
    else // if (AsmImplicitGemmV4R1_1x1 == kernel_type)
        kernel_name_prefix = std::string("igemm_v4r1_1x1_dynamic_");

    return kernel_name_prefix + std::to_string(config.KPerBlock) + "x" +
           std::to_string(config.BPerBlock * config.GemmNRepeat * config.GemmNPerThreadSubC) + "x" +
           std::to_string(config.EPerBlock) + "_" + std::to_string(ThreadTileM) + "x" +
           std::to_string(ThreadTileN) + "_" + std::to_string(config.GemmMPerThreadSubC) + "x" +
           std::to_string(config.GemmMLevel0Cluster) + "x" +
           std::to_string(config.GemmMLevel1Cluster) + "x" +
           std::to_string(config.GemmNPerThreadSubC) + "x" +
           std::to_string(config.GemmNLevel0Cluster) + "x" +
           std::to_string(config.GemmNLevel1Cluster) + "_" +
           std::to_string(config.InBlockCopyClusterLengths_E) + "x" +
           std::to_string(config.InBlockCopyClusterLengths_N1) + "x" +
           std::to_string(config.InBlockCopyClusterLengths_B) + "x" +
           std::to_string(config.InBlockCopyClusterLengths_N2) + "_" +
           std::to_string(config.WeiBlockCopyClusterLengths_E) + "x" +
           std::to_string(config.WeiBlockCopyClusterLengths_K);
}

static inline int GetImplicitGemmV4R1DynamicBlockSize(const TunableImplicitGemmV4R1Dynamic& config)
{
    return config.GemmMLevel0Cluster * config.GemmNLevel0Cluster * config.GemmMLevel1Cluster *
           config.GemmNLevel1Cluster;
}

static inline int GetImplicitGemmV4R1DynamicGridSize(const ProblemDescription& problem,
                                                     const TunableImplicitGemmV4R1Dynamic& config)
{
    const auto& N1 = config.GemmNRepeat;
    const auto& N2 = config.GemmNPerThreadSubC;

    const int n  = problem.GetBatchSize();
    const int k  = problem.GetOutChannels();
    const int ho = problem.GetOutHeight();
    const int wo = problem.GetOutWidth();

    const auto& b = (static_cast<std::size_t>(n) * ho * wo) / (static_cast<std::size_t>(N1) * N2);
    const auto& b_per_block = config.BPerBlock;
    const auto& k_per_block = config.KPerBlock;

    return (b / b_per_block) * (k / k_per_block); // NOLINT
}

TunableImplicitGemmV4R1Dynamic::TunableImplicitGemmV4R1Dynamic(int BPerBlock_,
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
                                                               int InBlockCopyClusterLengths_N1_,
                                                               int InBlockCopyClusterLengths_B_,
                                                               int InBlockCopyClusterLengths_N2_,
                                                               int WeiBlockCopyClusterLengths_E_,
                                                               int WeiBlockCopyClusterLengths_K_)
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
      InBlockCopyClusterLengths_N1(InBlockCopyClusterLengths_N1_),
      InBlockCopyClusterLengths_B(InBlockCopyClusterLengths_B_),
      InBlockCopyClusterLengths_N2(InBlockCopyClusterLengths_N2_),
      WeiBlockCopyClusterLengths_E(WeiBlockCopyClusterLengths_E_),
      WeiBlockCopyClusterLengths_K(WeiBlockCopyClusterLengths_K_)
{
}

bool TunableImplicitGemmV4R1Dynamic::IsValid(const ExecutionContext& ctx,
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
    {
        return false;
    }

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
    {
        return false;
    }

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

bool ConvAsmImplicitGemmV4R1DynamicFwd::IsApplicable(const ExecutionContext& ctx,
                                                     const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1))
        return false;

    const auto device_name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(device_name, "gfx900") || StartsWith(device_name, "gfx906")))
        return false;

    if(!ctx.use_asm_kernels)
        return false;

    if(!problem.IsDirectionForward())
        return false;

    if(!problem.Is2d())
        return false;

    if(problem.HasNonPackedTensors())
        return false;

    if(!problem.AllTensorsDimsFitIntoInt())
        return false;

    if(!problem.IsFp32())
        return false;

    if(problem.IsTensorsCasted())
        return false;

    if(!ctx.rmv.IsV3())
        return false;

    if(problem.GetGroupCount() != 1)
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;
    auto tunables = GetImplicitGemmV4R1DynamicTunables();
    return !std::none_of(tunables.begin(), tunables.end(), [&](auto tunable) {
        return tunable.IsValid(ctx, problem);
    });
}

bool ConvAsmImplicitGemmV4R1DynamicFwd_1x1::IsApplicable(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_V4R1_1X1))
        return false;

    const auto device_name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(device_name, "gfx900") || StartsWith(device_name, "gfx906")))
        return false;

    if(!ctx.use_asm_kernels)
        return false;

    if(!problem.IsDirectionForward())
        return false;

    if(!problem.Is2d())
        return false;

    if(!problem.AllTensorsDimsFitIntoInt())
        return false;

    if(!problem.IsFp32())
        return false;

    if(!ctx.rmv.IsV3())
        return false;

    if(problem.GetGroupCount() != 1)
        return false;

    if((problem.GetWeightsHeight() != 1) || (problem.GetWeightsWidth() != 1))
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;
    auto tunables = GetImplicitGemmV4R1DynamicTunables();
    return !std::none_of(tunables.begin(), tunables.end(), [&](auto tunable) {
        return tunable.IsValid(ctx, problem);
    });
}

static inline ConvSolution GetSolutionBase(const ExecutionContext& ctx,
                                           const ProblemDescription& problem,
                                           const TunableImplicitGemmV4R1Dynamic& config,
                                           const AsmImplicitGemmKernelV4R1Fwd_t& kernel_type)
{
    ConvSolution result;

    std::string kernel_name = GetKernelNameImplicitGemmV4R1Dynamic(config, kernel_type);

    int block_size     = GetImplicitGemmV4R1DynamicBlockSize(config);
    int grid_size      = GetImplicitGemmV4R1DynamicGridSize(problem, config);
    bool kernel_is_1x1 = (kernel_name.find("igemm_v4r1_1x1_dynamic") == 0);

    KernelInfo kernel;
    std::ostringstream options;

    kernel.kernel_file = "igemm_v4r1_dynamic.s";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    /* Note here, for API like hipHccModuleLaunchKernel(), hipExtModuleLaunchKernel()
     * grid dims is in unit of work item.
     * But for api like hipModuleLaunchKernel(), grid dim is in unit of block.
     */
    kernel.g_wk.push_back(static_cast<std::size_t>(grid_size) * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);

    kernel.comp_options = options.str();

    MIOPEN_LOG_I2(kernel.kernel_file << ":" << kernel.kernel_name);

    if(kernel_is_1x1)
    {
        result.invoker_factory = miopen::conv::MakeImplGemmDynamicForward1x1InvokerFactory(problem);
    }
    else
    {
        int packed_value = 0;
        result.invoker_factory =
            miopen::conv::MakeImplGemmDynamicForwardInvokerFactory<int>(problem, packed_value);
    }
    result.construction_params.push_back(kernel);
    return result;
}

ConvSolution ConvAsmImplicitGemmV4R1DynamicFwd::GetSolution(const ExecutionContext& ctx,
                                                            const ProblemDescription& problem) const
{
    auto tunables = GetImplicitGemmV4R1DynamicTunables();
    auto it       = std::find_if(tunables.begin(), tunables.end(), [&](auto tunable) {
        return tunable.IsValid(ctx, problem);
    });

    if(it == tunables.end())
    {
        MIOPEN_THROW(
            miopenStatusInternalError,
            "no solution found in igemm v4r1 dynamic fwd, should call IsApplicable() first.");
    }

    return GetSolutionBase(ctx, problem, *it, AsmImplicitGemmV4R1);
}

ConvSolution
ConvAsmImplicitGemmV4R1DynamicFwd_1x1::GetSolution(const ExecutionContext& ctx,
                                                   const ProblemDescription& problem) const
{
    auto tunables = GetImplicitGemmV4R1DynamicTunables();
    auto it       = std::find_if(tunables.begin(), tunables.end(), [&](auto tunable) {
        return tunable.IsValid(ctx, problem);
    });

    if(it == tunables.end())
    {
        MIOPEN_THROW(
            miopenStatusInternalError,
            "no solution found in igemm v4r1 dynamic fwd 1x1, should call IsApplicable() first.");
    }

    return GetSolutionBase(ctx, problem, *it, AsmImplicitGemmV4R1_1x1);
}

} // namespace conv
} // namespace solver
} // namespace miopen
