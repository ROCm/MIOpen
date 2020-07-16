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
#include "miopen/solver.hpp"
#include "miopen/handle.hpp"
#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/conv/invokers/impl_gemm_dynamic.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include "implicitgemm_util.hpp"

namespace miopen {
namespace solver {

static inline const std::vector<PerformanceImplicitGemmWrwV4R1Dynamic>&
GetImplicitGemmWrwV4R1DynamicTunables()
{
    // clang-format off
    static const std::vector<PerformanceImplicitGemmWrwV4R1Dynamic> tunables = {
        {  16, 128,  16,   2,   4,   4,   4,   4,   4,   4,  16,   1,  16,   1,   4,  64, 1},
        //{  16, 128,   8,   2,   4,   4,   4,   4,   4,   4,   8,   2,  16,   1,   2, 128},
        //{   8, 128,   8,   2,   4,   4,   4,   4,   4,   2,   8,   1,   8,   2,   2,  64},
        //{   8,  64,   8,   2,   4,   4,   4,   2,   2,   4,   8,   1,   8,   1,   4,  16},
        //{  16,  32,   4,   2,   4,   4,   1,   4,   4,   4,   4,   1,  16,   1,   4,  16},
        //{  16,  16,   4,   2,   2,   2,   2,   4,   2,   4,   4,   1,  16,   1,   4,  16},
        //{   8,  32,   4,   2,   2,   2,   2,   4,   4,   2,   4,   2,   8,   1,   4,  16},
        {  16, 128,  16,   2,   4,   4,   4,   4,   4,   4,  16,   1,  16,   1,   16,  16, 1}
    };
    // clang-format on
    return tunables;
}

static inline std::string
GetKernelNameImplicitGemmWrwV4R1Dynamic(const PerformanceImplicitGemmWrwV4R1Dynamic& config)
{
    int GemmMRepeat = config.KPerBlock / (config.GemmMPerThreadSubC * config.GemmMLevel0Cluster *
                                          config.GemmMLevel1Cluster);
    int ThreadTileM = GemmMRepeat * config.GemmMPerThreadSubC;
    int ThreadTileN = config.GemmNRepeat * config.GemmNPerThreadSubC;
    std::string kernel_name_prefix;
    kernel_name_prefix = std::string("igemm_v4r1_dynamic_wrw_");

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

static inline int
GetImplicitGemmWrwV4R1DynamicBlockSize(const PerformanceImplicitGemmWrwV4R1Dynamic& config)
{
    return config.GemmMLevel0Cluster * config.GemmNLevel0Cluster * config.GemmMLevel1Cluster *
           config.GemmNLevel1Cluster;
}

static inline int
GetImplicitGemmWrwV4R1DynamicGridSize(const ConvolutionContext& ctx,
                                   const PerformanceImplicitGemmWrwV4R1Dynamic& config)
{
    const auto& N1 = config.GemmNRepeat;
    const auto& N2 = config.GemmNPerThreadSubC;
    const auto& gemmk_groups = config.GemmKGroups;

    const auto& n  = ctx.n_inputs;
    const auto& k  = ctx.n_outputs;
    const auto& ho = ctx.kernel_size_h;
    const auto& wo = ctx.kernel_size_w;

    const auto& b = (static_cast<std::size_t>(n) * ho * wo) / (static_cast<std::size_t>(N1) * N2);
    const auto& b_per_block = config.BPerBlock;
    const auto& k_per_block = config.KPerBlock;

    return (b / b_per_block) * (k / k_per_block) * gemmk_groups; // NOLINT
}

// Remove This function when invoker is fully re-factored
template <typename BotBufType, typename TopBufType, typename WeiBufType>
static inline int RunAndMeasureSolutionDynamicBase(miopen::Handle& profile_h,
                                                   BotBufType bot_buf,
                                                   TopBufType top_buf,
                                                   WeiBufType wei_buf,
                                                   const ConvolutionContext& ctx,
                                                   const ConvSolution& solution,
                                                   float& elapsed_time)
{

#ifdef NDEBUG
    try
#endif
    {
        elapsed_time = float(0);
        std::vector<KernelInvoke> kernels;

        for(auto& k_info : solution.construction_params)
        {
            auto kernel = profile_h.AddKernel("",
                                              "",
                                              k_info.kernel_file,
                                              k_info.kernel_name,
                                              k_info.l_wk,
                                              k_info.g_wk,
                                              k_info.comp_options);
            kernels.push_back(kernel);
        }
        float time =
            conv::CallImplicitGemmWrwDynamic(profile_h, ctx, bot_buf, top_buf, wei_buf, 1, kernels);
        elapsed_time += time;
    }
#ifdef NDEBUG
    catch(miopen::Exception& ex)
    {
        MIOPEN_LOG_WE(ex.what());
        return -1;
    }
#endif
    return 0;
}

bool PerformanceImplicitGemmWrwV4R1Dynamic::
operator==(const PerformanceImplicitGemmWrwV4R1Dynamic& other) const
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
        && GemmKGroups == other.GemmKGroups;
    // clang-format on
}

bool PerformanceImplicitGemmWrwV4R1Dynamic::IsValid(const ConvolutionContext& ctx) const
{
    std::size_t N = ctx.n_inputs;
    std::size_t K = ctx.n_outputs;
    std::size_t C = ctx.batch_sz;

    std::size_t Ho = ctx.kernel_size_h;
    std::size_t Wo = ctx.kernel_size_w;

    std::size_t Y = ctx.out_height;
    std::size_t X = ctx.out_width;

    const int N1 = GemmNRepeat;
    const int N2 = GemmNPerThreadSubC;
    if(N % (N1 * N2) != 0)
        return false; // wrong! cannot divice N evenly among thread

    const auto N0 = N / (N1 * N2);

    const auto B = N0 * Ho * Wo;

    const auto nonVectorizedC = C / GetEPackLength(ctx, false);
    const auto E              = nonVectorizedC * Y * X;

    if(!(EPerBlock % InBlockCopyClusterLengths_E == 0 &&
         EPerBlock % WeiBlockCopyClusterLengths_E == 0 &&
         BPerBlock % InBlockCopyClusterLengths_B == 0 &&
         KPerBlock % WeiBlockCopyClusterLengths_K == 0 && N1 % InBlockCopyClusterLengths_N1 == 0 &&
         N2 % InBlockCopyClusterLengths_N2 == 0))
        return false;

    // divide block work by [K, B]
    if(!(K % KPerBlock == 0 && B % BPerBlock == 0 && E % (EPerBlock * GemmKGroups) == 0))
        return false; // wrong! cannot divide N evenly among thread

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

    if(!(GemmMRepeat == 2 && GemmNRepeat == 2))
        return false;

    const int InBlockCopySubLengths_E  = EPerBlock / InBlockCopyClusterLengths_E;
    const int InBlockCopySubLengths_B  = BPerBlock / InBlockCopyClusterLengths_B;
    const int WeiBlockCopySubLengths_K = KPerBlock / WeiBlockCopyClusterLengths_K;

    const std::size_t lds_size = ComputeLDSRequiredSize(ctx,
                                                        BPerBlock,
                                                        KPerBlock,
                                                        EPerBlock,
                                                        GemmMPerThreadSubC,
                                                        GemmNPerThreadSubC,
                                                        InBlockCopySubLengths_B,
                                                        WeiBlockCopySubLengths_K,
                                                        GetEPackLength(ctx, false));

    if(lds_size > 64 * 1024)
        return false;

    return (InBlockCopySubLengths_E == 1 && InBlockCopySubLengths_B == 1);
}

void PerformanceImplicitGemmWrwV4R1Dynamic::EuristicInit(const ConvolutionContext& config)
{
    auto tunables = GetImplicitGemmWrwV4R1DynamicTunables();
    auto it       = std::find_if(
        tunables.begin(), tunables.end(), [&](auto tunable) { return tunable.IsValid(config); });

    if(it == tunables.end())
    {
        MIOPEN_LOG_E("All attempts failed");
        assert(false);
    }

    Copy(*it);
}

bool PerformanceImplicitGemmWrwV4R1Dynamic::IsValidValue() const
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
        && IsTwoPower<16,128>(WeiBlockCopyClusterLengths_K)
        && IsTwoPower<1,32>(GemmKGroups); // clang-format on
}

void PerformanceImplicitGemmWrwV4R1Dynamic::Copy(const PerformanceImplicitGemmWrwV4R1Dynamic& other)
{
    BPerBlock                    = other.BPerBlock;
    KPerBlock                    = other.KPerBlock;
    EPerBlock                    = other.EPerBlock;
    GemmNRepeat                  = other.GemmNRepeat;
    GemmMPerThreadSubC           = other.GemmMPerThreadSubC;
    GemmNPerThreadSubC           = other.GemmNPerThreadSubC;
    GemmMLevel0Cluster           = other.GemmMLevel0Cluster;
    GemmNLevel0Cluster           = other.GemmNLevel0Cluster;
    GemmMLevel1Cluster           = other.GemmMLevel1Cluster;
    GemmNLevel1Cluster           = other.GemmNLevel1Cluster;
    InBlockCopyClusterLengths_E  = other.InBlockCopyClusterLengths_E;
    InBlockCopyClusterLengths_N1 = other.InBlockCopyClusterLengths_N1;
    InBlockCopyClusterLengths_B  = other.InBlockCopyClusterLengths_B;
    InBlockCopyClusterLengths_N2 = other.InBlockCopyClusterLengths_N2;
    WeiBlockCopyClusterLengths_E = other.WeiBlockCopyClusterLengths_E;
    WeiBlockCopyClusterLengths_K = other.WeiBlockCopyClusterLengths_K;
    GemmKGroups                  = other.GemmKGroups;
}

bool PerformanceImplicitGemmWrwV4R1Dynamic::SetNextValue()
{
    do
    {
        size_t total_kernels = GetImplicitGemmWrwV4R1DynamicTunables().size();
        PreGeneratedKernelIndex++;
        if(PreGeneratedKernelIndex >= total_kernels)
            return false;
        Copy(GetImplicitGemmWrwV4R1DynamicTunables()[PreGeneratedKernelIndex]);
    } while(false);

    return true;
}

std::string PerformanceImplicitGemmWrwV4R1Dynamic::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceImplicitGemmWrwV4R1Dynamic::PerformanceImplicitGemmWrwV4R1Dynamic(bool)
    : PerformanceImplicitGemmWrwV4R1Dynamic(
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
{
    // get the minimal routine
    Copy(GetImplicitGemmWrwV4R1DynamicTunables()[0]);
    PreGeneratedKernelIndex = 0;
}

PerformanceImplicitGemmWrwV4R1Dynamic::PerformanceImplicitGemmWrwV4R1Dynamic(
    int BPerBlock_,
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
    int WeiBlockCopyClusterLengths_K_,
    int GemmKGroups_)
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
      WeiBlockCopyClusterLengths_K(WeiBlockCopyClusterLengths_K_),
      GemmKGroups(GemmKGroups_)
{
    PreGeneratedKernelIndex = 0;
}

size_t ConvAsmImplicitGemmV4R1DynamicWrw::GetWorkspaceSize(const ConvolutionContext& ctx) const
{
    if (1){
        int n = ctx.n_inputs;
        int k = ctx.n_outputs;
        int ho = ctx.kernel_size_h;
        int wo = ctx.kernel_size_w;

        return n * k * ho * wo * 32;
    }
    else
        return 0;
    
}

bool ConvAsmImplicitGemmV4R1DynamicWrw::IsApplicable(const ConvolutionContext& ctx) const
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(device_name, "gfx900") || StartsWith(device_name, "gfx906")))
        return false;

    if(!ctx.direction.IsBackwardWrW())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.IsFp32())
        return false;

    if(!ctx.rmv.IsV3())
        return false;

    if(ctx.group_counts != 1)
        return false;

    auto tunables = GetImplicitGemmWrwV4R1DynamicTunables();
    return !std::none_of(
        tunables.begin(), tunables.end(), [&](auto tunable) { return tunable.IsValid(ctx); });
}

PerformanceImplicitGemmWrwV4R1Dynamic
ConvAsmImplicitGemmV4R1DynamicWrw::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmWrwV4R1Dynamic>(ctx);
}


bool ConvAsmImplicitGemmV4R1DynamicWrw::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmWrwV4R1Dynamic& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

int ConvAsmImplicitGemmV4R1DynamicWrw::RunAndMeasureSolution(miopen::Handle& profile_h,
                                                             ConstData_t bot_buf,
                                                             ConstData_t top_buf,
                                                             Data_t wei_buf,
                                                             ConstData_t bias_buf,
                                                             const ConvolutionContext& ctx,
                                                             const ConvSolution& solution,
                                                             float& elapsed_time) const
{
    assert(bias_buf == nullptr);
    (void)bias_buf;

    return RunAndMeasureSolutionDynamicBase(
        profile_h, bot_buf, top_buf, wei_buf, ctx, solution, elapsed_time);
}

PerformanceImplicitGemmWrwV4R1Dynamic
ConvAsmImplicitGemmV4R1DynamicWrw::Search(const ConvolutionContext& context) const
{
    // gridwise reduction needs workspace if GemmKGroups > 1
    if(1)
        return GenericSearchWrW(*this, context, SearchTweak::WorkspaceInsteadOfWeightsBuffer);
    else
        return GenericSearchWrW(*this, context);
}

ConvSolution ConvAsmImplicitGemmV4R1DynamicWrw::GetSolution(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmWrwV4R1Dynamic& config, bool) const
{
    ConvSolution result;

    std::string kernel_name = GetKernelNameImplicitGemmWrwV4R1Dynamic(config);

    int block_size = GetImplicitGemmWrwV4R1DynamicBlockSize(config);
    int grid_size  = GetImplicitGemmWrwV4R1DynamicGridSize(ctx, config);

    KernelInfo kernel;
    std::ostringstream options;

    kernel.kernel_file = "igemm_v4r1_wrw_dynamic.s";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    /* Note here, for API like hipHccModuleLaunchKernel(), hipExtModuleLaunchKernel()
    * grid dims is in unit of work item.
    * But for api like hipModuleLaunchKernel(), grid dim is in unit of block.
    */
    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);

    kernel.comp_options = options.str();

    MIOPEN_LOG_I2(kernel.kernel_file + ":" + kernel.kernel_name);

    result.workspce_sz = GetWorkspaceSize(ctx);

    result.invoker_factory = conv::MakeImplGemmDynamicDataInvokerFactory(ctx);
    result.construction_params.push_back(kernel);
    return result;
}


} // namespace solver
} // namespace miopen
