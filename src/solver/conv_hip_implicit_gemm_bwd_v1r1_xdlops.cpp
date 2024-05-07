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
#include <miopen/conv/invokers/impl_gemm.hpp>
#include <miopen/solver.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/solver/implicitgemm_util.hpp>

#include <cstddef>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

PerformanceImplicitGemmBwdV1R1Xdlops::PerformanceImplicitGemmBwdV1R1Xdlops()
    : PerformanceImplicitGemmBwdV1R1Xdlops::PerformanceImplicitGemmBwdV1R1Xdlops(
          4, 4, 1, 4, 4, 1, false, false)
{
}

PerformanceImplicitGemmBwdV1R1Xdlops::PerformanceImplicitGemmBwdV1R1Xdlops(
    int GemmMPerBlock_,
    int GemmNPerBlock_,
    int GemmKPerBlock_,
    int GemmMPerWave_,
    int GemmNPerWave_,
    int GemmKPack_,
    bool GemmAThreadCopyMoreGemmK_,
    bool GemmBThreadCopyMoreGemmKPack_)
    : GemmMPerBlock(GemmMPerBlock_),
      GemmNPerBlock(GemmNPerBlock_),
      GemmKPerBlock(GemmKPerBlock_),
      GemmMPerWave(GemmMPerWave_),
      GemmNPerWave(GemmNPerWave_),
      GemmKPack(GemmKPack_),
      GemmAThreadCopyMoreGemmK(GemmAThreadCopyMoreGemmK_),
      GemmBThreadCopyMoreGemmKPack(GemmBThreadCopyMoreGemmKPack_)
{
}

bool PerformanceImplicitGemmBwdV1R1Xdlops::operator==(
    const PerformanceImplicitGemmBwdV1R1Xdlops& other) const
{
    // clang-format off
    return GemmMPerBlock == other.GemmMPerBlock
        && GemmNPerBlock == other.GemmNPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmMPerWave == other.GemmMPerWave
        && GemmNPerWave == other.GemmNPerWave
        && GemmKPack == other.GemmKPack
        && GemmAThreadCopyMoreGemmK  == other.GemmAThreadCopyMoreGemmK
        && GemmBThreadCopyMoreGemmKPack  == other.GemmBThreadCopyMoreGemmKPack;
    // clang-format on
}

bool PerformanceImplicitGemmBwdV1R1Xdlops::SetNextValue(const ProblemDescription&)
{
    do
    {
        // list performance parameters in reverse order, in order for tuning to iterate over the
        // range in normal order
        if(!NextFlag<false, true>(GemmBThreadCopyMoreGemmKPack))
            break;
        if(!NextFlag<false, true>(GemmAThreadCopyMoreGemmK))
            break;
        if(!NextTwoPower<1, 8>(GemmKPack))
            break;
        if(!NextTwoPower<4, 128>(GemmNPerWave))
            break;
        if(!NextTwoPower<4, 128>(GemmMPerWave))
            break;
        if(!NextTwoPower<1, 8>(GemmKPerBlock))
            break;
        if(!NextTwoPower<4, 256>(GemmNPerBlock))
            break;
        if(!NextTwoPower<4, 256>(GemmMPerBlock))
            break;
        return false;
    } while(false);

    return true;
}

void PerformanceImplicitGemmBwdV1R1Xdlops::HeuristicInit(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem)
{
    PerformanceImplicitGemmBwdV1R1Xdlops tmp;
    // loop over certain ranges of tuning parameter
    auto get_euristic_config = [&](auto is_valid_func) {
        if(problem.IsFp32())
        {
            tmp = {256, 256, 8, 128, 128, 4, false, true};

            bool all_visited = false;
            do
            {
                do
                {
                    // list in reverse order of importance,
                    // and favor large GEMM
                    if(!PreviousTwoPower<1, 8>(tmp.GemmKPerBlock))
                        break;
                    if(!PreviousTwoPower<1, 4>(tmp.GemmKPack))
                        break;
                    if(!PreviousTwoPower<4, 128>(tmp.GemmNPerWave))
                        break;
                    if(!PreviousTwoPower<4, 128>(tmp.GemmMPerWave))
                        break;
                    if(!PreviousTwoPower<4, 256>(tmp.GemmNPerBlock))
                        break;
                    if(!PreviousTwoPower<4, 256>(tmp.GemmMPerBlock))
                        break;

                    all_visited = true;
                } while(false);

                if(is_valid_func(tmp))
                    break;
            } while(!all_visited);
        }
        else if(problem.IsFp16())
        {
            tmp = {256, 256, 8, 128, 128, 8, false, true};

            bool all_visited = false;
            do
            {
                do
                {
                    // list in reverse order of importance,
                    // and favor large GEMM
                    if(!PreviousTwoPower<1, 8>(tmp.GemmKPerBlock))
                        break;
                    if(!PreviousTwoPower<4, 8>(tmp.GemmKPack))
                        break;
                    if(!PreviousTwoPower<4, 128>(tmp.GemmNPerWave))
                        break;
                    if(!PreviousTwoPower<4, 128>(tmp.GemmMPerWave))
                        break;
                    if(!PreviousTwoPower<4, 256>(tmp.GemmNPerBlock))
                        break;
                    if(!PreviousTwoPower<4, 256>(tmp.GemmMPerBlock))
                        break;

                    all_visited = true;
                } while(false);

                if(is_valid_func(tmp))
                    break;
            } while(!all_visited);
        }
        else if(problem.IsBfp16())
        {
            tmp = {256, 256, 8, 128, 128, 8, false, true};

            bool all_visited = false;
            do
            {
                do
                {
                    // list in reverse order of importance,
                    // and favor large GEMM
                    if(!PreviousTwoPower<1, 8>(tmp.GemmKPerBlock))
                        break;
                    if(!PreviousTwoPower<2, 8>(tmp.GemmKPack))
                        break;
                    if(!PreviousTwoPower<4, 128>(tmp.GemmNPerWave))
                        break;
                    if(!PreviousTwoPower<4, 128>(tmp.GemmMPerWave))
                        break;
                    if(!PreviousTwoPower<4, 256>(tmp.GemmNPerBlock))
                        break;
                    if(!PreviousTwoPower<4, 256>(tmp.GemmMPerBlock))
                        break;

                    all_visited = true;
                } while(false);

                if(is_valid_func(tmp))
                    break;
            } while(!all_visited);
        }
        else
        {
            MIOPEN_LOG_E("Only fp32, fp16, and bfp16 are supported");
            assert(false);
        }
    };

    // first round: really valid and fast
    get_euristic_config([&ctx, &problem](const auto& config) {
        return config.IsReallyValid(problem) && config.IsFastToBeUsedForTuning(ctx, problem);
    });

    // second round: really valid
    if(!tmp.IsReallyValid(problem))
    {
        get_euristic_config([&ctx, &problem](const auto& config) {
            std::ignore = ctx;
            return config.IsReallyValid(problem);
        });
    }

    // final check
    if(!tmp.IsReallyValid(problem))
    {
        MIOPEN_LOG_I("All attempts unsuccessful");
    }
    *this = tmp;
    MIOPEN_LOG_I(ToString());
}

std::tuple<int, bool> PerformanceImplicitGemmBwdV1R1Xdlops::CalculateBlockSize() const
{
    int block_size = 0;

    try
    {
        if(!(GemmMPerBlock % GemmMPerWave == 0 && GemmNPerBlock % GemmNPerWave == 0))
            MIOPEN_THROW("invalid performance parameter");

        const auto WaveSize = 64;
        block_size = (GemmNPerBlock * GemmMPerBlock) / (GemmMPerWave * GemmNPerWave) * WaveSize;
    }
    catch(...)
    {
        return std::make_tuple(-1, false);
    }

    return std::make_tuple(block_size, true);
}

std::tuple<int, bool>
PerformanceImplicitGemmBwdV1R1Xdlops::CalculateGridSize(const ProblemDescription& problem) const
{
    int GridSize = 0;

    try
    {
        int gemm_g = -1;
        int gemm_m = -1;
        int gemm_n = -1;

        std::tie(gemm_g, gemm_m, gemm_n, std::ignore) =
            ConvHipImplicitGemmBwdDataV1R1Xdlops::CalculateGemmSize(problem);

        if(!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0))
            MIOPEN_THROW("invalid performance parameter");

        GridSize = gemm_g * (gemm_m / GemmMPerBlock) * (gemm_n / GemmNPerBlock);
    }
    catch(...)
    {
        return std::make_tuple(-1, false);
    }

    return std::make_tuple(GridSize, true);
}

std::tuple<int, int, int, int, int, bool>
PerformanceImplicitGemmBwdV1R1Xdlops::CalculateGemmABlockCopyPerformanceParameters(
    const ProblemDescription& problem) const
{
    // A tensor shape [GemmG, GemmK, GemmM, GemmKPack]

    int ClusterLengths_GemmK      = -1;
    int ClusterLengths_GemmM      = -1;
    int ClusterLengths_GemmKPack  = -1;
    int SrcDataPerRead_GemmM      = problem.IsFp32() ? amd_buffer_load_max_length<float>()
                                                     : amd_buffer_load_max_length<half_float::half>();
    int DstDataPerWrite_GemmKPack = problem.IsFp32() ? amd_lds_write_max_length<float>()
                                                     : amd_lds_write_max_length<half_float::half>();

    try
    {
        bool valid = false;

        int block_size = -1;

        std::tie(block_size, valid) = CalculateBlockSize();

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        if(!((GemmKPerBlock * GemmMPerBlock * GemmKPack) % block_size == 0))
            MIOPEN_THROW("invalid performance parameter");

        // GemmM is src vector read dimension
        SrcDataPerRead_GemmM = gcd(SrcDataPerRead_GemmM, GemmMPerBlock);

        // calculate threadwise copy size
        const auto data_per_thread_copy =
            std::max(1, (GemmKPerBlock * GemmMPerBlock * GemmKPack) / block_size);

        // SrcDataPerRead bounded by size of threadwise copy
        SrcDataPerRead_GemmM = gcd(SrcDataPerRead_GemmM, data_per_thread_copy);

        const auto data_per_thread_copy_gemmm = SrcDataPerRead_GemmM;
        const auto tmp                        = data_per_thread_copy / data_per_thread_copy_gemmm;

        int data_per_thread_copy_gemmk     = -1;
        int data_per_thread_copy_gemmkpack = -1;

        if(GemmAThreadCopyMoreGemmK)
        {
            data_per_thread_copy_gemmk     = gcd(GemmKPerBlock, tmp);
            data_per_thread_copy_gemmkpack = tmp / data_per_thread_copy_gemmk;
        }
        else
        {
            data_per_thread_copy_gemmkpack = gcd(GemmKPack, tmp);
            data_per_thread_copy_gemmk     = tmp / data_per_thread_copy_gemmkpack;
        }

        // vector write into LDS
        DstDataPerWrite_GemmKPack = gcd(DstDataPerWrite_GemmKPack, data_per_thread_copy_gemmkpack);

        // NOLINTNEXTLINE
        if(!(GemmKPerBlock % data_per_thread_copy_gemmk == 0 &&
             GemmMPerBlock % data_per_thread_copy_gemmm == 0 &&
             GemmKPack % data_per_thread_copy_gemmkpack == 0))
            MIOPEN_THROW("invalid performance parameter");

        ClusterLengths_GemmK     = GemmKPerBlock / data_per_thread_copy_gemmk;
        ClusterLengths_GemmM     = GemmMPerBlock / data_per_thread_copy_gemmm;
        ClusterLengths_GemmKPack = GemmKPack / data_per_thread_copy_gemmkpack;
    }
    catch(...)
    {
        return std::make_tuple(-1, -1, -1, -1, -1, false);
    }

    return std::make_tuple(ClusterLengths_GemmK,
                           ClusterLengths_GemmM,
                           ClusterLengths_GemmKPack,
                           SrcDataPerRead_GemmM,
                           DstDataPerWrite_GemmKPack,
                           true);
}

std::tuple<int, int, int, int, int, bool>
PerformanceImplicitGemmBwdV1R1Xdlops::CalculateGemmBBlockCopyPerformanceParameters(
    const ProblemDescription& problem) const
{
    // B tensor shape [GemmG, GemmK, GemmN, GemmKPack]

    int ClusterLengths_GemmK      = -1;
    int ClusterLengths_GemmN      = -1;
    int ClusterLengths_GemmKPack  = -1;
    int SrcDataPerRead_GemmN      = problem.IsFp32() ? amd_buffer_load_max_length<float>()
                                                     : amd_buffer_load_max_length<half_float::half>();
    int DstDataPerWrite_GemmKPack = problem.IsFp32() ? amd_lds_write_max_length<float>()
                                                     : amd_lds_write_max_length<half_float::half>();

    try
    {
        bool valid = false;

        int block_size = -1;

        std::tie(block_size, valid) = CalculateBlockSize();

        if(!valid)
            MIOPEN_THROW("invalid performance parameter");

        if(!((GemmKPerBlock * GemmNPerBlock * GemmKPack) % block_size == 0))
            MIOPEN_THROW("invalid performance parameter");

        // calculate threadwise copy size
        const auto data_per_thread_copy =
            std::max(1, (GemmKPerBlock * GemmNPerBlock * GemmKPack) / block_size);

        // GemmN is src vector read dimension
        // calculate vector length on gemmn dimension based on global tensor layout
        const auto ho        = ProblemInterpreter::GetOutputHeightHo(problem);
        const auto wo        = ProblemInterpreter::GetOutputWidthWo(problem);
        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, ho * wo);

        // SrcDataPerRead_GemmN bounded by size of threadwise copy
        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, data_per_thread_copy);

        // SrcDataPerRead also bounded by GemmKPack;
        SrcDataPerRead_GemmN = gcd(SrcDataPerRead_GemmN, GemmNPerBlock);

        const auto data_per_thread_copy_gemmn = SrcDataPerRead_GemmN;
        const auto tmp                        = data_per_thread_copy / data_per_thread_copy_gemmn;

        int data_per_thread_copy_gemmkpack = -1;
        int data_per_thread_copy_gemmk     = -1;

        if(GemmBThreadCopyMoreGemmKPack)
        {
            data_per_thread_copy_gemmkpack = gcd(GemmKPack, tmp);
            data_per_thread_copy_gemmk     = tmp / data_per_thread_copy_gemmkpack;
        }
        else
        {
            data_per_thread_copy_gemmk     = gcd(GemmKPerBlock, tmp);
            data_per_thread_copy_gemmkpack = tmp / data_per_thread_copy_gemmk;
        }

        // vector write into LDS
        DstDataPerWrite_GemmKPack = gcd(DstDataPerWrite_GemmKPack, data_per_thread_copy_gemmkpack);

        if(!(GemmKPerBlock % data_per_thread_copy_gemmk == 0 &&
             GemmNPerBlock % data_per_thread_copy_gemmn == 0 &&
             GemmKPack % data_per_thread_copy_gemmkpack == 0))
            MIOPEN_THROW("invalid performance parameter");

        ClusterLengths_GemmK     = GemmKPerBlock / data_per_thread_copy_gemmk;
        ClusterLengths_GemmN     = GemmNPerBlock / data_per_thread_copy_gemmn;
        ClusterLengths_GemmKPack = GemmKPack / data_per_thread_copy_gemmkpack;
    }
    catch(...)
    {
        return std::make_tuple(-1, -1, -1, -1, -1, false);
    }

    return std::make_tuple(ClusterLengths_GemmK,
                           ClusterLengths_GemmN,
                           ClusterLengths_GemmKPack,
                           SrcDataPerRead_GemmN,
                           DstDataPerWrite_GemmKPack,
                           true);
}

// Used by IsReallyValid()
bool PerformanceImplicitGemmBwdV1R1Xdlops::IsValidValue() const
{
    // clang-format off
    return IsTwoPower<4, 256>(GemmMPerBlock)
        && IsTwoPower<4, 256>(GemmNPerBlock)
        && IsTwoPower<1, 8>(GemmKPerBlock)
        && IsTwoPower<4, 128>(GemmMPerWave)
        && IsTwoPower<4, 128>(GemmNPerWave)
        && IsTwoPower<1, 8>(GemmKPack);
    // clang-format on
}

// Used by HeuristicInit()
// Only return false if a performance config will violate requirements given by kernel algorithm
bool PerformanceImplicitGemmBwdV1R1Xdlops::IsReallyValid(const ProblemDescription& problem) const
{
    if(!IsValidValue())
        return false;

    if(!IsValidBlockwiseGemmXdlops(problem,
                                   GemmMPerBlock,
                                   GemmNPerBlock,
                                   GemmKPerBlock,
                                   GemmMPerWave,
                                   GemmNPerWave,
                                   GemmKPack))
        return false;

    bool valid = false;

    // check blockwise GEMM size
    {
        int gemm_m       = -1;
        int gemm_n       = -1;
        int gemm_k_total = -1;

        std::tie(std::ignore, gemm_m, gemm_n, gemm_k_total) =
            ConvHipImplicitGemmBwdDataV1R1Xdlops::CalculateGemmSize(problem);

        if(gemm_k_total % GemmKPack != 0)
            return false;

        const auto gemm_k = gemm_k_total / GemmKPack;

        if(!(gemm_m % GemmMPerBlock == 0 && gemm_n % GemmNPerBlock == 0 &&
             gemm_k % GemmKPerBlock == 0))
            return false;
    }

    // check blockwise copy of A matrix
    {
        std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, valid) =
            CalculateGemmABlockCopyPerformanceParameters(problem);

        if(!valid)
            return false;
    }

    // check blockwise copy of B matrix
    {
        std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, valid) =
            CalculateGemmBBlockCopyPerformanceParameters(problem);

        if(!valid)
            return false;
    }

    // check LDS allocation
    std::size_t lds_size      = 0;
    std::tie(lds_size, valid) = CalculateLdsNumberOfByte(problem);

    return (valid and lds_size <= get_lds_max_number_of_byte());
}

// Used by GenericSearch, not used by HeuristicInit
// Return false if a performance config is known to be sub-optimal, comparing to other performance
// config inside tuning range
bool PerformanceImplicitGemmBwdV1R1Xdlops::IsFastToBeUsedForTuning(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    // somehow, 128x128 wave-wise GEMM tend to spill register
    // TODO revisit this when 128x128 wave-wise GEMM become efficient
    {
        if(GemmMPerWave * GemmNPerWave > 64 * 128)
            return false;
    }

    // don't need too many blocks
    {
        int gemm_m = 0;
        int gemm_n = 0;

        std::tie(std::ignore, gemm_m, gemm_n, std::ignore) =
            ConvHipImplicitGemmBwdDataV1R1Xdlops::CalculateGemmSize(problem);

        // this is grid size using current blockwise-GEMM
        const int grid_size = (gemm_m * gemm_n) / (GemmMPerBlock * GemmNPerBlock);

        // this the the biggest blockwise-GEMM you can do
        int max_blockwise_gemm_size =
            std::max(gcd(256, gemm_m) * gcd(128, gemm_n), gcd(128, gemm_m) * gcd(256, gemm_n));

        // this is the grid size using the biggest blockwise-GEMM
        auto grid_size_max_blockwise_gemm =
            (std::size_t(gemm_m) * gemm_n) / max_blockwise_gemm_size;

        const float ratio = float(grid_size) / grid_size_max_blockwise_gemm;

        const auto num_cu = ctx.GetStream().GetMaxComputeUnits();

        // heuristic to exclude performance paramater that result in very large number of blocks
        if(grid_size_max_blockwise_gemm > 5 * num_cu)
        {
            if(ratio > 2.81)
                return false;
        }
        else if(grid_size_max_blockwise_gemm > 4 * num_cu)
        {
            if(ratio > 3.61)
                return false;
        }
        else if(grid_size_max_blockwise_gemm > 3 * num_cu)
        {
            if(ratio > 4.41)
                return false;
        }
        else if(grid_size_max_blockwise_gemm > 2 * num_cu)
        {
            if(ratio > 6.41)
                return false;
        }
        else if(grid_size_max_blockwise_gemm > num_cu)
        {
            if(ratio > 12.41)
                return false;
        }
    }

    // don't need too many waves per block
    {
        const int wave_per_block = (GemmMPerBlock / GemmMPerWave) * (GemmNPerBlock / GemmNPerWave);

        if(!(wave_per_block > 1 && wave_per_block <= 4))
        {
            return false;
        }
    }

    // avoid skinny blockwise GEMM whenever possible
    {
        int gemm_m = 0;
        int gemm_n = 0;

        std::tie(std::ignore, gemm_m, gemm_n, std::ignore) =
            ConvHipImplicitGemmBwdDataV1R1Xdlops::CalculateGemmSize(problem);

        if(GemmMPerBlock > 2 * GemmNPerBlock)
        {
            if(gemm_n % (2 * GemmNPerBlock) == 0)
                return false;
        }

        if(GemmNPerBlock > 2 * GemmMPerBlock)
        {
            if(gemm_m % (2 * GemmMPerBlock) == 0)
                return false;
        }
    }

    // avoid skinny wavewise GEMM whenever possible
    {
        if(GemmMPerWave > 2 * GemmNPerWave)
        {
            if(GemmNPerBlock % (2 * GemmNPerWave) == 0)
                return false;
        }

        if(GemmNPerWave > 2 * GemmMPerWave)
        {
            if(GemmMPerBlock % (2 * GemmMPerWave) == 0)
                return false;
        }
    }

    // each thread should not too much data
    {
        const int block_size = (GemmMPerBlock / GemmMPerWave) * (GemmNPerBlock / GemmNPerWave) * 64;

        const int a_data_per_thread_copy = (GemmKPerBlock * GemmMPerBlock * GemmKPack) / block_size;
        const int b_data_per_thread_copy = (GemmKPerBlock * GemmNPerBlock * GemmKPack) / block_size;

        if(problem.IsFp32())
        {
            if(a_data_per_thread_copy > 16 || b_data_per_thread_copy > 16)
                return false;
        }
        else if(problem.IsFp16() || problem.IsBfp16())
        {
            if(a_data_per_thread_copy > 32 || b_data_per_thread_copy > 32)
                return false;
        }
    }

    return true;
}

// Used by GenericSearch, not used by HeuristicInit
// Return false, if you don't want to this to be included in tuning range used by generic search
// A performance config may still be valid w.r.t algorithm correctness, even when IsValid() return
// false
bool PerformanceImplicitGemmBwdV1R1Xdlops::IsValid(const ExecutionContext& ctx,
                                                   const ProblemDescription& problem) const
{
    return IsReallyValid(problem) && IsFastToBeUsedForTuning(ctx, problem);
}

// Used by GenericSearch, not used by HeuristicInit
bool ConvHipImplicitGemmBwdDataV1R1Xdlops::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription& problem,
    const PerformanceImplicitGemmBwdV1R1Xdlops& config) const
{
    return config.IsReallyValid(problem);
}

std::tuple<int, int, int, int>
ConvHipImplicitGemmBwdDataV1R1Xdlops::CalculateGemmSize(const ProblemDescription& problem)
{
    const auto g  = ProblemInterpreter::GetGroupCountG(problem);
    const auto n  = ProblemInterpreter::GetBatchN(problem);
    const auto k  = ProblemInterpreter::GetOutputChannelK(problem);
    const auto c  = ProblemInterpreter::GetInputChannelC(problem);
    const auto ho = ProblemInterpreter::GetOutputHeightHo(problem);
    const auto wo = ProblemInterpreter::GetOutputWidthWo(problem);
    const auto y  = ProblemInterpreter::GetFilterHeightY(problem);
    const auto x  = ProblemInterpreter::GetFilterWidthX(problem);

    const auto k_per_group = k / g;
    const auto c_per_group = c / g;

    const auto gemm_g       = g;
    const auto gemm_m       = c_per_group * y * x;
    const auto gemm_n       = n * ho * wo;
    const auto gemm_k_total = k_per_group;

    return std::make_tuple(gemm_g, gemm_m, gemm_n, gemm_k_total);
}

PerformanceImplicitGemmBwdV1R1Xdlops
ConvHipImplicitGemmBwdDataV1R1Xdlops::GetDefaultPerformanceConfig(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmBwdV1R1Xdlops>(ctx, problem);
}

std::tuple<std::size_t, bool> PerformanceImplicitGemmBwdV1R1Xdlops::CalculateLdsNumberOfByte(
    const ProblemDescription& problem) const
{
    const auto a_block_space = GemmKPerBlock * GemmMPerBlock * GemmKPack;
    const auto b_block_space = GemmKPerBlock * GemmNPerBlock * GemmKPack;

    std::size_t lds_size = (a_block_space + b_block_space) *
                           (problem.IsFp32() ? sizeof(float) : sizeof(half_float::half));

    return std::make_tuple(lds_size, true);
}

std::size_t
ConvHipImplicitGemmBwdDataV1R1Xdlops::GetWorkspaceSize(const ExecutionContext&,
                                                       const ProblemDescription& problem) const
{
    if(problem.IsFp32())
    {
        return 0;
    }
    else
    {
        const auto y          = ProblemInterpreter::GetFilterHeightY(problem);
        const auto x          = ProblemInterpreter::GetFilterWidthX(problem);
        const auto stride_h   = ProblemInterpreter::GetAdjustedConvolutionStrideH(problem);
        const auto stride_w   = ProblemInterpreter::GetAdjustedConvolutionStrideW(problem);
        const auto dilation_h = ProblemInterpreter::GetAdjustedConvolutionDilationH(problem);
        const auto dilation_w = ProblemInterpreter::GetAdjustedConvolutionDilationW(problem);

        if((stride_h >= dilation_h * (y - 1) + 1) && (stride_w >= dilation_w * (x - 1) + 1))
        {
            return 0;
        }
        else
        {
            // In case of fp16/bfp16, because there is no atomic add ISA,
            // reduction via atomic add ISA is done via fp32. As a result,
            // workspace is computed with miopenFloat data type.
            // Later, a separate kernel is invoked that casts from fp32 to fp16/bfp16
            std::size_t n  = ProblemInterpreter::GetBatchN(problem);
            std::size_t c  = ProblemInterpreter::GetInputChannelC(problem);
            std::size_t hi = ProblemInterpreter::GetInputHeightHi(problem);
            std::size_t wi = ProblemInterpreter::GetInputWidthWi(problem);
            return n * c * hi * wi * miopen::GetTypeSize(miopenFloat);
        }
    }
}

bool ConvHipImplicitGemmBwdDataV1R1Xdlops::IsApplicable(const ExecutionContext& ctx,
                                                        const ProblemDescription& problem) const
{
#if WORKAROUND_SWDEV_251757
    if(!env::enabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS))
        return false;
#endif
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS))
        return false;

    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;

    if(problem.GetConv().attribute.deterministic)
        return false;

    if(!IsComposableKernelSupportedHardware(ctx))
        return false;

    if(!ctx.use_hip_kernels)
        return false;

    if(!IsXdlopsSupport(ctx))
        return false;

    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16()))
        return false;

    if(!problem.IsDirectionBackwardData())
        return false;

    if(!problem.Is2d())
        return false;

    if(problem.HasNonPackedTensors())
        return false;

    if(!problem.AllTensorsDimsFitIntoInt())
        return false;

    if(problem.IsTensorsCasted())
        return false;

    if(ctx.GetStream().GetDeviceName() == "gfx90a" && problem.IsGfx90aFp16altRequired())
        return false;

    if(!IsIndexRangeLargeEnough(problem))
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    // gemm size
    int gemm_g       = -1;
    int gemm_m       = -1;
    int gemm_n       = -1;
    int gemm_k_total = -1;

    std::tie(gemm_g, gemm_m, gemm_n, gemm_k_total) = CalculateGemmSize(problem);

    return IsValidGridGemmXdlops(gemm_m, gemm_n, gemm_k_total);
}

PerformanceImplicitGemmBwdV1R1Xdlops
ConvHipImplicitGemmBwdDataV1R1Xdlops::Search(const ExecutionContext& ctx,
                                             const ProblemDescription& problem,
                                             const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution ConvHipImplicitGemmBwdDataV1R1Xdlops::GetSolution(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceImplicitGemmBwdV1R1Xdlops& config) const
{
    ConvSolution result;

    if(!config.IsReallyValid(problem))
    {
        MIOPEN_LOG_E("invalid performance parameter");
        assert(false);
    }

    KernelInfo construction_parameters;

    // clang-format off
    construction_parameters.kernel_file =
        "static_kernel_gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw.cpp";

    construction_parameters.kernel_name =
        "gridwise_convolution_backward_data_implicit_gemm_v1r1_xdlops_nchw_kcyx_nkhw";
    // clang-format on

    result.workspace_sz = GetWorkspaceSize(ctx, problem);

    int grid_size  = 0;
    int block_size = 0;

    std::tie(grid_size, std::ignore)  = config.CalculateGridSize(problem);
    std::tie(block_size, std::ignore) = config.CalculateBlockSize();

    construction_parameters.l_wk.push_back(block_size);
    construction_parameters.l_wk.push_back(1);
    construction_parameters.l_wk.push_back(1);

    construction_parameters.g_wk.push_back(static_cast<size_t>(block_size) * grid_size);
    construction_parameters.g_wk.push_back(1);
    construction_parameters.g_wk.push_back(1);

    int GemmABlockCopyClusterLengths_GemmK      = -1;
    int GemmABlockCopyClusterLengths_GemmM      = -1;
    int GemmABlockCopyClusterLengths_GemmKPack  = -1;
    int GemmABlockCopySrcDataPerRead_GemmM      = -1;
    int GemmABlockCopyDstDataPerWrite_GemmKPack = -1;

    int GemmBBlockCopyClusterLengths_GemmK      = -1;
    int GemmBBlockCopyClusterLengths_GemmN      = -1;
    int GemmBBlockCopyClusterLengths_GemmKPack  = -1;
    int GemmBBlockCopySrcDataPerRead_GemmN      = -1;
    int GemmBBlockCopyDstDataPerWrite_GemmKPack = -1;

    std::tie(GemmABlockCopyClusterLengths_GemmK,
             GemmABlockCopyClusterLengths_GemmM,
             GemmABlockCopyClusterLengths_GemmKPack,
             GemmABlockCopySrcDataPerRead_GemmM,
             GemmABlockCopyDstDataPerWrite_GemmKPack,
             std::ignore) = config.CalculateGemmABlockCopyPerformanceParameters(problem);

    std::tie(GemmBBlockCopyClusterLengths_GemmK,
             GemmBBlockCopyClusterLengths_GemmN,
             GemmBBlockCopyClusterLengths_GemmKPack,
             GemmBBlockCopySrcDataPerRead_GemmN,
             GemmBBlockCopyDstDataPerWrite_GemmKPack,
             std::ignore) = config.CalculateGemmBBlockCopyPerformanceParameters(problem);

    // clang-format off
    construction_parameters.comp_options =
        std::string(" -DCK_PARAM_PROBLEM_G=") + std::to_string(ProblemInterpreter::GetGroupCountG(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(ProblemInterpreter::GetBatchN(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ProblemInterpreter::GetOutputChannelK(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ProblemInterpreter::GetInputChannelC(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ProblemInterpreter::GetInputHeightHi(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ProblemInterpreter::GetInputWidthWi(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ProblemInterpreter::GetOutputHeightHo(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ProblemInterpreter::GetOutputWidthWo(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_Y=") + std::to_string(ProblemInterpreter::GetFilterHeightY(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_X=") + std::to_string(ProblemInterpreter::GetFilterWidthX(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(ProblemInterpreter::GetAdjustedConvolutionStrideH(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_H=") + std::to_string(ProblemInterpreter::GetAdjustedConvolutionDilationH(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_W=") + std::to_string(ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_IN_LEFT_PAD_H=") + std::to_string(ProblemInterpreter::GetInputLeftPadH(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_IN_LEFT_PAD_W=") + std::to_string(ProblemInterpreter::GetInputLeftPadW(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_H=") + std::to_string(ProblemInterpreter::GetAdjustedInputRightPadH(problem)) +
        std::string(" -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_W=") + std::to_string(ProblemInterpreter::GetAdjustedInputRightPadW(problem)) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_BLOCK=") + std::to_string(config.GemmMPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_BLOCK=") + std::to_string(config.GemmNPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_K_PER_BLOCK=") + std::to_string(config.GemmKPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_WAVE=") + std::to_string(config.GemmMPerWave) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_WAVE=") + std::to_string(config.GemmNPerWave) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_KPACK=") + std::to_string(config.GemmKPack) +
        std::string(" -DCK_PARAM_DEPENDENT_BLOCK_SIZE=") + std::to_string(block_size) +
        std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
        std::string(" -DCK_PARAM_DEPENDENT_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(GemmABlockCopyClusterLengths_GemmK) +
        std::string(" -DCK_PARAM_DEPENDENT_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M=") + std::to_string(GemmABlockCopyClusterLengths_GemmM) +
        std::string(" -DCK_PARAM_DEPENDENT_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_KPACK=") + std::to_string(GemmABlockCopyClusterLengths_GemmKPack) +
        std::string(" -DCK_PARAM_DEPENDENT_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_M=") + std::to_string(GemmABlockCopySrcDataPerRead_GemmM) +
        std::string(" -DCK_PARAM_DEPENDENT_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") + std::to_string(GemmABlockCopyDstDataPerWrite_GemmKPack) +
        std::string(" -DCK_PARAM_DEPENDENT_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(GemmBBlockCopyClusterLengths_GemmK) +
        std::string(" -DCK_PARAM_DEPENDENT_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=") + std::to_string(GemmBBlockCopyClusterLengths_GemmN) +
        std::string(" -DCK_PARAM_DEPENDENT_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_KPACK=") + std::to_string(GemmBBlockCopyClusterLengths_GemmKPack) +
        std::string(" -DCK_PARAM_DEPENDENT_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N=") + std::to_string(GemmBBlockCopySrcDataPerRead_GemmN) +
        std::string(" -DCK_PARAM_DEPENDENT_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") + std::to_string(GemmBBlockCopyDstDataPerWrite_GemmKPack) +
        std::string(" -DCK_USE_AMD_XDLOPS=") + std::to_string(IsXdlopsSupport(ctx) ? 1 : 0) +
        std::string(" -DCK_USE_AMD_XDLOPS_INLINE_ASM=") + (env::enabled(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM) ? '1' : '0') +
        std::string(" -DCK_USE_AMD_XDLOPS_EMULATE=") + (env::enabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE) ? '1' : '0') +
        get_static_ck_common_compiler_flag(ctx) +
        ctx.general_compile_options;

    result.invoker_factory = miopen::conv::MakeImplGemmDataInvokerFactory(problem);
    result.construction_params.push_back(construction_parameters);
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
