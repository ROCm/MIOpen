/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#define CONV_OCL_DIR2D_BWDWRW_2_CPP

#include <miopen/conv/solvers.hpp>

#include <miopen/conv/invokers/ocl_wrw_rdc.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/bfloat16.hpp>
#include <miopen/visit_float.hpp>

#include <algorithm>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2_SEARCH_OPTIMIZED)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

inline static bool Is_1_to_8(const int& v)
{
    // full: {1,2,4,8}, optimized: {1,3,8}
    switch(v)
    {
    case 1:
    case 2:
    case 3:
    case 4:
    case 8: return true;
    default: return false;
    }
}

inline static bool Inc_1_to_8(int& v)
{
    assert(Is_1_to_8(v));
    if(v == 8)
    {
        v = 1;
        return true;
    }
    v = v * 2;
    return false;
}

inline static bool Inc_1_to_8_optimized(int& v)
{
    assert(Is_1_to_8(v));
    switch(v)
    {
    case 1: v = 3; return false;
    case 3: v = 8; return false;
    default:
    case 8: v = 1; return true;
    }
}

inline static bool Is_6_to_12(const int& v) { return 6 <= v && v <= 12; }

inline static bool Inc_6_to_12(int& v)
{
    assert(Is_6_to_12(v));
    if(++v <= 12)
        return false;
    v = 6;
    return true;
}

inline static bool Inc_6_to_12_optimized(int& v)
{
    assert(Is_6_to_12(v));
    // {6,8,10,12}, {7,9,11}...
    switch(v)
    {
    case 12: v = 7; return true;
    case 11: v = 6; return true;
    default: v += 2; return false; // 6,8,10,7,9
    }
}

inline static bool Is_2_to_11(const int& v) { return 2 <= v && v <= 11; }

inline static bool Inc_2_to_11(int& v)
{
    assert(Is_2_to_11(v));
    if(++v <= 11)
        return false;
    v = 2;
    return true;
}

inline static bool Inc_2_to_11_optimized(int& v)
{
    // {2 3 5 7 9 11}
    assert(Is_2_to_11(v));
    switch(v)
    {
    case 2: v = 3; return false;
    default: v += 2; return false;
    case 11: v = 2; return true;
    }
}

// Workaround for issue 1185.
// OpenCL fails to allocate a single piece of GPU memory, if the require mem size is too large
// However, the limit of mem size is unknown currently, and has been raised as a remaining question
// in issue 1289
#define WORKAROUND_ISSUE_1185 1

static bool IsTunableBase(const ProblemDescription& problem)
{
    return !(problem.GetGroupCount() == 1 &&
             ((problem.GetWeightsWidth() == 3 && problem.GetWeightsHeight() == 3) ||
              (problem.GetWeightsWidth() == 1 && problem.GetWeightsHeight() == 1)));
}

bool ConvOclBwdWrW2NonTunable::IsApplicable(const ExecutionContext& ctx,
                                            const ProblemDescription& problem) const
{
    // At present, auto-tuning is disabled for non-group 3x3 and 1x1 filters for multiple
    // reasons: after tuning ocl kernel for 3x3 and 1x1 filters, assembly kernel still
    // dominates. Thus, this solver is used for non-group 3x3 and 1x1 filters only.
    const auto tunable = ConvOclBwdWrW2<1>{};
    return tunable.IsApplicableBase(ctx, problem) && !IsTunableBase(problem);
}

size_t ConvOclBwdWrW2NonTunable::GetWorkspaceSize(const ExecutionContext& ctx,
                                                  const ProblemDescription& problem) const
{
    const auto tunable = ConvOclBwdWrW2<1>{};
    return tunable.GetWorkspaceSize(ctx, problem);
}

ConvSolution ConvOclBwdWrW2NonTunable::GetSolution(const ExecutionContext& ctx,
                                                   const ProblemDescription& problem) const
{
    // Invoking base class GetSolution with default values for params obtained
    // from GetDefaultPerformanceConfig()
    const auto tunable = ConvOclBwdWrW2<1>{};
    return tunable.GetSolution(ctx, problem, tunable.GetDefaultPerformanceConfig(ctx, problem));
}

template <int N_BATCH_LOOPS>
bool PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>::operator==(
    const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& other) const
{
    // clang-format off
    return n_waves == other.n_waves
        && read_size == other.read_size
        && n_out_channels_per_tile == other.n_out_channels_per_tile
        && n_out_channels_tiles == other.n_out_channels_tiles
        && n_out_rows_in_lcl == other.n_out_rows_in_lcl; // clang-format on
}

template <int N_BATCH_LOOPS>
bool PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>::SetNextValue(const ProblemDescription&)
{
    // Increment with wrap-around:
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2_SEARCH_OPTIMIZED))
    {
        do
        {
            if(!Inc_1_to_8(n_waves))
                break;
            if(!Inc_6_to_12(read_size))
                break;
            if(!Inc_1_to_8(n_out_channels_per_tile))
                break;
            if(!Inc_1_to_8(n_out_channels_tiles))
                break;
            if(!Inc_2_to_11(n_out_rows_in_lcl))
                break;
            return false;
        } while(false);
    }
    else
    {
        do
        {
            if(!Inc_1_to_8_optimized(n_waves))
                break;
            if(!Inc_6_to_12_optimized(read_size))
                break;
            if(!Inc_1_to_8_optimized(n_out_channels_per_tile))
                break;
            if(!Inc_1_to_8_optimized(n_out_channels_tiles))
                break;
            if(!Inc_2_to_11_optimized(n_out_rows_in_lcl))
                break;
            return false;
        } while(false);
    }
    return true;
}

template <int N_BATCH_LOOPS>
bool PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>::IsValidValue() const
{
    // clang-format off
    return Is_1_to_8(n_waves)
        && Is_6_to_12(read_size)
        && Is_1_to_8(n_out_channels_per_tile)
        && Is_1_to_8(n_out_channels_tiles)
        && Is_2_to_11(n_out_rows_in_lcl); // clang-format on
}

static const int N_STACKS = 1; // number  of batch iterations

template <int N_BATCH_LOOPS>
static size_t GetNBatchBlks(const ProblemDescription& problem)
{
    return std::ceil(static_cast<float>(problem.GetBatchSize()) / (N_BATCH_LOOPS * N_STACKS));
}

template <int N_BATCH_LOOPS>
bool PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>::IsValid(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    if(!IsValidValue())
    {
        return false;
    }

    ConvSolution result;
    result.n_in_data_tiles = 1;

    // Check 1: n_back_loops
    // Ensure that the total amount of system memory used by intermediate object
    // that holds the weights of x number of batches doesn't exceed system memory
    size_t wei_cstride = problem.GetWeightsHeight() * problem.GetWeightsWidth();
    size_t wei_bstride = (problem.GetOutChannels() / problem.GetGroupCount()) * wei_cstride;

    // number  of batch iterations
    const size_t n_batch_blks = GetNBatchBlks<N_BATCH_LOOPS>(problem);

    // guard not to grab too much system memory
    if(n_batch_blks < 1 || (wei_bstride * problem.GetInChannels() * n_batch_blks) >
                               ctx.GetStream().GetMaxMemoryAllocSize())
    {
        return false;
    }

    // Check 2: read size
    if(problem.GetInWidth() < read_size)
    {
        return false;
    }
    size_t aligned_out_scan_lane =
        std::ceil(static_cast<float>(problem.GetInWidth()) / read_size); // image aligned scan

    // Check 3: n_out_channels_tiles
    if(problem.GetGroupCount() > 1 && n_out_channels_tiles > 1)
    {
        return false;
    }

    size_t n_output_channels_per_group = problem.GetInChannels() / problem.GetGroupCount();

    // Check 4: n_out_channels_per_tile
    if(problem.GetGroupCount() > 1 && n_out_channels_per_tile > n_output_channels_per_group)
    {
        return false;
    }

    // group config requires n_out_channels_tiles to be 1 or else
    // kernel doesn't work.
    if(problem.GetGroupCount() > 1 && n_out_channels_tiles != 1)
    {
        return false;
    }

    size_t total_out_channels =
        static_cast<std::size_t>(n_out_channels_tiles) * n_out_channels_per_tile;
    if(total_out_channels > n_output_channels_per_group)
    {
        return false;
    }

    if(n_out_rows_in_lcl < problem.GetWeightsHeight())
    {
        return false;
    }

    // Check 5: n_out_rows_in_lcl  should exceed LDS limit
    size_t in_lcl_height =
        static_cast<std::size_t>(n_out_rows_in_lcl - 1) * problem.GetKernelStrideH() +
        problem.GetWeightsHeight();
    size_t in_lcl_sz = 0;
    {
        // Chao: Reserve space in LDS for left padding, it also reserve
        //   enough space in LDS for right padding and the footprint of scaning (from
        //   device into LDS). To reduces LDS consumption, right padding of one row
        //   is overlapped with the left padding of the next row.
        //   Also, for the last row, right padding is needed.
        // Revisit this if encounter failure
        size_t in_lcl_width = 0;
        size_t in_width     = problem.GetOutWidth(); // out is in, in is out
        size_t out_width    = problem.GetInWidth();

        size_t in_lcl_width_effective = std::max<size_t>(
            in_width + 2ULL * problem.GetPadW(),
            std::max(problem.GetPadW() + ((in_width + read_size - 1) / read_size) * read_size,
                     problem.GetWeightsWidth() + (out_width - 1) * problem.GetKernelStrideW()));

        size_t in_lcl_width_right_buffer = std::max(
            static_cast<int>(in_lcl_width_effective - (in_width + 2ULL * problem.GetPadW())), 0);

        in_lcl_width = problem.GetPadW() + in_width + in_lcl_width_right_buffer;

        // Chao: attempt to reduce LDS bank conflict during reading input image from LDS
        // Revisit this if performance regress
        if(problem.GetOutDataType() == miopenFloat)
        {
            in_lcl_width = (in_lcl_width / 2) * 2 + 1;
        }

        // make enough room for right padding and buffer for the last row in LDS
        in_lcl_sz = in_lcl_width * in_lcl_height + problem.GetPadW() + in_lcl_width_right_buffer;
    }

    // check LDS consumption
    size_t wei_per_wkitem = (problem.GetWeightsWidth() <= 7 ||
                             (((problem.GetWeightsWidth() / 2) * 2) != problem.GetWeightsWidth()))
                                ? problem.GetWeightsWidth()
                                : problem.GetWeightsWidth() / 2;

    {
        size_t n_lcl_batchs   = N_STACKS;
        size_t n_lcl_in_maps  = result.n_in_data_tiles;
        size_t n_lcl_out_maps = n_out_channels_tiles;

        // LDS used for "bot"
        size_t total_in_lcl_sz = in_lcl_sz * n_lcl_batchs * n_lcl_in_maps;

        // LDS used for workgroup-level reduction of weights
        size_t wei_lcl_sz        = 0;
        size_t max_wei_blk       = 0;
        size_t out_wei_scan_loop = 0;
        size_t out_width         = problem.GetInWidth(); // out is in, in is out

        {
            const auto hw_wave_size   = 64; // TBD Obtain this from handle.
            const auto workgroup_size = hw_wave_size * n_waves;

            if(wei_per_wkitem == 0)
                return false;
            size_t wei_blk_sz0 =
                std::ceil(static_cast<float>(problem.GetWeightsWidth()) / wei_per_wkitem);
            size_t wei_blk_sz = problem.GetWeightsHeight() * wei_blk_sz0;
            if(wei_blk_sz == 0)
                return false;
            size_t n_wei_blk = workgroup_size / wei_blk_sz;
            if(n_wei_blk == 0)
            { /// \todo This is quickfix for DIV/0, see ROCm/MIOpen/issues/70.
                MIOPEN_LOG_I2("ConvOClBwdWrW2: GRP_SZ < wei_blk_sz, not applicable?");
                return false;
            }
            out_wei_scan_loop = std::ceil(static_cast<float>(out_width) / n_wei_blk);
            if(out_wei_scan_loop == 0)
                return false;

            max_wei_blk = std::min(
                n_wei_blk,
                static_cast<size_t>(std::ceil(static_cast<float>(out_width) / out_wei_scan_loop)));
            wei_lcl_sz = wei_blk_sz * wei_per_wkitem * max_wei_blk;
        }
        size_t total_wei_lcl_sz = wei_lcl_sz * n_lcl_in_maps * n_lcl_out_maps;

        // LDS use for "top_df"
        size_t out_horiz_pix_ext_sz =
            std::max(out_wei_scan_loop * max_wei_blk, aligned_out_scan_lane * read_size);
        size_t total_out_lcl_sz =
            (n_out_rows_in_lcl * out_horiz_pix_ext_sz) * n_lcl_batchs * n_lcl_out_maps;

        size_t total_lcl_mem_sz = std::max(total_in_lcl_sz + total_out_lcl_sz, total_wei_lcl_sz) *
                                  GetTypeSize(problem.GetOutDataType());

        const auto lds_size = 64 * 1024; // TBD Obtain this from handle.
        if(total_lcl_mem_sz > lds_size)
        {
            return false;
        }
    }

    if(n_batch_blks > 1)
    {
        size_t data_len     = GetTypeSize(problem.GetOutDataType());
        result.workspace_sz = wei_bstride * problem.GetInChannels() * n_batch_blks * data_len;

#if WORKAROUND_ISSUE_1185
        if(result.workspace_sz >
           (std::size_t(6) * std::size_t(1024) * std::size_t(1024) * std::size_t(1024)))
        {
            return false;
        }
#endif
    }

    return true;
}

template <int N_BATCH_LOOPS>
void PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>::HeuristicInit(
    const ProblemDescription& problem)
{
    n_waves                                = 1;
    read_size                              = 6;
    const auto n_output_channels_per_group = problem.GetInChannels() / problem.GetGroupCount();
    // NOLINTBEGIN(*-braces-around-statements)
    if(n_output_channels_per_group % 4 == 0)
        n_out_channels_per_tile = 4;
    else if(n_output_channels_per_group % 3 == 0)
        n_out_channels_per_tile = 3;
    else if(n_output_channels_per_group % 2 == 0)
        n_out_channels_per_tile = 2;
    else
        n_out_channels_per_tile = 1;
    // NOLINTEND(*-braces-around-statements)
    n_out_channels_tiles = 1;
    n_out_rows_in_lcl    = problem.GetWeightsHeight();
}

template <int N_BATCH_LOOPS>
bool ConvOclBwdWrW2<N_BATCH_LOOPS>::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& config) const
{
    return config.IsValidValue() && config.IsValid(ctx, problem);
}

template <int N_BATCH_LOOPS>
bool ConvOclBwdWrW2<N_BATCH_LOOPS>::IsApplicableBase(const ExecutionContext& ctx,
                                                     const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2))
        return false;
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;
    if(!ctx.use_opencl_convolutions)
        return false;
    if(!problem.Is2d())
        return false;
    if(!problem.IsDirectionBackwardWrW())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsAsymmetricPadH() || problem.IsAsymmetricPadW())
        return false;
    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16()))
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    if(problem.IsTensorsCasted())
        return false;

    return problem.GetDilationW() == 1 && problem.GetDilationH() == 1 &&
#if 0
           // There is a stronger restriction than this one, which make this one unnecessary.
           // The kernel read stripes (in height direction, one stripe at a time) of input into LDS,
           // the height of stripe is (MLO_N_ALIGNED_OUT_SCAN_BLK - 1) * MLO_FILTER_STRIDE1 +
           // MLO_FILTER_SIZE1, (MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1) of it is reusable from
           // previous read, (MLO_N_ALIGNED_OUT_SCAN_BLK * MLO_FILTER_STRIDE1) of it is fresh read
           // from device memory. So (MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1) need no less than 0.
           // TODO: chao: revisit this if failure is encountered.
           problem.GetWeightsHeight() >= problem.GetKernelStrideH() &&
#endif

           // The first scan of stripe of the input into LDS will read a strip of height
           // (kernel_size_h - kernel_stride_h), this stripe should include the whole lower bound
           // padding, as well as some or none of the input.
           static_cast<int>(problem.GetWeightsHeight()) - problem.GetKernelStrideH() >=
               problem.GetPadH() &&
           problem.GetBatchSize() >= N_BATCH_LOOPS &&
           /// \todo Workaround for issue 1693
           !(problem.GetWeightsWidth() >= 8 && problem.GetWeightsWidth() % 2 == 0 &&
             !( // Allow these configs to avoid perf drops:
                 (problem.GetKernelStrideH() == 2 && problem.GetKernelStrideW() == 2) &&
                 (problem.GetWeightsHeight() == 5 &&
                  (problem.GetWeightsWidth() == 10 || problem.GetWeightsWidth() == 20)) &&
                 ((problem.GetOutHeight() == 79 && problem.GetOutWidth() == 341) ||
                  (problem.GetOutHeight() == 161 && problem.GetOutWidth() == 700)))) &&
           /// Avoid LDS & Workspace over-allocation.
           /// \note Required LDS depends on PerformanceConfig.
           /// We use the default PerformanceConfig here. This guarantees that at least
           /// one config will pass the LDS constraint check during auto-tuning.
           /// This works also for non-tunable solver.
           IsValidPerformanceConfig(ctx, problem, GetDefaultPerformanceConfig(ctx, problem));
}

template <int N_BATCH_LOOPS>
bool ConvOclBwdWrW2<N_BATCH_LOOPS>::IsApplicable(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem) const
{
    return IsApplicableBase(ctx, problem) && IsTunableBase(problem);
}

template <int N_BATCH_LOOPS>
PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>
ConvOclBwdWrW2<N_BATCH_LOOPS>::GetDefaultPerformanceConfig(const ExecutionContext&,
                                                           const ProblemDescription& problem) const
{
    PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS> pp;
    pp.HeuristicInit(problem);
    return pp;
}

template <int N_BATCH_LOOPS>
size_t ConvOclBwdWrW2<N_BATCH_LOOPS>::GetWorkspaceSize(const ExecutionContext&,
                                                       const ProblemDescription& problem) const
{
    const size_t n_batch_blks = GetNBatchBlks<N_BATCH_LOOPS>(problem);
    if(n_batch_blks > 1)
    {
        const auto n_input_channels_per_group = problem.GetOutChannels() / problem.GetGroupCount();
        const auto wei_cstride = problem.GetWeightsWidth() * problem.GetWeightsHeight();
        const auto wei_bstride = n_input_channels_per_group * wei_cstride;
        const auto data_len    = GetTypeSize(problem.GetOutDataType());
        return wei_bstride * problem.GetInChannels() * n_batch_blks * data_len;
    }
    else
        return 0;
}

template <int N_BATCH_LOOPS>
ConvSolution ConvOclBwdWrW2<N_BATCH_LOOPS>::GetSolution(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& config) const
{
    ConvSolution result;
    const auto hw_wave_size   = 64;
    const auto workgroup_size = hw_wave_size * config.n_waves;

    const int n_input_channels_per_group  = problem.GetOutChannels() / problem.GetGroupCount();
    const int n_output_channels_per_group = problem.GetInChannels() / problem.GetGroupCount();
    const int wei_cstride                 = problem.GetWeightsWidth() * problem.GetWeightsHeight();
    const auto wei_bstride                = n_input_channels_per_group * wei_cstride;

    result.n_in_data_tiles    = 1;
    const size_t n_batch_blks = GetNBatchBlks<N_BATCH_LOOPS>(problem);
    size_t total_out_maps     = config.n_out_channels_per_tile * config.n_out_channels_tiles;
    size_t wei_per_wkitem     = (problem.GetWeightsWidth() <= 7 ||
                             (((problem.GetWeightsWidth() / 2) * 2) != problem.GetWeightsWidth()))
                                    ? problem.GetWeightsWidth()
                                    : problem.GetWeightsWidth() / 2;

    // each wave is a filter row
    std::string READ_TYPE =
        (config.read_size == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((config.read_size));
    size_t aligned_out_scan_lane = std::ceil(static_cast<float>(problem.GetInWidth()) /
                                             config.read_size); // image aligned scan
    size_t n_out_blk =
        std::ceil(static_cast<float>(problem.GetInHeight()) / config.n_out_rows_in_lcl);
    size_t in_lcl_height =
        (config.n_out_rows_in_lcl - 1) * problem.GetKernelStrideH() + problem.GetWeightsHeight();
    size_t in_lcl_width = 0;
    size_t in_lcl_sz    = 0;
    {
        // Chao: Reserve space in LDS for left padding, it also reserve
        //   enough space in LDS for right padding and the footprint of scaning (from
        //   device into LDS). To reduces LDS consumption, right padding of one row
        //   is overlapped with the left padding of the next row.
        //   Also, for the last row, right padding is needed.
        // Revisit this if encounter failure
        size_t in_width  = problem.GetOutWidth(); // out is in, in is out
        size_t out_width = problem.GetInWidth();

        size_t in_lcl_width_effective = std::max(
            in_width + 2 * static_cast<size_t>(problem.GetPadW()),
            std::max(
                static_cast<size_t>(problem.GetPadW()) +
                    static_cast<size_t>(std::ceil(static_cast<float>(in_width) / config.read_size) *
                                        config.read_size),
                problem.GetWeightsWidth() + (out_width - 1) * problem.GetKernelStrideW()));

        size_t in_lcl_width_right_buffer = std::max(
            static_cast<int>(in_lcl_width_effective - (in_width + 2ULL * problem.GetPadW())), 0);

        in_lcl_width = problem.GetPadW() + in_width + in_lcl_width_right_buffer;

        // Chao: attempt to reduce LDS bank conflict during reading input image from LDS
        // Revisit this if performance regress
        if(problem.GetOutDataType() == miopenFloat)
        {
            in_lcl_width = (in_lcl_width / 2) * 2 + 1;
        }

        // make enough room for right padding and buffer for the last row in LDS
        in_lcl_sz = in_lcl_width * in_lcl_height + problem.GetPadW() + in_lcl_width_right_buffer;
    }

    size_t out_n_pixels_off =
        problem.GetInWidth() - (problem.GetInWidth() / config.read_size) * config.read_size;

    result.grp_tile0       = workgroup_size;
    result.grp_tile1       = 1;
    unsigned int grp_tile2 = 1;

    // utility parameters
    unsigned int n_utility_waves        = 4;
    unsigned int utility_workgroup_size = hw_wave_size * n_utility_waves;
    unsigned int utility_read_unit      = ((wei_cstride / 4) * 4 == wei_cstride)   ? 4
                                          : ((wei_cstride / 2) * 2 == wei_cstride) ? 2
                                                                                   : 1;
    std::string UT_READ_TYPE =
        (utility_read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((utility_read_unit));

    if(!problem.IsDirectionBackwardWrW())
        MIOPEN_THROW("!problem.IsDirectionBackwardWrW()");
    // it's backward - inputs are outputs and vs versa
    const auto comp_options =
        std::string(" -DMLO_DIR_FORWARD=0") + std::string(" -DMLO_GRP_SZ=") +
        std::to_string((workgroup_size)) + std::string(" -DMLO_GRP_SZ0=") +
        std::to_string((result.grp_tile0)) + std::string(" -DMLO_GRP_SZ1=") +
        std::to_string((result.grp_tile1)) + std::string(" -DMLO_GRP_SZ2=") +
        std::to_string((grp_tile2)) + std::string(" -DMLO_FILTER_SIZE0=") +
        std::to_string(problem.GetWeightsWidth()) + std::string(" -DMLO_FILTER_SIZE1=") +
        std::to_string(problem.GetWeightsHeight()) + std::string(" -DMLO_FILTER_PAD0=") +
        std::to_string(problem.GetPadW()) + std::string(" -DMLO_FILTER_PAD1=") +
        std::to_string(problem.GetPadH()) + std::string(" -DMLO_FILTER_STRIDE0=") +
        std::to_string(problem.GetKernelStrideW()) + std::string(" -DMLO_FILTER_STRIDE1=") +
        std::to_string(problem.GetKernelStrideH()) + std::string(" -DMLO_N_OUTPUTS=") +
        std::to_string(problem.GetInChannels()) + std::string(" -DMLO_N_INPUTS=") +
        std::to_string(problem.GetOutChannels()) + std::string(" -DMLO_BATCH_SZ=") +
        std::to_string(problem.GetBatchSize()) + std::string(" -DMLO_N_BATCH_LOOPS=") +
        std::to_string(N_BATCH_LOOPS) + std::string(" -DMLO_N_BATCH_BLKS=") +
        std::to_string(n_batch_blks) + std::string(" -DMLO_OUT_BATCH_STRIDE=") +
        std::to_string((problem.GetInBatchStride())) + std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
        std::to_string((problem.GetInChannelStride())) + std::string(" -DMLO_OUT_STRIDE=") +
        std::to_string((problem.GetInStrideH())) + std::string(" -DMLO_IN_BATCH_STRIDE=") +
        std::to_string((problem.GetOutBatchStride())) + std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
        std::to_string((problem.GetOutChannelStride())) + std::string(" -DMLO_IN_STRIDE=") +
        std::to_string((problem.GetOutStrideH())) + std::string(" -DMLO_WEI_BATCH_STRIDE=") +
        std::to_string((wei_bstride)) + std::string(" -DMLO_WEI_CHANNEL_STRIDE=") +
        std::to_string((wei_cstride)) + std::string(" -DMLO_IN_WIDTH=") +
        std::to_string((problem.GetOutWidth())) + std::string(" -DMLO_IN_HEIGHT=") +
        std::to_string(problem.GetOutHeight()) + std::string(" -DMLO_OUT_WIDTH=") +
        std::to_string(problem.GetInWidth()) + std::string(" -DMLO_OUT_HEIGHT=") +
        std::to_string(problem.GetInHeight()) + std::string(" -DMLO_N_LCL_OUT_MAPS=") +
        std::to_string(config.n_out_channels_tiles) +
        // # output pixel tiles per wk-item (ALU)
        std::string(" -DMLO_N_LCL_IN_MAPS=") + std::to_string(result.n_in_data_tiles) +
        // total # of blocks of different inputs in LDS
        std::string(" -DMLO_N_WAVES=") + std::to_string(config.n_waves) +
        std::string(" -DMLO_READ_TYPE=") + READ_TYPE + std::string(" -DMLO_READ_UNIT=") +
        std::to_string(config.read_size) + std::string(" -DMLO_ALIGNED_OUT_SCAN_LN=") +
        std::to_string(aligned_out_scan_lane) + // image aligned scan
        std::string(" -DMLO_N_ALIGNED_OUT_SCAN_BLK=") + std::to_string(config.n_out_rows_in_lcl) +
        std::string(" -DMLO_WEI_WKITEM=") + std::to_string(wei_per_wkitem) +
        std::string(" -DMLO_N_OUT_BLK_GRP=") + std::to_string(config.n_out_channels_per_tile) +
        std::string(" -DMLO_N_OUT_BLK=") + std::to_string(n_out_blk) +
        std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(hw_wave_size) +
        std::string(" -DMLO_OUT_N_PIXS_OFF=") + std::to_string(out_n_pixels_off) +
        std::string(" -DMLO_IN_LCL_WIDTH=") + std::to_string(in_lcl_width) +
        std::string(" -DMLO_IN_LCL_SZ=") + std::to_string(in_lcl_sz) +
        std::string(" -DMLO_CONV_BIAS=") + std::to_string(problem.GetBias()) +
        std::string(" -DMLO_UT_READ_TYPE=") + UT_READ_TYPE + std::string(" -DMLO_UT_READ_UNIT=") +
        std::to_string(utility_read_unit) + std::string(" -DMLO_UT_GRP_SZ0=") +
        std::to_string(utility_workgroup_size) + std::string(" -DMLO_GROUP_COUNTS=") +
        std::to_string(problem.GetGroupCount()) + std::string(" -DMLO_N_INPUTS_PER_GROUP=") +
        std::to_string(n_input_channels_per_group) + std::string(" -DMLO_N_OUTPUTS_PER_GROUP=") +
        std::to_string(n_output_channels_per_group) + ctx.general_compile_options;

    // wrt to W
    {
        KernelInfo kernel;
        kernel.l_wk.push_back(result.grp_tile0);
        kernel.l_wk.push_back(result.grp_tile1);
        kernel.l_wk.push_back(grp_tile2);

        assert(total_out_maps != 0);
        size_t gbl_wk1 = std::ceil(static_cast<float>(problem.GetInChannels()) / total_out_maps);
        size_t gbl_wk2 = n_batch_blks;
        size_t gbl_wk0 = workgroup_size;

        if(problem.GetGroupCount() > 1)
        {
            gbl_wk0 *= n_input_channels_per_group;
            kernel.kernel_file = "MIOpenGroupConvBwdWrWS2.cl";
            kernel.kernel_name = "MIOpenCvBwdWrW";
        }
        else
        {
            gbl_wk0 *= problem.GetOutChannels();
            kernel.kernel_file = "MIOpenConvBwdWrWS2.cl";
            kernel.kernel_name = "MIOpenCvBwdWrW";
        }

        kernel.g_wk.push_back(gbl_wk0);
        kernel.g_wk.push_back(gbl_wk1);
        kernel.g_wk.push_back(gbl_wk2);

        kernel.comp_options = comp_options;

        result.construction_params.push_back(kernel);
    }

    // sum over batch
    if(n_batch_blks > 1)
    {
        KernelInfo kernel;

        kernel.kernel_file  = "MIOpenConvBwdWrWS2.cl";
        kernel.kernel_name  = "MIOpenCvBwdWrW_rdc";
        kernel.comp_options = comp_options;

        kernel.l_wk.push_back(utility_workgroup_size);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);

        assert(utility_read_unit != 0);
        unsigned gbl_ut_wk0 = wei_bstride * problem.GetInChannels() / utility_read_unit;

        kernel.g_wk.push_back(gbl_ut_wk0);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);
        result.construction_params.push_back(kernel);
    }

    const auto ws_sz       = GetWorkspaceSize(ctx, problem);
    result.workspace_sz    = ws_sz;
    result.invoker_factory = miopen::conv::MakeOclWrWRdcInvokerFactory(n_batch_blks > 1, ws_sz);

    return result;
}

template <int N_BATCH_LOOPS>
PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>
ConvOclBwdWrW2<N_BATCH_LOOPS>::Search(const ExecutionContext& ctx,
                                      const ProblemDescription& problem,
                                      const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

/// We need to instantiate required classes implicitly.
/// The reason is that we do not define the whole template class
/// in the header, only declaring it there.
template struct PerformanceConfigConvOclBwdWrw2<1>;
template struct PerformanceConfigConvOclBwdWrw2<2>;
template struct PerformanceConfigConvOclBwdWrw2<4>;
template struct PerformanceConfigConvOclBwdWrw2<8>;
template struct PerformanceConfigConvOclBwdWrw2<16>;

template struct ConvOclBwdWrW2<1>;
template struct ConvOclBwdWrW2<2>;
template struct ConvOclBwdWrW2<4>;
template struct ConvOclBwdWrW2<8>;
template struct ConvOclBwdWrW2<16>;

} // namespace conv
} // namespace solver
} // namespace miopen
