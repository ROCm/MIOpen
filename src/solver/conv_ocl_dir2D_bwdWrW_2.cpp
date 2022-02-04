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

#include <miopen/solver.hpp>

#include <miopen/conv/invokers/ocl_wrw_rdc.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/bfloat16.hpp>
#include <miopen/mlo_utils.hpp>
#include <miopen/visit_float.hpp>

#include <algorithm>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2_SEARCH_OPTIMIZED)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2)

namespace miopen {
namespace solver {

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

static bool IsTunable(const ConvolutionContext& params)
{
    return !(params.group_counts == 1 &&
             ((params.kernel_size_w == 3 && params.kernel_size_h == 3) ||
              (params.kernel_size_w == 1 && params.kernel_size_h == 1)));
}

bool ConvOclBwdWrW2NonTunable::IsApplicable(const ConvolutionContext& params) const
{
    // At present, auto-tuning is disabled for non-group 3x3 and 1x1 filters for multiple
    // reasons: after tuning ocl kernel for 3x3 and 1x1 filters, assembly kernel still
    // dominates. Thus, this solver is used for non-group 3x3 and 1x1 filters only.
    return ConvOclBwdWrW2<1>::IsApplicableBase(params) && !IsTunable(params);
}

ConvSolution ConvOclBwdWrW2NonTunable::GetSolution(const ConvolutionContext& params) const
{
    // Invoking base class GetSolution with default values for params obtained
    // from GetPerformanceConfig(params)
    return ConvOclBwdWrW2<1>::GetSolution(params, GetPerformanceConfig(params));
}

template <int N_BATCH_LOOPS>
inline bool PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>::operator==(
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
bool PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>::SetNextValue(
    const ConvolutionContext& /*config*/)
{
    // Increment with wrap-around:
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2_SEARCH_OPTIMIZED{}))
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
size_t GetNBatchBlks(const ConvolutionContext& params)
{
    return std::ceil(static_cast<float>(params.batch_sz) / (N_BATCH_LOOPS * N_STACKS));
}

template <int N_BATCH_LOOPS>
bool PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>::IsValid(const ConvolutionContext& params) const
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
    size_t wei_cstride = params.kernel_size_h * params.kernel_size_w;
    size_t wei_bstride = (params.n_outputs / params.group_counts) * wei_cstride;

    // number  of batch iterations
    const size_t n_batch_blks = GetNBatchBlks<N_BATCH_LOOPS>(params);

    // guard not to grab too much system memory
    if(n_batch_blks < 1 ||
       (wei_bstride * params.n_inputs * n_batch_blks) > params.GetStream().GetMaxMemoryAllocSize())
    {
        return false;
    }

    // Check 2: read size
    if(params.in_width < read_size)
    {
        return false;
    }
    size_t aligned_out_scan_lane =
        std::ceil(static_cast<float>(params.in_width) / read_size); // image aligned scan

    // Check 3: n_out_channels_tiles
    if(params.group_counts > 1 && n_out_channels_tiles > 1)
    {
        return false;
    }

    size_t n_output_channels_per_group = params.n_inputs / params.group_counts;

    // Check 4: n_out_channels_per_tile
    if(params.group_counts > 1 && n_out_channels_per_tile > n_output_channels_per_group)
    {
        return false;
    }

    // group config requires n_out_channels_tiles to be 1 or else
    // kernel doesn't work.
    if(params.group_counts > 1 && n_out_channels_tiles != 1)
    {
        return false;
    }

    size_t total_out_channels = n_out_channels_tiles * n_out_channels_per_tile;
    if(total_out_channels > n_output_channels_per_group)
    {
        return false;
    }

    if(n_out_rows_in_lcl < params.kernel_size_h)
    {
        return false;
    }

    // Check 5: n_out_rows_in_lcl  should exceed LDS limit
    size_t in_lcl_height = (n_out_rows_in_lcl - 1) * params.kernel_stride_h + params.kernel_size_h;
    size_t in_lcl_sz     = 0;
    {
        // Chao: Reserve space in LDS for left padding, it also reserve
        //   enough space in LDS for right padding and the footprint of scaning (from
        //   device into LDS). To reduces LDS consumption, right padding of one row
        //   is overlapped with the left padding of the next row.
        //   Also, for the last row, right padding is needed.
        // Revisit this if encounter failure
        size_t in_lcl_width = 0;
        size_t in_width     = params.out_width; // out is in, in is out
        size_t out_width    = params.in_width;

        size_t in_lcl_width_effective =
            std::max(in_width + 2 * params.pad_w,
                     std::max(params.pad_w + ((in_width + read_size - 1) / read_size) * read_size,
                              params.kernel_size_w + (out_width - 1) * params.kernel_stride_w));

        size_t in_lcl_width_right_buffer =
            std::max(static_cast<int>(in_lcl_width_effective - (in_width + 2 * params.pad_w)), 0);

        in_lcl_width = params.pad_w + in_width + in_lcl_width_right_buffer;

        // Chao: attempt to reduce LDS bank conflict during reading input image from LDS
        // Revisit this if performance regress
        if(params.out_data_type == miopenFloat)
        {
            in_lcl_width = (in_lcl_width / 2) * 2 + 1;
        }

        // make enough room for right padding and buffer for the last row in LDS
        in_lcl_sz = in_lcl_width * in_lcl_height + params.pad_w + in_lcl_width_right_buffer;
    }

    // check LDS consumption
    size_t wei_per_wkitem =
        (params.kernel_size_w <= 7 || (((params.kernel_size_w / 2) * 2) != params.kernel_size_w))
            ? params.kernel_size_w
            : params.kernel_size_w / 2;

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
        size_t out_width         = params.in_width; // out is in, in is out

        {
            const auto hw_wave_size   = 64; // TBD Obtain this from handle.
            const auto workgroup_size = hw_wave_size * n_waves;

            if(wei_per_wkitem == 0)
                return false;
            size_t wei_blk_sz0 =
                std::ceil(static_cast<float>(params.kernel_size_w) / wei_per_wkitem);
            size_t wei_blk_sz = params.kernel_size_h * wei_blk_sz0;
            if(wei_blk_sz == 0)
                return false;
            size_t n_wei_blk = workgroup_size / wei_blk_sz;
            if(n_wei_blk == 0)
            { /// \todo This is quickfix for DIV/0, see ROCmSoftwarePlatform/MIOpen/issues/70.
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
                                  GetTypeSize(params.out_data_type);

        const auto lds_size = 64 * 1024; // TBD Obtain this from handle.
        if(total_lcl_mem_sz > lds_size)
        {
            return false;
        }
    }

    if(n_batch_blks > 1)
    {
        size_t data_len    = GetTypeSize(params.out_data_type);
        result.workspce_sz = static_cast<std::size_t>(wei_bstride) *
                             static_cast<std::size_t>(params.n_inputs) * n_batch_blks *
                             static_cast<std::size_t>(data_len);

#if WORKAROUND_ISSUE_1185
        if(result.workspce_sz >
           (std::size_t(6) * std::size_t(1024) * std::size_t(1024) * std::size_t(1024)))
        {
            return false;
        }
#endif
    }

    return true;
}

template <int N_BATCH_LOOPS>
void PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>::HeuristicInit(const ConvolutionContext& params)
{
    n_waves                                = 1;
    read_size                              = 6;
    const auto n_output_channels_per_group = params.n_inputs / params.group_counts;
    if(n_output_channels_per_group % 4 == 0)
        n_out_channels_per_tile = 4;
    else if(n_output_channels_per_group % 3 == 0)
        n_out_channels_per_tile = 3;
    else if(n_output_channels_per_group % 2 == 0)
        n_out_channels_per_tile = 2;
    else
        n_out_channels_per_tile = 1;
    n_out_channels_tiles = 1;
    n_out_rows_in_lcl    = params.kernel_size_h;
}

template <int N_BATCH_LOOPS>
std::string PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>::ToString() const
{
    std::ostringstream ss;
    ss << (*this);
    return ss.str();
}

template <int N_BATCH_LOOPS>
bool ConvOclBwdWrW2<N_BATCH_LOOPS>::IsValidPerformanceConfig(
    const ConvolutionContext& params,
    const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& perfConfig) const
{
    return perfConfig.IsValidValue() && perfConfig.IsValid(params);
}

template <int N_BATCH_LOOPS>
bool ConvOclBwdWrW2<N_BATCH_LOOPS>::IsApplicableBase(const ConvolutionContext& params) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2{}))
        return false;
    if(!params.use_opencl_convolutions)
        return false;
    if(!params.Is2d())
        return false;
    if(!params.direction.IsBackwardWrW())
        return false;
    if(params.IsAsymmetricPadH() || params.IsAsymmetricPadW())
        return false;
    if(!(params.IsFp32() || params.IsFp16() || params.IsBfp16()))
        return false;
    if(!params.IsLayoutDefault())
    {
        return false;
    }

    return params.kernel_dilation_w == 1 && params.kernel_dilation_h == 1 &&
#if 0
           // There is a stronger restriction than this one, which make this one unnecessary.
           // The kernel read stripes (in height direction, one stripe at a time) of input into LDS,
           // the height of stripe is (MLO_N_ALIGNED_OUT_SCAN_BLK - 1) * MLO_FILTER_STRIDE1 +
           // MLO_FILTER_SIZE1, (MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1) of it is reusable from
           // previous read, (MLO_N_ALIGNED_OUT_SCAN_BLK * MLO_FILTER_STRIDE1) of it is fresh read
           // from device memory. So (MLO_FILTER_SIZE1 - MLO_FILTER_STRIDE1) need no less than 0.
           // TODO: chao: revisit this if failure is encountered.
           params.kernel_size_h - params.kernel_stride_h >= 0 &&
#endif

           // The first scan of stripe of the input into LDS will read a strip of height
           // (kernel_size_h - kernel_stride_h), this stripe should include the whole lower bound
           // padding, as well as some or none of the input.
           params.kernel_size_h - params.kernel_stride_h >= params.pad_h &&
           params.batch_sz >= N_BATCH_LOOPS &&
           /// \todo Workaround for issue 1693
           !(params.kernel_size_w >= 8 && params.kernel_size_w % 2 == 0 &&
             !( // Allow these configs to avoid perf drops:
                 (params.kernel_stride_h == 2 && params.kernel_stride_w == 2) &&
                 (params.kernel_size_h == 5 &&
                  (params.kernel_size_w == 10 || params.kernel_size_w == 20)) &&
                 ((params.out_height == 79 && params.out_width == 341) ||
                  (params.out_height == 161 && params.out_width == 700)))) &&
           /// Avoid LDS & Workspace over-allocation.
           /// \note Required LDS depends on PerformanceConfig.
           /// We use the default PerformanceConfig here. This guarantees that at least
           /// one config will pass the LDS constraint check during auto-tuning.
           /// This works also for non-tunable solver.
           IsValidPerformanceConfig(params, GetPerformanceConfig(params));
}

template <int N_BATCH_LOOPS>
bool ConvOclBwdWrW2<N_BATCH_LOOPS>::IsApplicable(const ConvolutionContext& params) const
{
    return IsApplicableBase(params) && IsTunable(params);
}

template <int N_BATCH_LOOPS>
PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>
ConvOclBwdWrW2<N_BATCH_LOOPS>::GetPerformanceConfig(const ConvolutionContext& params) const
{
    PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS> pp;
    pp.HeuristicInit(params);
    return pp;
}

template <int N_BATCH_LOOPS>
size_t ConvOclBwdWrW2<N_BATCH_LOOPS>::GetWorkspaceSize(const ConvolutionContext& params) const
{
    const size_t n_batch_blks = GetNBatchBlks<N_BATCH_LOOPS>(params);
    if(n_batch_blks > 1)
    {
        const auto n_input_channels_per_group = params.n_outputs / params.group_counts;
        const auto wei_cstride                = params.kernel_size_w * params.kernel_size_h;
        const auto wei_bstride                = n_input_channels_per_group * wei_cstride;
        int data_len                          = GetTypeSize(params.out_data_type);
        return static_cast<std::size_t>(wei_bstride) * static_cast<std::size_t>(params.n_inputs) *
               static_cast<std::size_t>(n_batch_blks) * static_cast<std::size_t>(data_len);
    }
    else
        return 0;
}

template <int N_BATCH_LOOPS>
ConvSolution ConvOclBwdWrW2<N_BATCH_LOOPS>::GetSolution(
    const ConvolutionContext& params,
    const PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>& config,
    bool) const
{
    ConvSolution result;
    const auto hw_wave_size   = 64;
    const auto workgroup_size = hw_wave_size * config.n_waves;

    const auto n_input_channels_per_group  = params.n_outputs / params.group_counts;
    const auto n_output_channels_per_group = params.n_inputs / params.group_counts;
    const auto wei_cstride                 = params.kernel_size_w * params.kernel_size_h;
    const auto wei_bstride                 = n_input_channels_per_group * wei_cstride;

    result.n_in_data_tiles    = 1;
    const size_t n_batch_blks = GetNBatchBlks<N_BATCH_LOOPS>(params);
    size_t total_out_maps     = config.n_out_channels_per_tile * config.n_out_channels_tiles;
    size_t wei_per_wkitem =
        (params.kernel_size_w <= 7 || (((params.kernel_size_w / 2) * 2) != params.kernel_size_w))
            ? params.kernel_size_w
            : params.kernel_size_w / 2;

    // each wave is a filter row
    std::string READ_TYPE =
        (config.read_size == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((config.read_size));
    size_t aligned_out_scan_lane =
        std::ceil(static_cast<float>(params.in_width) / config.read_size); // image aligned scan
    size_t n_out_blk = std::ceil(static_cast<float>(params.in_height) / config.n_out_rows_in_lcl);
    size_t in_lcl_height =
        (config.n_out_rows_in_lcl - 1) * params.kernel_stride_h + params.kernel_size_h;
    size_t in_lcl_width = 0;
    size_t in_lcl_sz    = 0;
    {
        // Chao: Reserve space in LDS for left padding, it also reserve
        //   enough space in LDS for right padding and the footprint of scaning (from
        //   device into LDS). To reduces LDS consumption, right padding of one row
        //   is overlapped with the left padding of the next row.
        //   Also, for the last row, right padding is needed.
        // Revisit this if encounter failure
        size_t in_width  = params.out_width; // out is in, in is out
        size_t out_width = params.in_width;

        size_t in_lcl_width_effective =
            std::max(in_width + 2 * static_cast<size_t>(params.pad_w),
                     std::max(static_cast<size_t>(params.pad_w) +
                                  static_cast<size_t>(
                                      std::ceil(static_cast<float>(in_width) / config.read_size) *
                                      config.read_size),
                              static_cast<size_t>(params.kernel_size_w) +
                                  (out_width - 1) * params.kernel_stride_w));

        size_t in_lcl_width_right_buffer =
            std::max(static_cast<int>(in_lcl_width_effective - (in_width + 2 * params.pad_w)), 0);

        in_lcl_width = params.pad_w + in_width + in_lcl_width_right_buffer;

        // Chao: attempt to reduce LDS bank conflict during reading input image from LDS
        // Revisit this if performance regress
        if(params.out_data_type == miopenFloat)
        {
            in_lcl_width = (in_lcl_width / 2) * 2 + 1;
        }

        // make enough room for right padding and buffer for the last row in LDS
        in_lcl_sz = in_lcl_width * in_lcl_height + params.pad_w + in_lcl_width_right_buffer;
    }

    size_t out_n_pixels_off =
        params.in_width - (params.in_width / config.read_size) * config.read_size;

    result.grp_tile0       = workgroup_size;
    result.grp_tile1       = 1;
    unsigned int grp_tile2 = 1;

    // utility parameters
    unsigned int n_utility_waves        = 4;
    unsigned int utility_workgroup_size = hw_wave_size * n_utility_waves;
    unsigned int utility_read_unit =
        ((wei_cstride / 4) * 4 == wei_cstride) ? 4 : ((wei_cstride / 2) * 2 == wei_cstride) ? 2 : 1;
    std::string UT_READ_TYPE =
        (utility_read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((utility_read_unit));

    if(!params.direction.IsBackwardWrW())
        MIOPEN_THROW("!params.direction.IsBackwardWrW()");
    // it's backward - inputs are outputs and vs versa
    const auto comp_options =
        std::string(" -DMLO_DIR_FORWARD=0") + std::string(" -DMLO_GRP_SZ=") +
        std::to_string((workgroup_size)) + std::string(" -DMLO_GRP_SZ0=") +
        std::to_string((result.grp_tile0)) + std::string(" -DMLO_GRP_SZ1=") +
        std::to_string((result.grp_tile1)) + std::string(" -DMLO_GRP_SZ2=") +
        std::to_string((grp_tile2)) + std::string(" -DMLO_FILTER_SIZE0=") +
        std::to_string(params.kernel_size_w) + std::string(" -DMLO_FILTER_SIZE1=") +
        std::to_string(params.kernel_size_h) + std::string(" -DMLO_FILTER_PAD0=") +
        std::to_string(params.pad_w) + std::string(" -DMLO_FILTER_PAD1=") +
        std::to_string(params.pad_h) + std::string(" -DMLO_FILTER_STRIDE0=") +
        std::to_string(params.kernel_stride_w) + std::string(" -DMLO_FILTER_STRIDE1=") +
        std::to_string(params.kernel_stride_h) + std::string(" -DMLO_N_OUTPUTS=") +
        std::to_string(params.n_inputs) + std::string(" -DMLO_N_INPUTS=") +
        std::to_string(params.n_outputs) + std::string(" -DMLO_BATCH_SZ=") +
        std::to_string(params.batch_sz) + std::string(" -DMLO_N_BATCH_LOOPS=") +
        std::to_string(N_BATCH_LOOPS) + std::string(" -DMLO_N_BATCH_BLKS=") +
        std::to_string(n_batch_blks) + std::string(" -DMLO_OUT_BATCH_STRIDE=") +
        std::to_string((params.in_batch_stride)) + std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
        std::to_string((params.in_channel_stride)) + std::string(" -DMLO_OUT_STRIDE=") +
        std::to_string((params.in_stride)) + std::string(" -DMLO_IN_BATCH_STRIDE=") +
        std::to_string((params.out_batch_stride)) + std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
        std::to_string((params.out_channel_stride)) + std::string(" -DMLO_IN_STRIDE=") +
        std::to_string((params.out_stride)) + std::string(" -DMLO_WEI_BATCH_STRIDE=") +
        std::to_string((wei_bstride)) + std::string(" -DMLO_WEI_CHANNEL_STRIDE=") +
        std::to_string((wei_cstride)) + std::string(" -DMLO_IN_WIDTH=") +
        std::to_string((params.out_width)) + std::string(" -DMLO_IN_HEIGHT=") +
        std::to_string(params.out_height) + std::string(" -DMLO_OUT_WIDTH=") +
        std::to_string(params.in_width) + std::string(" -DMLO_OUT_HEIGHT=") +
        std::to_string(params.in_height) + std::string(" -DMLO_N_LCL_OUT_MAPS=") +
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
        std::string(" -DMLO_CONV_BIAS=") + std::to_string(params.bias) +
        std::string(" -DMLO_UT_READ_TYPE=") + UT_READ_TYPE + std::string(" -DMLO_UT_READ_UNIT=") +
        std::to_string(utility_read_unit) + std::string(" -DMLO_UT_GRP_SZ0=") +
        std::to_string(utility_workgroup_size) + std::string(" -DMLO_GROUP_COUNTS=") +
        std::to_string(params.group_counts) + std::string(" -DMLO_N_INPUTS_PER_GROUP=") +
        std::to_string(n_input_channels_per_group) + std::string(" -DMLO_N_OUTPUTS_PER_GROUP=") +
        std::to_string(n_output_channels_per_group) + params.general_compile_options;

    // wrt to W
    {
        KernelInfo kernel;
        kernel.l_wk.push_back(result.grp_tile0);
        kernel.l_wk.push_back(result.grp_tile1);
        kernel.l_wk.push_back(grp_tile2);

        assert(total_out_maps != 0);
        size_t gbl_wk1 = std::ceil(static_cast<float>(params.n_inputs) / total_out_maps);
        size_t gbl_wk2 = n_batch_blks;
        size_t gbl_wk0 = workgroup_size;

        if(params.group_counts > 1)
        {
            gbl_wk0 *= n_input_channels_per_group;
            kernel.kernel_file = "MIOpenGroupConvBwdWrWS2.cl";
            kernel.kernel_name = "MIOpenCvBwdWrW";
        }
        else
        {
            gbl_wk0 *= params.n_outputs;
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
        int gbl_ut_wk0 =
            static_cast<int>(static_cast<int>(wei_bstride) * params.n_inputs / utility_read_unit);

        kernel.g_wk.push_back(gbl_ut_wk0);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);
        result.construction_params.push_back(kernel);
    }

    const auto ws_sz       = GetWorkspaceSize(params);
    result.workspce_sz     = ws_sz;
    result.invoker_factory = conv::MakeOclWrWRdcInvokerFactory(n_batch_blks > 1, ws_sz);

    return result;
}

template <int N_BATCH_LOOPS>
PerformanceConfigConvOclBwdWrw2<N_BATCH_LOOPS>
ConvOclBwdWrW2<N_BATCH_LOOPS>::Search(const ConvolutionContext& context,
                                      const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, context, invoke_ctx);
}

/// We need to instantiate required classes implicitly.
/// The reason is that we do not define the whole template class
/// in the header, only declaring it there.
template struct ConvOclBwdWrW2<1>;
template struct ConvOclBwdWrW2<2>;
template struct ConvOclBwdWrW2<4>;
template struct ConvOclBwdWrW2<8>;
template struct ConvOclBwdWrW2<16>;

} // namespace solver
} // namespace miopen
