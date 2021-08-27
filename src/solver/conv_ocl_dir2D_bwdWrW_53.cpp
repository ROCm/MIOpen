/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017-2018 Advanced Micro Devices, Inc.
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

#include <miopen/conv/invokers/ocl_wrw_rdc.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53)

namespace miopen {
namespace solver {

// Once the compiler fix (SWDEV-168168) is available, that version of compiler needs to be
// checked to skip workarounds. Till then, true is returned in all cases so as to skip
// problematic configs.
static bool WorkaroundSwdev168168() { return true; }

bool ConvOclBwdWrW53::IsApplicable(const ConvolutionContext& params) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53{}))
        return false;
    if(!params.use_opencl_convolutions)
        return false;
    if(!params.Is2d())
        return false;
    if(params.IsAsymmetricPadH() || params.IsAsymmetricPadW())
        return false;
    if(!(params.IsFp32() || params.IsFp16() || params.IsBfp16()))
        return false;
    if(!params.direction.IsBackwardWrW())
        return false;
    if(!params.IsLayoutDefault())
    {
        return false;
    }

    bool workaround = false;

    if(WorkaroundSwdev168168())
    {
        // Workaround for issue 1173. These FP16 configs would cause clang-ocl compiler to crash
        // during kernel compilation, due to compiler bug
        workaround =
            workaround ||
            (params.out_data_type == miopenHalf &&
             ((params.kernel_size_w == 7 && params.kernel_size_h == 7 && params.pad_w == 3) ||
              (params.kernel_size_w == 7 && params.kernel_size_h == 7 && params.pad_w == 2) ||
              (params.kernel_size_w == 11 && params.kernel_size_h == 11 && params.pad_w == 5) ||
              (params.kernel_size_w == 11 && params.kernel_size_h == 11 && params.pad_w == 2) ||
              (params.kernel_size_w == 11 && params.kernel_size_h == 11 && params.pad_w == 1)));

        // Workaround for issue 1242. These FP32 configs produce wrong result if compiled with
        // OpenCL 1.2.0-2018090737 that comes with rocm 1.9, using -O2 flag or higher.
        // However, when compiled with older OpenCL that comes with rocm 1.8, this config
        // would pass
        workaround =
            workaround ||
            (params.out_data_type == miopenFloat &&
             ((params.kernel_size_w == 7 && params.kernel_size_h == 7 && params.pad_w == 3) ||
              (params.kernel_size_w == 7 && params.kernel_size_h == 7 && params.pad_w == 1)) &&
             (params.out_height % 112 == 0 || params.out_width % 112 == 0));

        // Workaround for issue 1479
        // The compiler issue causes the correctness failure of particular config
        // --input 1, 64, n, 1024 --weights 1, 64, 3, 3 -filter 2 2 1 1 1 1 --group-count 1
        // Disabling compiler optimization i.e. #pragma unroll in MIOpenConvBwdWrW_LxG_P53.cl
        // restores the correctness. Until, the compiler issue is fixed, all configs with width 1024
        // is skipped
        workaround = workaround ||
                     (params.IsFp32() && params.kernel_size_w == 3 && params.kernel_size_h == 3 &&
                      params.pad_h == 2 && params.pad_w == 2 && params.out_width == 1024);
    }

    /// Resolve NaN issue on gfx908, manifested on Jenkins.
    /// Note that there is another solver, ConvOclBwdWrW2, that has very similar
    /// performance and applicable for the affected "popular" configs (7x7 filter, 1x1 padding).
    const auto name = params.GetStream().GetDeviceName();
    workaround =
        workaround || (params.IsFp16() && (name == "gfx908") && params.kernel_size_w == 7 &&
                       params.kernel_size_h == 7 && params.pad_w == 1);

    return (params.kernel_dilation_w == 1 && params.kernel_dilation_h == 1) &&
           (params.kernel_stride_w == 1 && params.kernel_stride_h == 1) &&

           // This limitation is because of the way the kernel process data at lower vertical
           // boundary (including padding).
           (params.kernel_size_h >= params.pad_h + params.kernel_stride_h) &&

           // Input image height plus vertical paddings should be no less than filter vertical size.
           // TODO: chao: revisit this to make sure this is the actual limitation.
           // Remind that input is output, output is input.
           (params.kernel_size_h <= params.out_height + 2 * params.pad_h) &&

           // Input and output width and height need to match exactly,
           // meaning, filter's moving range should be the same as input plus padding.
           // TODO: chao: in order to remove this limitation, need to rewrite how kernel handle
           // right padding, when reading an input row into LDS. Also need to rewrite the vertical
           // loop.
           // Remind that input is output, output is input.
           (params.in_height == params.out_height + 2 * params.pad_h - params.kernel_size_h + 1) &&
           (params.in_width == params.out_width + 2 * params.pad_w - params.kernel_size_w + 1) &&

           // Avoid LDS over-allocation
           GetSolution(params).Succeeded() && !workaround;
}

/*! @brief ComputeInputParams
 *
 * Given the availabe LDS size, this function recomputes some of tunable parameters
 *
 * The goal is to reduce the amount of work that would be done in a workgroup that will
 * reduce the required LDS.
 *
 * The reduction is done in certain order: first cut down the number of input channels processed
 * in a single workgroup. If that doesn't suffice, cut down the number of input rows (verticals)
 * processed in a single workgroup. At last, cut down the input row into chunks.
 *
 */
static inline miopenStatus_t ComputeInputParams(
    const ConvolutionContext& params,
    int out_lcl_width,
    int& num_out_channels, // No. of input channels to be processed in a single workgroup
    int& out_n_horizon_reads,
    int& out_n_vert_reads,
    size_t workgroup_size)
{
    // In case where input (x) rows are splitted into chunks so that each chunk fits into LDS,
    // each chunk should also include pixels covering the complete filter size in horizonal
    // direction
    // so that the corresponding output chunk could be entirely processed.
    out_n_horizon_reads = out_lcl_width;

    // As each width chunk starts to get split,
    // it should include complete kernel filter in horizontal span.
    const uint filter_adjustment = params.kernel_size_w - 1;

    const auto lds_size         = 64 * 1024; /// TBD Obtain this from device info.
    const auto max_lds_elements = lds_size / (2 * GetTypeSize(params.in_data_type));

    while(num_out_channels * out_n_vert_reads * (out_n_horizon_reads + filter_adjustment) >
          max_lds_elements)
    {
        if(out_n_vert_reads < 2 && num_out_channels >= 2)
        {
            out_n_vert_reads = params.in_height;
            num_out_channels = std::ceil(static_cast<float>(num_out_channels) / 2);
        }
        else if(out_n_vert_reads >= params.kernel_size_h * 2)
        {
            out_n_vert_reads = std::ceil(static_cast<float>(out_n_vert_reads) / 2);
        }
        else if(out_n_vert_reads >= params.kernel_size_h && out_n_horizon_reads > 2)
        {
            out_n_horizon_reads = std::ceil(static_cast<float>(out_n_horizon_reads) / 2);
        }
        else
        {
            MIOPEN_LOG_I2("Can't fit input data into LDS of size "
                          << lds_size << " bytes despite row splitting");
            return miopenStatusNotInitialized;
        }
    }

    // LDS check based on weight blob
    // Kernel uses LDS for storing input data and weight accumulation
    if(workgroup_size * params.kernel_size_w > max_lds_elements)
    {
        MIOPEN_LOG_I2("For large filter size " << params.kernel_size_w
                                               << ", running out of LDS size (bytes) " << lds_size);
        return miopenStatusNotInitialized;
    }

    return miopenStatusSuccess;
}

/*! @brief ComputeOutputParams
 *
 *  Given output width chunk, this function computes how many output
 *  pixels are going to be processed in a single wkitem i.e. out_tile0. That in turn
 *  helps compute the number of spans needed to process the output width chunk.
 *
 *  Bear in mind that output width chunk is the same as output width if there was
 *  no split.
 */
static inline void ComputeOutputParams(int output_width,
                                       int workgroup_size,
                                       int output_width_chunk,
                                       int num_out_channels, // No. of output channels in config
                                       int group_counts,
                                       int& out_tile0,
                                       int& n_out_stacks)
{

    size_t out_pixels_per_wkitem_by_mod =
        (output_width_chunk % 4 == 0)
            ? 4
            : (output_width_chunk % 3 == 0) ? 3 : (output_width_chunk % 2 == 0) ? 2 : 1;

    assert(workgroup_size != 0);

    size_t out_pixels_per_wkitem_by_wkgrp =
        std::ceil(static_cast<float>(output_width_chunk) / static_cast<float>(workgroup_size));

    // work item in a group should cover at least 1 row of output image
    out_tile0 = std::max(out_pixels_per_wkitem_by_wkgrp, out_pixels_per_wkitem_by_mod);

    if(output_width == output_width_chunk)
    {
        // span size
        int n_spans =
            std::ceil(static_cast<float>(output_width_chunk) / static_cast<float>(out_tile0));

        // Only process 1 output channel in workgroup when group is specified
        // TBD: To support more than output channels in workgroup, more changes are required in
        // kernel
        // as which input channels are to be read from input is driven by combination of
        // weight input channel thread idx and output channel index.
        n_out_stacks = (group_counts == 1)
                           ? std::min(num_out_channels, std::max(1, workgroup_size / n_spans))
                           : 1;
    }
}

/*! @brief Compute the number of loops required to process the entire input width
 *         (applicable only when it doesn't fit in LDS in its entirety)
 *
 * In the event, when the input row is split into chunks to fit in LDS, the chunk
 * processing needs to rescan the (kernel size - 1) values on the left as one moves
 * from one chunk to another.
 *
 * In that case, the number of loops is computed based on the following
 * equation
 *     x = # of horizonal pixels in LDS
 *     n = # of loops
 *     y = input width
 *     fw = filter width
 *
 *     eq: x*1 + (x-fw+1)*n = y  -> n = ceil((y-x)/(x-fw+1))
 *
 * The number of loops required to process the entire input = n + 1
 * 1 is added to consider the initial unique x values (x*1).
 */
static inline void ComputeNumInputWidthLoops(
    int width,
    int padding,
    int n_horizon_reads,
    int filter_width,
    int& out_n_horizon_read_loops,
    int& out_horizon_last_chunk_valid_pixels // Last iteration may not process all reads
)
{
    if(width == n_horizon_reads)
    {
        out_n_horizon_read_loops            = 1;
        out_horizon_last_chunk_valid_pixels = 0;
    }
    else
    {
        int unpadded_width = width - 2 * padding;
        // # of pixels read fresh in each iteration except 1st iteration
        int n_fresh_horizon_reads = n_horizon_reads - filter_width + 1;
        if(n_fresh_horizon_reads <= 0)
        {
            MIOPEN_THROW(miopenStatusInvalidValue,
                         "Looks like input width split seems to lower than filter width - so no "
                         "read possible.");
        }
        out_n_horizon_read_loops =
            static_cast<int>(std::ceil(static_cast<float>(unpadded_width - n_horizon_reads) /
                                       static_cast<float>(n_fresh_horizon_reads))) +
            1;
        out_horizon_last_chunk_valid_pixels =
            (unpadded_width - n_horizon_reads) % n_fresh_horizon_reads + (filter_width - 1);
    }
}

size_t ConvOclBwdWrW53::GetWorkspaceSize(const ConvolutionContext& params) const
{
    int n_stacks = std::min(params.batch_sz, 1);
    int N_BATCH_LOOPS =
        (params.n_inputs * params.n_outputs <= 8 * 1024)
            ? 1
            : (params.batch_sz <= 16 || params.in_width <= 32) ? (params.batch_sz / n_stacks) : 4;
    int n_batch_blks =
        (params.batch_sz + N_BATCH_LOOPS * n_stacks - 1) / (N_BATCH_LOOPS * n_stacks);
    if(n_batch_blks > 1)
    {
        int wei_bstride = (params.n_outputs / params.group_counts) *
                          (params.kernel_size_w * params.kernel_size_h);
        int data_len = GetTypeSize(params.out_data_type);
        return wei_bstride * params.n_inputs * n_batch_blks * data_len;
    }
    else
        return 0;
}

ConvSolution ConvOclBwdWrW53::GetSolution(const ConvolutionContext& params) const
{
    ConvSolution result;

    const auto hw_wave_sz = 64;
    // inpout are outputs
    int wei_cstride = params.kernel_size_w * params.kernel_size_h;

    // At convolutionocl level, the assertion is present to ensure output channels are
    // in multiple of group counts
    int wei_bstride = (params.n_outputs / params.group_counts) * wei_cstride;

    // number  of batch iterations
    result.n_stacks = 1;
    result.n_stacks = std::min(params.batch_sz, result.n_stacks);
    // defines how to proceed : 1 grouop per batch or with a loop over all batches
    // loop over al batches make sense in 2 cases: a lot of small inputs/outputs or few batches
    int N_BATCH_LOOPS = (params.n_inputs * params.n_outputs <= 8 * 1024)
                            ? 1
                            : (params.batch_sz <= 16 || params.in_width <= 32)
                                  ? (params.batch_sz / result.n_stacks)
                                  : 4;
    int n_batch_blks =
        (params.batch_sz + N_BATCH_LOOPS * result.n_stacks - 1) / (N_BATCH_LOOPS * result.n_stacks);

    result.out_pix_tile0 = params.kernel_size_w;
    result.out_pix_tile1 = params.kernel_size_h;

    // n of wavefronts per group
    int n_waves = ((result.out_pix_tile0 * result.out_pix_tile1) <= 16 && (params.in_width > 8))
                      ? 4
                      : (params.in_width <= 16) ? 1 : 2;
    int GRP_SZ = hw_wave_sz * n_waves;
    result.n_in_data_tiles =
        (params.in_width <= 32 && (result.out_pix_tile0 * result.out_pix_tile1) <= 16) ? 4 : 1;

    result.n_in_data_tiles =
        std::min(result.n_in_data_tiles, (params.n_outputs / params.group_counts));

    static const int read_unit =
        (params.out_width % 4 == 0)
            ? 4
            : (params.out_width % 3 == 0) ? 3 : (params.out_width % 2 == 0) ? 2 : 1;

    static const std::string READ_TYPE =
        (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((read_unit));

    // calculate number of input scans in the input block
    int out_lcl_width =
        ((params.out_width + read_unit - 1) / read_unit) * read_unit + 2 * params.pad_w;

    // number of input map blocks being process at once
    int out_n_vert_reads = (params.out_height > 32 && params.out_width <= 64 &&
                            (result.out_pix_tile0 * result.out_pix_tile1) <= 16)
                               ? (params.out_height + 1) / 2
                               : params.out_height;

    // Given the availability of LDS, recomputes the params
    int out_n_horizon_reads = out_lcl_width;
    if(ComputeInputParams(params,
                          out_lcl_width,
                          result.n_in_data_tiles,
                          out_n_horizon_reads,
                          out_n_vert_reads,
                          GRP_SZ) != miopenStatusSuccess ||
       out_n_vert_reads <= 0)
    {
        return {miopenStatusNotInitialized};
    }

    int out_n_vert_read_loops = static_cast<int>(
        std::ceil(static_cast<float>(params.out_height) / static_cast<float>(out_n_vert_reads)));

    // When a row is split into chunks, each chunk should fully cover the entire filter in
    // horizontal dir
    out_n_horizon_reads = (out_n_horizon_reads == out_lcl_width)
                              ? out_lcl_width
                              : (out_n_horizon_reads + params.kernel_size_w - 1);

    int out_n_horizon_read_loops            = 1;
    int out_horizon_last_chunk_valid_pixels = 0;
    ComputeNumInputWidthLoops(out_lcl_width,
                              params.pad_w,
                              out_n_horizon_reads,
                              params.kernel_size_w,
                              out_n_horizon_read_loops,
                              out_horizon_last_chunk_valid_pixels);
    if(out_n_horizon_read_loops > 2 && params.pad_w != 0)
    {
        MIOPEN_LOG_I2("Padding where split is more than 2 ways is not supported.");
        return {miopenStatusNotInitialized};
    }
    if(out_n_horizon_read_loops > 1 && params.group_counts > 1)
    {
        MIOPEN_LOG_I2("For large images, group support is missing.");
        return {miopenStatusNotInitialized};
    }

    int out_horizon_last_chunk_valid_read_units = std::ceil(
        static_cast<float>(out_horizon_last_chunk_valid_pixels) / static_cast<float>(read_unit));
    int out_horizon_last_chunk_valid_pixels_in_last_read_unit =
        (out_horizon_last_chunk_valid_pixels % read_unit != 0)
            ? out_horizon_last_chunk_valid_pixels % read_unit
            : read_unit;

    // Compute in -> out in kernel i.e. dy
    int in_width_chunk = (out_n_horizon_read_loops == 1)
                             ? params.in_width
                             : (out_n_horizon_reads + params.pad_w - params.kernel_size_w + 1);
    int in_width_last_chunk_valid_pixels =
        (out_n_horizon_read_loops == 1) ? 0 : (params.in_width % in_width_chunk);

    result.in_tile1        = 1;
    result.n_out_pix_tiles = 1;
    int n_out_stacks       = 1;
    ComputeOutputParams(params.in_width,
                        GRP_SZ,
                        in_width_chunk,
                        params.n_inputs,
                        params.group_counts,
                        result.in_tile0,
                        n_out_stacks);

    if(result.in_tile0 <= 0)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Looks like tile size is set <=0.");
    }

    int in_width_last_chunk_valid_spans = std::ceil(
        static_cast<float>(in_width_last_chunk_valid_pixels) / static_cast<float>(result.in_tile0));
    int in_width_last_chunk_valid_pixels_in_last_span =
        (in_width_last_chunk_valid_pixels % result.in_tile0 != 0)
            ? in_width_last_chunk_valid_pixels % result.in_tile0
            : result.in_tile0;

    // select output mapping
    int total_out_maps = result.n_out_pix_tiles * n_out_stacks;
    total_out_maps     = (total_out_maps > params.n_inputs) ? params.n_inputs : total_out_maps;

    result.grp_tile0 = GRP_SZ;
    result.grp_tile1 = 1;
    int grp_tile2    = 1;

    // utility parameters
    int n_ut_waves = 4;
    int UT_GRP_SZ0 = hw_wave_sz * n_ut_waves;
    int ut_read_unit =
        ((wei_cstride / 4) * 4 == wei_cstride) ? 4 : ((wei_cstride / 2) * 2 == wei_cstride) ? 2 : 1;
    std::string UT_READ_TYPE =
        (ut_read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string((ut_read_unit));

    // group parameters
    int n_input_channels_per_group  = params.n_outputs / params.group_counts;
    int n_output_channels_per_group = params.n_inputs / params.group_counts;

    if(!params.direction.IsBackwardWrW())
        MIOPEN_THROW("!params.direction.IsBackwardWrW()");
    // it's backward - inputs are outputs and vs versa
    auto comp_options =
        std::string(" -DMLO_DIR_FORWARD=0") + std::string(" -DMLO_GRP_SZ=") +
        std::to_string(GRP_SZ) + std::string(" -DMLO_GRP_SZ0=") + std::to_string(result.grp_tile0) +
        std::string(" -DMLO_GRP_SZ1=") + std::to_string(result.grp_tile1) +
        std::string(" -DMLO_GRP_SZ2=") + std::to_string(grp_tile2) +
        std::string(" -DMLO_FILTER_SIZE0=") + std::to_string(params.kernel_size_w) +
        std::string(" -DMLO_FILTER_SIZE1=") + std::to_string(params.kernel_size_h) +
        std::string(" -DMLO_FILTER_PAD0=") + std::to_string(params.pad_w) +
        std::string(" -DMLO_FILTER_PAD1=") + std::to_string(params.pad_h) +
        std::string(" -DMLO_FILTER_STRIDE0=") + std::to_string(params.kernel_stride_w) +
        std::string(" -DMLO_FILTER_STRIDE1=") + std::to_string(params.kernel_stride_h) +
        std::string(" -DSTRIDE_W=") + std::to_string(params.kernel_stride_w) +
        std::string(" -DSTRIDE_H=") + std::to_string(params.kernel_stride_h) +
        std::string(" -DMLO_N_OUTPUTS=") + std::to_string(params.n_inputs) +
        std::string(" -DMLO_N_INPUTS=") + std::to_string(params.n_outputs) +
        std::string(" -DMLO_GROUP_COUNTS=") + std::to_string(params.group_counts) +
        std::string(" -DMLO_N_INPUTS_PER_GROUP=") + std::to_string(n_input_channels_per_group) +
        std::string(" -DMLO_N_OUTPUTS_PER_GROUP=") + std::to_string(n_output_channels_per_group) +
        std::string(" -DMLO_BATCH_SZ=") + std::to_string(params.batch_sz) +
        std::string(" -DMLO_N_BATCH_LOOPS=") + std::to_string(N_BATCH_LOOPS) +
        std::string(" -DMLO_OUT_BATCH_STRIDE=") + std::to_string(params.in_batch_stride) +
        std::string(" -DMLO_OUT_CHANNEL_STRIDE=") + std::to_string(params.in_channel_stride) +
        std::string(" -DMLO_OUT_STRIDE=") + std::to_string(params.in_stride) +
        std::string(" -DMLO_IN_BATCH_STRIDE=") + std::to_string(params.out_batch_stride) +
        std::string(" -DMLO_IN_CHANNEL_STRIDE=") + std::to_string(params.out_channel_stride) +
        std::string(" -DMLO_IN_STRIDE=") + std::to_string(params.out_stride) +
        std::string(" -DMLO_WEI_BATCH_STRIDE=") + std::to_string(wei_bstride) +
        std::string(" -DMLO_WEI_CHANNEL_STRIDE=") + std::to_string(wei_cstride) +
        std::string(" -DMLO_IN_WIDTH=") + std::to_string(params.out_width) +
        std::string(" -DMLO_IN_HEIGHT=") + std::to_string(params.out_height) +
        std::string(" -DMLO_OUT_WIDTH=") + std::to_string(params.in_width) +
        std::string(" -DMLO_OUT_HEIGHT=") + std::to_string(params.in_height) +
        std::string(" -DMLO_IN_TILE1=") + std::to_string(result.in_tile1) +
        std::string(" -DMLO_IN_TILE0=") + std::to_string(result.in_tile0) +
        std::string(" -DMLO_N_LCL_BATCHS=") +
        std::to_string(result.n_stacks) // # of diff stacks (part of batch).
        + std::string(" -DMLO_N_LCL_OUT_MAPS=") +
        std::to_string(result.n_out_pix_tiles) // # output pixel tiles per wk-item (ALU)
        + std::string(" -DMLO_N_LCL_IN_MAPS=") +
        std::to_string(result.n_in_data_tiles) // total # of blocks of different inputs in LDS
        + std::string(" -DMLO_OUT_TILE0=") +
        std::to_string(result.out_pix_tile0) // size of ouptput tile per wk-item (ALU)
        + std::string(" -DMLO_OUT_TILE1=") + std::to_string(result.out_pix_tile1) //
        + std::string(" -DMLO_OUT_STACKS=") + std::to_string(n_out_stacks) +
        std::string(" -DMLO_N_WAVES=") + std::to_string(n_waves) +
        std::string(" -DMLO_READ_TYPE=") + READ_TYPE + std::string(" -DMLO_READ_UNIT=") +
        std::to_string(read_unit) + std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(hw_wave_sz) +
        std::string(" -DMLO_LG2_PHYS_WAVE_SZ=") + std::to_string(mloLg2(hw_wave_sz)) +
        std::string(" -DMLO_IN_EXTENT1=") + std::to_string(out_n_vert_reads) +
        std::string(" -DMLO_IN_N_VERT_LOOPS=") + std::to_string(out_n_vert_read_loops) +
        std::string(" -DMLO_IN_WIDTH_CHUNK=") +
        std::to_string((out_n_horizon_read_loops == 1) ? params.out_width : out_n_horizon_reads) +
        std::string(" -DMLO_IN_WIDTH_N_LOOPS=") + std::to_string(out_n_horizon_read_loops) +
        std::string(" -DMLO_IN_WIDTH_LAST_CHUNK_VALID_READ_UNITS=") +
        std::to_string(out_horizon_last_chunk_valid_read_units) +
        std::string(" -DMLO_IN_WIDTH_LAST_CHUNK_VALID_PIXELS_IN_LAST_READ_UNIT=") +
        std::to_string(out_horizon_last_chunk_valid_pixels_in_last_read_unit) +
        std::string(" -DMLO_OUT_WIDTH_CHUNK=") + std::to_string(in_width_chunk) +
        std::string(" -DMLO_OUT_WIDTH_N_LOOPS=") + std::to_string(out_n_horizon_read_loops) +
        std::string(" -DMLO_OUT_WIDTH_LAST_CHUNK_VALID_SPANS=") +
        std::to_string(in_width_last_chunk_valid_spans) +
        std::string(" -DMLO_OUT_WIDTH_LAST_CHUNK_VALID_PIXELS_IN_LAST_SPAN=") +
        std::to_string(in_width_last_chunk_valid_pixels_in_last_span) +
        std::string(" -DMLO_CONV_BIAS=") + std::to_string(params.bias) +
        std::string(" -DMLO_UT_READ_TYPE=") + UT_READ_TYPE + std::string(" -DMLO_UT_READ_UNIT=") +
        std::to_string(ut_read_unit) + std::string(" -DMLO_UT_GRP_SZ0=") +
        std::to_string(UT_GRP_SZ0)

        //		+ std::string(" -limit-vector-registers=64 ")
        + params.general_compile_options;

    // On gfx908 hardware, the compiler doesn't seem to support #pragam unroll correctly
    // References: PR: #1962 and SWDEV-200074
    const auto name = params.GetStream().GetDeviceName();
    if(StartsWith(name, "gfx908"))
    {
        comp_options += " -DMLO_DISABLE_PRAGMA_UNROLL_COMPILER_SWDEV_200074_WORKAROUND=1";
    }

    // wrt to W
    {
        KernelInfo kernel;

        kernel.l_wk.push_back(result.grp_tile0);
        kernel.l_wk.push_back(result.grp_tile1);
        kernel.l_wk.push_back(grp_tile2);
        // input is output

        size_t gbl_wk1 = ((params.n_inputs + total_out_maps - 1) / total_out_maps);
        size_t gbl_wk2 = n_batch_blks;
        size_t gbl_wk0 = GRP_SZ;

        if(params.group_counts > 1)
        {
            gbl_wk0 *= (((params.n_outputs / params.group_counts) + result.n_in_data_tiles - 1) /
                        result.n_in_data_tiles);

            kernel.kernel_file = "MIOpenGroupConvBwdWrW_LxG_P53.cl";
            kernel.kernel_name = "MIOpenCvBwdWrW";
        }
        else
        {
            gbl_wk0 *= ((params.n_outputs + result.n_in_data_tiles - 1) / result.n_in_data_tiles);

            kernel.kernel_file = "MIOpenConvBwdWrW_LxG_P53.cl";
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

        kernel.kernel_file  = "MIOpenConvBwdWrW_LxG_P53.cl";
        kernel.kernel_name  = "MIOpenCvBwdWrW_rdc";
        kernel.comp_options = comp_options;

        kernel.l_wk.push_back(UT_GRP_SZ0);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);

        int gbl_ut_wk0 = wei_bstride * params.n_inputs / ut_read_unit;

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
} // namespace solver
} // namespace miopen
