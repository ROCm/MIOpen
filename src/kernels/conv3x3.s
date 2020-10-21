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

.if ROCM_METADATA_VERSION == 4
.hsa_code_object_version 2,1
.hsa_code_object_isa
.endif

.text
.globl miopenGcnAsmConv3x3U
.p2align 8
.type miopenGcnAsmConv3x3U,@function

.if ROCM_METADATA_VERSION == 4
.amdgpu_hsa_kernel miopenGcnAsmConv3x3U
.endif

.include "rocm_version.inc"
.include "gpr_alloc.inc"
.include "utilities.inc"
.include "conv_common.inc"
.include "inst_wrappers.inc"

// initial state (s[0:4] are overlapped with filtersA):
// s[0:1] - kernarg address
// s2 - wg x (output batch)
// s3 - wg y (line in feature map)
// s4 - wg z (image in minibatch)
kernarg = 0
gid_x = 2
gid_y = 3
gid_z = 4

// kernarg layout:
// dwords 0:1 - input buffer pointer
// dwords 2:3 - weights pointer
// dwords 4:5 - output buffer pointer
// dwords 6:7 - debug buffer pointer
.set in_ptr_off, 0x0
.set wei_ptr_off, 0x8
.set out_ptr_off, 0x10

.ifnotdef no_params_file
    //inliner-include-optional
    .include "params.ins"
    .ifndef params_file
        .set batch_size, 1
        .set img_width, 128
        .set img_height, 64
        .set input_channels, 8
        .set output_channels, 96
        .set output_lines_per_wave, 2
        .set filters_per_wave, 4
        .set weights_layout, 0 // 0 - KCHW, 1 - CKHW
        .set reverse_weights, 0 // for backward conv
        .set enable_debug_output, 0
        .set limit_wave_cnt, 0
    .endif
.endif

.set max_hw_wctn, 15
.set padding_x, 1
.set padding_y, 1
.set first_input_line_idx, 0
.set enable_zero_line_padding_on_read, 1
.set small_image, 0
.set disable_filter_prefetch, 0 // disable prefetch if we dont have enough sgprs to hold all weights
.set uneven_line_read_mode, 0
.set uneven_line_write_mode, 0

block_size_x = 1
enable_dpp_zero_column_padding = 1
block_size_y = 1

.set img_x_blocks, img_width
.set img_y_blocks, img_height
.set input_line_stride, 4 * img_x_blocks * block_size_x
.set input_feature_map_stride, 4 * img_x_blocks * img_y_blocks * block_size_x * block_size_y
.set output_line_stride, 4 * img_width
.set output_feature_map_stride, 4 * img_width * img_height

w64_chunks = (img_x_blocks + 63) / 64
active_lanes = (img_x_blocks + w64_chunks - 1) / w64_chunks
gprs_per_input_line = (img_x_blocks * block_size_x + active_lanes - 1) / active_lanes
gprs_per_output_line = (img_x_blocks + active_lanes - 1) / active_lanes
acc_x_blocks = gprs_per_input_line / block_size_x
acc_y_blocks = output_lines_per_wave
.if gprs_per_input_line % block_size_x
    .error "Fazoht!"
    .end
.endif
.if (w64_chunks * active_lanes) < img_x_blocks
    .error "Check w size"
    .end
.endif
.if img_x_blocks % active_lanes
    uneven_line_read_mode = 1
    full_input_chunks = img_x_blocks % w64_chunks
    partial_input_chunks = w64_chunks - full_input_chunks
.endif
.if img_width % active_lanes
    uneven_line_write_mode = 1
    full_output_chunks = img_width % gprs_per_output_line
    partial_output_chunks = gprs_per_output_line - full_output_chunks
.endif

.if img_height == output_lines_per_wave
    first_input_line_idx = 1
    enable_zero_line_padding_on_read = 0
.endif


.if img_height == output_lines_per_wave
    input_lines_per_wave = output_lines_per_wave
    acc_lines_per_wave = output_lines_per_wave
.else
    input_lines_per_wave = output_lines_per_wave + 2
    acc_lines_per_wave = output_lines_per_wave
.endif

.if uneven_line_read_mode
    mbufs_per_line = (full_input_chunks+3) / 4 + (partial_input_chunks+3)/4
.else
    mbufs_per_line = (gprs_per_input_line+3) / 4 // memory buffer instructions per line
.endif
.set mbufs_cnt, mbufs_per_line * input_lines_per_wave
.if mbufs_cnt > max_hw_wctn
    mbufs_cnt = max_hw_wctn
.endif

// compute how many lines can we load or store using immediate 12-bit instruction offset
// for all other lines we need additional sgpr to hold offset
input_lines_per_sgpr = 1 + (0x1000-1-4*mbufs_per_line) / input_line_stride
.if input_lines_per_sgpr == 1
    additional_input_sgprs = input_lines_per_wave - 1
.else
    additional_lines = input_lines_per_wave - input_lines_per_sgpr
    .if additional_lines <= 0
        additional_input_sgprs = 0
    .else
        .if enable_zero_line_padding_on_read
            input_lines_per_additional_sgpr = input_lines_per_sgpr - 1
        .else
            input_lines_per_additional_sgpr = input_lines_per_sgpr
        .endif
        additional_input_sgprs = (additional_lines + input_lines_per_additional_sgpr - 1) / input_lines_per_additional_sgpr
    .endif
.endif

output_lines_per_sgpr = 1 + (0x1000-1-4*mbufs_per_line) / output_line_stride
.if output_lines_per_sgpr == 1
    additional_output_sgprs = output_lines_per_wave
.else
    additional_lines = output_lines_per_wave - output_lines_per_sgpr
    .if additional_lines <= 0
        additional_output_sgprs = 0
    .else
        additional_output_sgprs = (additional_lines + output_lines_per_sgpr - 1) / output_lines_per_sgpr
    .endif
.endif


.if (!enable_zero_line_padding_on_read) && (input_lines_per_sgpr / (img_y_blocks * block_size_y) > 1) && (input_channels % 2 == 0) && (img_y_blocks % acc_y_blocks == 0)
    small_image = 1
    linesB_start_offset = input_feature_map_stride
    input_ptr_step = input_feature_map_stride * 2
.else
    linesB_start_offset = 0
    input_ptr_step = input_feature_map_stride
.endif
input_buffer_window = input_ptr_step
output_buffer_window = 4 * img_height * img_width
.if output_channels % filters_per_wave
    load_weights_using_buffer = 1
    uneven_outputs = 1
.else
    load_weights_using_buffer = 0
    uneven_outputs = 0
.endif

// weights
.set weights_w, 3
.set weights_h, 3
.set weights_per_filter, weights_w * weights_h
.if weights_layout == 0 // KCHW
    .set filter_c_stride, 4 * weights_per_filter
    .set filter_k_stride, 4 * weights_per_filter * input_channels / group_counts
.else // CKHW
    .set filter_c_stride, 4 * weights_per_filter * output_channels / group_counts
    .set filter_k_stride, 4 * weights_per_filter
.endif
.set filters_size, 4 * weights_per_filter * input_channels * output_channels / group_counts


.GPR_ALLOC_BEGIN
.SGPR_ALLOC_FROM 0
.if limit_wave_cnt
  .SET_MAX_WAVES_LIMIT limit_wave_cnt
.endif
//xnack disabled by default
//.SGPR_RESERVE_XNACK

// allocate filters
filter_part_size = weights_per_filter & 0x3
filter_base_size = weights_per_filter - filter_part_size
.if filter_part_size == 3
    .error "Unsupported filter size"
    .end
.endif
sgprs_to_allocate_after_filters = 4 + 2 + 2*load_weights_using_buffer + 2 + 3 + additional_input_sgprs
.if weights_layout == 0  // KCHW
    sgprs_to_allocate_for_filters = 2 * filters_per_wave * weights_per_filter
.else  // CKHW
    sgprs_to_allocate_for_filters = filters_per_wave * weights_per_filter + ((filters_per_wave * weights_per_filter + 3) / 4) * 4
.endif
.if sgprs_to_allocate_after_filters + sgprs_to_allocate_for_filters > .AVAILABLE_SGPRS
    disable_filter_prefetch = 1
.endif
.if weights_layout == 0 // KCHW
    .if disable_filter_prefetch
        .SGPR_ALLOC filtersA, weights_per_filter * filters_per_wave
        filtersA_part = filtersA + filter_base_size * filters_per_wave
        filtersB = filtersA
        filtersB_part = filtersA_part
    .else
        .SGPR_ALLOC filtersA, filter_base_size * filters_per_wave
        .if filter_part_size == 2
            .SGPR_ALLOC filtersA_part, 2 * filters_per_wave, 2
            .SGPR_ALLOC filtersB_part, 2 * filters_per_wave, 2
        .endif
        .SGPR_ALLOC filtersB, filter_base_size * filters_per_wave
        .if filter_part_size == 1
            .SGPR_ALLOC filtersA_part, 1 * filters_per_wave, 1
            .SGPR_ALLOC filtersB_part, 1 * filters_per_wave, 1
        .endif
    .endif
    .if filter_part_size == 0
        filtersA_part = 0xFFFF //should generate error if filters_part is used
        filtersB_part = 0xFFFF
    .endif
.else // CKHW
    .if disable_filter_prefetch
        .SGPR_ALLOC filtersA, weights_per_filter * filters_per_wave
        filtersB = filtersA
    .else
        .SGPR_ALLOC filtersA, weights_per_filter * filters_per_wave
        padding = (4 - (.SGPR_NEXT_FREE & 0x3)) & 0x3
        .SGPR_ALLOC_FROM (.SGPR_NEXT_FREE+padding) // padding to align filtersB to 4 sgprs
        .SGPR_ALLOC filtersB, weights_per_filter * filters_per_wave
    .endif
    filtersA_part = 0xFFFF // should generate error if filters_part is used
    filtersB_part = 0xFFFF
.endif


__sgprs_ptr = .SGPR_NEXT_FREE
.if .SGPR_NEXT_FREE % 2
    .SGPR_ALLOC_ONCE tmp
.endif
.if .SGPR_NEXT_FREE % 4
    .SGPR_ALLOC_ONCE out_ptr, 2
.endif
.SGPR_ALLOC in_desc, 4 // input buffer descriptor
.SGPR_ALLOC weights_ptr, 2  // weights_ptr
.if load_weights_using_buffer
    .SGPR_ALLOC wei_desc_hi, 2
    out_k = wei_desc_hi+1 // gfx8 hack: last V# reg is ignored for s_buffer_load
.endif
.SGPR_ALLOC_ONCE out_ptr, 2
.SGPR_ALLOC_ONCE loop_cnt
.SGPR_ALLOC_ONCE tmp
.SGPR_ALLOC img_offset, 1 + additional_input_sgprs, 1  // img_offset[0] - offset of the first line to read

__sgprs_allocated_after_filters = .SGPR_NEXT_FREE - __sgprs_ptr
.if sgprs_to_allocate_after_filters != __sgprs_allocated_after_filters
    .error "Error: check sgpr allocation"
    .end
.endif

.macro .get_wei w, base, part, k, x, y
    .if reverse_weights
        fx = weights_w - 1 - \x
        fy = weights_h - 1 - \y
    .else
        fx = \x
        fy = \y
    .endif
    wei_id = fy * weights_w + fx
    .if weights_layout == 1  // CKHW
        \w = \base + \k * weights_per_filter + wei_id
    .elseif wei_id < filter_base_size // KCHW
        \w = \base + \k * filter_base_size + wei_id
    .else  // KCHW
        \w = \part + \k * filter_part_size + wei_id - filter_base_size
    .endif
.endm



.VGPR_ALLOC_FROM 0
    .VGPR_ALLOC tid
.if enable_zero_line_padding_on_read
    .VGPR_ALLOC in_off // input start line offset (for zero padding)
.endif
.if uneven_line_read_mode || uneven_line_write_mode
    .VGPR_ALLOC in_off_p
.endif

// input lines
.if k_group_size_is_power_of_two || gprs_per_input_line * input_lines_per_wave >= 4
    .VGPR_ALLOC linesA, gprs_per_input_line * input_lines_per_wave
.else
    .VGPR_ALLOC linesA, 4
.endif

.if k_group_size_is_power_of_two || gprs_per_input_line * input_lines_per_wave >= 3
    .VGPR_ALLOC linesB, gprs_per_input_line * input_lines_per_wave
.else
    .VGPR_ALLOC linesB, 3
.endif

.if enable_dpp_zero_column_padding
    .VGPR_ALLOC in_l // part of input line shifted left with zero padding
    .VGPR_ALLOC in_r // part of input line shifted right with zero padding
.endif

// output accumulators
.VGPR_ALLOC accums, gprs_per_input_line * filters_per_wave * acc_lines_per_wave
.macro .get_accum name, k, line
  .set \name, accums + (\k * acc_lines_per_wave + \line) * gprs_per_input_line
.endm

.if enable_debug_output
  .VGPR_ALLOC dbg_ptr, 2
  .VGPR_ALLOC dbg, 16
  .SGPR_ALLOC dbg_exec_lo
  .SGPR_ALLOC dbg_exec_hi
  .SGPR_RESERVE_VCC
.endif

.GPR_ALLOC_END


.macro .single_sload base, count
    .if ((vals_to_load - \count) >= 0) && vals_to_load > 0
        .if woff >= (1 << 20)
            .error "Error: Immediate offset is too large for s_load instruction"
            .end
        .endif
        .if load_weights_using_buffer
            .if \count == 1
                s_buffer_load_dword s[\base+vals_loaded], s[wei_desc:wei_desc+3], 0 + woff
            .else
                s_buffer_load_dwordx\count s[\base+vals_loaded:\base+vals_loaded+\count-1], s[wei_desc:wei_desc+3], 0 + woff
            .endif
        .else
            .if \count == 1
                s_load_dword s[\base+vals_loaded], s[weights_ptr:weights_ptr+1], 0 + woff
            .else
                s_load_dwordx\count s[\base+vals_loaded:\base+vals_loaded+\count-1], s[weights_ptr:weights_ptr+1], 0 + woff
            .endif
        .endif
        vals_to_load = vals_to_load - \count
        vals_loaded = vals_loaded + \count
        woff = woff + 4 * \count
    .endif
.endm

.macro .load_sgprs base, count
    .if (\count) > 0
        vals_to_load = \count
        vals_loaded = 0
        .rept (\count / 16)
            .single_sload \base, 16
        .endr
        .single_sload \base, 8
        .single_sload \base, 4
        .single_sload \base, 2
        .single_sload \base, 1
    .endif
.endm

.macro .load_filters base, part, w_offset
    woff = \w_offset
    .if weights_layout == 0 // KCHW
        b_base = \base
        b_part = \part
        .rept filters_per_wave
            .load_sgprs b_base, filter_base_size
            .load_sgprs b_part, filter_part_size
            //woff = \w_offset + i_k * filter_k_stride
            woff = woff + filter_k_stride - weights_per_filter * 4
            b_base = b_base + filter_base_size
            b_part = b_part + filter_part_size
        .endr
    .else // CKHW
        .load_sgprs \base, weights_per_filter * filters_per_wave
    .endif
.endm


.macro .convx3_line in0, acc0, w_base, w_part, fline, k
    .get_wei f0, \w_base, \w_part, \k, 0, \fline
    .get_wei f1, \w_base, \w_part, \k, 1, \fline
    .get_wei f2, \w_base, \w_part, \k, 2, \fline
    gpr_in_line = 0
    .rept gprs_per_input_line
        cur_input = \in0 + gpr_in_line
            .if (gpr_in_line == 0)
                prev_input = in_r
            .else
                prev_input = \in0 + gpr_in_line - 1
            .endif
            .if (gpr_in_line+1) < gprs_per_input_line
                next_input = \in0 + gpr_in_line + 1
            .else
                next_input = in_l
            .endif
            v_mac_f32 v[\acc0 + gpr_in_line], s[f0], v[prev_input]
            v_mac_f32 v[\acc0 + gpr_in_line], s[f1], v[cur_input]
            v_mac_f32 v[\acc0 + gpr_in_line], s[f2], v[next_input]
        gpr_in_line = gpr_in_line + 1
    .endr
.endm


.macro .conv3x3 in_base, w_base, w_part
    iline = 0
    .rept input_lines_per_wave // iterate over input lines
        in0 = \in_base + gprs_per_input_line * iline
        .if enable_dpp_zero_column_padding
            v_mov_b32 v[in_l], v[in0] wave_shl:1 bound_ctrl:0
            v_mov_b32 v[in_r], v[in0+gprs_per_input_line-1] wave_shr:1 bound_ctrl:0
        .endif

        k = 0
        .rept filters_per_wave // iterate over output channels
            aline = 0
            .rept acc_lines_per_wave // iterate over acc lines
                .get_accum acc0, k, aline
                    fline = iline - aline + first_input_line_idx
                    .if (fline >= 0) && (fline <= 2)
                        .convx3_line in0, acc0, \w_base, \w_part, fline, k
                    .endif
                aline = aline + 1
            .endr
            k = k + 1
        .endr

        iline = iline + 1
    .endr
.endm

.macro .single_vload base, s_offset, count, partial=0
    .if ((vals_to_load - \count) >= 0) && vals_to_load > 0
        .if imm_off >= (1 << 12)
            .error "Error: Immediate offset is too large for buffer_load instruction"
            .end
        .endif

        .if \count == 1
            .if \partial
                buffer_load_dword v[\base+vals_loaded], v[in_off_p], s[in_desc:in_desc+3], s[\s_offset] offen offset:0+imm_off
            .elseif enable_zero_line_padding_on_read
                buffer_load_dword v[\base+vals_loaded], v[in_off], s[in_desc:in_desc+3], s[\s_offset] offen offset:0+imm_off
            .else
                buffer_load_dword v[\base+vals_loaded], v[voffset], s[in_desc:in_desc+3], s[\s_offset] offen offset:0+imm_off
            .endif
        .else
            .if \partial
                buffer_load_dwordx\count v[\base+vals_loaded:\base+vals_loaded+\count-1], v[in_off_p], s[in_desc:in_desc+3], s[\s_offset] offen offset:0+imm_off
            .elseif enable_zero_line_padding_on_read
                buffer_load_dwordx\count v[\base+vals_loaded:\base+vals_loaded+\count-1], v[in_off], s[in_desc:in_desc+3], s[\s_offset] offen offset:0+imm_off
            .else
                buffer_load_dwordx\count v[\base+vals_loaded:\base+vals_loaded+\count-1], v[voffset], s[in_desc:in_desc+3], s[\s_offset] offen offset:0+imm_off
            .endif
        .endif

        vals_to_load = vals_to_load - \count
        vals_loaded = vals_loaded + \count
        imm_off = imm_off + 4 * \count
    .endif
.endm

.macro .load_input_line base, s_offset
    vals_to_load = gprs_per_input_line
    .if uneven_line_read_mode
        vals_to_load = full_input_chunks
    .endif
    vals_loaded = 0
    .rept (gprs_per_input_line / 4)
        .single_vload \base, \s_offset, 4
    .endr
    .single_vload \base, \s_offset, 3
    .single_vload \base, \s_offset, 2
    .single_vload \base, \s_offset, 1
    .if uneven_line_read_mode
        vals_to_load = partial_input_chunks
        .rept (gprs_per_input_line / 4)
            .single_vload \base, \s_offset, 4, 1
        .endr
        .single_vload \base, \s_offset, 3, 1
        .single_vload \base, \s_offset, 2, 1
        .single_vload \base, \s_offset, 1, 1
    .endif
    imm_off = imm_off + input_line_stride - 4 * gprs_per_input_line
.endm

.macro .load_input_lines_on_same_sgpr count
    .rept \count
        .if lines_to_load > 0
            .load_input_line line_base, s_off
            lines_to_load = lines_to_load -1
            line_base = line_base + gprs_per_input_line
        .endif
    .endr
    s_off = s_off + 1
.endm

.macro .load_input base, start_offset=0
    lines_to_load = input_lines_per_wave
    line_base = \base
    imm_off = \start_offset
    s_off = img_offset
    .load_input_lines_on_same_sgpr input_lines_per_sgpr

    .rept additional_input_sgprs
        .if enable_zero_line_padding_on_read
            imm_off = input_line_stride
        .else
            imm_off = 0
        .endif
        .load_input_lines_on_same_sgpr input_lines_per_additional_sgpr
    .endr
.endm

.macro .move_input_ptr, num // 0 - linesA, 1 - linesB
    .if (\num == 1) || !small_image
        s_add_u32 s[in_desc], s[in_desc], input_ptr_step
        s_addc_u32 s[in_desc+1], s[in_desc+1], 0
    .endif
.endm

.macro .move_wei_ptr, offset
    .if load_weights_using_buffer
        s_add_u32 s[wei_desc], s[wei_desc], \offset
        s_addc_u32 s[wei_desc+1], s[wei_desc+1], 0
        s_sub_u32 s[wei_desc+2], s[wei_desc+2], \offset
    .else
        s_add_u32 s[weights_ptr], s[weights_ptr], \offset
        s_addc_u32 s[weights_ptr+1], s[weights_ptr+1], 0
    .endif
.endm


//.text 0
//.p2align 8
miopenGcnAsmConv3x3U:

.if ROCM_METADATA_VERSION == 4
  .amd_kernel_code_t
     enable_sgpr_kernarg_segment_ptr = 1
     compute_pgm_rsrc2_tgid_x_en = 1
     compute_pgm_rsrc2_tgid_y_en = 1
     compute_pgm_rsrc2_tgid_z_en = 1
     is_ptr64 = 1
     compute_pgm_rsrc1_vgprs = .AUTO_VGPR_GRANULATED_COUNT
     compute_pgm_rsrc1_sgprs = .AUTO_SGPR_GRANULATED_COUNT
     compute_pgm_rsrc2_tidig_comp_cnt = 1
     compute_pgm_rsrc2_user_sgpr = 2
     kernarg_segment_byte_size = 56
     wavefront_sgpr_count = .AUTO_SGPR_COUNT
     workitem_vgpr_count = .AUTO_VGPR_COUNT
     float_mode = 192
     workgroup_group_segment_byte_size = 0
  .end_amd_kernel_code_t
.endif

.if accums < linesA || accums < linesB || linesA == linesB
    .error "Error: check vgpr allocation"
    // in case of back transformation data will be moved to
    // lower registers, so accums should be allocated after linesA and linesB
    .end
.endif

//.text 1
//.p2align 8
//.Lfunc_start0:
  // debug
  .if enable_debug_output
    s_load_dwordx2 s[6:7], s[kernarg:kernarg+1], 0x18 // load debug buffer pointer
    s_waitcnt 0
    // compute per lane address
    s_mov_b32 s[dbg_exec_lo], exec_lo
    s_mov_b32 s[dbg_exec_hi], exec_hi
    s_mov_b64 exec, -1
    v_mbcnt_lo_u32_b32 v[dbg_ptr], -1, 0
    v_mbcnt_hi_u32_b32 v[dbg_ptr], -1, v[dbg_ptr]
    v_mul_u32_u24 v[dbg_ptr], v[dbg_ptr], 4
    v_mov_b32 v[dbg_ptr+1], s[7]
   _v_add_co_u32 v[dbg_ptr], vcc, v[dbg_ptr], s[6]
    v_addc_u32 v[dbg_ptr+1], vcc, v[dbg_ptr+1], 0, vcc
    s_mov_b32 exec_lo, s[dbg_exec_lo]
    s_mov_b32 exec_hi, s[dbg_exec_hi]
    s_mov_b32 s[gid_x], debug_gid_x //debug output batch
    s_mov_b32 s[gid_y], debug_gid_y //debug line batch
    s_mov_b32 s[gid_z], debug_gid_z //debug image
  .endif

  num_wavefronts = output_channels / filters_per_wave
  //to-do add support of uneven_outputs into grouped conv
  static_assert(group_counts == 1 || ((num_wavefronts % group_counts == 0) && uneven_outputs == 0))
  static_assert(input_channels % (4 * group_counts) == 0) //to-do remove restriction that (n_inputs/group_counts) must be multiple of 4
  k_group_size = output_channels / filters_per_wave / group_counts
  c_group_size = input_channels / group_counts
  s_group_id = loop_cnt
  .if k_group_size_is_power_of_two
      log2 k_group_size_log2, k_group_size
      s_lshr_b32 s[s_group_id], s[gid_x], 0 + k_group_size_log2 // group_id
  .else
      stmp_udiv = in_desc
      vtmp_udiv = linesA
      gid = linesB
      v_group_id = linesB + 1
      group_size = linesB + 2
      v_mov_b32 v[gid], s[gid_x]
      v_mov_b32 v[group_size], 0 + k_group_size
      u32_div gid, group_size, v_group_id, vtmp_udiv, stmp_udiv
      v_readfirstlane_b32 s[s_group_id], v[v_group_id]
  .endif

  s_load_dwordx2 s[in_desc:in_desc+1], s[kernarg:kernarg+1], 0x0 + in_ptr_off
  s_load_dwordx2 s[weights_ptr:weights_ptr+1], s[kernarg:kernarg+1], 0x0 + wei_ptr_off
  s_load_dwordx2 s[out_ptr:out_ptr+1], s[kernarg:kernarg+1], 0x0 + out_ptr_off

  v_and_b32 v[tid], 0x3f, v[tid]
  v_mul_u32_u24 v[tid], v[tid], 4 * gprs_per_output_line
  .GPR_REUSE tid, voffset
  // compute offsets for input
  .if enable_zero_line_padding_on_read
    s_cmpk_eq_u32 s[gid_y], 0
    s_cselect_b32 s[img_offset], 0, -1 * input_line_stride
    s_cselect_b32 s[tmp], -1 * input_line_stride, 0
    v_mov_b32 v[in_off], s[tmp]
   _v_add_nc_u32 v[in_off], v[in_off], v[voffset]
    s_mul_i32 s[tmp], s[gid_y], 0 + input_line_stride * acc_lines_per_wave
    s_add_u32 s[img_offset], s[img_offset], s[tmp]
  .else
    s_mul_i32 s[img_offset], s[gid_y], 0 + input_line_stride * acc_lines_per_wave
  .endif
  i=0
  .rept additional_input_sgprs
    offset_in_lines = input_lines_per_sgpr + i * input_lines_per_additional_sgpr - enable_zero_line_padding_on_read
    s_add_u32 s[img_offset+i+1], s[img_offset], 0 + input_line_stride * offset_in_lines
    i=i+1
  .endr

  // construct v[in_off_p] that is used to mask out unnecessary memory operations when number of elements to read/write is not a multiple of active lanes
  .if uneven_line_read_mode || uneven_line_write_mode
    v_mov_b32 v[in_off_p], 0x7fffFFFF
    .if ((0x7fffFFFF - input_line_stride)/2) < input_feature_map_stride
        .error "Error: feature map is too big"
        .end
    .endif
    .if active_lanes > 32
        last_active_lane_mask = 1 << (active_lanes - 1 - 32)
        s_xor_b32 exec_hi, exec_hi, last_active_lane_mask
        v_mov_b32 v[in_off_p], 0
        s_xor_b32 exec_hi, exec_hi, last_active_lane_mask
    .else
        last_active_lane_mask = 1 << (active_lanes - 1)
        s_xor_b32 exec_lo, exec_lo, last_active_lane_mask
        v_mov_b32 v[in_off_p], 0
        s_xor_b32 exec_lo, exec_lo, last_active_lane_mask
    .endif
    .if enable_zero_line_padding_on_read
       _v_add_nc_u32 v[in_off_p], v[in_off_p], v[in_off]
    .else
       _v_add_nc_u32 v[in_off_p], v[in_off_p], v[voffset]
    .endif
  .endif

  s_waitcnt 0

  // compute offset for weights
  .if weights_layout == 0 || group_counts == 1
      s_mul_i32 s[tmp], s[gid_x], filters_per_wave * filter_k_stride
      s_add_u32 s[weights_ptr], s[weights_ptr], s[tmp]
      s_addc_u32 s[weights_ptr+1], s[weights_ptr+1], 0
      .if load_weights_using_buffer
        .GPR_REUSE weights_ptr, wei_desc
        s_sub_u32 s[wei_desc+2], filters_size, s[tmp]
      .endif
  .else
      s_mul_i32 s[tmp], s[s_group_id], k_group_size
      s_sub_i32 s[tmp], s[gid_x], s[tmp]
      s_mul_i32 s[tmp], s[tmp], filters_per_wave * filter_k_stride
      s_add_u32 s[weights_ptr], s[weights_ptr], s[tmp]
      s_addc_u32 s[weights_ptr+1], s[weights_ptr+1], 0
      .if load_weights_using_buffer
        .GPR_REUSE weights_ptr, wei_desc
        s_sub_u32 s[wei_desc+2], filters_size, s[tmp]
      .endif
      s_mul_i32 s[tmp], s[s_group_id], c_group_size * filter_c_stride // c_group_offset
      s_add_u32 s[weights_ptr], s[weights_ptr], s[tmp]
      s_addc_u32 s[weights_ptr+1], s[weights_ptr+1], 0
  .endif

  // construct input buffer descriptor
  .if batch_size > 1
    s_mul_i32 s[tmp], s[gid_z], input_feature_map_stride * input_channels
    s_add_u32 s[in_desc], s[in_desc], s[tmp] // add input image batch offset
    s_addc_u32 s[in_desc+1], s[in_desc+1], 0
  .endif
  s_mov_b32 s[in_desc+2], input_buffer_window // size
  s_mov_b32 s[in_desc+3], 0x00027000

.if group_counts > 1
  s_mul_i32 s[tmp], s[s_group_id], c_group_size * input_feature_map_stride // c_group_offset
  s_add_u32 s[in_desc], s[in_desc], s[tmp]
  s_addc_u32 s[in_desc+1], s[in_desc+1], 0
.endif

  .if uneven_outputs
    s_mul_i32 s[out_k], s[gid_x], filters_per_wave
  .endif
  s_mul_i32 s[tmp], s[gid_x], output_feature_map_stride * filters_per_wave // output image filter offset
  .if batch_size > 1 // add output image batch offset
    s_mul_i32 s[gid_z], s[gid_z], output_feature_map_stride * output_channels
    s_add_u32 s[tmp], s[tmp], s[gid_z]
  .endif
  s_add_u32 s[out_ptr], s[out_ptr], s[tmp]
  s_addc_u32 s[out_ptr+1], s[out_ptr+1], 0
  s_mul_i32 s[tmp], s[gid_y], output_line_stride * output_lines_per_wave // output line offset
  .GPR_REUSE tmp, out_img_off
  .GPR_INVALIDATE gid_x
  .GPR_INVALIDATE gid_y
  .GPR_INVALIDATE gid_z

  .GPR_INVALIDATE gid
  .GPR_INVALIDATE group_size
  .GPR_INVALIDATE s_group_id
  .GPR_INVALIDATE v_group_id
  .GPR_INVALIDATE vtmp_udiv


  // zeroing acc
  i=accums
  .rept gprs_per_input_line * filters_per_wave * acc_lines_per_wave
    v_mov_b32 v[i], 0
    i = i + 1
  .endr

  // zeroing loop counter
  s_mov_b32 s[loop_cnt], 0

.if disable_filter_prefetch
  .load_filters filtersA, filtersA_part, 0
  .load_input linesA
  .move_input_ptr 0

loop_begin:

  .load_input linesB, linesB_start_offset
  .move_input_ptr 1

  s_waitcnt vmcnt(1*mbufs_cnt) & lgkmcnt(0)

  .conv3x3 linesA, filtersA, filtersA_part
  // load 2nd set of filters and adjust weights pointer
  .load_filters filtersB, filtersB_part, filter_c_stride
  .move_wei_ptr filter_c_stride * 2

  // load input data and move ptr to next feature map
  .load_input linesA
  .move_input_ptr 0

  s_waitcnt vmcnt(1*mbufs_cnt) & lgkmcnt(0)
  .conv3x3 linesB, filtersB, filtersB_part
  .load_filters filtersA, filtersA_part, 0

loop_end:
  s_addk_i32 s[loop_cnt], 1
  s_cmpk_ge_u32 s[loop_cnt], 0 + c_group_size/2 - 1
  s_cbranch_scc0 loop_begin

  .load_input linesB, linesB_start_offset
  s_waitcnt vmcnt(1*mbufs_cnt) & lgkmcnt(0)
  .conv3x3 linesA, filtersA, filtersA_part
  .load_filters filtersB, filtersB_part, filter_c_stride
  s_waitcnt 0
  .conv3x3 linesB, filtersB, filtersB_part
.else
  // load input data and move ptr to next feature map
  .load_input linesA
  .move_input_ptr 0

  // load set of filters
  .load_filters filtersA, filtersA_part, 0

loop_begin:

  // load 2nd feature map and move to next feature map
  .load_input linesB, linesB_start_offset
  .move_input_ptr 1

  s_waitcnt vmcnt(1*mbufs_cnt) & lgkmcnt(0)

  // load 2nd set of filters and adjust weights pointer
  .load_filters filtersB, filtersB_part, filter_c_stride
  .move_wei_ptr filter_c_stride * 2

  // perform convolution A
  .conv3x3 linesA, filtersA, filtersA_part

  // load input data and move ptr to next feature map
  .load_input linesA
  .move_input_ptr 0

  s_waitcnt vmcnt(1*mbufs_cnt) & lgkmcnt(0)
  // load set of 4 filters
  .load_filters filtersA, filtersA_part, 0

  // perform convolution B
  .conv3x3 linesB, filtersB, filtersB_part

loop_end:
  s_addk_i32 s[loop_cnt], 1
  s_cmpk_ge_u32 s[loop_cnt], 0 + c_group_size/2 - 1
  s_cbranch_scc0 loop_begin


  // load 2nd feature map and move to next feature map
  .load_input linesB, linesB_start_offset
  s_waitcnt vmcnt(1*mbufs_cnt) & lgkmcnt(0)
  .load_filters filtersB, filtersB_part, filter_c_stride
  .conv3x3 linesA, filtersA, filtersA_part
  s_waitcnt 0
  .conv3x3 linesB, filtersB, filtersB_part
.endif

  // construct output descriptor
  .GPR_REUSE in_desc, out_desc
  s_mov_b64 s[out_desc:out_desc+1], s[out_ptr:out_ptr+1]
  s_mov_b32 s[out_desc+2], output_buffer_window
  i = 0
  .rept additional_output_sgprs
    i = i + 1
    s_add_u32 s[img_offset+i], s[out_img_off], 0 + output_line_stride * output_lines_per_sgpr * i
  .endr

  // store output
.macro .single_vstore base, s_offset, count, partial=0
    .if ((vals_to_store - \count) >= 0) && vals_to_store > 0
        .if imm_off >= (1 << 12)
            .error "Error: Immediate offset is too large for buffer_store instruction"
            .end
        .endif
        .if \count == 1
            .if \partial
                buffer_store_dword v[\base+vals_stored], v[in_off_p], s[out_desc:out_desc+3], s[\s_offset] offen offset:0+imm_off
            .else
                buffer_store_dword v[\base+vals_stored], v[voffset], s[out_desc:out_desc+3], s[\s_offset] offen offset:0+imm_off
            .endif
        .else
            .if \partial
                buffer_store_dwordx\count v[\base+vals_stored:\base+vals_stored+\count-1], v[in_off_p], s[out_desc:out_desc+3], s[\s_offset] offen offset:0+imm_off
            .else
                buffer_store_dwordx\count v[\base+vals_stored:\base+vals_stored+\count-1], v[voffset], s[out_desc:out_desc+3], s[\s_offset] offen offset:0+imm_off
            .endif
        .endif
        vals_to_store = vals_to_store - \count
        vals_stored = vals_stored + \count
        imm_off = imm_off + 4 * \count
    .endif
.endm

.macro .store_output_line base, s_offset
    .if uneven_line_write_mode
        vals_to_store = full_output_chunks
    .else
        vals_to_store = gprs_per_output_line
    .endif
    vals_stored = 0
    .rept (gprs_per_output_line / 4)
        .single_vstore \base, \s_offset, 4
    .endr
    .single_vstore \base, \s_offset, 3
    .single_vstore \base, \s_offset, 2
    .single_vstore \base, \s_offset, 1

    .if uneven_line_write_mode
        vals_to_store = partial_output_chunks
        .rept (gprs_per_output_line / 4)
            .single_vstore \base, \s_offset, 4, 1
        .endr
        .single_vstore \base, \s_offset, 3, 1
        .single_vstore \base, \s_offset, 2, 1
        .single_vstore \base, \s_offset, 1, 1
    .endif
    imm_off = imm_off + output_line_stride - 4 * gprs_per_output_line
.endm

.macro .store_output_lines_on_same_sgpr s_offset
    .rept output_lines_per_sgpr
        .if lines_to_store > 0
            .store_output_line line_base, \s_offset
            lines_to_store = lines_to_store -1
            line_base = line_base + gprs_per_output_line
        .endif
    .endr
    \s_offset = \s_offset + 1
.endm

.macro .store_output base
    lines_to_store = output_lines_per_wave
    line_base = \base
    imm_off = 0
    s_off = out_img_off
    .store_output_lines_on_same_sgpr s_off

    s_off = img_offset + 1
    .rept additional_output_sgprs
        imm_off = 0
        .store_output_lines_on_same_sgpr s_off
    .endr
.endm

  .if uneven_line_write_mode && enable_zero_line_padding_on_read
   _v_sub_nc_u32 v[in_off_p], v[in_off_p], v[in_off]
   _v_add_nc_u32 v[in_off_p], v[in_off_p], v[voffset]
  .endif


  .GPR_REUSE accums, outputs

  // store output
  out0 = outputs
  .rept filters_per_wave-1
    .store_output out0
    //s_add_u32 s[out_desc], s[out_desc], output_feature_map_stride
    //s_addc_u32 s[out_desc+1], s[out_desc+1], 0
    s_add_u32 s[out_img_off], s[out_img_off], 0 + output_feature_map_stride
    i = 0
    .rept additional_output_sgprs
      i = i + 1
      s_add_u32 s[img_offset+i], s[img_offset+i], 0 + output_feature_map_stride
    .endr
    out0 = out0 + output_lines_per_wave * gprs_per_output_line

    .if uneven_outputs
        s_addk_i32 s[out_k], 1
        s_cmpk_lt_u32 s[out_k], 0 + output_channels
        tmp = wei_desc_hi
        s_cselect_b32 s[tmp], 0 + output_feature_map_stride, 0
        s_add_u32 s[out_desc+2], s[out_desc+2], s[tmp]
    .else
        s_add_u32 s[out_desc+2], s[out_desc+2], 0 + output_feature_map_stride
    .endif
  .endr
  .store_output out0

  s_endpgm

.Lfunc_end0:
    .size miopenGcnAsmConv3x3U, .Lfunc_end0 - miopenGcnAsmConv3x3U

.if ROCM_METADATA_VERSION == 5
.rodata
.p2align 6
.amdhsa_kernel miopenGcnAsmConv3x3U
    //enable_sgpr_kernarg_segment_ptr = 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1

    // compute_pgm_rsrc2_tgid_x_en = 1
    // compute_pgm_rsrc2_tgid_y_en = 1
    // compute_pgm_rsrc2_tgid_z_en = 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1

    .amdhsa_next_free_vgpr .AUTO_VGPR_COUNT
    .amdhsa_next_free_sgpr __amdhsa_next_free_sgpr
    .amdhsa_reserve_vcc __sgpr_reserve_vcc
    .amdhsa_reserve_xnack_mask __sgpr_reserve_xnack
    .amdhsa_reserve_flat_scratch __sgpr_reserve_flatscr

    // compute_pgm_rsrc2_tidig_comp_cnt = 1
    .amdhsa_system_vgpr_workitem_id 1

    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.altmacro
.macro METADATA sc,wc,wg_x
.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: miopenGcnAsmConv3x3U
    .symbol: miopenGcnAsmConv3x3U.kd
    .sgpr_count: \sc
    .vgpr_count: \wc
    .language: "OpenCL C"
    .language_version: [ 1, 2 ]
    .kernarg_segment_size: 56
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .max_flat_workgroup_size: \wg_x
    .args:
    - { .size: 8, .offset:  0, .value_kind: global_buffer, .value_type: f32, .name: in,      .address_space: global, .is_const: true }
    - { .size: 8, .offset:  8, .value_kind: global_buffer, .value_type: f32, .name: weights, .address_space: global, .is_const: true }
    - { .size: 8, .offset: 16, .value_kind: global_buffer, .value_type: f32, .name: out,     .address_space: global, .is_const: false }
    - { .size: 4, .offset: 24, .value_kind: by_value,      .value_type: f32, .name: padding_val }
    - { .size: 8, .offset: 32, .value_kind: hidden_global_offset_x, .value_type: i64 }
    - { .size: 8, .offset: 40, .value_kind: hidden_global_offset_y, .value_type: i64 }
    - { .size: 8, .offset: 48, .value_kind: hidden_global_offset_z, .value_type: i64 }
...
.end_amdgpu_metadata
.endm // METADATA

METADATA %.AUTO_SGPR_COUNT, %.AUTO_VGPR_COUNT, %workgroup_size_x

.elseif ROCM_METADATA_VERSION == 4
.macro METADATA wg_x
    .amd_amdgpu_hsa_metadata
    { Version: [ 1, 0 ],
        Kernels:
        - { Name: miopenGcnAsmConv3x3U, SymbolName: 'miopenGcnAsmConv3x3U@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
            Attrs:
              { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
            CodeProps:
              { KernargSegmentSize: 56, GroupSegmentFixedSize: 0, PrivateSegmentFixedSize: 0, KernargSegmentAlign: 8, WavefrontSize: 64, MaxFlatWorkGroupSize: \wg_x }
            Args:
            - { Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', Name: in,          AddrSpaceQual: Global, AccQual: Default, IsConst: true }
            - { Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', Name: weights,     AddrSpaceQual: Global, AccQual: Default, IsConst: true }
            - { Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', Name: out,         AddrSpaceQual: Global, AccQual: Default }
            - { Size: 4, Align: 4, ValueKind: ByValue,      ValueType: F32, TypeName:  float,   Name: padding_val,                        AccQual: Default }
            - { Size: 8, Align: 8, ValueKind: HiddenGlobalOffsetX, ValueType: I64 }
            - { Size: 8, Align: 8, ValueKind: HiddenGlobalOffsetY, ValueType: I64 }
            - { Size: 8, Align: 8, ValueKind: HiddenGlobalOffsetZ, ValueType: I64 }
          }
    }
    .end_amd_amdgpu_hsa_metadata
.endm // METADATA

.altmacro
.macro METADATA_WRAPPER wg_x
    METADATA %\wg_x
.endm

.ifnotdef workgroup_size_x
    .error "workgroup_size_x must be defined"
    .end
.endif

METADATA_WRAPPER workgroup_size_x
.endif
