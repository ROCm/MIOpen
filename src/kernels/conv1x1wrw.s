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
.if ROCM_METADATA_VERSION == 4
.hsa_code_object_version 2,1
.hsa_code_object_isa
.endif

.text
.globl miopenGcnAsmConv1x1WrW
.p2align 8
.type miopenGcnAsmConv1x1WrW,@function

.if ROCM_METADATA_VERSION == 4
.amdgpu_hsa_kernel miopenGcnAsmConv1x1WrW
.endif

.include "rocm_version.inc"
.include "gpr_alloc.inc"
.include "utilities.inc"
.include "conv_common.inc"
.include "inst_wrappers.inc"

// initial state (s[0:4] are overlapped with filtersA):
// s[0:1] - kernarg address
// s2 - wg x (none), wave_id bits: ck_id, n_id
// s3 - wg y (C)
// s4 - wg z (K)
kernarg = 0
gid_x = 2
gid_y = 3
gid_z = 4

// kernarg layout:
// dwords 0:4 - n, c, H, W, k
// dwords 5:7 - not used
// dwords 8:9 - input buffer pointer
// dwords 10:11 - weights pointer
// dwords 12:13 - output buffer pointer
.set in_ptr_off, 0x20
.set wei_ptr_off, 0x28
.set out_ptr_off, 0x30
.set unused_ptr_off, 0x38
.set KERNEL_ARGUMENTS_SIZE, unused_ptr_off + 8

// gfx90a requires 64bit aligned vgpr tuples
// Tuples are used only in buffer_load_dwordx/buffer_store_dwordx instructions
//
// To meet this requirement, the following approach is used ('buffer_load_dwordx4 v[x:y]' as an example):
//    if 'x' 64bit aligned:
//       buffer_load_dwordx4 v[x:y], ...
//    if 'x' not 64bit aligned:
//       buffer_load_dword   v[x], ...
//       buffer_load_dwordx3 v[x+1:y], ...
.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor == 0 && .amdgcn.gfx_generation_stepping == 10)
   tuple_alignment = 1
.else
   tuple_alignment = 0
.endif

.ifnotdef do_not_use_default_perf_params
    default n_per_gpr, 4 // 1..4, 2^n
    default c_per_gpr, 4 // 1..16, 2^n
    default c_mult, 2 // 1..16, 2^n
    default k_per_gpr, 4 // 1..16, 2^n
    default k_mult, 2 // 1..16, 2^n
    default read_size, 1 // 1..4
    default chunk_size, 4 // 1..16, 2^n
    default n_part_cnt, 1 //1..8
.endif
default limit_wave_cnt, 0
default hw_per_gpr, 1 // 1..4, 2^n
default short_store, 0
default data_prefetch, 0


group_size = n_part_cnt
static_assert (pad_h == 0 && pad_w == 0)
static_assert (stride_h == 1) // || stride_h == 2)
static_assert (stride_w == 1) // || stride_w == 2)
static_assert (wei_h == 1 && wei_w == 1)
static_assert (1 <= group_size && group_size <= 8)
static_assert (c_per_gpr * chunk_size >= 16)
static_assert (chunk_size == 1 || c_per_gpr * chunk_size <= 16) // todo: remove restriction
static_assert (c_per_gpr * n_per_gpr * hw_per_gpr * chunk_size == wave_size)
static_assert (k_per_gpr * n_per_gpr * hw_per_gpr * chunk_size <= wave_size)

static_assert ((.option.machine_version_major == 8) || (.option.machine_version_major == 9))
static_assert (filter_c_stride < maxU24)
static_assert (filter_k_stride < maxU24)
static_assert (input_c_stride < maxU24)
static_assert (output_k_stride < maxU24)

lds_element_stride = 4
dword_size = 4

elements_in_dword = 1
.if(buf_type == TYPE_FP16 || buf_type == TYPE_INT16 || buf_type == TYPE_BFP16)
    elements_in_dword = 2
.elseif(buf_type == TYPE_INT8)
    elements_in_dword = 4
.elseif(buf_type == TYPE_INT4)
    elements_in_dword = 8
.endif

elements_per_lane = read_size * elements_in_dword

is_required_sequential_c_channels = ((weights_layout == LAYOUT_DATA_NCHW) || (weights_layout == LAYOUT_DATA_NHWC)) && !short_store
is_required_sequential_k_channels = ((weights_layout == LAYOUT_DATA_CNHW) || (weights_layout == LAYOUT_DATA_CHWN)) && !short_store
sequential_c_channels = 1
sequential_k_channels = 1

.if(is_required_sequential_c_channels)
    sequential_c_channels = elements_in_dword
.elseif(is_required_sequential_k_channels)
    sequential_k_channels = elements_in_dword
.endif
static_assert(c_mult % sequential_c_channels == 0)
static_assert(k_mult % sequential_k_channels == 0)
static_assert(input_channels % sequential_c_channels == 0)
static_assert(output_channels % sequential_k_channels == 0)

bfp16_native_support = 0
dot_instructions_available = 0
.if (.option.machine_version_major == 9) && (.option.machine_version_minor == 0) && (.option.machine_version_stepping >= 6)
    dot_instructions_available = 1
.endif
madmix_instructions_available = 0
fmamix_instructions_available = 0
madmix_fmamix_with_dpp_available = 0
.if (.option.machine_version_major == 9)
    .if(.option.machine_version_stepping > 2)
        fmamix_instructions_available = 1
    .else
        madmix_instructions_available = 1
    .endif
.endif

bit_convert_mult = 0
.if(buf_type == TYPE_BFP16 && !bfp16_native_support)
    bit_convert_mult = 1
.endif


log2 c_per_gpr_log2, c_per_gpr
log2 k_per_gpr_log2, k_per_gpr
log2 n_per_gpr_log2, n_per_gpr
log2 hw_per_gpr_log2, hw_per_gpr

// chunk parameters
c_quads = c_per_gpr * chunk_size / 16
.if k_per_gpr > c_quads
    k_ds_rotates = c_quads
    k_dpp_rotates = k_per_gpr / c_quads
.else
    k_ds_rotates = k_per_gpr
    k_dpp_rotates = 1
.endif
max_per_read = read_size * elements_in_dword
//max dwords per vector read

part1_chunks = max_per_read + 1
part2_chunks = max_per_read + 1
active_lanes_in_part1_chunks = 0
active_lanes_in_part2_chunks = 0

metachunk_size = chunk_size * hw_per_gpr // hw pieces are not contiguous in vgpr
log2 chunk_size_log2, chunk_size
log2 meatchunk_size_log2, metachunk_size
out_wh = out_w * out_h
full_chunk_reads = out_wh / elements_per_lane

full_reads_per_lane = full_chunk_reads / metachunk_size
partial_chunks = out_wh - full_reads_per_lane * metachunk_size * elements_per_lane
full_reads = full_reads_per_lane
full_chunks = full_reads * elements_per_lane
.if(full_reads_per_lane == 0)
    i_cnt = 1
.else
    i_cnt = ((2 * metachunk_size) * max_per_read - partial_chunks + full_reads_per_lane * elements_per_lane - 1) / (full_reads_per_lane* elements_per_lane)
    .if(i_cnt > metachunk_size)
        i_cnt = metachunk_size
    .elseif (i_cnt < 0)
        i_cnt = 0
        static_assert(0)
    .endif
.endif

i = 0
x = 0

.if(partial_chunks)
.rept i_cnt
    partial_points1 = full_reads_per_lane * elements_per_lane * i + partial_chunks

    x = max_per_read
    .rept max_per_read
        j = (partial_points1) / x
        .if(j > metachunk_size)
            j = metachunk_size
        .endif

        .if(j > 0)
            .if ( (partial_points1 % j == 0) && (x * j == partial_points1) )
                .if (part1_chunks + part2_chunks > x)
                    part1_chunks = x
                    active_lanes_in_part1_chunks = j
                    part2_chunks = 0
                    active_lanes_in_part2_chunks = 0
                    rem_lanes_ful = metachunk_size - i
                .endif
            .else
                .if ( (x % elements_in_dword == 0) )//&&  (x <= max_per_read))
                    partial_points2 = partial_points1 - x * j
                    max_per_read2 = max_per_read - x
                        y = 1
                        .rept max_per_read2
                            k = partial_points2 / y
                            .if(y * k == partial_points2 && k <= metachunk_size)
                                .if (partial_points2 % k == 0 && (y <= max_per_read2) && (part1_chunks + part2_chunks > x + y))
                                        part1_chunks = x
                                        active_lanes_in_part1_chunks = j
                                        part2_chunks = y
                                        active_lanes_in_part2_chunks = k
                                        rem_lanes_ful = metachunk_size - i
                                .endif
                            .endif ///(y * k == partial_points2)
                            y = y + 1
                        .endr
                .endif //x % elements_in_dword == 0
            .endif //partial_points1 % j == 0
        .endif
        x = x - 1
    .endr
    i = i + 1
.endr
.endif
adv_perf_param_comb = 0
.if(partial_chunks == 0)
    part1_chunks = 0
    part2_chunks = 0
    active_lanes_in_part1_chunks = 0
    active_lanes_in_part2_chunks = 0
    rem_lanes_ful = metachunk_size
.else

    .if( part2_chunks == max_per_read + 1 && part1_chunks == max_per_read + 1)
        adv_perf_param_comb = 1
        x = max_per_read
        .rept read_size
            j = (partial_chunks + x - 1)/ x
            .if( (j <= metachunk_size) && (j * x > partial_chunks) && ((j - 1) * x < partial_chunks) )
                part1_chunks = x
                active_lanes_in_part1_chunks = j - 1
                part2_chunks = 0
                active_lanes_in_part2_chunks = 0
                rem_lanes_ful = metachunk_size
            .endif
            x = x - 1
        .endr
        bound_elements_cnt = partial_chunks - part1_chunks * (active_lanes_in_part1_chunks)
    .else
        static_assert (part1_chunks * active_lanes_in_part1_chunks + part2_chunks * active_lanes_in_part2_chunks +  full_reads_per_lane * rem_lanes_ful * elements_per_lane == out_wh)
    .endif

    static_assert(part2_chunks <= max_per_read)
    static_assert(part1_chunks <= max_per_read)


    static_assert(part1_chunks + part2_chunks <= max_per_read)
    static_assert(part1_chunks > 0)
.endif
partial_chunks = (part1_chunks+ part2_chunks)
active_lanes_in_full_chunks = rem_lanes_ful



part2_offset = part1_chunks * input_w_stride * active_lanes_in_part1_chunks

.GPR_ALLOC_BEGIN
.if limit_wave_cnt
    .SET_MAX_WAVES_LIMIT limit_wave_cnt
.endif

.SGPR_ALLOC_FROM 5
.SGPR_ALLOC soffset_in
.SGPR_ALLOC soffset_out
.SGPR_ALLOC soffset_wei
.SGPR_ALLOC desc_in, 4 // input buffer descriptor
.SGPR_ALLOC desc_out, 4 // weights buffer descriptor
.SGPR_ALLOC desc_wei, 4 // output buffer descriptor
.SGPR_ALLOC bound_lanes_exec, 2
.SGPR_ALLOC loop_n_cnt
.SGPR_ALLOC loop_hw_cnt
.SGPR_ALLOC c_base
.SGPR_ALLOC k_base
.SGPR_ALLOC n_base
.SGPR_ALLOC stmp
.SGPR_ALLOC loop_begin_ptr, 2
.SGPR_ALLOC wave_id // wave_id in group

//xnack disabled by default
//.SGPR_RESERVE_XNACK
.SGPR_RESERVE_VCC

.VGPR_ALLOC_FROM 0
.VGPR_ALLOC tid
.VGPR_ALLOC voffset_in
.VGPR_ALLOC voffset_out
.VGPR_ALLOC voffset_part1_in
.VGPR_ALLOC voffset_part1_out
.VGPR_ALLOC voffset_part2_in
.VGPR_ALLOC voffset_part2_out
.VGPR_ALLOC voffset_ldsw
accums_cnt = wei_w * wei_h * k_per_gpr * c_mult * k_mult
.VGPR_ALLOC accums, accums_cnt
single_lane_vgpr_offset = read_size

inbuf_prefetch_vgpr_offset = single_lane_vgpr_offset * c_mult
inbuf_bit_convert_vgpr_offset = inbuf_prefetch_vgpr_offset * (data_prefetch + 1)
lines_in_cnt = inbuf_bit_convert_vgpr_offset + (bit_convert_mult * inbuf_prefetch_vgpr_offset)
.VGPR_ALLOC lines_in, lines_in_cnt

outbuf_prefetch_vgpr_offset = single_lane_vgpr_offset * k_mult
outbuf_bit_convert_vgpr_offset = outbuf_prefetch_vgpr_offset * (data_prefetch + 1)
lines_out_cnt = outbuf_bit_convert_vgpr_offset + (bit_convert_mult * outbuf_prefetch_vgpr_offset)
.VGPR_ALLOC lines_out, lines_out_cnt

.VGPR_ALLOC permute_addr
.VGPR_ALLOC n_id
.if (madmix_instructions_available == 0 && dot_instructions_available == 0 && fmamix_instructions_available == 0)
    .VGPR_ALLOC vtmp_cvt_fir
    .VGPR_ALLOC vtmp_cvt_sec
.endif


static_assert (((n_part_cnt - 1) * wave_size * 4 * accums_cnt) <= 65536) //LDS size
.LDS_ALLOC_FROM 0
.LDS_ALLOC accums_lds, (n_part_cnt - 1) * wave_size * 4 * accums_cnt // lds_read_size

.GPR_ALLOC_END

max_waves_per_CU = (256 / .AUTO_VGPR_COUNT) * 4
static_assert( max_waves_per_CU >= group_size )
//.text 0
//.p2align 8
miopenGcnAsmConv1x1WrW:
.if ROCM_METADATA_VERSION == 4
    .amd_kernel_code_t
     enable_sgpr_kernarg_segment_ptr = 1
     enable_sgpr_workgroup_id_x = 1
     enable_sgpr_workgroup_id_y = 1
     enable_sgpr_workgroup_id_z = 1
     is_ptr64 = 1
     granulated_workitem_vgpr_count = .AUTO_VGPR_GRANULATED_COUNT
     granulated_wavefront_sgpr_count = .AUTO_SGPR_GRANULATED_COUNT
     enable_vgpr_workitem_id = 1
     user_sgpr_count = 2
     kernarg_segment_byte_size = KERNEL_ARGUMENTS_SIZE
     wavefront_sgpr_count = .AUTO_SGPR_COUNT
     workitem_vgpr_count = .AUTO_VGPR_COUNT
     float_mode = 192
     workgroup_group_segment_byte_size = .AUTO_LDS_BYTE_SIZE
    .end_amd_kernel_code_t
.endif

    .macro mult_acc_fp16 v_acc, v_base_out, v_base_in, it, cnt
    .if( ( (\it * elements_in_dword) + elements_in_dword) <= \cnt)
        .if(dot_instructions_available)
            v_dot2_f32_f16 v[\v_acc], v[\v_base_out], v[\v_base_in], v[\v_acc]
        .elseif (madmix_instructions_available)
            v_mad_mix_f32 v[\v_acc], v[\v_base_out], v[\v_base_in], v[\v_acc] op_sel:[0,0,0] op_sel_hi:[1,1,0]
            v_mad_mix_f32 v[\v_acc], v[\v_base_out], v[\v_base_in], v[\v_acc] op_sel:[1,1,0] op_sel_hi:[1,1,0]
        .elseif fmamix_instructions_available
            v_fma_mix_f32 v[\v_acc], v[\v_base_out], v[\v_base_in], v[\v_acc] op_sel:[0,0,0] op_sel_hi:[1,1,0]
            v_fma_mix_f32 v[\v_acc], v[\v_base_out], v[\v_base_in], v[\v_acc] op_sel:[1,1,0] op_sel_hi:[1,1,0]
        .else
            v_cvt_f32_f16 v[vtmp_cvt_fir], v[\v_base_in]
            v_cvt_f32_f16 v[vtmp_cvt_sec], v[\v_base_out]
            v_mac_f32     v[\v_acc], v[vtmp_cvt_fir], v[vtmp_cvt_sec]

            v_lshrrev_b32 v[vtmp_cvt_fir], 16, v[\v_base_in]
            v_lshrrev_b32 v[vtmp_cvt_sec], 16, v[\v_base_out]

            v_cvt_f32_f16 v[vtmp_cvt_fir], v[vtmp_cvt_fir]
            v_cvt_f32_f16 v[vtmp_cvt_sec], v[vtmp_cvt_sec]
            v_mac_f32     v[\v_acc], v[vtmp_cvt_fir], v[vtmp_cvt_sec]
        .endif
    .else   //if partial read
        .if(madmix_instructions_available)
            v_mad_mix_f32 v[\v_acc], v[\v_base_out], v[\v_base_in], v[\v_acc] op_sel:[0,0,0] op_sel_hi:[1,1,0]
        .elseif fmamix_instructions_available
            v_fma_mix_f32 v[\v_acc], v[\v_base_out], v[\v_base_in], v[\v_acc] op_sel:[0,0,0] op_sel_hi:[1,1,0]
        .else
            v_cvt_f32_f16 v[vtmp_cvt_fir], v[\v_base_in]
            v_cvt_f32_f16 v[vtmp_cvt_sec], v[\v_base_out]
            v_mac_f32     v[\v_acc], v[vtmp_cvt_fir], v[vtmp_cvt_sec]
        .endif
    .endif
.endm

    .macro bfp16_fp32_convert bfp16_vgpr_ptr, second_fp32_res_ptr, cnt
    convert_i = 0
    .rept \cnt
        //v_lshlrev_b32 v[\second_fp32_res_ptr + convert_i], 16, v[\bfp16_vgpr_ptr + convert_i]
        //v_and_b32 v[\bfp16_vgpr_ptr + convert_i], 0 + 0xFFFF0000, v[\bfp16_vgpr_ptr + convert_i]
        v_and_b32 v[\second_fp32_res_ptr + convert_i], 0 + 0xFFFF0000, v[\bfp16_vgpr_ptr + convert_i]
        v_lshlrev_b32 v[\bfp16_vgpr_ptr + convert_i], 16, v[\bfp16_vgpr_ptr + convert_i]

        convert_i = convert_i + 1
    .endr
.endm

.macro m_conv_accums elements_cnt, ld_part_id
    rotates_inflight = 0
    k_ds = 0
    .if(\elements_cnt == 0)
        .exitm
    .endif

    .if(buf_type == TYPE_BFP16 && bfp16_native_support == 0)
        conv_elements_cnt = \elements_cnt
        fi_element_ptr = lines_in + (\ld_part_id * inbuf_prefetch_vgpr_offset)
        bfp16_fp32_convert fi_element_ptr, lines_in + inbuf_bit_convert_vgpr_offset, inbuf_prefetch_vgpr_offset
        fi_element_ptr = lines_out + (\ld_part_id * outbuf_prefetch_vgpr_offset)
        bfp16_fp32_convert fi_element_ptr, lines_out + outbuf_bit_convert_vgpr_offset, outbuf_prefetch_vgpr_offset
    .else
        conv_elements_cnt = (\elements_cnt + elements_in_dword - 1) / elements_in_dword
    .endif

    .rept k_ds_rotates
        i = 0
        .rept conv_elements_cnt
            kx = 0
            .rept k_mult
                base_out = lines_out + kx * read_size + (\ld_part_id * outbuf_prefetch_vgpr_offset)
                .if (buf_type == TYPE_BFP16 && bfp16_native_support == 0)
                    base_out = base_out - (i % 2) * (\ld_part_id * outbuf_prefetch_vgpr_offset)
                    base_out = base_out + (i % 2) * outbuf_bit_convert_vgpr_offset + (i / 2)
                .else
                    base_out = base_out + i
                .endif
                .if k_ds > 0
                    rotates_inflight = rotates_inflight - 1
                    s_wait , rotates_inflight
                .endif

                k_dpp = 0
                .rept k_dpp_rotates
                cx = 0

                .rept c_mult
                    base_in = lines_in + cx * read_size + (\ld_part_id * inbuf_prefetch_vgpr_offset)
                    acc = accums + k_per_gpr * (cx * k_mult + kx) + k_ds * k_dpp_rotates

                    .if(buf_type == TYPE_BFP16 && bfp16_native_support == 0)
                        base_in = base_in - (i % 2) * (\ld_part_id * inbuf_prefetch_vgpr_offset)
                        base_in = base_in + (i % 2) * inbuf_bit_convert_vgpr_offset + (i / 2)
                    .else
                        base_in = base_in + i
                    .endif

                    .if(elements_in_dword == 2 && ( (buf_type == TYPE_FP16) || (buf_type == TYPE_BFP16 && bfp16_native_support == 1) ))
                        .if(buf_type == TYPE_FP16)
                            mult_acc_fp16 (acc + k_dpp), (base_out), (base_in), i, \elements_cnt
                        .elseif (buf_type == TYPE_BFP16)
                            mult_acc_bfp16 (acc + k_dpp), (base_out), (base_in), i, \elements_cnt
                        .endif
                    .else   //if fp32 or converted bfp16
                        .if(k_dpp == 0)
                            v_mac_f32 v[acc], v[base_out], v[base_in]
                        .else
                            v_mac_f32 v[acc + k_dpp], v[base_out], v[base_in] row_ror:16*k_dpp/k_dpp_rotates
                        .endif
                    .endif
                    cx = cx + 1
                .endr

                    k_dpp = k_dpp + 1
                    .if(elements_in_dword == 2 && k_dpp_rotates > 1 && madmix_fmamix_with_dpp_available == 0 && buf_type != TYPE_BFP16)
                        v_mov_b32 v[base_out], v[base_out] row_ror:16/k_dpp_rotates
                        s_nop 1
                    .endif
                .endr

                .if (k_ds + 1) < k_ds_rotates
                    static_assert (c_quads == 2 || c_quads == 4)
                    .if c_quads == 2
                        ds_swizzle_b32 v[base_out], v[base_out] offset:0xc200
                    .elseif c_quads == 4
                        ds_bpermute_b32 v[base_out], v[permute_addr], v[base_out]
                    .endif
                    rotates_inflight = rotates_inflight + 1
                .endif

                kx = kx + 1
            .endr
            i = i + 1
        .endr
        k_ds = k_ds + 1
    .endr

.endm

.macro m_acc_reduction first_round, rounds
    i = 0
    .rept \rounds
        round = i + \first_round
        acc = accums
        .rept accums_cnt
            .if i >= 1 && accums_cnt <= 2
                s_nop 2 - accums_cnt
            .endif
            .if round == 0
                v_add_f32 v[acc], v[acc], v[acc] quad_perm:[1,0,3,2]
            .elseif round == 1
                v_add_f32 v[acc], v[acc], v[acc] quad_perm:[2,3,0,1]
            .elseif round == 2
                v_add_f32 v[acc], v[acc], v[acc] row_ror:12
            .elseif round == 3
                v_add_f32 v[acc], v[acc], v[acc] row_ror:8
            .elseif round == 4
                static_assert (0) //v_add_f32 v[acc], v[acc], v[acc] row_bcast:15
            .elseif round == 5
                static_assert (0) //v_add_f32 v[acc], v[acc], v[acc] row_bcast:31
            .else
                static_assert (0)
            .endif
            acc = acc + 1
        .endr
        i = i + 1
    .endr
.endm

    s_load_dwordx2 s[desc_in:desc_in+1], s[kernarg:kernarg+1], 0x0 + in_ptr_off
    s_load_dwordx2 s[desc_wei:desc_wei+1], s[kernarg:kernarg+1], 0x0 + wei_ptr_off
    s_load_dwordx2 s[desc_out:desc_out+1], s[kernarg:kernarg+1], 0x0 + out_ptr_off
    s_mov_b32 m0, -1

    v_readfirstlane_b32 s[wave_id], v[tid]
    s_lshr_b32 s[wave_id], s[wave_id], 0+wave_size_log2
    v_and_b32 v[tid], 0x3f, v[tid]

    // calculate input/output offsets
    // example for c_per_gpr=4, k_per_gpr=2, n_per_gpr=1
    // lanes  0-15: c0, k0, n0
    // lanes 16-31: c1, k0, n0
    // lanes 32-47: c2, k1, n0
    // lanes 48-63: c3, k1, n0
    vtmp = accums
    c_id = lines_in
    k_id = lines_out
    v_lshrrev_b32 v[n_id], 0 + wave_size_log2 - n_per_gpr_log2, v[tid]
    v_bfe_u32 v[c_id], v[tid], 0 + chunk_size_log2, 0 + c_per_gpr_log2
    v_bfe_u32 v[k_id], v[tid], 0 + chunk_size_log2 + c_per_gpr_log2 - k_per_gpr_log2, 0 + k_per_gpr_log2

    s_mov_b32 s[stmp], 0 + input_c_stride * sequential_c_channels
    v_mul_lo_u32 v[voffset_in], s[stmp], v[c_id]
    s_mov_b32 s[stmp], 0 + input_n_stride
    v_mul_lo_u32 v[vtmp], s[stmp], v[n_id]
    _v_add_nc_u32 v[voffset_in], v[voffset_in], v[vtmp] // c_off + n_off

    s_mov_b32 s[stmp], 0 + output_k_stride * sequential_k_channels
    v_mul_lo_u32 v[voffset_out], s[stmp], v[k_id]
    s_mov_b32 s[stmp], 0 + output_n_stride
    v_mul_lo_u32 v[vtmp], s[stmp], v[n_id]
    _v_add_nc_u32 v[voffset_out], v[voffset_out], v[vtmp] // k_off + n_off

    vtmp2 = permute_addr
    v_bfe_u32 v[vtmp], v[tid], 0 + chunk_size_log2 + c_per_gpr_log2, 0 + hw_per_gpr_log2 // hw peice id
    v_lshlrev_b32 v[vtmp], 0 + chunk_size_log2, v[vtmp]
    v_and_b32 v[vtmp2], 0 + chunk_size - 1, v[tid] // lane in chunk
    _v_add_nc_u32 v[vtmp2], v[vtmp2], v[vtmp] // lane in metachunk

    v_mul_u32_u24 v[vtmp], 0 + input_w_stride * part1_chunks, v[vtmp2]
    _v_add_nc_u32 v[voffset_part1_in],  v[voffset_in], v[vtmp] // +hw_off
    v_mul_u32_u24 v[vtmp], 0 + output_w_stride * part1_chunks, v[vtmp2]
    _v_add_nc_u32 v[voffset_part1_out], v[voffset_out], v[vtmp] // +hw_off

    v_mul_u32_u24 v[vtmp], 0 + input_w_stride * part2_chunks, v[vtmp2]
    _v_add_nc_u32 v[voffset_part2_in],  v[voffset_in], v[vtmp] // +hw_off
    v_mul_u32_u24 v[vtmp], 0 + output_w_stride * part2_chunks, v[vtmp2]
    _v_add_nc_u32 v[voffset_part2_out], v[voffset_out], v[vtmp] // +hw_off

    v_mul_u32_u24 v[vtmp], 0 + input_w_stride * elements_per_lane, v[vtmp2]
    _v_add_nc_u32 v[voffset_in], v[voffset_in], v[vtmp] // +hw_off
    v_mul_u32_u24 v[vtmp], 0 + output_w_stride * elements_per_lane, v[vtmp2]
    _v_add_nc_u32 v[voffset_out], v[voffset_out], v[vtmp] // +hw_off

    s_mul_i32 s[stmp], 0 + 4 * read_size * wave_size, s[wave_id]

    // calculate buffer scalar offsets
    s_mul_i32 s[c_base], 0 + c_per_gpr * c_mult, s[gid_y]
    s_mul_i32 s[k_base], 0 + k_per_gpr * k_mult, s[gid_z]
    s_mul_i32 s[n_base], 0 + n_per_gpr, s[wave_id]

    s_mul_i32 s[soffset_in], 0 + input_n_stride, s[n_base]
    s_mul_i32 s[stmp], 0 + input_c_stride, s[c_base]
    s_add_u32 s[soffset_in], s[soffset_in], s[stmp]

    s_mul_i32 s[soffset_out], 0 + output_n_stride, s[n_base]
    s_mul_i32 s[stmp], 0 + output_k_stride, s[k_base]
    s_add_u32 s[soffset_out], s[soffset_out], s[stmp]

    s_mul_i32 s[soffset_wei], 0 + filter_c_stride, s[c_base]
    s_mul_i32 s[stmp], 0 + filter_k_stride, s[k_base]
    s_add_u32 s[soffset_wei], s[soffset_wei], s[stmp]


    // mask unused lanes
    _v_add_nc_u32 v[c_id], s[c_base], v[c_id]
    _v_add_nc_u32 v[k_id], s[k_base], v[k_id]
    _v_add_nc_u32 v[n_id], s[n_base], v[n_id]
    v_cmp_gt_u32 vcc, 0 + input_channels, v[c_id]
    v_cndmask_b32_e32 v[voffset_in], -1, v[voffset_in], vcc
    v_cmp_gt_u32 vcc, 0 + output_channels, v[k_id]
    v_cndmask_b32_e32 v[voffset_out], -1, v[voffset_out], vcc

    v_mov_b32 v[vtmp], 0x7FFFFFFF
    v_cmp_gt_u32 vcc, 0 + active_lanes_in_full_chunks, v[vtmp2]
    v_cndmask_b32_e32 v[voffset_in],  v[vtmp], v[voffset_in], vcc
    v_cndmask_b32_e32 v[voffset_out], v[vtmp], v[voffset_out], vcc

    .if(adv_perf_param_comb)
        v_mov_b32 v[voffset_part2_in],  v[voffset_part1_in]
        v_mov_b32 v[voffset_part2_out],  v[voffset_part1_out]
    .else
        v_cmp_gt_u32 vcc, 0 + active_lanes_in_part2_chunks, v[vtmp2]
        v_cndmask_b32_e32 v[voffset_part2_in],  v[vtmp], v[voffset_part2_in], vcc
        v_cndmask_b32_e32 v[voffset_part2_out], v[vtmp], v[voffset_part2_out], vcc
    .endif

    v_cmp_gt_u32 vcc, 0 + active_lanes_in_part1_chunks, v[vtmp2]
    v_cndmask_b32_e32 v[voffset_part1_in],  v[vtmp], v[voffset_part1_in], vcc
    v_cndmask_b32_e32 v[voffset_part1_out], v[vtmp], v[voffset_part1_out], vcc

    v_cmp_eq_u32 vcc, 0 + active_lanes_in_part1_chunks, v[vtmp2]
    s_nop 4
    s_mov_b64 s[bound_lanes_exec:bound_lanes_exec+1], vcc

    .GPR_INVALIDATE c_id
    .GPR_INVALIDATE k_id
    .GPR_INVALIDATE vtmp
    .GPR_INVALIDATE vtmp2

    // fill format and size fields of buffer descriptors
    s_mov_b32 s[desc_in+2], input_buffer_size
    s_mov_b32 s[desc_in+3], 0x00027000
    s_mov_b32 s[desc_wei+2], filter_buffer_size
    s_mov_b32 s[desc_wei+3], 0x00027000
    s_mov_b32 s[desc_out+2], output_buffer_size
    s_mov_b32 s[desc_out+3], 0x00027000

    i = 0
    .rept accums_cnt
        v_mov_b32 v[accums+i], 0
        i = i + 1
    .endr

    s_waitcnt 0

    // calculate buffer offsets
    s_add_u32 s[desc_wei], s[desc_wei], s[soffset_wei]
    s_addc_u32 s[1+desc_wei], 0, s[1+desc_wei]
    s_sub_u32 s[2+desc_wei], s[2+desc_wei], s[soffset_wei]
    s_max_i32 s[2+desc_wei], 0, s[2+desc_wei]

    // compute permute_addr
    .if c_quads == 4
        _v_add_nc_u32 v[permute_addr], 0 + wave_size / k_ds_rotates, v[tid]
        v_lshlrev_b32 v[permute_addr], 2, v[permute_addr]
    .endif

    s_mov_b32 s[loop_n_cnt], s[n_base]

        .macro increase_ioffset_or_soffset i_offset, sgpr, sgpr_offset, rize_vall
        _buff = \i_offset + \rize_vall
        .if(_buff >= (1 << 12))
            s_add_u32 s[\sgpr], s[\sgpr], 0 + _buff
            \i_offset = 0
            \sgpr_offset = \sgpr_offset + _buff
        .else
            \i_offset = \i_offset + \rize_vall
        .endif
    .endm

    .macro m_load inout, total_adj, dwords1, voff1, w_cnt, ld_id, dwords2=0, voff2=0, shorts1=0, shorts2=0, ex_load=0
        .if lines_\inout == lines_in
            sequential_output_channels = sequential_c_channels
            ck_stride = input_c_stride
            mult = c_mult
            dst = lines_in + \ld_id * inbuf_prefetch_vgpr_offset
            desc = desc_in
            soff = soffset_in
            adj_size = c_per_gpr * input_c_stride
        .else
            mult = k_mult
            sequential_output_channels = sequential_k_channels
            ck_stride = output_k_stride
            dst = lines_out + \ld_id * outbuf_prefetch_vgpr_offset
            desc = desc_out
            soff = soffset_out
            adj_size = k_per_gpr * output_k_stride
        .endif

         mult = mult / sequential_output_channels
        adj_size = adj_size * sequential_output_channels

        _mult_it = 0
        .rept mult
            _sequential_output_channels_it = 0
            _sequential_ck_offset = 0
            _seq_ck_offset_in_soffset = 0
            .rept sequential_output_channels
                .if(_sequential_output_channels_it != 0)
                    increase_ioffset_or_soffset _sequential_ck_offset, soff, _seq_ck_offset_in_soffset, ck_stride
                .endif
                .if tuple_alignment && (\dwords1 > 1) && (dst % 2)
                    m_buffer_load_dwordx 1,          dst,              \voff1, desc, soff,  _sequential_ck_offset
                    m_buffer_load_dwordx \dwords1-1, dst+1,            \voff1, desc, soff, (_sequential_ck_offset + dword_size)
                    \w_cnt = \w_cnt + 1
                .else
                    m_buffer_load_dwordx \dwords1, dst,            \voff1, desc, soff, _sequential_ck_offset
                .endif
                .if(\dwords1>0)
                    \w_cnt = \w_cnt + 1
                .endif

                .if(\shorts1 != 0)
                    short_offset = \dwords1 * dword_size
                    m_buffer_load_ushort 1, \dwords1 + dst, \voff1, desc, soff, (_sequential_ck_offset + short_offset)
                    \w_cnt = \w_cnt + 1
                .else
                    .if(\ex_load)

                        s_mov_b64 exec, s[bound_lanes_exec:bound_lanes_exec+1]
                        bound_dwords_cnt = bound_elements_cnt / elements_in_dword
                        bound_shorts_cnt = bound_elements_cnt % elements_in_dword

                        .if tuple_alignment && (bound_dwords_cnt > 1) && (dst % 2)
                            m_buffer_load_dwordx 1,                  dst,   \voff2, desc, soff,  _sequential_ck_offset
                            m_buffer_load_dwordx bound_dwords_cnt-1, dst+1, \voff2, desc, soff, (_sequential_ck_offset + dword_size)
                            \w_cnt = \w_cnt + 1
                        .else
                            m_buffer_load_dwordx bound_dwords_cnt, dst, \voff2, desc, soff, _sequential_ck_offset
                        .endif
                        short_offset = bound_dwords_cnt * dword_size
                        m_buffer_load_ushort bound_shorts_cnt, dst + bound_dwords_cnt, \voff2, desc, soff, (_sequential_ck_offset + short_offset)

                        .if(bound_dwords_cnt )
                            \w_cnt = \w_cnt + 1
                        .endif
                        .if(bound_shorts_cnt)
                            \w_cnt = \w_cnt + 1
                        .endif

                        s_mov_b64 exec, -1
                    .else
                        .if tuple_alignment && (\dwords2 > 1) && (dst % 2)
                            m_buffer_load_dwordx 1,          dst + \dwords1,     \voff2, desc, soff, (_sequential_ck_offset + part2_offset)
                            m_buffer_load_dwordx \dwords2-1, dst + \dwords1 + 1, \voff2, desc, soff, (_sequential_ck_offset + part2_offset + dword_size)
                            \w_cnt = \w_cnt + 1
                        .else
                            m_buffer_load_dwordx \dwords2, dst + \dwords1, \voff2, desc, soff, (_sequential_ck_offset + part2_offset)
                        .endif
                        .if(\dwords2 > 0)
                            \w_cnt = \w_cnt + 1
                        .endif

                        .if(\shorts2 != 0)
                            short_offset = part2_offset + part2_dwords * dword_size
                            m_buffer_load_ushort 1, dst + \dwords1 + \dwords2, \voff2, desc, soff, (_sequential_ck_offset + short_offset)
                            \w_cnt = \w_cnt + 1
                        .endif
                    .endif
                .endif

                dst = dst + read_size //\dwords1
                _sequential_output_channels_it = _sequential_output_channels_it + 1
            .endr
            .if(_mult_it != (mult - 1))

                s_add_u32 s[soff], s[soff], 0 + adj_size - _seq_ck_offset_in_soffset
                \total_adj = \total_adj + adj_size
                _mult_it = _mult_it + 1
            .else
                \total_adj = \total_adj + _seq_ck_offset_in_soffset
            .endif
        .endr
    .endm

.if(full_reads > 0 && full_reads < data_prefetch + 1)
    data_prefetch = full_reads - 1
.endif

LD_PARTIAL_CHUNKS = 1
LD_FULL_CHUNKS = 0
LD_PART_A_ID = 0
LD_PART_B_ID = data_prefetch
last_free_ld_part = LD_PART_B_ID


.macro ld_buffers_inc_pointers_rept_waitcnt chunk_type, ld_part_id, rept_wait_cnt=-1
     wait_cnt = 0

    .if(\chunk_type == LD_PARTIAL_CHUNKS)
        m_load in,  c_off, part1_dwords, voffset_part1_in, wait_cnt, \ld_part_id, part2_dwords, voffset_part2_in, part1_shorts, part2_shorts, adv_perf_param_comb
        m_load out, k_off, part1_dwords, voffset_part1_out, wait_cnt, \ld_part_id, part2_dwords, voffset_part2_out, part1_shorts, part2_shorts, adv_perf_param_comb
    .else
        c_off = 0
        k_off = 0
        m_load in,  c_off, (elements_per_lane / elements_in_dword), voffset_in, wait_cnt, \ld_part_id
        m_load out, k_off, (elements_per_lane / elements_in_dword), voffset_out, wait_cnt, \ld_part_id
        s_add_u32 s[soffset_in],  s[soffset_in],  0 + active_lanes_in_full_chunks * (elements_per_lane * input_w_stride) - c_off
        s_add_u32 s[soffset_out], s[soffset_out], 0 + active_lanes_in_full_chunks * (elements_per_lane * output_w_stride) - k_off
    .endif
    .if(\rept_wait_cnt != -1)
        \rept_wait_cnt = wait_cnt
    .endif
.endm

.macro data_prefetch_init_q q_wait_cnt, singl_wait_cnt
    q_id = LD_PART_A_ID
    .rept data_prefetch
        ld_buffers_inc_pointers_rept_waitcnt LD_FULL_CHUNKS, q_id, \singl_wait_cnt
        \q_wait_cnt = \q_wait_cnt + \singl_wait_cnt
        q_id = (q_id + 1)
    .endr
.endm

.macro data_ld_prefetch_active_loop q_wait_cnt, loop_cnt=data_prefetch+1
    q_id_conv = LD_PART_A_ID
    q_id_data_ld = LD_PART_B_ID
    .rept \loop_cnt
        ld_buffers_inc_pointers_rept_waitcnt LD_FULL_CHUNKS, q_id_data_ld

        s_wait \q_wait_cnt

        m_conv_accums elements_per_lane, q_id_conv

        q_id_conv = ((q_id_conv + 1) % (data_prefetch + 1))
        q_id_data_ld = ((q_id_data_ld + 1) % (data_prefetch + 1))
    .endr
.endm

.macro data_prefetch_conv_finalizing q_wait_cnt, singl_wait_cnt, q_id_conv_off=0
    q_id_conv = ((LD_PART_A_ID + \q_id_conv_off) % (data_prefetch + 1))

    .rept data_prefetch
        \q_wait_cnt = (\q_wait_cnt - \singl_wait_cnt)
        s_wait \q_wait_cnt
        m_conv_accums elements_per_lane, q_id_conv
        q_id_conv = ((q_id_conv + 1) % (data_prefetch + 1))
    .endr
.endm

S_GETPC_B64 s[loop_begin_ptr:loop_begin_ptr+1]
loop_n_begin: // loop over batch (n)
    s_mov_b32 s[loop_hw_cnt], 0

    c_off = 0
    k_off = 0
    q_wait_vec_load_full = 0
    single_wait_vec_load_full = 0
    .if full_reads
        loop_resi = 0

        data_prefetch_init_q q_wait_vec_load_full, single_wait_vec_load_full

        .if(full_reads >= 2 * data_prefetch + 1)
            loop_hw_begin:
                data_ld_prefetch_active_loop q_wait_vec_load_full

            loop_hw_end:
                s_addk_i32 s[loop_hw_cnt], 1 + data_prefetch
                s_cmpk_gt_u32 s[loop_hw_cnt], 0 + full_reads - (2 * data_prefetch + 1)
                s_cbranch_scc0 loop_hw_begin
        .endif
        .if( (full_reads - data_prefetch) % (data_prefetch + 1) != 0)
            loop_resi = ((full_reads - data_prefetch) % (data_prefetch + 1))

            data_ld_prefetch_active_loop q_wait_vec_load_full, loop_resi

            last_free_ld_part = ((LD_PART_B_ID + loop_resi) % (data_prefetch + 1))
        .endif
    .endif

    c_off = full_chunks * input_w_stride * active_lanes_in_full_chunks
    k_off = full_chunks * output_w_stride * active_lanes_in_full_chunks
    .if partial_chunks
        wait_vec_load_part = 0
        part1_dwords = part1_chunks  / elements_in_dword
        part2_dwords = part2_chunks / elements_in_dword
        part1_shorts = part1_chunks % elements_in_dword
        part2_shorts = part2_chunks % elements_in_dword
        ld_buffers_inc_pointers_rept_waitcnt LD_PARTIAL_CHUNKS, last_free_ld_part, wait_vec_load_part
        q_wait_vec_load_full = q_wait_vec_load_full + wait_vec_load_part
    .endif

    .if(full_reads != 0 && data_prefetch != 0)
        data_prefetch_conv_finalizing q_wait_vec_load_full, single_wait_vec_load_full, loop_resi
    .endif

    .if(partial_chunks)
        s_wait 0
        m_conv_accums partial_chunks, last_free_ld_part
     .endif

    s_add_u32 s[soffset_in],  s[soffset_in],  0 + input_n_stride * n_per_gpr * n_part_cnt - c_off
    s_add_u32 s[soffset_out], s[soffset_out], 0 + output_n_stride * n_per_gpr * n_part_cnt- k_off
loop_n_end:
    _v_add_nc_u32 v[n_id], 0 + (n_per_gpr * n_part_cnt) , v[n_id]

    s_addk_i32 s[loop_n_cnt], 0 + n_per_gpr * n_part_cnt
    s_cmpk_ge_u32 s[loop_n_cnt], 0 + batch_size
    //s_cbranch_scc0 loop_n_begin
    s_cbranch_scc1 loop_exit

    s_setpc_b64 s[loop_begin_ptr:loop_begin_ptr+1]
loop_exit:

    // reduction inside chunk
    m_acc_reduction 0, chunk_size_log2

    // reduction across n and hw pieces
    .GPR_REUSE voffset_out, vtmp
    .if n_per_gpr * hw_per_gpr > 1
        .if chunk_size >= 4
            v_lshlrev_b32 v[permute_addr], 2 + chunk_size_log2, v[tid]
            m_bpermute accums, accums_cnt, permute_addr
            // acc layout [n/hw][c]:
            // c0n0 c1n0 c2n0 ... c0n1 c1n1 c2n1 ...
            s_waitcnt 0 // todo: later
            m_acc_reduction c_per_gpr_log2, n_per_gpr_log2 + hw_per_gpr_log2
        .else
            v_lshrrev_b32 v[vtmp], 0 + n_per_gpr_log2 + hw_per_gpr_log2, v[tid]
            v_lshlrev_b32 v[permute_addr], 0 + c_per_gpr_log2, v[tid]
            v_and_b32 v[permute_addr], 0 + wave_size/chunk_size - 1, v[permute_addr]
            v_bfi_b32 v[permute_addr], 0 + c_per_gpr - 1, v[vtmp], v[permute_addr]
            v_lshlrev_b32 v[permute_addr], 2 + chunk_size_log2, v[permute_addr]
            m_bpermute accums, accums_cnt, permute_addr
            // acc layout [c][n/hw]:
            // c0n0 c0n1 c0n2 ... c1n0 c1n1 c1n2 ...
            s_waitcnt 0 // todo: more later

            m_acc_reduction 0, n_per_gpr_log2 + hw_per_gpr_log2

            v_lshlrev_b32 v[permute_addr], 2 + n_per_gpr_log2 + hw_per_gpr_log2, v[tid]
            m_bpermute accums, accums_cnt, permute_addr
            s_waitcnt 0 // todo: finally more later
        .endif
    .endif

    s_waitcnt 0

    .GPR_REUSE lines_in, lines_in_buffer
    .GPR_REUSE lines_out, lines_in_buffer2
    //used as one

    //use LDS to merge waves
    .macro acc_nvgpr_from_lds read_buffer, buffer_start, data_cnt

        acum_idx = \buffer_start / (n_part_cnt - 1)
        wave_idx = \buffer_start - acum_idx * (n_part_cnt - 1)
        read_it = 0

        .rept \data_cnt
            .if (wave_idx >= (n_part_cnt - 1))
                wave_idx = 0
                acum_idx = acum_idx + 1
            .endif
            .if (acum_idx >= accums_cnt)
                acum_idx = 0
            .endif

            v_add_f32 v[accums + acum_idx], v[accums + acum_idx], v[\read_buffer + read_it]

            read_it = read_it + 1
            wave_idx = wave_idx + 1
        .endr
    .endm

    .if (n_part_cnt > 1)
        lds_read_size = 1
        v_mul_u32_u24 v[voffset_ldsw], 0 + lds_element_stride * lds_read_size, v[tid]

        s_cmpk_eq_u32 s[wave_id], 0
        s_cbranch_scc1 lds_read_begin

        s_sub_u32 s[stmp], s[wave_id], 1

        s_mul_i32 s[stmp], 0 + lds_element_stride * lds_read_size * wave_size, s[stmp]
        _v_add_nc_u32 v[voffset_ldsw], s[stmp], v[voffset_ldsw]

        lds_wr_id = 0
        .rept accums_cnt
            lds_acc_off = wave_size * (n_part_cnt - 1 )* lds_element_stride * lds_wr_id
            ds_write_b32 v[voffset_ldsw], v[accums + lds_wr_id], offset:0+lds_acc_off + accums_lds
            lds_wr_id = lds_wr_id + 1
        .endr
        s_wait , 0

        s_endpgm

        lds_read_begin:
        s_barrier

        lines_io_size = read_size * (c_mult + k_mult)
        lines_in_id = 0
        first_element = 0
        lds_acc_off = 0

        .rept accums_cnt * (n_part_cnt - 1)
            .if(lines_in_id >= lines_io_size)
                s_wait , 0
                acc_nvgpr_from_lds lines_in_buffer, first_element, lines_in_id
                first_element = first_element + lines_in_id
                lines_in_id = 0
            .endif
            ds_read_b32 v[lines_in_buffer + lines_in_id], v[voffset_ldsw], offset:0+lds_acc_off + accums_lds
            lds_acc_off = lds_acc_off + wave_size * lds_element_stride * lds_read_size
            lines_in_id = lines_in_id + 1
        .endr
        s_wait , 0
        acc_nvgpr_from_lds lines_in_buffer, first_element, lines_in_id
    .endif

    .GPR_REUSE lines_in_buffer, lines_in
    .GPR_REUSE lines_in_buffer2, lines_out

    // STORE
    // prepare output addresses
    .GPR_REUSE voffset_in, voffset_wei
    .GPR_REUSE lines_in, c_off
    .GPR_REUSE lines_out, k_off
    .GPR_REUSE voffset_part1_in, c_gid
    .GPR_REUSE voffset_part2_in, k_gid
    .GPR_REUSE voffset_part1_out, c_off_masked
    .GPR_REUSE voffset_part2_out, k_off_masked
    .GPR_REUSE n_id, invalid_addr
    v_mov_b32 v[invalid_addr], 0x7FFFFFFF

    v_mul_u32_u24 v[c_gid], 0 + sequential_c_channels, v[tid]
    _v_add_nc_u32 v[c_gid], s[c_base], v[c_gid]
    v_mul_u32_u24 v[c_off], 0 + filter_c_stride * sequential_c_channels, v[tid]
    v_cmp_gt_i32 vcc, 0 + c_per_gpr, v[tid]
    v_cndmask_b32_e32 v[c_off], v[invalid_addr], v[c_off], vcc

    .macro _v_add_nc_u32_ror dst, src0, src1, ror
        .long 0x320000FA + ((\src1) << 9) + ((\dst) << 17)
        .long 0xFF012100 + \src0 + ((\ror - 1) << 8)
    .endm

    v_bfe_u32 v[k_off], v[tid], 0 + c_per_gpr_log2 - k_per_gpr_log2, 0 + k_per_gpr_log2
    _v_add_nc_u32 v[k_gid], s[k_base], v[k_off]
    v_mul_u32_u24 v[k_off], 0 + filter_k_stride * sequential_k_channels, v[k_off]

    _v_add_nc_u32 v[permute_addr], 0 + wave_size / k_ds_rotates, v[tid]
    v_lshlrev_b32 v[permute_addr], 2, v[permute_addr]

    // store accums
    k_ds = 0
    rotates_inflight = 0
    .rept k_ds_rotates

        .if k_ds > 0
            rotates_inflight = rotates_inflight - 2
            s_wait , rotates_inflight
        .endif

        kx = 0

        k_mult_packed_cnt = (k_mult + sequential_k_channels - 1) / sequential_k_channels
        .rept k_mult_packed_cnt
            v_cmp_gt_i32 vcc, 0 + output_channels - kx * k_per_gpr, v[k_gid]
            v_cndmask_b32_e32 v[k_off_masked], v[invalid_addr], v[k_off], vcc
            cx = 0

            c_mult_packed_cnt = (c_mult + sequential_c_channels - 1) / sequential_c_channels
            .rept c_mult_packed_cnt
                v_cmp_gt_i32 vcc, 0 + input_channels - cx * c_per_gpr, v[c_gid]
                v_cndmask_b32_e32 v[c_off_masked], v[invalid_addr], v[c_off], vcc
                k_dpp = 0
                .rept k_dpp_rotates
                    b = (k_dpp * c_per_gpr / k_per_gpr) % 16 // lanes to ror
                    .if b == 0
                        _v_add_nc_u32 v[voffset_wei], v[k_off_masked], v[c_off_masked]
                    .else
                        .if (.option.machine_version_major == 8) // workaround for asm
                            _v_add_nc_u32_ror voffset_wei, k_off_masked, c_off_masked, b
                        .else
                            _v_add_nc_u32 v[voffset_wei], v[k_off_masked], v[c_off_masked] row_ror:b
                        .endif
                    .endif

                    acc = accums + k_per_gpr * (cx * k_mult + kx) + k_ds * k_dpp_rotates
                    .if( (buf_type == TYPE_FP16 || buf_type == TYPE_BFP16) && acc_type == TYPE_FP32)
                        acc2_cx = cx + sequential_c_channels - 1
                        acc2_kx = kx + sequential_k_channels - 1
                        acc2 = accums + k_per_gpr * ( (acc2_cx * k_mult) + acc2_kx) + k_ds * k_dpp_rotates
                        .if(buf_type == TYPE_FP16)
                            .if (!short_store)
                                v_cvt_pkrtz_f16_f32 v[acc+k_dpp], v[acc + k_dpp], v[acc2 + k_dpp]
                            .else
                                v_cvt_f16_f32 v[acc+k_dpp], v[acc + k_dpp]
                            .endif
                        .else
                            v_lshrrev_b32 v[acc + k_dpp], 16, v[acc + k_dpp]
                            .if (!short_store)
                                v_and_b32 v[acc2 + k_dpp], 0xFFFF0000, v[acc2 + k_dpp]
                                v_or_b32 v[acc + k_dpp], v[acc + k_dpp], v[acc2 + k_dpp]
                            .endif
                        .endif
                    .endif
                    s_mov_b32 s[stmp], 0 + cx * c_per_gpr * filter_c_stride + kx * k_per_gpr * filter_k_stride

                    .if(short_store)
                        buffer_store_short v[acc+k_dpp], v[voffset_wei], s[desc_wei:desc_wei+3], s[stmp] offen
                    .else
                        buffer_store_dword v[acc+k_dpp], v[voffset_wei], s[desc_wei:desc_wei+3], s[stmp] offen
                    .endif

                    k_dpp = k_dpp + 1
                .endr
                cx = cx + sequential_c_channels
            .endr
            kx = kx + sequential_k_channels
        .endr
        k_ds = k_ds + 1

        .if k_ds < k_ds_rotates
            static_assert (c_quads == 2 || c_quads == 4)
            .if c_quads == 2
                ds_swizzle_b32 v[k_off], v[k_off] offset:0xc200
                ds_swizzle_b32 v[k_gid], v[k_gid] offset:0xc200
            .elseif c_quads == 4
                ds_bpermute_b32 v[k_off], v[permute_addr], v[k_off]
                ds_bpermute_b32 v[k_gid], v[permute_addr], v[k_gid]
            .endif
            rotates_inflight = rotates_inflight + 1
        .endif
    .endr

s_endpgm


.Lfunc_end0:
    .size miopenGcnAsmConv1x1WrW, .Lfunc_end0 - miopenGcnAsmConv1x1WrW

.ifdef n_per_group
.error "n_per_group must NOT be defined"
.end
.endif
.set n_per_group, n_part_cnt

workgroup_size_x = n_per_group * 64

.if ROCM_METADATA_VERSION == 5
.rodata
.p2align 6
.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_stepping == 10)
.amdhsa_kernel miopenGcnAsmConv1x1WrW
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 1
        .amdhsa_system_sgpr_workgroup_id_z 1
        .amdhsa_system_vgpr_workitem_id 1
        .amdhsa_next_free_sgpr __amdhsa_next_free_sgpr
        .amdhsa_next_free_vgpr .AUTO_VGPR_COUNT
        .amdhsa_group_segment_fixed_size .AUTO_LDS_BYTE_SIZE
        .amdhsa_dx10_clamp 0
        .amdhsa_ieee_mode 0
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 0
        .amdhsa_float_denorm_mode_16_64 3
        .amdhsa_reserve_flat_scratch __sgpr_reserve_flatscr
        .amdhsa_reserve_xnack_mask __sgpr_reserve_xnack
        .amdhsa_reserve_vcc __sgpr_reserve_vcc
        .amdhsa_accum_offset ((.AUTO_VGPR_COUNT + 4 - 1) / 4) * 4
.end_amdhsa_kernel
.else
.amdhsa_kernel miopenGcnAsmConv1x1WrW
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 1
        .amdhsa_system_sgpr_workgroup_id_z 1
        .amdhsa_system_vgpr_workitem_id 1
        .amdhsa_next_free_sgpr __amdhsa_next_free_sgpr
        .amdhsa_next_free_vgpr .AUTO_VGPR_COUNT
        .amdhsa_group_segment_fixed_size .AUTO_LDS_BYTE_SIZE
        .amdhsa_dx10_clamp 0
        .amdhsa_ieee_mode 0
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 0
        .amdhsa_float_denorm_mode_16_64 3
        .amdhsa_reserve_flat_scratch __sgpr_reserve_flatscr
        .amdhsa_reserve_xnack_mask __sgpr_reserve_xnack
        .amdhsa_reserve_vcc __sgpr_reserve_vcc
.end_amdhsa_kernel
.endif

.altmacro
.macro METADATA sc, vc, wg_x, lds_size, kernarg_size
.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: miopenGcnAsmConv1x1WrW
    .symbol: miopenGcnAsmConv1x1WrW.kd
    .sgpr_count: \sc
    .vgpr_count: \vc
    .language: "OpenCL C"
    .language_version: [ 1, 2 ]
    .kernarg_segment_size: \kernarg_size
    .kernarg_segment_align: 8
    .group_segment_fixed_size: \lds_size
    .private_segment_fixed_size: 0
    .reqd_workgroup_size: [ \wg_x, 1, 1 ]
    .max_flat_workgroup_size: \wg_x
    .wavefront_size: 64
    .args:
    - { .size: 4, .offset:  0, .value_kind: by_value, .value_type: i32, .name: N }
    - { .size: 4, .offset:  4, .value_kind: by_value, .value_type: i32, .name: C }
    - { .size: 4, .offset:  8, .value_kind: by_value, .value_type: i32, .name: H }
    - { .size: 4, .offset: 12, .value_kind: by_value, .value_type: i32, .name: W }
    - { .size: 4, .offset: 16, .value_kind: by_value, .value_type: i32, .name: K }
    - { .size: 4, .offset: 20, .value_kind: by_value, .value_type: i32, .name: n_groups }
    - { .size: 4, .offset: 24, .value_kind: by_value, .value_type: i32, .name: unused_0 }
    - { .size: 4, .offset: 28, .value_kind: by_value, .value_type: i32, .name: unused_1 }
    - { .size: 8, .offset: 32, .value_kind: global_buffer, .value_type: f32, .name: x,        .address_space: global, .is_const: true }
    - { .size: 8, .offset: 40, .value_kind: global_buffer, .value_type: f32, .name: dw,       .address_space: global, .is_const: false }
    - { .size: 8, .offset: 48, .value_kind: global_buffer, .value_type: f32, .name: dy,       .address_space: global, .is_const: true }
    - { .size: 8, .offset: 56, .value_kind: global_buffer, .value_type: i32, .name: ret_addr, .address_space: global, .is_const: false }
...
.end_amdgpu_metadata
.endm // METADATA

.elseif ROCM_METADATA_VERSION == 4
.altmacro
.macro METADATA sc, vc, wg_x, lds_size, kernarg_size
    .amd_amdgpu_hsa_metadata
    { Version: [ 1, 0 ],
        Kernels:
        - { Name: miopenGcnAsmConv1x1WrW, SymbolName: 'miopenGcnAsmConv1x1WrW@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
            Attrs:
              { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
            CodeProps:
              { KernargSegmentSize: \kernarg_size, GroupSegmentFixedSize: \lds_size, PrivateSegmentFixedSize: 0, KernargSegmentAlign: 8, WavefrontSize: 64, MaxFlatWorkGroupSize: \wg_x }
            Args:
            - { Name: N       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: C       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: H       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: W       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: K       , Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: n_groups, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: unused_0, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: unused_1, Size: 4, Align: 4, ValueKind: ByValue, ValueType: I32, TypeName: 'int', AccQual: Default, IsConst: true }
            - { Name: x       , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true }
            - { Name: dw      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default }
            - { Name: dy      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true }
            - { Name: ret_addr, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: I32, TypeName: 'int*'  , AddrSpaceQual: Global, AccQual: Default }
          }
    }
    .end_amd_amdgpu_hsa_metadata
.endm
.endif

METADATA %.AUTO_SGPR_COUNT, %.AUTO_VGPR_COUNT, %workgroup_size_x, %.AUTO_LDS_BYTE_SIZE, %KERNEL_ARGUMENTS_SIZE

