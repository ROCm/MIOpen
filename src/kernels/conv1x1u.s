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
.globl miopenGcnAsmConv1x1U
.p2align 8
.type miopenGcnAsmConv1x1U,@function

.if ROCM_METADATA_VERSION == 4
.amdgpu_hsa_kernel miopenGcnAsmConv1x1U
.endif

.include "rocm_version.inc"
.include "gpr_alloc.inc"
.include "utilities.inc"
.include "conv_common.inc"
.include "inst_wrappers.inc"

// initial state:
// s[0:1] - kernarg address
// s2 - wg x (HW)
// s3 - wg y (K)
// s4 - wg z (N)
kernarg = 0
gid_x = 2
gid_y = 3
gid_z = 4
gid_hw = gid_x
gid_k = gid_y
gid_n = gid_z

// kernarg layout:
// dwords 0:4 - n, c, H, W, k
// dwords 5:7 - not used
// dwords 8:9 - input buffer pointer
// dwords 10:11 - weights pointer
// dwords 12:13 - output buffer pointer

.set in_ptr_off, 0x20
.set wei_ptr_off, 0x28
.set out_ptr_off, 0x30
.set dbg_ptr_off, 0x38
.set KERNEL_ARGUMENTS_SIZE, dbg_ptr_off + 8

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

maxU24 = 1 << 24
invalid_addr_lit = 0x7FFFFFFF
static_assert (filter_c_stride < maxU24)
static_assert (filter_k_stride < maxU24)
static_assert (input_n_stride < maxU24)
static_assert (input_c_stride < maxU24)
static_assert (output_n_stride < maxU24)
static_assert (output_k_stride < maxU24)
static_assert (input_channels % vec_c_in == 0)
static_assert (output_channels % vec_k_out == 0)//TODO change last conv step

static_assert (pad_h == 0 && pad_w == 0)
static_assert (stride_h == 1 && stride_w == 1)
static_assert (wei_h == 1 && wei_w == 1)

//TODO take them from host, as symbols
// krenel supports fp32 regular layout or fp16_vec2 or int8_vec4
static_assert ((acc_type == TYPE_FP32 && buf_type == TYPE_FP32&& vec_c_in == 1) || (acc_type == TYPE_FP32&& buf_type == TYPE_FP16 && (vec_c_in == 2 || vec_c_in == 1)) || (acc_type == TYPE_INT32 && buf_type == TYPE_INT16 && vec_c_in == 2) || (acc_type == TYPE_INT32 && buf_type == TYPE_INT8 && vec_c_in == 4) || (acc_type == TYPE_INT32 && buf_type == TYPE_INT4 && vec_c_in == 8))
static_assert (k_mult % vec_k_out == 0)

static_assert( (vec_c_in == 1) || (vec_c_in == 2) || (vec_c_in == 4) || (vec_c_in == 8) )
static_assert( (vec_k_out == 1) || (vec_k_out == 2) || (vec_k_out == 4) || (vec_k_out == 8) )
static_assert( (vec_c_in == vec_k_out) || (vec_k_out == 1) )


elements_in_dword = 1
output_dword_chunks_cnt = 1
input_dword_chunks_cnt = 1
.if(buf_type == TYPE_FP16 || buf_type == TYPE_INT16)
    elements_in_dword = 2
.elseif(buf_type == TYPE_INT8)
    elements_in_dword = 4
.elseif(buf_type == TYPE_INT4)
    elements_in_dword = 8
.endif

.if(vec_k_out > elements_in_dword)
    static_assert(vec_k_out == elements_in_dword)
.else
    output_dword_chunks_cnt = elements_in_dword / vec_k_out
.endif

.if(vec_c_in > elements_in_dword)
    static_assert(vec_c_in == elements_in_dword)
.else
    input_dword_chunks_cnt = elements_in_dword / vec_c_in
.endif


img_hw = img_h * img_w
img_hw_vec = (img_h * img_w + input_dword_chunks_cnt - 1) / input_dword_chunks_cnt
rem_hw_in  = (img_h * img_w) % input_dword_chunks_cnt


static_assert( chunks_per_wave % output_dword_chunks_cnt == 0)
static_assert( chunks_per_wave % input_dword_chunks_cnt == 0)

hi_input_channels = (input_channels + vec_c_in - 1) / vec_c_in
hi_output_channels = (output_channels + vec_k_out - 1) / vec_k_out

s_pack_instructions_available = 0
dot_instructions_available = 0
.if (.option.machine_version_major == 9) && (.option.machine_version_minor == 0) && (.option.machine_version_stepping >= 6)
    dot_instructions_available = 1
.endif
madmix_instructions_available = 0
fmamix_instructions_available = 0
.if (.option.machine_version_major == 9)
    .if(.option.machine_version_stepping > 2)
        fmamix_instructions_available = 1
    .else
        madmix_instructions_available = 1
    .endif
    s_pack_instructions_available = 1
.endif

// perf params
default read_size, 2 // 1, 2, 3, 4
default k_mult, 4 // 1..32 (preffer 4*n)
default c_mult, 1 // 1, 2..32 (4*n) //TODO solve sgpr align isue
default chunks_per_wave, 2 // 1..16
default chunk_size, 16 // 1..64
default balanced_n, 1 // 0..1 (deprecated)
default balanced_chunk, 1 // 0..1 (deprecated)
default n_mult, 1 // 1..8
default waves_c_in_group, 1 // 1..8 (preffer 1..4)
default waves_k_in_group, 1 // 1,2,4,8 (preffer 1,2,4,8)
default lds_limit, .MAX_LDS / 8
default disable_case_opt, 0

static_assert ( (read_size * input_dword_chunks_cnt) <= chunks_per_wave)
static_assert (waves_c_in_group <= hi_input_channels && waves_c_in_group <= 16)
static_assert (waves_k_in_group <= hi_output_channels && waves_k_in_group <= 16)
//static_assert (hi_output_channels % k_mult == 0)
default use_saturated_integer_cast, 0

// chunk parameters
n_per_gpr = 64 / chunk_size
static_assert (n_per_gpr * chunk_size == 64)
total_n_blocks = (batch_size + n_per_gpr - 1) / n_per_gpr
.if balanced_n
    active_n_per_gpr = (batch_size + total_n_blocks - 1) / total_n_blocks
.else
    active_n_per_gpr = n_per_gpr
.endif
n_per_wave = n_mult * n_per_gpr
active_n_per_wave = n_mult * active_n_per_gpr


total_chunks = (img_hw + chunk_size - 1) / chunk_size
.if total_chunks < chunks_per_wave
    total_chunks = chunks_per_wave
.endif
.if balanced_chunk
    active_chunk_lanes = (img_hw + total_chunks - 1) / total_chunks
.else
    active_chunk_lanes = chunk_size
.endif
hw_per_wave = chunk_size * chunks_per_wave
active_hw_per_wave = active_chunk_lanes * chunks_per_wave

in_gprs = (n_mult * c_mult * chunks_per_wave  + elements_in_dword - 1) / elements_in_dword

//since we use mix-precision, which accumulates fp16 into fp32, we need vec_size
//times fp16 registers for accumulation
accums_cnt = k_mult * chunks_per_wave * n_mult

// exec mask
log2 chunk_size_log2, chunk_size
.if active_chunk_lanes < 64
    chunk_mask = (1 << active_chunk_lanes) - 1
.else
    chunk_mask = -1
.endif
active_mask = chunk_mask
.rept active_n_per_gpr-1
    active_mask = (active_mask << chunk_size) + chunk_mask
.endr
active_mask_lo = active_mask & 0xFFFFFFFF
active_mask_hi = active_mask >> 32

// group parameters
hi_c_per_wave = (hi_input_channels + waves_c_in_group - 1) / waves_c_in_group
last_wave_hi_c_per_wave = hi_input_channels - hi_c_per_wave * (waves_c_in_group-1)
lds_per_group = 0
double_lds = 0
.if waves_c_in_group > 1
    lds_per_wave = lds_limit / (waves_c_in_group - 1) / waves_k_in_group
    lds_gprs_per_wave = lds_per_wave / (4 * .WAVE_SIZE)
    .if lds_gprs_per_wave >= accums_cnt
        sync_loops = 1
        lds_gprs_per_loop = accums_cnt
        lds_per_group = lds_gprs_per_loop * 4 * .WAVE_SIZE * (waves_c_in_group - 1) * waves_k_in_group
    .else
        lds_gprs_per_loop = lds_gprs_per_wave / 2
        sync_loops = (accums_cnt + lds_gprs_per_loop - 1) / lds_gprs_per_loop
        lds_per_group = 2 * lds_gprs_per_loop * 4 * .WAVE_SIZE * (waves_c_in_group - 1) * waves_k_in_group
        double_lds = 1
    .endif
.endif


raw_filter_dword_k_cnt = 1
.if(weights_layout == 0)
    static_assert ((hi_c_per_wave * vec_c_in) % c_mult == 0 && (last_wave_hi_c_per_wave * vec_c_in )% c_mult == 0)

    filter_c_gpr_stride = 1
    filter_k_gpr_stride = c_mult / elements_in_dword
    sequential_read_size= c_mult / elements_in_dword
    sequential_read_stride = filter_k_stride
    sequential_reads_cnt = k_mult
    static_assert(c_mult % vec_c_filter == 0)
.else
    static_assert ((hi_c_per_wave * vec_c_in) % c_mult == 0 && (last_wave_hi_c_per_wave * vec_c_in )% c_mult == 0)
    raw_filter_dword_k_cnt = elements_in_dword / vec_c_filter
    static_assert(k_mult % (elements_in_dword / vec_c_filter) == 0)
    static_assert (output_channels % k_mult == 0)
    filter_c_gpr_stride = k_mult / (elements_in_dword / vec_c_filter)
    filter_k_gpr_stride = 1
    sequential_read_size= k_mult / (elements_in_dword / vec_c_filter)
    sequential_read_stride = filter_c_stride
    sequential_reads_cnt = c_mult / vec_c_filter

.endif

.GPR_ALLOC_BEGIN

    .SGPR_ALLOC_FROM 5
    .SGPR_ALLOC soffset_in
    .SGPR_ALLOC soffset_out
    .SGPR_ALLOC soffset_wei
    .SGPR_ALLOC desc_in, 4 // input buffer descriptor
    .SGPR_ALLOC desc_out, 4 // weights buffer descriptor
    .SGPR_ALLOC desc_wei, 4 // output buffer descriptor
    .SGPR_ALLOC filtersA, k_mult * c_mult, 1
    .if .SGPR_NEXT_FREE % 4
        .SGPR_ALLOC_ONCE wave_c_id // wave_c_id in group
    .endif
    .if .SGPR_NEXT_FREE % 4
        .SGPR_ALLOC_ONCE loop_cnt
    .endif
    .if .SGPR_NEXT_FREE % 4
        .SGPR_ALLOC_ONCE stmp_offset
    .endif
    .SGPR_ALLOC filtersB, k_mult * c_mult, 1
    .SGPR_ALLOC_ONCE wave_c_id // wave_c_id in group
    .SGPR_ALLOC_ONCE wave_k_id // wave_k_id in group
    .SGPR_ALLOC_ONCE loop_cnt
    .SGPR_ALLOC_ONCE stmp_offset
    .SGPR_ALLOC_ONCE stmp
    //xnack disabled by default
    //.SGPR_RESERVE_XNACK
    .SGPR_RESERVE_VCC

    .VGPR_ALLOC_FROM 0
    .VGPR_ALLOC tid
    .VGPR_ALLOC voffset_in
    .VGPR_ALLOC voffset_out
    .VGPR_ALLOC inputA, in_gprs
    .VGPR_ALLOC inputB, in_gprs
    .VGPR_ALLOC accums, accums_cnt
    .VGPR_ALLOC vtmp
    .if (madmix_instructions_available == 0 && dot_instructions_available == 0 && fmamix_instructions_available == 0)
        .VGPR_ALLOC vtmp_f_cvt
    .endif
    .if (rem_hw_in)
        .VGPR_ALLOC current_hw_in
    .endif

    .LDS_ALLOC_FROM 0
    .LDS_ALLOC accums_lds, lds_per_group

.GPR_ALLOC_END

max_waves_per_CU = (256 / .AUTO_VGPR_COUNT) * 4
static_assert( max_waves_per_CU >= waves_c_in_group * waves_k_in_group)

.macro get_rcp reg, val
   .if \val == 1
      s_mov_b32 s[\reg], 1.0
   .elseif \val == 2
      s_mov_b32 s[\reg], 0.5
   .elseif \val == 3
      s_mov_b32 s[\reg], 0.33333333333
   .elseif \val == 4
      s_mov_b32 s[\reg], 0.25
   .elseif \val == 5
      s_mov_b32 s[\reg], 0.2
   .elseif \val == 6
      s_mov_b32 s[\reg], 0.16666666666
   .elseif \val == 7
      s_mov_b32 s[\reg], 0.14285714285
   .elseif \val == 8
      s_mov_b32 s[\reg], 0.125
   .else
      .error "val > 8"
   .endif
.endm

miopenGcnAsmConv1x1U:
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

    s_load_dwordx2 s[desc_in:desc_in+1], s[kernarg:kernarg+1], 0x0 + in_ptr_off
    s_load_dwordx2 s[desc_wei:desc_wei+1], s[kernarg:kernarg+1], 0x0 + wei_ptr_off
    s_load_dwordx2 s[desc_out:desc_out+1], s[kernarg:kernarg+1], 0x0 + out_ptr_off

    // mask off unused lanes
    s_mov_b32 exec_lo, active_mask_lo
    s_mov_b32 exec_hi, active_mask_hi

    // fill format and size fields of buffer descriptors
    static_assert ((.option.machine_version_major == 8) || (.option.machine_version_major == 9))
    s_mov_b32 s[desc_in+2], input_buffer_size
    s_mov_b32 s[desc_in+3], 0x00027000
    s_mov_b32 s[desc_wei+2], filter_buffer_size
    s_mov_b32 s[desc_wei+3], 0x00027000
    s_mov_b32 s[desc_out+2], output_buffer_size
    s_mov_b32 s[desc_out+3], 0x00027000

    v_lshrrev_b32 v[vtmp], 6, v[tid]
    v_readfirstlane_b32 s[wave_c_id], v[vtmp]
    //wave_k_id = wave_id / waves_c_in_group
    v_cvt_f32_u32 v[vtmp], v[vtmp]
    get_rcp stmp, waves_c_in_group
    v_mul_f32 v[vtmp], v[vtmp], s[stmp]
    v_cvt_u32_f32 v[vtmp], v[vtmp]
    v_readfirstlane_b32 s[wave_k_id], v[vtmp]
    // wave_c_id = wave_id % waves_c_in_group
    s_mul_i32 s[stmp], s[wave_k_id], waves_c_in_group
    s_sub_i32 s[wave_c_id], s[wave_c_id], s[stmp]
    v_and_b32 v[tid], 0x3f, v[tid]

    // calculate input/output offsets
    v_lshrrev_b32 v[vtmp], 0 + chunk_size_log2, v[tid] // vtmp = wave part id
    v_mul_u32_u24 v[voffset_in], 0 + input_n_stride, v[vtmp]
    v_mul_u32_u24 v[voffset_out], 0 + output_n_stride, v[vtmp]

    v_and_b32 v[vtmp], 0 + chunk_size - 1, v[tid] // vtmp = lane in wave part
    v_mul_u32_u24 v[vtmp], 0 + input_w_stride * chunks_per_wave, v[vtmp]
    _v_add_nc_u32 v[voffset_in], v[voffset_in], v[vtmp]

    v_and_b32 v[vtmp], 0 + chunk_size - 1, v[tid] // vtmp = lane in wave part
    v_mul_u32_u24 v[vtmp], 0 + output_w_stride * chunks_per_wave, v[vtmp]
    _v_add_nc_u32 v[voffset_out], v[voffset_out], v[vtmp]

    s_mul_i32 s[soffset_in], s[gid_n], 0 + input_n_stride * active_n_per_wave
    s_mul_i32 s[soffset_out], s[gid_n], 0 + output_n_stride * active_n_per_wave

    s_mul_i32 s[stmp], s[gid_hw], 0 + active_hw_per_wave * input_w_stride
    s_add_u32 s[soffset_in], s[soffset_in], s[stmp]

    s_mul_i32 s[stmp], s[gid_hw], 0 + active_hw_per_wave * output_w_stride
    s_add_u32 s[soffset_out], s[soffset_out], s[stmp]

    s_mul_i32 s[stmp], s[wave_c_id], 0 + hi_c_per_wave * input_c_stride
    s_add_u32 s[soffset_in], s[soffset_in], s[stmp]

    s_mul_i32 s[stmp], s[gid_k], 0 + output_k_stride * k_mult * waves_k_in_group / vec_k_out
    s_add_u32 s[soffset_out], s[soffset_out], s[stmp]
    s_mul_i32 s[stmp], s[wave_k_id], 0 + output_k_stride * k_mult / vec_k_out
    s_add_u32 s[soffset_out], s[soffset_out], s[stmp]
    s_mul_i32 s[soffset_wei], s[gid_k], 0 + k_mult * filter_k_stride * waves_k_in_group
    s_mul_i32 s[stmp], s[wave_k_id], 0 + k_mult * filter_k_stride
    s_add_u32 s[soffset_wei], s[soffset_wei], s[stmp]

    static_assert(vec_c_in == vec_c_filter)
    s_mul_i32 s[stmp], s[wave_c_id], 0 + hi_c_per_wave * filter_c_stride
    s_add_u32 s[soffset_wei], s[soffset_wei], s[stmp]

    s_waitcnt 0

    .macro xsload base, xx, cnt
        .rept \xx
            .if \cnt == 1
                s_buffer_load_dword s[\base], s[desc_wei:desc_wei+3], s[soffset_wei]
            .else
                s_buffer_load_dwordx\cnt s[\base:\base+\cnt-1], s[desc_wei:desc_wei+3], s[soffset_wei]
            .endif
            \base = \base + \cnt
            s_add_u32 s[soffset_wei], s[soffset_wei], 0+4*\cnt
        .endr
    .endm

    .macro load_filters base, seq_size, seq_cnt, seq_stride
        seq_it = 0
        fbase = \base
        .rept \seq_cnt
            x16_chunks = \seq_size / 16
            rest = \seq_size - x16_chunks * 16
            x8_chunks = rest / 8
            rest = rest - x8_chunks * 8
            x4_chunks = rest / 4
            rest = rest - x4_chunks * 4
            x2_chunks = rest / 2
            rest = rest - x2_chunks * 2
            x1_chunks = rest
            imm_off = 0

            xsload fbase, x16_chunks, 16
            xsload fbase, x8_chunks, 8
            xsload fbase, x4_chunks, 4
            xsload fbase, x2_chunks, 2
            xsload fbase, x1_chunks, 1

            seq_it = seq_it + 1
            .if(weights_layout == 0 && seq_it == \seq_cnt)
                s_add_u32 s[soffset_wei], s[soffset_wei], 0 - \seq_stride * (\seq_cnt - 1)
            .else
                s_add_u32 s[soffset_wei], s[soffset_wei], 0 + \seq_stride - 4 * \seq_size
            .endif
        .endr
    .endm

    .if chunks_per_wave % (read_size * input_dword_chunks_cnt)
        mbufs_cnt = (c_mult / elements_in_dword) * n_mult * (1 + chunks_per_wave / (read_size * input_dword_chunks_cnt))
    .else
        mbufs_cnt = (c_mult / elements_in_dword) * n_mult * (chunks_per_wave / (read_size * input_dword_chunks_cnt))
    .endif
    .macro load_input base, mbufs_inflight
        ibase = \base
        hi_c_mult = c_mult / vec_c_in
        full_loads =              chunks_per_wave / (read_size * input_dword_chunks_cnt)
        partial_load_chunks_cnt = chunks_per_wave % (read_size * input_dword_chunks_cnt)
        partial_load_dwords = (partial_load_chunks_cnt + input_dword_chunks_cnt -1) / input_dword_chunks_cnt
        partial_load_short  =  partial_load_chunks_cnt % input_dword_chunks_cnt
        nb = 0
        .rept n_mult
            c_it = 0
            s_mov_b32 s[stmp_offset], s[soffset_in]
            .rept hi_c_mult // input and filter must be vectorized
                s_cmpk_le_i32 s[loop_cnt], 0 + c_it
                s_cmov_b32 s[stmp_offset], 0 + invalid_addr_lit

                ld_it = 0
                imm_off = 0
                current_read_cnt = read_size
                rem_ibase = ibase
                .rept full_loads + 1
                    .if(ld_it == full_loads)
                        current_read_cnt = partial_load_dwords
                    .endif
                    .if tuple_alignment && (current_read_cnt > 1) && (ibase % 2)
                        m_buffer_load_dwordx 1,                  ibase,   voffset_in, desc_in, stmp_offset, imm_off
                        m_buffer_load_dwordx current_read_cnt-1, ibase+1, voffset_in, desc_in, stmp_offset, imm_off+4
                        \mbufs_inflight = \mbufs_inflight + 1
                    .else
                        m_buffer_load_dwordx current_read_cnt, ibase, voffset_in, desc_in, stmp_offset, imm_off
                    .endif
                    ibase = ibase + current_read_cnt
                    imm_off = imm_off + 4 * current_read_cnt
                    ld_it = ld_it + 1
                    //TODO change step size
                .endr
                .if elements_in_dword == 2 && rem_hw_in

                    chunk_id = img_hw / (chunks_per_wave)
                    rem_dword_id = (img_hw % (chunks_per_wave)) / input_dword_chunks_cnt
                    rem_ibase = rem_ibase + rem_dword_id
                    v_cmpx_eq_i32 vcc, 0 + chunk_id * (chunks_per_wave), v[current_hw_in]

                    m_buffer_load_ushort 1,  rem_ibase, voffset_in, desc_in, stmp_offset, rem_dword_id * 4
                    s_mov_b32 exec_lo, active_mask_lo
                    s_mov_b32 exec_hi, active_mask_hi
                .endif
                c_it = c_it + 1
                s_add_u32 s[stmp_offset], s[stmp_offset], input_c_stride
            .endr
            nb = nb + 1

            .if nb == n_mult
                s_add_u32 s[soffset_in], s[soffset_in], 0 + (input_c_stride * hi_c_mult) - input_n_stride * active_n_per_gpr*(n_mult - 1)
            .else
                s_add_u32 s[soffset_in], s[soffset_in], 0 + input_n_stride * active_n_per_gpr
            .endif
        .endr

        s_addk_i32 s[loop_cnt], 0 - (1 * c_mult)
        .if (1|| hi_c_per_wave % 4 || hi_input_channels % hi_c_per_wave)
            s_cmpk_le_i32 s[loop_cnt], 0
            s_cmov_b32 s[desc_in+2], 0
        .endif

    .endm

    .macro get_acc_idx acc, k, n, chunk
        \acc = accums + chunks_per_wave * n_mult * \k + \n * chunks_per_wave + \chunk
    .endm

    //repack imgs between two vgpr
    .macro exch_img, img_c0, img_c1
        v_mov_b32 v[vtmp], v[\img_c0]
        v_mov_b32_sdwa v[\img_c0], v[\img_c1] dst_sel:WORD_1 src0_sel:WORD_0
        v_mov_b32_sdwa v[\img_c1], v[vtmp] dst_sel:WORD_0 src0_sel:WORD_1
    .endm

	.macro exch_filter filter_c0, filter_c1, tmp0, tmp1
        static_assert(\filter_c0 != \filter_c1 && \filter_c0 != \tmp0 && \filter_c1 != \tmp0)
        .if s_pack_instructions_available
            s_mov_b32         s[\tmp0],       s[\filter_c0]
            s_pack_ll_b32_b16 s[\filter_c0],  s[\filter_c0],  s[\filter_c1]
            s_pack_hh_b32_b16 s[\filter_c1],  s[stmp_offset], s[\filter_c1]
        .else
            static_assert(\tmp1 != \filter_c0 && \tmp1 != \filter_c1 && \tmp1 != \tmp0)
            s_lshr_b32 s[\tmp1],      s[\filter_c0], 16
            s_and_b32  s[\tmp0],      s[\filter_c0], 0x0000ffff
            s_lshl_b32 s[\filter_c0], s[\filter_c1], 16
            s_or_b32   s[\filter_c0], s[\filter_c0], s[\tmp0]
            s_and_b32  s[\filter_c1], s[\filter_c1], 0xffff0000
            s_or_b32   s[\filter_c1], s[\filter_c1], s[\tmp1]
        .endif
    .endm

    //repack input across channels
    .macro trans_input ibase
      .if(input_dword_chunks_cnt == 2)
        c = 0
        .rept c_mult / input_dword_chunks_cnt
            n = 0
            .rept n_mult
                ch_gpr = 0
                dwords_with_chunks_from_cx_lane = chunks_per_wave / input_dword_chunks_cnt
                .rept (dwords_with_chunks_from_cx_lane)
                    c_gpr_inp = c * dwords_with_chunks_from_cx_lane
                    n_gpr_inp = n * c_mult * dwords_with_chunks_from_cx_lane
                    img = \ibase + ch_gpr + n_gpr_inp + c_gpr_inp
                    exch_img img, img + dwords_with_chunks_from_cx_lane
                    ch_gpr = ch_gpr + 1
                .endr
                n = n + 1
            .endr
            c = c + input_dword_chunks_cnt
        .endr
      .endif
    .endm

    //repack filter across channels
    .macro trans_filter fbase
      .if(elements_in_dword == 2 &&  raw_filter_dword_k_cnt == 2)
        .if(weights_layout != 0)
            c = 0
            .rept sequential_reads_cnt / raw_filter_dword_k_cnt
                k = 0
                .rept filter_c_gpr_stride
                    c_gpr_filter = (c) * filter_c_gpr_stride
                    k_gpr_filter = k * filter_k_gpr_stride
                    wei = \fbase + k_gpr_filter + c_gpr_filter
                    exch_filter wei, wei + filter_c_gpr_stride, stmp_offset, stmp
                    k = k + 1
                .endr
                c = c + raw_filter_dword_k_cnt
            .endr
        .endif
      .endif
    .endm
    .macro conv ibase, fbase
        chunk_lo_intrans_gpr_stride = chunks_per_wave / input_dword_chunks_cnt
        chunk_hi_intrans_gpr_stride = 1
        hi_c_mult = c_mult / elements_in_dword
        k_lo_gpr_stride = filter_c_gpr_stride
        k_hi_gpr_stride = filter_k_gpr_stride
        c_hi_ftrans_gpr_stride = filter_c_gpr_stride * raw_filter_dword_k_cnt
        c_hi_intrans_gpr_stride = chunks_per_wave
        n_input_gpr_stride = hi_c_mult * c_hi_intrans_gpr_stride
        c_hi = 0
        .rept hi_c_mult
            k = 0
            .rept k_mult
                nb = 0
                .rept n_mult
                    chunk = 0
                    .rept chunks_per_wave
                        get_acc_idx acc, k, nb, chunk
                        k_lo = (k % raw_filter_dword_k_cnt) * k_lo_gpr_stride
                        k_hi = (k / raw_filter_dword_k_cnt) * k_hi_gpr_stride
                        k_gpr_filter = k_lo + k_hi

                        c_gpr_filter = c_hi * c_hi_ftrans_gpr_stride
                        f_gpr = \fbase + k_gpr_filter + c_gpr_filter

                        c_gpr_inp = c_hi * c_hi_intrans_gpr_stride
                        n_gpr_inp = nb * n_input_gpr_stride

                        chunk_lo = ((chunk) % input_dword_chunks_cnt) * chunk_lo_intrans_gpr_stride
                        chunk_hi = (chunk / input_dword_chunks_cnt) * chunk_hi_intrans_gpr_stride
                        inp_gpr = \ibase + c_gpr_inp + n_gpr_inp + chunk_lo + chunk_hi

                        .if acc_type == TYPE_FP32 && buf_type == TYPE_FP32 && vec_c_in == 1
                            v_mac_f32 v[acc], s[f_gpr], v[inp_gpr]
                        .elseif acc_type == TYPE_FP32 && buf_type == TYPE_FP16
                            .if dot_instructions_available
                                v_dot2_f32_f16 v[acc], s[f_gpr], v[inp_gpr], v[acc]
                            .elseif madmix_instructions_available
                                v_mad_mix_f32 v[acc], s[f_gpr], v[inp_gpr], v[acc] op_sel:[0,0,0] op_sel_hi:[1,1,0]
                                v_mad_mix_f32 v[acc], s[f_gpr], v[inp_gpr], v[acc] op_sel:[1,1,0] op_sel_hi:[1,1,0]
                            .elseif fmamix_instructions_available
                                v_fma_mix_f32 v[acc], s[f_gpr], v[inp_gpr], v[acc] op_sel:[0,0,0] op_sel_hi:[1,1,0]
                                v_fma_mix_f32 v[acc], s[f_gpr], v[inp_gpr], v[acc] op_sel:[1,1,0] op_sel_hi:[1,1,0]
                            .else
                                v_mov_b32 v[vtmp_f_cvt], s[f_gpr]
                                v_cvt_f32_f16 v[vtmp], v[inp_gpr]
                                v_cvt_f32_f16 v[vtmp_f_cvt], v[vtmp_f_cvt]
                                v_mac_f32     v[acc], v[vtmp], v[vtmp_f_cvt]

                                v_mov_b32 v[vtmp_f_cvt], s[f_gpr]
                                v_lshrrev_b32 v[vtmp], 16, v[inp_gpr]
                                v_lshrrev_b32 v[vtmp_f_cvt], 16, v[vtmp_f_cvt]

                                v_cvt_f32_f16 v[vtmp], v[vtmp]
                                v_cvt_f32_f16 v[vtmp_f_cvt], v[vtmp_f_cvt]
                                v_mac_f32     v[acc], v[vtmp], v[vtmp_f_cvt]
                            .endif
                        .elseif acc_type == TYPE_INT32 && buf_type == TYPE_INT16 && vec_c_in == 2
                            .if dot_instructions_available
                                v_dot2_i32_i16 v[acc], s[f_gpr], v[inp_gpr], v[acc]
                            .else
                                v_lshlrev_b32 v[vtmp], 16, v[inp_gpr]
                                v_ashrrev_i32 v[vtmp], 16, v[vtmp]
                                s_sext_i32_i16 s[stmp], s[f_gpr]
                                v_mul_i32_i24 v[vtmp], s[stmp], v[vtmp]
                                _v_add_nc_u32 v[acc], v[acc], v[vtmp]

                                v_ashrrev_i32 v[vtmp], 16, v[inp_gpr]
                                s_ashr_i32 s[stmp], s[f_gpr], 16
                                v_mul_i32_i24 v[vtmp], s[stmp], v[vtmp]
                                _v_add_nc_u32 v[acc], v[acc], v[vtmp]
                            .endif
                        .elseif acc_type == TYPE_INT32 && buf_type == TYPE_INT8 && vec_c_in == 4
                            .if dot_instructions_available
                                v_dot4_i32_i8 v[acc], s[f_gpr], v[inp_gpr], v[acc]
                            .else
                                i = 0
                                .rept 4
                                    v_lshlrev_b32 v[vtmp], 24 - 8*i, v[inp_gpr]
                                    v_ashrrev_i32 v[vtmp], 24, v[vtmp]
                                    s_lshl_b32 s[stmp], s[f_gpr], 24 - 8*i
                                    s_ashr_i32 s[stmp], s[stmp], 24

                                    v_mul_i32_i24 v[vtmp], s[stmp], v[vtmp]
                                    _v_add_nc_u32 v[acc], v[acc], v[vtmp]
                                    i = i + 1
                                .endr
                            .endif
                        .elseif acc_type == TYPE_INT32 && buf_type == TYPE_INT4 && vec_c_in == 8
                            .if dot_instructions_available
                                v_dot8_i32_i4 v[acc], s[f_gpr], v[inp_gpr], v[acc]
                            .else
                                i = 0
                                .rept 8
                                    v_lshlrev_b32 v[vtmp], 28 - 4*i, v[inp_gpr]
                                    v_ashrrev_i32 v[vtmp], 28, v[vtmp]
                                    s_lshl_b32 s[stmp], s[f_gpr], 28 - 4*i
                                    s_ashr_i32 s[stmp], s[stmp], 28

                                    v_mul_i32_i24 v[vtmp], s[stmp], v[vtmp]
                                    _v_add_nc_u32 v[acc], v[acc], v[vtmp]
                                    i = i + 1
                                .endr
                            .endif
                        .else
                            static_assert(0)
                        .endif
                        chunk = chunk + 1
                    .endr
                    nb = nb + 1
                .endr
                k = k + 1
            .endr
            c_hi = c_hi + 1
        .endr
    .endm

    .if(rem_hw_in)
        v_and_b32 v[current_hw_in], 0 + chunk_size - 1, v[tid]
        v_mul_u32_u24 v[current_hw_in], 0 + chunks_per_wave, v[current_hw_in]
        s_mul_i32 s[stmp], s[gid_hw], 0 + active_hw_per_wave
        _v_add_nc_u32 v[current_hw_in],  s[stmp], v[current_hw_in]
    .endif
    s_mov_b32 s[loop_cnt], 0 + hi_c_per_wave * vec_c_in
    s_cmpk_eq_u32 s[wave_c_id], 0 + waves_c_in_group - 1
    s_cmov_b32 s[loop_cnt], 0 + last_wave_hi_c_per_wave * vec_c_in

    mbufs_cnt_A = 0
    load_input inputA, mbufs_cnt_A
    load_filters filtersA, sequential_read_size, sequential_reads_cnt, sequential_read_stride

    // zeroing accums
    i = 0
    .rept accums_cnt
        v_mov_b32 v[accums + i], 0
        i = i + 1
    .endr


loop_begin:
    mbufs_cnt_B = 0
    load_input inputB, mbufs_cnt_B
    s_wait (mbufs_cnt+mbufs_cnt_B), 0
    load_filters filtersB, sequential_read_size, sequential_reads_cnt, sequential_read_stride
    trans_input inputA
    trans_filter filtersA
    conv inputA, filtersA

    mbufs_cnt_A = 0
    load_input inputA, mbufs_cnt_A
    s_wait (mbufs_cnt+mbufs_cnt_A), 0
    load_filters filtersA, sequential_read_size, sequential_reads_cnt, sequential_read_stride
    trans_input inputB
    trans_filter filtersB
    conv inputB, filtersB


loop_end:
    s_cmpk_gt_i32 s[loop_cnt], 1 * c_mult
    s_cbranch_scc1 loop_begin

    mbufs_cnt_B = 0
    load_input inputB, mbufs_cnt_B
    s_wait (mbufs_cnt+mbufs_cnt_B), 0
    load_filters filtersB, sequential_read_size, sequential_reads_cnt, sequential_read_stride
    trans_input inputA
    trans_filter filtersA
    conv inputA, filtersA
    s_waitcnt 0

    trans_input inputB
    trans_filter filtersB
    conv inputB, filtersB

    // reduction across waves in group
    // all waves but last store accums to LDS and dies
    // last wave survives and read LDS
    .GPR_REUSE voffset_in, lds_off
    .GPR_REUSE soffset_in, lds_off_k
    .if waves_c_in_group > 1
        s_mul_i32 s[lds_off_k], s[wave_k_id], 4 * .WAVE_SIZE * lds_gprs_per_loop * (waves_c_in_group-1) * (double_lds + 1)
        s_mov_b32 m0, -1
        s_cmpk_eq_u32 s[wave_c_id], 0 + waves_c_in_group - 1
        s_cbranch_scc1 last_wave

        s_mul_i32 s[stmp], s[wave_c_id], 4 * .WAVE_SIZE * lds_gprs_per_loop

        v_lshlrev_b32 v[lds_off], 2, v[tid]
        _v_add_nc_u32 v[lds_off], s[stmp], v[lds_off]
        _v_add_nc_u32 v[lds_off], s[lds_off_k], v[lds_off]
        acc_id = 0
        sync_loop = 0
        .rept sync_loops
            imm_off = (sync_loop % 2) * lds_gprs_per_loop * (waves_c_in_group-1) * 4 * .WAVE_SIZE
            .rept lds_gprs_per_loop
                .if acc_id < accums_cnt
                    ds_write_b32 v[lds_off], v[accums + acc_id], offset:0+imm_off
                .endif
                acc_id = acc_id + 1
                imm_off = imm_off + 4 * .WAVE_SIZE
            .endr
            s_waitcnt 0
            s_barrier
            sync_loop = sync_loop + 1
        .endr

        s_endpgm
last_wave:
        acc_id = 0
        sync_loop = 0
        .rept sync_loops
            v_lshlrev_b32 v[lds_off], 2, v[tid]
            _v_add_nc_u32 v[lds_off], s[lds_off_k], v[lds_off]
            s_barrier
            gpr = 0
            .rept lds_gprs_per_loop
                wave = 0
                .rept waves_c_in_group-1
                    imm_off = 4 * .WAVE_SIZE * (gpr + wave * lds_gprs_per_loop + (sync_loop % 2) * lds_gprs_per_loop * (waves_c_in_group-1))
                    .if acc_id < accums_cnt
                        ds_read_b32 v[vtmp], v[lds_off] offset:0+imm_off
                        s_waitcnt 0
                        .if acc_type == TYPE_FP32
                            v_add_f32 v[accums + acc_id], v[vtmp], v[accums + acc_id]
                        .elseif acc_type == TYPE_INT32
                            _v_add_nc_u32 v[accums + acc_id], v[vtmp], v[accums + acc_id]
                        .endif
                    .endif
                    wave = wave + 1
                .endr
                acc_id = acc_id + 1
                gpr = gpr + 1
            .endr
            sync_loop = sync_loop + 1
        .endr
    .endif

    // Pack output

    .macro set_acc_idx idx, k, nb, chunk
        get_acc_idx acc\idx, \k, \nb, \chunk
    .endm
    .altmacro
    .macro set_all_acc_idx  rounded_ck_base, vec_ck, nb, chunk_base
        _idx = 1
        _ck_local = 0
        _chunk_local = 0
        .rep elements_in_dword
            set_acc_idx %_idx, (\rounded_ck_base * \vec_ck + _ck_local), \nb, \chunk_base + _chunk_local
            .if(\vec_ck == 1)
                _chunk_local = _chunk_local + 1
            .else
                _ck_local = _ck_local + 1
            .endif
            _idx = _idx + 1
        .endr
    .endm

    .if (vec_k_out > 1) || (elements_in_dword > 1)
        nb = 0
        .rept n_mult
            hi_chunk = 0
            .rept chunks_per_wave / (output_dword_chunks_cnt)
                hi_k = 0
                .rept k_mult / vec_k_out
                    set_all_acc_idx hi_k, vec_k_out, nb, hi_chunk  * output_dword_chunks_cnt
                    .if acc_type == TYPE_FP32 && buf_type == TYPE_FP16
                        v_cvt_pkrtz_f16_f32 v[acc1], v[acc1], v[acc2]
                    .elseif acc_type == TYPE_INT32 && buf_type == TYPE_INT16
                        v_cvt_pk_i16_i32 v[acc1], v[acc1], v[acc2]
                    .elseif acc_type == TYPE_INT32 && buf_type == TYPE_INT8
                        v_mov_b32 v[vtmp], 127
                        s_mov_b32 s[stmp], -128
                        .if use_saturated_integer_cast
                            v_med3_i32 v[acc1], v[acc1], v[vtmp], s[stmp]
                            v_med3_i32 v[acc2], v[acc2], v[vtmp], s[stmp]
                            v_med3_i32 v[acc3], v[acc3], v[vtmp], s[stmp]
                            v_med3_i32 v[acc4], v[acc4], v[vtmp], s[stmp]
                        .endif
                        v_and_b32 v[acc1], 0xFF, v[acc1]
                        v_and_b32 v[acc2], 0xFF, v[acc2]
                        v_and_b32 v[acc3], 0xFF, v[acc3]
                        v_lshlrev_b32 v[acc2], 8, v[acc2]
                        v_lshlrev_b32 v[acc3], 16, v[acc3]
                        v_lshlrev_b32 v[acc4], 24, v[acc4]
                        v_or_b32 v[acc1], v[acc1], v[acc4]
                        v_or3_b32 v[acc1], v[acc1], v[acc2], v[acc3]
                    .elseif acc_type == TYPE_INT32 && buf_type == TYPE_INT4
                        i = 0
                        .rept elements_in_dword
                            .if(vec_k_out == 1)
                                get_acc_idx acci, (hi_k * vec_k_out), nb, hi_chunk  + i
                            .else
                                get_acc_idx acci, (hi_k * vec_k_out + i), nb, hi_chunk
                            .endif

                            .if use_saturated_integer_cast
                                v_med3_i32 v[acci], v[acci], -8, 7
                            .endif
                            .if i < (elements_in_dword-1)
                                v_and_b32 v[acci], 0xF, v[acci]
                            .endif
                            .if i > 0
                                v_lshlrev_b32 v[acci], 0 + i*elements_in_dword, v[acci]
                            .endif
                            i = i + 1
                        .endr
                        v_or3_b32 v[acc1], v[acc1], v[acc2], v[acc3]
                        v_or3_b32 v[acc4], v[acc4], v[acc5], v[acc6]
                        v_or_b32 v[acc7], v[acc7], v[acc8]
                        v_or3_b32 v[acc1], v[acc1], v[acc4], v[acc7]
                    .else
                        static_assert(0)
                    .endif
                    hi_k = hi_k + 1
                .endr
                hi_chunk = hi_chunk + 1
            .endr
            nb = nb + 1
        .endr
    .endif

    // store output
    .GPR_REUSE stmp_offset, current_k

    s_mul_i32 s[current_k], s[gid_k], 0 + k_mult * waves_k_in_group / vec_k_out
    s_mul_i32 s[stmp], s[wave_k_id], 0 + k_mult / vec_k_out
    s_add_u32 s[current_k], s[current_k], s[stmp]


    .if(!rem_hw_in)
        .GPR_REUSE inputA, current_hw
        v_and_b32 v[current_hw], 0 + chunk_size - 1, v[tid]
        v_mul_u32_u24 v[current_hw], 0 + chunks_per_wave, v[current_hw]
        s_mul_i32 s[stmp], s[gid_hw], 0 + active_hw_per_wave
        _v_add_nc_u32 v[current_hw],  s[stmp], v[current_hw]
    .else
        .GPR_REUSE current_hw_in, current_hw
    .endif
    .macro store_result
        rem_hw_out = (img_h * img_w) % output_dword_chunks_cnt
        k = 0
        .rept k_mult / vec_k_out
            nb = 0
            s_cmpk_ge_i32 s[current_k], 0 + hi_output_channels - k
            s_cmov_b32 s[desc_out+2], 0

            .rept n_mult
                s_mov_b32 exec_lo, active_mask_lo
                s_mov_b32 exec_hi, active_mask_hi
                chunk = 0
                .rept chunks_per_wave / output_dword_chunks_cnt
                    v_cmpx_ge_i32 vcc, 0 + (img_hw - rem_hw_out) - (chunk + 1) * output_dword_chunks_cnt, v[current_hw]
                    get_acc_idx acc, vec_k_out * k, nb, chunk * output_dword_chunks_cnt
                    buffer_store_dword v[acc], v[voffset_out], s[desc_out:desc_out+3], s[soffset_out] offen offset:0+4 * chunk
                    chunk = chunk + 1
                .endr
                .if(rem_hw_out != 0)
                    //TODO add support for int8
                    s_mov_b32 exec_lo, active_mask_lo
                    s_mov_b32 exec_hi, active_mask_hi

                    chunk_id = img_hw / (chunks_per_wave)

                    v_cmpx_eq_i32 vcc, 0 + chunk_id * chunks_per_wave, v[current_hw]

                    last_dword = (img_hw % chunks_per_wave) / output_dword_chunks_cnt
                    get_acc_idx acc, vec_k_out * k, nb, last_dword * output_dword_chunks_cnt
                    buffer_store_short v[acc], v[voffset_out], s[desc_out:desc_out+3], s[soffset_out] offen offset:0+4 * last_dword
                .endif
                nb = nb + 1
                .if nb == n_mult
                    s_add_u32 s[soffset_out], s[soffset_out], 0 + output_k_stride - active_n_per_gpr * output_n_stride * (n_mult - 1)
                .else
                    s_add_u32 s[soffset_out], s[soffset_out], 0 + active_n_per_gpr * output_n_stride
                .endif
            .endr
            k = k + 1
        .endr
    .endm

    store_result

s_endpgm

.Lfunc_end0:
    .size miopenGcnAsmConv1x1U, .Lfunc_end0 - miopenGcnAsmConv1x1U

waves_in_group = waves_c_in_group * waves_k_in_group
workgroup_size_x = waves_in_group * 64

.if ROCM_METADATA_VERSION == 5
.rodata
.p2align 6
.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_stepping == 10)
.amdhsa_kernel miopenGcnAsmConv1x1U
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 1
        .amdhsa_system_sgpr_workgroup_id_z 1
        .amdhsa_system_vgpr_workitem_id 1

        .amdhsa_next_free_sgpr __amdhsa_next_free_sgpr
        .amdhsa_next_free_vgpr .AUTO_VGPR_COUNT
        .amdhsa_reserve_vcc __sgpr_reserve_vcc
        .amdhsa_reserve_xnack_mask __sgpr_reserve_xnack
        .amdhsa_reserve_flat_scratch __sgpr_reserve_flatscr

        .amdhsa_group_segment_fixed_size .AUTO_LDS_BYTE_SIZE
        .amdhsa_dx10_clamp 0
        .amdhsa_ieee_mode 0
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 0
        .amdhsa_float_denorm_mode_16_64 3
        .amdhsa_accum_offset ((.AUTO_VGPR_COUNT + 4 - 1) / 4) * 4
.end_amdhsa_kernel
.else
.amdhsa_kernel miopenGcnAsmConv1x1U
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 1
        .amdhsa_system_sgpr_workgroup_id_z 1
        .amdhsa_system_vgpr_workitem_id 1

        .amdhsa_next_free_sgpr __amdhsa_next_free_sgpr
        .amdhsa_next_free_vgpr .AUTO_VGPR_COUNT
        .amdhsa_reserve_vcc __sgpr_reserve_vcc
        .amdhsa_reserve_xnack_mask __sgpr_reserve_xnack
        .amdhsa_reserve_flat_scratch __sgpr_reserve_flatscr

        .amdhsa_group_segment_fixed_size .AUTO_LDS_BYTE_SIZE
        .amdhsa_dx10_clamp 0
        .amdhsa_ieee_mode 0
        .amdhsa_float_round_mode_32 0
        .amdhsa_float_round_mode_16_64 0
        .amdhsa_float_denorm_mode_32 0
        .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.endif

.altmacro
.macro METADATA sc,vc,wg_x,lds_sz,kernarg_size
.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: miopenGcnAsmConv1x1U
    .symbol: miopenGcnAsmConv1x1U.kd
    .sgpr_count: \sc
    .vgpr_count: \vc
    .language: "OpenCL C"
    .language_version: [ 1, 2 ]
    .kernarg_segment_size: \kernarg_size
    .kernarg_segment_align: 8
    .group_segment_fixed_size: \lds_sz
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
    - { .size: 8, .offset: 40, .value_kind: global_buffer, .value_type: f32, .name: w,        .address_space: global, .is_const: true }
    - { .size: 8, .offset: 48, .value_kind: global_buffer, .value_type: f32, .name: y,        .address_space: global, .is_const: false }
    - { .size: 8, .offset: 56, .value_kind: global_buffer, .value_type: i32, .name: ret_addr, .address_space: global, .is_const: false }
...
.end_amdgpu_metadata
.endm // METADATA

METADATA %.AUTO_SGPR_COUNT, %.AUTO_VGPR_COUNT, %workgroup_size_x, %.AUTO_LDS_BYTE_SIZE, %KERNEL_ARGUMENTS_SIZE

.elseif ROCM_METADATA_VERSION == 4
.macro METADATA wg_x, lds_size, kernarg_size
    .amd_amdgpu_hsa_metadata
    { Version: [ 1, 0 ],
        Kernels:
        - { Name: miopenGcnAsmConv1x1U, SymbolName: 'miopenGcnAsmConv1x1U@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
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
            - { Name: w       , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true }
            - { Name: y       , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default }
            - { Name: ret_addr, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: I32, TypeName: 'int*'  , AddrSpaceQual: Global, AccQual: Default }
          }
    }
    .end_amd_amdgpu_hsa_metadata
.endm

.altmacro
.macro METADATA_WRAPPER wg_x, lds_size, kernarg_size
    METADATA %\wg_x, %\lds_size, %\kernarg_size
.endm

METADATA_WRAPPER workgroup_size_x, .AUTO_LDS_BYTE_SIZE, KERNEL_ARGUMENTS_SIZE
.endif

