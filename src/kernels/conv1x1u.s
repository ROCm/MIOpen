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

.hsa_code_object_version 2,1
.hsa_code_object_isa

.text
.globl gcnAsmConv1x1U
.p2align 8
.type gcnAsmConv1x1U,@function
.amdgpu_hsa_kernel gcnAsmConv1x1U

.include "gpr_alloc.inc"
.include "common.inc"
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

.set vec_size, 2

static_assert(c_mult % vec_size == 0)
static_assert(k_mult % vec_size == 0)
static_assert(c_mult % 2 == 0)
static_assert(k_mult % 2 == 0)

.include "conv_sizes.inc"

static_assert ((.option.machine_version_major == 8) || (.option.machine_version_major == 9))

maxU24 = 1 << 24
invalid_addr_lit = 0x7FFFFFFF
static_assert (filter_c_stride < maxU24)
static_assert (filter_k_stride < maxU24)
static_assert (input_n_stride < maxU24)
static_assert (output_n_stride < maxU24)
static_assert (input_c_stride < maxU24)
static_assert (output_k_stride < maxU24)

static_assert (pad_h == 0 && pad_w == 0)
static_assert (stride_h == 1 && stride_w == 1)
static_assert (wei_h == 1 && wei_w == 1)

dot_instructions_available = 0
.if (.option.machine_version_major == 9) && (.option.machine_version_minor == 0) && (.option.machine_version_stepping == 6)
    dot_instructions_available = 1
.endif


// perf params

.ifnotdef do_not_use_default_perf_params
	default read_size, 2 // 1, 2, 3, 4
	default k_mult, 4 // 1, 2..32 (4*n)
	default c_mult, 1 // 1, 2..32 (4*n)
	default chunks_per_wave, 2 // 1..16
	default chunk_size, 16 // 1..64
	default n_mult, 1 // 1..8
	default waves_in_group, 1 // 1..16
.endif
default balanced_n, 1
default balanced_chunk, 1

default lds_limit, .MAX_LDS / 8
default disable_case_opt, 1


static_assert (read_size <= chunks_per_wave)
static_assert (waves_in_group <= input_channels && waves_in_group <= 8)
//static_assert (output_channels % k_mult == 0)


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

in_gprs = chunks_per_wave * n_mult * c_mult

//since we use mix-precision, which accumulates fp16 into fp32, we need vec_size
//times fp16 registers for accumulation
accums_cnt = k_mult * chunks_per_wave * n_mult * vec_size

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
c_per_wave = (input_channels + waves_in_group - 1) / waves_in_group
lds_per_group = 0
.if waves_in_group > 1
    lds_per_wave = lds_limit / (waves_in_group - 1)
    lds_gprs_per_wave = lds_per_wave / (4 * .WAVE_SIZE)
    .if lds_gprs_per_wave >= accums_cnt
        sync_loops = 1
        lds_gprs_per_loop = accums_cnt
        lds_per_group = lds_gprs_per_loop * 4 * .WAVE_SIZE * (waves_in_group - 1)
    .else
        lds_gprs_per_loop = lds_gprs_per_wave / 2
        sync_loops = (accums_cnt + lds_gprs_per_loop - 1) / lds_gprs_per_loop
        lds_per_group = 2 * lds_gprs_per_loop * 4 * .WAVE_SIZE * (waves_in_group - 1)
    .endif
.endif

.if(weights_layout == 0)
    filter_c_gpr_stride = 1
    filter_k_gpr_stride = c_mult / vec_size
    sequential_read_size= c_mult / vec_size
    sequential_read_stride = filter_k_stride
    sequential_reads_cnt = k_mult
.else
    filter_c_gpr_stride = k_mult / vec_size
    filter_k_gpr_stride = 1
    sequential_read_size= k_mult / vec_size
    sequential_read_stride = filter_c_stride
    sequential_reads_cnt = c_mult
.endif


input_buffer_size = input_n_stride * batch_size + 4 / vec_size //padding the last fp16 element for odd hw
filter_buffer_size = filters_size
output_buffer_size = output_n_stride * batch_size + 4 / vec_size //padding the last fp16 element for odd hw

//static_assert(input_channels % (c_mult * waves_in_group) == 0) //todo: remove me

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
        .SGPR_ALLOC_ONCE wave_id // wave_id in group
    .endif
    .if .SGPR_NEXT_FREE % 4
        .SGPR_ALLOC_ONCE loop_cnt
    .endif
    .if .SGPR_NEXT_FREE % 4
        .SGPR_ALLOC_ONCE stmp_offset
    .endif
    .SGPR_ALLOC filtersB, k_mult * c_mult, 1
    .SGPR_ALLOC_ONCE wave_id // wave_id in group
    .SGPR_ALLOC_ONCE loop_cnt
    .SGPR_ALLOC_ONCE stmp_offset
    .SGPR_ALLOC_ONCE stmp
    .SGPR_RESERVE_XNACK
    
    .VGPR_ALLOC_FROM 0
    .VGPR_ALLOC tid
    .VGPR_ALLOC voffset_in
    .VGPR_ALLOC voffset_out
    .VGPR_ALLOC inputA, in_gprs
    .VGPR_ALLOC inputB, in_gprs
    .VGPR_ALLOC accums, accums_cnt
    .VGPR_ALLOC vtmp
    
    .LDS_ALLOC_FROM 0
    .LDS_ALLOC accums_lds, lds_per_group
    
.GPR_ALLOC_END

max_waves_per_CU = (256 / .AUTO_VGPR_COUNT) * 4
static_assert( max_waves_per_CU >= waves_in_group )

gcnAsmConv1x1U:
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
     kernarg_segment_byte_size = 64
     wavefront_sgpr_count = .AUTO_SGPR_COUNT
     workitem_vgpr_count = .AUTO_VGPR_COUNT
     float_mode = 192
     workgroup_group_segment_byte_size = .AUTO_LDS_BYTE_SIZE
    .end_amd_kernel_code_t


    
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
    v_readfirstlane_b32 s[wave_id], v[vtmp]
    v_and_b32 v[tid], 0x3f, v[tid]
    
    // calculate input/output offsets
    v_lshrrev_b32 v[vtmp], 0 + chunk_size_log2, v[tid] // vtmp = wave part id
    v_mul_u32_u24 v[voffset_in], 0 + input_n_stride, v[vtmp]
    v_mul_u32_u24 v[voffset_out], 0 + output_n_stride, v[vtmp]
    v_and_b32 v[vtmp], 0 + chunk_size - 1, v[tid] // vtmp = lane in wave part
    v_mul_u32_u24 v[vtmp], 4 * chunks_per_wave, v[vtmp]
    _v_add_nc_u32 v[voffset_in], v[voffset_in], v[vtmp]
    _v_add_nc_u32 v[voffset_out], v[voffset_out], v[vtmp]
    s_mul_i32 s[soffset_in], s[gid_n], 0 + input_n_stride * active_n_per_wave
    s_mul_i32 s[soffset_out], s[gid_n], 0 + output_n_stride * active_n_per_wave
    s_mul_i32 s[stmp], s[gid_hw], 0 + active_hw_per_wave * 4
    s_add_u32 s[soffset_in], s[soffset_in], s[stmp]
    s_add_u32 s[soffset_out], s[soffset_out], s[stmp]
    s_mul_i32 s[stmp], s[wave_id], 0 + c_per_wave * input_c_stride
    s_add_u32 s[soffset_in], s[soffset_in], s[stmp]
    s_mul_i32 s[stmp], s[gid_k], 0 + output_k_stride * k_mult
    s_add_u32 s[soffset_out], s[soffset_out], s[stmp]
    s_mul_i32 s[soffset_wei], s[gid_k], 0 + k_mult * filter_k_stride
    s_mul_i32 s[stmp], s[wave_id], 0 + c_per_wave * filter_c_stride
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

    .if chunks_per_wave % read_size
        mbufs_cnt = c_mult * n_mult * (1 + chunks_per_wave / read_size)
    .else
        mbufs_cnt = c_mult * n_mult * (chunks_per_wave / read_size)
    .endif
    .macro load_input base
        ibase = \base
        full_loads = chunks_per_wave / read_size
        partial_load_size = chunks_per_wave % read_size
        nb = 0
        .rept n_mult
            c_it = 0
            s_mov_b32 s[stmp_offset], s[soffset_in]
            .rept c_mult
                s_cmpk_le_i32 s[loop_cnt], 0 + c_it
                s_cmov_b32 s[stmp_offset], 0 + invalid_addr_lit

                imm_off = 0
                .rept full_loads
                    m_buffer_load_dwordx read_size, ibase, voffset_in, desc_in, stmp_offset, imm_off
                    ibase = ibase + read_size
                    imm_off = imm_off + 4 * read_size
                    //TODO change step size
                .endr
                m_buffer_load_dwordx partial_load_size, ibase, voffset_in, desc_in, stmp_offset, imm_off
                ibase = ibase + partial_load_size
                c_it = c_it + 1
                s_add_u32 s[stmp_offset], s[stmp_offset], input_c_stride
            .endr
            nb = nb + 1
            .if nb == n_mult
                s_add_u32 s[soffset_in], s[soffset_in], 0 + (input_c_stride * c_mult) - input_n_stride * (active_n_per_wave - active_n_per_gpr)
            .else
                s_add_u32 s[soffset_in], s[soffset_in], 0 + input_n_stride * active_n_per_gpr
            .endif
        .endr

        s_addk_i32 s[loop_cnt], 0 - (1 * c_mult)
        .if (disable_case_opt || c_per_wave % 4 || input_channels % c_per_wave)
            s_cmpk_le_i32 s[loop_cnt], 0
            s_cmov_b32 s[desc_in+2], 0
        .endif

    .endm
    
    //perf conv on packed img vgpr 
    .macro conv_line acc, wei, img
        .if dot_instructions_available
            v_dot2_f32_f16 v[\acc], s[\wei], v[\img], v[\acc]
        .else
            v_mad_mix_f32 v[\acc], s[\wei], v[\img], v[\acc] op_sel:[0,0,0] op_sel_hi:[1,1,0]
            v_mad_mix_f32 v[\acc], s[\wei], v[\img], v[\acc] op_sel:[1,1,0] op_sel_hi:[1,1,0]
        .endif
        #v_fma_mix_f32 v[\acc], s[\wei], v[\img], v[\acc] op_sel:[0,0,0] op_sel_hi:[1,1,0]
        #v_fma_mix_f32 v[\acc], s[\wei], v[\img], v[\acc] op_sel:[1,1,0] op_sel_hi:[1,1,0]
    .endm

    //repack imgs between two vgpr
    .macro exch_img, img_c0, img_c1
        v_mov_b32 v[vtmp], v[\img_c0]
        v_mov_b32_sdwa v[\img_c0], v[\img_c1] dst_sel:WORD_1 src0_sel:WORD_0
        v_mov_b32_sdwa v[\img_c1], v[vtmp] dst_sel:WORD_0 src0_sel:WORD_1
    .endm

    //repack filter between two sgpr
    .macro exch_filter, filter_c0, filter_c1
        s_mov_b32 s[stmp], s[\filter_c0]
        s_pack_ll_b32_b16 s[\filter_c0], s[\filter_c0], s[\filter_c1] 
        s_pack_hh_b32_b16 s[\filter_c1], s[stmp], s[\filter_c1]
    .endm

    //repack input across channels
    .macro trans_input ibase
        c = 0
        .rept c_mult / vec_size
            n = 0
            .rept n_mult
                ch_gpr = 0
                .rept chunks_per_wave
                    c_gpr_inp = c * chunks_per_wave
                    n_gpr_inp = n * c_mult * chunks_per_wave
                    img = \ibase + ch_gpr + n_gpr_inp + c_gpr_inp
                    exch_img img, img + chunks_per_wave
                    ch_gpr = ch_gpr + 1
                .endr
                n = n + 1
            .endr
            c = c + vec_size
        .endr
    .endm

    //repack filter across channels
    .macro trans_filter fbase
        .if(weights_layout != 0)
            c = 0
            .rept sequential_reads_cnt 
                k = 0
                .rept filter_c_gpr_stride 
                    c_gpr_filter = c * filter_c_gpr_stride
                    k_gpr_filter = k * filter_k_gpr_stride
                    wei = \fbase + k_gpr_filter + c_gpr_filter
                    exch_filter wei, wei + filter_c_gpr_stride 
                    k = k + 1
                .endr
                c = c + vec_size 
            .endr
        .endif
    .endm

    .macro conv ibase, fbase
        c = 0
        .rept c_mult / vec_size
            k = 0
            .rept k_mult
                n = 0
                .rept n_mult
                    ch_gpr = 0
                    .rept chunks_per_wave
                        c_gpr_inp = c * chunks_per_wave
                        n_gpr_inp = n * c_mult * chunks_per_wave
                        n_gpr_acc = n * chunks_per_wave
                        k_gpr_acc = k * n_mult * chunks_per_wave

                        img = \ibase + ch_gpr + n_gpr_inp + c_gpr_inp * vec_size
                        acc = accums + (k_gpr_acc + n_gpr_acc + ch_gpr) * vec_size

                        .if(weights_layout == 0) //wei[k][c]
                            c_gpr_filter = c * filter_c_gpr_stride
                            k_gpr_filter = k * filter_k_gpr_stride
                            wei = \fbase + k_gpr_filter + c_gpr_filter
                        .else //wei[c][k]
                            x = k / vec_size
                            y = k % vec_size
                            wei = \fbase + x + y * filter_c_gpr_stride + k_mult * c
                        .endif

                        #v_mov_b32 v[img], 0x3C003C00
                        #v_mov_b32 v[img + chunks_per_wave], 0x3C003C00
                        #s_mov_b32 s[wei], 0x3C003C00

                        .if vec_size == 2
                            conv_line acc, wei, img
                            conv_line acc+1, wei, img + chunks_per_wave
                        .else
                            v_mac_f32 v[accums + k_gpr_acc + n_gpr_acc + ch_gpr], s[\fbase + k_gpr_filter + c_gpr_filter], v[\ibase + ch_gpr + n_gpr_inp + c_gpr_inp]
                        .endif

                        ch_gpr = ch_gpr + 1
                    .endr
                    n = n + 1
                .endr
                k = k + 1
            .endr
            c = c + 1 
        .endr
    .endm

    s_mov_b32 s[loop_cnt], 0 + c_per_wave
    s_cmpk_eq_u32 s[wave_id], 0 + waves_in_group - 1
    s_cmov_b32 s[loop_cnt], 0 + input_channels - c_per_wave * (waves_in_group-1)

    load_input inputA
    load_filters filtersA, sequential_read_size, sequential_reads_cnt, sequential_read_stride
    
    // zeroing accums
    i = 0
    .rept accums_cnt
        v_mov_b32 v[accums + i], 0
        i = i + 1
    .endr


loop_begin:
    load_input inputB
    s_wait mbufs_cnt, 0
    load_filters filtersB, sequential_read_size, sequential_reads_cnt, sequential_read_stride
    trans_input inputA
    trans_filter filtersA
    conv inputA, filtersA

    load_input inputA
    s_wait mbufs_cnt, 0
    load_filters filtersA, sequential_read_size, sequential_reads_cnt, sequential_read_stride
    trans_input inputB
    trans_filter filtersB
    conv inputB, filtersB

   
loop_end:
    //load vec_size * c_mult channels once
    s_cmpk_gt_i32 s[loop_cnt], 1 * c_mult
    s_cbranch_scc1 loop_begin

    load_input inputB
    s_wait mbufs_cnt, 0
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
    .if waves_in_group > 1
        s_mov_b32 m0, -1
        s_cmpk_eq_u32 s[wave_id], 0 + waves_in_group - 1
        s_cbranch_scc1 last_wave
        
        s_mul_i32 s[stmp], s[wave_id], 4 * .WAVE_SIZE * lds_gprs_per_loop

        v_lshlrev_b32 v[lds_off], 2, v[tid]
        _v_add_nc_u32 v[lds_off], s[stmp], v[lds_off]
        acc_id = 0
        sync_loop = 0
        .rept sync_loops 
            imm_off = (sync_loop % 2) * lds_gprs_per_loop * (waves_in_group-1) * 4 * .WAVE_SIZE
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
            s_barrier
            gpr = 0
            .rept lds_gprs_per_loop
                wave = 0
                .rept waves_in_group-1
                    imm_off = 4 * .WAVE_SIZE * (gpr + wave * lds_gprs_per_loop + (sync_loop % 2) * lds_gprs_per_loop * (waves_in_group-1))
                    .if acc_id < accums_cnt
                        ds_read_b32 v[vtmp], v[lds_off] offset:0+imm_off
                        s_waitcnt 0
                        v_add_f32 v[accums + acc_id], v[vtmp], v[accums + acc_id]
                    .endif
                    wave = wave + 1
                .endr
                acc_id = acc_id + 1
                gpr = gpr + 1
            .endr
            sync_loop = sync_loop + 1
        .endr
    .endif
    
    // store output
    .GPR_REUSE stmp_offset, current_k
    .GPR_REUSE inputA, current_hw
    s_mul_i32 s[current_k], s[gid_k], 0 + k_mult
    v_and_b32 v[current_hw], 0 + chunk_size - 1, v[tid]
    v_mul_u32_u24 v[current_hw], 0 + chunks_per_wave, v[current_hw]
    s_mul_i32 s[stmp], s[gid_hw], 0 + active_hw_per_wave
    _v_add_nc_u32 v[current_hw],  s[stmp], v[current_hw]
    
    k = 0
    acc = accums
    .rept k_mult
        nb = 0
        .rept n_mult
            s_mov_b32 exec_lo, active_mask_lo
            s_mov_b32 exec_hi, active_mask_hi
            chunk = 0
            acc_t = acc
            .rept chunks_per_wave
                v_cmpx_gt_i32 vcc, 0 + (img_hw - odd_hw) - chunk, v[current_hw]
                //cvt and packed two fp32 into a fp32
                v_cvt_pkrtz_f16_f32 v[acc], v[acc], v[acc+1]
                buffer_store_dword v[acc], v[voffset_out], s[desc_out:desc_out+3], s[soffset_out] offen offset:0+4*chunk
                chunk = chunk + 1
                acc = acc + vec_size
            .endr
            .if odd_hw
                s_mov_b32 exec_lo, active_mask_lo
                s_mov_b32 exec_hi, active_mask_hi
                v_cmpx_eq_i32 vcc, 0 + img_hw - odd_hw, v[current_hw]
                v_cvt_f16_f32 v[acc_t], v[acc_t]
                buffer_store_short v[acc_t], v[voffset_out], s[desc_out:desc_out+3], s[soffset_out] offen offset:0
            .endif
            nb = nb + 1
            .if nb == n_mult
                s_add_u32 s[soffset_out], s[soffset_out], 0 + input_c_stride - (active_n_per_wave-active_n_per_gpr) * output_n_stride
            .else
                s_add_u32 s[soffset_out], s[soffset_out], 0 + active_n_per_gpr * output_n_stride
            .endif
        .endr
        .if (disable_case_opt || output_channels % k_mult)
            s_cmpk_ge_i32 s[current_k], 0 + output_channels - k - 1
            s_cmov_b32 s[desc_out+2], 0
            k = k + 1
        .endif
    .endr
    
    
s_endpgm

.Lfunc_end0:
    .size gcnAsmConv1x1U, .Lfunc_end0 - gcnAsmConv1x1U

.ifndef ROCM_METADATA_VERSION
.error "ROCM_METADATA_VERSION must be defined"
.end
.endif

.macro metadata wg_x
  .if ROCM_METADATA_VERSION == 3
    .amdgpu_code_object_metadata
    { Version: [ 3, 0 ],
        Kernels:
        - { Name: gcnAsmConv1x1U, Language: OpenCL C, LanguageVersion: [ 1, 2 ],
            Attrs:
              { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
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
    .end_amdgpu_code_object_metadata
  .endif
  .if ROCM_METADATA_VERSION == 4
    .amd_amdgpu_hsa_metadata
    { Version: [ 1, 0 ],
        Kernels:
        - { Name: gcnAsmConv1x1U, SymbolName: 'gcnAsmConv1x1U@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
            Attrs:
              { ReqdWorkGroupSize: [ \wg_x, 1, 1 ] }
            CodeProps:
              { KernargSegmentSize: 64, GroupSegmentFixedSize: 0, PrivateSegmentFixedSize: 0, KernargSegmentAlign: 8, WavefrontSize: 64, MaxFlatWorkGroupSize: 512 }
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
  .endif
.endm

.if waves_in_group == 8
    metadata 512
.elseif waves_in_group == 7
    metadata 448
.elseif waves_in_group == 6
    metadata 384
.elseif waves_in_group == 5
    metadata 320
.elseif waves_in_group == 4
    metadata 256
.elseif waves_in_group == 3
    metadata 192
.elseif waves_in_group == 2
    metadata 128
.else
    metadata 64
.endif
