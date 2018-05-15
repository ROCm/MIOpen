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

.include "conv_sizes.inc"

static_assert ((.option.machine_version_major == 8) || (.option.machine_version_major == 9))
maxU24 = 1 << 24
static_assert (filter_c_stride < maxU24)
static_assert (filter_k_stride < maxU24)
static_assert (input_stack_size < maxU24)
static_assert (output_stack_size < maxU24)
static_assert (input_feature_map_size < maxU24)
static_assert (output_feature_map_size < maxU24)
static_assert (pad_h == 0 && pad_w == 0)
static_assert (stride_h == 1 && stride_w == 1)
static_assert (wei_h == 1 && wei_w == 1)

// perf params
//default read_size, 2 // 1, 2, 3, 4
//default k_mult, 4 // 1..32 (prefer {1,[4,8,12,..32]})
//default chunks_per_wave, 2 // 1..16
//default chunk_size, 16 // 1..64
default balanced_n, 1 // 0..1 (deprecated)
default balanced_chunk, 1 // 0..1 (deprecated)
//default n_blocks_per_wave, 1 // 1..8
//default waves_in_group, 1 // 1..16 (prefer 1..8)
default lds_limit, .MAX_LDS / 8
default disable_case_opt, 0

static_assert (read_size <= chunks_per_wave)
static_assert (waves_in_group <= input_channels && waves_in_group <= 8) // was 16
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
n_per_wave = n_blocks_per_wave * n_per_gpr
active_n_per_wave = n_blocks_per_wave * active_n_per_gpr

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

in_gprs = chunks_per_wave * n_blocks_per_wave
accums_cnt = k_mult * chunks_per_wave * n_blocks_per_wave

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

//wg_cnt_n = (batch_size + n_per_wave - 1) / n_per_wave
//wg_cnt_k = (output_channels + k_mult - 1) / k_mult
//wg_cnt_hw = (img_hw + hw_per_wave - 1) / hw_per_wave

input_buffer_size = input_stack_size * batch_size
filter_buffer_size = filters_size
output_buffer_size = output_stack_size * batch_size

.GPR_ALLOC_BEGIN
    .SGPR_ALLOC_FROM 5
    .SGPR_ALLOC soffset_in
    .SGPR_ALLOC soffset_out
    .SGPR_ALLOC soffset_wei
    .SGPR_ALLOC desc_in, 4 // input buffer descriptor
    .SGPR_ALLOC desc_out, 4    // weights buffer descriptor
    .SGPR_ALLOC desc_wei, 4    // output buffer descriptor
    .SGPR_ALLOC filtersA, weights_per_filter * k_mult, 1
    .if .SGPR_NEXT_FREE % 4
        .SGPR_ALLOC_ONCE wave_id // wave_id in group
    .endif
    .if .SGPR_NEXT_FREE % 4
        .SGPR_ALLOC_ONCE loop_cnt
    .endif
    .if .SGPR_NEXT_FREE % 4
        .SGPR_ALLOC_ONCE current_c
    .endif
    .SGPR_ALLOC filtersB, weights_per_filter * k_mult, 1
    .SGPR_ALLOC_ONCE wave_id // wave_id in group
    .SGPR_ALLOC_ONCE loop_cnt
    .SGPR_ALLOC_ONCE current_c
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
    s_mov_b32 s[desc_wei+2], filters_size
    s_mov_b32 s[desc_wei+3], 0x00027000
    s_mov_b32 s[desc_out+2], output_buffer_size
    s_mov_b32 s[desc_out+3], 0x00027000

    v_lshrrev_b32 v[vtmp], 6, v[tid]
    v_readfirstlane_b32 s[wave_id], v[vtmp]
    v_and_b32 v[tid], 0x3f, v[tid]

    // calculate input/output offsets
    v_lshrrev_b32 v[vtmp], 0 + chunk_size_log2, v[tid] // vtmp = wave part id
    v_mul_u32_u24 v[voffset_in], 0 + input_stack_size, v[vtmp]
    v_mul_u32_u24 v[voffset_out], 0 + output_stack_size, v[vtmp]
    v_and_b32 v[vtmp], 0 + chunk_size - 1, v[tid] // vtmp = lane in wave part
    v_mul_u32_u24 v[vtmp], 4 * chunks_per_wave, v[vtmp]
   _v_add_co_u32 v[voffset_in], vcc, v[voffset_in], v[vtmp]
   _v_add_co_u32 v[voffset_out], vcc, v[voffset_out], v[vtmp]
    s_mul_i32 s[soffset_in], s[gid_n], 0 + input_stack_size * active_n_per_wave
    s_mul_i32 s[soffset_out], s[gid_n], 0 + output_stack_size * active_n_per_wave
    s_mul_i32 s[stmp], s[gid_hw], 0 + active_hw_per_wave * 4
    s_add_u32 s[soffset_in], s[soffset_in], s[stmp]
    s_add_u32 s[soffset_out], s[soffset_out], s[stmp]
    s_mul_i32 s[stmp], s[wave_id], 0 + c_per_wave * input_feature_map_size
    s_add_u32 s[soffset_in], s[soffset_in], s[stmp]
    s_mul_i32 s[stmp], s[gid_k], 0 + output_feature_map_size * k_mult
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

    .macro load_filters base
        .if weights_layout == 0
            i = 0
            .rept k_mult
                s_buffer_load_dword s[\base + i], s[desc_wei:desc_wei+3], s[soffset_wei]
                i = i + 1
                .if i == k_mult
                    s_add_u32 s[soffset_wei], s[soffset_wei], 0 + filter_c_stride - filter_k_stride * (k_mult - 1)
                .else
                    s_add_u32 s[soffset_wei], s[soffset_wei], 0 + filter_k_stride
                .endif
            .endr
        .else
            x16_chunks = k_mult / 16
            rest = k_mult - x16_chunks * 16
            x8_chunks = rest / 8
            rest = rest - x8_chunks * 8
            x4_chunks = rest / 4
            rest = rest - x4_chunks * 4
            x2_chunks = rest / 2
            rest = rest - x2_chunks * 2
            x1_chunks = rest
            imm_off = 0
            fbase = \base
            xsload fbase, x16_chunks, 16
            xsload fbase, x8_chunks, 8
            xsload fbase, x4_chunks, 4
            xsload fbase, x2_chunks, 2
            xsload fbase, x1_chunks, 1
            s_add_u32 s[soffset_wei], s[soffset_wei], 0 + filter_c_stride - 4*k_mult
        .endif
    .endm

    .macro vload base, count, imm_off
        .if \count == 1
            buffer_load_dword v[\base], v[voffset_in], s[desc_in:desc_in+3], s[soffset_in] offen offset:0+\imm_off
        .elseif \count == 2
            buffer_load_dwordx2 v[\base:\base+1], v[voffset_in], s[desc_in:desc_in+3], s[soffset_in] offen offset:0+\imm_off
        .elseif \count == 3
            buffer_load_dwordx3 v[\base:\base+2], v[voffset_in], s[desc_in:desc_in+3], s[soffset_in] offen offset:0+\imm_off
        .elseif \count == 4
            buffer_load_dwordx4 v[\base:\base+3], v[voffset_in], s[desc_in:desc_in+3], s[soffset_in] offen offset:0+\imm_off
        .endif
    .endm

    .if chunks_per_wave % read_size
        mbufs_cnt = n_blocks_per_wave * (1 + chunks_per_wave / read_size)
    .else
        mbufs_cnt = n_blocks_per_wave * (chunks_per_wave / read_size)
    .endif
    .macro load_input base
        ibase = \base
        full_loads = chunks_per_wave / read_size
        partial_load_size = chunks_per_wave % read_size
        nb = 0
        .rept n_blocks_per_wave
            imm_off = 0
            .rept full_loads
                vload ibase, read_size, imm_off
                ibase = ibase + read_size
                imm_off = imm_off + 4 * read_size
            .endr
            vload ibase, partial_load_size, imm_off
            ibase = ibase + partial_load_size
            nb = nb + 1
            .if nb == n_blocks_per_wave
                s_add_u32 s[soffset_in], s[soffset_in], 0 + input_feature_map_size - input_stack_size * (active_n_per_wave - active_n_per_gpr)
            .else
                s_add_u32 s[soffset_in], s[soffset_in], 0 + input_stack_size * active_n_per_gpr
            .endif
        .endr
        s_addk_i32 s[loop_cnt], -1
        .if (disable_case_opt || c_per_wave % 4 || input_channels % c_per_wave)
            s_cmpk_le_i32 s[loop_cnt], 0
            s_cmov_b32 s[desc_in+2], 0
        .endif
    .endm

    .macro conv ibase, fbase
        k = 0
        .rept k_mult
            gpr = 0
            .rept in_gprs
                v_mac_f32 v[accums + in_gprs * k + gpr], s[\fbase+k], v[\ibase + gpr]
                gpr = gpr + 1
            .endr
            k = k + 1
        .endr
    .endm

    s_mov_b32 s[loop_cnt], 0 + c_per_wave
    s_cmpk_eq_u32 s[wave_id], 0 + waves_in_group - 1
    s_cmov_b32 s[loop_cnt], 0 + input_channels - c_per_wave * (waves_in_group-1)

    load_input inputA
    load_filters filtersA

    // zeroing accums
    i = 0
    .rept accums_cnt
        v_mov_b32 v[accums + i], 0
        i = i + 1
    .endr

loop_begin:
    load_input inputB
    s_wait mbufs_cnt, 0
    load_filters filtersB
    conv inputA, filtersA

    load_input inputA
    s_wait mbufs_cnt, 0
    load_filters filtersA
    conv inputB, filtersB

loop_end:
    s_cmpk_gt_i32 s[loop_cnt], 1
    s_cbranch_scc1 loop_begin

    load_input inputB
    s_wait mbufs_cnt, 0
    load_filters filtersB
    conv inputA, filtersA
    s_waitcnt 0
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
        //v_lshlrev_b32 v[lds_off], 4, v[stmp]
        //_v_add_co_u32 v[lds_off], vcc, s[stmp], v[lds_off]
        //.ds_write_all
        v_lshlrev_b32 v[lds_off], 2, v[tid]
       _v_add_co_u32 v[lds_off], vcc, s[stmp], v[lds_off]
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
    .GPR_REUSE current_c, current_k
    .GPR_REUSE inputA, current_hw
    s_mul_i32 s[current_k], s[gid_k], 0 + k_mult
    v_and_b32 v[current_hw], 0 + chunk_size - 1, v[tid]
    v_mul_u32_u24 v[current_hw], 0 + chunks_per_wave, v[current_hw]
    s_mul_i32 s[stmp], s[gid_hw], 0 + active_hw_per_wave
   _v_add_co_u32 v[current_hw], vcc, s[stmp], v[current_hw]

    k = 0
    acc = accums
    .rept k_mult
        nb = 0
        .rept n_blocks_per_wave
            s_mov_b32 exec_lo, active_mask_lo
            s_mov_b32 exec_hi, active_mask_hi
            chunk = 0
            .rept chunks_per_wave
                v_cmpx_gt_i32 vcc, 0 + img_hw - chunk, v[current_hw]
                buffer_store_dword v[acc], v[voffset_out], s[desc_out:desc_out+3], s[soffset_out] offen offset:0+4*chunk
                chunk = chunk + 1
                acc = acc + 1
            .endr
            nb = nb + 1
            .if nb == n_blocks_per_wave
                s_add_u32 s[soffset_out], s[soffset_out], 0 + input_feature_map_size - (active_n_per_wave-active_n_per_gpr) * output_stack_size
            .else
                s_add_u32 s[soffset_out], s[soffset_out], 0 + active_n_per_gpr* output_stack_size
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
