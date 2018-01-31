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
.globl gcnAsmConv1x1WrW
.p2align 8
.type gcnAsmConv1x1WrW,@function
.amdgpu_hsa_kernel gcnAsmConv1x1WrW

.include "gpr_alloc.inc"
.include "common.inc"
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
// dwords 14:15 - debug buffer pointer
.set in_ptr_off, 0x20
.set wei_ptr_off, 0x28
.set out_ptr_off, 0x30
.set dbg_ptr_off, 0x38

.ifnotdef do_not_use_default_perf_params
    default n_per_gpr, 4 // 1..4, 2^n
    default c_per_gpr, 4 // 1..16, 2^n
    default c_mult, 2 // 1..16, 2^n
    default k_per_gpr, 4 // 1..16, 2^n
    default k_mult, 2 // 1..16, 2^n
    default read_size, 1 // 1..4
    default pipe_depth, 1 // 1..8(16)
    default chunk_size, 4 // 1..16, 2^n
    default hw_per_gpr, 1 // 1..4, 2^n
.endif
default limit_wave_cnt, 0
default reverse_inout, 0 // 0 or 1
default c_per_wg, 1 // 1..8
default k_per_wg, 1 // 1..8
default n_per_wg, 1 // 1..8
//default hw_per_group, 1 // 1..8
default weights_layout, 0 // 0 or 1

.if reverse_inout
    swap input_channels, output_channels
    swap in_ptr_off, out_ptr_off
    swap gid_y, gid_z
    weights_layout = !weights_layout
.endif

.include "conv_sizes.inc"

group_size = c_per_wg * k_per_wg * n_per_wg
static_assert (c_per_wg == 1 || k_per_wg == 1)
static_assert (pad_h == 0 && pad_w == 0)
static_assert (stride_h == 1) // || stride_h == 2)
static_assert (stride_w == 1) // || stride_w == 2)
static_assert (wei_h == 1 && wei_w == 1)
//static_assert (pipe_depth <= img_h)
.if (c_mult > 1) || (k_mult > 1) // todo: remove restriction
    static_assert (input_channels % (c_per_gpr * c_mult) == 0)
    static_assert (output_channels % (k_per_gpr * k_mult) == 0)
.endif
static_assert (1 <= group_size && group_size <= 8)
static_assert (n_per_gpr * hw_per_gpr == 4)
static_assert (c_per_gpr * n_per_gpr * hw_per_gpr * chunk_size == wave_size)
static_assert (k_per_gpr * n_per_gpr * hw_per_gpr * chunk_size <= wave_size)

static_assert ((.option.machine_version_major == 8) || (.option.machine_version_major == 9))
static_assert (filter_c_stride < maxU24)
static_assert (filter_k_stride < maxU24)
static_assert (input_feature_map_size < maxU24)
static_assert (output_feature_map_size < maxU24)

log2 c_per_gpr_log2, c_per_gpr
log2 k_per_gpr_log2, k_per_gpr
log2 n_per_gpr_log2, n_per_gpr
log2 hw_per_gpr_log2, hw_per_gpr

// chunk parameters
metachunk_size = chunk_size * hw_per_gpr // hw pieces are not contiguous in vgpr
log2 chunk_size_log2, chunk_size
log2 meatchunk_size_log2, metachunk_size
out_wh = out_w * out_h
chunks = (out_wh + metachunk_size - 1) / metachunk_size
full_reads = chunks / read_size
.if full_reads * read_size * metachunk_size > out_wh // last chunk could be incomplete
    full_reads = full_reads - 1
.endif
full_chunks = full_reads * read_size
partial_chunks = chunks - full_chunks
points_in_full_chunks = full_chunks * metachunk_size
points_in_part_chunks = out_wh - points_in_full_chunks
j=0
.if full_reads > 0 && partial_chunks > 1
    // try to use single read for partial chunks
    max_points_in_part_chunks = partial_chunks * metachunk_size
    steps = (max_points_in_part_chunks - points_in_part_chunks) / full_chunks
    static_assert (steps < metachunk_size)
    i = 0
    .rept steps
        i = i + 1
        .if (points_in_part_chunks + i * full_chunks) % partial_chunks == 0
            j = i
        .endif
    .endr
.endif
active_lanes_in_full_chunks = metachunk_size - j
points_in_full_chunks = full_chunks * active_lanes_in_full_chunks
points_in_part_chunks = out_wh - points_in_full_chunks
.if partial_chunks
    active_lanes_in_part1_chunks = (points_in_part_chunks + partial_chunks - 1) / partial_chunks
    active_lanes_in_part2_chunks = active_lanes_in_part1_chunks - 1
    .if points_in_part_chunks % partial_chunks == 0
        part1_chunks = partial_chunks
    .else
        part1_chunks = points_in_part_chunks % partial_chunks
    .endif
    part2_chunks = partial_chunks - part1_chunks
.else
    part1_chunks = 0
    part2_chunks = 0
    active_lanes_in_part1_chunks = 0
    active_lanes_in_part2_chunks = 0
.endif
static_assert (part1_chunks * active_lanes_in_part1_chunks + part2_chunks * active_lanes_in_part2_chunks == points_in_part_chunks)
part2_offset = part1_chunks * 4 * active_lanes_in_part1_chunks
hw_step = 4 * active_lanes_in_full_chunks * read_size
//static_assert (pipe_depth <= (full_reads + (?)partial_rdd))
// todo: try if part1_chunks/part1_chunks optimization


input_buffer_size = input_stack_size * batch_size
output_buffer_size = output_stack_size * batch_size

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
.SGPR_ALLOC loop_n_cnt
.SGPR_ALLOC loop_hw_cnt
.SGPR_ALLOC c_base
.SGPR_ALLOC k_base
.SGPR_ALLOC n_base
.SGPR_ALLOC stmp
.SGPR_ALLOC wave_id // wave_id in group
.SGPR_RESERVE_XNACK



.VGPR_ALLOC_FROM 0
.VGPR_ALLOC tid
.VGPR_ALLOC voffset_in
.VGPR_ALLOC voffset_out
.VGPR_ALLOC voffset_part1_in
.VGPR_ALLOC voffset_part1_out
.VGPR_ALLOC voffset_part2_in
.VGPR_ALLOC voffset_part2_out
accums_cnt = wei_w * wei_h * k_per_gpr * c_per_wg * k_per_wg * c_mult * k_mult
.VGPR_ALLOC accums, accums_cnt
.VGPR_ALLOC lines_in, pipe_depth * read_size * c_mult
.VGPR_ALLOC lines_out, pipe_depth * read_size * k_mult
.VGPR_ALLOC permute_addr
.VGPR_ALLOC n_id


.LDS_ALLOC_FROM 0

.GPR_ALLOC_END

max_waves_per_CU = (256 / .AUTO_VGPR_COUNT) * 4
static_assert( max_waves_per_CU >= group_size )
//.text 0
//.p2align 8
gcnAsmConv1x1WrW:

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

    .include "macro1x1wrw.inc"
    s_load_dwordx2 s[desc_in:desc_in+1], s[kernarg:kernarg+1], 0x0 + in_ptr_off
    s_load_dwordx2 s[desc_wei:desc_wei+1], s[kernarg:kernarg+1], 0x0 + wei_ptr_off
    s_load_dwordx2 s[desc_out:desc_out+1], s[kernarg:kernarg+1], 0x0 + out_ptr_off
    
    vtmp = accums
    v_lshrrev_b32 v[vtmp], 6, v[tid]
    v_readfirstlane_b32 s[wave_id], v[vtmp]
    v_and_b32 v[tid], 0x3f, v[tid]
    
    // calculate input/output offsets
    // example for c_per_gpr=4, k_per_gpr=2, n_per_gpr=1
    // lanes  0-15: c0, k0, n0
    // lanes 16-31: c1, k0, n0
    // lanes 32-47: c2, k1, n0
    // lanes 48-63: c3, k1, n0
    c_id = lines_in
    k_id = lines_out
    v_lshrrev_b32 v[n_id], 0 + wave_size_log2 - n_per_gpr_log2, v[tid]
    v_bfe_u32 v[c_id], v[tid], 0 + chunk_size_log2, 0 + c_per_gpr_log2
    v_bfe_u32 v[k_id], v[tid], 0 + chunk_size_log2 + c_per_gpr_log2 - k_per_gpr_log2, 0 + k_per_gpr_log2
    
    s_mov_b32 s[stmp], 0 + input_feature_map_size
    v_mul_lo_u32 v[voffset_in], s[stmp], v[c_id]
    s_mov_b32 s[stmp], 0 + input_stack_size
    v_mul_lo_u32 v[vtmp], s[stmp], v[n_id]
   _v_add_nc_u32 v[voffset_in], v[voffset_in], v[vtmp] // c_off + n_off
    
    s_mov_b32 s[stmp], 0 + output_feature_map_size
    v_mul_lo_u32 v[voffset_out], s[stmp], v[k_id]
    s_mov_b32 s[stmp], 0 + output_stack_size
    v_mul_lo_u32 v[vtmp], s[stmp], v[n_id]
   _v_add_nc_u32 v[voffset_out], v[voffset_out], v[vtmp] // k_off + n_off
    
    vtmp2 = permute_addr
    v_bfe_u32 v[vtmp], v[tid], 0 + chunk_size_log2 + c_per_gpr_log2, 0 + hw_per_gpr_log2 // hw peice id
    v_lshlrev_b32 v[vtmp], 0 + chunk_size_log2, v[vtmp]
    v_and_b32 v[vtmp2], 0 + chunk_size - 1, v[tid] // lane in chunk
   _v_add_nc_u32 v[vtmp2], v[vtmp2], v[vtmp] // lane in metachunk
    
    v_mul_u32_u24 v[vtmp], 4 * part1_chunks, v[vtmp2]
   _v_add_nc_u32 v[voffset_part1_in],  v[voffset_in], v[vtmp] // +hw_off
   _v_add_nc_u32 v[voffset_part1_out], v[voffset_out], v[vtmp] // +hw_off
    
    v_mul_u32_u24 v[vtmp], 4 * part2_chunks, v[vtmp2]
   _v_add_nc_u32 v[voffset_part2_in],  v[voffset_in], v[vtmp] // +hw_off
   _v_add_nc_u32 v[voffset_part2_out], v[voffset_out], v[vtmp] // +hw_off
    
    v_mul_u32_u24 v[vtmp], 4 * read_size, v[vtmp2]
   _v_add_nc_u32 v[voffset_in], v[voffset_in], v[vtmp] // +hw_off
   _v_add_nc_u32 v[voffset_out], v[voffset_out], v[vtmp] // +hw_off
    
    
    
    // calculate buffer scalar offsets
    s_mul_i32 s[c_base], 0 + c_per_gpr * c_per_wg * c_mult, s[gid_y]
    s_mul_i32 s[k_base], 0 + k_per_gpr * k_per_wg * k_mult, s[gid_z]
    s_mul_i32 s[n_base], 0 + n_per_gpr * n_per_wg, s[wave_id] // todo: adjust c_base, k_base, n_base according to c_per_wg, k_per_wg, n_per_wg
    // todo: wrong(?) n_base
    
    s_mul_i32 s[soffset_in], 0 + input_stack_size, s[n_base]
    s_mul_i32 s[stmp], 0 + input_feature_map_size, s[c_base]
    s_add_u32 s[soffset_in], s[soffset_in], s[stmp]
    
    s_mul_i32 s[soffset_out], 0 + output_stack_size, s[n_base]
    s_mul_i32 s[stmp], 0 + output_feature_map_size, s[k_base]
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
    
    v_cmp_gt_u32 vcc, 0 + active_lanes_in_part1_chunks, v[vtmp2]
    v_cndmask_b32_e32 v[voffset_part1_in],  v[vtmp], v[voffset_part1_in], vcc
    v_cndmask_b32_e32 v[voffset_part1_out], v[vtmp], v[voffset_part1_out], vcc
    
    v_cmp_gt_u32 vcc, 0 + active_lanes_in_part2_chunks, v[vtmp2]
    v_cndmask_b32_e32 v[voffset_part2_in],  v[vtmp], v[voffset_part2_in], vcc
    v_cndmask_b32_e32 v[voffset_part2_out], v[vtmp], v[voffset_part2_out], vcc
    
    .GPR_INVALIDATE c_id
    .GPR_INVALIDATE k_id
    .GPR_INVALIDATE vtmp
    .GPR_INVALIDATE vtmp2
    
    // fill format and size fields of buffer descriptors
    s_mov_b32 s[desc_in+2], input_buffer_size
    s_mov_b32 s[desc_in+3], 0x00027000
    s_mov_b32 s[desc_wei+2], filters_size
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
    .adjust_sptr_gpr desc_in, s[soffset_in]
    .adjust_sptr_gpr desc_out, s[soffset_out]
    .adjust_sptr_gpr desc_wei, s[soffset_wei]
    
    s_mov_b32 s[loop_n_cnt], 0
    
    .macro m_load inout, total_adj, dwords1, voff1, dwords2=0, voff2=0
        .if lines_\inout == lines_in
            mult = c_mult
            dst = lines_in
            desc = desc_in
            adj_size = c_per_gpr * input_feature_map_size
        .else
            mult = k_mult
            dst = lines_out
            desc = desc_out
            adj_size = k_per_gpr * output_feature_map_size
        .endif
        .rept mult-1
            m_buffer_load_dwordx \dwords1, dst,            \voff1, desc
            m_buffer_load_dwordx \dwords2, dst + \dwords1, \voff2, desc, part2_offset
            dst = dst + \dwords1 + \dwords2
            .adjust_sptr desc, 0 + adj_size
            \total_adj = \total_adj + adj_size
        .endr
        m_buffer_load_dwordx \dwords1, dst,            \voff1, desc
        m_buffer_load_dwordx \dwords2, dst + \dwords1, \voff2, desc, part2_offset
    .endm
    
loop_n_begin: // loop over batch (n)
    s_mov_b32 s[loop_hw_cnt], 0

    c_off = 0
    k_off = 0
    .if full_reads
        loop_hw_begin:
            m_load in,  c_off, read_size, voffset_in
            m_load out, k_off, read_size, voffset_out
            .adjust_sptr desc_in,  0 + active_lanes_in_full_chunks * read_size * 4 - c_off // todo: remove redundant adjust_sptr
            .adjust_sptr desc_out, 0 + active_lanes_in_full_chunks * read_size * 4 - k_off
            s_waitcnt 0
            
            m_conv_accums read_size
        
        loop_hw_end:
            s_addk_i32 s[loop_hw_cnt], 1
            s_cmpk_ge_u32 s[loop_hw_cnt], 0+full_reads
            s_cbranch_scc0 loop_hw_begin
    .endif
    
    c_off = full_chunks * 4 * active_lanes_in_full_chunks
    k_off = full_chunks * 4 * active_lanes_in_full_chunks
    
    .if partial_chunks
        m_load in,  c_off, part1_chunks, voffset_part1_in,  part2_chunks, voffset_part2_in
        m_load out, k_off, part1_chunks, voffset_part1_out, part2_chunks, voffset_part2_out
        s_waitcnt 0
        
        m_conv_accums partial_chunks
    .endif
    
    .adjust_sptr desc_in, 0 + input_stack_size * n_per_wg * n_per_gpr - c_off
    .adjust_sptr desc_out, 0 + output_stack_size * n_per_wg * n_per_gpr - k_off
    s_sub_u32 s[desc_in+2], s[desc_in+2], 0 + input_stack_size * n_per_wg * n_per_gpr //todo: adjust for n_per_wg
    s_sub_u32 s[desc_out+2], s[desc_out+2], 0 + output_stack_size * n_per_wg * n_per_gpr
loop_n_end:
   _v_add_nc_u32 v[n_id], 0 + n_per_gpr * n_per_wg, v[n_id]
    s_addk_i32 s[loop_n_cnt], 1
    s_cmpk_ge_u32 s[loop_n_cnt], 0 + (batch_size + n_per_wg * n_per_gpr - 1) / (n_per_wg * n_per_gpr)
    s_cbranch_scc0 loop_n_begin

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
    
    // STORE
    // prepare output addresses
    .GPR_REUSE voffset_in, voffset_wei
    .GPR_REUSE lines_in, c_off
    .GPR_REUSE lines_out, k_off
    invalid_addr = permute_addr
    v_mov_b32 v[invalid_addr], 0x7FFFFFFF
    //v_mov_b32 v[invalid_addr], 0x40000000
    
   _v_add_nc_u32 v[vtmp], s[c_base], v[tid]
    v_mul_u32_u24 v[c_off], 0 + filter_c_stride, v[tid]
    v_cmp_gt_u32 vcc, 0 + input_channels, v[vtmp]
    v_cndmask_b32_e32 v[c_off], v[invalid_addr], v[c_off], vcc
    
    v_bfe_u32 v[k_off], v[tid], 0 + c_per_gpr_log2 - k_per_gpr_log2, 0 + k_per_gpr_log2
   _v_add_nc_u32 v[vtmp], s[k_base], v[k_off]
    v_mul_u32_u24 v[k_off], 0 + filter_k_stride, v[k_off]
    v_cmp_gt_u32 vcc, 0 + output_channels, v[vtmp]
    v_cndmask_b32_e32 v[k_off], v[invalid_addr], v[k_off], vcc
    v_cmp_gt_u32 vcc, 0 + c_per_gpr, v[tid]
    //v_cndmask_b32_e32 v[k_off], v[invalid_addr], v[k_off], vcc
    v_cndmask_b32_e32 v[c_off], v[invalid_addr], v[c_off], vcc
    .GPR_INVALIDATE invalid_addr
    
    .macro vadd_u32_ror dst, src0, src1, ror
        .long 0x320000FA + ((\src1) << 9) + ((\dst) << 17)
        .long 0xFF012100 + \src0 + ((\ror - 1) << 8)
    .endm
    
    // store accums
    k = 0
    .rept k_per_gpr
        b = (k * c_per_gpr / k_per_gpr) % 16 // lanes to ror
        .if b == 0
           _v_add_nc_u32 v[voffset_wei], v[k_off], v[c_off]
        .else
            .if (.option.machine_version_major == 8) // workaround for asm
                vadd_u32_ror voffset_wei, k_off, c_off, b
            .else
               _v_add_nc_u32 v[voffset_wei], v[k_off], v[c_off] row_ror:b
            .endif
        .endif
        cx = 0
        .rept c_mult
            kx = 0
            .rept k_mult
                acc = accums + k_per_gpr * (cx * k_mult + kx) + k
                s_mov_b32 s[stmp], 0 + cx * c_per_gpr * filter_c_stride + kx * k_per_gpr * filter_k_stride
                buffer_store_dword v[acc], v[voffset_wei], s[desc_wei:desc_wei+3], s[stmp] offen
                kx = kx + 1
            .endr
            cx = cx + 1
        .endr
        k = k + 1
    .endr
    
s_endpgm

.Lfunc_end0:
    .size gcnAsmConv1x1WrW, .Lfunc_end0 - gcnAsmConv1x1WrW

.ifndef ROCM_METADATA_VERSION
.error "ROCM_METADATA_VERSION must be defined"
.end
.endif

.ifdef n_per_group
.error "n_per_group must NOT be defined"
.end
.endif
.set n_per_group, 1

.macro metadata wg_x
  .if ROCM_METADATA_VERSION == 3
    .amdgpu_code_object_metadata
    { Version: [ 3, 0 ],
        Kernels:
        - { Name: gcnAsmConv1x1WrW, Language: OpenCL C, LanguageVersion: [ 1, 2 ],
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
            - { Name: dw      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default }
            - { Name: dy      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true }
            - { Name: ret_addr, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: I32, TypeName: 'int*'  , AddrSpaceQual: Global, AccQual: Default }
          }
    }
    .end_amdgpu_code_object_metadata
  .endif
  .if ROCM_METADATA_VERSION == 4
    .amd_amdgpu_hsa_metadata
    { Version: [ 1, 0 ],
        Kernels:
        - { Name: gcnAsmConv1x1WrW, SymbolName: 'gcnAsmConv1x1WrW@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
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
            - { Name: dw      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default }
            - { Name: dy      , Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', AddrSpaceQual: Global, AccQual: Default, IsConst: true }
            - { Name: ret_addr, Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: I32, TypeName: 'int*'  , AddrSpaceQual: Global, AccQual: Default }
          }
    }
    .end_amd_amdgpu_hsa_metadata
  .endif
.endm

.if n_per_group == 8
    metadata 512
.elseif n_per_group == 7
    metadata 448
.elseif n_per_group == 6
    metadata 384
.elseif n_per_group == 5
    metadata 320
.elseif n_per_group == 4
    metadata 256
.elseif n_per_group == 3
    metadata 192
.elseif n_per_group == 2
    metadata 128
.else
    metadata 64
.endif
