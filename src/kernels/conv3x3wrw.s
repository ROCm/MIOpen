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
.include "rocm_version.inc"
.include "inst_wrappers.inc"

.if ROCM_METADATA_VERSION == 4
.hsa_code_object_version 2,1
.hsa_code_object_isa
.endif

.text
.globl miopenGcnAsmConv3x3WrW
.p2align 8
.type miopenGcnAsmConv3x3WrW,@function

.if ROCM_METADATA_VERSION == 4
.amdgpu_hsa_kernel miopenGcnAsmConv3x3WrW
.endif

.include "gpr_alloc.inc"

// initial state (s[0:4] are overlapped with filtersA):
// s[0:1] - kernarg address
// s2 - wg x (none)
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
.set unused_ptr_off, 0x38
.set KERNEL_ARGUMENTS_SIZE, unused_ptr_off + 8

.include "utilities.inc"
.include "conv_common.inc"

default c_per_wave, 4
default k_per_wave, 4
default n_per_group, 1
default pipe_lines_depth, 2
default chunk_size, 16
default reverse_inout, 0
default weights_layout, 0
default reverse_weights, 0

// gfx90a requires 64bit aligned vgpr tuples
// Tuples are used only in buffer_load_dwordx/buffer_store_dwordx instructions
//
// To meet this requirement, the following approach is used ('buffer_load_dwordx4 v[x:y]' as an example):
//    if 'x' 64bit aligned:
//       buffer_load_dwordx4 v[x:y], ...
//    if 'x' not 64bit aligned:
//       buffer_load_dword   v[x], ...
//       buffer_load_dwordx3 v[x+1:y], ...
.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_stepping == 10)
   tuple_alignment = 1
.else
   tuple_alignment = 0
.endif

default elements_in_dword, 1
static_assert(elements_in_dword == 1 || elements_in_dword == 2)
.if elements_in_dword == 2
   static_assert ((batch_size % elements_in_dword) == 0)
   static_assert (.option.machine_version_major >= 9)
.endif
vec_size = elements_in_dword

.if reverse_inout
   static_assert (stride_h == 1 && stride_w == 1)
   swap input_channels, output_channels
   swap in_ptr_off, out_ptr_off
   swap gid_y, gid_z
   reverse_weights = !reverse_weights
   weights_layout = !weights_layout
.endif

static_assert (pad_h == 1 && pad_w == 1)
static_assert (stride_h == 1 || stride_h == 2)
static_assert (stride_w == 1 || stride_w == 2)
.if reverse_inout
   static_assert (stride_h == 1 && stride_w == 1)
.endif
static_assert (wei_h == 3 && wei_w == 3)
static_assert (img_w <= 512)
static_assert (pipe_lines_depth <= img_h)
static_assert (pad_h < wei_h)
static_assert (input_channels % c_per_wave == 0)
static_assert (output_channels % k_per_wave == 0)
static_assert (1 <= n_per_group && n_per_group <= 8)
static_assert (n_per_group <= batch_size)
static_assert (c_per_wave * chunk_size == 64)
static_assert (k_per_wave * chunk_size <= 64)

log2 c_per_wave_log2, c_per_wave
log2 k_per_wave_log2, k_per_wave

.include "conv_sizes.inc"

maxU24 = 1 << 24
static_assert (filter_c_stride < maxU24)
static_assert (filter_k_stride < maxU24)
static_assert (input_feature_map_size < maxU24)
static_assert (output_feature_map_size < maxU24)

w_half_in = img_w % elements_in_dword
w_half_out = out_w % elements_in_dword
img_w_vec = (img_w + elements_in_dword - 1) / elements_in_dword
out_w_vec = (out_w + elements_in_dword - 1) / elements_in_dword

// chunk parameters
log2 chunk_size_log2, chunk_size
chunks_in = (img_w_vec + chunk_size - 1) / chunk_size
.if (chunk_size != 16)
   // force chunks to have enough zeros for padding
   chunks_in = (img_w_vec + chunk_size - pad_w - 1) / (chunk_size - pad_w)
.endif
.if (chunks_in % stride_w) && (chunks_in > 1)
   // force chunks to be aligned with stride
   chunks_in = chunks_in + chunks_in % stride_w
.endif
.if chunks_in > 1
   chunks_out = chunks_in / stride_w
   static_assert ((chunks_out * stride_w) == chunks_in)
.else
   chunks_out = 1
.endif
active_in_lanes = (img_w_vec + chunks_in - 1) / chunks_in // active lanes in chunk
active_out_lanes = (out_w_vec + chunks_out - 1) / chunks_out
static_assert (active_in_lanes == active_out_lanes || chunks_in == 1)
active_lanes = active_in_lanes
full_chunks_in = img_w_vec % chunks_in
full_chunks_out = out_w_vec % chunks_out
.if full_chunks_in == 0
   full_chunks_in = chunks_in
.endif
.if full_chunks_out == 0
   full_chunks_out = chunks_out
.endif
partial_chunks_in = chunks_in - full_chunks_in
partial_chunks_out = chunks_out - full_chunks_out
mbufs_per_line_in = (full_chunks_in + 3) / 4 + (partial_chunks_in + 3) / 4 // memory buffer instructions per line
mbufs_per_line_out = (full_chunks_out + 3) / 4 + (partial_chunks_out + 3) / 4 // memory buffer instructions per line
mbufs_per_line_in = mbufs_per_line_in + w_half_in
mbufs_per_line_out = mbufs_per_line_out + w_half_out
gprs_per_line_in = full_chunks_in + partial_chunks_in
gprs_per_line_out = full_chunks_out + partial_chunks_out
gprs_per_batch_in = gprs_per_line_in * lines_cnt_in
gprs_per_batch_out = gprs_per_line_out * lines_cnt_out


static_assert ((chunk_size == 16) || (chunk_size == 64) || (active_lanes < chunk_size)) // 64 for future expansion
static_assert (chunk_size == 8 || chunk_size == 16)
active_lanes_mask = (1 << active_lanes) - 1
partial_lanes_mask = 1 << (active_lanes - 1)
shift = chunk_size
.rept 5 - chunk_size_log2
   active_lanes_mask = active_lanes_mask + (active_lanes_mask << shift)
   partial_lanes_mask = partial_lanes_mask + (partial_lanes_mask << shift)
   shift = shift * 2
.endr

static_assert(active_lanes_mask <= 0xffffffff)
static_assert(partial_lanes_mask <= 0xffffffff)

input_buffer_size = input_stack_size * batch_size
output_buffer_size = output_stack_size * batch_size

.if (.option.machine_version_major == 8)
   .set max_hw_vctn, 15
.elseif (.option.machine_version_major == 9)
   .set max_hw_vctn, 63
.endif
max_hw_lcnt = 15

.GPR_ALLOC_BEGIN
.if limit_wave_cnt
   .SET_MAX_WAVES_LIMIT limit_wave_cnt
.endif

.SGPR_ALLOC_FROM 5
.SGPR_ALLOC soffset_in
.SGPR_ALLOC soffset_out
.SGPR_ALLOC soffset_wei
.SGPR_ALLOC desc_in, 4 // input buffer descriptor
.SGPR_ALLOC desc_out, 4   // weights buffer descriptor
.SGPR_ALLOC desc_wei, 4   // output buffer descriptor
.if k_group_size_is_power_of_two
    .SGPR_ALLOC stmp
.else
    .SGPR_ALLOC stmp, 4
.endif
.SGPR_ALLOC loop_n_cnt
.SGPR_ALLOC loop_h_cnt
.SGPR_ALLOC wave_id // wave_id in group
//xnack disabled by default
//.SGPR_RESERVE_XNACK
.SGPR_RESERVE_VCC


.VGPR_ALLOC_FROM 0
.VGPR_ALLOC tid
.VGPR_ALLOC voffset_in
.VGPR_ALLOC voffset_out
.VGPR_ALLOC voffset_part_in
.VGPR_ALLOC voffset_part_out
accums_cnt = wei_w * wei_h * c_per_wave * k_per_wave * chunk_size / 64
lines_cnt_in = pipe_lines_depth + wei_h - 1
lines_cnt_out = (pipe_lines_depth + stride_h - 1) / stride_h
.VGPR_ALLOC accums, accums_cnt
.VGPR_ALLOC lines_in, gprs_per_line_in * lines_cnt_in * elements_in_dword
.VGPR_ALLOC lines_out, gprs_per_line_out * lines_cnt_out * elements_in_dword
.VGPR_ALLOC permute_addr
.if elements_in_dword == 2
   .VGPR_ALLOC shfl
.endif

.if k_group_size_is_power_of_two || gprs_per_line_in * lines_cnt_in * elements_in_dword >= 4
    vtmp_udiv = lines_in
.else
    .VGPR_ALLOC vtmp_udiv, 4
.endif

.if k_group_size_is_power_of_two || gprs_per_line_out * lines_cnt_out * elements_in_dword >= 3
    gid = lines_out
    group_id = lines_out + 1
    group_size = lines_out + 2
.else
    .VGPR_ALLOC gid
    .VGPR_ALLOC group_id
    .VGPR_ALLOC group_size
.endif

.LDS_ALLOC_FROM 0
.LDS_ALLOC accums_lds, (n_per_group - 1) * 64 * 4 * accums_cnt

.GPR_ALLOC_END

max_waves_per_CU = (256 / .AUTO_VGPR_COUNT) * 4
static_assert( max_waves_per_CU >= n_per_group )
//.text 0
//.p2align 8
miopenGcnAsmConv3x3WrW:
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


   // fill format and size fields of buffer descriptors
   static_assert ((.option.machine_version_major == 8) || (.option.machine_version_major == 9))
   s_mov_b32 s[desc_in+2], input_buffer_size
   s_mov_b32 s[desc_in+3], 0x00027000
   s_mov_b32 s[desc_wei+2], filters_size
   s_mov_b32 s[desc_wei+3], 0x00027000
   s_mov_b32 s[desc_out+2], output_buffer_size
   s_mov_b32 s[desc_out+3], 0x00027000

   vtmp = accums
   vtmp2 = voffset_part_in
   v_lshrrev_b32 v[vtmp], 6, v[tid]
   v_readfirstlane_b32 s[wave_id], v[vtmp]
   v_and_b32 v[tid], 0x3f, v[tid]

   // calculate input/output offsets
   // example for c_per_wave=4, k_per_wave=2
   // lanes  0-15: c0, k0
   // lanes 16-31: c1, k1
   // lanes 32-47: c2, k0
   // lanes 48-63: c3, k1

   v_lshrrev_b32 v[vtmp], 0 + chunk_size_log2, v[tid] // vtmp = wave part id
   v_mul_u32_u24 v[voffset_in], 0 + input_feature_map_size, v[vtmp]
   v_and_b32 v[voffset_out], 0 + (1 << k_per_wave_log2) - 1, v[vtmp]
   v_mul_u32_u24 v[voffset_out], 0 + output_feature_map_size, v[voffset_out]

   i = 0
   .rept accums_cnt
      v_mov_b32 v[accums+i], 0
      i = i + 1
   .endr

   v_and_b32 v[vtmp2], 0 + (1 << chunk_size_log2) - 1, v[tid] // vtmp = lane in wave part
   v_mul_u32_u24 v[vtmp], 4 * gprs_per_line_in, v[vtmp2]
  _v_add_co_u32 v[voffset_in], vcc, v[voffset_in], v[vtmp]
   .if stride_w == 1 || chunks_in > 1
      v_mul_u32_u24 v[vtmp], 4 * gprs_per_line_out, v[vtmp2]
     _v_add_co_u32 v[voffset_out], vcc, v[voffset_out], v[vtmp]
   .else
      static_assert (stride_w == 2)
      v_lshrrev_b32 v[vtmp2], 1, v[vtmp2]
      v_mul_u32_u24 v[vtmp], 4 * gprs_per_line_out, v[vtmp2]
     _v_add_co_u32 v[voffset_out], vcc, v[voffset_out], v[vtmp]
      s_mov_b32 exec_lo, 0xAAAAAAAA
      s_mov_b32 exec_hi, 0xAAAAAAAA
      v_mov_b32 v[voffset_out], 0x80000000
      s_mov_b32 exec_lo, -1
      s_mov_b32 exec_hi, -1
   .endif

   .GPR_INVALIDATE vtmp
   .GPR_INVALIDATE vtmp2

   // calculate offsets for partial chunks
   v_mov_b32 v[voffset_part_in], v[voffset_in]
   v_mov_b32 v[voffset_part_out], v[voffset_out]
   s_mov_b32 exec_lo, partial_lanes_mask
   s_mov_b32 exec_hi, partial_lanes_mask
   v_mov_b32 v[voffset_part_in], 0x80000000
   v_mov_b32 v[voffset_part_out], v[voffset_part_in]

   s_mov_b32 exec_lo, active_lanes_mask
   s_mov_b32 exec_hi, active_lanes_mask

   s_waitcnt 0

   // calculate buffer offsets
   s_mul_i32 s[stmp], s[wave_id], input_stack_size // image in batch (n)
   s_mul_i32 s[soffset_in], s[gid_y], c_per_wave * input_feature_map_size // input feature map (c)
   s_add_u32 s[soffset_in], s[soffset_in], s[stmp]
   s_mul_i32 s[stmp], s[wave_id], output_stack_size // image in batch (n)
   s_mul_i32 s[soffset_out], s[gid_z], k_per_wave * output_feature_map_size // output feature map (k)
   s_add_u32 s[soffset_out], s[soffset_out], s[stmp]
   s_mul_i32 s[soffset_wei], s[gid_y], c_per_wave * filter_c_stride
   s_mul_i32 s[stmp], s[gid_z], k_per_wave * filter_k_stride
   s_add_u32 s[soffset_wei], s[soffset_wei], s[stmp]

   // calculate group offsets
   static_assert(output_channels % (k_per_wave * group_counts) == 0)
   static_assert(input_channels % (c_per_wave * group_counts) == 0)
   .if reverse_inout
       c_group_size = output_channels / k_per_wave / group_counts
       k_group_size = input_channels / c_per_wave / group_counts
       .if k_group_size_is_power_of_two
           log2 k_group_size_log2, k_group_size
           s_lshr_b32 s[stmp], s[gid_y], 0 + k_group_size_log2 // group_id
       .else
           v_mov_b32 v[gid], s[gid_y]
           v_mov_b32 v[group_size], 0 + k_group_size
           u32_div gid, group_size, group_id, vtmp_udiv, stmp
           v_readfirstlane_b32 s[stmp], v[group_id]
       .endif
       s_mul_i32 s[stmp], s[stmp], c_group_size * k_per_wave * output_feature_map_size // k_group_offset
       s_add_u32 s[soffset_out], s[soffset_out], s[stmp]
   .else
       k_group_size = output_channels / k_per_wave / group_counts
       c_group_size = input_channels / c_per_wave / group_counts
       .if k_group_size_is_power_of_two
           log2 k_group_size_log2, k_group_size
           s_lshr_b32 s[stmp], s[gid_z], 0 + k_group_size_log2 // group_id
       .else
           v_mov_b32 v[gid], s[gid_z]
           v_mov_b32 v[group_size], 0 + k_group_size
           u32_div gid, group_size, group_id, vtmp_udiv, stmp
           v_readfirstlane_b32 s[stmp], v[group_id]
       .endif
       s_mul_i32 s[stmp], s[stmp], c_group_size * c_per_wave * input_feature_map_size // c_group_offset
       s_add_u32 s[soffset_in], s[soffset_in], s[stmp]
   .endif

   .GPR_INVALIDATE gid
   .GPR_INVALIDATE group_size
   .GPR_INVALIDATE group_id
   .GPR_INVALIDATE vtmp_udiv

   s_mov_b32 s[loop_n_cnt], 0

   .macro .single_vload base, v_offset, s_offset, desc, mbufs_inflight, count
      .if ((vals_to_load - \count) >= 0) && vals_to_load > 0
         .if imm_off >= (1 << 12)
            .error "Error: Immediate offset is too large for buffer_load instruction"
         .endif

         .if \count == 1
            buffer_load_dword v[\base+vals_loaded], v[\v_offset], s[\desc:\desc+3], s[\s_offset] offen offset:0+imm_off
         .else
            buffer_load_dwordx\count v[\base+vals_loaded:\base+vals_loaded+\count-1], v[\v_offset], s[\desc:\desc+3], s[\s_offset] offen offset:0+imm_off
         .endif

         \mbufs_inflight = \mbufs_inflight + 1
         vals_to_load = vals_to_load - \count
         vals_loaded = vals_loaded + \count
         imm_off = imm_off + 4 * \count
      .endif
   .endm

   .macro .next_line inout, line
      .if \line >= lines_cnt_\inout - 1
         \line = 0
      .else
         \line = \line + 1
      .endif
   .endm

   .macro exch_vgpr, img_c0, img_c1
      v_mov_b32 v[shfl], v[\img_c0]
      v_mov_b32_sdwa v[\img_c0], v[\img_c1] dst_sel:WORD_1 src0_sel:WORD_0
      v_mov_b32_sdwa v[\img_c1], v[shfl] dst_sel:WORD_0 src0_sel:WORD_1
   .endm

   .macro exch_line_f16 inout
      .if elements_in_dword == 2
         v_cnt = 0
         line_base = lines_\inout + gprs_per_line_\inout * exch_line_\inout
         .rept gprs_per_line_\inout
            exch_vgpr line_base + v_cnt, line_base + v_cnt + gprs_per_batch_\inout
            v_cnt = v_cnt + 1
         .endr
         exch_line_\inout = exch_line_\inout + 1
         .if(exch_line_\inout == lines_cnt_\inout)
            exch_line_\inout = 0
         .endif
      .endif
   .endm

   .macro exch_step_f16
      .if g_line_exch < img_h
          .if g_line_exch < img_h - 1
             exch_line_f16 in
          .endif
          .if !(g_line_exch % stride_h)
             exch_line_f16 out
          .endif
      .endif
      g_line_exch = g_line_exch + 1
   .endm

   .macro .load_line inout, line, mbufs_inflight, go_to_next_line = 1
      so = soffset_\inout
      //reserve soffset
      s_mov_b32 s[stmp], s[so]
      n_cnt = 0
      .rept elements_in_dword
          vo = voffset_\inout
          desc = desc_\inout
          vals_to_load = full_chunks_\inout
          vals_loaded = 0
          imm_off = 0
          line_base = lines_\inout + gprs_per_line_\inout * \line + n_cnt * gprs_per_batch_\inout

          .if tuple_alignment && ((line_base+vals_loaded) % 2)
             .single_vload line_base, vo, so, desc, \mbufs_inflight, 1
          .endif
          .rept (full_chunks_\inout / 4)
             .single_vload line_base, vo, so, desc, \mbufs_inflight, 4
          .endr
          .single_vload line_base, vo, so, desc, \mbufs_inflight, 3
          .single_vload line_base, vo, so, desc, \mbufs_inflight, 2
          .single_vload line_base, vo, so, desc, \mbufs_inflight, 1

          vals_to_load = partial_chunks_\inout
          vo = voffset_part_\inout
          .if tuple_alignment && ((line_base+vals_loaded) % 2)
             .single_vload line_base, vo, so, desc, \mbufs_inflight, 1
          .endif
          .rept (partial_chunks_\inout / 4)
             .single_vload line_base, vo, so, desc, \mbufs_inflight, 4
          .endr
          .single_vload line_base, vo, so, desc, \mbufs_inflight, 3
          .single_vload line_base, vo, so, desc, \mbufs_inflight, 2
          .single_vload line_base, vo, so, desc, \mbufs_inflight, 1

          .if w_half_\inout
             s_mov_b32 exec_lo, partial_lanes_mask
             s_mov_b32 exec_hi, partial_lanes_mask
             buffer_load_ushort v[line_base + vals_loaded - partial_chunks_\inout - 1], v[voffset_\inout], s[desc:desc+3], s[so] offen offset:0 + (vals_loaded - partial_chunks_\inout - 1) * 4
             s_mov_b32 exec_lo, active_lanes_mask
             s_mov_b32 exec_hi, active_lanes_mask
          .endif

         .if so == soffset_in
            s_add_u32 s[so], s[so], 0 + input_stack_size * n_per_group
         .else
            s_add_u32 s[so], s[so], 0 + output_stack_size * n_per_group
         .endif
         n_cnt = n_cnt + 1
      .endr

      //restore soffset
      s_mov_b32 s[so], s[stmp]

      .if \go_to_next_line
         .next_line \inout, \line
         .if so == soffset_in
            s_add_u32 s[so], s[so], input_line_size
         .else
            s_add_u32 s[so], s[so], output_line_size
         .endif
      .endif
   .endm

   .if stride_w == 2
      line_adjust = gprs_per_batch_in
      line_adjust_1 = 1
   .else
      line_adjust = 0
      line_adjust_1 = gprs_per_batch_in
   .endif

   .macro conv_line in_line, out_line, acc_line, acc_batch, sync=0, swizzle=0
      in_base = lines_in + gprs_per_line_in * \in_line
      out_base = lines_out + gprs_per_line_out * \out_line
      acc_base = accums + wei_w * \acc_line + weights_per_filter * \acc_batch
      out_x = 0 // current gpr in line
      .rept gprs_per_line_out
         acc_x = 0
         .if \sync
            s_wait , gprs_per_line_out-out_x-1
         .endif
         .rept wei_w
            in_x = out_x * stride_w - pad_w + acc_x
            .if reverse_weights
               acc_off = wei_w - acc_x - 1
            .else
               acc_off = acc_x
            .endif
            .if in_x < 0
               .if elements_in_dword == 2
                  v_mov_b32 v[shfl], v[in_base + gprs_per_line_in - 1 + gprs_per_batch_in] row_shr:1 bound_ctrl:0
                  v_dot2  acc_base + acc_off, shfl, out_base + out_x
                  v_dot2  acc_base + acc_off, in_base + line_adjust, out_base + out_x + gprs_per_batch_out
               .else
                  v_mac_f32 v[acc_base + acc_off], v[in_base+gprs_per_line_in-1], v[out_base + out_x] row_shr:1 bound_ctrl:0
               .endif
            .elseif in_x >= gprs_per_line_in
                  .if elements_in_dword == 2
                     v_dot2  acc_base + acc_off, in_base + gprs_per_line_in - 1 + gprs_per_batch_in, out_base + out_x
                     .if gprs_per_line_in == 1
                         v_mov_b32 v[shfl], v[in_base + line_adjust] row_shl:1 bound_ctrl:0
                     .else
                         v_mov_b32 v[shfl], v[in_base] row_shl:1 bound_ctrl:0
                     .endif
                     v_dot2  acc_base + acc_off, shfl, out_base + out_x + gprs_per_batch_out
               .else
                  v_mac_f32 v[acc_base + acc_off], v[in_base], v[out_base + out_x] row_shl:1 bound_ctrl:0
               .endif
            .else
               .if elements_in_dword == 2
                  .if acc_x == 1
                     v_dot2  acc_base + acc_off, in_base + in_x, out_base + out_x
                     .if stride_w == 2 && gprs_per_line_in == 1
                        v_mov_b32 v[shfl], v[in_base] row_shl:1 bound_ctrl:0
                        v_dot2  acc_base + acc_off, shfl, out_base + out_x + gprs_per_batch_out
                     .else
                        v_dot2 acc_base + acc_off, in_base + in_x + line_adjust_1, out_base + out_x + gprs_per_batch_out
                     .endif
                  .elseif acc_x == 0
                     v_dot2  acc_base + acc_off, in_base + in_x + gprs_per_batch_in, out_base + out_x
                     v_dot2  acc_base + acc_off, in_base + in_x + 1 + line_adjust, out_base + out_x + gprs_per_batch_out
                  .elseif acc_x == 2
                     v_dot2  acc_base + acc_off, in_base + in_x - 1 + gprs_per_batch_in, out_base + out_x
                     v_dot2  acc_base + acc_off, in_base + in_x + line_adjust, out_base + out_x + gprs_per_batch_out
                  .else
                     static_assert(0)
                  .endif
               .else
                  v_mac_f32 v[acc_base + acc_off], v[in_base + in_x], v[out_base + out_x]
               .endif
            .endif
            acc_x = acc_x + 1
         .endr
         .if \swizzle==32 // swaps each 32 lanes
            // lanes[ 0:31] <-> lanes[32:63]
            ds_bpermute_b32 v[out_base+out_x], v[permute_addr], v[out_base+out_x]
            .if elements_in_dword == 2
               ds_bpermute_b32 v[out_base+out_x+gprs_per_batch_out], v[permute_addr], v[out_base+out_x+gprs_per_batch_out]
            .endif
         .elseif \swizzle==16  // swaps each 16 lanes
            // lanes[ 0:15] <-> lanes[16:31]
            // lanes[32:47] <-> lanes[48:63]
            ds_swizzle_b32 v[out_base+out_x], v[out_base+out_x] offset:0x401F
            .if elements_in_dword == 2
               ds_swizzle_b32 v[out_base+out_x+gprs_per_batch_out], v[out_base+out_x+gprs_per_batch_out] offset:0x401F
            .endif
         .elseif \swizzle==8  // swaps each 8 lanes
            // lanes[0:7] <-> lanes[8:15]
            // ...
            ds_swizzle_b32 v[out_base+out_x], v[out_base+out_x] offset:0x201F
            .if elements_in_dword == 2
               ds_swizzle_b32 v[out_base+out_x+gprs_per_batch_out], v[out_base+out_x+gprs_per_batch_out] offset:0x201F
            .endif
         .elseif \swizzle != 0
            .error "Wrong swizzle parameter"
         .endif
         out_x = out_x + 1
      .endr
   .endm


   // construct address fo ds_bpermute to swap lanes [0:31] <-> lanes [32:63]
   v_xor_b32 v[permute_addr], 0 + (1 << 5), v[tid]
   v_lshlrev_b32 v[permute_addr], 2, v[permute_addr]

loop_n_begin: // loop over batch (n)

   g_line_conv = 0
   g_line_fetch = 0
   g_line_exch = 0
   exch_line_in = 0
   exch_line_out = 0
   line_conv_in = lines_cnt_in - 1
   line_conv_out = 0
   line_fetch_in = 0
   line_fetch_out = 0

   .macro conv_filter in_line0, in_line1, in_line2, out_line, acc_batch, sync=0, swizzle=0
      static_assert (wei_h == 3)
      .if reverse_weights
         accl0 = 2
         accl1 = 1
         accl2 = 0
      .else
         accl0 = 0
         accl1 = 1
         accl2 = 2
      .endif
      .if img_h == 1
         conv_line \in_line1, \out_line, accl1, \acc_batch, \sync, \swizzle
      .elseif g_line_conv == 0
         conv_line \in_line1, \out_line, accl1, \acc_batch, \sync
         conv_line \in_line2, \out_line, accl2, \acc_batch, 0, \swizzle
      .elseif g_line_conv == img_h - 1
         conv_line \in_line0, \out_line, accl0, \acc_batch, \sync
         conv_line \in_line1, \out_line, accl1, \acc_batch, 0, \swizzle
      .else
         conv_line \in_line0, \out_line, accl0, \acc_batch, \sync
         conv_line \in_line1, \out_line, accl1, \acc_batch
         conv_line \in_line2, \out_line, accl2, \acc_batch, 0, \swizzle
      .endif
   .endm

   .macro fetch_step mbufs_inflight
      .if g_line_fetch < img_h
         .if g_line_fetch < img_h - 1
            .load_line in, line_fetch_in, \mbufs_inflight
         .else
            skipped_fetches = skipped_fetches + 1
         .endif
         .if !(g_line_fetch % stride_h)
            .load_line out, line_fetch_out, \mbufs_inflight
         .else
            skipped_fetches = skipped_fetches + 1
         .endif
      .else
         skipped_fetches = skipped_fetches + 2
      .endif
      g_line_fetch = g_line_fetch + 1
   .endm

   .macro conv_step
      .if g_line_conv < img_h
         l2 = line_conv_in
         l0 = l2
         .next_line in, l2
         l1 = l2
         .next_line in, l2
         .next_line in, line_conv_in

         .if !(g_line_conv % stride_h)
            .if k_per_wave == 1
               conv_filter l0, l1, l2, line_conv_out, 0
            .elseif k_per_wave == 2
               conv_filter l0, l1, l2, line_conv_out, 0, 0, chunk_size
               conv_filter l0, l1, l2, line_conv_out, 1, 1, 0
            .elseif k_per_wave == 4
               conv_filter l0, l1, l2, line_conv_out, 0, 0, chunk_size
               conv_filter l0, l1, l2, line_conv_out, 1, 1, chunk_size * 2
               conv_filter l0, l1, l2, line_conv_out, 2, 1, chunk_size
               conv_filter l0, l1, l2, line_conv_out, 3, 1, 0
            .elseif k_per_wave == 8
               conv_filter l0, l1, l2, line_conv_out, 0, 0, chunk_size
               conv_filter l0, l1, l2, line_conv_out, 1, 1, chunk_size * 2
               conv_filter l0, l1, l2, line_conv_out, 2, 1, chunk_size
               conv_filter l0, l1, l2, line_conv_out, 3, 1, chunk_size * 4
               conv_filter l0, l1, l2, line_conv_out, 4, 1, chunk_size
               conv_filter l0, l1, l2, line_conv_out, 5, 1, chunk_size * 2
               conv_filter l0, l1, l2, line_conv_out, 6, 1, chunk_size
               conv_filter l0, l1, l2, line_conv_out, 7, 1, 0
            .else
               static_assert(0)
            .endif
            .next_line out, line_conv_out
         .endif

      .endif

      g_line_conv = g_line_conv + 1
   .endm

   mbufs_cur = 0
   .load_line in, line_fetch_in, mbufs_cur
   s_mov_b32 s[loop_h_cnt], 0

   // pipe prologue
   skipped_fetches = 0
   fetch_step mbufs_cur
   mbufs_1row = mbufs_cur
   .rept pipe_lines_depth-1
      fetch_step mbufs_cur
   .endr
   vmcnt_per_step = (mbufs_per_line_in + mbufs_per_line_out) * elements_in_dword

   mbufs_piped = mbufs_cur - mbufs_1row
   s_wait mbufs_piped
   exch_line_f16 in
   exch_step_f16
   conv_step

   // software pipeline
loop_h_begin:
   period = lines_cnt_in * lines_cnt_out * stride_h
   unroll_factor = 1 * period
   pipelined_steps = img_h - pipe_lines_depth - 1
   .if pipelined_steps < 0
      pipelined_steps = 0
   .endif
   h_cycles = pipelined_steps / unroll_factor
   .if h_cycles > 0
      cur_conv_line = g_line_conv
      cur_fetch_line = g_line_fetch
      .rept unroll_factor
         fetch_step mbufs_cur
         s_wait mbufs_piped
         exch_step_f16
         conv_step
      .endr
      s_addk_i32 s[loop_h_cnt], 1
      s_cmpk_ge_u32 s[loop_h_cnt], 0+h_cycles
      s_cbranch_scc0 loop_h_begin
      g_line_conv = cur_conv_line + unroll_factor * h_cycles
      g_line_fetch = cur_fetch_line + unroll_factor * h_cycles
   .endif
   non_looped_pipelined_steps = pipelined_steps - unroll_factor * h_cycles
loop_h_end:

   .rept non_looped_pipelined_steps
      fetch_step mbufs_cur
      s_wait mbufs_piped
      exch_step_f16
      conv_step
   .endr

   // pipe epilogue
   fetch_step mbufs_cur
   iii = 0
   .rept pipe_lines_depth
      iii = iii + 1
      s_wait mbufs_piped - iii * vmcnt_per_step
      exch_step_f16
      conv_step
   .endr


   s_add_u32 s[soffset_in], s[soffset_in], 0 + input_stack_size * n_per_group * elements_in_dword - input_feature_map_size
   s_add_u32 s[soffset_out], s[soffset_out], 0 + output_stack_size * n_per_group * elements_in_dword - output_feature_map_size
loop_n_end:
   s_addk_i32 s[loop_n_cnt], 0 + elements_in_dword
   s_cmpk_ge_u32 s[loop_n_cnt], 0 + (batch_size + n_per_group - 1) / n_per_group
   s_cbranch_scc0 loop_n_begin

   // reduction across waves in group
   // all waves but last store accums to LDS and dies
   // last wave survives and read LDS
   .GPR_REUSE voffset_in, lds_off
   .if n_per_group > 1
      s_mov_b32 m0, -1
      s_cmpk_eq_u32 s[wave_id], 0 + n_per_group - 1
      s_cbranch_scc1 last_wave

      s_mulk_i32 s[wave_id], 4 * 64 * accums_cnt
      //v_lshlrev_b32 v[lds_off], 4, v[tid]
      //_v_add_co_u32 v[lds_off], vcc, s[wave_id], v[lds_off]
      //.ds_write_all
      v_lshlrev_b32 v[lds_off], 2, v[tid]
     _v_add_co_u32 v[lds_off], vcc, s[wave_id], v[lds_off]
      imm_off = 0
      cur_accum = accums
      .rept accums_cnt
         ds_write_b32 v[lds_off], v[cur_accum], offset:0+imm_off
         cur_accum = cur_accum + 1
         imm_off = imm_off + 4 * 64
      .endr
      s_waitcnt 0
      s_barrier
      s_endpgm
last_wave:
      s_barrier

      v_lshlrev_b32 v[lds_off], 2, v[tid]
      .rept n_per_group-1
         imm_off = 0
         cur_accum = accums
         tmp_accum = lines_in
         .rept accums_cnt
            ds_read_b32 v[tmp_accum], v[lds_off] offset:0+imm_off
            s_waitcnt 0
            v_add_f32 v[cur_accum], v[tmp_accum], v[cur_accum]
            imm_off = imm_off + 4 * 64
            cur_accum = cur_accum + 1
         .endr
        _v_add_co_u32 v[lds_off], vcc, 4 * 64 * accums_cnt, v[lds_off]
      .endr
   .endif


   // reduction inside each chunk
   s_mov_b64 exec, -1
   reduction_steps = chunk_size_log2

   static_assert(accums_cnt > 2)
   .macro reduction max_step
      .irpc step, 123456
         .if \step <= \max_step
            acc = accums
            .rept accums_cnt
               .if \step == 1
                  v_add_f32 v[acc], v[acc], v[acc] quad_perm:[1,0,3,2]
               .elseif \step == 2
                  v_add_f32 v[acc], v[acc], v[acc] quad_perm:[2,3,1,0]
               .elseif \step == 3
                  v_add_f32 v[acc], v[acc], v[acc] row_ror:12
               .elseif \step == 4
                  v_add_f32 v[acc], v[acc], v[acc] row_ror:8
               .elseif \step == 5
                  static_assert (0) //v_add_f32 v[acc], v[acc], v[acc] row_bcast:15
               .elseif \step == 6
                  static_assert (0) //v_add_f32 v[acc], v[acc], v[acc] row_bcast:31
               .endif
               acc = acc + 1
            .endr
         .endif
      .endr
   .endm


   reduction reduction_steps


   //storing result
   static_assert(chunk_size == 16 || chunk_size == 8)
   .if chunk_size == 16
      s_mov_b32 exec_hi, 0x00010001
   .else
      s_mov_b32 exec_hi, 0x01010101
   .endif
   s_mov_b32 exec_lo, exec_hi

   .GPR_REUSE lds_off, c_pattern
   .GPR_REUSE voffset_out, k_pattern
   .GPR_REUSE lines_in, voffset_wei
   .GPR_REUSE lines_out, c_offset
   .GPR_REUSE permute_addr, tid_wei

   .macro store_accums acc
      static_assert (weights_per_filter == 9)
      acc_base = accums + \acc * weights_per_filter
      .if elements_in_dword == 2
         v_cvt_pkrtz_f16_f32 v[acc_base], v[acc_base], v[acc_base+1]
         v_cvt_pkrtz_f16_f32 v[acc_base+1], v[acc_base+2], v[acc_base+3]
         v_cvt_pkrtz_f16_f32 v[acc_base+2], v[acc_base+4], v[acc_base+5]
         v_cvt_pkrtz_f16_f32 v[acc_base+3], v[acc_base+6], v[acc_base+7]
         .if tuple_alignment && (acc_base % 2)
            buffer_store_dword v[acc_base], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0
            buffer_store_dwordx3 v[acc_base+1:acc_base+3], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0+4
         .else
            buffer_store_dwordx4 v[acc_base:acc_base+3], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0
         .endif   
         v_cvt_f16_f32 v[acc_base+8], v[acc_base+8]
         buffer_store_short v[acc_base+8], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0+4*4
      .else
         .if tuple_alignment && (acc_base % 2)
            buffer_store_dword v[acc_base], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0
            buffer_store_dwordx4 v[acc_base+1:acc_base+4], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0+1*4
            buffer_store_dwordx4 v[acc_base+5:acc_base+8], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0+5*4
         .else
            buffer_store_dwordx4 v[acc_base+0:acc_base+3], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0
            buffer_store_dwordx4 v[acc_base+4:acc_base+7], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0+4*4
            buffer_store_dword v[acc_base+8], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0+8*4
         .endif
      .endif
   .endm

   v_mbcnt_hi_u32_b32 v[tid_wei], exec_lo, 0
   v_mbcnt_lo_u32_b32 v[tid_wei], exec_hi, v[tid_wei]

   v_mov_b32 v[c_pattern], v[tid_wei]
   v_and_b32 v[k_pattern], 0 + (1 << k_per_wave_log2) - 1, v[tid_wei]
   v_mul_u32_u24 v[c_offset], 0 + filter_c_stride, v[c_pattern]

   // k_pattern sequence for all accumulators is a gray code sequence.
   // Following macro produces bit mask (\A) that is used to produce next
   // number in a gray sequence

   .macro gray_walker A, B
      .if \B == 1
         v_xor_b32 v[k_pattern], 0 + \A, v[k_pattern]
         v_mul_u32_u24 v[voffset_wei], 0 + filter_k_stride, v[k_pattern]
        _v_add_co_u32 v[voffset_wei], vcc, v[voffset_wei], v[c_offset]
         store_accums curr_accum
         curr_accum = curr_accum + 1
      .else
         gray_walker \A, \B/2
         gray_walker \B/2, \B/2
      .endif
   .endm

   curr_accum = 0
   gray_walker 0, k_per_wave

s_endpgm

.Lfunc_end0:
   .size miopenGcnAsmConv3x3WrW, .Lfunc_end0 - miopenGcnAsmConv3x3WrW

workgroup_size_x = n_per_group * 64

.if ROCM_METADATA_VERSION == 5
.rodata
.p2align 6
.if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_stepping == 10)
.amdhsa_kernel miopenGcnAsmConv3x3WrW
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_system_sgpr_workgroup_id_y 1
        .amdhsa_system_sgpr_workgroup_id_z 1
        .amdhsa_system_vgpr_workitem_id 1
        .amdhsa_next_free_sgpr __amdhsa_next_free_sgpr
        .amdhsa_accum_offset ((.AUTO_VGPR_COUNT + 3) / 4) * 4
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
.else
.amdhsa_kernel miopenGcnAsmConv3x3WrW
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
  - .name: miopenGcnAsmConv3x3WrW
    .symbol: miopenGcnAsmConv3x3WrW.kd
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
        - { Name: miopenGcnAsmConv3x3WrW, SymbolName: 'miopenGcnAsmConv3x3WrW@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
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
