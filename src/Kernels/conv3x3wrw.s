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
.globl gcnAsmConv3x3WrW
.p2align 8
.type gcnAsmConv3x3WrW,@function
.amdgpu_hsa_kernel gcnAsmConv3x3WrW

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
.set dbg_ptr_off, 0x38


.macro default symbol, value
   .ifnotdef \symbol
      \symbol = \value
   .endif
.endm

.macro static_assert fufufu
   .if !\fufufu
      .error "\fufufu is false"
   .endif
.endm

.macro swap a, b
   __tmp = \a
   \a = \b
   \b = __tmp
.endm

default c_per_wave, 4
default k_per_wave, 4
default n_per_group, 1
default pipe_lines_depth, 2
default chunk_size, 16
default reverse_inout, 0
default weights_layout, 0
default reverse_weights, 0

.if reverse_inout
   static_assert (stride_h == 1 && stride_w == 1)
   swap input_channels, output_channels
   swap in_ptr_off, out_ptr_off
   swap gid_y, gid_z
   reverse_weights = !reverse_weights
   weights_layout = !weights_layout
.endif

static_assert (pad_h == 1 && pad_w == 1)
static_assert (stride_h == 1 && stride_w == 1)
static_assert (wei_h == 3 && wei_w == 3)
static_assert (img_w <= 256)
static_assert (pipe_lines_depth <= img_h)
static_assert (pad_h < wei_h)
static_assert (input_channels % c_per_wave == 0)
static_assert (output_channels % k_per_wave == 0)
static_assert (1 <= n_per_group && n_per_group <= 8)
static_assert (n_per_group <= batch_size)


.macro log2 lg2, num, max_bits=8
   \lg2 = 0
   lg_i = \num
   .rept \max_bits
      lg_i = lg_i / 2
      .if lg_i > 0
         \lg2 = \lg2 + 1
      .endif
   .endr
.endm

log2 c_per_wave_log2, c_per_wave
log2 k_per_wave_log2, k_per_wave


// weights
.set weights_per_filter, wei_w * wei_h
.if weights_layout == 0 // KCHW
   .set filter_c_stride, 4 * weights_per_filter
   .set filter_k_stride, 4 * weights_per_filter * input_channels
.else // CKHW
   .set filter_c_stride, 4 * weights_per_filter * output_channels
   .set filter_k_stride, 4 * weights_per_filter
.endif
.set filters_size, 4 * weights_per_filter * input_channels * output_channels


// input/output
input_line_size = 4 * img_w
input_feature_map_size = input_line_size * img_h
input_stack_size = input_feature_map_size * input_channels
output_line_size = 4 * img_w
output_feature_map_size = output_line_size * img_h
output_stack_size = output_feature_map_size * output_channels

maxU24 = 1 << 24
static_assert (filter_c_stride < maxU24)
static_assert (filter_k_stride < maxU24)
static_assert (input_feature_map_size < maxU24)
static_assert (output_feature_map_size < maxU24)

// chunk parameters
log2 chunk_size_log2, chunk_size
chunks = (img_w + chunk_size - 1) / chunk_size
.if (chunk_size != 16)
	// force chunks to have enough zeros for padding
	chunks = (img_w + chunk_size - pad_w - 1) / (chunk_size - pad_w)
.endif
active_lanes = (img_w + chunks - 1) / chunks // active lanes in chunk
.if img_w % chunks == 0
   full_chunks = chunks
   partial_chunks = 0
.else
   full_chunks =  img_w % chunks
   partial_chunks = chunks - full_chunks
.endif
mbufs_per_line = (full_chunks + 3) / 4 + (partial_chunks + 3) / 4 // memory buffer instructions per line
gprs_per_line = full_chunks + partial_chunks


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

input_buffer_size = input_stack_size * batch_size
output_buffer_size = output_stack_size * batch_size

.set max_hw_wctn, 15
.macro s_wait vmcnt=max_hw_wctn, lgkmcnt=max_hw_wctn
   vm_cnt = \vmcnt
   lgkm_cnt = \lgkmcnt
   .if vm_cnt > max_hw_wctn
      vm_cnt = max_hw_wctn
   .elseif vm_cnt < 0
      vm_cnt = 0
   .endif
   .if lgkm_cnt > max_hw_wctn
      lgkm_cnt = max_hw_wctn
   .elseif lgkm_cnt < 0
      lgkm_cnt = 0
   .endif
   s_waitcnt vmcnt(0 + vm_cnt) & lgkmcnt(0 + lgkm_cnt)
.endm


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
.SGPR_ALLOC loop_n_cnt
.SGPR_ALLOC loop_h_cnt
.SGPR_ALLOC wave_id // wave_id in group
.SGPR_ALLOC stmp
.SGPR_RESERVE_XNACK



.VGPR_ALLOC_FROM 0
.VGPR_ALLOC tid
.VGPR_ALLOC voffset_in
.VGPR_ALLOC voffset_out
.VGPR_ALLOC voffset_part_in
.VGPR_ALLOC voffset_part_out
accums_cnt = wei_w * wei_h * c_per_wave * k_per_wave * chunk_size / 64
lines_cnt_in = pipe_lines_depth + wei_h - 1
lines_cnt_out = pipe_lines_depth
.VGPR_ALLOC accums, accums_cnt
.VGPR_ALLOC lines_in, gprs_per_line * lines_cnt_in
.VGPR_ALLOC lines_out, gprs_per_line * lines_cnt_out
.VGPR_ALLOC permute_addr

.LDS_ALLOC_FROM 0
.LDS_ALLOC accums_lds, (n_per_group - 1) * 64 * 4 * accums_cnt

.GPR_ALLOC_END

max_waves_per_CU = (256 / .AUTO_VGPR_COUNT) * 4
static_assert( max_waves_per_CU >= n_per_group )
//.text 0
//.p2align 8
gcnAsmConv3x3WrW:

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


   // fill format and size fields of buffer descriptors
   static_assert ((.option.machine_version_major == 8) || (.option.machine_version_major == 9))
   s_mov_b32 s[desc_in+2], input_buffer_size
   s_mov_b32 s[desc_in+3], 0x00804fac
   s_mov_b32 s[desc_wei+2], filters_size
   s_mov_b32 s[desc_wei+3], 0x00804fac
   s_mov_b32 s[desc_out+2], output_buffer_size
   s_mov_b32 s[desc_out+3], 0x00804fac

   vtmp = accums
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

   v_and_b32 v[vtmp], 0 + (1 << chunk_size_log2) - 1, v[tid] // vtmp = lane in wave part
   v_mul_u32_u24 v[vtmp], 4 * gprs_per_line, v[vtmp]
   v_add_u32 v[voffset_in], vcc, v[voffset_in], v[vtmp]
   v_add_u32 v[voffset_out], vcc, v[voffset_out], v[vtmp]

   .GPR_INVALIDATE vtmp

   i = 0
   .rept accums_cnt
      v_mov_b32 v[accums+i], 0
      i = i + 1
   .endr

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

   s_mov_b32 s[loop_n_cnt], 0

   .macro .single_vload base, v_offset, s_offset, desc, count
      .if ((vals_to_load - \count) >= 0) && vals_to_load > 0
         .if imm_off >= (1 << 12)
            .error "Error: Immediate offset is too large for buffer_load instruction"
         .endif

         .if \count == 1
            buffer_load_dword v[\base+vals_loaded], v[\v_offset], s[\desc:\desc+3], s[\s_offset] offen offset:0+imm_off
         .else
            buffer_load_dwordx\count v[\base+vals_loaded:\base+vals_loaded+\count-1], v[\v_offset], s[\desc:\desc+3], s[\s_offset] offen offset:0+imm_off
         .endif

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

   .macro .load_line inout, line, go_to_next_line = 1
      vals_to_load = full_chunks
      vals_loaded = 0
      imm_off = 0
      line_base = lines_\inout + gprs_per_line * \line
      vo = voffset_\inout
      so = soffset_\inout
      desc = desc_\inout
      .rept (full_chunks / 4)
         .single_vload line_base, vo, so, desc, 4
      .endr
      .single_vload line_base, vo, so, desc, 3
      .single_vload line_base, vo, so, desc, 2
      .single_vload line_base, vo, so, desc, 1

      vals_to_load = partial_chunks
      vo = voffset_part_\inout
      .rept (full_chunks / 4)
         .single_vload line_base, vo, so, desc, 4
      .endr
      .single_vload line_base, vo, so, desc, 3
      .single_vload line_base, vo, so, desc, 2
      .single_vload line_base, vo, so, desc, 1

      .if \go_to_next_line
         .next_line \inout, \line
         .if so == soffset_in
            s_add_u32 s[so], s[so], input_line_size
         .else
            s_add_u32 s[so], s[so], output_line_size
         .endif
      .endif
   .endm

   .macro conv_line in_line, out_line, acc_line, acc_batch, sync=0, swizzle=0
      in_base = lines_in + gprs_per_line * \in_line
      out_base = lines_out + gprs_per_line * \out_line
      acc_base = accums + wei_w * \acc_line + weights_per_filter * \acc_batch
      out_x = 0 // current gpr in line
      .rept gprs_per_line
         acc_x = 0
         .if \sync
            s_wait , gprs_per_line-out_x-1
         .endif
         .rept wei_w
            in_x = out_x - pad_w + acc_x
            .if reverse_weights
               acc_off = wei_w - acc_x - 1
            .else
               acc_off = acc_x
            .endif
            .if in_x < 0
               v_mac_f32 v[acc_base + acc_off], v[in_base+gprs_per_line-1], v[out_base + out_x] row_shr:1 bound_ctrl:0
            .elseif in_x >= gprs_per_line
               v_mac_f32 v[acc_base + acc_off], v[in_base], v[out_base + out_x] row_shl:1 bound_ctrl:0
            .else
               v_mac_f32 v[acc_base + acc_off], v[in_base + in_x], v[out_base + out_x]
            .endif
            acc_x = acc_x + 1
         .endr
         .if \swizzle==32 // swaps each 32 lanes
            // lanes[ 0:31] <-> lanes[32:63]
            ds_bpermute_b32 v[out_base+out_x], v[permute_addr], v[out_base+out_x]
         .elseif \swizzle==16  // swaps each 16 lanes
            // lanes[ 0:15] <-> lanes[16:31]
            // lanes[32:47] <-> lanes[48:63]
            ds_swizzle_b32 v[out_base+out_x], v[out_base+out_x] offset:0x401F
         .elseif \swizzle==8  // swaps each 8 lanes
            // lanes[0:7] <-> lanes[8:15]
            // ...
            ds_swizzle_b32 v[out_base+out_x], v[out_base+out_x] offset:0x201F
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

   .macro fetch_step
      .if g_line_fetch < img_h
         .if g_line_fetch < img_h - 1
            .load_line in, line_fetch_in
         .else
            skipped_fetches = skipped_fetches + 1
         .endif
         .load_line out, line_fetch_out
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

         .next_line in, line_conv_in
         .next_line out, line_conv_out
      .endif

      g_line_conv = g_line_conv + 1
   .endm

   .load_line in, line_fetch_in
   s_mov_b32 s[loop_h_cnt], 0

   // pipe prologue
   skipped_fetches = 0
   .rept pipe_lines_depth
      fetch_step
   .endr
   vmcnt_per_step = mbufs_per_line * 2

   s_wait vmcnt_per_step*(pipe_lines_depth-1) - skipped_fetches * mbufs_per_line
   conv_step

   // software pipeline
loop_h_begin:
   period = lines_cnt_in * lines_cnt_out
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
         fetch_step
         s_wait vmcnt_per_step*(pipe_lines_depth-1)
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
      fetch_step
      s_wait vmcnt_per_step*(pipe_lines_depth-1)
      conv_step
   .endr

   // pipe epilogue
   fetch_step
   s_wait 0 // todo: add proper counter (last line is different)
   .rept pipe_lines_depth
      conv_step
   .endr


   s_add_u32 s[soffset_in], s[soffset_in], 0 + input_stack_size * n_per_group - input_feature_map_size
   s_add_u32 s[soffset_out], s[soffset_out], 0 + output_stack_size * n_per_group - output_feature_map_size
loop_n_end:
   s_addk_i32 s[loop_n_cnt], 1
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
      //v_add_u32 v[lds_off], vcc, s[wave_id], v[lds_off]
      //.ds_write_all
      v_lshlrev_b32 v[lds_off], 2, v[tid]
      v_add_u32 v[lds_off], vcc, s[wave_id], v[lds_off]
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
         v_add_u32 v[lds_off], vcc, 4 * 64 * accums_cnt, v[lds_off]
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
      buffer_store_dwordx4 v[acc_base+0:acc_base+3], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0
      buffer_store_dwordx4 v[acc_base+4:acc_base+7], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0+4*4
      buffer_store_dword v[acc_base+8], v[voffset_wei], s[desc_wei:desc_wei+3], s[soffset_wei] offen offset:0+8*4
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
         v_add_u32 v[voffset_wei], vcc, v[voffset_wei], v[c_offset]
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
   .size gcnAsmConv3x3WrW, .Lfunc_end0 - gcnAsmConv3x3WrW


.macro metadata wg_x
    .amdgpu_code_object_metadata
    { Version: [ 3, 0 ],
        Kernels:
        - { Name: gcnAsmConv3x3WrW, Language: OpenCL C, LanguageVersion: [ 1, 2 ],
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
