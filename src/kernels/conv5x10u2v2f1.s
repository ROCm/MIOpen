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
/*
 * Convolution Kernel for 5x10 kernel with stride equal to 2 (i.e., -x 10 -y 5 -u 2 -v 2)
 * works on devices compatible with GCN3 ISA, but not XNACK.
 */
.include "inst_wrappers.inc"

.hsa_code_object_version 2,1
.hsa_code_object_isa
.if (.option.machine_version_major != 8) && (.option.machine_version_major != 9)
.error "ERROR: specified target machine not supported"
.endif

///////////////////////////////////////////////////
// ******* global-work and work-group-size
//  work-group-size = [64, 8, 1]
//  global-work = [alignUp(out_w,64), (alignUp(out_h,4)/4)*alignUp(wei_k/2,8), batch_n]
//    * def alignUp(a,b) = ((a + b - 1)/b)*b
//    * def out_w = (inp_w + 2*pad_w + inp_u - wei_w) / inp_u
//    * def out_h = (inp_h + 2*pad_h + inp_v - wei_h) / inp_v
//  NOTE: Each workgroup will process 1x16x4x64(NCHW) tile of output tensor.
//        So there will some loss of performance when output tensor size
//        is not an integer multiple of 1x16x4x64 tile.
///////////////////////////////////////////////////
// ******* changeable configuration parameters
//   inp_w          - input tensor width
//   inp_h          - input tensor height
//   wei_c          - input tensor channels
//   wei_k          - output tensor channels (must be multiple of 2)
//   wei_layout     - weights layout 0:"KCHW" or 1:"CKHW"
//   pad_w          - input padding on left and right
//   pad_h          - input padding on top and bottom
.ifndef inp_w
.error "ERROR: configurable parameter: inp_w must be defined"
.endif
.ifndef inp_h
.error "ERROR: configurable parameter: inp_h must be defined"
.endif
.ifndef wei_c
.error "ERROR: configurable parameter: wei_c must be defined"
.endif
.ifndef wei_k
.error "ERROR: configurable parameter: wei_k must be defined"
.endif
.ifndef wei_layout
.error "ERROR: configurable parameter: wei_layout must be defined"
.endif
.if (wei_k % 2) != 0
.error "ERROR: wei_k must be multiple of 2"
.endif
.ifndef pad_w
.error "ERROR: configurable parameter: pad_w must be defined"
.endif
.ifndef pad_h
.error "ERROR: configurable parameter: pad_h must be defined"
.endif
// ******* build flags
.set use_ds_read_b128, 0
// ******* fixed configuration parameters
.set wei_w       ,  10
.set wei_h       ,   5
.set inp_u       ,   2
.set inp_v       ,   2
// ******* LDS allocation
.set LDS_SIZE    ,4*(64*inp_u-1+(wei_w-1))*(4*inp_v-1+(wei_h-1))+32 // = 6016 bytes
// ******* SGPR allocation
// For used during initialization or as temporary
.set sreg_karg   ,   0   // [2]
.set sreg_group_0,   2   // [1]
.set sreg_group_1,   3   // [1]
.set sreg_group_2,   4   // [1]
.set sreg_tmp0   ,   5   // [1]
.set sreg_tmp1   ,   6   // [1]
.set sreg_tmp2   ,   7   // [1]
.set sreg_iinc   ,   8   // [1]
.set sreg_winc   ,   9   // [1]
.set sreg_inp_addr, 10   // [2]
.set sreg_out_addr, 12   // [2]/[4]
.set sreg_dswr1vcc, 14   // [2]
.set sreg_k       , 16   // [1]
.set sreg_c       , 17   // [1]
.set sreg_dy      , 18   // [1]
.set sreg_oinc    , 19   // [1]
.set sreg_dsrd0vcc, 20   // [2]
.set sreg_dsrd1vcc, 22   // [2]
.set sreg_dspl0vcc, 24   // [2]
.set sreg_dspl1vcc, 26   // [2]
.set sreg_dspr0vcc, 28   // [2]
.set sreg_dspr1vcc, 30   // [2]
.set sreg_pad_val , 32   // [1]
.set sreg_sx      , 33   // [1]
.set sreg_sy      , 34   // [1]
// For use during core-loop and later
.set sreg_wval    ,  0   // [100]
.set sreg_wei_addr,100   // [2]
.set sreg_vcc_resv,102   // [2]
.set SGPR_COUNT   ,108   // COUNT
// ******* VGPR allocation
// For used during initialization
.set vreg_local_0 ,  0   // [1]
.set vreg_local_1 ,  1   // [1]
.set vreg_local_2 ,  2   // [1] unused
.set vreg_tmp0    ,  3   // [1]
.set vreg_tmp1    ,  4   // [1]
.set vreg_tmp2    ,  5   // [1]
.set vreg_tmp3    ,  6   // [1]
.set vreg_lx0     ,  7   // [1]
.set vreg_ly0     ,  8   // [1]
.set vreg_lx1     ,  9   // [1]
.set vreg_ly1     , 10   // [1]
.set vreg_sx0     , 11   // [1]
.set vreg_sy0     , 12   // [1]
.set vreg_sx1     , 13   // [1]
.set vreg_sy1     , 14   // [1]
// For use during core-loop and later
.set vreg_ival    ,  0   // [8]
.set vreg_oval    ,  8   // [8]
.set vreg_iinc0   , 16   // [1]
.set vreg_iinc1   , 17   // [1]
.set vreg_inp_dswr0,18   // [1]
.set vreg_inp_dswr1,19   // [1]
.set vreg_inp_dsrd0,20   // [1]
.set vreg_dx      , 21   // [1]
.set vreg_save    , 22   // [1]
.set VGPR_COUNT   , 24   // COUNT
// Lane# of saved sreg_* in vreg_save
.set lane_vreg_save_wr1vcc0   ,  0
.set lane_vreg_save_wr1vcc1   ,  1
.set lane_vreg_save_c         ,  2
.set lane_vreg_save_k         ,  3
.set lane_vreg_save_dy        ,  4
.set lane_vreg_save_inp_addr0 ,  5
.set lane_vreg_save_inp_addr1 ,  6
.set lane_vreg_save_out_addr0 ,  7
.set lane_vreg_save_out_addr1 ,  8
.set lane_vreg_save_pad_val   ,  9
.set lane_vreg_save_rd0vcc0   , 10
.set lane_vreg_save_rd0vcc1   , 11
.set lane_vreg_save_rd1vcc0   , 12
.set lane_vreg_save_rd1vcc1   , 13
.set lane_vreg_save_pl0vcc0   , 14
.set lane_vreg_save_pl0vcc1   , 15
.set lane_vreg_save_pr0vcc0   , 16
.set lane_vreg_save_pr0vcc1   , 17
.set lane_vreg_save_pl1vcc0   , 18
.set lane_vreg_save_pl1vcc1   , 19
.set lane_vreg_save_pr1vcc0   , 20
.set lane_vreg_save_pr1vcc1   , 21
// ******* derived constants
.set out_w       ,(inp_w + 2*pad_w + inp_u - wei_w) / inp_u
.set out_h       ,(inp_h + 2*pad_h + inp_v - wei_h) / inp_v
.set inp_stride_y,(inp_w * 4)
.set inp_stride_c,(inp_h * inp_stride_y)
.set inp_stride_n,(wei_c * inp_stride_c)
.set out_stride_y,(out_w * 4)
.set out_stride_k,(out_h * out_stride_y)
.set out_stride_n,(wei_k * out_stride_k)
.if wei_layout == 0 // KCHW
.set wei_stride_c,(wei_h * wei_w * 4)
.set wei_stride_k,(wei_c * wei_stride_c)
.elseif wei_layout == 1 // CKHW
.set wei_stride_k,(wei_h * wei_w * 4)
.set wei_stride_c,(wei_k * wei_stride_k)
.else
.error "ERROR: wei_layout should be 0 (for:KCHW) or 1 (for:CKHW)"
.endif
.macro .bitcount, n, bits
  .if (1 << \bits) < \n
    .set \bits, \bits + 1
    .bitcount wei_k, wei_k_bits
  .endif
.endm
.set wei_k_bits  , 0
.bitcount wei_k  , wei_k_bits
.set wei_k_mask  , ((1 << wei_k_bits) - 1)
// ******* derived flags
.set padding_enabled     , ((pad_w != 0) || (pad_h != 0))
.set dspl0_enabled       ,  (pad_w % 2) == 1
.set dspl1_enabled       ,  (pad_w % 2) == 1
.set dspr0_enabled       ,  (pad_w > 0) && (((inp_w + pad_w) % 2) == 1) && (((inp_w + pad_w) % 128) >= 8)
.set dspr1_enabled       ,  (pad_w > 0) && (((inp_w + pad_w) % 2) == 1)
// ******* macros for ds_read_b128
.macro macro_ds_read_b128 dest, addr, offset=0
  .short 0x0+\offset, 0xd9fe
  .byte 0x0+\addr, 0x0, 0x0, 0x0+\dest
.endm

///////////////////////////////////////////////////
// ******* text section of the kernels
///////////////////////////////////////////////////
.text
.p2align 8
.global conv5x10u2v2f1
.type conv5x10u2v2f1, @function
.amdgpu_hsa_kernel conv5x10u2v2f1
conv5x10u2v2f1:

    .amd_kernel_code_t
        amd_machine_version_major = .option.machine_version_major
        amd_machine_version_minor = .option.machine_version_minor
        amd_machine_version_stepping = .option.machine_version_stepping
        is_ptr64 = 1
        float_mode = 192
        user_sgpr_count = 2
        is_xnack_enabled = 0
        enable_sgpr_workgroup_id_x = 1
        enable_sgpr_workgroup_id_y = 1
        enable_sgpr_workgroup_id_z = 1
        enable_vgpr_workitem_id = 1
        enable_sgpr_kernarg_segment_ptr = 1
        workitem_vgpr_count = VGPR_COUNT
        wavefront_sgpr_count = SGPR_COUNT
        workgroup_group_segment_byte_size = LDS_SIZE
        kernarg_segment_byte_size = 56
        granulated_workitem_vgpr_count = (VGPR_COUNT-1)/4
        granulated_wavefront_sgpr_count = (SGPR_COUNT-1)/8
    .end_amd_kernel_code_t

    //////////////////////////////////////////////////////////////////////////////
    // initialization
    //  - work-items:
    //      work-group-size = [64, 8, 1]
    //      global-work = [alignUp(out_w,64), (alignUp(out_h,4)/4)*alignUp(wei_k/2,8), batch_n]
    //      work-item relation to output buffer:
    //        dx =  global-work[0]
    //        dy = (global-work[1] >> (wei_k_bits-4)) * 4
    //        k  = (global-work[1] << 1) & wei_k_mask
    //        n  =  global-work[2]
    //      calculation:
    //        dx =  group_id(0) * 64 + local_id(0)
    //        dy = (group_id(1) >> (wei_k_bits-4)) * 4
    //        k  =((group_id(1) << 3) + local_id(1))*2 & wei_k_mask
    //        n  =  group_id(2)
    // - calculate sreg_wei_addr for current wave
    //      sreg_wei_addr += min(k,wei_k-1) * wei_stride_k
    //  - calculate sreg_out_addr for current wave
    //      sreg_out_addr += n * out_stride_n + k * out_stride_k + dy * out_stride_y
    //  - calculate sreg_inp_addr for current wave
    //      sreg_inp_addr += n * inp_stride_n
    //  - calculate registers for managing input in LDS
    //      lx0 = local_id(0) * 2
    //      ly0 = local_id(1)
    //      lx1 = (local_id(1) < 3) ? ((local_id(0)  & 3)*2 + 128) ?  lx0
    //      ly1 = (local_id(1) < 3) ?  (local_id(0) >> 2)          : (ly0 + 8)
    //      vreg_inp_dswr0 = lx0 * 4 + ly0 * 136 * 4
    //      vreg_inp_dswr1 = lx1 * 4 + ly1 * 136 * 4
    //      vreg_inp_dsrd0 = lx0 * 4
    //      sx  = 2 * group_id(0) * 64
    //      sy  = 2 * sreg_dy
    //      sx0 = sx + lx0 - pad_w
    //      sy0 = sy + ly0 - pad_h
    //      sx1 = sx + lx1 - pad_w
    //      sy1 = sy + ly1 - pad_h
    //      vreg_iinc0 = sy0 * inp_stride_y + max(0, sx0) * 4
    //      vreg_iinc1 = sy1 * inp_stride_y + max(0, sx1) * 4
    //      dsr_right_pos = inp_w + (inp_w & 1) - (pad_w & inp_w & 1)
    //      sreg_dsrd0vcc = (sx0 >= -1 && sx0 < dsr_right_pos && sy0 >= 0 && sy0 < inp_h)
    //      sreg_dsrd1vcc = (sx1 >= -1 && sx1 < dsr_right_pos && sy1 >= 0 && sy1 < inp_h) && (ly1 < 11)
    //      sreg_dswr1vcc = (ly1 < 11)
    //      dsp_right_pos = inp_w - (inp_w & 1) - (pad_w & 1)
    //      sreg_dspl0vcc = (sx0 == -1)
    //      sreg_dspr0vcc = (sx0 == dsp_right_pos)
    //      sreg_dspl1vcc = (sx1 == -1)
    //      sreg_dspr0vcc = (sx1 == dsp_right_pos)
    //  - save sreg values in vreg_save
    //  - initialize sreg_c and output values
    //////////////////////////////////////////////////////////////////////////////
    s_mov_b32 m0, LDS_SIZE
    // load for parameters
    s_load_dwordx2 s[sreg_inp_addr:sreg_inp_addr+1], s[sreg_karg:sreg_karg+1], 0x00
    s_load_dwordx2 s[sreg_wei_addr:sreg_wei_addr+1], s[sreg_karg:sreg_karg+1], 0x08
    s_load_dwordx2 s[sreg_out_addr:sreg_out_addr+1], s[sreg_karg:sreg_karg+1], 0x10
.if padding_enabled
    s_load_dword   s[sreg_pad_val                 ], s[sreg_karg:sreg_karg+1], 0x18
.endif
    // compute: sreg_dx, sreg_dy, sreg_k
    s_lshl_b32 s[sreg_tmp0], s[sreg_group_0], 6
   _v_add_co_u32 v[vreg_dx], vcc, s[sreg_tmp0], v[vreg_local_0]
    s_lshr_b32 s[sreg_dy], s[sreg_group_1], wei_k_bits-4
    s_lshl_b32 s[sreg_dy], s[sreg_dy], 2
    v_readfirstlane_b32 s[sreg_k], v[vreg_local_1]
    s_lshl_b32 s[sreg_tmp0], s[sreg_group_1], 3
    s_add_u32  s[sreg_k], s[sreg_k], s[sreg_tmp0]
    s_lshl_b32 s[sreg_k], s[sreg_k], 1
    s_and_b32  s[sreg_k], s[sreg_k], wei_k_mask
    // compute: sreg_winc = min(k,wei_k-1) * wei_stride_k
    s_min_i32  s[sreg_winc], s[sreg_k], wei_k-1
    s_mul_i32  s[sreg_winc], s[sreg_winc], wei_stride_k
    // compute: sreg_oinc = group_id(2) * out_stride_n + k * out_stride_k + dy * out_stride_y
    s_mul_i32  s[sreg_oinc], s[sreg_group_2], out_stride_n
    s_mul_i32  s[sreg_tmp1], s[sreg_k], out_stride_k
    s_add_u32  s[sreg_oinc], s[sreg_oinc], s[sreg_tmp1]
    s_mul_i32  s[sreg_tmp1], s[sreg_dy], out_stride_y
    s_add_u32  s[sreg_oinc], s[sreg_oinc], s[sreg_tmp1]
    // compute: registers for transfering input into LDS
    v_lshlrev_b32 v[vreg_lx0], 1, v[vreg_local_0]
    v_mov_b32     v[vreg_ly0], v[vreg_local_1]
    v_and_b32     v[vreg_lx1], 3, v[vreg_local_0]
    v_lshlrev_b32 v[vreg_lx1], 1, v[vreg_lx1]
   _v_add_co_u32  v[vreg_lx1], vcc, 128, v[vreg_lx1]
    v_lshrrev_b32 v[vreg_ly1], 2, v[vreg_local_0]
   _v_add_co_u32  v[vreg_tmp0], vcc, 8, v[vreg_ly0]
    v_cmp_eq_u32  vcc, 3, v[vreg_local_1]
    v_cndmask_b32 v[vreg_lx1], v[vreg_lx0], v[vreg_lx1], vcc
    v_cndmask_b32 v[vreg_ly1], v[vreg_tmp0], v[vreg_ly1], vcc
    v_lshlrev_b32 v[vreg_inp_dsrd0], 2, v[vreg_lx0]
    s_movk_i32    s[sreg_tmp0], 136*4
    v_lshlrev_b32 v[vreg_tmp1], 2, v[vreg_lx1]
    v_mad_u32_u24 v[vreg_inp_dswr0], v[vreg_ly0], s[sreg_tmp0], v[vreg_inp_dsrd0]
    v_mad_u32_u24 v[vreg_inp_dswr1], v[vreg_ly1], s[sreg_tmp0], v[vreg_tmp1]
    s_lshl_b32    s[sreg_sx], s[sreg_group_0], 7
    s_lshl_b32    s[sreg_sy], s[sreg_dy], 1
    s_sub_u32     s[sreg_sx], s[sreg_sx], 0+pad_w
    s_sub_u32     s[sreg_sy], s[sreg_sy], 0+pad_h
   _v_add_co_u32  v[vreg_sx0], vcc, s[sreg_sx], v[vreg_lx0]
   _v_add_co_u32  v[vreg_sy0], vcc, s[sreg_sy], v[vreg_ly0]
   _v_add_co_u32  v[vreg_sx1], vcc, s[sreg_sx], v[vreg_lx1]
   _v_add_co_u32  v[vreg_sy1], vcc, s[sreg_sy], v[vreg_ly1]
    v_max_i32     v[vreg_iinc0], 0, v[vreg_sx0]
    v_max_i32     v[vreg_iinc1], 0, v[vreg_sx1]
    v_lshlrev_b32 v[vreg_iinc0], 2, v[vreg_iinc0]
    v_lshlrev_b32 v[vreg_iinc1], 2, v[vreg_iinc1]
    v_mov_b32     v[vreg_tmp0], 0+inp_stride_y
    v_mad_u32_u24 v[vreg_iinc0], v[vreg_sy0], v[vreg_tmp0], v[vreg_iinc0]
    v_mad_u32_u24 v[vreg_iinc1], v[vreg_sy1], v[vreg_tmp0], v[vreg_iinc1]
.if padding_enabled
    v_mov_b32     v[vreg_tmp0], 0+inp_w+(inp_w & 1)-(inp_w & pad_w & 1)
    v_mov_b32     v[vreg_tmp1], 0+inp_h
    v_cmp_le_i32  vcc, -1, v[vreg_sx0]
    s_mov_b64     s[sreg_dsrd0vcc:sreg_dsrd0vcc+1], vcc
    v_cmp_gt_i32  vcc, v[vreg_tmp0], v[vreg_sx0]
    s_and_b64     s[sreg_dsrd0vcc:sreg_dsrd0vcc+1], s[sreg_dsrd0vcc:sreg_dsrd0vcc+1], vcc
    v_cmp_le_i32  vcc, 0, v[vreg_sy0]
    s_and_b64     s[sreg_dsrd0vcc:sreg_dsrd0vcc+1], s[sreg_dsrd0vcc:sreg_dsrd0vcc+1], vcc
    v_cmp_gt_i32  vcc, v[vreg_tmp1], v[vreg_sy0]
    s_and_b64     s[sreg_dsrd0vcc:sreg_dsrd0vcc+1], s[sreg_dsrd0vcc:sreg_dsrd0vcc+1], vcc
.endif
    v_cmp_gt_i32  vcc, 11, v[vreg_ly1]
    s_mov_b64     s[sreg_dswr1vcc:sreg_dswr1vcc+1], vcc
.if padding_enabled
    v_cmp_le_i32  vcc, -1, v[vreg_sx1]
    s_and_b64     s[sreg_dsrd1vcc:sreg_dsrd1vcc+1], s[sreg_dswr1vcc:sreg_dswr1vcc+1], vcc
    v_cmp_gt_i32  vcc, v[vreg_tmp0], v[vreg_sx1]
    s_and_b64     s[sreg_dsrd1vcc:sreg_dsrd1vcc+1], s[sreg_dsrd1vcc:sreg_dsrd1vcc+1], vcc
    v_cmp_le_i32  vcc, 0, v[vreg_sy1]
    s_and_b64     s[sreg_dsrd1vcc:sreg_dsrd1vcc+1], s[sreg_dsrd1vcc:sreg_dsrd1vcc+1], vcc
    v_cmp_gt_i32  vcc, v[vreg_tmp1], v[vreg_sy1]
    s_and_b64     s[sreg_dsrd1vcc:sreg_dsrd1vcc+1], s[sreg_dsrd1vcc:sreg_dsrd1vcc+1], vcc
.if dspr0_enabled || dspr1_enabled
    v_mov_b32     v[vreg_tmp0], 0+inp_w-(inp_w & 1)-(pad_w & 1)
.endif
.if dspl0_enabled
    v_cmp_eq_i32  vcc, -1, v[vreg_sx0]
    s_mov_b64     s[sreg_dspl0vcc:sreg_dspl0vcc+1], vcc
.endif
.if dspr0_enabled
    v_cmp_eq_i32  vcc, v[vreg_tmp0], v[vreg_sx0]
    s_mov_b64     s[sreg_dspr0vcc:sreg_dspr0vcc+1], vcc
.endif
.if dspl1_enabled
    v_cmp_eq_i32  vcc, -1, v[vreg_sx1]
    s_mov_b64     s[sreg_dspl1vcc:sreg_dspl1vcc+1], vcc
.endif
.if dspr1_enabled
    v_cmp_eq_i32  vcc, v[vreg_tmp0], v[vreg_sx1]
    s_mov_b64     s[sreg_dspr1vcc:sreg_dspr1vcc+1], vcc
.endif
.endif
    // wait for load completion
    s_waitcnt lgkmcnt(0)
    // update address registers
    //   sreg_inp_addr += group_id(2) * inp_stride_n
    //   sreg_wei_addr += sreg_winc
    //   sreg_out_addr += sreg_oinc
    s_add_u32  s[sreg_wei_addr], s[sreg_wei_addr], s[sreg_winc]
    s_addc_u32 s[sreg_wei_addr+1], s[sreg_wei_addr+1], 0
    s_add_u32  s[sreg_out_addr], s[sreg_out_addr], s[sreg_oinc]
    s_addc_u32 s[sreg_out_addr+1], s[sreg_out_addr+1], 0
    s_mul_i32  s[sreg_tmp0], s[sreg_group_2], inp_stride_n
    s_add_u32  s[sreg_inp_addr], s[sreg_inp_addr], s[sreg_tmp0]
    s_addc_u32 s[sreg_inp_addr+1], s[sreg_inp_addr+1], 0
    // initialize output values and channel count
    s_movk_i32 s[sreg_c], 0+wei_c
    v_mov_b32 v[vreg_oval+0], 0
    v_mov_b32 v[vreg_oval+1], 0
    v_mov_b32 v[vreg_oval+2], 0
    v_mov_b32 v[vreg_oval+3], 0
    v_mov_b32 v[vreg_oval+4], 0
    v_mov_b32 v[vreg_oval+5], 0
    v_mov_b32 v[vreg_oval+6], 0
    v_mov_b32 v[vreg_oval+7], 0
    // save sreg_c, sreg_k, sreg_dy, sreg_inp/out_addr, sreg_*vcc:sreg_*vcc+1, sreg_pad_val into vreg_save
    v_writelane_b32 v[vreg_save], s[sreg_dswr1vcc], 0+lane_vreg_save_wr1vcc0
    v_writelane_b32 v[vreg_save], s[sreg_dswr1vcc+1], 0+lane_vreg_save_wr1vcc1
    v_writelane_b32 v[vreg_save], s[sreg_c], 0+lane_vreg_save_c
    v_writelane_b32 v[vreg_save], s[sreg_k], 0+lane_vreg_save_k
    v_writelane_b32 v[vreg_save], s[sreg_dy], 0+lane_vreg_save_dy
    v_writelane_b32 v[vreg_save], s[sreg_inp_addr+0], 0+lane_vreg_save_inp_addr0
    v_writelane_b32 v[vreg_save], s[sreg_inp_addr+1], 0+lane_vreg_save_inp_addr1
    v_writelane_b32 v[vreg_save], s[sreg_out_addr+0], 0+lane_vreg_save_out_addr0
    v_writelane_b32 v[vreg_save], s[sreg_out_addr+1], 0+lane_vreg_save_out_addr1
.if padding_enabled
    v_writelane_b32 v[vreg_save], s[sreg_pad_val], 0+lane_vreg_save_pad_val
    v_writelane_b32 v[vreg_save], s[sreg_dsrd0vcc+0], 0+lane_vreg_save_rd0vcc0
    v_writelane_b32 v[vreg_save], s[sreg_dsrd0vcc+1], 0+lane_vreg_save_rd0vcc1
    v_writelane_b32 v[vreg_save], s[sreg_dsrd1vcc+0], 0+lane_vreg_save_rd1vcc0
    v_writelane_b32 v[vreg_save], s[sreg_dsrd1vcc+1], 0+lane_vreg_save_rd1vcc1
.if dspl0_enabled
    v_writelane_b32 v[vreg_save], s[sreg_dspl0vcc+0], 0+lane_vreg_save_pl0vcc0
    v_writelane_b32 v[vreg_save], s[sreg_dspl0vcc+1], 0+lane_vreg_save_pl0vcc1
.endif
.if dspl1_enabled
    v_writelane_b32 v[vreg_save], s[sreg_dspl1vcc+0], 0+lane_vreg_save_pl1vcc0
    v_writelane_b32 v[vreg_save], s[sreg_dspl1vcc+1], 0+lane_vreg_save_pl1vcc1
.endif
.if dspr0_enabled
    v_writelane_b32 v[vreg_save], s[sreg_dspr0vcc+0], 0+lane_vreg_save_pr0vcc0
    v_writelane_b32 v[vreg_save], s[sreg_dspr0vcc+1], 0+lane_vreg_save_pr0vcc1
.endif
.if dspr1_enabled
    v_writelane_b32 v[vreg_save], s[sreg_dspr1vcc+0], 0+lane_vreg_save_pr1vcc0
    v_writelane_b32 v[vreg_save], s[sreg_dspr1vcc+1], 0+lane_vreg_save_pr1vcc1
.endif
.endif

    //////////////////////////////////////////////////////////////////////////////
    // loop though all channels:
    // registers with valid data from initialization
    //  - s[sreg_wei_addr:sreg_wei_addr+1]
    //  - v[vreg_save]
    //  - v[vreg_dx]
    //  - v[vreg_iinc0]
    //  - v[vreg_iinc1]
    //  - v[vreg_dswr0]
    //  - v[vreg_dswr1]
    //  - v[vreg_dsrd0]
    //  - v[vreg_oval:vreg_oval+7]
    // temporary registers used inside this loop:
    //  - s[sreg_wval:sreg_wval+99]
    //  - v[vreg_ival:vreg_ival+7]
    //////////////////////////////////////////////////////////////////////////////
loop_channel:
    // load input row into LDS
.if padding_enabled
    v_readlane_b32 s[sreg_dsrd0vcc+0], v[vreg_save], 0+lane_vreg_save_rd0vcc0
    v_readlane_b32 s[sreg_dsrd0vcc+1], v[vreg_save], 0+lane_vreg_save_rd0vcc1
    v_readlane_b32 s[sreg_dsrd1vcc+0], v[vreg_save], 0+lane_vreg_save_rd1vcc0
    v_readlane_b32 s[sreg_dsrd1vcc+1], v[vreg_save], 0+lane_vreg_save_rd1vcc1
.endif
    v_readlane_b32 s[sreg_wval+0], v[vreg_save], 0+lane_vreg_save_inp_addr0
    v_readlane_b32 s[sreg_wval+1], v[vreg_save], 0+lane_vreg_save_inp_addr1
    v_readlane_b32 s[sreg_dswr1vcc+0], v[vreg_save], 0+lane_vreg_save_wr1vcc0
    v_readlane_b32 s[sreg_dswr1vcc+1], v[vreg_save], 0+lane_vreg_save_wr1vcc1
    s_mov_b32      s[sreg_wval+2], 0+inp_stride_n
    s_mov_b32      s[sreg_wval+3], 0x00020000
.if padding_enabled
    v_readlane_b32 s[sreg_pad_val], v[vreg_save], 0+lane_vreg_save_pad_val
    v_mov_b32 v[vreg_ival+0], s[sreg_pad_val]
    v_mov_b32 v[vreg_ival+1], s[sreg_pad_val]
    v_mov_b32 v[vreg_ival+2], s[sreg_pad_val]
    v_mov_b32 v[vreg_ival+3], s[sreg_pad_val]
    s_mov_b64 exec, s[sreg_dsrd0vcc:sreg_dsrd0vcc+1]
.else
    s_nop 0
.endif
    buffer_load_dwordx2 v[vreg_ival+0:vreg_ival+1], v[vreg_iinc0], s[sreg_wval+0:sreg_wval+3], 0 offen offset:0
.if padding_enabled
    s_mov_b64 exec, s[sreg_dsrd1vcc:sreg_dsrd1vcc+1]
.else
    s_mov_b64 exec, s[sreg_dswr1vcc:sreg_dswr1vcc+1]
.endif
    buffer_load_dwordx2 v[vreg_ival+2:vreg_ival+3], v[vreg_iinc1], s[sreg_wval+0:sreg_wval+3], 0 offen offset:0
    s_mov_b64 exec, -1
    v_mov_b32  v[vreg_ival+4], 0+inp_stride_c
   _v_add_co_u32 v[vreg_iinc0], vcc, v[vreg_iinc0], v[vreg_ival+4]
   _v_add_co_u32 v[vreg_iinc1], vcc, v[vreg_iinc1], v[vreg_ival+4]
.if padding_enabled
.if dspl0_enabled
    v_readlane_b32 s[sreg_dspl0vcc+0], v[vreg_save], 0+lane_vreg_save_pl0vcc0
    v_readlane_b32 s[sreg_dspl0vcc+1], v[vreg_save], 0+lane_vreg_save_pl0vcc1
.endif
.if dspr0_enabled
    v_readlane_b32 s[sreg_dspr0vcc+0], v[vreg_save], 0+lane_vreg_save_pr0vcc0
    v_readlane_b32 s[sreg_dspr0vcc+1], v[vreg_save], 0+lane_vreg_save_pr0vcc1
.endif
.if dspl1_enabled
    v_readlane_b32 s[sreg_dspl1vcc+0], v[vreg_save], 0+lane_vreg_save_pl1vcc0
    v_readlane_b32 s[sreg_dspl1vcc+1], v[vreg_save], 0+lane_vreg_save_pl1vcc1
.endif
.if dspr1_enabled
    v_readlane_b32 s[sreg_dspr1vcc+0], v[vreg_save], 0+lane_vreg_save_pr1vcc0
    v_readlane_b32 s[sreg_dspr1vcc+1], v[vreg_save], 0+lane_vreg_save_pr1vcc1
.endif
.endif
    s_waitcnt lgkmcnt(0) vmcnt(0)
.if padding_enabled
.if dspl0_enabled
    s_mov_b64 exec, s[sreg_dspl0vcc:sreg_dspl0vcc+1]
    v_mov_b32 v[vreg_ival+1], v[vreg_ival+0]
    v_mov_b32 v[vreg_ival+0], s[sreg_pad_val]
.endif
.if dspr0_enabled
    s_mov_b64 exec, s[sreg_dspr0vcc:sreg_dspr0vcc+1]
    v_mov_b32 v[vreg_ival+1], s[sreg_pad_val]
.endif
.if dspl1_enabled
    s_mov_b64 exec, s[sreg_dspl1vcc:sreg_dspl1vcc+1]
    v_mov_b32 v[vreg_ival+3], v[vreg_ival+2]
    v_mov_b32 v[vreg_ival+2], s[sreg_pad_val]
.endif
.if dspr1_enabled
    s_mov_b64 exec, s[sreg_dspr1vcc:sreg_dspr1vcc+1]
    v_mov_b32 v[vreg_ival+3], s[sreg_pad_val]
.endif
.if dspl0_enabled || dspr0_enabled || dspl1_enabled || dspr1_enabled
    s_mov_b64 exec, -1
.endif
.endif
    s_barrier
    ds_write_b64 v[vreg_inp_dswr0], v[vreg_ival+0:vreg_ival+1]
    s_mov_b64 exec, s[sreg_dswr1vcc:sreg_dswr1vcc+1]
    ds_write_b64 v[vreg_inp_dswr1], v[vreg_ival+2:vreg_ival+3]
    s_mov_b64 exec, -1
    s_waitcnt lgkmcnt(0)
    s_barrier
    // load channel weights and update sreg_wei_addr for next loop iteration
    s_load_dwordx16 s[sreg_wval   :sreg_wval+15], s[sreg_wei_addr:sreg_wei_addr+1], 0
    s_load_dwordx16 s[sreg_wval+16:sreg_wval+31], s[sreg_wei_addr:sreg_wei_addr+1], 4*16
    s_load_dwordx16 s[sreg_wval+32:sreg_wval+47], s[sreg_wei_addr:sreg_wei_addr+1], 4*32
    s_load_dwordx2  s[sreg_wval+96:sreg_wval+97], s[sreg_wei_addr:sreg_wei_addr+1], 4*48
    s_add_u32  s[sreg_wei_addr], s[sreg_wei_addr], wei_stride_k
    s_addc_u32 s[sreg_wei_addr+1], s[sreg_wei_addr+1], 0
    s_load_dwordx16 s[sreg_wval+48:sreg_wval+63], s[sreg_wei_addr:sreg_wei_addr+1], 0
    s_load_dwordx16 s[sreg_wval+64:sreg_wval+79], s[sreg_wei_addr:sreg_wei_addr+1], 4*16
    s_load_dwordx16 s[sreg_wval+80:sreg_wval+95], s[sreg_wei_addr:sreg_wei_addr+1], 4*32
    s_load_dwordx2  s[sreg_wval+98:sreg_wval+99], s[sreg_wei_addr:sreg_wei_addr+1], 4*48
.if wei_layout == 0 // KCHW
    s_add_u32  s[sreg_wei_addr], s[sreg_wei_addr], wei_stride_c-wei_stride_k
    s_addc_u32 s[sreg_wei_addr+1], s[sreg_wei_addr+1], -1
.else // CKHW
    s_add_u32  s[sreg_wei_addr], s[sreg_wei_addr], wei_stride_c-wei_stride_k
    s_addc_u32 s[sreg_wei_addr+1], s[sreg_wei_addr+1], 0
.endif
    s_waitcnt lgkmcnt(0) vmcnt(0)

    // compute 2D conv
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+0, vreg_inp_dsrd0 4*(0*136+0)
    macro_ds_read_b128 vreg_ival+4, vreg_inp_dsrd0 4*(0*136+4)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(0*136+0)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(0*136+2)
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(0*136+4)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(0*136+6)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+0], s[sreg_wval+0*10+0]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+1], s[sreg_wval+0*10+1]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+2], s[sreg_wval+0*10+2]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+3], s[sreg_wval+0*10+3]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+0], s[sreg_wval+0*10+0+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+1], s[sreg_wval+0*10+1+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+2], s[sreg_wval+0*10+2+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+3], s[sreg_wval+0*10+3+48]
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(0*136+8)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(1*136+0)
    s_waitcnt lgkmcnt(2)
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+4], s[sreg_wval+0*10+4]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+5], s[sreg_wval+0*10+5]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+6], s[sreg_wval+0*10+6]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+7], s[sreg_wval+0*10+7]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+4], s[sreg_wval+0*10+4+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+5], s[sreg_wval+0*10+5+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+6], s[sreg_wval+0*10+6+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+7], s[sreg_wval+0*10+7+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+4, vreg_inp_dsrd0, 4*(1*136+2)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(1*136+2)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(1*136+4)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+0], s[sreg_wval+0*10+8]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+1], s[sreg_wval+0*10+9]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+2], s[sreg_wval+1*10+0]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+3], s[sreg_wval+1*10+1]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+0], s[sreg_wval+0*10+8+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+1], s[sreg_wval+0*10+9+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+2], s[sreg_wval+1*10+0+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+3], s[sreg_wval+1*10+1+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+0, vreg_inp_dsrd0, 4*(1*136+6)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(1*136+6)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(1*136+8)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+4], s[sreg_wval+1*10+2]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+5], s[sreg_wval+1*10+3]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+6], s[sreg_wval+1*10+4]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+7], s[sreg_wval+1*10+5]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+4], s[sreg_wval+1*10+2+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+5], s[sreg_wval+1*10+3+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+6], s[sreg_wval+1*10+4+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+7], s[sreg_wval+1*10+5+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+4, vreg_inp_dsrd0, 4*(2*136+0)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(2*136+0)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(2*136+2)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+0], s[sreg_wval+1*10+6]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+1], s[sreg_wval+1*10+7]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+2], s[sreg_wval+1*10+8]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+3], s[sreg_wval+1*10+9]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+0], s[sreg_wval+1*10+6+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+1], s[sreg_wval+1*10+7+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+2], s[sreg_wval+1*10+8+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+3], s[sreg_wval+1*10+9+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+0, vreg_inp_dsrd0, 4*(2*136+4)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(2*136+4)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(2*136+6)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+4], s[sreg_wval+2*10+0]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+5], s[sreg_wval+2*10+1]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+6], s[sreg_wval+2*10+2]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+7], s[sreg_wval+2*10+3]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+4], s[sreg_wval+0*10+0]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+5], s[sreg_wval+0*10+1]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+6], s[sreg_wval+0*10+2]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+7], s[sreg_wval+0*10+3]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+4], s[sreg_wval+2*10+0+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+5], s[sreg_wval+2*10+1+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+6], s[sreg_wval+2*10+2+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+7], s[sreg_wval+2*10+3+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+4], s[sreg_wval+0*10+0+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+5], s[sreg_wval+0*10+1+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+6], s[sreg_wval+0*10+2+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+7], s[sreg_wval+0*10+3+48]
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(2*136+8)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(3*136+0)
    s_waitcnt lgkmcnt(2)
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+0], s[sreg_wval+2*10+4]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+1], s[sreg_wval+2*10+5]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+2], s[sreg_wval+2*10+6]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+3], s[sreg_wval+2*10+7]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+0], s[sreg_wval+0*10+4]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+1], s[sreg_wval+0*10+5]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+2], s[sreg_wval+0*10+6]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+3], s[sreg_wval+0*10+7]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+0], s[sreg_wval+2*10+4+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+1], s[sreg_wval+2*10+5+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+2], s[sreg_wval+2*10+6+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+3], s[sreg_wval+2*10+7+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+0], s[sreg_wval+0*10+4+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+1], s[sreg_wval+0*10+5+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+2], s[sreg_wval+0*10+6+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+3], s[sreg_wval+0*10+7+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+0, vreg_inp_dsrd0, 4*(3*136+2)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(3*136+2)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(3*136+4)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+4], s[sreg_wval+2*10+8]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+5], s[sreg_wval+2*10+9]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+6], s[sreg_wval+3*10+0]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+7], s[sreg_wval+3*10+1]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+4], s[sreg_wval+0*10+8]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+5], s[sreg_wval+0*10+9]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+6], s[sreg_wval+1*10+0]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+7], s[sreg_wval+1*10+1]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+4], s[sreg_wval+2*10+8+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+5], s[sreg_wval+2*10+9+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+6], s[sreg_wval+3*10+0+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+7], s[sreg_wval+3*10+1+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+4], s[sreg_wval+0*10+8+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+5], s[sreg_wval+0*10+9+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+6], s[sreg_wval+1*10+0+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+7], s[sreg_wval+1*10+1+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+4, vreg_inp_dsrd0, 4*(3*136+6)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(3*136+6)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(3*136+8)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+0], s[sreg_wval+3*10+2]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+1], s[sreg_wval+3*10+3]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+2], s[sreg_wval+3*10+4]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+3], s[sreg_wval+3*10+5]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+0], s[sreg_wval+1*10+2]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+1], s[sreg_wval+1*10+3]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+2], s[sreg_wval+1*10+4]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+3], s[sreg_wval+1*10+5]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+0], s[sreg_wval+3*10+2+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+1], s[sreg_wval+3*10+3+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+2], s[sreg_wval+3*10+4+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+3], s[sreg_wval+3*10+5+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+0], s[sreg_wval+1*10+2+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+1], s[sreg_wval+1*10+3+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+2], s[sreg_wval+1*10+4+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+3], s[sreg_wval+1*10+5+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+0, vreg_inp_dsrd0, 4*(4*136+0)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(4*136+0)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(4*136+2)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+4], s[sreg_wval+3*10+6]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+5], s[sreg_wval+3*10+7]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+6], s[sreg_wval+3*10+8]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+7], s[sreg_wval+3*10+9]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+4], s[sreg_wval+1*10+6]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+5], s[sreg_wval+1*10+7]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+6], s[sreg_wval+1*10+8]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+7], s[sreg_wval+1*10+9]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+4], s[sreg_wval+3*10+6+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+5], s[sreg_wval+3*10+7+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+6], s[sreg_wval+3*10+8+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+7], s[sreg_wval+3*10+9+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+4], s[sreg_wval+1*10+6+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+5], s[sreg_wval+1*10+7+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+6], s[sreg_wval+1*10+8+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+7], s[sreg_wval+1*10+9+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+4, vreg_inp_dsrd0, 4*(4*136+4)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(4*136+4)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(4*136+6)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+0], s[sreg_wval+4*10+0]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+1], s[sreg_wval+4*10+1]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+2], s[sreg_wval+4*10+2]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+3], s[sreg_wval+4*10+3]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+0], s[sreg_wval+2*10+0]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+1], s[sreg_wval+2*10+1]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+2], s[sreg_wval+2*10+2]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+3], s[sreg_wval+2*10+3]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+0], s[sreg_wval+0*10+0]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+1], s[sreg_wval+0*10+1]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+2], s[sreg_wval+0*10+2]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+3], s[sreg_wval+0*10+3]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+0], s[sreg_wval+4*10+0+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+1], s[sreg_wval+4*10+1+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+2], s[sreg_wval+4*10+2+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+3], s[sreg_wval+4*10+3+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+0], s[sreg_wval+2*10+0+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+1], s[sreg_wval+2*10+1+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+2], s[sreg_wval+2*10+2+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+3], s[sreg_wval+2*10+3+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+0], s[sreg_wval+0*10+0+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+1], s[sreg_wval+0*10+1+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+2], s[sreg_wval+0*10+2+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+3], s[sreg_wval+0*10+3+48]
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(4*136+8)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(5*136+0)
    s_waitcnt lgkmcnt(2)
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+4], s[sreg_wval+4*10+4]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+5], s[sreg_wval+4*10+5]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+6], s[sreg_wval+4*10+6]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+7], s[sreg_wval+4*10+7]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+4], s[sreg_wval+2*10+4]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+5], s[sreg_wval+2*10+5]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+6], s[sreg_wval+2*10+6]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+7], s[sreg_wval+2*10+7]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+4], s[sreg_wval+0*10+4]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+5], s[sreg_wval+0*10+5]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+6], s[sreg_wval+0*10+6]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+7], s[sreg_wval+0*10+7]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+4], s[sreg_wval+4*10+4+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+5], s[sreg_wval+4*10+5+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+6], s[sreg_wval+4*10+6+48]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+7], s[sreg_wval+4*10+7+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+4], s[sreg_wval+2*10+4+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+5], s[sreg_wval+2*10+5+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+6], s[sreg_wval+2*10+6+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+7], s[sreg_wval+2*10+7+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+4], s[sreg_wval+0*10+4+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+5], s[sreg_wval+0*10+5+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+6], s[sreg_wval+0*10+6+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+7], s[sreg_wval+0*10+7+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+4, vreg_inp_dsrd0, 4*(5*136+2)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(5*136+2)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(5*136+4)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+0], s[sreg_wval+4*10+8+48]
    v_mac_f32 v[vreg_oval+0], v[vreg_ival+1], s[sreg_wval+4*10+9+48]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+0], s[sreg_wval+2*10+8]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+1], s[sreg_wval+2*10+9]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+2], s[sreg_wval+3*10+0]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+3], s[sreg_wval+3*10+1]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+0], s[sreg_wval+0*10+8]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+1], s[sreg_wval+0*10+9]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+2], s[sreg_wval+1*10+0]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+3], s[sreg_wval+1*10+1]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+0], s[sreg_wval+4*10+8+50]
    v_mac_f32 v[vreg_oval+4], v[vreg_ival+1], s[sreg_wval+4*10+9+50]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+0], s[sreg_wval+2*10+8+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+1], s[sreg_wval+2*10+9+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+2], s[sreg_wval+3*10+0+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+3], s[sreg_wval+3*10+1+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+0], s[sreg_wval+0*10+8+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+1], s[sreg_wval+0*10+9+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+2], s[sreg_wval+1*10+0+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+3], s[sreg_wval+1*10+1+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+0, vreg_inp_dsrd0, 4*(5*136+6)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(5*136+6)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(5*136+8)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+4], s[sreg_wval+3*10+2]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+5], s[sreg_wval+3*10+3]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+6], s[sreg_wval+3*10+4]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+7], s[sreg_wval+3*10+5]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+4], s[sreg_wval+1*10+2]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+5], s[sreg_wval+1*10+3]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+6], s[sreg_wval+1*10+4]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+7], s[sreg_wval+1*10+5]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+4], s[sreg_wval+3*10+2+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+5], s[sreg_wval+3*10+3+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+6], s[sreg_wval+3*10+4+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+7], s[sreg_wval+3*10+5+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+4], s[sreg_wval+1*10+2+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+5], s[sreg_wval+1*10+3+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+6], s[sreg_wval+1*10+4+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+7], s[sreg_wval+1*10+5+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+4, vreg_inp_dsrd0, 4*(6*136+0)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(6*136+0)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(6*136+2)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+0], s[sreg_wval+3*10+6]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+1], s[sreg_wval+3*10+7]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+2], s[sreg_wval+3*10+8]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+3], s[sreg_wval+3*10+9]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+0], s[sreg_wval+1*10+6]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+1], s[sreg_wval+1*10+7]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+2], s[sreg_wval+1*10+8]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+3], s[sreg_wval+1*10+9]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+0], s[sreg_wval+3*10+6+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+1], s[sreg_wval+3*10+7+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+2], s[sreg_wval+3*10+8+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+3], s[sreg_wval+3*10+9+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+0], s[sreg_wval+1*10+6+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+1], s[sreg_wval+1*10+7+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+2], s[sreg_wval+1*10+8+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+3], s[sreg_wval+1*10+9+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+0, vreg_inp_dsrd0, 4*(6*136+4)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(6*136+4)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(6*136+6)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+4], s[sreg_wval+4*10+0]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+5], s[sreg_wval+4*10+1]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+6], s[sreg_wval+4*10+2]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+7], s[sreg_wval+4*10+3]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+4], s[sreg_wval+2*10+0]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+5], s[sreg_wval+2*10+1]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+6], s[sreg_wval+2*10+2]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+7], s[sreg_wval+2*10+3]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+4], s[sreg_wval+0*10+0]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+5], s[sreg_wval+0*10+1]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+6], s[sreg_wval+0*10+2]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+7], s[sreg_wval+0*10+3]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+4], s[sreg_wval+4*10+0+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+5], s[sreg_wval+4*10+1+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+6], s[sreg_wval+4*10+2+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+7], s[sreg_wval+4*10+3+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+4], s[sreg_wval+2*10+0+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+5], s[sreg_wval+2*10+1+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+6], s[sreg_wval+2*10+2+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+7], s[sreg_wval+2*10+3+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+4], s[sreg_wval+0*10+0+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+5], s[sreg_wval+0*10+1+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+6], s[sreg_wval+0*10+2+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+7], s[sreg_wval+0*10+3+48]
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(6*136+8)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(7*136+0)
    s_waitcnt lgkmcnt(2)
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+0], s[sreg_wval+4*10+4]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+1], s[sreg_wval+4*10+5]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+2], s[sreg_wval+4*10+6]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+3], s[sreg_wval+4*10+7]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+0], s[sreg_wval+2*10+4]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+1], s[sreg_wval+2*10+5]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+2], s[sreg_wval+2*10+6]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+3], s[sreg_wval+2*10+7]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+0], s[sreg_wval+0*10+4]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+1], s[sreg_wval+0*10+5]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+2], s[sreg_wval+0*10+6]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+3], s[sreg_wval+0*10+7]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+0], s[sreg_wval+4*10+4+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+1], s[sreg_wval+4*10+5+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+2], s[sreg_wval+4*10+6+48]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+3], s[sreg_wval+4*10+7+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+0], s[sreg_wval+2*10+4+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+1], s[sreg_wval+2*10+5+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+2], s[sreg_wval+2*10+6+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+3], s[sreg_wval+2*10+7+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+0], s[sreg_wval+0*10+4+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+1], s[sreg_wval+0*10+5+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+2], s[sreg_wval+0*10+6+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+3], s[sreg_wval+0*10+7+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+0, vreg_inp_dsrd0, 4*(7*136+2)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(7*136+2)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(7*136+4)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+4], s[sreg_wval+4*10+8+48]
    v_mac_f32 v[vreg_oval+1], v[vreg_ival+5], s[sreg_wval+4*10+9+48]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+4], s[sreg_wval+2*10+8]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+5], s[sreg_wval+2*10+9]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+6], s[sreg_wval+3*10+0]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+7], s[sreg_wval+3*10+1]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+4], s[sreg_wval+0*10+8]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+5], s[sreg_wval+0*10+9]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+6], s[sreg_wval+1*10+0]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+7], s[sreg_wval+1*10+1]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+4], s[sreg_wval+4*10+8+50]
    v_mac_f32 v[vreg_oval+5], v[vreg_ival+5], s[sreg_wval+4*10+9+50]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+4], s[sreg_wval+2*10+8+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+5], s[sreg_wval+2*10+9+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+6], s[sreg_wval+3*10+0+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+7], s[sreg_wval+3*10+1+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+4], s[sreg_wval+0*10+8+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+5], s[sreg_wval+0*10+9+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+6], s[sreg_wval+1*10+0+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+7], s[sreg_wval+1*10+1+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+4, vreg_inp_dsrd0, 4*(7*136+6)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(7*136+6)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(7*136+8)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+0], s[sreg_wval+3*10+2]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+1], s[sreg_wval+3*10+3]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+2], s[sreg_wval+3*10+4]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+3], s[sreg_wval+3*10+5]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+0], s[sreg_wval+1*10+2]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+1], s[sreg_wval+1*10+3]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+2], s[sreg_wval+1*10+4]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+3], s[sreg_wval+1*10+5]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+0], s[sreg_wval+3*10+2+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+1], s[sreg_wval+3*10+3+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+2], s[sreg_wval+3*10+4+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+3], s[sreg_wval+3*10+5+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+0], s[sreg_wval+1*10+2+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+1], s[sreg_wval+1*10+3+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+2], s[sreg_wval+1*10+4+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+3], s[sreg_wval+1*10+5+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+0, vreg_inp_dsrd0, 4*(8*136+0)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(8*136+0)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(8*136+2)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+4], s[sreg_wval+3*10+6]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+5], s[sreg_wval+3*10+7]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+6], s[sreg_wval+3*10+8]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+7], s[sreg_wval+3*10+9]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+4], s[sreg_wval+1*10+6]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+5], s[sreg_wval+1*10+7]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+6], s[sreg_wval+1*10+8]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+7], s[sreg_wval+1*10+9]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+4], s[sreg_wval+3*10+6+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+5], s[sreg_wval+3*10+7+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+6], s[sreg_wval+3*10+8+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+7], s[sreg_wval+3*10+9+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+4], s[sreg_wval+1*10+6+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+5], s[sreg_wval+1*10+7+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+6], s[sreg_wval+1*10+8+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+7], s[sreg_wval+1*10+9+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+4, vreg_inp_dsrd0, 4*(8*136+4)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(8*136+4)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(8*136+6)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+0], s[sreg_wval+4*10+0]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+1], s[sreg_wval+4*10+1]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+2], s[sreg_wval+4*10+2]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+3], s[sreg_wval+4*10+3]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+0], s[sreg_wval+2*10+0]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+1], s[sreg_wval+2*10+1]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+2], s[sreg_wval+2*10+2]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+3], s[sreg_wval+2*10+3]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+0], s[sreg_wval+4*10+0+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+1], s[sreg_wval+4*10+1+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+2], s[sreg_wval+4*10+2+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+3], s[sreg_wval+4*10+3+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+0], s[sreg_wval+2*10+0+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+1], s[sreg_wval+2*10+1+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+2], s[sreg_wval+2*10+2+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+3], s[sreg_wval+2*10+3+48]
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(8*136+8)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(9*136+0)
    s_waitcnt lgkmcnt(2)
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+4], s[sreg_wval+4*10+4]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+5], s[sreg_wval+4*10+5]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+6], s[sreg_wval+4*10+6]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+7], s[sreg_wval+4*10+7]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+4], s[sreg_wval+2*10+4]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+5], s[sreg_wval+2*10+5]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+6], s[sreg_wval+2*10+6]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+7], s[sreg_wval+2*10+7]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+4], s[sreg_wval+4*10+4+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+5], s[sreg_wval+4*10+5+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+6], s[sreg_wval+4*10+6+48]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+7], s[sreg_wval+4*10+7+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+4], s[sreg_wval+2*10+4+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+5], s[sreg_wval+2*10+5+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+6], s[sreg_wval+2*10+6+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+7], s[sreg_wval+2*10+7+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+4, vreg_inp_dsrd0, 4*(9*136+2)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(9*136+2)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(9*136+4)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+0], s[sreg_wval+4*10+8+48]
    v_mac_f32 v[vreg_oval+2], v[vreg_ival+1], s[sreg_wval+4*10+9+48]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+0], s[sreg_wval+2*10+8]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+1], s[sreg_wval+2*10+9]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+2], s[sreg_wval+3*10+0]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+3], s[sreg_wval+3*10+1]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+0], s[sreg_wval+4*10+8+50]
    v_mac_f32 v[vreg_oval+6], v[vreg_ival+1], s[sreg_wval+4*10+9+50]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+0], s[sreg_wval+2*10+8+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+1], s[sreg_wval+2*10+9+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+2], s[sreg_wval+3*10+0+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+3], s[sreg_wval+3*10+1+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+0, vreg_inp_dsrd0, 4*(9*136+6)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(9*136+6)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(9*136+8)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+4], s[sreg_wval+3*10+2]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+5], s[sreg_wval+3*10+3]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+6], s[sreg_wval+3*10+4]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+7], s[sreg_wval+3*10+5]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+4], s[sreg_wval+3*10+2+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+5], s[sreg_wval+3*10+3+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+6], s[sreg_wval+3*10+4+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+7], s[sreg_wval+3*10+5+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+4, vreg_inp_dsrd0, 4*(10*136+0)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(10*136+0)
    ds_read_b64 v[vreg_ival+6:vreg_ival+7], v[vreg_inp_dsrd0] offset:4*(10*136+2)
    .endif
    s_waitcnt lgkmcnt(2)
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+0], s[sreg_wval+3*10+6]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+1], s[sreg_wval+3*10+7]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+2], s[sreg_wval+3*10+8]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+3], s[sreg_wval+3*10+9]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+0], s[sreg_wval+3*10+6+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+1], s[sreg_wval+3*10+7+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+2], s[sreg_wval+3*10+8+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+3], s[sreg_wval+3*10+9+48]
    .if use_ds_read_b128
    macro_ds_read_b128 vreg_ival+0, vreg_inp_dsrd0, 4*(10*136+4)
    s_waitcnt lgkmcnt(1)
    .else
    ds_read_b64 v[vreg_ival+0:vreg_ival+1], v[vreg_inp_dsrd0] offset:4*(10*136+4)
    ds_read_b64 v[vreg_ival+2:vreg_ival+3], v[vreg_inp_dsrd0] offset:4*(10*136+6)
    s_waitcnt lgkmcnt(2)
    .endif
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+4], s[sreg_wval+4*10+0]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+5], s[sreg_wval+4*10+1]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+6], s[sreg_wval+4*10+2]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+7], s[sreg_wval+4*10+3]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+4], s[sreg_wval+4*10+0+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+5], s[sreg_wval+4*10+1+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+6], s[sreg_wval+4*10+2+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+7], s[sreg_wval+4*10+3+48]
    ds_read_b64 v[vreg_ival+4:vreg_ival+5], v[vreg_inp_dsrd0] offset:4*(10*136+8)
    s_waitcnt lgkmcnt(1)
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+0], s[sreg_wval+4*10+4]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+1], s[sreg_wval+4*10+5]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+2], s[sreg_wval+4*10+6]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+3], s[sreg_wval+4*10+7]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+0], s[sreg_wval+4*10+4+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+1], s[sreg_wval+4*10+5+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+2], s[sreg_wval+4*10+6+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+3], s[sreg_wval+4*10+7+48]
    s_waitcnt lgkmcnt(0)
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+4], s[sreg_wval+4*10+8+48]
    v_mac_f32 v[vreg_oval+3], v[vreg_ival+5], s[sreg_wval+4*10+9+48]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+4], s[sreg_wval+4*10+8+50]
    v_mac_f32 v[vreg_oval+7], v[vreg_ival+5], s[sreg_wval+4*10+9+50]

    // loop if more channels needs to be processed
    v_readlane_b32 s[sreg_c], v[vreg_save], 0+lane_vreg_save_c
    s_sub_u32  s[sreg_c], s[sreg_c], 1
    v_writelane_b32 v[vreg_save], s[sreg_c], 0+lane_vreg_save_c
    s_cmp_gt_u32 s[sreg_c], 0
    s_cbranch_scc1 loop_channel

    //////////////////////////////////////////////////////////////////////////////
    // write output values
    //  - do bound checks before writing
    //  - use s[sreg_wei_addr:sreg_wei_addr+1] as temporary registers
    v_readlane_b32 s[sreg_k], v[vreg_save], 0+lane_vreg_save_k
    v_readlane_b32 s[sreg_dy], v[vreg_save], 0+lane_vreg_save_dy
    v_readlane_b32 s[sreg_wval+0], v[vreg_save], 0+lane_vreg_save_out_addr0
    v_readlane_b32 s[sreg_wval+1], v[vreg_save], 0+lane_vreg_save_out_addr1
    s_mov_b32      s[sreg_wval+2], 0+out_stride_k*2
    s_mov_b32      s[sreg_wval+3], 0x00020000
    s_mov_b32      s[sreg_wval+4], 0+out_stride_k
    v_cmpx_gt_u32 vcc, 0+out_w, v[vreg_dx]
    v_lshlrev_b32 v[vreg_dx], 2, v[vreg_dx]
    s_cmpk_ge_u32 s[sreg_k], 0+wei_k
    s_cbranch_scc1 skip_write
    buffer_store_dword v[vreg_oval+0], v[vreg_dx], s[sreg_wval+0:sreg_wval+3], 0              offen offset:0*out_stride_y
    buffer_store_dword v[vreg_oval+4], v[vreg_dx], s[sreg_wval+0:sreg_wval+3], s[sreg_wval+4] offen offset:0*out_stride_y
    s_cmpk_ge_i32 s[sreg_dy], 0+out_h-1
    s_cbranch_scc1 skip_write
    buffer_store_dword v[vreg_oval+1], v[vreg_dx], s[sreg_wval+0:sreg_wval+3], 0              offen offset:1*out_stride_y
    buffer_store_dword v[vreg_oval+5], v[vreg_dx], s[sreg_wval+0:sreg_wval+3], s[sreg_wval+4] offen offset:1*out_stride_y
    s_cmpk_ge_i32 s[sreg_dy], 0+out_h-2
    s_cbranch_scc1 skip_write
    buffer_store_dword v[vreg_oval+2], v[vreg_dx], s[sreg_wval+0:sreg_wval+3], 0              offen offset:2*out_stride_y
    buffer_store_dword v[vreg_oval+6], v[vreg_dx], s[sreg_wval+0:sreg_wval+3], s[sreg_wval+4] offen offset:2*out_stride_y
    s_cmpk_ge_i32 s[sreg_dy], 0+out_h-3
    s_cbranch_scc1 skip_write
    buffer_store_dword v[vreg_oval+3], v[vreg_dx], s[sreg_wval+0:sreg_wval+3], 0              offen offset:3*out_stride_y
    buffer_store_dword v[vreg_oval+7], v[vreg_dx], s[sreg_wval+0:sreg_wval+3], s[sreg_wval+4] offen offset:3*out_stride_y
skip_write:
    s_endpgm

///////////////////////////////////////////////////
// ******* meta-data section of the kernels
///////////////////////////////////////////////////
.ifndef ROCM_METADATA_VERSION
.error "ROCM_METADATA_VERSION must be defined"
.endif
.if ROCM_METADATA_VERSION == 4
.amd_amdgpu_hsa_metadata
{ Version: [ 1, 0 ],
    Kernels:
    - { Name: conv5x10u2v2f1, SymbolName: 'conv5x10u2v2f1@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
        Attrs:
          { ReqdWorkGroupSize: [ 64, 8, 1 ] }
        CodeProps:
          { KernargSegmentSize: 56, GroupSegmentFixedSize: 0, PrivateSegmentFixedSize: 0, KernargSegmentAlign: 8, WavefrontSize: 64, MaxFlatWorkGroupSize: 256 }
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
.endif
.if ROCM_METADATA_VERSION == 3
.amdgpu_code_object_metadata
{ Version: [ 3, 0 ],
    Kernels:
    - { Name: conv5x10u2v2f1, Language: OpenCL C, LanguageVersion: [ 1, 2 ],
        Attrs:
          { ReqdWorkGroupSize: [ 64, 8, 1 ] }
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
.end_amdgpu_code_object_metadata
.endif
.if ROCM_METADATA_VERSION == 2
.amdgpu_runtime_metadata
{ amd.MDVersion: [ 2, 1 ],
    amd.Kernels:
    - { amd.KernelName: conv5x10u2v2f1, amd.Language: OpenCL C, amd.LanguageVersion: [ 1, 2 ],
        amd.ReqdWorkGroupSize: [ 64, 8, 1 ], 
        amd.Args:
        - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 8, amd.ArgTypeName: 'float*', amd.ArgName: in,          amd.ArgAddrQual: 1, amd.ArgAccQual: 0, amd.ArgIsConst: 1 }
        - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 8, amd.ArgTypeName: 'float*', amd.ArgName: weights,     amd.ArgAddrQual: 1, amd.ArgAccQual: 0, amd.ArgIsConst: 1 }
        - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 8, amd.ArgTypeName: 'float*', amd.ArgName: out,         amd.ArgAddrQual: 1, amd.ArgAccQual: 0 }
        - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 8, amd.ArgTypeName:  float,   amd.ArgName: padding_val,                     amd.ArgAccQual: 0 }
      }
}
.end_amdgpu_runtime_metadata
.endif
.if ROCM_METADATA_VERSION == 1
.section .note
    // old ROCm metadata
    .long 4
    .long .Lmeta_end - .Lmeta_begin
    .long 7
    .asciz "AMD"
    .p2align 2
    .Lmeta_begin:
    .long  0x02010001, 0x00780300
    .short 0x0604, 14, 0
    .ascii "conv5x10u2v2f1"
    .long  0x00080907, 0x080a0000, 0x0b000000, 0x00000006
    .long  0x616f6c66, 0x030c2a74, 0x69000000, 0x010d706e
    .long  0x1000080e, 0x08010f00, 0x00080907, 0x080a0000
    .long  0x0b000000, 0x00000006, 0x616f6c66, 0x070c2a74
    .long  0x77000000, 0x68676965, 0x010d7374, 0x1000080e
    .long  0x08010f00, 0x00080907, 0x080a0000, 0x0b000000
    .long  0x00000006, 0x616f6c66, 0x030c2a74, 0x6f000000
    .long  0x010d7475, 0x1000080e, 0x08010f00, 0x00040907
    .long  0x040a0000, 0x0b000000, 0x00000005, 0x616f6c66
    .long  0x000b0c74, 0x61700000, 0x6e696464, 0x61765f67
    .long  0x0e000d6c, 0x00100008, 0x08090708, 0x0a000000
    .long  0x00000008, 0x090e070d, 0x09070800, 0x00000008
    .long  0x0000080a, 0x0e080d00, 0x07080009, 0x00000809
    .long  0x00080a00, 0x090d0000, 0x0800090e, 0x00004015
    .long  0x00000800, 0x00000100, 0x00000500
    .Lmeta_end:
    .p2align 2
.endif
