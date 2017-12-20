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
.include "inst_wrappers.inc"

.hsa_code_object_version 2,1
.hsa_code_object_isa
.if (.option.machine_version_major != 8) && (.option.machine_version_major != 9)
.error "ERROR: specified target machine not supported"
.endif

///////////////////////////////////////////////////
// ******* global-work and work-group-size
//  work-group-size = [64, 8, 1]
//  global-work = [alignUp(out_w,64), (alignUp(out_h,4)/4)*alignUp(wei_c/2,8), batch_n]
//    * def alignUp(a,b) = ((a + b - 1)/b)*b
//    * def out_w = (inp_w + 2*pad_w + inp_u - wei_w) / inp_u
//    * def out_h = (inp_h + 2*pad_h + inp_v - wei_h) / inp_v
///////////////////////////////////////////////////
// ******* changeable configuration parameters
//   inp_w          - input tensor width
//   inp_h          - input tensor height
//   wei_c          - input tensor channels (must be multiple of 16)
//   wei_k          - output tensor channels
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
.if (wei_c % 16) != 0
.error "ERROR: wei_c must be multiple of 16"
.endif
.if wei_k < 8
.error "ERROR: wei_k must be >= 8"
.endif
.if inp_w < 138
.error "ERROR: inp_w must be >= 138"
.endif
.if inp_h < 16
.error "ERROR: inp_w must be >= 16"
.endif
// ******* fixed configuration parameters
.set wei_w       ,  10
.set wei_h       ,   5
.set inp_u       ,   2
.set inp_v       ,   2
.set pad_w       ,   0
.set pad_h       ,   0
// ******* derived constants
.set out_w       ,(inp_w + 2*pad_w + inp_u - wei_w) / inp_u
.set out_h       ,(inp_h + 2*pad_h + inp_v - wei_h) / inp_v
.set inp_stride_y,(inp_w * 4)
.set inp_stride_c,(inp_h * inp_stride_y)
.set inp_stride_n,(wei_c * inp_stride_c)
.set out_stride_y,(out_w * 4)
.set out_stride_k,(out_h * out_stride_y)
.set out_stride_n,(wei_k * out_stride_k)
.set wei_stride_c,(wei_h * wei_w * 4)
.set wei_stride_k,(wei_c * wei_stride_c)
.macro .bitcount, n, bits
  .if (1 << \bits) < \n
    .set \bits, \bits + 1
    .bitcount wei_c, wei_c_bits
  .endif
.endm
.set wei_c_bits  , 0
.bitcount wei_c  , wei_c_bits
.set wei_c_mask  , ((1 << wei_c_bits) - 1)

.text
.p2align 8
.global conv5x10u2v2b1
.type conv5x10u2v2b1, @function
.amdgpu_hsa_kernel conv5x10u2v2b1
conv5x10u2v2b1:
   .amd_kernel_code_t
      amd_machine_version_major = .option.machine_version_major
      amd_machine_version_minor = .option.machine_version_minor
      amd_machine_version_stepping = .option.machine_version_stepping
      is_ptr64 = 1
      float_mode = 0
      user_sgpr_count = 2
      is_xnack_enabled = 0
      enable_sgpr_workgroup_id_x = 1
      enable_sgpr_workgroup_id_y = 1
      enable_sgpr_workgroup_id_z = 1
      enable_vgpr_workitem_id = 1
      enable_sgpr_kernarg_segment_ptr = 1
      workitem_vgpr_count = 40
      wavefront_sgpr_count = 108
      workgroup_group_segment_byte_size = 3072
      kernarg_segment_byte_size = 56
      granulated_workitem_vgpr_count = 9
      granulated_wavefront_sgpr_count = 13
  .end_amd_kernel_code_t
  s_mov_b32 m0, 0x00000c00
  s_load_dwordx2 s[16:17], s[0:1], 0
  s_load_dwordx2 s[100:101], s[0:1], 8
  s_load_dwordx2 s[12:13], s[0:1], 16
  s_mul_i32 s22, s2, 0x00000200
  s_lshl_b32 s8, s2, 6
 _v_add_co_u32 v12, vcc, s8, v0
 _v_subrev_co_u32 v12, vcc, 4, v12
  s_lshr_b32 s33, s3, wei_c_bits-4
  s_mul_i32 s23, s33, 8
  s_mul_i32 s33, s33, 4
 _v_add_co_u32 v13, vcc, s33, v1
 _v_subrev_co_u32 v13, vcc, 2, v13
  v_readfirstlane_b32 s21, v1
  s_lshl_b32 s8, s3, 3
  s_add_u32 s21, s21, s8
  s_mul_i32 s21, s21, 2
  s_and_b32 s21, s21, wei_c_mask
  s_mul_i32 s5, s21, wei_stride_c
  s_mul_i32 s32, s4, inp_stride_n
  s_mul_i32 s9, s21, inp_stride_c
  s_add_u32 s32, s32, s9
  v_lshlrev_b32 v6, 2, v0
  s_movk_i32 s8, 512
  v_mad_u32_u24 v5, v1, s8, v6
  v_lshlrev_b32 v4, 2, v12
  v_mov_b32 v8, 0+out_stride_y
  v_mad_u32_u24 v4, v13, v8, v4
  v_mov_b32 v8, 0+out_w
  v_mov_b32 v9, 0+out_h
  v_mov_b32 v10, 0+out_w-64
  v_cmp_gt_i32 vcc, 6, v1
  s_mov_b64 s[6:7], vcc
  v_cmp_le_i32 vcc, 0, v12
  s_and_b64 s[24:25], s[6:7], vcc
  v_cmp_gt_i32 vcc, v8, v12
  s_and_b64 s[24:25], s[24:25], vcc
  v_cmp_le_i32 vcc, 0, v13
  s_and_b64 s[24:25], s[24:25], vcc
  v_cmp_gt_i32 vcc, v9, v13
  s_and_b64 s[24:25], s[24:25], vcc
  v_cmp_le_i32 vcc, 0xffffffc0, v12
  s_and_b64 s[26:27], s[6:7], vcc
  v_cmp_gt_i32 vcc, v10, v12
  s_and_b64 s[26:27], s[26:27], vcc
  v_cmp_le_i32 vcc, 0, v13
  s_and_b64 s[26:27], s[26:27], vcc
  v_cmp_gt_i32 vcc, v9, v13
  s_and_b64 s[26:27], s[26:27], vcc
  s_waitcnt lgkmcnt(0)
  s_add_u32 s100, s100, s5
  s_addc_u32 s101, s101, 0
  s_add_u32 s12, s12, s32
  s_addc_u32 s13, s13, 0
  s_mul_i32 s8, s4, out_stride_n
  s_add_u32 s16, s16, s8
  s_addc_u32 s17, s17, 0
  v_mov_b32 v8, 0
  v_mov_b32 v9, 0
  v_mov_b32 v10, 0
  v_mov_b32 v11, 0
  v_mov_b32 v12, 0
  v_mov_b32 v13, 0
  v_mov_b32 v14, 0
  v_mov_b32 v15, 0
  v_mov_b32 v16, 0
  v_mov_b32 v17, 0
  v_mov_b32 v18, 0
  v_mov_b32 v19, 0
  v_mov_b32 v20, 0
  v_mov_b32 v21, 0
  v_mov_b32 v22, 0
  v_mov_b32 v23, 0
  v_mov_b32 v24, 0
  v_mov_b32 v25, 0
  v_mov_b32 v26, 0
  v_mov_b32 v27, 0
  v_mov_b32 v28, 0
  v_mov_b32 v29, 0
  v_mov_b32 v30, 0
  v_mov_b32 v31, 0
  v_mov_b32 v32, 0
  v_mov_b32 v33, 0
  v_mov_b32 v34, 0
  v_mov_b32 v35, 0
  v_mov_b32 v36, 0
  v_mov_b32 v37, 0
  v_mov_b32 v38, 0
  v_mov_b32 v39, 0
  s_movk_i32 s20, 0+wei_k
  v_writelane_b32 v7, s6, 0
  v_writelane_b32 v7, s7, 1
  v_writelane_b32 v7, s21, 2
  v_writelane_b32 v7, s20, 3
  v_writelane_b32 v7, s23, 4
  v_writelane_b32 v7, s22, 5
  v_writelane_b32 v7, s16, 10
  v_writelane_b32 v7, s17, 11
  v_writelane_b32 v7, s12, 6
  v_writelane_b32 v7, s13, 7
  v_writelane_b32 v7, s24, 14
  v_writelane_b32 v7, s25, 15
  v_writelane_b32 v7, s26, 16
  v_writelane_b32 v7, s27, 17
loop_channel:
  v_readlane_b32 s16, v7, 10
  v_readlane_b32 s17, v7, 11
  s_mov_b32 s18, out_stride_n
  s_mov_b32 s19, 0x00020000
  v_readlane_b32 s24, v7, 14
  v_readlane_b32 s25, v7, 15
  v_readlane_b32 s26, v7, 16
  v_readlane_b32 s27, v7, 17
  v_mov_b32 v0, 0
  v_mov_b32 v1, 0
  s_mov_b64 exec, s[24:25]
  buffer_load_dword v0, v4, s[16:19], 0 offen offset:0
  s_mov_b64 exec, s[26:27]
  buffer_load_dword v1, v4, s[16:19], 0 offen offset:256
  s_mov_b64 exec, -1
  s_mov_b32 s8, out_stride_k
 _v_add_co_u32 v4, vcc, s8, v4
  v_readlane_b32 s6, v7, 0
  v_readlane_b32 s7, v7, 1
  s_waitcnt vmcnt(0) lgkmcnt(0)
  s_barrier
  s_mov_b64 exec, s[6:7]
  ds_write_b32 v5, v0 offset:0
  ds_write_b32 v5, v1 offset:256
  s_mov_b64 exec, -1
  s_waitcnt lgkmcnt(0)
  s_barrier
  s_load_dwordx16 s[0:15], s[100:101], 0
  s_load_dwordx16 s[16:31], s[100:101], 64
  s_load_dwordx16 s[32:47], s[100:101], 0x00000080
  s_load_dwordx2 s[96:97], s[100:101], 0x000000c0
  s_add_u32 s100, s100, 0x000000c8
  s_addc_u32 s101, s101, 0
  s_load_dwordx16 s[48:63], s[100:101], 0
  s_load_dwordx16 s[64:79], s[100:101], 64
  s_load_dwordx16 s[80:95], s[100:101], 0x00000080
  s_load_dwordx2 s[98:99], s[100:101], 0x000000c0
  s_add_u32 s100, s100, wei_stride_k-wei_stride_c
  s_addc_u32 s101, s101, 0
  s_waitcnt vmcnt(0) lgkmcnt(0)
  ds_read_b32 v0, v6 offset:0
  ds_read_b32 v1, v6 offset:4
  ds_read_b32 v2, v6 offset:8
  ds_read_b32 v3, v6 offset:12
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v0, s96
  v_mac_f32 v8, v1, s46
  v_mac_f32 v9, v0, s97
  v_mac_f32 v9, v1, s47
  v_mac_f32 v24, v0, s98
  v_mac_f32 v24, v1, s94
  v_mac_f32 v25, v0, s99
  v_mac_f32 v25, v1, s95
  ds_read_b32 v0, v6 offset:16
  ds_read_b32 v1, v6 offset:512
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v2, s44
  v_mac_f32 v8, v3, s42
  v_mac_f32 v9, v2, s45
  v_mac_f32 v9, v3, s43
  v_mac_f32 v24, v2, s92
  v_mac_f32 v24, v3, s90
  v_mac_f32 v25, v2, s93
  v_mac_f32 v25, v3, s91
  ds_read_b32 v2, v6 offset:516
  ds_read_b32 v3, v6 offset:520
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v0, s40
  v_mac_f32 v8, v1, s28
  v_mac_f32 v9, v0, s41
  v_mac_f32 v9, v1, s29
  v_mac_f32 v10, v1, s38
  v_mac_f32 v11, v1, s39
  v_mac_f32 v12, v1, s96
  v_mac_f32 v13, v1, s97
  v_mac_f32 v24, v0, s88
  v_mac_f32 v24, v1, s76
  v_mac_f32 v25, v0, s89
  v_mac_f32 v25, v1, s77
  v_mac_f32 v26, v1, s86
  v_mac_f32 v27, v1, s87
  v_mac_f32 v28, v1, s98
  v_mac_f32 v29, v1, s99
  ds_read_b32 v0, v6 offset:524
  ds_read_b32 v1, v6 offset:528
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v2, s26
  v_mac_f32 v8, v3, s24
  v_mac_f32 v9, v2, s27
  v_mac_f32 v9, v3, s25
  v_mac_f32 v10, v2, s36
  v_mac_f32 v10, v3, s34
  v_mac_f32 v11, v2, s37
  v_mac_f32 v11, v3, s35
  v_mac_f32 v12, v2, s46
  v_mac_f32 v12, v3, s44
  v_mac_f32 v13, v2, s47
  v_mac_f32 v13, v3, s45
  v_mac_f32 v24, v2, s74
  v_mac_f32 v24, v3, s72
  v_mac_f32 v25, v2, s75
  v_mac_f32 v25, v3, s73
  v_mac_f32 v26, v2, s84
  v_mac_f32 v26, v3, s82
  v_mac_f32 v27, v2, s85
  v_mac_f32 v27, v3, s83
  v_mac_f32 v28, v2, s94
  v_mac_f32 v28, v3, s92
  v_mac_f32 v29, v2, s95
  v_mac_f32 v29, v3, s93
  ds_read_b32 v2, v6 offset:1024
  ds_read_b32 v3, v6 offset:1028
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v0, s22
  v_mac_f32 v8, v1, s20
  v_mac_f32 v9, v0, s23
  v_mac_f32 v9, v1, s21
  v_mac_f32 v10, v0, s32
  v_mac_f32 v10, v1, s30
  v_mac_f32 v11, v0, s33
  v_mac_f32 v11, v1, s31
  v_mac_f32 v12, v0, s42
  v_mac_f32 v12, v1, s40
  v_mac_f32 v13, v0, s43
  v_mac_f32 v13, v1, s41
  v_mac_f32 v24, v0, s70
  v_mac_f32 v24, v1, s68
  v_mac_f32 v25, v0, s71
  v_mac_f32 v25, v1, s69
  v_mac_f32 v26, v0, s80
  v_mac_f32 v26, v1, s78
  v_mac_f32 v27, v0, s81
  v_mac_f32 v27, v1, s79
  v_mac_f32 v28, v0, s90
  v_mac_f32 v28, v1, s88
  v_mac_f32 v29, v0, s91
  v_mac_f32 v29, v1, s89
  ds_read_b32 v0, v6 offset:1032
  ds_read_b32 v1, v6 offset:1036
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v2, s8
  v_mac_f32 v8, v3, s6
  v_mac_f32 v9, v2, s9
  v_mac_f32 v9, v3, s7
  v_mac_f32 v10, v2, s18
  v_mac_f32 v10, v3, s16
  v_mac_f32 v11, v2, s19
  v_mac_f32 v11, v3, s17
  v_mac_f32 v12, v2, s28
  v_mac_f32 v12, v3, s26
  v_mac_f32 v13, v2, s29
  v_mac_f32 v13, v3, s27
  v_mac_f32 v14, v2, s38
  v_mac_f32 v14, v3, s36
  v_mac_f32 v15, v2, s39
  v_mac_f32 v15, v3, s37
  v_mac_f32 v16, v2, s96
  v_mac_f32 v16, v3, s46
  v_mac_f32 v17, v2, s97
  v_mac_f32 v17, v3, s47
  v_mac_f32 v24, v2, s56
  v_mac_f32 v24, v3, s54
  v_mac_f32 v25, v2, s57
  v_mac_f32 v25, v3, s55
  v_mac_f32 v26, v2, s66
  v_mac_f32 v26, v3, s64
  v_mac_f32 v27, v2, s67
  v_mac_f32 v27, v3, s65
  v_mac_f32 v28, v2, s76
  v_mac_f32 v28, v3, s74
  v_mac_f32 v29, v2, s77
  v_mac_f32 v29, v3, s75
  v_mac_f32 v30, v2, s86
  v_mac_f32 v30, v3, s84
  v_mac_f32 v31, v2, s87
  v_mac_f32 v31, v3, s85
  v_mac_f32 v32, v2, s98
  v_mac_f32 v32, v3, s94
  v_mac_f32 v33, v2, s99
  v_mac_f32 v33, v3, s95
  ds_read_b32 v2, v6 offset:1040
  ds_read_b32 v3, v6 offset:1536
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v0, s4
  v_mac_f32 v8, v1, s2
  v_mac_f32 v9, v0, s5
  v_mac_f32 v9, v1, s3
  v_mac_f32 v10, v0, s14
  v_mac_f32 v10, v1, s12
  v_mac_f32 v11, v0, s15
  v_mac_f32 v11, v1, s13
  v_mac_f32 v12, v0, s24
  v_mac_f32 v12, v1, s22
  v_mac_f32 v13, v0, s25
  v_mac_f32 v13, v1, s23
  v_mac_f32 v14, v0, s34
  v_mac_f32 v14, v1, s32
  v_mac_f32 v15, v0, s35
  v_mac_f32 v15, v1, s33
  v_mac_f32 v16, v0, s44
  v_mac_f32 v16, v1, s42
  v_mac_f32 v17, v0, s45
  v_mac_f32 v17, v1, s43
  v_mac_f32 v24, v0, s52
  v_mac_f32 v24, v1, s50
  v_mac_f32 v25, v0, s53
  v_mac_f32 v25, v1, s51
  v_mac_f32 v26, v0, s62
  v_mac_f32 v26, v1, s60
  v_mac_f32 v27, v0, s63
  v_mac_f32 v27, v1, s61
  v_mac_f32 v28, v0, s72
  v_mac_f32 v28, v1, s70
  v_mac_f32 v29, v0, s73
  v_mac_f32 v29, v1, s71
  v_mac_f32 v30, v0, s82
  v_mac_f32 v30, v1, s80
  v_mac_f32 v31, v0, s83
  v_mac_f32 v31, v1, s81
  v_mac_f32 v32, v0, s92
  v_mac_f32 v32, v1, s90
  v_mac_f32 v33, v0, s93
  v_mac_f32 v33, v1, s91
  ds_read_b32 v0, v6 offset:1540
  ds_read_b32 v1, v6 offset:1544
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v2, s0
  v_mac_f32 v9, v2, s1
  v_mac_f32 v10, v2, s10
  v_mac_f32 v11, v2, s11
  v_mac_f32 v12, v2, s20
  v_mac_f32 v12, v3, s8
  v_mac_f32 v13, v2, s21
  v_mac_f32 v13, v3, s9
  v_mac_f32 v14, v2, s30
  v_mac_f32 v14, v3, s18
  v_mac_f32 v15, v2, s31
  v_mac_f32 v15, v3, s19
  v_mac_f32 v16, v2, s40
  v_mac_f32 v16, v3, s28
  v_mac_f32 v17, v2, s41
  v_mac_f32 v17, v3, s29
  v_mac_f32 v18, v3, s38
  v_mac_f32 v19, v3, s39
  v_mac_f32 v20, v3, s96
  v_mac_f32 v21, v3, s97
  v_mac_f32 v24, v2, s48
  v_mac_f32 v25, v2, s49
  v_mac_f32 v26, v2, s58
  v_mac_f32 v27, v2, s59
  v_mac_f32 v28, v2, s68
  v_mac_f32 v28, v3, s56
  v_mac_f32 v29, v2, s69
  v_mac_f32 v29, v3, s57
  v_mac_f32 v30, v2, s78
  v_mac_f32 v30, v3, s66
  v_mac_f32 v31, v2, s79
  v_mac_f32 v31, v3, s67
  v_mac_f32 v32, v2, s88
  v_mac_f32 v32, v3, s76
  v_mac_f32 v33, v2, s89
  v_mac_f32 v33, v3, s77
  v_mac_f32 v34, v3, s86
  v_mac_f32 v35, v3, s87
  v_mac_f32 v36, v3, s98
  v_mac_f32 v37, v3, s99
  ds_read_b32 v2, v6 offset:1548
  ds_read_b32 v3, v6 offset:1552
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v12, v0, s6
  v_mac_f32 v12, v1, s4
  v_mac_f32 v13, v0, s7
  v_mac_f32 v13, v1, s5
  v_mac_f32 v14, v0, s16
  v_mac_f32 v14, v1, s14
  v_mac_f32 v15, v0, s17
  v_mac_f32 v15, v1, s15
  v_mac_f32 v16, v0, s26
  v_mac_f32 v16, v1, s24
  v_mac_f32 v17, v0, s27
  v_mac_f32 v17, v1, s25
  v_mac_f32 v18, v0, s36
  v_mac_f32 v18, v1, s34
  v_mac_f32 v19, v0, s37
  v_mac_f32 v19, v1, s35
  v_mac_f32 v20, v0, s46
  v_mac_f32 v20, v1, s44
  v_mac_f32 v21, v0, s47
  v_mac_f32 v21, v1, s45
  v_mac_f32 v28, v0, s54
  v_mac_f32 v28, v1, s52
  v_mac_f32 v29, v0, s55
  v_mac_f32 v29, v1, s53
  v_mac_f32 v30, v0, s64
  v_mac_f32 v30, v1, s62
  v_mac_f32 v31, v0, s65
  v_mac_f32 v31, v1, s63
  v_mac_f32 v32, v0, s74
  v_mac_f32 v32, v1, s72
  v_mac_f32 v33, v0, s75
  v_mac_f32 v33, v1, s73
  v_mac_f32 v34, v0, s84
  v_mac_f32 v34, v1, s82
  v_mac_f32 v35, v0, s85
  v_mac_f32 v35, v1, s83
  v_mac_f32 v36, v0, s94
  v_mac_f32 v36, v1, s92
  v_mac_f32 v37, v0, s95
  v_mac_f32 v37, v1, s93
  ds_read_b32 v0, v6 offset:2048
  ds_read_b32 v1, v6 offset:2052
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v12, v2, s2
  v_mac_f32 v12, v3, s0
  v_mac_f32 v13, v2, s3
  v_mac_f32 v13, v3, s1
  v_mac_f32 v14, v2, s12
  v_mac_f32 v14, v3, s10
  v_mac_f32 v15, v2, s13
  v_mac_f32 v15, v3, s11
  v_mac_f32 v16, v2, s22
  v_mac_f32 v16, v3, s20
  v_mac_f32 v17, v2, s23
  v_mac_f32 v17, v3, s21
  v_mac_f32 v18, v2, s32
  v_mac_f32 v18, v3, s30
  v_mac_f32 v19, v2, s33
  v_mac_f32 v19, v3, s31
  v_mac_f32 v20, v2, s42
  v_mac_f32 v20, v3, s40
  v_mac_f32 v21, v2, s43
  v_mac_f32 v21, v3, s41
  v_mac_f32 v28, v2, s50
  v_mac_f32 v28, v3, s48
  v_mac_f32 v29, v2, s51
  v_mac_f32 v29, v3, s49
  v_mac_f32 v30, v2, s60
  v_mac_f32 v30, v3, s58
  v_mac_f32 v31, v2, s61
  v_mac_f32 v31, v3, s59
  v_mac_f32 v32, v2, s70
  v_mac_f32 v32, v3, s68
  v_mac_f32 v33, v2, s71
  v_mac_f32 v33, v3, s69
  v_mac_f32 v34, v2, s80
  v_mac_f32 v34, v3, s78
  v_mac_f32 v35, v2, s81
  v_mac_f32 v35, v3, s79
  v_mac_f32 v36, v2, s90
  v_mac_f32 v36, v3, s88
  v_mac_f32 v37, v2, s91
  v_mac_f32 v37, v3, s89
  ds_read_b32 v2, v6 offset:2056
  ds_read_b32 v3, v6 offset:2060
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v16, v0, s8
  v_mac_f32 v16, v1, s6
  v_mac_f32 v17, v0, s9
  v_mac_f32 v17, v1, s7
  v_mac_f32 v18, v0, s18
  v_mac_f32 v18, v1, s16
  v_mac_f32 v19, v0, s19
  v_mac_f32 v19, v1, s17
  v_mac_f32 v20, v0, s28
  v_mac_f32 v20, v1, s26
  v_mac_f32 v21, v0, s29
  v_mac_f32 v21, v1, s27
  v_mac_f32 v22, v0, s38
  v_mac_f32 v22, v1, s36
  v_mac_f32 v23, v0, s39
  v_mac_f32 v23, v1, s37
  v_mac_f32 v32, v0, s56
  v_mac_f32 v32, v1, s54
  v_mac_f32 v33, v0, s57
  v_mac_f32 v33, v1, s55
  v_mac_f32 v34, v0, s66
  v_mac_f32 v34, v1, s64
  v_mac_f32 v35, v0, s67
  v_mac_f32 v35, v1, s65
  v_mac_f32 v36, v0, s76
  v_mac_f32 v36, v1, s74
  v_mac_f32 v37, v0, s77
  v_mac_f32 v37, v1, s75
  v_mac_f32 v38, v0, s86
  v_mac_f32 v38, v1, s84
  v_mac_f32 v39, v0, s87
  v_mac_f32 v39, v1, s85
  ds_read_b32 v0, v6 offset:2064
  ds_read_b32 v1, v6 offset:2560
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v16, v2, s4
  v_mac_f32 v16, v3, s2
  v_mac_f32 v17, v2, s5
  v_mac_f32 v17, v3, s3
  v_mac_f32 v18, v2, s14
  v_mac_f32 v18, v3, s12
  v_mac_f32 v19, v2, s15
  v_mac_f32 v19, v3, s13
  v_mac_f32 v20, v2, s24
  v_mac_f32 v20, v3, s22
  v_mac_f32 v21, v2, s25
  v_mac_f32 v21, v3, s23
  v_mac_f32 v22, v2, s34
  v_mac_f32 v22, v3, s32
  v_mac_f32 v23, v2, s35
  v_mac_f32 v23, v3, s33
  v_mac_f32 v32, v2, s52
  v_mac_f32 v32, v3, s50
  v_mac_f32 v33, v2, s53
  v_mac_f32 v33, v3, s51
  v_mac_f32 v34, v2, s62
  v_mac_f32 v34, v3, s60
  v_mac_f32 v35, v2, s63
  v_mac_f32 v35, v3, s61
  v_mac_f32 v36, v2, s72
  v_mac_f32 v36, v3, s70
  v_mac_f32 v37, v2, s73
  v_mac_f32 v37, v3, s71
  v_mac_f32 v38, v2, s82
  v_mac_f32 v38, v3, s80
  v_mac_f32 v39, v2, s83
  v_mac_f32 v39, v3, s81
  ds_read_b32 v2, v6 offset:2564
  ds_read_b32 v3, v6 offset:2568
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v16, v0, s0
  v_mac_f32 v17, v0, s1
  v_mac_f32 v18, v0, s10
  v_mac_f32 v19, v0, s11
  v_mac_f32 v20, v0, s20
  v_mac_f32 v20, v1, s8
  v_mac_f32 v21, v0, s21
  v_mac_f32 v21, v1, s9
  v_mac_f32 v22, v0, s30
  v_mac_f32 v22, v1, s18
  v_mac_f32 v23, v0, s31
  v_mac_f32 v23, v1, s19
  v_mac_f32 v32, v0, s48
  v_mac_f32 v33, v0, s49
  v_mac_f32 v34, v0, s58
  v_mac_f32 v35, v0, s59
  v_mac_f32 v36, v0, s68
  v_mac_f32 v36, v1, s56
  v_mac_f32 v37, v0, s69
  v_mac_f32 v37, v1, s57
  v_mac_f32 v38, v0, s78
  v_mac_f32 v38, v1, s66
  v_mac_f32 v39, v0, s79
  v_mac_f32 v39, v1, s67
  ds_read_b32 v0, v6 offset:2572
  ds_read_b32 v1, v6 offset:2576
  s_waitcnt lgkmcnt(1)
  v_mac_f32 v20, v2, s6
  v_mac_f32 v20, v3, s4
  v_mac_f32 v20, v0, s2
  v_mac_f32 v21, v2, s7
  v_mac_f32 v21, v3, s5
  v_mac_f32 v21, v0, s3
  v_mac_f32 v22, v2, s16
  v_mac_f32 v22, v3, s14
  v_mac_f32 v22, v0, s12
  v_mac_f32 v23, v2, s17
  v_mac_f32 v23, v3, s15
  v_mac_f32 v23, v0, s13
  v_mac_f32 v36, v2, s54
  v_mac_f32 v36, v3, s52
  v_mac_f32 v36, v0, s50
  v_mac_f32 v37, v2, s55
  v_mac_f32 v37, v3, s53
  v_mac_f32 v37, v0, s51
  v_mac_f32 v38, v2, s64
  v_mac_f32 v38, v3, s62
  v_mac_f32 v38, v0, s60
  v_mac_f32 v39, v2, s65
  v_mac_f32 v39, v3, s63
  v_mac_f32 v39, v0, s61
  s_waitcnt lgkmcnt(0)
  v_mac_f32 v20, v1, s0
  v_mac_f32 v21, v1, s1
  v_mac_f32 v22, v1, s10
  v_mac_f32 v23, v1, s11
  v_mac_f32 v36, v1, s48
  v_mac_f32 v37, v1, s49
  v_mac_f32 v38, v1, s58
  v_mac_f32 v39, v1, s59
  v_readlane_b32 s20, v7, 3
  s_sub_u32 s20, s20, 1
  v_writelane_b32 v7, s20, 3
  s_cmp_gt_u32 s20, 0
  s_cbranch_scc1 loop_channel
  v_readlane_b32 s21, v7, 2
  v_readlane_b32 s23, v7, 4
  v_readlane_b32 s22, v7, 5
  v_readlane_b32 s12, v7, 6
  v_readlane_b32 s13, v7, 7
  s_mov_b32 s14, 2*inp_stride_c
  s_mov_b32 s15, 0x00020000
  v_mul_u32_u24 v0, 2, v6
 _v_add_co_u32 v0, vcc, s22, v0
  v_cmp_gt_u32 vcc, 4 * inp_w, v0
  s_mov_b64 s[0:1], vcc
  v_cmp_gt_u32 vcc, 4 * (inp_w - 1), v0
  s_mov_b64 s[2:3], vcc
  s_mul_i32 s32, s23, inp_stride_y
 _v_add_co_u32 v0, vcc, s32, v0
  s_mov_b32 s8, inp_stride_c
 _v_add_co_u32 v1, vcc, s8, v0
  s_mov_b32 s8, 0
  s_mov_b64 exec, s[0:1]
  buffer_store_dword v8, v0, s[12:15], s8 offen offset:0
  buffer_store_dword v24, v1, s[12:15], s8 offen offset:0
  s_mov_b64 exec, s[2:3]
  buffer_store_dword v9, v0, s[12:15], s8 offen offset:4
  buffer_store_dword v25, v1, s[12:15], s8 offen offset:4
.if (inp_h % 8) == 1
  s_cmp_ge_u32 s23, 0+inp_h-1
  s_cbranch_scc1 skip_write
.endif
  s_mov_b32 s8, 1*inp_stride_y
  s_mov_b64 exec, s[0:1]
  buffer_store_dword v10, v0, s[12:15], s8 offen offset:0
  buffer_store_dword v26, v1, s[12:15], s8 offen offset:0
  s_mov_b64 exec, s[2:3]
  buffer_store_dword v11, v0, s[12:15], s8 offen offset:4
  buffer_store_dword v27, v1, s[12:15], s8 offen offset:4
.if (inp_h % 8) == 2
  s_cmp_ge_u32 s23, 0+inp_h-2
  s_cbranch_scc1 skip_write
.endif
  s_mov_b32 s8, 2*inp_stride_y
  s_mov_b64 exec, s[0:1]
  buffer_store_dword v12, v0, s[12:15], s8 offen offset:0
  buffer_store_dword v28, v1, s[12:15], s8 offen offset:0
  s_mov_b64 exec, s[2:3]
  buffer_store_dword v13, v0, s[12:15], s8 offen offset:4
  buffer_store_dword v29, v1, s[12:15], s8 offen offset:4
.if (inp_h % 8) == 3
  s_cmp_ge_u32 s23, 0+inp_h-3
  s_cbranch_scc1 skip_write
.endif
  s_mov_b32 s8, 3*inp_stride_y
  s_mov_b64 exec, s[0:1]
  buffer_store_dword v14, v0, s[12:15], s8 offen offset:0
  buffer_store_dword v30, v1, s[12:15], s8 offen offset:0
  s_mov_b64 exec, s[2:3]
  buffer_store_dword v15, v0, s[12:15], s8 offen offset:4
  buffer_store_dword v31, v1, s[12:15], s8 offen offset:4
.if (inp_h % 8) == 4
  s_cmp_ge_u32 s23, 0+inp_h-4
  s_cbranch_scc1 skip_write
.endif
  s_mov_b32 s8, 4*inp_stride_y
  s_mov_b64 exec, s[0:1]
  buffer_store_dword v16, v0, s[12:15], s8 offen offset:0
  buffer_store_dword v32, v1, s[12:15], s8 offen offset:0
  s_mov_b64 exec, s[2:3]
  buffer_store_dword v17, v0, s[12:15], s8 offen offset:4
  buffer_store_dword v33, v1, s[12:15], s8 offen offset:4
.if (inp_h % 8) == 5
  s_cmp_ge_u32 s23, 0+inp_h-5
  s_cbranch_scc1 skip_write
.endif
  s_mov_b32 s8, 5*inp_stride_y
  s_mov_b64 exec, s[0:1]
  buffer_store_dword v18, v0, s[12:15], s8 offen offset:0
  buffer_store_dword v34, v1, s[12:15], s8 offen offset:0
  s_mov_b64 exec, s[2:3]
  buffer_store_dword v19, v0, s[12:15], s8 offen offset:4
  buffer_store_dword v35, v1, s[12:15], s8 offen offset:4
.if (inp_h % 8) == 6
  s_cmp_ge_u32 s23, 0+inp_h-6
  s_cbranch_scc1 skip_write
.endif
  s_mov_b32 s8, 6*inp_stride_y
  s_mov_b64 exec, s[0:1]
  buffer_store_dword v20, v0, s[12:15], s8 offen offset:0
  buffer_store_dword v36, v1, s[12:15], s8 offen offset:0
  s_mov_b64 exec, s[2:3]
  buffer_store_dword v21, v0, s[12:15], s8 offen offset:4
  buffer_store_dword v37, v1, s[12:15], s8 offen offset:4
.if (inp_h % 8) == 7
  s_cmp_ge_u32 s23, 0+inp_h-7
  s_cbranch_scc1 skip_write
.endif
  s_mov_b32 s8, 7*inp_stride_y
  s_mov_b64 exec, s[0:1]
  buffer_store_dword v22, v0, s[12:15], s8 offen offset:0
  buffer_store_dword v38, v1, s[12:15], s8 offen offset:0
  s_mov_b64 exec, s[2:3]
  buffer_store_dword v23, v0, s[12:15], s8 offen offset:4
  buffer_store_dword v39, v1, s[12:15], s8 offen offset:4
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
    - { Name: conv5x10u2v2b1, SymbolName: 'conv5x10u2v2b1@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
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
    - { Name: conv5x10u2v2b1, Language: OpenCL C, LanguageVersion: [ 1, 2 ],
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
    .ascii "conv5x10u2v2b1"
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
