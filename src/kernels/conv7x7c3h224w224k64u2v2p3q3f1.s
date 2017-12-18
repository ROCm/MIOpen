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
.text
.p2align 8
.global gcnAsmConv7x7c3h224w224k64u2v2p3q3f1
.type gcnAsmConv7x7c3h224w224k64u2v2p3q3f1, @function
.amdgpu_hsa_kernel gcnAsmConv7x7c3h224w224k64u2v2p3q3f1
gcnAsmConv7x7c3h224w224k64u2v2p3q3f1:
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
      workitem_vgpr_count = 24
      wavefront_sgpr_count = 108
      workgroup_group_segment_byte_size = 7040
      kernarg_segment_byte_size = 56
      granulated_workitem_vgpr_count = 5
      granulated_wavefront_sgpr_count = 13
  .end_amd_kernel_code_t
  s_mov_b32 m0, 7040
  s_load_dwordx2 s[12:13], s[0:1], 0
  s_load_dwordx2 s[100:101], s[0:1], 8
  s_load_dwordx2 s[16:17], s[0:1], 16
  s_load_dword s36, s[0:1], 24
  s_lshl_b32 s8, s2, 6
 _v_add_co_u32 v21, vcc, s8, v0
  s_lshr_b32 s22, s3, 2
  s_lshl_b32 s22, s22, 2
  v_readfirstlane_b32 s20, v1
  s_lshl_b32 s8, s3, 3
  s_add_u32 s20, s20, s8
  s_mul_i32 s20, s20, 2
  s_and_b32 s20, s20, 63
  s_mul_i32 s5, s20, 588
  s_mul_i32 s23, s4, 3211264
  s_mul_i32 s9, s20, 50176
  s_add_u32 s23, s23, s9
  s_mul_i32 s9, s22, 448
  s_add_u32 s23, s23, s9
  v_lshlrev_b32 v7, 1, v0
  v_mov_b32 v8, v1
 _v_subrev_co_u32 v4, vcc, 5, v1
  v_lshlrev_b32 v4, 6, v4
 _v_add_co_u32 v4, vcc, v4, v0
  s_movk_i32 s8, 5462
  v_mul_u32_u24 v10, s8, v4
  v_lshrrev_b32 v10, 14, v10
  v_mul_u32_u24 v9, 3, v10
 _v_sub_co_u32 v9, vcc, v4, v9
  v_lshlrev_b32 v9, 1, v9
 _v_add_co_u32 v9, vcc, 128, v9
 _v_add_co_u32 v3, vcc, 8, v8
  v_cmp_le_u32 vcc, 5, v1
  v_cndmask_b32 v9, v7, v9, vcc
  v_cndmask_b32 v10, v3, v10, vcc
  v_lshlrev_b32 v20, 2, v7
  s_movk_i32 s8, 536
  v_lshlrev_b32 v4, 2, v9
  v_mad_u32_u24 v18, v8, s8, v20
  v_mad_u32_u24 v19, v10, s8, v4
  s_lshl_b32 s37, s2, 7
  s_lshl_b32 s38, s22, 1
  s_sub_u32 s37, s37, 3
  s_sub_u32 s38, s38, 3
 _v_add_co_u32 v11, vcc, s37, v7
 _v_add_co_u32 v12, vcc, s38, v8
 _v_add_co_u32 v13, vcc, s37, v9
 _v_add_co_u32 v14, vcc, s38, v10
  v_max_i32 v16, 0, v11
  v_lshlrev_b32 v16, 2, v16
  v_max_i32 v17, 0, v13
  v_lshlrev_b32 v17, 2, v17
  v_mov_b32 v3, 896
  v_mad_u32_u24 v16, v12, v3, v16
  v_mad_u32_u24 v17, v14, v3, v17
  v_mov_b32 v3, 224
  v_mov_b32 v4, 224
  v_cmp_le_i32 vcc, -1, v11
  s_mov_b64 s[24:25], vcc
  v_cmp_gt_i32 vcc, v3, v11
  s_and_b64 s[24:25], s[24:25], vcc
  v_cmp_le_i32 vcc, 0, v12
  s_and_b64 s[24:25], s[24:25], vcc
  v_cmp_gt_i32 vcc, v4, v12
  s_and_b64 s[24:25], s[24:25], vcc
  v_cmp_gt_i32 vcc, 13, v10
  s_mov_b64 s[6:7], vcc
  v_cmp_le_i32 vcc, -1, v13
  s_and_b64 s[26:27], s[6:7], vcc
  v_cmp_gt_i32 vcc, v3, v13
  s_and_b64 s[26:27], s[26:27], vcc
  v_cmp_le_i32 vcc, 0, v14
  s_and_b64 s[26:27], s[26:27], vcc
  v_cmp_gt_i32 vcc, v4, v14
  s_and_b64 s[26:27], s[26:27], vcc
  v_mov_b32 v3, 223
  v_cmp_eq_i32 vcc, -1, v11
  s_and_b64 s[28:29], s[24:25], vcc
  v_cmp_eq_i32 vcc, v3, v11
  s_and_b64 s[32:33], s[24:25], vcc
  v_cmp_eq_i32 vcc, -1, v13
  s_and_b64 s[30:31], s[26:27], vcc
  v_cmp_eq_i32 vcc, v3, v13
  s_and_b64 s[34:35], s[26:27], vcc
  s_waitcnt lgkmcnt(0)
  s_add_u32 s100, s100, s5
  s_addc_u32 s101, s101, 0
  s_add_u32 s16, s16, s23
  s_addc_u32 s17, s17, 0
  s_mul_i32 s8, s4, 602112
  s_add_u32 s12, s12, s8
  s_addc_u32 s13, s13, 0
  v_mov_b32 v8, 0
  v_mov_b32 v9, 0
  v_mov_b32 v10, 0
  v_mov_b32 v11, 0
  v_mov_b32 v12, 0
  v_mov_b32 v13, 0
  v_mov_b32 v14, 0
  v_mov_b32 v15, 0
  s_movk_i32 s21, 3
  v_writelane_b32 v22, s6, 0
  v_writelane_b32 v22, s7, 1
  v_writelane_b32 v22, s21, 2
  v_writelane_b32 v22, s12, 6
  v_writelane_b32 v22, s13, 7
  v_writelane_b32 v22, s20, 3
  v_writelane_b32 v22, s22, 4
  v_writelane_b32 v22, s16, 10
  v_writelane_b32 v22, s17, 11
  v_writelane_b32 v22, s36, 5
  v_writelane_b32 v22, s24, 14
  v_writelane_b32 v22, s25, 15
  v_writelane_b32 v22, s26, 16
  v_writelane_b32 v22, s27, 17
  v_writelane_b32 v22, s28, 18
  v_writelane_b32 v22, s29, 19
  v_writelane_b32 v22, s30, 22
  v_writelane_b32 v22, s31, 23
  v_writelane_b32 v22, s32, 20
  v_writelane_b32 v22, s33, 21
  v_writelane_b32 v22, s34, 24
  v_writelane_b32 v22, s35, 25
loop_channel:
  v_readlane_b32 s24, v22, 14
  v_readlane_b32 s25, v22, 15
  v_readlane_b32 s26, v22, 16
  v_readlane_b32 s27, v22, 17
  v_readlane_b32 s6, v22, 0
  v_readlane_b32 s7, v22, 1
  v_readlane_b32 s12, v22, 6
  v_readlane_b32 s13, v22, 7
  s_mov_b32 s14, 602112
  s_mov_b32 s15, 0x00020000
  v_readlane_b32 s36, v22, 5
  v_mov_b32 v0, s36
  v_mov_b32 v1, s36
  v_mov_b32 v2, s36
  v_mov_b32 v3, s36
  s_mov_b64 exec, s[24:25]
  buffer_load_dwordx2 v[0:1], v16, s[12:15], 0 offen offset:0
  s_mov_b64 exec, s[26:27]
  buffer_load_dwordx2 v[2:3], v17, s[12:15], 0 offen offset:0
  s_mov_b64 exec, -1
  v_mov_b32 v4, 200704
 _v_add_co_u32 v16, vcc, v16, v4
 _v_add_co_u32 v17, vcc, v17, v4
  v_readlane_b32 s28, v22, 18
  v_readlane_b32 s29, v22, 19
  v_readlane_b32 s32, v22, 20
  v_readlane_b32 s33, v22, 21
  v_readlane_b32 s30, v22, 22
  v_readlane_b32 s31, v22, 23
  v_readlane_b32 s34, v22, 24
  v_readlane_b32 s35, v22, 25
  s_waitcnt lgkmcnt(0) vmcnt(0)
  s_mov_b64 exec, s[28:29]
  v_mov_b32 v1, v0
  v_mov_b32 v0, s36
  s_mov_b64 exec, s[32:33]
  v_mov_b32 v1, s36
  s_mov_b64 exec, s[30:31]
  v_mov_b32 v3, v2
  v_mov_b32 v2, s36
  s_mov_b64 exec, s[34:35]
  v_mov_b32 v3, s36
  s_mov_b64 exec, -1
  s_barrier
  ds_write_b64 v18, v[0:1] offset:0
  s_mov_b64 exec, s[6:7]
  ds_write_b64 v19, v[2:3] offset:0
  s_mov_b64 exec, -1
  s_waitcnt lgkmcnt(0)
  s_barrier
  s_load_dwordx16 s[0:15], s[100:101], 0
  s_load_dwordx16 s[16:31], s[100:101], 64
  s_load_dwordx16 s[32:47], s[100:101], 128
  s_load_dword s96, s[100:101], 192
  s_add_u32 s100, s100, 588
  s_addc_u32 s101, s101, 0
  s_load_dwordx16 s[48:63], s[100:101], 0
  s_load_dwordx16 s[64:79], s[100:101], 64
  s_load_dwordx16 s[80:95], s[100:101], 128
  s_load_dword s97, s[100:101], 192
  s_add_u32 s100, s100, 4294966904
  s_addc_u32 s101, s101, -1
  s_waitcnt lgkmcnt(0) vmcnt(0)
  ds_read_b64 v[0:1], v20 offset:0
  ds_read_b64 v[2:3], v20 offset:8
  ds_read_b64 v[4:5], v20 offset:16
  ds_read_b32 v6, v20 offset:24
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v0, s0
  v_mac_f32 v8, v1, s1
  v_mac_f32 v8, v2, s2
  v_mac_f32 v8, v3, s3
  v_mac_f32 v12, v0, s48
  v_mac_f32 v12, v1, s49
  v_mac_f32 v12, v2, s50
  v_mac_f32 v12, v3, s51
  ds_read_b64 v[0:1], v20 offset:536
  ds_read_b64 v[2:3], v20 offset:544
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v4, s4
  v_mac_f32 v8, v5, s5
  v_mac_f32 v8, v6, s6
  v_mac_f32 v12, v4, s52
  v_mac_f32 v12, v5, s53
  v_mac_f32 v12, v6, s54
  ds_read_b64 v[4:5], v20 offset:552
  ds_read_b32 v6, v20 offset:560
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v0, s7
  v_mac_f32 v8, v1, s8
  v_mac_f32 v8, v2, s9
  v_mac_f32 v8, v3, s10
  v_mac_f32 v12, v0, s55
  v_mac_f32 v12, v1, s56
  v_mac_f32 v12, v2, s57
  v_mac_f32 v12, v3, s58
  ds_read_b64 v[0:1], v20 offset:1072
  ds_read_b64 v[2:3], v20 offset:1080
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v4, s11
  v_mac_f32 v8, v5, s12
  v_mac_f32 v8, v6, s13
  v_mac_f32 v12, v4, s59
  v_mac_f32 v12, v5, s60
  v_mac_f32 v12, v6, s61
  ds_read_b64 v[4:5], v20 offset:1088
  ds_read_b32 v6, v20 offset:1096
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v0, s14
  v_mac_f32 v8, v1, s15
  v_mac_f32 v8, v2, s16
  v_mac_f32 v8, v3, s17
  v_mac_f32 v9, v0, s0
  v_mac_f32 v9, v1, s1
  v_mac_f32 v9, v2, s2
  v_mac_f32 v9, v3, s3
  v_mac_f32 v12, v0, s62
  v_mac_f32 v12, v1, s63
  v_mac_f32 v12, v2, s64
  v_mac_f32 v12, v3, s65
  v_mac_f32 v13, v0, s48
  v_mac_f32 v13, v1, s49
  v_mac_f32 v13, v2, s50
  v_mac_f32 v13, v3, s51
  ds_read_b64 v[0:1], v20 offset:1608
  ds_read_b64 v[2:3], v20 offset:1616
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v4, s18
  v_mac_f32 v8, v5, s19
  v_mac_f32 v8, v6, s20
  v_mac_f32 v9, v4, s4
  v_mac_f32 v9, v5, s5
  v_mac_f32 v9, v6, s6
  v_mac_f32 v12, v4, s66
  v_mac_f32 v12, v5, s67
  v_mac_f32 v12, v6, s68
  v_mac_f32 v13, v4, s52
  v_mac_f32 v13, v5, s53
  v_mac_f32 v13, v6, s54
  ds_read_b64 v[4:5], v20 offset:1624
  ds_read_b32 v6, v20 offset:1632
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v0, s21
  v_mac_f32 v8, v1, s22
  v_mac_f32 v8, v2, s23
  v_mac_f32 v8, v3, s24
  v_mac_f32 v9, v0, s7
  v_mac_f32 v9, v1, s8
  v_mac_f32 v9, v2, s9
  v_mac_f32 v9, v3, s10
  v_mac_f32 v12, v0, s69
  v_mac_f32 v12, v1, s70
  v_mac_f32 v12, v2, s71
  v_mac_f32 v12, v3, s72
  v_mac_f32 v13, v0, s55
  v_mac_f32 v13, v1, s56
  v_mac_f32 v13, v2, s57
  v_mac_f32 v13, v3, s58
  ds_read_b64 v[0:1], v20 offset:2144
  ds_read_b64 v[2:3], v20 offset:2152
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v4, s25
  v_mac_f32 v8, v5, s26
  v_mac_f32 v8, v6, s27
  v_mac_f32 v9, v4, s11
  v_mac_f32 v9, v5, s12
  v_mac_f32 v9, v6, s13
  v_mac_f32 v12, v4, s73
  v_mac_f32 v12, v5, s74
  v_mac_f32 v12, v6, s75
  v_mac_f32 v13, v4, s59
  v_mac_f32 v13, v5, s60
  v_mac_f32 v13, v6, s61
  ds_read_b64 v[4:5], v20 offset:2160
  ds_read_b32 v6, v20 offset:2168
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v0, s28
  v_mac_f32 v8, v1, s29
  v_mac_f32 v8, v2, s30
  v_mac_f32 v8, v3, s31
  v_mac_f32 v9, v0, s14
  v_mac_f32 v9, v1, s15
  v_mac_f32 v9, v2, s16
  v_mac_f32 v9, v3, s17
  v_mac_f32 v10, v0, s0
  v_mac_f32 v10, v1, s1
  v_mac_f32 v10, v2, s2
  v_mac_f32 v10, v3, s3
  v_mac_f32 v12, v0, s76
  v_mac_f32 v12, v1, s77
  v_mac_f32 v12, v2, s78
  v_mac_f32 v12, v3, s79
  v_mac_f32 v13, v0, s62
  v_mac_f32 v13, v1, s63
  v_mac_f32 v13, v2, s64
  v_mac_f32 v13, v3, s65
  v_mac_f32 v14, v0, s48
  v_mac_f32 v14, v1, s49
  v_mac_f32 v14, v2, s50
  v_mac_f32 v14, v3, s51
  ds_read_b64 v[0:1], v20 offset:2680
  ds_read_b64 v[2:3], v20 offset:2688
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v4, s32
  v_mac_f32 v8, v5, s33
  v_mac_f32 v8, v6, s34
  v_mac_f32 v9, v4, s18
  v_mac_f32 v9, v5, s19
  v_mac_f32 v9, v6, s20
  v_mac_f32 v10, v4, s4
  v_mac_f32 v10, v5, s5
  v_mac_f32 v10, v6, s6
  v_mac_f32 v12, v4, s80
  v_mac_f32 v12, v5, s81
  v_mac_f32 v12, v6, s82
  v_mac_f32 v13, v4, s66
  v_mac_f32 v13, v5, s67
  v_mac_f32 v13, v6, s68
  v_mac_f32 v14, v4, s52
  v_mac_f32 v14, v5, s53
  v_mac_f32 v14, v6, s54
  ds_read_b64 v[4:5], v20 offset:2696
  ds_read_b32 v6, v20 offset:2704
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v0, s35
  v_mac_f32 v8, v1, s36
  v_mac_f32 v8, v2, s37
  v_mac_f32 v8, v3, s38
  v_mac_f32 v9, v0, s21
  v_mac_f32 v9, v1, s22
  v_mac_f32 v9, v2, s23
  v_mac_f32 v9, v3, s24
  v_mac_f32 v10, v0, s7
  v_mac_f32 v10, v1, s8
  v_mac_f32 v10, v2, s9
  v_mac_f32 v10, v3, s10
  v_mac_f32 v12, v0, s83
  v_mac_f32 v12, v1, s84
  v_mac_f32 v12, v2, s85
  v_mac_f32 v12, v3, s86
  v_mac_f32 v13, v0, s69
  v_mac_f32 v13, v1, s70
  v_mac_f32 v13, v2, s71
  v_mac_f32 v13, v3, s72
  v_mac_f32 v14, v0, s55
  v_mac_f32 v14, v1, s56
  v_mac_f32 v14, v2, s57
  v_mac_f32 v14, v3, s58
  ds_read_b64 v[0:1], v20 offset:3216
  ds_read_b64 v[2:3], v20 offset:3224
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v4, s39
  v_mac_f32 v8, v5, s40
  v_mac_f32 v8, v6, s41
  v_mac_f32 v9, v4, s25
  v_mac_f32 v9, v5, s26
  v_mac_f32 v9, v6, s27
  v_mac_f32 v10, v4, s11
  v_mac_f32 v10, v5, s12
  v_mac_f32 v10, v6, s13
  v_mac_f32 v12, v4, s87
  v_mac_f32 v12, v5, s88
  v_mac_f32 v12, v6, s89
  v_mac_f32 v13, v4, s73
  v_mac_f32 v13, v5, s74
  v_mac_f32 v13, v6, s75
  v_mac_f32 v14, v4, s59
  v_mac_f32 v14, v5, s60
  v_mac_f32 v14, v6, s61
  ds_read_b64 v[4:5], v20 offset:3232
  ds_read_b32 v6, v20 offset:3240
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v0, s42
  v_mac_f32 v8, v1, s43
  v_mac_f32 v8, v2, s44
  v_mac_f32 v8, v3, s45
  v_mac_f32 v9, v0, s28
  v_mac_f32 v9, v1, s29
  v_mac_f32 v9, v2, s30
  v_mac_f32 v9, v3, s31
  v_mac_f32 v10, v0, s14
  v_mac_f32 v10, v1, s15
  v_mac_f32 v10, v2, s16
  v_mac_f32 v10, v3, s17
  v_mac_f32 v11, v0, s0
  v_mac_f32 v11, v1, s1
  v_mac_f32 v11, v2, s2
  v_mac_f32 v11, v3, s3
  v_mac_f32 v12, v0, s90
  v_mac_f32 v12, v1, s91
  v_mac_f32 v12, v2, s92
  v_mac_f32 v12, v3, s93
  v_mac_f32 v13, v0, s76
  v_mac_f32 v13, v1, s77
  v_mac_f32 v13, v2, s78
  v_mac_f32 v13, v3, s79
  v_mac_f32 v14, v0, s62
  v_mac_f32 v14, v1, s63
  v_mac_f32 v14, v2, s64
  v_mac_f32 v14, v3, s65
  v_mac_f32 v15, v0, s48
  v_mac_f32 v15, v1, s49
  v_mac_f32 v15, v2, s50
  v_mac_f32 v15, v3, s51
  ds_read_b64 v[0:1], v20 offset:3752
  ds_read_b64 v[2:3], v20 offset:3760
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v8, v4, s46
  v_mac_f32 v8, v5, s47
  v_mac_f32 v8, v6, s96
  v_mac_f32 v9, v4, s32
  v_mac_f32 v9, v5, s33
  v_mac_f32 v9, v6, s34
  v_mac_f32 v10, v4, s18
  v_mac_f32 v10, v5, s19
  v_mac_f32 v10, v6, s20
  v_mac_f32 v11, v4, s4
  v_mac_f32 v11, v5, s5
  v_mac_f32 v11, v6, s6
  v_mac_f32 v12, v4, s94
  v_mac_f32 v12, v5, s95
  v_mac_f32 v12, v6, s97
  v_mac_f32 v13, v4, s80
  v_mac_f32 v13, v5, s81
  v_mac_f32 v13, v6, s82
  v_mac_f32 v14, v4, s66
  v_mac_f32 v14, v5, s67
  v_mac_f32 v14, v6, s68
  v_mac_f32 v15, v4, s52
  v_mac_f32 v15, v5, s53
  v_mac_f32 v15, v6, s54
  ds_read_b64 v[4:5], v20 offset:3768
  ds_read_b32 v6, v20 offset:3776
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v9, v0, s35
  v_mac_f32 v9, v1, s36
  v_mac_f32 v9, v2, s37
  v_mac_f32 v9, v3, s38
  v_mac_f32 v10, v0, s21
  v_mac_f32 v10, v1, s22
  v_mac_f32 v10, v2, s23
  v_mac_f32 v10, v3, s24
  v_mac_f32 v11, v0, s7
  v_mac_f32 v11, v1, s8
  v_mac_f32 v11, v2, s9
  v_mac_f32 v11, v3, s10
  v_mac_f32 v13, v0, s83
  v_mac_f32 v13, v1, s84
  v_mac_f32 v13, v2, s85
  v_mac_f32 v13, v3, s86
  v_mac_f32 v14, v0, s69
  v_mac_f32 v14, v1, s70
  v_mac_f32 v14, v2, s71
  v_mac_f32 v14, v3, s72
  v_mac_f32 v15, v0, s55
  v_mac_f32 v15, v1, s56
  v_mac_f32 v15, v2, s57
  v_mac_f32 v15, v3, s58
  ds_read_b64 v[0:1], v20 offset:4288
  ds_read_b64 v[2:3], v20 offset:4296
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v9, v4, s39
  v_mac_f32 v9, v5, s40
  v_mac_f32 v9, v6, s41
  v_mac_f32 v10, v4, s25
  v_mac_f32 v10, v5, s26
  v_mac_f32 v10, v6, s27
  v_mac_f32 v11, v4, s11
  v_mac_f32 v11, v5, s12
  v_mac_f32 v11, v6, s13
  v_mac_f32 v13, v4, s87
  v_mac_f32 v13, v5, s88
  v_mac_f32 v13, v6, s89
  v_mac_f32 v14, v4, s73
  v_mac_f32 v14, v5, s74
  v_mac_f32 v14, v6, s75
  v_mac_f32 v15, v4, s59
  v_mac_f32 v15, v5, s60
  v_mac_f32 v15, v6, s61
  ds_read_b64 v[4:5], v20 offset:4304
  ds_read_b32 v6, v20 offset:4312
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v9, v0, s42
  v_mac_f32 v9, v1, s43
  v_mac_f32 v9, v2, s44
  v_mac_f32 v9, v3, s45
  v_mac_f32 v10, v0, s28
  v_mac_f32 v10, v1, s29
  v_mac_f32 v10, v2, s30
  v_mac_f32 v10, v3, s31
  v_mac_f32 v11, v0, s14
  v_mac_f32 v11, v1, s15
  v_mac_f32 v11, v2, s16
  v_mac_f32 v11, v3, s17
  v_mac_f32 v13, v0, s90
  v_mac_f32 v13, v1, s91
  v_mac_f32 v13, v2, s92
  v_mac_f32 v13, v3, s93
  v_mac_f32 v14, v0, s76
  v_mac_f32 v14, v1, s77
  v_mac_f32 v14, v2, s78
  v_mac_f32 v14, v3, s79
  v_mac_f32 v15, v0, s62
  v_mac_f32 v15, v1, s63
  v_mac_f32 v15, v2, s64
  v_mac_f32 v15, v3, s65
  ds_read_b64 v[0:1], v20 offset:4824
  ds_read_b64 v[2:3], v20 offset:4832
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v9, v4, s46
  v_mac_f32 v9, v5, s47
  v_mac_f32 v9, v6, s96
  v_mac_f32 v10, v4, s32
  v_mac_f32 v10, v5, s33
  v_mac_f32 v10, v6, s34
  v_mac_f32 v11, v4, s18
  v_mac_f32 v11, v5, s19
  v_mac_f32 v11, v6, s20
  v_mac_f32 v13, v4, s94
  v_mac_f32 v13, v5, s95
  v_mac_f32 v13, v6, s97
  v_mac_f32 v14, v4, s80
  v_mac_f32 v14, v5, s81
  v_mac_f32 v14, v6, s82
  v_mac_f32 v15, v4, s66
  v_mac_f32 v15, v5, s67
  v_mac_f32 v15, v6, s68
  ds_read_b64 v[4:5], v20 offset:4840
  ds_read_b32 v6, v20 offset:4848
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v10, v0, s35
  v_mac_f32 v10, v1, s36
  v_mac_f32 v10, v2, s37
  v_mac_f32 v10, v3, s38
  v_mac_f32 v11, v0, s21
  v_mac_f32 v11, v1, s22
  v_mac_f32 v11, v2, s23
  v_mac_f32 v11, v3, s24
  v_mac_f32 v14, v0, s83
  v_mac_f32 v14, v1, s84
  v_mac_f32 v14, v2, s85
  v_mac_f32 v14, v3, s86
  v_mac_f32 v15, v0, s69
  v_mac_f32 v15, v1, s70
  v_mac_f32 v15, v2, s71
  v_mac_f32 v15, v3, s72
  ds_read_b64 v[0:1], v20 offset:5360
  ds_read_b64 v[2:3], v20 offset:5368
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v10, v4, s39
  v_mac_f32 v10, v5, s40
  v_mac_f32 v10, v6, s41
  v_mac_f32 v11, v4, s25
  v_mac_f32 v11, v5, s26
  v_mac_f32 v11, v6, s27
  v_mac_f32 v14, v4, s87
  v_mac_f32 v14, v5, s88
  v_mac_f32 v14, v6, s89
  v_mac_f32 v15, v4, s73
  v_mac_f32 v15, v5, s74
  v_mac_f32 v15, v6, s75
  ds_read_b64 v[4:5], v20 offset:5376
  ds_read_b32 v6, v20 offset:5384
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v10, v0, s42
  v_mac_f32 v10, v1, s43
  v_mac_f32 v10, v2, s44
  v_mac_f32 v10, v3, s45
  v_mac_f32 v11, v0, s28
  v_mac_f32 v11, v1, s29
  v_mac_f32 v11, v2, s30
  v_mac_f32 v11, v3, s31
  v_mac_f32 v14, v0, s90
  v_mac_f32 v14, v1, s91
  v_mac_f32 v14, v2, s92
  v_mac_f32 v14, v3, s93
  v_mac_f32 v15, v0, s76
  v_mac_f32 v15, v1, s77
  v_mac_f32 v15, v2, s78
  v_mac_f32 v15, v3, s79
  ds_read_b64 v[0:1], v20 offset:5896
  ds_read_b64 v[2:3], v20 offset:5904
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v10, v4, s46
  v_mac_f32 v10, v5, s47
  v_mac_f32 v10, v6, s96
  v_mac_f32 v11, v4, s32
  v_mac_f32 v11, v5, s33
  v_mac_f32 v11, v6, s34
  v_mac_f32 v14, v4, s94
  v_mac_f32 v14, v5, s95
  v_mac_f32 v14, v6, s97
  v_mac_f32 v15, v4, s80
  v_mac_f32 v15, v5, s81
  v_mac_f32 v15, v6, s82
  ds_read_b64 v[4:5], v20 offset:5912
  ds_read_b32 v6, v20 offset:5920
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v11, v0, s35
  v_mac_f32 v11, v1, s36
  v_mac_f32 v11, v2, s37
  v_mac_f32 v11, v3, s38
  v_mac_f32 v15, v0, s83
  v_mac_f32 v15, v1, s84
  v_mac_f32 v15, v2, s85
  v_mac_f32 v15, v3, s86
  ds_read_b64 v[0:1], v20 offset:6432
  ds_read_b64 v[2:3], v20 offset:6440
  s_waitcnt lgkmcnt(2)
  v_mac_f32 v11, v4, s39
  v_mac_f32 v11, v5, s40
  v_mac_f32 v11, v6, s41
  v_mac_f32 v15, v4, s87
  v_mac_f32 v15, v5, s88
  v_mac_f32 v15, v6, s89
  ds_read_b64 v[4:5], v20 offset:6448
  ds_read_b32 v6, v20 offset:6456
  s_waitcnt lgkmcnt(1)
  v_mac_f32 v11, v0, s42
  v_mac_f32 v11, v1, s43
  v_mac_f32 v11, v2, s44
  v_mac_f32 v11, v3, s45
  v_mac_f32 v11, v4, s46
  v_mac_f32 v11, v5, s47
  v_mac_f32 v15, v0, s90
  v_mac_f32 v15, v1, s91
  v_mac_f32 v15, v2, s92
  v_mac_f32 v15, v3, s93
  v_mac_f32 v15, v4, s94
  v_mac_f32 v15, v5, s95
  s_waitcnt lgkmcnt(0)
  v_mac_f32 v11, v6, s96
  v_mac_f32 v15, v6, s97
  v_readlane_b32 s21, v22, 2
  s_sub_u32 s21, s21, 1
  v_writelane_b32 v22, s21, 2
  s_cmp_gt_u32 s21, 0
  s_cbranch_scc1 loop_channel
  v_readlane_b32 s20, v22, 3
  v_readlane_b32 s22, v22, 4
  v_readlane_b32 s16, v22, 10
  v_readlane_b32 s17, v22, 11
  s_mov_b32 s18, 100352
  s_mov_b32 s19, 0x00020000
  v_cmpx_gt_u32 vcc, 112, v21
  v_lshlrev_b32 v21, 2, v21
  s_mov_b32 s8, 0
  buffer_store_dword v8, v21, s[16:19], s8 offen offset:0
  s_mov_b32 s8, 448
  buffer_store_dword v9, v21, s[16:19], s8 offen offset:0
  s_mov_b32 s8, 896
  buffer_store_dword v10, v21, s[16:19], s8 offen offset:0
  s_mov_b32 s8, 1344
  buffer_store_dword v11, v21, s[16:19], s8 offen offset:0
  s_mov_b32 s8, 50176
  buffer_store_dword v12, v21, s[16:19], s8 offen offset:0
  s_mov_b32 s8, 50624
  buffer_store_dword v13, v21, s[16:19], s8 offen offset:0
  s_mov_b32 s8, 51072
  buffer_store_dword v14, v21, s[16:19], s8 offen offset:0
  s_mov_b32 s8, 51520
  buffer_store_dword v15, v21, s[16:19], s8 offen offset:0
  s_endpgm
.ifndef ROCM_METADATA_VERSION
.error "ROCM_METADATA_VERSION must be defined"
.endif
.if ROCM_METADATA_VERSION == 3
.amdgpu_code_object_metadata
{ Version: [ 3, 0 ],
    Kernels:
    - {
        Name: gcnAsmConv7x7c3h224w224k64u2v2p3q3f1, Language: OpenCL C, LanguageVersion: [ 1, 2 ],
        Attrs: { ReqdWorkGroupSize: [ 64, 8, 1 ] }
        Args:
        - { Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', Name: in,          AddrSpaceQual: Global, AccQual: Default, IsConst: true }
        - { Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', Name: weights,     AddrSpaceQual: Global, AccQual: Default, IsConst: true }
        - { Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', Name: out,         AddrSpaceQual: Global, AccQual: Default }
        - { Size: 4, Align: 4, ValueKind: ByValue,      ValueType: F32, TypeName:  float,   Name: padding_val,                        AccQual: Default }
      }
}
.end_amdgpu_code_object_metadata
.endif
.if ROCM_METADATA_VERSION == 4
.amd_amdgpu_hsa_metadata
{ Version: [ 1, 0 ],
    Kernels:
    - {
        Name: gcnAsmConv7x7c3h224w224k64u2v2p3q3f1, SymbolName: 'gcnAsmConv7x7c3h224w224k64u2v2p3q3f1@kd', Language: OpenCL C, LanguageVersion: [ 1, 2 ],
        Attrs: { ReqdWorkGroupSize: [ 64, 8, 1 ] }
        CodeProps:
          { KernargSegmentSize: 32, GroupSegmentFixedSize: 0, PrivateSegmentFixedSize: 0, KernargSegmentAlign: 8, WavefrontSize: 64, MaxFlatWorkGroupSize: 256 }
        Args:
        - { Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', Name: in,          AddrSpaceQual: Global, AccQual: Default, IsConst: true }
        - { Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', Name: weights,     AddrSpaceQual: Global, AccQual: Default, IsConst: true }
        - { Size: 8, Align: 8, ValueKind: GlobalBuffer, ValueType: F32, TypeName: 'float*', Name: out,         AddrSpaceQual: Global, AccQual: Default }
        - { Size: 4, Align: 4, ValueKind: ByValue,      ValueType: F32, TypeName:  float,   Name: padding_val,                        AccQual: Default }
      }
}
.end_amd_amdgpu_hsa_metadata
.endif
