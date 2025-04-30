/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

.macro _v_add_f16_e32_1_gfx9x vdst:req, vsrc:req
    .long  0x3E0000FF + (\vdst << 17) + (\vsrc << 9)
    .long  0x00003C00
.endm

v_mov_b32_e32 v0, v0
s_mov_b32 s0, 0
v_mov_b32_e32 v115, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s79, 0xc220
s_mov_b32 s78, 0xc220
v_readfirstlane_b32 s84, v0
s_lshr_b32 s84, s84, 5
s_add_u32 s84, s84, 8
s_and_b32 s74, s84, 20
s_mov_b64 s[84:85], s[2:3]
s_load_dwordx16 s[8:23], s[84:85], 0x0
s_load_dwordx4 s[24:27], s[84:85], 0x40
s_load_dwordx2 s[28:29], s[84:85], 0x50
s_waitcnt lgkmcnt(0)
s_and_b32 s14, s14, 0xffff
s_bitcmp1_b32 s14, 6
s_cbranch_scc0 12
s_and_b32 s17, s17, 0xffff
s_and_b32 s19, s19, 0xffff
s_and_b32 s21, s21, 0xffff
s_load_dwordx2 s[16:17], s[16:17], 0x0
s_load_dwordx2 s[18:19], s[18:19], 0x0
s_load_dwordx2 s[20:21], s[20:21], 0x0
s_bitcmp1_b32 s14, 7
s_cbranch_scc0 2
s_load_dwordx2 s[30:31], s[84:85], 0x58
s_mov_b32 s32, 1.0
s_mov_b32 s33, 1.0
s_getpc_b64 s[82:83]
s_add_u32 s82, s82, 0x46a8
s_addc_u32 s83, s83, 0
s_bitcmp1_b32 s14, 14
s_cbranch_scc1 3
s_bitcmp1_b32 s14, 8
s_cbranch_scc0 12
s_branch 16
s_load_dword s34, s[84:85], 0xc8
s_waitcnt lgkmcnt(0)
s_and_b32 s34, s34, 0xff
s_cmp_eq_u32 s34, 2
s_cbranch_scc1 16
s_cmp_eq_u32 s34, 3
s_cbranch_scc1 19
s_cmp_eq_u32 s34, 1
s_cbranch_scc1 5
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x4988
s_addc_u32 s81, s81, 0
s_branch 18
s_load_dword s32, s[84:85], 0x60
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x4c14
s_addc_u32 s81, s81, 0
s_branch 11
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x5088
s_addc_u32 s81, s81, 0
s_branch 6
s_load_dwordx2 s[32:33], s[84:85], 0x60
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x5554
s_addc_u32 s81, s81, 0
s_bitcmp1_b32 s14, 7
s_cbranch_scc0 7
s_bitcmp1_b32 s14, 6
s_cbranch_scc0 5
s_waitcnt lgkmcnt(0)
s_and_b32 s31, s31, 0xffff
s_load_dwordx2 s[30:31], s[30:31], 0x0
s_bitcmp1_b32 s14, 9
s_cbranch_scc0 83
s_mov_b32 s86, 0x8c
s_mov_b32 s87, 0x9c
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cmp_eq_u64 0, vcc
s_cselect_b32 s86, s87, s86
s_load_dword s44, s[84:85], 0x88
s_load_dword s49, s[84:85], 0x90
s_load_dword s72, s[84:85], 0x98
s_load_dword s48, s[84:85], s86
s_load_dword s73, s[84:85], 0xa0
s_load_dword s45, s[84:85], 0xa8
s_load_dword s46, s[84:85], 0xac
s_load_dword s50, s[84:85], 0xb0
s_bitcmp1_b32 s14, 10
s_cbranch_scc0 79
s_load_dwordx4 s[92:95], s[84:85], 0xb8
v_ffbh_u32_e32 v118, s13
v_lshlrev_b32_e64 v119, v118, s13
v_and_b32_e32 v120, 0xffffff00, v119
v_cmp_eq_u32_e32 vcc, 0x80000000, v119
v_cvt_f32_u32_e32 v120, v120
v_rcp_f32_e32 v116, v120
v_subb_co_u32_e32 v117, vcc, 32, v118, vcc
v_cvt_f32_ubyte0_e32 v118, v119
v_fma_f32 v120, v120, v116, -1.0
v_fma_f32 v120, v118, v116, v120
v_fmaak_f32 v120, v120, v116, 0x9f000000
v_mul_f32_e32 v120, 0x5f800000, v120
v_mov_b32_e32 v118, 0
v_cvt_flr_i32_f32_e64 v120, -v120
v_lshl_add_u32 v116, v116, 9, v120
v_mad_u64_u32 v[118:119], vcc, v119, v116, v[118:119]
v_subb_co_u32_e64 v116, vcc, v116, -1, vcc
v_mul_hi_u32 v118, s4, v116
v_add_co_u32_e64 v116, vcc, v118, s4
v_addc_co_u32_e64 v118, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v117
v_cndmask_b32_e32 v116, v116, v118, vcc
v_alignbit_b32 v116, v118, v116, v117
s_nop 0
v_readfirstlane_b32 s88, v116
s_mul_i32 s89, s88, s13
s_sub_u32 s4, s4, s89
s_waitcnt lgkmcnt(0)
s_lshl_b32 s93, s93, 1
s_lshl_b64 s[94:95], s[94:95], 1
s_mul_i32 s89, s93, s88
s_add_u32 s16, s16, s89
s_addc_u32 s17, s17, 0
s_mul_i32 s89, s94, s88
s_add_u32 s18, s18, s89
s_addc_u32 s19, s19, 0
s_mul_i32 s89, s95, s88
s_add_u32 s20, s20, s89
s_addc_u32 s21, s21, 0
s_branch 22
s_mov_b32 s49, s11
s_mul_i32 s48, s10, s11
s_mul_i32 s44, s48, s9
s_mov_b32 s50, s29
s_mul_i32 s46, s28, s29
s_mul_i32 s45, s46, s12
s_bitcmp1_b32 s14, 13
s_cbranch_scc0 2
s_load_dwordx8 s[92:99], s[84:85], 0x68
s_mov_b32 s73, s25
s_mul_i32 s86, s24, s25
s_bitcmp1_b32 s14, 2
s_cselect_b32 s87, s12, s9
s_mul_i32 s87, s86, s87
s_bitcmp1_b32 s14, 2
s_cselect_b32 s100, s87, s86
s_cselect_b32 s72, s86, s87
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cmp_eq_u64 0, vcc
s_cselect_b32 s48, s100, s48
s_waitcnt lgkmcnt(0)
s_lshl_b32 s47, s48, 1
s_and_b32 s17, s17, 0xffff
s_and_b32 s19, s19, 0xffff
s_and_b32 s21, s21, 0xffff
s_and_b32 s31, s31, 0xffff
s_bitcmp1_b32 s14, 13
s_cbranch_scc0 8
s_add_u32 s16, s16, s92
s_addc_u32 s17, s17, s93
s_add_u32 s18, s18, s94
s_addc_u32 s19, s19, s95
s_add_u32 s20, s20, s96
s_addc_u32 s21, s21, s97
s_add_u32 s30, s30, s98
s_addc_u32 s31, s31, s99
v_cvt_f16_f32_e32 v116, s32
v_cvt_f16_f32_e32 v117, s33
v_readfirstlane_b32 s32, v116
v_readfirstlane_b32 s33, v117
s_and_b32 s88, 0, s26
s_addc_u32 s88, s28, 0
s_ashr_i32 s88, s88, 0
s_add_u32 s86, s88, 2
v_mov_b32_e32 v117, 0x55555556
v_mul_hi_u32 v117, v117, s86
s_nop 0
v_readfirstlane_b32 s86, v117
s_andn2_b32 s88, 0, s27
s_addc_u32 s88, s29, 0
s_ashr_i32 s88, s88, 0
s_add_u32 s87, s88, 2
v_mov_b32_e32 v117, 0x55555556
v_mul_hi_u32 v117, v117, s87
s_nop 0
v_readfirstlane_b32 s87, v117
s_sub_u32 s57, 0, s87
s_sub_u32 s56, 0, s86
s_add_u32 s1, s24, 1
v_mov_b32_e32 v117, 0x80000000
v_mul_hi_u32 v117, v117, s1
s_nop 0
v_readfirstlane_b32 s1, v117
s_add_u32 s5, s25, 1
v_mov_b32_e32 v117, 0x80000000
v_mul_hi_u32 v117, v117, s5
s_nop 0
v_readfirstlane_b32 s5, v117
v_mad_i32_i24 v116, 2, s1, -1
v_sub_co_u32_e64 v116, vcc, v116, s24
v_addc_co_u32_e64 v116, vcc, 0, 0, vcc
s_nop 0
v_readfirstlane_b32 s88, v116
s_and_b32 s88, s88, 0
s_and_b32 s88, s88, s1
s_add_u32 s1, s1, s88
v_readfirstlane_b32 s89, v0
s_and_b32 s90, s89, 64
s_cselect_b32 s90, 0x80000, 0
s_or_b32 s14, s14, s90
s_lshl_b32 s51, s47, 1
s_sub_u32 s52, 0, s51
s_subb_u32 s53, 0, 0
s_bitcmp1_b32 s14, 12
s_cselect_b32 s88, 0, -1
s_bitcmp1_b32 s14, 11
s_cselect_b32 s88, s88, 1
s_cmp_gt_u32 s5, s88
s_cbranch_scc0 8
s_bitset1_b32 s14, 23
s_bitset1_b32 s14, 20
s_bitset0_b32 s14, 19
s_ashr_i32 s51, s51, 1
s_ashr_i64 s[52:53], s[52:53], 1
s_add_u32 s5, s5, 1
s_and_b32 s5, s5, -2
s_branch 16
s_and_b32 s90, s9, 3
s_cselect_b32 s90, 0, 0x1000000
s_bitcmp1_b32 s14, 2
s_cselect_b32 s90, 0, s90
s_or_b32 s14, s14, s90
s_cmp_eq_u32 s90, 0
s_cselect_b32 s51, s47, s51
s_cselect_b32 s52, s47, s52
s_cselect_b32 s53, 0, s53
s_bitcmp0_b32 s89, 8
s_cselect_b32 s90, s90, 0
s_cmp_eq_u32 s90, 0
s_cselect_b32 s90, 0, 0x80000
s_andn2_b32 s14, s14, s90
s_add_u32 s52, s52, s51
s_addc_u32 s53, s53, 0
s_add_u32 s52, s52, s51
s_addc_u32 s53, s53, 0
v_bfe_u32 v117, v0, 2, 6
v_lshrrev_b32_e32 v110, 1, v117
s_bitcmp0_b32 s89, 8
s_cselect_b32 s90, 0x1000000, 0
s_or_b32 s90, s90, 0x100000
s_and_b32 s90, s14, s90
s_cselect_b32 s90, 0, 15
v_bfi_b32 v110, s90, v117, v110
s_mul_i32 s70, s8, s86
s_sub_u32 s70, s70, 1
s_lshr_b32 s70, s70, 0
s_add_u32 s70, s70, 1
s_lshr_b32 s89, -1, 16
s_and_b32 s89, s89, s70
s_lshr_b32 s90, s70, 16
s_mul_i32 s90, s90, s87
s_mul_i32 s70, s89, s87
s_lshl_b32 s89, s90, 16
s_lshr_b32 s90, s90, 16
s_add_u32 s70, s89, s70
s_addc_u32 s71, s90, 0
s_sub_u32 s70, s70, 1
s_subb_u32 s71, s71, 0
s_lshr_b64 s[70:71], s[70:71], 5
s_add_u32 s70, s70, 1
s_addc_u32 s71, s71, 0
v_mov_b32_e32 v117, s4
v_mov_b32_e32 v118, s13
v_and_b32_e32 v119, 3, v0
v_cmp_eq_u32_e32 vcc, 2, v119
v_cndmask_b32_e32 v117, v117, v118, vcc
v_cmp_eq_u32_e32 vcc, 1, v119
v_cndmask_b32_e32 v120, 0, v110, vcc
s_bitcmp1_b32 s14, 20
s_cbranch_scc0 4
v_add_co_u32_e64 v118, vcc, v110, 8
v_cmp_eq_u32_e32 vcc, 0, v119
v_cndmask_b32_e32 v120, v120, v118, vcc
v_cmp_eq_u32_e64 s[90:91], 3, v119
v_bfe_u32 v108, v120, 0, 5
v_mad_u32_u24 v108, v117, 32, v108
v_ffbh_u32_e32 v122, s87
v_lshlrev_b32_e64 v123, v122, s87
v_and_b32_e32 v124, 0xffffff00, v123
v_cmp_eq_u32_e32 vcc, 0x80000000, v123
v_cvt_f32_u32_e32 v124, v124
v_rcp_f32_e32 v109, v124
v_subb_co_u32_e32 v121, vcc, 32, v122, vcc
v_cvt_f32_ubyte0_e32 v122, v123
v_fma_f32 v124, v124, v109, -1.0
v_fma_f32 v124, v122, v109, v124
v_fmaak_f32 v124, v124, v109, 0x9f000000
v_mul_f32_e32 v124, 0x5f800000, v124
v_mov_b32_e32 v122, 0
v_cvt_flr_i32_f32_e64 v124, -v124
v_lshl_add_u32 v109, v109, 9, v124
v_mad_u64_u32 v[122:123], vcc, v123, v109, v[122:123]
v_subb_co_u32_e64 v109, vcc, v109, -1, vcc
v_mul_hi_u32 v122, v108, v109
v_add_co_u32_e32 v109, vcc, v122, v108
v_addc_co_u32_e64 v122, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v121
v_cndmask_b32_e32 v109, v109, v122, vcc
v_alignbit_b32 v109, v122, v109, v121
v_mad_i32_i24 v107, v109, s57, v108
v_lshrrev_b32_e32 v108, 5, v120
v_mad_u32_u24 v108, v109, 1, v108
v_cndmask_b32_e64 v108, v108, 1, s[90:91]
v_ffbh_u32_e32 v122, s86
v_lshlrev_b32_e64 v123, v122, s86
v_and_b32_e32 v124, 0xffffff00, v123
v_cmp_eq_u32_e32 vcc, 0x80000000, v123
v_cvt_f32_u32_e32 v124, v124
v_rcp_f32_e32 v109, v124
v_subb_co_u32_e32 v121, vcc, 32, v122, vcc
v_cvt_f32_ubyte0_e32 v122, v123
v_fma_f32 v124, v124, v109, -1.0
v_fma_f32 v124, v122, v109, v124
v_fmaak_f32 v124, v124, v109, 0x9f000000
v_mul_f32_e32 v124, 0x5f800000, v124
v_mov_b32_e32 v122, 0
v_cvt_flr_i32_f32_e64 v124, -v124
v_lshl_add_u32 v109, v109, 9, v124
v_mad_u64_u32 v[122:123], vcc, v123, v109, v[122:123]
v_subb_co_u32_e64 v109, vcc, v109, -1, vcc
v_mul_hi_u32 v122, v108, v109
v_add_co_u32_e32 v109, vcc, v122, v108
v_addc_co_u32_e64 v122, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v121
v_cndmask_b32_e32 v109, v109, v122, vcc
v_alignbit_b32 v109, v122, v109, v121
v_mad_i32_i24 v108, v109, s56, v108
v_readlane_b32 s58, v107, 2
v_readlane_b32 s59, v108, 2
v_readlane_b32 s60, v109, 2
v_readlane_b32 s61, v108, 3
v_readlane_b32 s62, v109, 3
v_add_co_u32_e64 v107, vcc, v107, s57
v_add_co_u32_e64 v108, vcc, v108, s56
v_mov_b32_dpp v109, v109 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v107, v107 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v108, v108 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x20000
s_mov_b32 s86, 0x80000000
s_mov_b32 s87, 0x20000
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 7
v_xor_b32_dpp v111, v0, v0 quad_perm:[2,3,2,1] row_mask:0xf bank_mask:0xf
v_subrev_co_u32_e32 v111, vcc, 1, v111
v_cvt_f16_i16_e32 v111, v111
v_pk_add_f16 v111, v111, 0 op_sel_hi:[0,0]
s_branch 6
v_xor_b32_dpp v111, v0, v0 quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32_e32 v111, vcc, 1, v111
v_cvt_f16_i16_e32 v111, v111
v_pk_add_f16 v111, v111, 0 op_sel_hi:[0,0]
v_mov_b32_e32 v112, 1
v_xor_b32_dpp v112, v0, v0 quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v112, v0, v0 quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32_e32 v112, vcc, 1, v112
v_mov_b32_e32 v113, 1
v_xor_b32_dpp v113, v0, v0 quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v113, v0, v0 quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32_e32 v113, vcc, 1, v113
v_cvt_f32_i32_e32 v112, v112
v_cvt_f32_i32_e32 v113, v113
v_lshrrev_b32_e64 v116, 2, s74
v_and_b32_e32 v117, 3, v0
v_bfe_u32 v118, v0, 4, 3
v_mad_u32_u24 v106, v118, 4, v117
v_lshlrev_b32_e32 v106, 4, v106
v_mad_u32_u24 v101, v116, 4, v117
v_lshlrev_b32_e32 v101, 4, v101
v_bfe_u32 v116, v0, 2, 2
v_and_b32_e32 v117, 1, v116
v_mad_u32_u24 v119, v116, 16, v117
v_lshlrev_b32_e32 v119, 6, v119
v_xor_b32_e32 v101, v101, v119
v_mul_u32_u24_e32 v119, 0x400, v116
v_xor_b32_e32 v106, v106, v119
s_lshr_b32 s74, s74, 0
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 47
s_and_b32 s3, s14, 0x1100000
s_addc_u32 s3, 0, 0
v_lshrrev_b32_e32 v119, 1, v0
s_mul_i32 s2, 60, s3
s_sub_u32 s2, 63, s2
v_bfi_b32 v119, s2, v0, v119
v_and_b32_e32 v116, 1, v119
v_bfe_u32 v117, v119, 1, 1
v_xor_b32_e32 v116, v116, v117
v_bfe_u32 v118, v119, 3, 1
v_mad_u32_u24 v117, v117, 2, v118
v_mul_u32_u24_e32 v116, 0x118, v116
v_bfe_u32 v118, v119, 2, 1
v_mad_u32_u24 v117, v117, 2, v116
v_xor_b32_e32 v117, v117, v118
v_and_b32_e32 v118, 0xf0, v119
v_xor_b32_e32 v117, v117, v118
s_mul_i32 s2, 4, s3
s_sub_u32 s2, 6, s2
v_bfe_u32 v119, v0, s2, 1
v_mul_u32_u24_e32 v119, 0x1040, v119
v_xor_b32_e32 v103, 0x314, v117
v_xor_b32_e32 v104, 0x31c, v117
v_xor_b32_e32 v105, 8, v117
v_mov_b32_e32 v102, v117
v_mad_u32_u24 v102, 4, v102, v119
v_mad_u32_u24 v103, 4, v103, v119
v_mad_u32_u24 v104, 4, v104, v119
v_mad_u32_u24 v105, 4, v105, v119
s_branch 44
s_bfe_u32 s3, s14, 0x10014
v_lshrrev_b32_e32 v119, 1, v0
s_mul_i32 s2, 60, s3
s_sub_u32 s2, 63, s2
v_bfi_b32 v119, s2, v0, v119
v_and_b32_e32 v116, 1, v119
v_bfe_u32 v117, v119, 1, 1
v_bfe_u32 v118, v119, 3, 1
v_xor_b32_e32 v116, v116, v117
v_mad_u32_u24 v117, v117, 2, v118
v_mul_u32_u24_e32 v116, 0x109, v116
v_bfe_u32 v118, v119, 2, 1
v_mad_u32_u24 v117, v117, 2, v116
v_xor_b32_e32 v117, v117, v118
v_and_b32_e32 v118, 0xf0, v119
v_or_b32_e32 v117, v117, v118
s_mul_i32 s2, 4, s3
s_sub_u32 s2, 6, s2
v_bfe_u32 v119, v0, s2, 1
v_mul_u32_u24_e32 v119, 0x1040, v119
v_mad_u32_u24 v102, 4, v117, v119
v_xor_b32_e32 v103, 0x307, v117
v_mad_u32_u24 v103, 4, v103, v119
v_xor_b32_e32 v104, 0x30f, v117
v_mad_u32_u24 v104, 4, v104, v119
v_xor_b32_e32 v105, 8, v117
v_mad_u32_u24 v105, 4, v105, v119
v_subrev_co_u32_e32 v107, vcc, s58, v107
v_mov_b32_e32 v117, s57
v_cmp_lt_i32_e32 vcc, v107, v117
v_subb_co_u32_e64 v116, vcc, 0, 0, vcc
v_mad_i32_i24 v107, v116, s57, v107
v_mad_i32_i24 v109, v116, s62, v109
v_mad_i32_i24 v108, v116, s61, v108
v_mov_b32_e32 v117, s56
v_cmp_lt_i32_e32 vcc, v108, v117
v_subb_co_u32_e64 v116, vcc, 0, 0, vcc
v_add_co_u32_e32 v109, vcc, v109, v116
v_mad_i32_i24 v108, v116, v117, v108
v_subrev_co_u32_e32 v108, vcc, s59, v108
v_cmp_lt_i32_e32 vcc, v108, v117
v_subb_co_u32_e64 v116, vcc, 0, 0, vcc
v_add_co_u32_e32 v109, vcc, v109, v116
v_mad_i32_i24 v108, v116, s56, v108
v_subrev_co_u32_e32 v109, vcc, s60, v109
s_mov_b32 s35, 0
s_mov_b32 s38, s24
s_mov_b32 s39, 1
s_mov_b32 s66, 0
s_mov_b32 s67, s12
s_mov_b32 s65, s67
s_sub_u32 s75, -1, s74
s_sub_u32 s75, s75, 32
s_bitset1_b32 s14, 21
s_mov_b32 s87, 0
s_mov_b32 s91, 0
s_mov_b32 s76, 34
s_mov_b32 s64, 0
s_bitset1_b32 s14, 26
s_call_b64 s[36:37], 2706
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 1
s_branch 1441
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v76, v88, v76
v_pack_b32_f16 v77, v89, v77
v_pack_b32_f16 v78, v90, v78
v_pack_b32_f16 v79, v91, v79
v_pk_fma_f16 v76, v78, -1.0, v76 op_sel_hi:[1,0,1]
v_pk_mul_f16 v76, v76, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v79, v77, -1.0, v79 op_sel_hi:[1,0,1]
v_pk_mul_f16 v79, v79, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v77, v78, v77
v_pk_mul_f16 v77, v77, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v64, v1, s[40:43], 0 idxen
buffer_load_short_d16 v66, v3, s[40:43], 0 idxen
buffer_load_short_d16 v65, v2, s[40:43], 0 idxen
buffer_load_short_d16 v67, v36, s[40:43], 0 idxen
ds_write_b32 v102, v68
ds_read_b128 v[38:41], v106 offset:29440
ds_write_b32 v103, v69
ds_read_b128 v[46:49], v101 offset:28928
ds_read_b128 v[50:53], v101 offset:29056
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 6
s_call_b64 s[36:37], 2581
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_dot2_f32_f16 v33, v43, v61, v33
v_pk_fma_f16 v78, v77, -1.0, v78 op_sel_hi:[1,0,1]
v_mov_b32_dpp v114, v76 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v76, v114, v111, v76
v_mov_b32_dpp v114, v77 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v77, v114, v111, v77
v_mov_b32_dpp v114, v78 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v78, v114, v111, v78
v_mov_b32_dpp v114, v79 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v79, v114, v111, v79
s_setprio 0
s_nop 0
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_setprio 1
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v88, v1, s[40:43], 0 idxen
buffer_load_short_d16 v90, v3, s[40:43], 0 idxen
buffer_load_short_d16 v89, v2, s[40:43], 0 idxen
buffer_load_short_d16 v91, v36, s[40:43], 0 idxen
ds_write_b32 v104, v74 offset:8256
ds_read_b128 v[42:45], v106 offset:33536
ds_write_b32 v105, v75 offset:8256
ds_read_b128 v[54:57], v101 offset:33024
ds_read_b128 v[58:61], v101 offset:33152
s_waitcnt vmcnt(16) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 8
s_call_b64 s[36:37], 2463
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v80, v92, v80
v_pack_b32_f16 v81, v93, v81
v_pack_b32_f16 v82, v94, v82
v_pack_b32_f16 v83, v95, v83
v_pk_fma_f16 v80, v82, -1.0, v80 op_sel_hi:[1,0,1]
v_pk_mul_f16 v80, v80, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v83, v81, -1.0, v83 op_sel_hi:[1,0,1]
v_pk_mul_f16 v83, v83, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v81, v82, v81
v_pk_mul_f16 v81, v81, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v68, v1, s[40:43], 0 idxen
buffer_load_short_d16 v70, v3, s[40:43], 0 idxen
buffer_load_short_d16 v69, v2, s[40:43], 0 idxen
buffer_load_short_d16 v71, v36, s[40:43], 0 idxen
ds_write_b32 v102, v72 offset:8256
ds_read_b128 v[38:41], v106 offset:37696
ds_write_b32 v103, v73 offset:8256
ds_read_b128 v[46:49], v101 offset:37184
ds_read_b128 v[50:53], v101 offset:37312
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 6
s_call_b64 s[36:37], 2341
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_pk_fma_f16 v82, v81, -1.0, v82 op_sel_hi:[1,0,1]
v_mov_b32_dpp v114, v80 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v80, v114, v111, v80
v_mov_b32_dpp v114, v81 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v81, v114, v111, v81
v_mov_b32_dpp v114, v82 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v82, v114, v111, v82
v_mov_b32_dpp v114, v83 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v83, v114, v111, v83
s_setprio 0
s_nop 0
v_dot2_f32_f16 v33, v43, v61, v33
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_barrier
s_setprio 1
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v92, v1, s[40:43], 0 idxen
buffer_load_short_d16 v94, v3, s[40:43], 0 idxen
buffer_load_short_d16 v93, v2, s[40:43], 0 idxen
buffer_load_short_d16 v95, v36, s[40:43], 0 idxen
ds_write_b32 v104, v78 offset:16512
ds_read_b128 v[42:45], v106 offset:41792
ds_write_b32 v105, v79 offset:16512
ds_read_b128 v[54:57], v101 offset:41280
ds_read_b128 v[58:61], v101 offset:41408
s_waitcnt vmcnt(16) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 7
s_call_b64 s[36:37], 2222
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v84, v96, v84
v_pack_b32_f16 v85, v97, v85
v_pack_b32_f16 v86, v98, v86
v_pack_b32_f16 v87, v99, v87
v_pk_fma_f16 v84, v86, -1.0, v84 op_sel_hi:[1,0,1]
v_pk_mul_f16 v84, v84, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v87, v85, -1.0, v87 op_sel_hi:[1,0,1]
v_pk_mul_f16 v87, v87, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v85, v86, v85
v_pk_mul_f16 v85, v85, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v72, v1, s[40:43], 0 idxen
buffer_load_short_d16 v74, v3, s[40:43], 0 idxen
buffer_load_short_d16 v73, v2, s[40:43], 0 idxen
buffer_load_short_d16 v75, v36, s[40:43], 0 idxen
ds_write_b32 v102, v76 offset:16512
ds_read_b128 v[38:41], v106 offset:45952
ds_write_b32 v103, v77 offset:16512
ds_read_b128 v[46:49], v101 offset:45440
ds_read_b128 v[50:53], v101 offset:45568
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 6
s_call_b64 s[36:37], 2101
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_dot2_f32_f16 v33, v43, v61, v33
v_pk_fma_f16 v86, v85, -1.0, v86 op_sel_hi:[1,0,1]
v_mov_b32_dpp v114, v84 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v84, v114, v111, v84
v_mov_b32_dpp v114, v85 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v85, v114, v111, v85
v_mov_b32_dpp v114, v86 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v86, v114, v111, v86
v_mov_b32_dpp v114, v87 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v87, v114, v111, v87
s_setprio 0
s_nop 0
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_setprio 1
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v96, v1, s[40:43], 0 idxen
buffer_load_short_d16 v98, v3, s[40:43], 0 idxen
buffer_load_short_d16 v97, v2, s[40:43], 0 idxen
buffer_load_short_d16 v99, v36, s[40:43], 0 idxen
ds_write_b32 v104, v82 offset:24768
ds_read_b128 v[42:45], v106 offset:512
ds_write_b32 v105, v83 offset:24768
ds_read_b128 v[54:57], v101
ds_read_b128 v[58:61], v101 offset:128
s_waitcnt vmcnt(16) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 8
s_call_b64 s[36:37], 1983
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v64, v88, v64
v_pack_b32_f16 v65, v89, v65
v_pack_b32_f16 v66, v90, v66
v_pack_b32_f16 v67, v91, v67
v_pk_fma_f16 v64, v66, -1.0, v64 op_sel_hi:[1,0,1]
v_pk_mul_f16 v64, v64, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v67, v65, -1.0, v67 op_sel_hi:[1,0,1]
v_pk_mul_f16 v67, v67, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v65, v66, v65
v_pk_mul_f16 v65, v65, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v76, v1, s[40:43], 0 idxen
buffer_load_short_d16 v78, v3, s[40:43], 0 idxen
buffer_load_short_d16 v77, v2, s[40:43], 0 idxen
buffer_load_short_d16 v79, v36, s[40:43], 0 idxen
ds_write_b32 v102, v80 offset:24768
ds_read_b128 v[38:41], v106 offset:4672
ds_write_b32 v103, v81 offset:24768
ds_read_b128 v[46:49], v101 offset:4160
ds_read_b128 v[50:53], v101 offset:4288
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 6
s_call_b64 s[36:37], 1861
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_pk_fma_f16 v66, v65, -1.0, v66 op_sel_hi:[1,0,1]
v_mov_b32_dpp v114, v64 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v64, v114, v111, v64
v_mov_b32_dpp v114, v65 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v65, v114, v111, v65
v_mov_b32_dpp v114, v66 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v66, v114, v111, v66
v_mov_b32_dpp v114, v67 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v67, v114, v111, v67
s_setprio 0
s_nop 0
v_dot2_f32_f16 v33, v43, v61, v33
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_barrier
s_setprio 1
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v88, v1, s[40:43], 0 idxen
buffer_load_short_d16 v90, v3, s[40:43], 0 idxen
buffer_load_short_d16 v89, v2, s[40:43], 0 idxen
buffer_load_short_d16 v91, v36, s[40:43], 0 idxen
ds_write_b32 v104, v86 offset:33024
ds_read_b128 v[42:45], v106 offset:8768
ds_write_b32 v105, v87 offset:33024
ds_read_b128 v[54:57], v101 offset:8256
ds_read_b128 v[58:61], v101 offset:8384
s_waitcnt vmcnt(16) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 7
s_call_b64 s[36:37], 1742
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v68, v92, v68
v_pack_b32_f16 v69, v93, v69
v_pack_b32_f16 v70, v94, v70
v_pack_b32_f16 v71, v95, v71
v_pk_fma_f16 v68, v70, -1.0, v68 op_sel_hi:[1,0,1]
v_pk_mul_f16 v68, v68, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v71, v69, -1.0, v71 op_sel_hi:[1,0,1]
v_pk_mul_f16 v71, v71, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v69, v70, v69
v_pk_mul_f16 v69, v69, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v80, v1, s[40:43], 0 idxen
buffer_load_short_d16 v82, v3, s[40:43], 0 idxen
buffer_load_short_d16 v81, v2, s[40:43], 0 idxen
buffer_load_short_d16 v83, v36, s[40:43], 0 idxen
ds_write_b32 v102, v84 offset:33024
ds_read_b128 v[38:41], v106 offset:12928
ds_write_b32 v103, v85 offset:33024
ds_read_b128 v[46:49], v101 offset:12416
ds_read_b128 v[50:53], v101 offset:12544
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 6
s_call_b64 s[36:37], 1621
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_dot2_f32_f16 v33, v43, v61, v33
v_pk_fma_f16 v70, v69, -1.0, v70 op_sel_hi:[1,0,1]
v_mov_b32_dpp v114, v68 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v68, v114, v111, v68
v_mov_b32_dpp v114, v69 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v69, v114, v111, v69
v_mov_b32_dpp v114, v70 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v70, v114, v111, v70
v_mov_b32_dpp v114, v71 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v71, v114, v111, v71
s_setprio 0
s_nop 0
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_setprio 1
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v92, v1, s[40:43], 0 idxen
buffer_load_short_d16 v94, v3, s[40:43], 0 idxen
buffer_load_short_d16 v93, v2, s[40:43], 0 idxen
buffer_load_short_d16 v95, v36, s[40:43], 0 idxen
ds_write_b32 v104, v66 offset:41280
ds_read_b128 v[42:45], v106 offset:17024
ds_write_b32 v105, v67 offset:41280
ds_read_b128 v[54:57], v101 offset:16512
ds_read_b128 v[58:61], v101 offset:16640
s_waitcnt vmcnt(16) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 8
s_call_b64 s[36:37], 1503
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v72, v96, v72
v_pack_b32_f16 v73, v97, v73
v_pack_b32_f16 v74, v98, v74
v_pack_b32_f16 v75, v99, v75
v_pk_fma_f16 v72, v74, -1.0, v72 op_sel_hi:[1,0,1]
v_pk_mul_f16 v72, v72, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v75, v73, -1.0, v75 op_sel_hi:[1,0,1]
v_pk_mul_f16 v75, v75, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v73, v74, v73
v_pk_mul_f16 v73, v73, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v84, v1, s[40:43], 0 idxen
buffer_load_short_d16 v86, v3, s[40:43], 0 idxen
buffer_load_short_d16 v85, v2, s[40:43], 0 idxen
buffer_load_short_d16 v87, v36, s[40:43], 0 idxen
ds_write_b32 v102, v64 offset:41280
ds_read_b128 v[38:41], v106 offset:21184
ds_write_b32 v103, v65 offset:41280
ds_read_b128 v[46:49], v101 offset:20672
ds_read_b128 v[50:53], v101 offset:20800
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 6
s_call_b64 s[36:37], 1381
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_pk_fma_f16 v74, v73, -1.0, v74 op_sel_hi:[1,0,1]
v_mov_b32_dpp v114, v72 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v72, v114, v111, v72
v_mov_b32_dpp v114, v73 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v73, v114, v111, v73
v_mov_b32_dpp v114, v74 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v74, v114, v111, v74
v_mov_b32_dpp v114, v75 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v75, v114, v111, v75
s_setprio 0
s_nop 0
v_dot2_f32_f16 v33, v43, v61, v33
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_barrier
s_setprio 1
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v96, v1, s[40:43], 0 idxen
buffer_load_short_d16 v98, v3, s[40:43], 0 idxen
buffer_load_short_d16 v97, v2, s[40:43], 0 idxen
buffer_load_short_d16 v99, v36, s[40:43], 0 idxen
ds_write_b32 v104, v70
ds_read_b128 v[42:45], v106 offset:25280
ds_write_b32 v105, v71
ds_read_b128 v[54:57], v101 offset:24768
ds_read_b128 v[58:61], v101 offset:24896
s_waitcnt vmcnt(16) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 64103
s_call_b64 s[36:37], 1262
s_branch 64101
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v76, v88, v76
v_pack_b32_f16 v77, v89, v77
v_pack_b32_f16 v78, v90, v78
v_pack_b32_f16 v79, v91, v79
v_mov_b32_dpp v114, v76 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v76, v114, v111, v76
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v64, v1, s[40:43], 0 idxen
buffer_load_short_d16 v67, v36, s[40:43], 0 idxen
ds_write_b32 v102, v68
ds_read_b128 v[38:41], v106 offset:29440
ds_write_b32 v103, v69
ds_read_b128 v[46:49], v101 offset:28928
ds_read_b128 v[50:53], v101 offset:29056
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 2
s_call_b64 s[36:37], 1153
s_nop 0
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_dot2_f32_f16 v33, v43, v61, v33
v_mov_b32_dpp v114, v79 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v79, v114, v111, v79
v_pk_add_f16 v77, v76, v79
v_pk_mul_f16 v77, v77, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v78, -1.0, v79, v76 op_sel_hi:[0,1,1]
v_pk_mul_f16 v78, v78, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v88, v1, s[40:43], 0 idxen
buffer_load_short_d16 v91, v36, s[40:43], 0 idxen
ds_write_b32 v104, v74 offset:8256
ds_read_b128 v[42:45], v106 offset:33536
ds_write_b32 v105, v75 offset:8256
ds_read_b128 v[54:57], v101 offset:33024
ds_read_b128 v[58:61], v101 offset:33152
s_waitcnt vmcnt(8) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 2
s_call_b64 s[36:37], 1049
s_nop 0
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v80, v92, v80
v_pack_b32_f16 v81, v93, v81
v_pack_b32_f16 v82, v94, v82
v_pack_b32_f16 v83, v95, v83
v_mov_b32_dpp v114, v80 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v80, v114, v111, v80
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v68, v1, s[40:43], 0 idxen
buffer_load_short_d16 v71, v36, s[40:43], 0 idxen
ds_write_b32 v102, v72 offset:8256
ds_read_b128 v[38:41], v106 offset:37696
ds_write_b32 v103, v73 offset:8256
ds_read_b128 v[46:49], v101 offset:37184
ds_read_b128 v[50:53], v101 offset:37312
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 2
s_call_b64 s[36:37], 945
s_nop 0
s_barrier
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_mov_b32_dpp v114, v83 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v83, v114, v111, v83
v_pk_add_f16 v81, v80, v83
v_pk_mul_f16 v81, v81, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v82, -1.0, v83, v80 op_sel_hi:[0,1,1]
v_pk_mul_f16 v82, v82, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v33, v43, v61, v33
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v92, v1, s[40:43], 0 idxen
buffer_load_short_d16 v95, v36, s[40:43], 0 idxen
ds_write_b32 v104, v78 offset:16512
ds_read_b128 v[42:45], v106 offset:41792
ds_write_b32 v105, v79 offset:16512
ds_read_b128 v[54:57], v101 offset:41280
ds_read_b128 v[58:61], v101 offset:41408
s_waitcnt vmcnt(8) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 1
s_call_b64 s[36:37], 840
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v84, v96, v84
v_pack_b32_f16 v85, v97, v85
v_pack_b32_f16 v86, v98, v86
v_pack_b32_f16 v87, v99, v87
v_mov_b32_dpp v114, v84 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v84, v114, v111, v84
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v72, v1, s[40:43], 0 idxen
buffer_load_short_d16 v75, v36, s[40:43], 0 idxen
ds_write_b32 v102, v76 offset:16512
ds_read_b128 v[38:41], v106 offset:45952
ds_write_b32 v103, v77 offset:16512
ds_read_b128 v[46:49], v101 offset:45440
ds_read_b128 v[50:53], v101 offset:45568
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 2
s_call_b64 s[36:37], 737
s_nop 0
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_dot2_f32_f16 v33, v43, v61, v33
v_mov_b32_dpp v114, v87 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v87, v114, v111, v87
v_pk_add_f16 v85, v84, v87
v_pk_mul_f16 v85, v85, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v86, -1.0, v87, v84 op_sel_hi:[0,1,1]
v_pk_mul_f16 v86, v86, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v96, v1, s[40:43], 0 idxen
buffer_load_short_d16 v99, v36, s[40:43], 0 idxen
ds_write_b32 v104, v82 offset:24768
ds_read_b128 v[42:45], v106 offset:512
ds_write_b32 v105, v83 offset:24768
ds_read_b128 v[54:57], v101
ds_read_b128 v[58:61], v101 offset:128
s_waitcnt vmcnt(8) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 2
s_call_b64 s[36:37], 633
s_nop 0
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v64, v88, v64
v_pack_b32_f16 v65, v89, v65
v_pack_b32_f16 v66, v90, v66
v_pack_b32_f16 v67, v91, v67
v_mov_b32_dpp v114, v64 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v64, v114, v111, v64
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v76, v1, s[40:43], 0 idxen
buffer_load_short_d16 v79, v36, s[40:43], 0 idxen
ds_write_b32 v102, v80 offset:24768
ds_read_b128 v[38:41], v106 offset:4672
ds_write_b32 v103, v81 offset:24768
ds_read_b128 v[46:49], v101 offset:4160
ds_read_b128 v[50:53], v101 offset:4288
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 2
s_call_b64 s[36:37], 529
s_nop 0
s_barrier
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_mov_b32_dpp v114, v67 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v67, v114, v111, v67
v_pk_add_f16 v65, v64, v67
v_pk_mul_f16 v65, v65, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v66, -1.0, v67, v64 op_sel_hi:[0,1,1]
v_pk_mul_f16 v66, v66, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v33, v43, v61, v33
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v88, v1, s[40:43], 0 idxen
buffer_load_short_d16 v91, v36, s[40:43], 0 idxen
ds_write_b32 v104, v86 offset:33024
ds_read_b128 v[42:45], v106 offset:8768
ds_write_b32 v105, v87 offset:33024
ds_read_b128 v[54:57], v101 offset:8256
ds_read_b128 v[58:61], v101 offset:8384
s_waitcnt vmcnt(8) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 1
s_call_b64 s[36:37], 424
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v68, v92, v68
v_pack_b32_f16 v69, v93, v69
v_pack_b32_f16 v70, v94, v70
v_pack_b32_f16 v71, v95, v71
v_mov_b32_dpp v114, v68 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v68, v114, v111, v68
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v80, v1, s[40:43], 0 idxen
buffer_load_short_d16 v83, v36, s[40:43], 0 idxen
ds_write_b32 v102, v84 offset:33024
ds_read_b128 v[38:41], v106 offset:12928
ds_write_b32 v103, v85 offset:33024
ds_read_b128 v[46:49], v101 offset:12416
ds_read_b128 v[50:53], v101 offset:12544
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 2
s_call_b64 s[36:37], 321
s_nop 0
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_dot2_f32_f16 v33, v43, v61, v33
v_mov_b32_dpp v114, v71 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v71, v114, v111, v71
v_pk_add_f16 v69, v68, v71
v_pk_mul_f16 v69, v69, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v70, -1.0, v71, v68 op_sel_hi:[0,1,1]
v_pk_mul_f16 v70, v70, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v92, v1, s[40:43], 0 idxen
buffer_load_short_d16 v95, v36, s[40:43], 0 idxen
ds_write_b32 v104, v66 offset:41280
ds_read_b128 v[42:45], v106 offset:17024
ds_write_b32 v105, v67 offset:41280
ds_read_b128 v[54:57], v101 offset:16512
ds_read_b128 v[58:61], v101 offset:16640
s_waitcnt vmcnt(8) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 2
s_call_b64 s[36:37], 217
s_nop 0
v_dot2_f32_f16 v4, v38, v46, v4
v_dot2_f32_f16 v5, v39, v46, v5
v_dot2_f32_f16 v6, v40, v46, v6
v_dot2_f32_f16 v7, v41, v46, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v38, v47, v8
v_dot2_f32_f16 v9, v39, v47, v9
v_dot2_f32_f16 v10, v40, v47, v10
v_dot2_f32_f16 v11, v41, v47, v11
v_dot2_f32_f16 v12, v38, v48, v12
v_dot2_f32_f16 v13, v39, v48, v13
v_dot2_f32_f16 v14, v40, v48, v14
v_dot2_f32_f16 v15, v41, v48, v15
v_dot2_f32_f16 v16, v38, v49, v16
v_dot2_f32_f16 v17, v39, v49, v17
v_dot2_f32_f16 v18, v40, v49, v18
v_dot2_f32_f16 v19, v41, v49, v19
v_dot2_f32_f16 v20, v38, v50, v20
v_dot2_f32_f16 v21, v39, v50, v21
v_dot2_f32_f16 v22, v40, v50, v22
v_dot2_f32_f16 v23, v41, v50, v23
v_dot2_f32_f16 v24, v38, v51, v24
v_dot2_f32_f16 v25, v39, v51, v25
v_dot2_f32_f16 v26, v40, v51, v26
v_dot2_f32_f16 v27, v41, v51, v27
v_dot2_f32_f16 v28, v38, v52, v28
v_dot2_f32_f16 v29, v39, v52, v29
v_dot2_f32_f16 v30, v40, v52, v30
v_dot2_f32_f16 v31, v41, v52, v31
v_dot2_f32_f16 v32, v38, v53, v32
v_dot2_f32_f16 v33, v39, v53, v33
v_pack_b32_f16 v72, v96, v72
v_pack_b32_f16 v73, v97, v73
v_pack_b32_f16 v74, v98, v74
v_pack_b32_f16 v75, v99, v75
v_mov_b32_dpp v114, v72 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v72, v114, v111, v72
v_dot2_f32_f16 v34, v40, v53, v34
v_dot2_f32_f16 v35, v41, v53, v35
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v84, v1, s[40:43], 0 idxen
buffer_load_short_d16 v87, v36, s[40:43], 0 idxen
ds_write_b32 v102, v64 offset:41280
ds_read_b128 v[38:41], v106 offset:21184
ds_write_b32 v103, v65 offset:41280
ds_read_b128 v[46:49], v101 offset:20672
ds_read_b128 v[50:53], v101 offset:20800
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 2
s_call_b64 s[36:37], 113
s_nop 0
s_barrier
v_dot2_f32_f16 v4, v42, v54, v4
v_dot2_f32_f16 v5, v43, v54, v5
v_dot2_f32_f16 v6, v44, v54, v6
v_dot2_f32_f16 v7, v45, v54, v7
s_setprio 1
s_nop 0
v_dot2_f32_f16 v8, v42, v55, v8
v_dot2_f32_f16 v9, v43, v55, v9
v_dot2_f32_f16 v10, v44, v55, v10
v_dot2_f32_f16 v11, v45, v55, v11
v_dot2_f32_f16 v12, v42, v56, v12
v_dot2_f32_f16 v13, v43, v56, v13
v_dot2_f32_f16 v14, v44, v56, v14
v_dot2_f32_f16 v15, v45, v56, v15
v_dot2_f32_f16 v16, v42, v57, v16
v_dot2_f32_f16 v17, v43, v57, v17
v_dot2_f32_f16 v18, v44, v57, v18
v_dot2_f32_f16 v19, v45, v57, v19
v_dot2_f32_f16 v20, v42, v58, v20
v_dot2_f32_f16 v21, v43, v58, v21
v_dot2_f32_f16 v22, v44, v58, v22
v_dot2_f32_f16 v23, v45, v58, v23
v_dot2_f32_f16 v24, v42, v59, v24
v_dot2_f32_f16 v25, v43, v59, v25
v_dot2_f32_f16 v26, v44, v59, v26
v_dot2_f32_f16 v27, v45, v59, v27
v_dot2_f32_f16 v28, v42, v60, v28
v_dot2_f32_f16 v29, v43, v60, v29
v_dot2_f32_f16 v30, v44, v60, v30
v_dot2_f32_f16 v31, v45, v60, v31
v_dot2_f32_f16 v32, v42, v61, v32
v_mov_b32_dpp v114, v75 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v75, v114, v111, v75
v_pk_add_f16 v73, v72, v75
v_pk_mul_f16 v73, v73, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v74, -1.0, v75, v72 op_sel_hi:[0,1,1]
v_pk_mul_f16 v74, v74, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v33, v43, v61, v33
v_dot2_f32_f16 v34, v44, v61, v34
v_dot2_f32_f16 v35, v45, v61, v35
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v96, v1, s[40:43], 0 idxen
buffer_load_short_d16 v99, v36, s[40:43], 0 idxen
ds_write_b32 v104, v70
ds_read_b128 v[42:45], v106 offset:25280
ds_write_b32 v105, v71
ds_read_b128 v[54:57], v101 offset:24768
ds_read_b128 v[58:61], v101 offset:24896
s_waitcnt vmcnt(8) lgkmcnt(5)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 64289
s_call_b64 s[36:37], 8
s_branch 64287
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_nop
s_cmp_eq_u32 s64, 0
s_cbranch_scc0 8
s_branch 612
s_add_u32 s64, s64, 3
s_andn2_b32 s64, s64, 3
s_bitcmp0_b32 s14, 26
s_cselect_b32 s92, s51, s52
s_cselect_b32 s93, 0, s53
s_sub_u32 s40, s40, s92
s_subb_u32 s41, s41, s93
s_cmp_eq_u32 s76, 0
s_cbranch_scc0 5
s_cbranch_scc1 618
s_nop 0
s_nop 0
s_add_u32 s76, s76, 1
s_andn2_b32 s76, s76, 1
s_min_u32 s54, s64, s76
s_sub_u32 s64, s64, s54
s_sub_u32 s76, s76, s54
s_sub_u32 s54, s54, 2
s_setpc_b64 s[36:37]
s_nop 0
s_bitcmp1_b32 s14, 17
s_cbranch_scc1 241
s_add_u32 s70, s70, s13
s_cmp_eq_u32 s70, 0
s_cbranch_scc1 238
s_mov_b32 s71, 0
s_bitcmp1_b32 s14, 16
s_cbranch_scc1 227
s_add_u32 s69, s12, 31
s_lshr_b32 s69, s69, 5
v_mov_b32_e32 v117, s70
v_mul_u32_u24_e32 v117, s69, v117
v_add_co_u32_e32 v117, vcc, s13, v117
v_sub_co_u32_e64 v117, vcc, v117, 1
v_ffbh_u32_e32 v120, s13
v_lshlrev_b32_e64 v121, v120, s13
v_and_b32_e32 v122, 0xffffff00, v121
v_cmp_eq_u32_e32 vcc, 0x80000000, v121
v_cvt_f32_u32_e32 v122, v122
v_rcp_f32_e32 v116, v122
v_subb_co_u32_e32 v119, vcc, 32, v120, vcc
v_cvt_f32_ubyte0_e32 v120, v121
v_fma_f32 v122, v122, v116, -1.0
v_fma_f32 v122, v120, v116, v122
v_fmaak_f32 v122, v122, v116, 0x9f000000
v_mul_f32_e32 v122, 0x5f800000, v122
v_mov_b32_e32 v120, 0
v_cvt_flr_i32_f32_e64 v122, -v122
v_lshl_add_u32 v116, v116, 9, v122
v_mad_u64_u32 v[120:121], vcc, v121, v116, v[120:121]
v_subb_co_u32_e64 v116, vcc, v116, -1, vcc
v_mul_hi_u32 v120, v117, v116
v_add_co_u32_e32 v116, vcc, v120, v117
v_addc_co_u32_e64 v120, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v119
v_cndmask_b32_e32 v116, v116, v120, vcc
v_alignbit_b32 v116, v120, v116, v119
s_nop 0
v_readfirstlane_b32 s68, v116
v_mul_u32_u24_e64 v116, v116, s4
v_ffbh_u32_e32 v120, s69
v_lshlrev_b32_e64 v121, v120, s69
v_and_b32_e32 v122, 0xffffff00, v121
v_cmp_eq_u32_e32 vcc, 0x80000000, v121
v_cvt_f32_u32_e32 v122, v122
v_rcp_f32_e32 v117, v122
v_subb_co_u32_e32 v119, vcc, 32, v120, vcc
v_cvt_f32_ubyte0_e32 v120, v121
v_fma_f32 v122, v122, v117, -1.0
v_fma_f32 v122, v120, v117, v122
v_fmaak_f32 v122, v122, v117, 0x9f000000
v_mul_f32_e32 v122, 0x5f800000, v122
v_mov_b32_e32 v120, 0
v_cvt_flr_i32_f32_e64 v122, -v122
v_lshl_add_u32 v117, v117, 9, v122
v_mad_u64_u32 v[120:121], vcc, v121, v117, v[120:121]
v_subb_co_u32_e64 v117, vcc, v117, -1, vcc
v_mul_hi_u32 v120, v116, v117
v_add_co_u32_e32 v117, vcc, v120, v116
v_addc_co_u32_e64 v120, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v119
v_cndmask_b32_e32 v117, v117, v120, vcc
v_alignbit_b32 v117, v120, v117, v119
v_readfirstlane_b32 s2, v116
v_readfirstlane_b32 s66, v117
s_mul_i32 s66, s66, s69
s_sub_u32 s66, s2, s66
v_sub_co_u32_e32 v117, vcc, s4, v117
v_sub_co_u32_e32 v117, vcc, s13, v117
v_and_b32_e64 v119, v0, 63
v_cmp_eq_u32_e64 vcc, v119, 0
v_cndmask_b32_e32 v117, 1, v117, vcc
s_sub_u32 s3, 0, s57
s_sub_u32 s15, 0, s56
v_mul_u32_u24_e64 v121, v117, 32
v_ffbh_u32_e32 v124, s3
v_lshlrev_b32_e64 v125, v124, s3
v_and_b32_e32 v123, 0xffffff00, v125
v_cmp_eq_u32_e32 vcc, 0x80000000, v125
v_cvt_f32_u32_e32 v123, v123
v_rcp_f32_e32 v119, v123
v_subb_co_u32_e32 v122, vcc, 32, v124, vcc
v_cvt_f32_ubyte0_e32 v124, v125
v_fma_f32 v123, v123, v119, -1.0
v_fma_f32 v123, v124, v119, v123
v_fmaak_f32 v123, v123, v119, 0x9f000000
v_mul_f32_e32 v123, 0x5f800000, v123
v_mov_b32_e32 v124, 0
v_cvt_flr_i32_f32_e64 v123, -v123
v_lshl_add_u32 v119, v119, 9, v123
v_mad_u64_u32 v[124:125], vcc, v125, v119, v[124:125]
v_subb_co_u32_e64 v119, vcc, v119, -1, vcc
v_mul_hi_u32 v123, v121, v119
v_add_co_u32_e32 v119, vcc, v123, v121
v_addc_co_u32_e64 v123, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v122
v_cndmask_b32_e32 v119, v119, v123, vcc
v_alignbit_b32 v119, v123, v119, v122
v_mad_i32_i24 v120, v119, s57, v121
v_mul_u32_u24_e64 v121, v119, 1
v_ffbh_u32_e32 v124, s15
v_lshlrev_b32_e64 v125, v124, s15
v_and_b32_e32 v123, 0xffffff00, v125
v_cmp_eq_u32_e32 vcc, 0x80000000, v125
v_cvt_f32_u32_e32 v123, v123
v_rcp_f32_e32 v119, v123
v_subb_co_u32_e32 v122, vcc, 32, v124, vcc
v_cvt_f32_ubyte0_e32 v124, v125
v_fma_f32 v123, v123, v119, -1.0
v_fma_f32 v123, v124, v119, v123
v_fmaak_f32 v123, v123, v119, 0x9f000000
v_mul_f32_e32 v123, 0x5f800000, v123
v_mov_b32_e32 v124, 0
v_cvt_flr_i32_f32_e64 v123, -v123
v_lshl_add_u32 v119, v119, 9, v123
v_mad_u64_u32 v[124:125], vcc, v125, v119, v[124:125]
v_subb_co_u32_e64 v119, vcc, v119, -1, vcc
v_mul_hi_u32 v123, v121, v119
v_add_co_u32_e32 v119, vcc, v123, v121
v_addc_co_u32_e64 v123, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v122
v_cndmask_b32_e32 v119, v119, v123, vcc
v_alignbit_b32 v119, v123, v119, v122
v_mad_i32_i24 v121, v119, s56, v121
v_readfirstlane_b32 s58, v120
v_readfirstlane_b32 s59, v121
v_readfirstlane_b32 s60, v119
v_add_co_u32_e32 v107, vcc, s58, v107
v_addc_co_u32_e64 v122, vcc, 0, 0, vcc
v_mad_i32_i24 v107, v122, s57, v107
v_mad_i32_i24 v109, v122, s62, v109
v_mad_i32_i24 v108, v122, s61, v108
v_cmp_ge_i32_e64 vcc, v108, 0
v_addc_co_u32_e64 v122, vcc, 0, 0, vcc
v_add_co_u32_e32 v109, vcc, v109, v122
v_mad_i32_i24 v108, v122, s56, v108
v_add_co_u32_e32 v108, vcc, s59, v108
v_addc_co_u32_e64 v122, vcc, 0, 0, vcc
v_add_co_u32_e32 v109, vcc, v109, v122
v_mad_i32_i24 v108, v122, s56, v108
v_add_co_u32_e32 v109, vcc, s60, v109
v_readlane_b32 s58, v120, 1
v_readlane_b32 s59, v121, 1
v_readlane_b32 s60, v119, 1
s_add_u32 s67, s66, s68
s_cmp_le_u32 s67, s69
s_cselect_b32 s92, 0x20000, 0
s_cselect_b32 s67, s67, s69
s_or_b32 s14, s14, s92
s_lshl_b32 s66, s66, 5
s_lshl_b32 s67, s67, 5
s_min_u32 s67, s67, s12
s_cmp_eq_u32 s4, s13
s_cselect_b32 s92, 0x20000, 0
s_or_b32 s14, s14, s92
s_bitset1_b32 s14, 16
s_branch 43
s_lshr_b32 s66, s66, 5
s_add_u32 s67, s66, s68
s_sub_u32 s67, s67, s69
s_mov_b32 s66, 0
s_lshl_b32 s67, s67, 5
s_min_u32 s67, s67, s12
s_bitset1_b32 s14, 17
s_branch 12
s_bitset1_b32 s14, 18
s_mov_b32 s43, 0
s_mov_b32 s55, -1
s_mov_b32 s64, 40
s_branch 31
s_add_u32 s65, s65, 32
s_cmp_ge_u32 s65, s67
s_cbranch_scc0 28
s_bitset1_b32 s14, 22
s_sub_u32 s70, s70, s13
s_subb_u32 s71, s71, 0
s_cbranch_scc1 65281
v_add_co_u32_e32 v107, vcc, s58, v107
v_addc_co_u32_e64 v116, vcc, 0, 0, vcc
v_mad_i32_i24 v107, v116, s57, v107
v_mad_i32_i24 v109, v116, s62, v109
v_mad_i32_i24 v108, v116, s61, v108
v_cmp_ge_i32_e64 vcc, v108, 0
v_addc_co_u32_e64 v116, vcc, 0, 0, vcc
v_add_co_u32_e32 v109, vcc, v109, v116
v_mad_i32_i24 v108, v116, s56, v108
v_add_co_u32_e32 v108, vcc, s59, v108
v_addc_co_u32_e64 v116, vcc, 0, 0, vcc
v_add_co_u32_e32 v109, vcc, v109, v116
v_mad_i32_i24 v108, v116, s56, v108
v_add_co_u32_e32 v109, vcc, s60, v109
s_mov_b32 s65, s66
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccz 172
v_subrev_co_u32_e32 v116, vcc, s57, v107
v_subrev_co_u32_e32 v117, vcc, s56, v108
s_bitcmp1_b32 s14, 22
s_cbranch_scc0 64
s_bitset0_b32 s14, 22
s_bfe_u32 s2, s14, 0x10014
v_mul_u32_u24_e32 v119, 3, v116
v_mul_u32_u24_e32 v120, 3, v117
v_cvt_pk_u16_u32 v122, v119, v120
v_and_b32_e64 v119, v0, 1
v_cmp_eq_u32_e64 vcc, v119, 1
v_cndmask_b32_e32 v122, v109, v122, vcc
v_lshrrev_b32_e32 v118, 1, v0
v_bfe_u32 v123, v118, s2, 1
v_lshrrev_b32_e32 v118, 1, v0
v_bfi_b32 v118, 1, v0, v118
v_lshrrev_b32_e32 v119, 2, v0
v_bfi_b32 v119, 1, v0, v119
v_cmp_eq_u32_e64 vcc, s2, 0
v_cndmask_b32_e32 v118, v119, v118, vcc
s_sub_u32 s2, 1, s2
v_lshrrev_b32_e32 v119, s2, v118
v_bfi_b32 v118, 32, v119, v118
v_and_b32_e32 v118, 63, v118
v_add_co_u32_e32 v119, vcc, 16, v118
v_and_b32_e64 v120, v0, 2
v_cmp_eq_u32_e64 vcc, v120, 0
v_cndmask_b32_e32 v119, v119, v118, vcc
v_lshlrev_b32_e32 v120, 14, v123
v_mad_u32_u24 v119, 4, v119, v120
v_add_co_u32_e32 v118, vcc, s78, v119
ds_write_b32 v118, v122
v_writelane_b32 v120, s14, 0
v_writelane_b32 v120, s67, 1
v_writelane_b32 v120, s66, 2
v_and_b32_e64 v118, v0, 63
v_cmp_ge_u32_e64 vcc, v118, 3
v_mov_b32_e32 v121, 0x4000
v_cndmask_b32_e32 v118, v118, v121, vcc
v_mad_i32_i24 v118, v118, 4, s78
ds_write_b32 v118, v120 offset:256
s_add_u32 s78, s78, 0x18c
s_cmp_eq_u32 s78, 0x10000
s_cselect_b32 s78, 0xc220, s78
v_mov_b32_dpp v118, v109 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v116, v116 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v117, v117 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_readfirstlane_b32 s63, v118
v_sub_co_u32_e64 v119, vcc, v118, s63
v_mul_lo_u32 v119, v119, s44
v_and_b32_e64 v123, v0, 3
v_ashrrev_i32_e64 v124, 0, s27
v_subrev_co_u32_e32 v123, vcc, v124, v123
v_ashrrev_i32_e64 v124, 0, s35
v_mad_i32_i24 v120, v124, 2, v123
s_bfe_u32 s2, s14, 0x10014
v_lshrrev_b32_e32 v122, 2, v0
v_and_b32_e32 v122, s2, v122
v_mad_i32_i24 v120, v122, 2, v120
v_add_co_u32_e64 v121, vcc, 0, s38
v_ashrrev_i32_e32 v121, 0, v121
v_add_co_u32_e64 v122, vcc, 0, s26
v_ashrrev_i32_e32 v122, 0, v122
v_sub_i32 v121, v121, v122
v_cmp_ge_u32_e64 s[2:3], v118, s8
v_mad_i32_i24 v116, v116, 3, v120
v_cmp_ge_u32_e64 s[22:23], v116, s11
v_add_co_u32_e32 v116, vcc, v116, v119
s_or_b64 s[22:23], s[22:23], s[2:3]
v_mad_i32_i24 v117, v117, 3, v121
v_cmp_ge_u32_e64 s[96:97], v117, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v1, v117, s49, v116
v_cndmask_b32_e64 v1, v1, -1, s[96:97]
v_add_co_u32_e32 v117, vcc, 1, v117
v_cmp_ge_u32_e64 s[96:97], v117, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v2, v117, s49, v116
v_cndmask_b32_e64 v2, v2, -1, s[96:97]
v_add_co_u32_e32 v117, vcc, 1, v117
v_cmp_ge_u32_e64 s[96:97], v117, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v3, v117, s49, v116
v_cndmask_b32_e64 v3, v3, -1, s[96:97]
v_add_co_u32_e32 v117, vcc, 1, v117
v_cmp_ge_u32_e64 s[96:97], v117, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v36, v117, s49, v116
v_cndmask_b32_e64 v36, v36, -1, s[96:97]
s_bitcmp1_b32 s14, 18
s_cbranch_scc1 160
s_lshr_b32 s15, -1, 16
s_and_b32 s15, s15, s44
s_lshr_b32 s22, s44, 16
s_mul_i32 s22, s22, s63
s_mul_i32 s40, s15, s63
s_lshl_b32 s15, s22, 16
s_lshr_b32 s22, s22, 16
s_add_u32 s40, s15, s40
s_addc_u32 s41, s22, 0
s_lshl_b64 s[40:41], s[40:41], 1
s_add_u32 s40, s40, s16
s_addc_u32 s41, s41, s17
s_lshr_b32 s2, s14, 6
s_xor_b32 s2, s2, s14
s_and_b32 s2, s2, 0x80000
s_cselect_b32 s2, s48, 0
s_and_b32 s3, s14, 0x1100000
s_cselect_b32 s3, 0, 1
s_lshl_b32 s2, s2, s3
s_lshl_b32 s2, s2, 1
s_add_u32 s40, s40, s2
s_addc_u32 s41, s41, 0
s_add_u32 s41, s41, 0x20000
s_branch 101
s_bitcmp1_b32 s14, 18
s_cbranch_scc1 131
s_bfe_u32 s2, s14, 0x10014
v_xor_b32_dpp v116, v0, v0 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_bfe_u32 v118, v0, 2, s2
v_mad_u32_u24 v116, v118, 2, v116
v_mad_u32_u24 v116, s35, 2, v116
v_sub_co_u32_e32 v118, vcc, s25, v116
v_sub_co_u32_e64 v118, vcc, v118, 1
s_bfe_u32 s2, s14, 0x10001
v_cmp_eq_u32_e64 vcc, s2, 1
v_cndmask_b32_e32 v116, v116, v118, vcc
v_cmp_ge_u32_e64 s[2:3], v116, s25
s_bfe_u32 s15, s14, 0x10018
v_bfe_u32 v119, v0, 2, s15
v_mul_lo_u32 v119, s48, v119
v_add_co_u32_e32 v116, vcc, v116, v119
v_mul_lo_u32 v117, s72, v110
v_add_co_u32_e32 v117, vcc, v117, v116
s_sub_u32 s15, s24, s38
s_sub_u32 s15, s15, 2
s_bitcmp1_b32 s14, 0
s_cselect_b32 s15, s15, s38
v_mov_b32_e32 v119, s15
v_cmp_ge_u32_e64 s[22:23], v119, s24
v_mad_i32_i24 v1, v119, s73, v117
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v1, v1, -1, s[22:23]
v_mov_b32_e32 v2, v1
v_add_co_u32_e64 v119, vcc, v119, 1
v_cmp_ge_u32_e64 s[22:23], v119, s24
v_mad_i32_i24 v36, v119, s73, v117
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v36, v36, -1, s[22:23]
v_add_co_u32_e64 v119, vcc, v119, 1
v_cmp_ge_u32_e64 s[22:23], v119, s24
v_mad_i32_i24 v3, v119, s73, v117
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v3, v3, -1, s[22:23]
s_bitcmp1_b32 s14, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v1, v2, v36, vcc
v_cndmask_b32_e32 v36, v36, v2, vcc
v_add_co_u32_e64 v116, vcc, v110, s65
v_cmp_lt_u32_e64 vcc, v116, s12
v_cndmask_b32_e32 v1, -1, v1, vcc
v_cndmask_b32_e32 v2, -1, v2, vcc
v_cndmask_b32_e32 v3, -1, v3, vcc
v_cndmask_b32_e32 v36, -1, v36, vcc
s_lshr_b32 s15, -1, 16
s_and_b32 s15, s15, s72
s_lshr_b32 s22, s72, 16
s_mul_i32 s22, s22, s65
s_mul_i32 s40, s15, s65
s_lshl_b32 s15, s22, 16
s_lshr_b32 s22, s22, 16
s_add_u32 s40, s15, s40
s_addc_u32 s41, s22, 0
s_lshl_b64 s[40:41], s[40:41], 1
s_add_u32 s40, s40, s18
s_addc_u32 s41, s41, s19
s_lshr_b32 s2, s14, 6
s_xor_b32 s2, s2, s14
s_and_b32 s2, s2, 0x80000
s_cselect_b32 s2, s48, 0
s_and_b32 s3, s14, 0x1100000
s_cselect_b32 s3, 0, 1
s_lshl_b32 s2, s2, s3
s_lshl_b32 s2, s2, 1
s_add_u32 s40, s40, s2
s_addc_u32 s41, s41, 0
s_add_u32 s41, s41, 0x20000
s_mov_b32 s43, 0x20000
s_mov_b32 s55, -1
s_bitcmp0_b32 s9, 0
s_cbranch_scc1 4
s_mov_b32 s43, 0
s_mov_b32 s55, 1
s_sub_u32 s40, s40, s47
s_subb_u32 s41, s41, 0
s_add_u32 s93, s9, 1
s_and_b32 s93, s93, -2
s_bfe_u32 s92, s14, 0x10014
s_lshl_b32 s64, s93, s92
s_bfe_u32 s92, s14, 0x10013
s_bfe_u32 s94, s14, 0x10019
s_xor_b32 s92, s92, s94
s_cselect_b32 s92, 2, 0
s_cselect_b32 s43, 0x20000, s43
s_and_b32 s92, s92, s64
s_sub_u32 s64, s64, s92
s_bitcmp1_b32 s14, 20
s_cselect_b32 s92, 0, 0x2000000
s_bitcmp1_b32 s93, 1
s_cselect_b32 s92, s92, 0
s_xor_b32 s14, s14, s92
s_cmp_eq_u32 s64, 0
s_cbranch_scc1 5
s_branch 64928
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_and_b32 s92, 0x900000, s14
s_subb_u32 s35, s35, 1
s_cbranch_scc0 65219
s_and_b32 s92, 0x900000, s14
s_subb_u32 s35, s5, 1
s_add_u32 s38, s38, 2
s_cmp_ge_u32 s38, s24
s_cbranch_scc0 65213
s_mov_b32 s38, 0
s_branch 65180
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s2, 0x3c3c3c3c
s_mov_b32 s3, s2
v_mov_b32_e32 v116, v4
v_mov_b32_e32 v117, v5
v_mov_b32_e32 v118, v6
v_mov_b32_e32 v119, v7
v_add_f32_dpp v116, v4, v4 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v117, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v118, v6, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v119, v7, v7 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v6, v6, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v7, v7, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v4, v4, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v5, v5, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v5, v6 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v4, v7 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v117, v118, v117 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v116, v119, v116 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v119, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v5, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v118, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v4, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v7, v117
v_add_f32_dpp v7, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v119 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v118, v119 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v6, v116
v_add_f32_dpp v6, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v118 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v119, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v5, v5, v4 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v4, v7, v6 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v6, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v6, v119 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v118, v118 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v5, v118, v5, s[2:3]
v_mov_b32_dpp v6, v6 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v6, v6 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v4, v4
v_cvt_f16_f32_e32 v5, v5
v_cvt_f16_f32_e32 v6, v6
v_mov_b32_e32 v116, v8
v_mov_b32_e32 v117, v9
v_mov_b32_e32 v118, v10
v_mov_b32_e32 v119, v11
v_add_f32_dpp v116, v8, v8 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v117, v9, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v118, v10, v10 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v119, v11, v11 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v10, v10, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v11, v11, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v8, v8, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v9, v9, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v9, v10 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v8, v11 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v117, v118, v117 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v116, v119, v116 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v119, v9, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v9, v9, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v118, v8, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v8, v8, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v11, v117
v_add_f32_dpp v11, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v119 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v118, v119 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v10, v116
v_add_f32_dpp v10, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v118 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v119, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v9, v9, v8 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v7, v11, v10 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v10, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v10, v119 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v118, v118 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v8, v118, v9, s[2:3]
v_mov_b32_dpp v9, v10 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v9, v10 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v7, v7
v_cvt_f16_f32_e32 v8, v8
v_cvt_f16_f32_e32 v9, v9
v_mov_b32_e32 v116, v12
v_mov_b32_e32 v117, v13
v_mov_b32_e32 v118, v14
v_mov_b32_e32 v119, v15
v_add_f32_dpp v116, v12, v12 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v117, v13, v13 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v118, v14, v14 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v119, v15, v15 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v14, v14, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v15, v15, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v12, v12, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v13, v13, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v13, v14 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v12, v15 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v117, v118, v117 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v116, v119, v116 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v119, v13, v13 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v13, v13, v13 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v118, v12, v12 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v12, v12, v12 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v15, v117
v_add_f32_dpp v15, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v119 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v118, v119 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v14, v116
v_add_f32_dpp v14, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v118 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v119, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v13, v13, v12 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v10, v15, v14 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v14, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v14, v119 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v118, v118 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v11, v118, v13, s[2:3]
v_mov_b32_dpp v12, v14 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v12, v14 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v10, v10
v_cvt_f16_f32_e32 v11, v11
v_cvt_f16_f32_e32 v12, v12
v_mov_b32_e32 v116, v16
v_mov_b32_e32 v117, v17
v_mov_b32_e32 v118, v18
v_mov_b32_e32 v119, v19
v_add_f32_dpp v116, v16, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v117, v17, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v118, v18, v18 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v119, v19, v19 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v18, v18, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v19, v19, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v16, v16, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v17, v17, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v17, v18 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v16, v19 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v117, v118, v117 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v116, v119, v116 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v119, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v17, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v118, v16, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v16, v16, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v19, v117
v_add_f32_dpp v19, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v119 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v118, v119 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v18, v116
v_add_f32_dpp v18, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v118 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v119, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v17, v17, v16 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v13, v19, v18 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v18, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v18, v119 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v118, v118 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v14, v118, v17, s[2:3]
v_mov_b32_dpp v15, v18 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v15, v18 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v13, v13
v_cvt_f16_f32_e32 v14, v14
v_cvt_f16_f32_e32 v15, v15
v_mov_b32_e32 v116, v20
v_mov_b32_e32 v117, v21
v_mov_b32_e32 v118, v22
v_mov_b32_e32 v119, v23
v_add_f32_dpp v116, v20, v20 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v117, v21, v21 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v118, v22, v22 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v119, v23, v23 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v22, v22, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v23, v23, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v20, v20, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v21, v21, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v21, v22 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v20, v23 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v117, v118, v117 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v116, v119, v116 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v119, v21, v21 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v21, v21, v21 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v118, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v20, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v23, v117
v_add_f32_dpp v23, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v119 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v118, v119 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v22, v116
v_add_f32_dpp v22, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v118 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v119, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v21, v21, v20 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v16, v23, v22 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v22, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v22, v119 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v118, v118 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v17, v118, v21, s[2:3]
v_mov_b32_dpp v18, v22 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v18, v22 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v16, v16
v_cvt_f16_f32_e32 v17, v17
v_cvt_f16_f32_e32 v18, v18
v_mov_b32_e32 v116, v24
v_mov_b32_e32 v117, v25
v_mov_b32_e32 v118, v26
v_mov_b32_e32 v119, v27
v_add_f32_dpp v116, v24, v24 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v117, v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v118, v26, v26 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v119, v27, v27 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v26, v26, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v27, v27, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v24, v24, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v25, v25, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v25, v26 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v24, v27 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v117, v118, v117 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v116, v119, v116 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v119, v25, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v25, v25, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v118, v24, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v24, v24, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v27, v117
v_add_f32_dpp v27, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v119 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v118, v119 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v26, v116
v_add_f32_dpp v26, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v118 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v119, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v25, v25, v24 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v19, v27, v26 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v26, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v26, v119 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v118, v118 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v20, v118, v25, s[2:3]
v_mov_b32_dpp v21, v26 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v21, v26 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v19, v19
v_cvt_f16_f32_e32 v20, v20
v_cvt_f16_f32_e32 v21, v21
v_mov_b32_e32 v116, v28
v_mov_b32_e32 v117, v29
v_mov_b32_e32 v118, v30
v_mov_b32_e32 v119, v31
v_add_f32_dpp v116, v28, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v117, v29, v29 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v118, v30, v30 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v119, v31, v31 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v30, v30, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v31, v31, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v28, v28, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v29, v29, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v29, v30 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v28, v31 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v117, v118, v117 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v116, v119, v116 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v119, v29, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v29, v29, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v118, v28, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v28, v28, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v31, v117
v_add_f32_dpp v31, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v119 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v118, v119 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v30, v116
v_add_f32_dpp v30, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v118 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v119, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v29, v29, v28 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v22, v31, v30 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v30, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v30, v119 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v118, v118 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v23, v118, v29, s[2:3]
v_mov_b32_dpp v24, v30 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v24, v30 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v22, v22
v_cvt_f16_f32_e32 v23, v23
v_cvt_f16_f32_e32 v24, v24
s_setprio 1
v_mov_b32_e32 v116, v32
v_mov_b32_e32 v117, v33
v_mov_b32_e32 v118, v34
v_mov_b32_e32 v119, v35
v_add_f32_dpp v116, v32, v32 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v117, v33, v33 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v118, v34, v34 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v119, v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v34, v34, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v35, v35, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v32, v32, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v33, v33, v112 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v33, v34 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v32, v35 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v117, v118, v117 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v116, v119, v116 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v119, v33, v33 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v33, v33, v33 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v118, v32, v32 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v32, v32, v32 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v35, v117
v_add_f32_dpp v35, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v119 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v118, v119 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v34, v116
v_add_f32_dpp v34, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v118, v118 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v119, v117, v117 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v33, v33, v32 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v25, v35, v34 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v34, v116, v116 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v34, v119 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v118, v118 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v26, v118, v33, s[2:3]
v_mov_b32_dpp v27, v34 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v27, v34 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v25, v25
v_cvt_f16_f32_e32 v26, v26
v_cvt_f16_f32_e32 v27, v27
s_waitcnt vmcnt(0)
s_setpc_b64 s[80:81]
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v9, 0
v_mov_b32_e32 v10, 0
v_mov_b32_e32 v11, 0
v_mov_b32_e32 v12, 0
v_mov_b32_e32 v13, 0
v_mov_b32_e32 v14, 0
v_mov_b32_e32 v15, 0
v_mov_b32_e32 v16, 0
v_mov_b32_e32 v17, 0
v_mov_b32_e32 v18, 0
v_mov_b32_e32 v19, 0
v_mov_b32_e32 v20, 0
v_mov_b32_e32 v21, 0
v_mov_b32_e32 v22, 0
v_mov_b32_e32 v23, 0
v_mov_b32_e32 v24, 0
v_mov_b32_e32 v25, 0
v_mov_b32_e32 v26, 0
v_mov_b32_e32 v27, 0
v_mov_b32_e32 v28, 0
v_mov_b32_e32 v29, 0
v_mov_b32_e32 v30, 0
v_mov_b32_e32 v31, 0
v_mov_b32_e32 v32, 0
v_mov_b32_e32 v33, 0
v_mov_b32_e32 v34, 0
v_mov_b32_e32 v35, 0
s_xor_b32 s14, s14, 0x200000
s_bitcmp1_b32 s9, 0
s_addc_u32 s92, s9, 0
s_mul_i32 s76, s1, s5
s_mul_i32 s76, s76, s92
s_add_u32 s92, s75, s74
s_cmp_lt_i32 s92, 0
s_cbranch_scc0 158
v_and_b32_e32 v37, 0x7f, v0
v_lshrrev_b32_e32 v37, 1, v37
v_bfi_b32 v37, 1, v0, v37
v_and_b32_e64 v62, v0, 2
v_mad_u32_u24 v37, v62, 16, v37
v_lshlrev_b32_e32 v37, 2, v37
v_add_co_u32_e64 v37, vcc, v37, s79
v_and_b32_e32 v62, 3, v0
v_lshlrev_b32_e32 v62, 2, v62
v_add_co_u32_e64 v62, vcc, v62, s79
ds_read_b32 v116, v62 offset:256
ds_read_b32 v37, v37
s_add_u32 s79, s79, 0x18c
s_cmp_eq_u32 s79, 0x10000
s_cselect_b32 s79, 0xc220, s79
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s77, v37
v_readlane_b32 s94, v116, 0
s_bitcmp1_b32 s94, 18
s_cbranch_scc1 1411
v_readlane_b32 s92, v116, 1
v_readlane_b32 s93, v116, 2
s_add_u32 s75, s74, s93
s_lshr_b32 s3, -1, 16
s_and_b32 s3, s3, s45
s_lshr_b32 s15, s45, 16
s_mul_i32 s15, s15, s77
s_mul_i32 s84, s3, s77
s_lshl_b32 s3, s15, 16
s_lshr_b32 s15, s15, 16
s_add_u32 s84, s3, s84
s_addc_u32 s85, s15, 0
s_lshl_b64 s[84:85], s[84:85], 1
s_add_u32 s84, s84, s20
s_addc_u32 s85, s85, s21
s_mul_i32 s22, s46, s75
s_lshl_b32 s22, s22, 1
s_add_u32 s84, s84, s22
s_addc_u32 s85, s85, 0
s_add_u32 s85, s85, 0x20000
s_mov_b32 s87, 0x20000
s_bitcmp1_b32 s14, 7
s_cselect_b32 s91, 0x20000, 0
s_lshl_b32 s2, s75, 1
s_add_u32 s88, s30, s2
s_addc_u32 s89, s31, 0
s_add_u32 s89, s89, 0x20000
s_sub_u32 s90, s92, s75
s_cselect_b32 s91, 0, s91
s_sub_u32 s75, s92, s93
s_sub_u32 s75, s75, 1
s_sub_u32 s75, s75, s74
s_cselect_b32 s87, 0, s87
v_bfe_u32 v116, v37, 16, 16
v_bfe_u32 v117, v37, 0, 16
v_and_b32_e64 v118, v0, 7
v_sub_co_u32_e32 v119, vcc, 7, v118
v_min_u32_e32 v118, v118, v119
v_bfe_u32 v119, v118, 1, 1
v_bfe_u32 v118, v118, 0, 1
v_mov_b32_dpp v116, v116 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v117, v117 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_add_co_u32_e32 v116, vcc, v116, v119
v_add_co_u32_e32 v117, vcc, v117, v118
v_mov_b32_dpp v118, v37 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[2:3], v118, s8
v_sub_co_u32_e64 v118, vcc, v118, s77
v_mul_lo_u32 v118, v118, s45
v_xor_b32_dpp v119, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v119, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v120, v0, v0 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v120, v0, v0 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_add_co_u32_e32 v120, vcc, v117, v120
v_add_co_u32_e32 v119, vcc, v116, v119
v_mad_i32_i24 v37, v119, s50, v120
v_add_co_u32_e32 v37, vcc, v37, v118
v_cmp_ge_u32_e64 s[96:97], v120, s29
s_or_b64 s[96:97], s[96:97], s[2:3]
v_cmp_ge_u32_e64 s[22:23], v119, s28
s_or_b64 s[96:97], s[96:97], s[22:23]
v_cndmask_b32_e64 v37, v37, -1, s[96:97]
v_xor_b32_dpp v119, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v119, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v120, v0, v0 quad_perm:[1,1,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32_e32 v120, vcc, v117, v120
v_add_co_u32_e32 v119, vcc, v116, v119
v_mad_i32_i24 v62, v119, s50, v120
v_add_co_u32_e32 v62, vcc, v62, v118
v_cmp_ge_u32_e64 s[96:97], v120, s29
s_or_b64 s[96:97], s[96:97], s[2:3]
v_cmp_ge_u32_e64 s[22:23], v119, s28
s_or_b64 s[96:97], s[96:97], s[22:23]
v_cndmask_b32_e64 v62, v62, -1, s[96:97]
v_xor_b32_dpp v119, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v119, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v120, v0, v0 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v120, v0, v0 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xa
v_add_co_u32_e32 v120, vcc, v117, v120
v_add_co_u32_e32 v119, vcc, v116, v119
v_mad_i32_i24 v63, v119, s50, v120
v_add_co_u32_e32 v63, vcc, v63, v118
v_cmp_ge_u32_e64 s[96:97], v120, s29
s_or_b64 s[96:97], s[96:97], s[2:3]
v_cmp_ge_u32_e64 s[22:23], v119, s28
s_or_b64 s[96:97], s[96:97], s[22:23]
v_cndmask_b32_e64 v63, v63, -1, s[96:97]
v_and_b32_e64 v115, v0, 63
buffer_load_ushort v115, v115, s[88:91], 0 idxen
s_branch 64150
v_readlane_b32 s99, v115, 0
v_add_f16_e64 v4, v4, s99
v_add_f16_e64 v5, v5, s99
v_add_f16_e64 v6, v6, s99
buffer_store_short v4, v37, s[84:87], 0 idxen
buffer_store_short v5, v62, s[84:87], 0 idxen
buffer_store_short v6, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 1
v_add_f16_e64 v7, v7, s99
v_add_f16_e64 v8, v8, s99
v_add_f16_e64 v9, v9, s99
buffer_store_short v7, v37, s[84:87], 0 idxen
buffer_store_short v8, v62, s[84:87], 0 idxen
buffer_store_short v9, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 2
v_add_f16_e64 v10, v10, s99
v_add_f16_e64 v11, v11, s99
v_add_f16_e64 v12, v12, s99
buffer_store_short v10, v37, s[84:87], 0 idxen
buffer_store_short v11, v62, s[84:87], 0 idxen
buffer_store_short v12, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 3
v_add_f16_e64 v13, v13, s99
v_add_f16_e64 v14, v14, s99
v_add_f16_e64 v15, v15, s99
buffer_store_short v13, v37, s[84:87], 0 idxen
buffer_store_short v14, v62, s[84:87], 0 idxen
buffer_store_short v15, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s99, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 4
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 8
v_add_f16_e64 v16, v16, s99
v_add_f16_e64 v17, v17, s99
v_add_f16_e64 v18, v18, s99
buffer_store_short v16, v37, s[84:87], 0 idxen
buffer_store_short v17, v62, s[84:87], 0 idxen
buffer_store_short v18, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 9
v_add_f16_e64 v19, v19, s99
v_add_f16_e64 v20, v20, s99
v_add_f16_e64 v21, v21, s99
buffer_store_short v19, v37, s[84:87], 0 idxen
buffer_store_short v20, v62, s[84:87], 0 idxen
buffer_store_short v21, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 10
v_add_f16_e64 v22, v22, s99
v_add_f16_e64 v23, v23, s99
v_add_f16_e64 v24, v24, s99
buffer_store_short v22, v37, s[84:87], 0 idxen
buffer_store_short v23, v62, s[84:87], 0 idxen
buffer_store_short v24, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 11
v_add_f16_e64 v25, v25, s99
v_add_f16_e64 v26, v26, s99
v_add_f16_e64 v27, v27, s99
buffer_store_short v25, v37, s[84:87], 0 idxen
buffer_store_short v26, v62, s[84:87], 0 idxen
buffer_store_short v27, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 20
s_cselect_b32 s87, 0, s87
s_cselect_b32 s91, 0, s91
s_add_u32 s88, s88, 64
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, 32
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
v_readlane_b32 s99, v115, 0
v_add_f16_e64 v4, v4, s99
v_mul_f16_e64 v117, v4, s32
v_cmp_lt_f16_e64 vcc, v4, 0
v_cndmask_b32_e32 v4, v4, v117, vcc
v_add_f16_e64 v5, v5, s99
v_mul_f16_e64 v117, v5, s32
v_cmp_lt_f16_e64 vcc, v5, 0
v_cndmask_b32_e32 v5, v5, v117, vcc
v_add_f16_e64 v6, v6, s99
v_mul_f16_e64 v117, v6, s32
v_cmp_lt_f16_e64 vcc, v6, 0
v_cndmask_b32_e32 v6, v6, v117, vcc
buffer_store_short v4, v37, s[84:87], 0 idxen
buffer_store_short v5, v62, s[84:87], 0 idxen
buffer_store_short v6, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 1
v_add_f16_e64 v7, v7, s99
v_mul_f16_e64 v117, v7, s32
v_cmp_lt_f16_e64 vcc, v7, 0
v_cndmask_b32_e32 v7, v7, v117, vcc
v_add_f16_e64 v8, v8, s99
v_mul_f16_e64 v117, v8, s32
v_cmp_lt_f16_e64 vcc, v8, 0
v_cndmask_b32_e32 v8, v8, v117, vcc
v_add_f16_e64 v9, v9, s99
v_mul_f16_e64 v117, v9, s32
v_cmp_lt_f16_e64 vcc, v9, 0
v_cndmask_b32_e32 v9, v9, v117, vcc
buffer_store_short v7, v37, s[84:87], 0 idxen
buffer_store_short v8, v62, s[84:87], 0 idxen
buffer_store_short v9, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 2
v_add_f16_e64 v10, v10, s99
v_mul_f16_e64 v117, v10, s32
v_cmp_lt_f16_e64 vcc, v10, 0
v_cndmask_b32_e32 v10, v10, v117, vcc
v_add_f16_e64 v11, v11, s99
v_mul_f16_e64 v117, v11, s32
v_cmp_lt_f16_e64 vcc, v11, 0
v_cndmask_b32_e32 v11, v11, v117, vcc
v_add_f16_e64 v12, v12, s99
v_mul_f16_e64 v117, v12, s32
v_cmp_lt_f16_e64 vcc, v12, 0
v_cndmask_b32_e32 v12, v12, v117, vcc
buffer_store_short v10, v37, s[84:87], 0 idxen
buffer_store_short v11, v62, s[84:87], 0 idxen
buffer_store_short v12, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 3
v_add_f16_e64 v13, v13, s99
v_mul_f16_e64 v117, v13, s32
v_cmp_lt_f16_e64 vcc, v13, 0
v_cndmask_b32_e32 v13, v13, v117, vcc
v_add_f16_e64 v14, v14, s99
v_mul_f16_e64 v117, v14, s32
v_cmp_lt_f16_e64 vcc, v14, 0
v_cndmask_b32_e32 v14, v14, v117, vcc
v_add_f16_e64 v15, v15, s99
v_mul_f16_e64 v117, v15, s32
v_cmp_lt_f16_e64 vcc, v15, 0
v_cndmask_b32_e32 v15, v15, v117, vcc
buffer_store_short v13, v37, s[84:87], 0 idxen
buffer_store_short v14, v62, s[84:87], 0 idxen
buffer_store_short v15, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s99, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 4
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 8
v_add_f16_e64 v16, v16, s99
v_mul_f16_e64 v117, v16, s32
v_cmp_lt_f16_e64 vcc, v16, 0
v_cndmask_b32_e32 v16, v16, v117, vcc
v_add_f16_e64 v17, v17, s99
v_mul_f16_e64 v117, v17, s32
v_cmp_lt_f16_e64 vcc, v17, 0
v_cndmask_b32_e32 v17, v17, v117, vcc
v_add_f16_e64 v18, v18, s99
v_mul_f16_e64 v117, v18, s32
v_cmp_lt_f16_e64 vcc, v18, 0
v_cndmask_b32_e32 v18, v18, v117, vcc
buffer_store_short v16, v37, s[84:87], 0 idxen
buffer_store_short v17, v62, s[84:87], 0 idxen
buffer_store_short v18, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 9
v_add_f16_e64 v19, v19, s99
v_mul_f16_e64 v117, v19, s32
v_cmp_lt_f16_e64 vcc, v19, 0
v_cndmask_b32_e32 v19, v19, v117, vcc
v_add_f16_e64 v20, v20, s99
v_mul_f16_e64 v117, v20, s32
v_cmp_lt_f16_e64 vcc, v20, 0
v_cndmask_b32_e32 v20, v20, v117, vcc
v_add_f16_e64 v21, v21, s99
v_mul_f16_e64 v117, v21, s32
v_cmp_lt_f16_e64 vcc, v21, 0
v_cndmask_b32_e32 v21, v21, v117, vcc
buffer_store_short v19, v37, s[84:87], 0 idxen
buffer_store_short v20, v62, s[84:87], 0 idxen
buffer_store_short v21, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 10
v_add_f16_e64 v22, v22, s99
v_mul_f16_e64 v117, v22, s32
v_cmp_lt_f16_e64 vcc, v22, 0
v_cndmask_b32_e32 v22, v22, v117, vcc
v_add_f16_e64 v23, v23, s99
v_mul_f16_e64 v117, v23, s32
v_cmp_lt_f16_e64 vcc, v23, 0
v_cndmask_b32_e32 v23, v23, v117, vcc
v_add_f16_e64 v24, v24, s99
v_mul_f16_e64 v117, v24, s32
v_cmp_lt_f16_e64 vcc, v24, 0
v_cndmask_b32_e32 v24, v24, v117, vcc
buffer_store_short v22, v37, s[84:87], 0 idxen
buffer_store_short v23, v62, s[84:87], 0 idxen
buffer_store_short v24, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 11
v_add_f16_e64 v25, v25, s99
v_mul_f16_e64 v117, v25, s32
v_cmp_lt_f16_e64 vcc, v25, 0
v_cndmask_b32_e32 v25, v25, v117, vcc
v_add_f16_e64 v26, v26, s99
v_mul_f16_e64 v117, v26, s32
v_cmp_lt_f16_e64 vcc, v26, 0
v_cndmask_b32_e32 v26, v26, v117, vcc
v_add_f16_e64 v27, v27, s99
v_mul_f16_e64 v117, v27, s32
v_cmp_lt_f16_e64 vcc, v27, 0
v_cndmask_b32_e32 v27, v27, v117, vcc
buffer_store_short v25, v37, s[84:87], 0 idxen
buffer_store_short v26, v62, s[84:87], 0 idxen
buffer_store_short v27, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 20
s_cselect_b32 s87, 0, s87
s_cselect_b32 s91, 0, s91
s_add_u32 s88, s88, 64
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, 32
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
v_readlane_b32 s99, v115, 0
v_add_f16_e64 v4, v4, s99
v_mul_f16_e32 v4, 0xbdc5, v4
v_exp_f16_e32 v4, v4
_v_add_f16_e32_1_gfx9x 4, 4
v_rcp_f16_e32 v4, v4
v_add_f16_e64 v5, v5, s99
v_mul_f16_e32 v5, 0xbdc5, v5
v_exp_f16_e32 v5, v5
_v_add_f16_e32_1_gfx9x 5, 5
v_rcp_f16_e32 v5, v5
v_add_f16_e64 v6, v6, s99
v_mul_f16_e32 v6, 0xbdc5, v6
v_exp_f16_e32 v6, v6
_v_add_f16_e32_1_gfx9x 6, 6
v_rcp_f16_e32 v6, v6
buffer_store_short v4, v37, s[84:87], 0 idxen
buffer_store_short v5, v62, s[84:87], 0 idxen
buffer_store_short v6, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 1
v_add_f16_e64 v7, v7, s99
v_mul_f16_e32 v7, 0xbdc5, v7
v_exp_f16_e32 v7, v7
_v_add_f16_e32_1_gfx9x 7, 7
v_rcp_f16_e32 v7, v7
v_add_f16_e64 v8, v8, s99
v_mul_f16_e32 v8, 0xbdc5, v8
v_exp_f16_e32 v8, v8
_v_add_f16_e32_1_gfx9x 8, 8
v_rcp_f16_e32 v8, v8
v_add_f16_e64 v9, v9, s99
v_mul_f16_e32 v9, 0xbdc5, v9
v_exp_f16_e32 v9, v9
_v_add_f16_e32_1_gfx9x 9, 9
v_rcp_f16_e32 v9, v9
buffer_store_short v7, v37, s[84:87], 0 idxen
buffer_store_short v8, v62, s[84:87], 0 idxen
buffer_store_short v9, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 2
v_add_f16_e64 v10, v10, s99
v_mul_f16_e32 v10, 0xbdc5, v10
v_exp_f16_e32 v10, v10
_v_add_f16_e32_1_gfx9x 10, 10
v_rcp_f16_e32 v10, v10
v_add_f16_e64 v11, v11, s99
v_mul_f16_e32 v11, 0xbdc5, v11
v_exp_f16_e32 v11, v11
_v_add_f16_e32_1_gfx9x 11, 11
v_rcp_f16_e32 v11, v11
v_add_f16_e64 v12, v12, s99
v_mul_f16_e32 v12, 0xbdc5, v12
v_exp_f16_e32 v12, v12
_v_add_f16_e32_1_gfx9x 12, 12
v_rcp_f16_e32 v12, v12
buffer_store_short v10, v37, s[84:87], 0 idxen
buffer_store_short v11, v62, s[84:87], 0 idxen
buffer_store_short v12, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 3
v_add_f16_e64 v13, v13, s99
v_mul_f16_e32 v13, 0xbdc5, v13
v_exp_f16_e32 v13, v13
_v_add_f16_e32_1_gfx9x 13, 13
v_rcp_f16_e32 v13, v13
v_add_f16_e64 v14, v14, s99
v_mul_f16_e32 v14, 0xbdc5, v14
v_exp_f16_e32 v14, v14
_v_add_f16_e32_1_gfx9x 14, 14
v_rcp_f16_e32 v14, v14
v_add_f16_e64 v15, v15, s99
v_mul_f16_e32 v15, 0xbdc5, v15
v_exp_f16_e32 v15, v15
_v_add_f16_e32_1_gfx9x 15, 15
v_rcp_f16_e32 v15, v15
buffer_store_short v13, v37, s[84:87], 0 idxen
buffer_store_short v14, v62, s[84:87], 0 idxen
buffer_store_short v15, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s99, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 4
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 8
v_add_f16_e64 v16, v16, s99
v_mul_f16_e32 v16, 0xbdc5, v16
v_exp_f16_e32 v16, v16
_v_add_f16_e32_1_gfx9x 16, 16
v_rcp_f16_e32 v16, v16
v_add_f16_e64 v17, v17, s99
v_mul_f16_e32 v17, 0xbdc5, v17
v_exp_f16_e32 v17, v17
_v_add_f16_e32_1_gfx9x 17, 17
v_rcp_f16_e32 v17, v17
v_add_f16_e64 v18, v18, s99
v_mul_f16_e32 v18, 0xbdc5, v18
v_exp_f16_e32 v18, v18
_v_add_f16_e32_1_gfx9x 18, 18
v_rcp_f16_e32 v18, v18
buffer_store_short v16, v37, s[84:87], 0 idxen
buffer_store_short v17, v62, s[84:87], 0 idxen
buffer_store_short v18, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 9
v_add_f16_e64 v19, v19, s99
v_mul_f16_e32 v19, 0xbdc5, v19
v_exp_f16_e32 v19, v19
_v_add_f16_e32_1_gfx9x 19, 19
v_rcp_f16_e32 v19, v19
v_add_f16_e64 v20, v20, s99
v_mul_f16_e32 v20, 0xbdc5, v20
v_exp_f16_e32 v20, v20
_v_add_f16_e32_1_gfx9x 20, 20
v_rcp_f16_e32 v20, v20
v_add_f16_e64 v21, v21, s99
v_mul_f16_e32 v21, 0xbdc5, v21
v_exp_f16_e32 v21, v21
_v_add_f16_e32_1_gfx9x 21, 21
v_rcp_f16_e32 v21, v21
buffer_store_short v19, v37, s[84:87], 0 idxen
buffer_store_short v20, v62, s[84:87], 0 idxen
buffer_store_short v21, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 10
v_add_f16_e64 v22, v22, s99
v_mul_f16_e32 v22, 0xbdc5, v22
v_exp_f16_e32 v22, v22
_v_add_f16_e32_1_gfx9x 22, 22
v_rcp_f16_e32 v22, v22
v_add_f16_e64 v23, v23, s99
v_mul_f16_e32 v23, 0xbdc5, v23
v_exp_f16_e32 v23, v23
_v_add_f16_e32_1_gfx9x 23, 23
v_rcp_f16_e32 v23, v23
v_add_f16_e64 v24, v24, s99
v_mul_f16_e32 v24, 0xbdc5, v24
v_exp_f16_e32 v24, v24
_v_add_f16_e32_1_gfx9x 24, 24
v_rcp_f16_e32 v24, v24
buffer_store_short v22, v37, s[84:87], 0 idxen
buffer_store_short v23, v62, s[84:87], 0 idxen
buffer_store_short v24, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 11
v_add_f16_e64 v25, v25, s99
v_mul_f16_e32 v25, 0xbdc5, v25
v_exp_f16_e32 v25, v25
_v_add_f16_e32_1_gfx9x 25, 25
v_rcp_f16_e32 v25, v25
v_add_f16_e64 v26, v26, s99
v_mul_f16_e32 v26, 0xbdc5, v26
v_exp_f16_e32 v26, v26
_v_add_f16_e32_1_gfx9x 26, 26
v_rcp_f16_e32 v26, v26
v_add_f16_e64 v27, v27, s99
v_mul_f16_e32 v27, 0xbdc5, v27
v_exp_f16_e32 v27, v27
_v_add_f16_e32_1_gfx9x 27, 27
v_rcp_f16_e32 v27, v27
buffer_store_short v25, v37, s[84:87], 0 idxen
buffer_store_short v26, v62, s[84:87], 0 idxen
buffer_store_short v27, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 20
s_cselect_b32 s87, 0, s87
s_cselect_b32 s91, 0, s91
s_add_u32 s88, s88, 64
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, 32
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
v_readlane_b32 s99, v115, 0
v_add_f16_e64 v4, v4, s99
v_mul_f16_e64 v4, v4, s33
v_mul_f16_e32 v117, 0x3dc5, v4
v_mul_f16_e32 v118, 0xbdc5, v4
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v4, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v4, v4
v_mul_f16_e32 v4, v4, v117
v_mul_f16_e64 v4, v4, s32
v_add_f16_e64 v5, v5, s99
v_mul_f16_e64 v5, v5, s33
v_mul_f16_e32 v117, 0x3dc5, v5
v_mul_f16_e32 v118, 0xbdc5, v5
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v5, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v5, v5
v_mul_f16_e32 v5, v5, v117
v_mul_f16_e64 v5, v5, s32
v_add_f16_e64 v6, v6, s99
v_mul_f16_e64 v6, v6, s33
v_mul_f16_e32 v117, 0x3dc5, v6
v_mul_f16_e32 v118, 0xbdc5, v6
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v6, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v6, v6
v_mul_f16_e32 v6, v6, v117
v_mul_f16_e64 v6, v6, s32
buffer_store_short v4, v37, s[84:87], 0 idxen
buffer_store_short v5, v62, s[84:87], 0 idxen
buffer_store_short v6, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 1
v_add_f16_e64 v7, v7, s99
v_mul_f16_e64 v7, v7, s33
v_mul_f16_e32 v117, 0x3dc5, v7
v_mul_f16_e32 v118, 0xbdc5, v7
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v7, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v7, v7
v_mul_f16_e32 v7, v7, v117
v_mul_f16_e64 v7, v7, s32
v_add_f16_e64 v8, v8, s99
v_mul_f16_e64 v8, v8, s33
v_mul_f16_e32 v117, 0x3dc5, v8
v_mul_f16_e32 v118, 0xbdc5, v8
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v8, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v8, v8
v_mul_f16_e32 v8, v8, v117
v_mul_f16_e64 v8, v8, s32
v_add_f16_e64 v9, v9, s99
v_mul_f16_e64 v9, v9, s33
v_mul_f16_e32 v117, 0x3dc5, v9
v_mul_f16_e32 v118, 0xbdc5, v9
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v9, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v9, v9
v_mul_f16_e32 v9, v9, v117
v_mul_f16_e64 v9, v9, s32
buffer_store_short v7, v37, s[84:87], 0 idxen
buffer_store_short v8, v62, s[84:87], 0 idxen
buffer_store_short v9, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 2
v_add_f16_e64 v10, v10, s99
v_mul_f16_e64 v10, v10, s33
v_mul_f16_e32 v117, 0x3dc5, v10
v_mul_f16_e32 v118, 0xbdc5, v10
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v10, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v10, v10
v_mul_f16_e32 v10, v10, v117
v_mul_f16_e64 v10, v10, s32
v_add_f16_e64 v11, v11, s99
v_mul_f16_e64 v11, v11, s33
v_mul_f16_e32 v117, 0x3dc5, v11
v_mul_f16_e32 v118, 0xbdc5, v11
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v11, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v11, v11
v_mul_f16_e32 v11, v11, v117
v_mul_f16_e64 v11, v11, s32
v_add_f16_e64 v12, v12, s99
v_mul_f16_e64 v12, v12, s33
v_mul_f16_e32 v117, 0x3dc5, v12
v_mul_f16_e32 v118, 0xbdc5, v12
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v12, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v12, v12
v_mul_f16_e32 v12, v12, v117
v_mul_f16_e64 v12, v12, s32
buffer_store_short v10, v37, s[84:87], 0 idxen
buffer_store_short v11, v62, s[84:87], 0 idxen
buffer_store_short v12, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 3
v_add_f16_e64 v13, v13, s99
v_mul_f16_e64 v13, v13, s33
v_mul_f16_e32 v117, 0x3dc5, v13
v_mul_f16_e32 v118, 0xbdc5, v13
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v13, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v13, v13
v_mul_f16_e32 v13, v13, v117
v_mul_f16_e64 v13, v13, s32
v_add_f16_e64 v14, v14, s99
v_mul_f16_e64 v14, v14, s33
v_mul_f16_e32 v117, 0x3dc5, v14
v_mul_f16_e32 v118, 0xbdc5, v14
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v14, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v14, v14
v_mul_f16_e32 v14, v14, v117
v_mul_f16_e64 v14, v14, s32
v_add_f16_e64 v15, v15, s99
v_mul_f16_e64 v15, v15, s33
v_mul_f16_e32 v117, 0x3dc5, v15
v_mul_f16_e32 v118, 0xbdc5, v15
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v15, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v15, v15
v_mul_f16_e32 v15, v15, v117
v_mul_f16_e64 v15, v15, s32
buffer_store_short v13, v37, s[84:87], 0 idxen
buffer_store_short v14, v62, s[84:87], 0 idxen
buffer_store_short v15, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s99, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 4
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 8
v_add_f16_e64 v16, v16, s99
v_mul_f16_e64 v16, v16, s33
v_mul_f16_e32 v117, 0x3dc5, v16
v_mul_f16_e32 v118, 0xbdc5, v16
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v16, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v16, v16
v_mul_f16_e32 v16, v16, v117
v_mul_f16_e64 v16, v16, s32
v_add_f16_e64 v17, v17, s99
v_mul_f16_e64 v17, v17, s33
v_mul_f16_e32 v117, 0x3dc5, v17
v_mul_f16_e32 v118, 0xbdc5, v17
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v17, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v17, v17
v_mul_f16_e32 v17, v17, v117
v_mul_f16_e64 v17, v17, s32
v_add_f16_e64 v18, v18, s99
v_mul_f16_e64 v18, v18, s33
v_mul_f16_e32 v117, 0x3dc5, v18
v_mul_f16_e32 v118, 0xbdc5, v18
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v18, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v18, v18
v_mul_f16_e32 v18, v18, v117
v_mul_f16_e64 v18, v18, s32
buffer_store_short v16, v37, s[84:87], 0 idxen
buffer_store_short v17, v62, s[84:87], 0 idxen
buffer_store_short v18, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 9
v_add_f16_e64 v19, v19, s99
v_mul_f16_e64 v19, v19, s33
v_mul_f16_e32 v117, 0x3dc5, v19
v_mul_f16_e32 v118, 0xbdc5, v19
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v19, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v19, v19
v_mul_f16_e32 v19, v19, v117
v_mul_f16_e64 v19, v19, s32
v_add_f16_e64 v20, v20, s99
v_mul_f16_e64 v20, v20, s33
v_mul_f16_e32 v117, 0x3dc5, v20
v_mul_f16_e32 v118, 0xbdc5, v20
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v20, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v20, v20
v_mul_f16_e32 v20, v20, v117
v_mul_f16_e64 v20, v20, s32
v_add_f16_e64 v21, v21, s99
v_mul_f16_e64 v21, v21, s33
v_mul_f16_e32 v117, 0x3dc5, v21
v_mul_f16_e32 v118, 0xbdc5, v21
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v21, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v21, v21
v_mul_f16_e32 v21, v21, v117
v_mul_f16_e64 v21, v21, s32
buffer_store_short v19, v37, s[84:87], 0 idxen
buffer_store_short v20, v62, s[84:87], 0 idxen
buffer_store_short v21, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 10
v_add_f16_e64 v22, v22, s99
v_mul_f16_e64 v22, v22, s33
v_mul_f16_e32 v117, 0x3dc5, v22
v_mul_f16_e32 v118, 0xbdc5, v22
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v22, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v22, v22
v_mul_f16_e32 v22, v22, v117
v_mul_f16_e64 v22, v22, s32
v_add_f16_e64 v23, v23, s99
v_mul_f16_e64 v23, v23, s33
v_mul_f16_e32 v117, 0x3dc5, v23
v_mul_f16_e32 v118, 0xbdc5, v23
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v23, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v23, v23
v_mul_f16_e32 v23, v23, v117
v_mul_f16_e64 v23, v23, s32
v_add_f16_e64 v24, v24, s99
v_mul_f16_e64 v24, v24, s33
v_mul_f16_e32 v117, 0x3dc5, v24
v_mul_f16_e32 v118, 0xbdc5, v24
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v24, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v24, v24
v_mul_f16_e32 v24, v24, v117
v_mul_f16_e64 v24, v24, s32
buffer_store_short v22, v37, s[84:87], 0 idxen
buffer_store_short v23, v62, s[84:87], 0 idxen
buffer_store_short v24, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v115, 11
v_add_f16_e64 v25, v25, s99
v_mul_f16_e64 v25, v25, s33
v_mul_f16_e32 v117, 0x3dc5, v25
v_mul_f16_e32 v118, 0xbdc5, v25
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v25, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v25, v25
v_mul_f16_e32 v25, v25, v117
v_mul_f16_e64 v25, v25, s32
v_add_f16_e64 v26, v26, s99
v_mul_f16_e64 v26, v26, s33
v_mul_f16_e32 v117, 0x3dc5, v26
v_mul_f16_e32 v118, 0xbdc5, v26
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v26, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v26, v26
v_mul_f16_e32 v26, v26, v117
v_mul_f16_e64 v26, v26, s32
v_add_f16_e64 v27, v27, s99
v_mul_f16_e64 v27, v27, s33
v_mul_f16_e32 v117, 0x3dc5, v27
v_mul_f16_e32 v118, 0xbdc5, v27
v_exp_f16_e32 v117, v117
v_exp_f16_e32 v118, v118
v_add_f16_e32 v27, v117, v118
v_sub_f16_e32 v117, v117, v118
v_rcp_f16_e32 v27, v27
v_mul_f16_e32 v27, v27, v117
v_mul_f16_e64 v27, v27, s32
buffer_store_short v25, v37, s[84:87], 0 idxen
buffer_store_short v26, v62, s[84:87], 0 idxen
buffer_store_short v27, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 1
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 20
s_cselect_b32 s87, 0, s87
s_cselect_b32 s91, 0, s91
s_add_u32 s88, s88, 64
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, 32
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
s_endpgm
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
