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

v_mov_b32_e32 v0, v0
s_mov_b32 s0, 0
v_mov_b32_e32 v111, 0
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
s_add_u32 s82, s82, 0x3448
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
s_add_u32 s80, s80, 0x3868
s_addc_u32 s81, s81, 0
s_branch 18
s_load_dword s32, s[84:85], 0x60
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x3ad4
s_addc_u32 s81, s81, 0
s_branch 11
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x3f28
s_addc_u32 s81, s81, 0
s_branch 6
s_load_dwordx2 s[32:33], s[84:85], 0x60
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x4374
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
v_ffbh_u32_e32 v114, s13
v_lshlrev_b32_e64 v115, v114, s13
v_and_b32_e32 v116, 0xffffff00, v115
v_cmp_eq_u32_e32 vcc, 0x80000000, v115
v_cvt_f32_u32_e32 v116, v116
v_rcp_f32_e32 v112, v116
v_subb_co_u32_e32 v113, vcc, 32, v114, vcc
v_cvt_f32_ubyte0_e32 v114, v115
v_fma_f32 v116, v116, v112, -1.0
v_fma_f32 v116, v114, v112, v116
v_fmaak_f32 v116, v116, v112, 0x9f000000
v_mul_f32_e32 v116, 0x5f800000, v116
v_mov_b32_e32 v114, 0
v_cvt_flr_i32_f32_e64 v116, -v116
v_lshl_add_u32 v112, v112, 9, v116
v_mad_u64_u32 v[114:115], vcc, v115, v112, v[114:115]
v_subb_co_u32_e64 v112, vcc, v112, -1, vcc
v_mul_hi_u32 v114, s4, v112
v_add_co_u32_e64 v112, vcc, v114, s4
v_addc_co_u32_e64 v114, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v113
v_cndmask_b32_e32 v112, v112, v114, vcc
v_alignbit_b32 v112, v114, v112, v113
s_nop 0
v_readfirstlane_b32 s88, v112
s_mul_i32 s89, s88, s13
s_sub_u32 s4, s4, s89
s_waitcnt lgkmcnt(0)
s_lshl_b32 s93, s93, 2
s_lshl_b64 s[94:95], s[94:95], 2
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
s_lshl_b32 s47, s48, 2
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
s_and_b32 s88, 1, s26
s_addc_u32 s88, s28, 1
s_ashr_i32 s88, s88, 1
s_add_u32 s86, s88, 2
v_mov_b32_e32 v113, 0x55555556
v_mul_hi_u32 v113, v113, s86
s_nop 0
v_readfirstlane_b32 s86, v113
s_andn2_b32 s88, 1, s27
s_addc_u32 s88, s29, 1
s_ashr_i32 s88, s88, 1
s_add_u32 s87, s88, 2
v_mov_b32_e32 v113, 0x55555556
v_mul_hi_u32 v113, v113, s87
s_nop 0
v_readfirstlane_b32 s87, v113
s_sub_u32 s57, 0, s87
s_sub_u32 s56, 0, s86
s_add_u32 s1, s24, 1
v_mov_b32_e32 v113, 0x80000000
v_mul_hi_u32 v113, v113, s1
s_nop 0
v_readfirstlane_b32 s1, v113
s_add_u32 s5, s25, 1
v_mov_b32_e32 v113, 0x80000000
v_mul_hi_u32 v113, v113, s5
s_nop 0
v_readfirstlane_b32 s5, v113
v_mad_i32_i24 v112, 2, s1, -1
v_sub_co_u32_e64 v112, vcc, v112, s24
v_addc_co_u32_e64 v112, vcc, 0, 0, vcc
s_nop 0
v_readfirstlane_b32 s88, v112
s_and_b32 s88, s88, 1
s_and_b32 s88, s88, s1
s_add_u32 s1, s1, s88
v_readfirstlane_b32 s89, v0
s_and_b32 s90, s89, 64
s_cselect_b32 s90, 0x80000, 0
s_or_b32 s14, s14, s90
s_lshl_b32 s51, s47, 1
s_mov_b64 s[52:53], 0
s_bitset1_b32 s14, 23
s_mov_b32 s51, s47
s_mov_b32 s52, s47
s_mov_b32 s53, 0
s_add_u32 s5, s5, 1
s_and_b32 s5, s5, -2
s_branch 16
s_and_b32 s90, s9, 1
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
v_bfe_u32 v113, v0, 2, 6
v_lshrrev_b32_e32 v106, 1, v113
s_bitcmp0_b32 s89, 8
s_cselect_b32 s90, 0x1000000, 0
s_or_b32 s90, s90, 0x100000
s_and_b32 s90, s14, s90
s_cselect_b32 s90, 0, 15
v_bfi_b32 v106, s90, v113, v106
v_bfe_u32 v113, s89, 8, 1
v_xor_b32_e64 v113, v113, 1
v_lshrrev_b32_e32 v106, v113, v106
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
v_mov_b32_e32 v113, s4
v_mov_b32_e32 v114, s13
v_and_b32_e32 v115, 3, v0
v_cmp_eq_u32_e32 vcc, 2, v115
v_cndmask_b32_e32 v113, v113, v114, vcc
v_cmp_eq_u32_e32 vcc, 1, v115
v_cndmask_b32_e32 v116, 0, v106, vcc
s_bitcmp1_b32 s14, 20
s_cbranch_scc0 4
v_add_co_u32_e64 v114, vcc, v106, 8
v_cmp_eq_u32_e32 vcc, 0, v115
v_cndmask_b32_e32 v116, v116, v114, vcc
v_cmp_eq_u32_e64 s[90:91], 3, v115
v_bfe_u32 v104, v116, 0, 5
v_mad_u32_u24 v104, v113, 32, v104
v_ffbh_u32_e32 v118, s87
v_lshlrev_b32_e64 v119, v118, s87
v_and_b32_e32 v120, 0xffffff00, v119
v_cmp_eq_u32_e32 vcc, 0x80000000, v119
v_cvt_f32_u32_e32 v120, v120
v_rcp_f32_e32 v105, v120
v_subb_co_u32_e32 v117, vcc, 32, v118, vcc
v_cvt_f32_ubyte0_e32 v118, v119
v_fma_f32 v120, v120, v105, -1.0
v_fma_f32 v120, v118, v105, v120
v_fmaak_f32 v120, v120, v105, 0x9f000000
v_mul_f32_e32 v120, 0x5f800000, v120
v_mov_b32_e32 v118, 0
v_cvt_flr_i32_f32_e64 v120, -v120
v_lshl_add_u32 v105, v105, 9, v120
v_mad_u64_u32 v[118:119], vcc, v119, v105, v[118:119]
v_subb_co_u32_e64 v105, vcc, v105, -1, vcc
v_mul_hi_u32 v118, v104, v105
v_add_co_u32_e32 v105, vcc, v118, v104
v_addc_co_u32_e64 v118, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v117
v_cndmask_b32_e32 v105, v105, v118, vcc
v_alignbit_b32 v105, v118, v105, v117
v_mad_i32_i24 v103, v105, s57, v104
v_lshrrev_b32_e32 v104, 5, v116
v_mad_u32_u24 v104, v105, 1, v104
v_cndmask_b32_e64 v104, v104, 1, s[90:91]
v_ffbh_u32_e32 v118, s86
v_lshlrev_b32_e64 v119, v118, s86
v_and_b32_e32 v120, 0xffffff00, v119
v_cmp_eq_u32_e32 vcc, 0x80000000, v119
v_cvt_f32_u32_e32 v120, v120
v_rcp_f32_e32 v105, v120
v_subb_co_u32_e32 v117, vcc, 32, v118, vcc
v_cvt_f32_ubyte0_e32 v118, v119
v_fma_f32 v120, v120, v105, -1.0
v_fma_f32 v120, v118, v105, v120
v_fmaak_f32 v120, v120, v105, 0x9f000000
v_mul_f32_e32 v120, 0x5f800000, v120
v_mov_b32_e32 v118, 0
v_cvt_flr_i32_f32_e64 v120, -v120
v_lshl_add_u32 v105, v105, 9, v120
v_mad_u64_u32 v[118:119], vcc, v119, v105, v[118:119]
v_subb_co_u32_e64 v105, vcc, v105, -1, vcc
v_mul_hi_u32 v118, v104, v105
v_add_co_u32_e32 v105, vcc, v118, v104
v_addc_co_u32_e64 v118, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v117
v_cndmask_b32_e32 v105, v105, v118, vcc
v_alignbit_b32 v105, v118, v105, v117
v_mad_i32_i24 v104, v105, s56, v104
v_readlane_b32 s58, v103, 2
v_readlane_b32 s59, v104, 2
v_readlane_b32 s60, v105, 2
v_readlane_b32 s61, v104, 3
v_readlane_b32 s62, v105, 3
v_add_co_u32_e64 v103, vcc, v103, s57
v_add_co_u32_e64 v104, vcc, v104, s56
v_mov_b32_dpp v105, v105 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v103, v103 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v104, v104 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x20000
s_mov_b32 s86, 0x80000000
s_mov_b32 s87, 0x20000
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 5
v_xor_b32_dpp v107, v0, v0 quad_perm:[2,3,2,1] row_mask:0xf bank_mask:0xf
v_subrev_co_u32_e32 v107, vcc, 1, v107
v_cvt_f32_i32_e32 v107, v107
s_branch 4
v_xor_b32_dpp v107, v0, v0 quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32_e32 v107, vcc, 1, v107
v_cvt_f32_i32_e32 v107, v107
v_mov_b32_e32 v108, 1
v_xor_b32_dpp v108, v0, v0 quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v108, v0, v0 quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32_e32 v108, vcc, 1, v108
v_mov_b32_e32 v109, 1
v_xor_b32_dpp v109, v0, v0 quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v109, v0, v0 quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32_e32 v109, vcc, 1, v109
v_cvt_f32_i32_e32 v108, v108
v_cvt_f32_i32_e32 v109, v109
v_lshrrev_b32_e64 v112, 2, s74
v_and_b32_e32 v113, 3, v0
v_bfe_u32 v114, v0, 4, 3
v_mad_u32_u24 v102, v114, 4, v113
v_lshlrev_b32_e32 v102, 4, v102
v_mad_u32_u24 v97, v112, 4, v113
v_lshlrev_b32_e32 v97, 4, v97
v_bfe_u32 v112, v0, 2, 2
v_and_b32_e32 v113, 1, v112
v_mad_u32_u24 v115, v112, 16, v113
v_lshlrev_b32_e32 v115, 6, v115
v_xor_b32_e32 v97, v97, v115
v_mul_u32_u24_e32 v115, 0x400, v112
v_xor_b32_e32 v102, v102, v115
s_lshr_b32 s74, s74, 1
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 47
s_and_b32 s3, s14, 0x1100000
s_addc_u32 s3, 0, 0
v_lshrrev_b32_e32 v115, 1, v0
s_mul_i32 s2, 60, s3
s_sub_u32 s2, 63, s2
v_bfi_b32 v115, s2, v0, v115
v_and_b32_e32 v112, 1, v115
v_bfe_u32 v113, v115, 1, 1
v_xor_b32_e32 v112, v112, v113
v_bfe_u32 v114, v115, 3, 1
v_mad_u32_u24 v113, v113, 2, v114
v_mul_u32_u24_e32 v112, 0x118, v112
v_bfe_u32 v114, v115, 2, 1
v_mad_u32_u24 v113, v113, 2, v112
v_xor_b32_e32 v113, v113, v114
v_and_b32_e32 v114, 0xf0, v115
v_xor_b32_e32 v113, v113, v114
s_mul_i32 s2, 4, s3
s_sub_u32 s2, 6, s2
v_bfe_u32 v115, v0, s2, 1
v_mul_u32_u24_e32 v115, 0x1040, v115
v_xor_b32_e32 v99, 0x314, v113
v_xor_b32_e32 v100, 0x31c, v113
v_xor_b32_e32 v101, 8, v113
v_mov_b32_e32 v98, v113
v_mad_u32_u24 v98, 4, v98, v115
v_mad_u32_u24 v99, 4, v99, v115
v_mad_u32_u24 v100, 4, v100, v115
v_mad_u32_u24 v101, 4, v101, v115
s_branch 44
s_bfe_u32 s3, s14, 0x10014
v_lshrrev_b32_e32 v115, 1, v0
s_mul_i32 s2, 60, s3
s_sub_u32 s2, 63, s2
v_bfi_b32 v115, s2, v0, v115
v_and_b32_e32 v112, 1, v115
v_bfe_u32 v113, v115, 1, 1
v_bfe_u32 v114, v115, 3, 1
v_xor_b32_e32 v112, v112, v113
v_mad_u32_u24 v113, v113, 2, v114
v_mul_u32_u24_e32 v112, 0x109, v112
v_bfe_u32 v114, v115, 2, 1
v_mad_u32_u24 v113, v113, 2, v112
v_xor_b32_e32 v113, v113, v114
v_and_b32_e32 v114, 0xf0, v115
v_or_b32_e32 v113, v113, v114
s_mul_i32 s2, 4, s3
s_sub_u32 s2, 6, s2
v_bfe_u32 v115, v0, s2, 1
v_mul_u32_u24_e32 v115, 0x1040, v115
v_mad_u32_u24 v98, 4, v113, v115
v_xor_b32_e32 v99, 0x307, v113
v_mad_u32_u24 v99, 4, v99, v115
v_xor_b32_e32 v100, 0x30f, v113
v_mad_u32_u24 v100, 4, v100, v115
v_xor_b32_e32 v101, 8, v113
v_mad_u32_u24 v101, 4, v101, v115
v_subrev_co_u32_e32 v103, vcc, s58, v103
v_mov_b32_e32 v113, s57
v_cmp_lt_i32_e32 vcc, v103, v113
v_subb_co_u32_e64 v112, vcc, 0, 0, vcc
v_mad_i32_i24 v103, v112, s57, v103
v_mad_i32_i24 v105, v112, s62, v105
v_mad_i32_i24 v104, v112, s61, v104
v_mov_b32_e32 v113, s56
v_cmp_lt_i32_e32 vcc, v104, v113
v_subb_co_u32_e64 v112, vcc, 0, 0, vcc
v_add_co_u32_e32 v105, vcc, v105, v112
v_mad_i32_i24 v104, v112, v113, v104
v_subrev_co_u32_e32 v104, vcc, s59, v104
v_cmp_lt_i32_e32 vcc, v104, v113
v_subb_co_u32_e64 v112, vcc, 0, 0, vcc
v_add_co_u32_e32 v105, vcc, v105, v112
v_mad_i32_i24 v104, v112, s56, v104
v_subrev_co_u32_e32 v105, vcc, s60, v105
s_mov_b32 s35, 0
s_mov_b32 s38, s24
s_mov_b32 s39, 1
s_mov_b32 s66, 0
s_mov_b32 s67, s12
s_mov_b32 s65, s67
s_sub_u32 s75, -1, s74
s_sub_u32 s75, s75, 16
s_bitset1_b32 s14, 21
s_mov_b32 s87, 0
s_mov_b32 s91, 0
v_add_co_u32_e32 v112, vcc, 2, v0
v_bfe_u32 v112, v112, 2, 1
v_cmp_ne_u32_e64 vcc, v112, 1
s_mov_b64 s[6:7], vcc
s_mov_b32 s76, 17
s_mov_b32 s64, 0
s_bitset1_b32 s14, 26
s_call_b64 s[36:37], 1569
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 1
s_branch 791
s_mov_b64 vcc, s[6:7]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v72, v74, v72 div:2
v_subrev_f32_e64 v75, v73, v75 div:2
v_add_f32_e64 v73, v74, v73 div:2
v_fma_f32 v74, v74, 1.0, -v73
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v64, v1, s[40:43], 0 idxen
buffer_load_dword v66, v3, s[40:43], 0 idxen
buffer_load_dword v65, v2, s[40:43], 0 idxen
buffer_load_dword v67, v36, s[40:43], 0 idxen
ds_write_b32 v98, v68
ds_write_b32 v99, v69
ds_read_b128 v[38:41], v102 offset:29440
ds_read_b128 v[46:49], v97 offset:28928
ds_read_b128 v[50:53], v97 offset:29056
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 1494
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_dpp v72, v72, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v73, v73, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v74, v74, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v75, v75, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_setprio 1
ds_write_b32 v100, v74 offset:8256
ds_write_b32 v101, v75 offset:8256
ds_read_b128 v[42:45], v102 offset:33536
ds_read_b128 v[54:57], v97 offset:33024
ds_read_b128 v[58:61], v97 offset:33152
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 1
s_call_b64 s[36:37], 1432
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v76, v78, v76 div:2
v_subrev_f32_e64 v79, v77, v79 div:2
v_add_f32_e64 v77, v78, v77 div:2
v_fma_f32 v78, v78, 1.0, -v77
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v68, v1, s[40:43], 0 idxen
buffer_load_dword v70, v3, s[40:43], 0 idxen
buffer_load_dword v69, v2, s[40:43], 0 idxen
buffer_load_dword v71, v36, s[40:43], 0 idxen
ds_write_b32 v98, v72 offset:8256
ds_write_b32 v99, v73 offset:8256
ds_read_b128 v[38:41], v102 offset:37696
ds_read_b128 v[46:49], v97 offset:37184
ds_read_b128 v[50:53], v97 offset:37312
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 1366
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_dpp v76, v76, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v77, v77, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v78, v78, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v79, v79, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_barrier
s_setprio 1
ds_write_b32 v100, v78 offset:16512
ds_write_b32 v101, v79 offset:16512
ds_read_b128 v[42:45], v102 offset:41792
ds_read_b128 v[54:57], v97 offset:41280
ds_read_b128 v[58:61], v97 offset:41408
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 8
s_call_b64 s[36:37], 1303
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v80, v82, v80 div:2
v_subrev_f32_e64 v83, v81, v83 div:2
v_add_f32_e64 v81, v82, v81 div:2
v_fma_f32 v82, v82, 1.0, -v81
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v72, v1, s[40:43], 0 idxen
buffer_load_dword v74, v3, s[40:43], 0 idxen
buffer_load_dword v73, v2, s[40:43], 0 idxen
buffer_load_dword v75, v36, s[40:43], 0 idxen
ds_write_b32 v98, v76 offset:16512
ds_write_b32 v99, v77 offset:16512
ds_read_b128 v[38:41], v102 offset:45952
ds_read_b128 v[46:49], v97 offset:45440
ds_read_b128 v[50:53], v97 offset:45568
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 1230
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_dpp v80, v80, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v81, v81, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v82, v82, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v83, v83, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_setprio 1
ds_write_b32 v100, v82 offset:24768
ds_write_b32 v101, v83 offset:24768
ds_read_b128 v[42:45], v102 offset:512
ds_read_b128 v[54:57], v97
ds_read_b128 v[58:61], v97 offset:128
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 1
s_call_b64 s[36:37], 1168
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v84, v86, v84 div:2
v_subrev_f32_e64 v87, v85, v87 div:2
v_add_f32_e64 v85, v86, v85 div:2
v_fma_f32 v86, v86, 1.0, -v85
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v76, v1, s[40:43], 0 idxen
buffer_load_dword v78, v3, s[40:43], 0 idxen
buffer_load_dword v77, v2, s[40:43], 0 idxen
buffer_load_dword v79, v36, s[40:43], 0 idxen
ds_write_b32 v98, v80 offset:24768
ds_write_b32 v99, v81 offset:24768
ds_read_b128 v[38:41], v102 offset:4672
ds_read_b128 v[46:49], v97 offset:4160
ds_read_b128 v[50:53], v97 offset:4288
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 1102
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_dpp v84, v84, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v85, v85, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v86, v86, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v87, v87, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_barrier
s_setprio 1
ds_write_b32 v100, v86 offset:33024
ds_write_b32 v101, v87 offset:33024
ds_read_b128 v[42:45], v102 offset:8768
ds_read_b128 v[54:57], v97 offset:8256
ds_read_b128 v[58:61], v97 offset:8384
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 8
s_call_b64 s[36:37], 1039
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v64, v66, v64 div:2
v_subrev_f32_e64 v67, v65, v67 div:2
v_add_f32_e64 v65, v66, v65 div:2
v_fma_f32 v66, v66, 1.0, -v65
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v80, v1, s[40:43], 0 idxen
buffer_load_dword v82, v3, s[40:43], 0 idxen
buffer_load_dword v81, v2, s[40:43], 0 idxen
buffer_load_dword v83, v36, s[40:43], 0 idxen
ds_write_b32 v98, v84 offset:33024
ds_write_b32 v99, v85 offset:33024
ds_read_b128 v[38:41], v102 offset:12928
ds_read_b128 v[46:49], v97 offset:12416
ds_read_b128 v[50:53], v97 offset:12544
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 966
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_dpp v64, v64, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v65, v65, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v66, v66, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v67, v67, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_setprio 1
ds_write_b32 v100, v66 offset:41280
ds_write_b32 v101, v67 offset:41280
ds_read_b128 v[42:45], v102 offset:17024
ds_read_b128 v[54:57], v97 offset:16512
ds_read_b128 v[58:61], v97 offset:16640
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 1
s_call_b64 s[36:37], 904
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v68, v70, v68 div:2
v_subrev_f32_e64 v71, v69, v71 div:2
v_add_f32_e64 v69, v70, v69 div:2
v_fma_f32 v70, v70, 1.0, -v69
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v84, v1, s[40:43], 0 idxen
buffer_load_dword v86, v3, s[40:43], 0 idxen
buffer_load_dword v85, v2, s[40:43], 0 idxen
buffer_load_dword v87, v36, s[40:43], 0 idxen
ds_write_b32 v98, v64 offset:41280
ds_write_b32 v99, v65 offset:41280
ds_read_b128 v[38:41], v102 offset:21184
ds_read_b128 v[46:49], v97 offset:20672
ds_read_b128 v[50:53], v97 offset:20800
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 838
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_dpp v68, v68, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v69, v69, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v70, v70, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v71, v71, v107 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_barrier
s_setprio 1
ds_write_b32 v100, v70
ds_write_b32 v101, v71
ds_read_b128 v[42:45], v102 offset:25280
ds_read_b128 v[54:57], v97 offset:24768
ds_read_b128 v[58:61], v97 offset:24896
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 64752
s_call_b64 s[36:37], 775
s_branch 64750
s_mov_b64 vcc, s[6:7]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
s_setprio 1
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_cndmask_b32_dpp v72, v72, v72, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v73, v73, v73, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v74, v74, v74, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v75, v75, v75, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v64, v1, s[40:43], 0 idxen
buffer_load_dword v67, v36, s[40:43], 0 idxen
ds_write_b32 v98, v68
ds_write_b32 v99, v69
ds_read_b128 v[38:41], v102 offset:29440
ds_read_b128 v[46:49], v97 offset:28928
ds_read_b128 v[50:53], v97 offset:29056
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 2
s_call_b64 s[36:37], 705
s_nop 0
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
s_setprio 1
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_dpp v72, v72, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v75, v75, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v73, v72, v75 div:2
v_add_f32_e64 v74, v72, -v75 div:2
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v100, v74 offset:8256
ds_write_b32 v101, v75 offset:8256
ds_read_b128 v[42:45], v102 offset:33536
ds_read_b128 v[54:57], v97 offset:33024
ds_read_b128 v[58:61], v97 offset:33152
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 8
s_call_b64 s[36:37], 647
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
s_setprio 1
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_cndmask_b32_dpp v76, v76, v76, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v77, v77, v77, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v78, v78, v78, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v79, v79, v79, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v68, v1, s[40:43], 0 idxen
buffer_load_dword v71, v36, s[40:43], 0 idxen
ds_write_b32 v98, v72 offset:8256
ds_write_b32 v99, v73 offset:8256
ds_read_b128 v[38:41], v102 offset:37696
ds_read_b128 v[46:49], v97 offset:37184
ds_read_b128 v[50:53], v97 offset:37312
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 2
s_call_b64 s[36:37], 577
s_nop 0
s_barrier
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
s_setprio 1
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_dpp v76, v76, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v79, v79, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v77, v76, v79 div:2
v_add_f32_e64 v78, v76, -v79 div:2
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v100, v78 offset:16512
ds_write_b32 v101, v79 offset:16512
ds_read_b128 v[42:45], v102 offset:41792
ds_read_b128 v[54:57], v97 offset:41280
ds_read_b128 v[58:61], v97 offset:41408
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 518
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
s_setprio 1
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_cndmask_b32_dpp v80, v80, v80, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v81, v81, v81, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v82, v82, v82, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v83, v83, v83, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v72, v1, s[40:43], 0 idxen
buffer_load_dword v75, v36, s[40:43], 0 idxen
ds_write_b32 v98, v76 offset:16512
ds_write_b32 v99, v77 offset:16512
ds_read_b128 v[38:41], v102 offset:45952
ds_read_b128 v[46:49], v97 offset:45440
ds_read_b128 v[50:53], v97 offset:45568
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 2
s_call_b64 s[36:37], 449
s_nop 0
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
s_setprio 1
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_dpp v80, v80, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v83, v83, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v81, v80, v83 div:2
v_add_f32_e64 v82, v80, -v83 div:2
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v100, v82 offset:24768
ds_write_b32 v101, v83 offset:24768
ds_read_b128 v[42:45], v102 offset:512
ds_read_b128 v[54:57], v97
ds_read_b128 v[58:61], v97 offset:128
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 8
s_call_b64 s[36:37], 391
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
s_setprio 1
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_cndmask_b32_dpp v84, v84, v84, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v85, v85, v85, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v86, v86, v86, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v87, v87, v87, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v76, v1, s[40:43], 0 idxen
buffer_load_dword v79, v36, s[40:43], 0 idxen
ds_write_b32 v98, v80 offset:24768
ds_write_b32 v99, v81 offset:24768
ds_read_b128 v[38:41], v102 offset:4672
ds_read_b128 v[46:49], v97 offset:4160
ds_read_b128 v[50:53], v97 offset:4288
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 2
s_call_b64 s[36:37], 321
s_nop 0
s_barrier
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
s_setprio 1
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_dpp v84, v84, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v87, v87, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v85, v84, v87 div:2
v_add_f32_e64 v86, v84, -v87 div:2
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v100, v86 offset:33024
ds_write_b32 v101, v87 offset:33024
ds_read_b128 v[42:45], v102 offset:8768
ds_read_b128 v[54:57], v97 offset:8256
ds_read_b128 v[58:61], v97 offset:8384
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 262
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
s_setprio 1
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_cndmask_b32_dpp v64, v64, v64, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v65, v65, v65, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v66, v66, v66, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v67, v67, v67, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v80, v1, s[40:43], 0 idxen
buffer_load_dword v83, v36, s[40:43], 0 idxen
ds_write_b32 v98, v84 offset:33024
ds_write_b32 v99, v85 offset:33024
ds_read_b128 v[38:41], v102 offset:12928
ds_read_b128 v[46:49], v97 offset:12416
ds_read_b128 v[50:53], v97 offset:12544
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 2
s_call_b64 s[36:37], 193
s_nop 0
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
s_setprio 1
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_dpp v64, v64, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v67, v67, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v65, v64, v67 div:2
v_add_f32_e64 v66, v64, -v67 div:2
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v100, v66 offset:41280
ds_write_b32 v101, v67 offset:41280
ds_read_b128 v[42:45], v102 offset:17024
ds_read_b128 v[54:57], v97 offset:16512
ds_read_b128 v[58:61], v97 offset:16640
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 8
s_call_b64 s[36:37], 135
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_e32 v4, v38, v46
v_fmac_f32_e32 v5, v39, v46
v_fmac_f32_e32 v6, v40, v46
s_setprio 1
v_fmac_f32_e32 v7, v41, v46
v_fmac_f32_e32 v8, v38, v47
v_fmac_f32_e32 v9, v39, v47
v_fmac_f32_e32 v10, v40, v47
v_fmac_f32_e32 v11, v41, v47
v_fmac_f32_e32 v12, v38, v48
v_fmac_f32_e32 v13, v39, v48
v_fmac_f32_e32 v14, v40, v48
v_fmac_f32_e32 v15, v41, v48
v_fmac_f32_e32 v16, v38, v49
v_fmac_f32_e32 v17, v39, v49
v_fmac_f32_e32 v18, v40, v49
v_fmac_f32_e32 v19, v41, v49
v_fmac_f32_e32 v20, v38, v50
v_fmac_f32_e32 v21, v39, v50
v_fmac_f32_e32 v22, v40, v50
v_fmac_f32_e32 v23, v41, v50
v_fmac_f32_e32 v24, v38, v51
v_fmac_f32_e32 v25, v39, v51
v_fmac_f32_e32 v26, v40, v51
v_fmac_f32_e32 v27, v41, v51
v_fmac_f32_e32 v28, v38, v52
v_fmac_f32_e32 v29, v39, v52
v_fmac_f32_e32 v30, v40, v52
v_fmac_f32_e32 v31, v41, v52
v_fmac_f32_e32 v32, v38, v53
v_fmac_f32_e32 v33, v39, v53
v_cndmask_b32_dpp v68, v68, v68, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v69, v69, v69, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v70, v70, v70, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v71, v71, v71, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v40, v53
v_fmac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v84, v1, s[40:43], 0 idxen
buffer_load_dword v87, v36, s[40:43], 0 idxen
ds_write_b32 v98, v64 offset:41280
ds_write_b32 v99, v65 offset:41280
ds_read_b128 v[38:41], v102 offset:21184
ds_read_b128 v[46:49], v97 offset:20672
ds_read_b128 v[50:53], v97 offset:20800
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 2
s_call_b64 s[36:37], 65
s_nop 0
s_barrier
v_fmac_f32_e32 v4, v42, v54
v_fmac_f32_e32 v5, v43, v54
v_fmac_f32_e32 v6, v44, v54
s_setprio 1
v_fmac_f32_e32 v7, v45, v54
v_fmac_f32_e32 v8, v42, v55
v_fmac_f32_e32 v9, v43, v55
v_fmac_f32_e32 v10, v44, v55
v_fmac_f32_e32 v11, v45, v55
v_fmac_f32_e32 v12, v42, v56
v_fmac_f32_e32 v13, v43, v56
v_fmac_f32_e32 v14, v44, v56
v_fmac_f32_e32 v15, v45, v56
v_fmac_f32_e32 v16, v42, v57
v_fmac_f32_e32 v17, v43, v57
v_fmac_f32_e32 v18, v44, v57
v_fmac_f32_e32 v19, v45, v57
v_fmac_f32_e32 v20, v42, v58
v_fmac_f32_e32 v21, v43, v58
v_fmac_f32_e32 v22, v44, v58
v_fmac_f32_e32 v23, v45, v58
v_fmac_f32_e32 v24, v42, v59
v_fmac_f32_e32 v25, v43, v59
v_fmac_f32_e32 v26, v44, v59
v_fmac_f32_e32 v27, v45, v59
v_fmac_f32_e32 v28, v42, v60
v_fmac_f32_e32 v29, v43, v60
v_fmac_f32_e32 v30, v44, v60
v_fmac_f32_e32 v31, v45, v60
v_fmac_f32_e32 v32, v42, v61
v_fmac_f32_dpp v68, v68, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v71, v71, v107 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v69, v68, v71 div:2
v_add_f32_e64 v70, v68, -v71 div:2
v_fmac_f32_e32 v33, v43, v61
v_fmac_f32_e32 v34, v44, v61
v_fmac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v100, v70
ds_write_b32 v101, v71
ds_read_b128 v[42:45], v102 offset:25280
ds_read_b128 v[54:57], v97 offset:24768
ds_read_b128 v[58:61], v97 offset:24896
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 64775
s_call_b64 s[36:37], 6
s_branch 64773
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_nop
s_cmp_eq_u32 s64, 0
s_cbranch_scc0 8
s_branch 596
s_add_u32 s64, s64, 1
s_andn2_b32 s64, s64, 1
s_bitcmp1_b32 0, 26
s_cselect_b32 s92, s51, s52
s_cselect_b32 s93, 0, s53
s_sub_u32 s40, s40, s92
s_subb_u32 s41, s41, s93
s_cmp_eq_u32 s76, 0
s_cbranch_scc0 3
s_cbranch_scc1 610
s_nop 0
s_nop 0
s_min_u32 s54, s64, s76
s_sub_u32 s64, s64, s54
s_sub_u32 s76, s76, s54
s_sub_u32 s54, s54, 1
s_setpc_b64 s[36:37]
s_nop 0
s_nop 0
s_nop 0
s_bitcmp1_b32 s14, 17
s_cbranch_scc1 241
s_add_u32 s70, s70, s13
s_cmp_eq_u32 s70, 0
s_cbranch_scc1 238
s_mov_b32 s71, 0
s_bitcmp1_b32 s14, 16
s_cbranch_scc1 227
s_add_u32 s69, s12, 15
s_lshr_b32 s69, s69, 4
v_mov_b32_e32 v113, s70
v_mul_u32_u24_e32 v113, s69, v113
v_add_co_u32_e32 v113, vcc, s13, v113
v_sub_co_u32_e64 v113, vcc, v113, 1
v_ffbh_u32_e32 v116, s13
v_lshlrev_b32_e64 v117, v116, s13
v_and_b32_e32 v118, 0xffffff00, v117
v_cmp_eq_u32_e32 vcc, 0x80000000, v117
v_cvt_f32_u32_e32 v118, v118
v_rcp_f32_e32 v112, v118
v_subb_co_u32_e32 v115, vcc, 32, v116, vcc
v_cvt_f32_ubyte0_e32 v116, v117
v_fma_f32 v118, v118, v112, -1.0
v_fma_f32 v118, v116, v112, v118
v_fmaak_f32 v118, v118, v112, 0x9f000000
v_mul_f32_e32 v118, 0x5f800000, v118
v_mov_b32_e32 v116, 0
v_cvt_flr_i32_f32_e64 v118, -v118
v_lshl_add_u32 v112, v112, 9, v118
v_mad_u64_u32 v[116:117], vcc, v117, v112, v[116:117]
v_subb_co_u32_e64 v112, vcc, v112, -1, vcc
v_mul_hi_u32 v116, v113, v112
v_add_co_u32_e32 v112, vcc, v116, v113
v_addc_co_u32_e64 v116, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v115
v_cndmask_b32_e32 v112, v112, v116, vcc
v_alignbit_b32 v112, v116, v112, v115
s_nop 0
v_readfirstlane_b32 s68, v112
v_mul_u32_u24_e64 v112, v112, s4
v_ffbh_u32_e32 v116, s69
v_lshlrev_b32_e64 v117, v116, s69
v_and_b32_e32 v118, 0xffffff00, v117
v_cmp_eq_u32_e32 vcc, 0x80000000, v117
v_cvt_f32_u32_e32 v118, v118
v_rcp_f32_e32 v113, v118
v_subb_co_u32_e32 v115, vcc, 32, v116, vcc
v_cvt_f32_ubyte0_e32 v116, v117
v_fma_f32 v118, v118, v113, -1.0
v_fma_f32 v118, v116, v113, v118
v_fmaak_f32 v118, v118, v113, 0x9f000000
v_mul_f32_e32 v118, 0x5f800000, v118
v_mov_b32_e32 v116, 0
v_cvt_flr_i32_f32_e64 v118, -v118
v_lshl_add_u32 v113, v113, 9, v118
v_mad_u64_u32 v[116:117], vcc, v117, v113, v[116:117]
v_subb_co_u32_e64 v113, vcc, v113, -1, vcc
v_mul_hi_u32 v116, v112, v113
v_add_co_u32_e32 v113, vcc, v116, v112
v_addc_co_u32_e64 v116, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v115
v_cndmask_b32_e32 v113, v113, v116, vcc
v_alignbit_b32 v113, v116, v113, v115
v_readfirstlane_b32 s2, v112
v_readfirstlane_b32 s66, v113
s_mul_i32 s66, s66, s69
s_sub_u32 s66, s2, s66
v_sub_co_u32_e32 v113, vcc, s4, v113
v_sub_co_u32_e32 v113, vcc, s13, v113
v_and_b32_e64 v115, v0, 63
v_cmp_eq_u32_e64 vcc, v115, 0
v_cndmask_b32_e32 v113, 1, v113, vcc
s_sub_u32 s3, 0, s57
s_sub_u32 s15, 0, s56
v_mul_u32_u24_e64 v117, v113, 32
v_ffbh_u32_e32 v120, s3
v_lshlrev_b32_e64 v121, v120, s3
v_and_b32_e32 v119, 0xffffff00, v121
v_cmp_eq_u32_e32 vcc, 0x80000000, v121
v_cvt_f32_u32_e32 v119, v119
v_rcp_f32_e32 v115, v119
v_subb_co_u32_e32 v118, vcc, 32, v120, vcc
v_cvt_f32_ubyte0_e32 v120, v121
v_fma_f32 v119, v119, v115, -1.0
v_fma_f32 v119, v120, v115, v119
v_fmaak_f32 v119, v119, v115, 0x9f000000
v_mul_f32_e32 v119, 0x5f800000, v119
v_mov_b32_e32 v120, 0
v_cvt_flr_i32_f32_e64 v119, -v119
v_lshl_add_u32 v115, v115, 9, v119
v_mad_u64_u32 v[120:121], vcc, v121, v115, v[120:121]
v_subb_co_u32_e64 v115, vcc, v115, -1, vcc
v_mul_hi_u32 v119, v117, v115
v_add_co_u32_e32 v115, vcc, v119, v117
v_addc_co_u32_e64 v119, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v118
v_cndmask_b32_e32 v115, v115, v119, vcc
v_alignbit_b32 v115, v119, v115, v118
v_mad_i32_i24 v116, v115, s57, v117
v_mul_u32_u24_e64 v117, v115, 1
v_ffbh_u32_e32 v120, s15
v_lshlrev_b32_e64 v121, v120, s15
v_and_b32_e32 v119, 0xffffff00, v121
v_cmp_eq_u32_e32 vcc, 0x80000000, v121
v_cvt_f32_u32_e32 v119, v119
v_rcp_f32_e32 v115, v119
v_subb_co_u32_e32 v118, vcc, 32, v120, vcc
v_cvt_f32_ubyte0_e32 v120, v121
v_fma_f32 v119, v119, v115, -1.0
v_fma_f32 v119, v120, v115, v119
v_fmaak_f32 v119, v119, v115, 0x9f000000
v_mul_f32_e32 v119, 0x5f800000, v119
v_mov_b32_e32 v120, 0
v_cvt_flr_i32_f32_e64 v119, -v119
v_lshl_add_u32 v115, v115, 9, v119
v_mad_u64_u32 v[120:121], vcc, v121, v115, v[120:121]
v_subb_co_u32_e64 v115, vcc, v115, -1, vcc
v_mul_hi_u32 v119, v117, v115
v_add_co_u32_e32 v115, vcc, v119, v117
v_addc_co_u32_e64 v119, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v118
v_cndmask_b32_e32 v115, v115, v119, vcc
v_alignbit_b32 v115, v119, v115, v118
v_mad_i32_i24 v117, v115, s56, v117
v_readfirstlane_b32 s58, v116
v_readfirstlane_b32 s59, v117
v_readfirstlane_b32 s60, v115
v_add_co_u32_e32 v103, vcc, s58, v103
v_addc_co_u32_e64 v118, vcc, 0, 0, vcc
v_mad_i32_i24 v103, v118, s57, v103
v_mad_i32_i24 v105, v118, s62, v105
v_mad_i32_i24 v104, v118, s61, v104
v_cmp_ge_i32_e64 vcc, v104, 0
v_addc_co_u32_e64 v118, vcc, 0, 0, vcc
v_add_co_u32_e32 v105, vcc, v105, v118
v_mad_i32_i24 v104, v118, s56, v104
v_add_co_u32_e32 v104, vcc, s59, v104
v_addc_co_u32_e64 v118, vcc, 0, 0, vcc
v_add_co_u32_e32 v105, vcc, v105, v118
v_mad_i32_i24 v104, v118, s56, v104
v_add_co_u32_e32 v105, vcc, s60, v105
v_readlane_b32 s58, v116, 1
v_readlane_b32 s59, v117, 1
v_readlane_b32 s60, v115, 1
s_add_u32 s67, s66, s68
s_cmp_le_u32 s67, s69
s_cselect_b32 s92, 0x20000, 0
s_cselect_b32 s67, s67, s69
s_or_b32 s14, s14, s92
s_lshl_b32 s66, s66, 4
s_lshl_b32 s67, s67, 4
s_min_u32 s67, s67, s12
s_cmp_eq_u32 s4, s13
s_cselect_b32 s92, 0x20000, 0
s_or_b32 s14, s14, s92
s_bitset1_b32 s14, 16
s_branch 43
s_lshr_b32 s66, s66, 4
s_add_u32 s67, s66, s68
s_sub_u32 s67, s67, s69
s_mov_b32 s66, 0
s_lshl_b32 s67, s67, 4
s_min_u32 s67, s67, s12
s_bitset1_b32 s14, 17
s_branch 12
s_bitset1_b32 s14, 18
s_mov_b32 s43, 0
s_mov_b32 s55, -1
s_mov_b32 s64, 40
s_branch 31
s_add_u32 s65, s65, 16
s_cmp_ge_u32 s65, s67
s_cbranch_scc0 28
s_bitset1_b32 s14, 22
s_sub_u32 s70, s70, s13
s_subb_u32 s71, s71, 0
s_cbranch_scc1 65281
v_add_co_u32_e32 v103, vcc, s58, v103
v_addc_co_u32_e64 v112, vcc, 0, 0, vcc
v_mad_i32_i24 v103, v112, s57, v103
v_mad_i32_i24 v105, v112, s62, v105
v_mad_i32_i24 v104, v112, s61, v104
v_cmp_ge_i32_e64 vcc, v104, 0
v_addc_co_u32_e64 v112, vcc, 0, 0, vcc
v_add_co_u32_e32 v105, vcc, v105, v112
v_mad_i32_i24 v104, v112, s56, v104
v_add_co_u32_e32 v104, vcc, s59, v104
v_addc_co_u32_e64 v112, vcc, 0, 0, vcc
v_add_co_u32_e32 v105, vcc, v105, v112
v_mad_i32_i24 v104, v112, s56, v104
v_add_co_u32_e32 v105, vcc, s60, v105
s_mov_b32 s65, s66
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccz 168
v_subrev_co_u32_e32 v112, vcc, s57, v103
v_subrev_co_u32_e32 v113, vcc, s56, v104
s_bitcmp1_b32 s14, 22
s_cbranch_scc0 64
s_bitset0_b32 s14, 22
s_bfe_u32 s2, s14, 0x10014
v_mul_u32_u24_e32 v115, 3, v112
v_mul_u32_u24_e32 v116, 3, v113
v_cvt_pk_u16_u32 v118, v115, v116
v_and_b32_e64 v115, v0, 1
v_cmp_eq_u32_e64 vcc, v115, 1
v_cndmask_b32_e32 v118, v105, v118, vcc
v_lshrrev_b32_e32 v114, 1, v0
v_bfe_u32 v119, v114, s2, 1
v_lshrrev_b32_e32 v114, 1, v0
v_bfi_b32 v114, 1, v0, v114
v_lshrrev_b32_e32 v115, 2, v0
v_bfi_b32 v115, 1, v0, v115
v_cmp_eq_u32_e64 vcc, s2, 0
v_cndmask_b32_e32 v114, v115, v114, vcc
s_sub_u32 s2, 1, s2
v_lshrrev_b32_e32 v115, s2, v114
v_bfi_b32 v114, 32, v115, v114
v_and_b32_e32 v114, 63, v114
v_add_co_u32_e32 v115, vcc, 16, v114
v_and_b32_e64 v116, v0, 2
v_cmp_eq_u32_e64 vcc, v116, 0
v_cndmask_b32_e32 v115, v115, v114, vcc
v_lshlrev_b32_e32 v116, 14, v119
v_mad_u32_u24 v115, 4, v115, v116
v_add_co_u32_e32 v114, vcc, s78, v115
ds_write_b32 v114, v118
v_writelane_b32 v116, s14, 0
v_writelane_b32 v116, s67, 1
v_writelane_b32 v116, s66, 2
v_and_b32_e64 v114, v0, 63
v_cmp_ge_u32_e64 vcc, v114, 3
v_mov_b32_e32 v117, 0x4000
v_cndmask_b32_e32 v114, v114, v117, vcc
v_mad_i32_i24 v114, v114, 4, s78
ds_write_b32 v114, v116 offset:256
s_add_u32 s78, s78, 0x18c
s_cmp_eq_u32 s78, 0x10000
s_cselect_b32 s78, 0xc220, s78
v_mov_b32_dpp v114, v105 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v112, v112 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v113, v113 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_readfirstlane_b32 s63, v114
v_sub_co_u32_e64 v115, vcc, v114, s63
v_mul_lo_u32 v115, v115, s44
v_and_b32_e64 v119, v0, 3
v_ashrrev_i32_e64 v120, 1, s27
v_subrev_co_u32_e32 v119, vcc, v120, v119
v_ashrrev_i32_e64 v120, 1, s35
v_mad_i32_i24 v116, v120, 2, v119
s_bfe_u32 s2, s14, 0x10014
v_lshrrev_b32_e32 v118, 2, v0
v_and_b32_e32 v118, s2, v118
v_mad_i32_i24 v116, v118, 2, v116
v_add_co_u32_e64 v117, vcc, 1, s38
v_ashrrev_i32_e32 v117, 1, v117
v_add_co_u32_e64 v118, vcc, 1, s26
v_ashrrev_i32_e32 v118, 1, v118
v_sub_i32 v117, v117, v118
v_cmp_ge_u32_e64 s[2:3], v114, s8
v_mad_i32_i24 v112, v112, 3, v116
v_cmp_ge_u32_e64 s[22:23], v112, s11
v_add_co_u32_e32 v112, vcc, v112, v115
s_or_b64 s[22:23], s[22:23], s[2:3]
v_mad_i32_i24 v113, v113, 3, v117
v_cmp_ge_u32_e64 s[96:97], v113, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v1, v113, s49, v112
v_cndmask_b32_e64 v1, v1, -1, s[96:97]
v_add_co_u32_e32 v113, vcc, 1, v113
v_cmp_ge_u32_e64 s[96:97], v113, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v2, v113, s49, v112
v_cndmask_b32_e64 v2, v2, -1, s[96:97]
v_add_co_u32_e32 v113, vcc, 1, v113
v_cmp_ge_u32_e64 s[96:97], v113, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v3, v113, s49, v112
v_cndmask_b32_e64 v3, v3, -1, s[96:97]
v_add_co_u32_e32 v113, vcc, 1, v113
v_cmp_ge_u32_e64 s[96:97], v113, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v36, v113, s49, v112
v_cndmask_b32_e64 v36, v36, -1, s[96:97]
s_bitcmp1_b32 s14, 18
s_cbranch_scc1 147
s_lshr_b32 s15, -1, 16
s_and_b32 s15, s15, s44
s_lshr_b32 s22, s44, 16
s_mul_i32 s22, s22, s63
s_mul_i32 s40, s15, s63
s_lshl_b32 s15, s22, 16
s_lshr_b32 s22, s22, 16
s_add_u32 s40, s15, s40
s_addc_u32 s41, s22, 0
s_lshl_b64 s[40:41], s[40:41], 2
s_add_u32 s40, s40, s16
s_addc_u32 s41, s41, s17
s_lshr_b32 s2, s14, 6
s_xor_b32 s2, s2, s14
s_and_b32 s2, s2, 0x80000
s_cselect_b32 s2, s48, 0
s_lshl_b32 s2, s2, 2
s_add_u32 s40, s40, s2
s_addc_u32 s41, s41, 0
s_add_u32 s41, s41, 0x40000
s_branch 100
s_bitcmp1_b32 s14, 18
s_cbranch_scc1 122
v_mad_u32_u24 v114, 5, v0, 2
v_lshlrev_b32_e32 v112, 1, v0
v_bfi_b32 v114, 4, v114, v112
v_bfe_u32 v112, v114, 2, 1
v_lshlrev_b32_e32 v114, 1, v112
v_bfe_u32 v112, v0, 1, 1
v_add_co_u32_e32 v112, vcc, v112, v114
v_mad_u32_u24 v112, s35, 2, v112
v_sub_co_u32_e32 v114, vcc, s25, v112
v_sub_co_u32_e64 v114, vcc, v114, 1
s_bfe_u32 s2, s14, 0x10001
v_cmp_eq_u32_e64 vcc, s2, 1
v_cndmask_b32_e32 v112, v112, v114, vcc
v_cmp_ge_u32_e64 s[2:3], v112, s25
s_bfe_u32 s15, s14, 0x10018
v_bfe_u32 v115, v0, 2, s15
v_mul_lo_u32 v115, s48, v115
v_add_co_u32_e32 v112, vcc, v112, v115
v_mul_lo_u32 v113, s72, v106
v_add_co_u32_e32 v113, vcc, v113, v112
s_sub_u32 s15, s24, s38
s_sub_u32 s15, s15, 3
s_bitcmp1_b32 s14, 0
s_cselect_b32 s15, s15, s38
v_mov_b32_e32 v115, s15
v_cmp_ge_u32_e64 s[22:23], v115, s24
v_mad_i32_i24 v1, v115, s73, v113
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v1, v1, -1, s[22:23]
v_mov_b32_e32 v2, v1
v_add_co_u32_e64 v115, vcc, v115, 2
v_cmp_ge_u32_e64 s[22:23], v115, s24
v_mad_i32_i24 v36, v115, s73, v113
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v36, v36, -1, s[22:23]
v_add_co_u32_e64 v115, vcc, v115, 2
v_cmp_ge_u32_e64 s[22:23], v115, s24
v_mad_i32_i24 v3, v115, s73, v113
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v3, v3, -1, s[22:23]
s_bitcmp1_b32 s14, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v1, v2, v36, vcc
v_cndmask_b32_e32 v36, v36, v2, vcc
v_add_co_u32_e64 v112, vcc, v106, s65
v_cmp_lt_u32_e64 vcc, v112, s12
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
s_lshl_b64 s[40:41], s[40:41], 2
s_add_u32 s40, s40, s18
s_addc_u32 s41, s41, s19
s_lshr_b32 s2, s14, 6
s_xor_b32 s2, s2, s14
s_and_b32 s2, s2, 0x80000
s_cselect_b32 s2, s48, 0
s_lshl_b32 s2, s2, 2
s_add_u32 s40, s40, s2
s_addc_u32 s41, s41, 0
s_add_u32 s41, s41, 0x40000
s_mov_b32 s43, 0x20000
s_mov_b32 s55, -1
s_bfe_u32 s92, s14, 0x10014
s_lshl_b32 s64, s9, s92
s_bfe_u32 s92, s14, 0x10013
s_bfe_u32 s94, s14, 0x10019
s_xor_b32 s92, s92, s94
s_cselect_b32 s92, 1, 0
s_cselect_b32 s43, 0x20000, s43
s_and_b32 s92, s92, s64
s_sub_u32 s64, s64, s92
s_bitcmp1_b32 s14, 20
s_cselect_b32 s92, 0, 0x2000000
s_bitcmp1_b32 s9, 0
s_cselect_b32 s92, s92, 0
s_xor_b32 s14, s14, s92
s_cmp_eq_u32 s64, 0
s_cbranch_scc1 2
s_mov_b64 vcc, s[6:7]
s_branch 64940
s_and_b32 s92, 0x900000, s14
s_subb_u32 s35, s35, 1
s_cbranch_scc0 65235
s_and_b32 s92, 0x900000, s14
s_subb_u32 s35, s5, 1
s_add_u32 s38, s38, 4
s_cmp_ge_u32 s38, s24
s_cbranch_scc0 65229
s_mov_b32 s38, 1
s_cmp_ge_u32 s38, s24
s_addc_u32 s39, s39, 1
s_cmp_gt_u32 s39, 1
s_cbranch_scc0 65224
s_mov_b32 s39, 0
s_mov_b32 s38, 0
s_branch 65190
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s2, 0x3c3c3c3c
s_mov_b32 s3, s2
v_mov_b32_e32 v112, v4
v_mov_b32_e32 v113, v5
v_mov_b32_e32 v114, v6
v_mov_b32_e32 v115, v7
v_add_f32_dpp v112, v4, v4 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v113, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v114, v6, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v115, v7, v7 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v6, v6, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v7, v7, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v4, v4, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v5, v5, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v5, v6 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v4, v7 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v113, v114, v113 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v112, v115, v112 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v115, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v5, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v114, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v4, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v7, v113
v_add_f32_dpp v7, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v115 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v114, v115 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v6, v112
v_add_f32_dpp v6, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v114 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v115, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v5, v5, v4 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v4, v7, v6 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v6, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v6, v115 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v114, v114 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v5, v114, v5, s[2:3]
v_mov_b32_dpp v6, v6 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v6, v6 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v112, v8
v_mov_b32_e32 v113, v9
v_mov_b32_e32 v114, v10
v_mov_b32_e32 v115, v11
v_add_f32_dpp v112, v8, v8 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v113, v9, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v114, v10, v10 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v115, v11, v11 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v10, v10, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v11, v11, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v8, v8, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v9, v9, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v9, v10 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v8, v11 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v113, v114, v113 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v112, v115, v112 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v115, v9, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v9, v9, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v114, v8, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v8, v8, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v11, v113
v_add_f32_dpp v11, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v115 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v114, v115 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v10, v112
v_add_f32_dpp v10, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v114 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v115, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v9, v9, v8 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v7, v11, v10 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v10, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v10, v115 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v114, v114 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v8, v114, v9, s[2:3]
v_mov_b32_dpp v9, v10 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v9, v10 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v112, v12
v_mov_b32_e32 v113, v13
v_mov_b32_e32 v114, v14
v_mov_b32_e32 v115, v15
v_add_f32_dpp v112, v12, v12 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v113, v13, v13 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v114, v14, v14 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v115, v15, v15 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v14, v14, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v15, v15, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v12, v12, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v13, v13, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v13, v14 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v12, v15 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v113, v114, v113 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v112, v115, v112 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v115, v13, v13 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v13, v13, v13 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v114, v12, v12 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v12, v12, v12 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v15, v113
v_add_f32_dpp v15, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v115 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v114, v115 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v14, v112
v_add_f32_dpp v14, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v114 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v115, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v13, v13, v12 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v10, v15, v14 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v14, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v14, v115 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v114, v114 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v11, v114, v13, s[2:3]
v_mov_b32_dpp v12, v14 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v12, v14 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v112, v16
v_mov_b32_e32 v113, v17
v_mov_b32_e32 v114, v18
v_mov_b32_e32 v115, v19
v_add_f32_dpp v112, v16, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v113, v17, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v114, v18, v18 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v115, v19, v19 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v18, v18, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v19, v19, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v16, v16, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v17, v17, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v17, v18 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v16, v19 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v113, v114, v113 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v112, v115, v112 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v115, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v17, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v114, v16, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v16, v16, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v19, v113
v_add_f32_dpp v19, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v115 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v114, v115 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v18, v112
v_add_f32_dpp v18, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v114 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v115, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v17, v17, v16 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v13, v19, v18 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v18, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v18, v115 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v114, v114 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v14, v114, v17, s[2:3]
v_mov_b32_dpp v15, v18 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v15, v18 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v112, v20
v_mov_b32_e32 v113, v21
v_mov_b32_e32 v114, v22
v_mov_b32_e32 v115, v23
v_add_f32_dpp v112, v20, v20 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v113, v21, v21 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v114, v22, v22 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v115, v23, v23 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v22, v22, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v23, v23, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v20, v20, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v21, v21, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v21, v22 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v20, v23 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v113, v114, v113 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v112, v115, v112 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v115, v21, v21 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v21, v21, v21 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v114, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v20, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v23, v113
v_add_f32_dpp v23, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v115 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v114, v115 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v22, v112
v_add_f32_dpp v22, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v114 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v115, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v21, v21, v20 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v16, v23, v22 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v22, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v22, v115 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v114, v114 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v17, v114, v21, s[2:3]
v_mov_b32_dpp v18, v22 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v18, v22 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v112, v24
v_mov_b32_e32 v113, v25
v_mov_b32_e32 v114, v26
v_mov_b32_e32 v115, v27
v_add_f32_dpp v112, v24, v24 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v113, v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v114, v26, v26 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v115, v27, v27 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v26, v26, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v27, v27, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v24, v24, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v25, v25, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v25, v26 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v24, v27 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v113, v114, v113 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v112, v115, v112 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v115, v25, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v25, v25, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v114, v24, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v24, v24, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v27, v113
v_add_f32_dpp v27, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v115 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v114, v115 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v26, v112
v_add_f32_dpp v26, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v114 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v115, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v25, v25, v24 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v19, v27, v26 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v26, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v26, v115 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v114, v114 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v20, v114, v25, s[2:3]
v_mov_b32_dpp v21, v26 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v21, v26 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v112, v28
v_mov_b32_e32 v113, v29
v_mov_b32_e32 v114, v30
v_mov_b32_e32 v115, v31
v_add_f32_dpp v112, v28, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v113, v29, v29 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v114, v30, v30 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v115, v31, v31 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v30, v30, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v31, v31, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v28, v28, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v29, v29, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v29, v30 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v28, v31 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v113, v114, v113 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v112, v115, v112 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v115, v29, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v29, v29, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v114, v28, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v28, v28, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v31, v113
v_add_f32_dpp v31, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v115 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v114, v115 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v30, v112
v_add_f32_dpp v30, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v114 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v115, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v29, v29, v28 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v22, v31, v30 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v30, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v30, v115 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v114, v114 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v23, v114, v29, s[2:3]
v_mov_b32_dpp v24, v30 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v24, v30 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
s_setprio 1
v_mov_b32_e32 v112, v32
v_mov_b32_e32 v113, v33
v_mov_b32_e32 v114, v34
v_mov_b32_e32 v115, v35
v_add_f32_dpp v112, v32, v32 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v113, v33, v33 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v114, v34, v34 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v115, v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v34, v34, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v35, v35, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v32, v32, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v33, v33, v108 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v33, v34 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v32, v35 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v113, v114, v113 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v112, v115, v112 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v115, v33, v33 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v33, v33, v33 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v114, v32, v32 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v32, v32, v32 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v35, v113
v_add_f32_dpp v35, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v115 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v114, v115 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v34, v112
v_add_f32_dpp v34, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v114, v114 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v115, v113, v113 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v33, v33, v32 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v25, v35, v34 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v34, v112, v112 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v34, v115 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v114, v114 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v26, v114, v33, s[2:3]
v_mov_b32_dpp v27, v34 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v27, v34 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
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
s_bitcmp0_b32 s14, 21
s_addc_u32 s76, s1, 0
s_lshr_b32 s76, s76, 1
s_mul_i32 s76, s76, s5
s_lshr_b32 s76, s76, 1
s_mul_i32 s76, s76, s9
s_cmp_eq_u32 s76, 0
s_cbranch_scc1 64952
s_add_u32 s92, s75, s74
s_cmp_lt_i32 s92, 0
s_cbranch_scc0 233
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
ds_read_b32 v112, v62 offset:256
ds_read_b32 v37, v37
s_add_u32 s79, s79, 0x18c
s_cmp_eq_u32 s79, 0x10000
s_cselect_b32 s79, 0xc220, s79
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s77, v37
v_readlane_b32 s94, v112, 0
s_bitcmp1_b32 s94, 18
s_cbranch_scc1 1431
v_readlane_b32 s92, v112, 1
v_readlane_b32 s93, v112, 2
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
s_lshl_b64 s[84:85], s[84:85], 2
s_add_u32 s84, s84, s20
s_addc_u32 s85, s85, s21
s_mul_i32 s22, s46, s75
s_lshl_b32 s22, s22, 2
s_add_u32 s84, s84, s22
s_addc_u32 s85, s85, 0
s_add_u32 s85, s85, 0x40000
s_mov_b32 s87, 0x20000
s_bitcmp1_b32 s14, 7
s_cselect_b32 s91, 0x20000, 0
s_lshl_b32 s2, s75, 2
s_add_u32 s88, s30, s2
s_addc_u32 s89, s31, 0
s_add_u32 s89, s89, 0x40000
s_sub_u32 s90, s92, s75
s_cselect_b32 s91, 0, s91
s_sub_u32 s75, s92, s93
s_sub_u32 s75, s75, 1
s_sub_u32 s75, s75, s74
s_cselect_b32 s87, 0, s87
v_bfe_u32 v112, v37, 16, 16
v_bfe_u32 v113, v37, 0, 16
v_and_b32_e64 v114, v0, 7
v_sub_co_u32_e32 v115, vcc, 7, v114
v_min_u32_e32 v114, v114, v115
v_bfe_u32 v115, v114, 1, 1
v_bfe_u32 v114, v114, 0, 1
v_mov_b32_dpp v112, v112 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v113, v113 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_add_co_u32_e32 v112, vcc, v112, v115
v_add_co_u32_e32 v113, vcc, v113, v114
v_mov_b32_dpp v114, v37 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[2:3], v114, s8
v_sub_co_u32_e64 v114, vcc, v114, s77
v_mul_lo_u32 v114, v114, s45
v_xor_b32_dpp v115, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v115, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v116, v0, v0 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v116, v0, v0 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_add_co_u32_e32 v116, vcc, v113, v116
v_add_co_u32_e32 v115, vcc, v112, v115
v_lshlrev_b32_e32 v116, 1, v116
s_and_b32 s22, 1, s27
v_add_co_u32_e32 v116, vcc, s22, v116
v_lshlrev_b32_e32 v115, 1, v115
s_and_b32 s22, 1, s26
v_subrev_co_u32_e32 v115, vcc, s22, v115
v_mad_i32_i24 v37, v115, s50, v116
v_add_co_u32_e32 v37, vcc, v37, v114
v_subrev_co_u32_e32 v88, vcc, 1, v37
v_add_co_u32_e32 v91, vcc, s50, v37
v_add_co_u32_e32 v94, vcc, s50, v88
v_cmp_ge_u32_e64 s[98:99], v116, s29
s_or_b64 s[96:97], s[98:99], s[2:3]
v_subrev_co_u32_e32 v116, vcc, 1, v116
v_cmp_ge_u32_e64 s[98:99], v116, s29
s_or_b64 s[98:99], s[98:99], s[2:3]
v_cmp_ge_u32_e64 s[22:23], v115, s28
s_or_b64 s[100:101], s[96:97], s[22:23]
v_cndmask_b32_e64 v37, v37, -1, s[100:101]
s_or_b64 s[100:101], s[98:99], s[22:23]
v_cndmask_b32_e64 v88, v88, -1, s[100:101]
v_add_co_u32_e32 v115, vcc, 1, v115
v_cmp_ge_u32_e64 s[22:23], v115, s28
s_or_b64 s[100:101], s[96:97], s[22:23]
v_cndmask_b32_e64 v91, v91, -1, s[100:101]
s_or_b64 s[100:101], s[98:99], s[22:23]
v_cndmask_b32_e64 v94, v94, -1, s[100:101]
v_xor_b32_dpp v115, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v115, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v116, v0, v0 quad_perm:[1,1,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32_e32 v116, vcc, v113, v116
v_add_co_u32_e32 v115, vcc, v112, v115
v_lshlrev_b32_e32 v116, 1, v116
s_and_b32 s22, 1, s27
v_add_co_u32_e32 v116, vcc, s22, v116
v_lshlrev_b32_e32 v115, 1, v115
s_and_b32 s22, 1, s26
v_subrev_co_u32_e32 v115, vcc, s22, v115
v_mad_i32_i24 v62, v115, s50, v116
v_add_co_u32_e32 v62, vcc, v62, v114
v_subrev_co_u32_e32 v89, vcc, 1, v62
v_add_co_u32_e32 v92, vcc, s50, v62
v_add_co_u32_e32 v95, vcc, s50, v89
v_cmp_ge_u32_e64 s[98:99], v116, s29
s_or_b64 s[96:97], s[98:99], s[2:3]
v_subrev_co_u32_e32 v116, vcc, 1, v116
v_cmp_ge_u32_e64 s[98:99], v116, s29
s_or_b64 s[98:99], s[98:99], s[2:3]
v_cmp_ge_u32_e64 s[22:23], v115, s28
s_or_b64 s[100:101], s[96:97], s[22:23]
v_cndmask_b32_e64 v62, v62, -1, s[100:101]
s_or_b64 s[100:101], s[98:99], s[22:23]
v_cndmask_b32_e64 v89, v89, -1, s[100:101]
v_add_co_u32_e32 v115, vcc, 1, v115
v_cmp_ge_u32_e64 s[22:23], v115, s28
s_or_b64 s[100:101], s[96:97], s[22:23]
v_cndmask_b32_e64 v92, v92, -1, s[100:101]
s_or_b64 s[100:101], s[98:99], s[22:23]
v_cndmask_b32_e64 v95, v95, -1, s[100:101]
v_xor_b32_dpp v115, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v115, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v116, v0, v0 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v116, v0, v0 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xa
v_add_co_u32_e32 v116, vcc, v113, v116
v_add_co_u32_e32 v115, vcc, v112, v115
v_lshlrev_b32_e32 v116, 1, v116
s_and_b32 s22, 1, s27
v_add_co_u32_e32 v116, vcc, s22, v116
v_lshlrev_b32_e32 v115, 1, v115
s_and_b32 s22, 1, s26
v_subrev_co_u32_e32 v115, vcc, s22, v115
v_mad_i32_i24 v63, v115, s50, v116
v_add_co_u32_e32 v63, vcc, v63, v114
v_subrev_co_u32_e32 v90, vcc, 1, v63
v_add_co_u32_e32 v93, vcc, s50, v63
v_add_co_u32_e32 v96, vcc, s50, v90
v_cmp_ge_u32_e64 s[98:99], v116, s29
s_or_b64 s[96:97], s[98:99], s[2:3]
v_subrev_co_u32_e32 v116, vcc, 1, v116
v_cmp_ge_u32_e64 s[98:99], v116, s29
s_or_b64 s[98:99], s[98:99], s[2:3]
v_cmp_ge_u32_e64 s[22:23], v115, s28
s_or_b64 s[100:101], s[96:97], s[22:23]
v_cndmask_b32_e64 v63, v63, -1, s[100:101]
s_or_b64 s[100:101], s[98:99], s[22:23]
v_cndmask_b32_e64 v90, v90, -1, s[100:101]
v_add_co_u32_e32 v115, vcc, 1, v115
v_cmp_ge_u32_e64 s[22:23], v115, s28
s_or_b64 s[100:101], s[96:97], s[22:23]
v_cndmask_b32_e64 v93, v93, -1, s[100:101]
s_or_b64 s[100:101], s[98:99], s[22:23]
v_cndmask_b32_e64 v96, v96, -1, s[100:101]
v_and_b32_e64 v111, v0, 63
buffer_load_dword v111, v111, s[88:91], 0 idxen
s_mov_b64 vcc, s[6:7]
s_branch 64102
s_mov_b64 s[98:99], s[84:85]
s_mov_b32 s97, s87
v_bfe_u32 v112, s14, 21, 1
v_sub_co_u32_e64 v112, vcc, v112, 1
v_cndmask_b32_e32 v113, v91, v37, vcc
v_cndmask_b32_e32 v116, v94, v88, vcc
v_cndmask_b32_e32 v114, v92, v62, vcc
v_cndmask_b32_e32 v117, v95, v89, vcc
v_cndmask_b32_e32 v115, v93, v63, vcc
v_cndmask_b32_e32 v118, v96, v90, vcc
v_readlane_b32 s96, v111, 0
v_add_f32_e64 v4, v4, s96
buffer_store_dword v4, v113, s[84:87], 0 idxen
v_add_f32_e64 v7, v7, s96
buffer_store_dword v7, v116, s[84:87], 0 idxen
v_add_f32_e64 v5, v5, s96
buffer_store_dword v5, v114, s[84:87], 0 idxen
v_add_f32_e64 v8, v8, s96
buffer_store_dword v8, v117, s[84:87], 0 idxen
v_add_f32_e64 v6, v6, s96
buffer_store_dword v6, v115, s[84:87], 0 idxen
v_add_f32_e64 v9, v9, s96
buffer_store_dword v9, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 1
v_add_f32_e64 v10, v10, s96
buffer_store_dword v10, v113, s[84:87], 0 idxen
v_add_f32_e64 v13, v13, s96
buffer_store_dword v13, v116, s[84:87], 0 idxen
v_add_f32_e64 v11, v11, s96
buffer_store_dword v11, v114, s[84:87], 0 idxen
v_add_f32_e64 v14, v14, s96
buffer_store_dword v14, v117, s[84:87], 0 idxen
v_add_f32_e64 v12, v12, s96
buffer_store_dword v12, v115, s[84:87], 0 idxen
v_add_f32_e64 v15, v15, s96
buffer_store_dword v15, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s96, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 2
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 4
v_add_f32_e64 v16, v16, s96
buffer_store_dword v16, v113, s[84:87], 0 idxen
v_add_f32_e64 v19, v19, s96
buffer_store_dword v19, v116, s[84:87], 0 idxen
v_add_f32_e64 v17, v17, s96
buffer_store_dword v17, v114, s[84:87], 0 idxen
v_add_f32_e64 v20, v20, s96
buffer_store_dword v20, v117, s[84:87], 0 idxen
v_add_f32_e64 v18, v18, s96
buffer_store_dword v18, v115, s[84:87], 0 idxen
v_add_f32_e64 v21, v21, s96
buffer_store_dword v21, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 5
v_add_f32_e64 v22, v22, s96
buffer_store_dword v22, v113, s[84:87], 0 idxen
v_add_f32_e64 v25, v25, s96
buffer_store_dword v25, v116, s[84:87], 0 idxen
v_add_f32_e64 v23, v23, s96
buffer_store_dword v23, v114, s[84:87], 0 idxen
v_add_f32_e64 v26, v26, s96
buffer_store_dword v26, v117, s[84:87], 0 idxen
v_add_f32_e64 v24, v24, s96
buffer_store_dword v24, v115, s[84:87], 0 idxen
v_add_f32_e64 v27, v27, s96
buffer_store_dword v27, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s46, 3
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 10
s_cselect_b32 s87, 0, s87
s_bitcmp1_b32 s14, 21
s_cselect_b32 s87, s87, s97
s_cselect_b32 s84, s84, s98
s_cselect_b32 s85, s85, s99
s_cselect_b32 s97, 0, 16
s_cselect_b32 s98, 16, 0
s_lshl_b32 s99, s98, 2
s_add_u32 s75, s75, s97
s_add_u32 s88, s88, s99
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, s98
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
s_mov_b64 s[98:99], s[84:85]
s_mov_b32 s97, s87
v_bfe_u32 v112, s14, 21, 1
v_sub_co_u32_e64 v112, vcc, v112, 1
v_cndmask_b32_e32 v113, v91, v37, vcc
v_cndmask_b32_e32 v116, v94, v88, vcc
v_cndmask_b32_e32 v114, v92, v62, vcc
v_cndmask_b32_e32 v117, v95, v89, vcc
v_cndmask_b32_e32 v115, v93, v63, vcc
v_cndmask_b32_e32 v118, v96, v90, vcc
v_readlane_b32 s96, v111, 0
v_add_f32_e64 v4, v4, s96
v_mul_f32_e64 v119, v4, s32
v_cmp_lt_f32_e64 vcc, v4, 0
v_cndmask_b32_e32 v4, v4, v119, vcc
buffer_store_dword v4, v113, s[84:87], 0 idxen
v_add_f32_e64 v7, v7, s96
v_mul_f32_e64 v119, v7, s32
v_cmp_lt_f32_e64 vcc, v7, 0
v_cndmask_b32_e32 v7, v7, v119, vcc
buffer_store_dword v7, v116, s[84:87], 0 idxen
v_add_f32_e64 v5, v5, s96
v_mul_f32_e64 v119, v5, s32
v_cmp_lt_f32_e64 vcc, v5, 0
v_cndmask_b32_e32 v5, v5, v119, vcc
buffer_store_dword v5, v114, s[84:87], 0 idxen
v_add_f32_e64 v8, v8, s96
v_mul_f32_e64 v119, v8, s32
v_cmp_lt_f32_e64 vcc, v8, 0
v_cndmask_b32_e32 v8, v8, v119, vcc
buffer_store_dword v8, v117, s[84:87], 0 idxen
v_add_f32_e64 v6, v6, s96
v_mul_f32_e64 v119, v6, s32
v_cmp_lt_f32_e64 vcc, v6, 0
v_cndmask_b32_e32 v6, v6, v119, vcc
buffer_store_dword v6, v115, s[84:87], 0 idxen
v_add_f32_e64 v9, v9, s96
v_mul_f32_e64 v119, v9, s32
v_cmp_lt_f32_e64 vcc, v9, 0
v_cndmask_b32_e32 v9, v9, v119, vcc
buffer_store_dword v9, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 1
v_add_f32_e64 v10, v10, s96
v_mul_f32_e64 v119, v10, s32
v_cmp_lt_f32_e64 vcc, v10, 0
v_cndmask_b32_e32 v10, v10, v119, vcc
buffer_store_dword v10, v113, s[84:87], 0 idxen
v_add_f32_e64 v13, v13, s96
v_mul_f32_e64 v119, v13, s32
v_cmp_lt_f32_e64 vcc, v13, 0
v_cndmask_b32_e32 v13, v13, v119, vcc
buffer_store_dword v13, v116, s[84:87], 0 idxen
v_add_f32_e64 v11, v11, s96
v_mul_f32_e64 v119, v11, s32
v_cmp_lt_f32_e64 vcc, v11, 0
v_cndmask_b32_e32 v11, v11, v119, vcc
buffer_store_dword v11, v114, s[84:87], 0 idxen
v_add_f32_e64 v14, v14, s96
v_mul_f32_e64 v119, v14, s32
v_cmp_lt_f32_e64 vcc, v14, 0
v_cndmask_b32_e32 v14, v14, v119, vcc
buffer_store_dword v14, v117, s[84:87], 0 idxen
v_add_f32_e64 v12, v12, s96
v_mul_f32_e64 v119, v12, s32
v_cmp_lt_f32_e64 vcc, v12, 0
v_cndmask_b32_e32 v12, v12, v119, vcc
buffer_store_dword v12, v115, s[84:87], 0 idxen
v_add_f32_e64 v15, v15, s96
v_mul_f32_e64 v119, v15, s32
v_cmp_lt_f32_e64 vcc, v15, 0
v_cndmask_b32_e32 v15, v15, v119, vcc
buffer_store_dword v15, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s96, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 2
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 4
v_add_f32_e64 v16, v16, s96
v_mul_f32_e64 v119, v16, s32
v_cmp_lt_f32_e64 vcc, v16, 0
v_cndmask_b32_e32 v16, v16, v119, vcc
buffer_store_dword v16, v113, s[84:87], 0 idxen
v_add_f32_e64 v19, v19, s96
v_mul_f32_e64 v119, v19, s32
v_cmp_lt_f32_e64 vcc, v19, 0
v_cndmask_b32_e32 v19, v19, v119, vcc
buffer_store_dword v19, v116, s[84:87], 0 idxen
v_add_f32_e64 v17, v17, s96
v_mul_f32_e64 v119, v17, s32
v_cmp_lt_f32_e64 vcc, v17, 0
v_cndmask_b32_e32 v17, v17, v119, vcc
buffer_store_dword v17, v114, s[84:87], 0 idxen
v_add_f32_e64 v20, v20, s96
v_mul_f32_e64 v119, v20, s32
v_cmp_lt_f32_e64 vcc, v20, 0
v_cndmask_b32_e32 v20, v20, v119, vcc
buffer_store_dword v20, v117, s[84:87], 0 idxen
v_add_f32_e64 v18, v18, s96
v_mul_f32_e64 v119, v18, s32
v_cmp_lt_f32_e64 vcc, v18, 0
v_cndmask_b32_e32 v18, v18, v119, vcc
buffer_store_dword v18, v115, s[84:87], 0 idxen
v_add_f32_e64 v21, v21, s96
v_mul_f32_e64 v119, v21, s32
v_cmp_lt_f32_e64 vcc, v21, 0
v_cndmask_b32_e32 v21, v21, v119, vcc
buffer_store_dword v21, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 5
v_add_f32_e64 v22, v22, s96
v_mul_f32_e64 v119, v22, s32
v_cmp_lt_f32_e64 vcc, v22, 0
v_cndmask_b32_e32 v22, v22, v119, vcc
buffer_store_dword v22, v113, s[84:87], 0 idxen
v_add_f32_e64 v25, v25, s96
v_mul_f32_e64 v119, v25, s32
v_cmp_lt_f32_e64 vcc, v25, 0
v_cndmask_b32_e32 v25, v25, v119, vcc
buffer_store_dword v25, v116, s[84:87], 0 idxen
v_add_f32_e64 v23, v23, s96
v_mul_f32_e64 v119, v23, s32
v_cmp_lt_f32_e64 vcc, v23, 0
v_cndmask_b32_e32 v23, v23, v119, vcc
buffer_store_dword v23, v114, s[84:87], 0 idxen
v_add_f32_e64 v26, v26, s96
v_mul_f32_e64 v119, v26, s32
v_cmp_lt_f32_e64 vcc, v26, 0
v_cndmask_b32_e32 v26, v26, v119, vcc
buffer_store_dword v26, v117, s[84:87], 0 idxen
v_add_f32_e64 v24, v24, s96
v_mul_f32_e64 v119, v24, s32
v_cmp_lt_f32_e64 vcc, v24, 0
v_cndmask_b32_e32 v24, v24, v119, vcc
buffer_store_dword v24, v115, s[84:87], 0 idxen
v_add_f32_e64 v27, v27, s96
v_mul_f32_e64 v119, v27, s32
v_cmp_lt_f32_e64 vcc, v27, 0
v_cndmask_b32_e32 v27, v27, v119, vcc
buffer_store_dword v27, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s46, 3
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 10
s_cselect_b32 s87, 0, s87
s_bitcmp1_b32 s14, 21
s_cselect_b32 s87, s87, s97
s_cselect_b32 s84, s84, s98
s_cselect_b32 s85, s85, s99
s_cselect_b32 s97, 0, 16
s_cselect_b32 s98, 16, 0
s_lshl_b32 s99, s98, 2
s_add_u32 s75, s75, s97
s_add_u32 s88, s88, s99
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, s98
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
s_mov_b64 s[98:99], s[84:85]
s_mov_b32 s97, s87
v_bfe_u32 v112, s14, 21, 1
v_sub_co_u32_e64 v112, vcc, v112, 1
v_cndmask_b32_e32 v113, v91, v37, vcc
v_cndmask_b32_e32 v116, v94, v88, vcc
v_cndmask_b32_e32 v114, v92, v62, vcc
v_cndmask_b32_e32 v117, v95, v89, vcc
v_cndmask_b32_e32 v115, v93, v63, vcc
v_cndmask_b32_e32 v118, v96, v90, vcc
v_readlane_b32 s96, v111, 0
v_add_f32_e64 v4, v4, s96
v_mul_f32_e32 v4, 0xbfb8aa3b, v4
v_exp_f32_e32 v4, v4
v_add_f32_e32 v4, 1.0, v4
v_rcp_f32_e32 v4, v4
buffer_store_dword v4, v113, s[84:87], 0 idxen
v_add_f32_e64 v7, v7, s96
v_mul_f32_e32 v7, 0xbfb8aa3b, v7
v_exp_f32_e32 v7, v7
v_add_f32_e32 v7, 1.0, v7
v_rcp_f32_e32 v7, v7
buffer_store_dword v7, v116, s[84:87], 0 idxen
v_add_f32_e64 v5, v5, s96
v_mul_f32_e32 v5, 0xbfb8aa3b, v5
v_exp_f32_e32 v5, v5
v_add_f32_e32 v5, 1.0, v5
v_rcp_f32_e32 v5, v5
buffer_store_dword v5, v114, s[84:87], 0 idxen
v_add_f32_e64 v8, v8, s96
v_mul_f32_e32 v8, 0xbfb8aa3b, v8
v_exp_f32_e32 v8, v8
v_add_f32_e32 v8, 1.0, v8
v_rcp_f32_e32 v8, v8
buffer_store_dword v8, v117, s[84:87], 0 idxen
v_add_f32_e64 v6, v6, s96
v_mul_f32_e32 v6, 0xbfb8aa3b, v6
v_exp_f32_e32 v6, v6
v_add_f32_e32 v6, 1.0, v6
v_rcp_f32_e32 v6, v6
buffer_store_dword v6, v115, s[84:87], 0 idxen
v_add_f32_e64 v9, v9, s96
v_mul_f32_e32 v9, 0xbfb8aa3b, v9
v_exp_f32_e32 v9, v9
v_add_f32_e32 v9, 1.0, v9
v_rcp_f32_e32 v9, v9
buffer_store_dword v9, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 1
v_add_f32_e64 v10, v10, s96
v_mul_f32_e32 v10, 0xbfb8aa3b, v10
v_exp_f32_e32 v10, v10
v_add_f32_e32 v10, 1.0, v10
v_rcp_f32_e32 v10, v10
buffer_store_dword v10, v113, s[84:87], 0 idxen
v_add_f32_e64 v13, v13, s96
v_mul_f32_e32 v13, 0xbfb8aa3b, v13
v_exp_f32_e32 v13, v13
v_add_f32_e32 v13, 1.0, v13
v_rcp_f32_e32 v13, v13
buffer_store_dword v13, v116, s[84:87], 0 idxen
v_add_f32_e64 v11, v11, s96
v_mul_f32_e32 v11, 0xbfb8aa3b, v11
v_exp_f32_e32 v11, v11
v_add_f32_e32 v11, 1.0, v11
v_rcp_f32_e32 v11, v11
buffer_store_dword v11, v114, s[84:87], 0 idxen
v_add_f32_e64 v14, v14, s96
v_mul_f32_e32 v14, 0xbfb8aa3b, v14
v_exp_f32_e32 v14, v14
v_add_f32_e32 v14, 1.0, v14
v_rcp_f32_e32 v14, v14
buffer_store_dword v14, v117, s[84:87], 0 idxen
v_add_f32_e64 v12, v12, s96
v_mul_f32_e32 v12, 0xbfb8aa3b, v12
v_exp_f32_e32 v12, v12
v_add_f32_e32 v12, 1.0, v12
v_rcp_f32_e32 v12, v12
buffer_store_dword v12, v115, s[84:87], 0 idxen
v_add_f32_e64 v15, v15, s96
v_mul_f32_e32 v15, 0xbfb8aa3b, v15
v_exp_f32_e32 v15, v15
v_add_f32_e32 v15, 1.0, v15
v_rcp_f32_e32 v15, v15
buffer_store_dword v15, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s96, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 2
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 4
v_add_f32_e64 v16, v16, s96
v_mul_f32_e32 v16, 0xbfb8aa3b, v16
v_exp_f32_e32 v16, v16
v_add_f32_e32 v16, 1.0, v16
v_rcp_f32_e32 v16, v16
buffer_store_dword v16, v113, s[84:87], 0 idxen
v_add_f32_e64 v19, v19, s96
v_mul_f32_e32 v19, 0xbfb8aa3b, v19
v_exp_f32_e32 v19, v19
v_add_f32_e32 v19, 1.0, v19
v_rcp_f32_e32 v19, v19
buffer_store_dword v19, v116, s[84:87], 0 idxen
v_add_f32_e64 v17, v17, s96
v_mul_f32_e32 v17, 0xbfb8aa3b, v17
v_exp_f32_e32 v17, v17
v_add_f32_e32 v17, 1.0, v17
v_rcp_f32_e32 v17, v17
buffer_store_dword v17, v114, s[84:87], 0 idxen
v_add_f32_e64 v20, v20, s96
v_mul_f32_e32 v20, 0xbfb8aa3b, v20
v_exp_f32_e32 v20, v20
v_add_f32_e32 v20, 1.0, v20
v_rcp_f32_e32 v20, v20
buffer_store_dword v20, v117, s[84:87], 0 idxen
v_add_f32_e64 v18, v18, s96
v_mul_f32_e32 v18, 0xbfb8aa3b, v18
v_exp_f32_e32 v18, v18
v_add_f32_e32 v18, 1.0, v18
v_rcp_f32_e32 v18, v18
buffer_store_dword v18, v115, s[84:87], 0 idxen
v_add_f32_e64 v21, v21, s96
v_mul_f32_e32 v21, 0xbfb8aa3b, v21
v_exp_f32_e32 v21, v21
v_add_f32_e32 v21, 1.0, v21
v_rcp_f32_e32 v21, v21
buffer_store_dword v21, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 5
v_add_f32_e64 v22, v22, s96
v_mul_f32_e32 v22, 0xbfb8aa3b, v22
v_exp_f32_e32 v22, v22
v_add_f32_e32 v22, 1.0, v22
v_rcp_f32_e32 v22, v22
buffer_store_dword v22, v113, s[84:87], 0 idxen
v_add_f32_e64 v25, v25, s96
v_mul_f32_e32 v25, 0xbfb8aa3b, v25
v_exp_f32_e32 v25, v25
v_add_f32_e32 v25, 1.0, v25
v_rcp_f32_e32 v25, v25
buffer_store_dword v25, v116, s[84:87], 0 idxen
v_add_f32_e64 v23, v23, s96
v_mul_f32_e32 v23, 0xbfb8aa3b, v23
v_exp_f32_e32 v23, v23
v_add_f32_e32 v23, 1.0, v23
v_rcp_f32_e32 v23, v23
buffer_store_dword v23, v114, s[84:87], 0 idxen
v_add_f32_e64 v26, v26, s96
v_mul_f32_e32 v26, 0xbfb8aa3b, v26
v_exp_f32_e32 v26, v26
v_add_f32_e32 v26, 1.0, v26
v_rcp_f32_e32 v26, v26
buffer_store_dword v26, v117, s[84:87], 0 idxen
v_add_f32_e64 v24, v24, s96
v_mul_f32_e32 v24, 0xbfb8aa3b, v24
v_exp_f32_e32 v24, v24
v_add_f32_e32 v24, 1.0, v24
v_rcp_f32_e32 v24, v24
buffer_store_dword v24, v115, s[84:87], 0 idxen
v_add_f32_e64 v27, v27, s96
v_mul_f32_e32 v27, 0xbfb8aa3b, v27
v_exp_f32_e32 v27, v27
v_add_f32_e32 v27, 1.0, v27
v_rcp_f32_e32 v27, v27
buffer_store_dword v27, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s46, 3
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 10
s_cselect_b32 s87, 0, s87
s_bitcmp1_b32 s14, 21
s_cselect_b32 s87, s87, s97
s_cselect_b32 s84, s84, s98
s_cselect_b32 s85, s85, s99
s_cselect_b32 s97, 0, 16
s_cselect_b32 s98, 16, 0
s_lshl_b32 s99, s98, 2
s_add_u32 s75, s75, s97
s_add_u32 s88, s88, s99
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, s98
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
s_mov_b64 s[98:99], s[84:85]
s_mov_b32 s97, s87
v_bfe_u32 v112, s14, 21, 1
v_sub_co_u32_e64 v112, vcc, v112, 1
v_cndmask_b32_e32 v113, v91, v37, vcc
v_cndmask_b32_e32 v116, v94, v88, vcc
v_cndmask_b32_e32 v114, v92, v62, vcc
v_cndmask_b32_e32 v117, v95, v89, vcc
v_cndmask_b32_e32 v115, v93, v63, vcc
v_cndmask_b32_e32 v118, v96, v90, vcc
v_readlane_b32 s96, v111, 0
v_add_f32_e64 v4, v4, s96
v_mul_f32_e64 v4, v4, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v4
v_mul_f32_e32 v120, 0xbfb8aa3b, v4
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v4, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v4, v4
v_mul_f32_e32 v4, v4, v119
v_mul_f32_e64 v4, v4, s32
buffer_store_dword v4, v113, s[84:87], 0 idxen
v_add_f32_e64 v7, v7, s96
v_mul_f32_e64 v7, v7, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v7
v_mul_f32_e32 v120, 0xbfb8aa3b, v7
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v7, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v7, v7
v_mul_f32_e32 v7, v7, v119
v_mul_f32_e64 v7, v7, s32
buffer_store_dword v7, v116, s[84:87], 0 idxen
v_add_f32_e64 v5, v5, s96
v_mul_f32_e64 v5, v5, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v5
v_mul_f32_e32 v120, 0xbfb8aa3b, v5
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v5, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v5, v5
v_mul_f32_e32 v5, v5, v119
v_mul_f32_e64 v5, v5, s32
buffer_store_dword v5, v114, s[84:87], 0 idxen
v_add_f32_e64 v8, v8, s96
v_mul_f32_e64 v8, v8, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v8
v_mul_f32_e32 v120, 0xbfb8aa3b, v8
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v8, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v8, v8
v_mul_f32_e32 v8, v8, v119
v_mul_f32_e64 v8, v8, s32
buffer_store_dword v8, v117, s[84:87], 0 idxen
v_add_f32_e64 v6, v6, s96
v_mul_f32_e64 v6, v6, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v6
v_mul_f32_e32 v120, 0xbfb8aa3b, v6
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v6, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v6, v6
v_mul_f32_e32 v6, v6, v119
v_mul_f32_e64 v6, v6, s32
buffer_store_dword v6, v115, s[84:87], 0 idxen
v_add_f32_e64 v9, v9, s96
v_mul_f32_e64 v9, v9, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v9
v_mul_f32_e32 v120, 0xbfb8aa3b, v9
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v9, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v9, v9
v_mul_f32_e32 v9, v9, v119
v_mul_f32_e64 v9, v9, s32
buffer_store_dword v9, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 1
v_add_f32_e64 v10, v10, s96
v_mul_f32_e64 v10, v10, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v10
v_mul_f32_e32 v120, 0xbfb8aa3b, v10
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v10, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v10, v10
v_mul_f32_e32 v10, v10, v119
v_mul_f32_e64 v10, v10, s32
buffer_store_dword v10, v113, s[84:87], 0 idxen
v_add_f32_e64 v13, v13, s96
v_mul_f32_e64 v13, v13, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v13
v_mul_f32_e32 v120, 0xbfb8aa3b, v13
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v13, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v13, v13
v_mul_f32_e32 v13, v13, v119
v_mul_f32_e64 v13, v13, s32
buffer_store_dword v13, v116, s[84:87], 0 idxen
v_add_f32_e64 v11, v11, s96
v_mul_f32_e64 v11, v11, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v11
v_mul_f32_e32 v120, 0xbfb8aa3b, v11
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v11, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v11, v11
v_mul_f32_e32 v11, v11, v119
v_mul_f32_e64 v11, v11, s32
buffer_store_dword v11, v114, s[84:87], 0 idxen
v_add_f32_e64 v14, v14, s96
v_mul_f32_e64 v14, v14, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v14
v_mul_f32_e32 v120, 0xbfb8aa3b, v14
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v14, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v14, v14
v_mul_f32_e32 v14, v14, v119
v_mul_f32_e64 v14, v14, s32
buffer_store_dword v14, v117, s[84:87], 0 idxen
v_add_f32_e64 v12, v12, s96
v_mul_f32_e64 v12, v12, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v12
v_mul_f32_e32 v120, 0xbfb8aa3b, v12
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v12, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v12, v12
v_mul_f32_e32 v12, v12, v119
v_mul_f32_e64 v12, v12, s32
buffer_store_dword v12, v115, s[84:87], 0 idxen
v_add_f32_e64 v15, v15, s96
v_mul_f32_e64 v15, v15, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v15
v_mul_f32_e32 v120, 0xbfb8aa3b, v15
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v15, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v15, v15
v_mul_f32_e32 v15, v15, v119
v_mul_f32_e64 v15, v15, s32
buffer_store_dword v15, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s96, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 2
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 4
v_add_f32_e64 v16, v16, s96
v_mul_f32_e64 v16, v16, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v16
v_mul_f32_e32 v120, 0xbfb8aa3b, v16
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v16, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v16, v16
v_mul_f32_e32 v16, v16, v119
v_mul_f32_e64 v16, v16, s32
buffer_store_dword v16, v113, s[84:87], 0 idxen
v_add_f32_e64 v19, v19, s96
v_mul_f32_e64 v19, v19, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v19
v_mul_f32_e32 v120, 0xbfb8aa3b, v19
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v19, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v19, v19
v_mul_f32_e32 v19, v19, v119
v_mul_f32_e64 v19, v19, s32
buffer_store_dword v19, v116, s[84:87], 0 idxen
v_add_f32_e64 v17, v17, s96
v_mul_f32_e64 v17, v17, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v17
v_mul_f32_e32 v120, 0xbfb8aa3b, v17
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v17, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v17, v17
v_mul_f32_e32 v17, v17, v119
v_mul_f32_e64 v17, v17, s32
buffer_store_dword v17, v114, s[84:87], 0 idxen
v_add_f32_e64 v20, v20, s96
v_mul_f32_e64 v20, v20, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v20
v_mul_f32_e32 v120, 0xbfb8aa3b, v20
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v20, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v20, v20
v_mul_f32_e32 v20, v20, v119
v_mul_f32_e64 v20, v20, s32
buffer_store_dword v20, v117, s[84:87], 0 idxen
v_add_f32_e64 v18, v18, s96
v_mul_f32_e64 v18, v18, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v18
v_mul_f32_e32 v120, 0xbfb8aa3b, v18
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v18, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v18, v18
v_mul_f32_e32 v18, v18, v119
v_mul_f32_e64 v18, v18, s32
buffer_store_dword v18, v115, s[84:87], 0 idxen
v_add_f32_e64 v21, v21, s96
v_mul_f32_e64 v21, v21, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v21
v_mul_f32_e32 v120, 0xbfb8aa3b, v21
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v21, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v21, v21
v_mul_f32_e32 v21, v21, v119
v_mul_f32_e64 v21, v21, s32
buffer_store_dword v21, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v111, 5
v_add_f32_e64 v22, v22, s96
v_mul_f32_e64 v22, v22, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v22
v_mul_f32_e32 v120, 0xbfb8aa3b, v22
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v22, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v22, v22
v_mul_f32_e32 v22, v22, v119
v_mul_f32_e64 v22, v22, s32
buffer_store_dword v22, v113, s[84:87], 0 idxen
v_add_f32_e64 v25, v25, s96
v_mul_f32_e64 v25, v25, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v25
v_mul_f32_e32 v120, 0xbfb8aa3b, v25
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v25, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v25, v25
v_mul_f32_e32 v25, v25, v119
v_mul_f32_e64 v25, v25, s32
buffer_store_dword v25, v116, s[84:87], 0 idxen
v_add_f32_e64 v23, v23, s96
v_mul_f32_e64 v23, v23, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v23
v_mul_f32_e32 v120, 0xbfb8aa3b, v23
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v23, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v23, v23
v_mul_f32_e32 v23, v23, v119
v_mul_f32_e64 v23, v23, s32
buffer_store_dword v23, v114, s[84:87], 0 idxen
v_add_f32_e64 v26, v26, s96
v_mul_f32_e64 v26, v26, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v26
v_mul_f32_e32 v120, 0xbfb8aa3b, v26
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v26, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v26, v26
v_mul_f32_e32 v26, v26, v119
v_mul_f32_e64 v26, v26, s32
buffer_store_dword v26, v117, s[84:87], 0 idxen
v_add_f32_e64 v24, v24, s96
v_mul_f32_e64 v24, v24, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v24
v_mul_f32_e32 v120, 0xbfb8aa3b, v24
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v24, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v24, v24
v_mul_f32_e32 v24, v24, v119
v_mul_f32_e64 v24, v24, s32
buffer_store_dword v24, v115, s[84:87], 0 idxen
v_add_f32_e64 v27, v27, s96
v_mul_f32_e64 v27, v27, s33
v_mul_f32_e32 v119, 0x3fb8aa3b, v27
v_mul_f32_e32 v120, 0xbfb8aa3b, v27
v_exp_f32_e32 v119, v119
v_exp_f32_e32 v120, v120
v_add_f32_e32 v27, v119, v120
v_sub_f32_e32 v119, v119, v120
v_rcp_f32_e32 v27, v27
v_mul_f32_e32 v27, v27, v119
v_mul_f32_e64 v27, v27, s32
buffer_store_dword v27, v118, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s46, 3
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 10
s_cselect_b32 s87, 0, s87
s_bitcmp1_b32 s14, 21
s_cselect_b32 s87, s87, s97
s_cselect_b32 s84, s84, s98
s_cselect_b32 s85, s85, s99
s_cselect_b32 s97, 0, 16
s_cselect_b32 s98, 16, 0
s_lshl_b32 s99, s98, 2
s_add_u32 s75, s75, s97
s_add_u32 s88, s88, s99
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, s98
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
