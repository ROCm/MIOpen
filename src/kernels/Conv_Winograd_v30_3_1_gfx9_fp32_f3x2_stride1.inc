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
v_mov_b32_e32 v103, 0
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
s_add_u32 s82, s82, 0x3348
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
s_add_u32 s80, s80, 0x3620
s_addc_u32 s81, s81, 0
s_branch 18
s_load_dword s32, s[84:85], 0x60
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x38b0
s_addc_u32 s81, s81, 0
s_branch 11
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x3d28
s_addc_u32 s81, s81, 0
s_branch 6
s_load_dwordx2 s[32:33], s[84:85], 0x60
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x4198
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
v_ffbh_u32_e32 v106, s13
v_lshlrev_b32_e64 v107, v106, s13
v_and_b32_e32 v108, 0xffffff00, v107
v_cmp_eq_u32_e32 vcc, 0x80000000, v107
v_cvt_f32_u32_e32 v108, v108
v_rcp_f32_e32 v104, v108
v_subb_co_u32_e32 v105, vcc, 32, v106, vcc
v_cvt_f32_ubyte0_e32 v106, v107
v_fma_f32 v108, v108, v104, -1.0
v_fma_f32 v108, v106, v104, v108
v_madak_f32 v108, v108, v104, 0x9f000000
v_mul_f32_e32 v108, 0x5f800000, v108
v_mov_b32_e32 v106, 0
v_cvt_flr_i32_f32_e64 v108, -v108
v_lshl_add_u32 v104, v104, 9, v108
v_mad_u64_u32 v[106:107], vcc, v107, v104, v[106:107]
v_subb_co_u32_e64 v104, vcc, v104, -1, vcc
v_mul_hi_u32 v106, s4, v104
v_add_co_u32_e64 v104, vcc, v106, s4
v_addc_co_u32_e64 v106, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v105
v_cndmask_b32_e32 v104, v104, v106, vcc
v_alignbit_b32 v104, v106, v104, v105
s_nop 0
v_readfirstlane_b32 s88, v104
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
s_and_b32 s88, 0, s26
s_addc_u32 s88, s28, 0
s_ashr_i32 s88, s88, 0
s_add_u32 s86, s88, 2
v_mov_b32_e32 v105, 0x55555556
v_mul_hi_u32 v105, v105, s86
s_nop 0
v_readfirstlane_b32 s86, v105
s_andn2_b32 s88, 0, s27
s_addc_u32 s88, s29, 0
s_ashr_i32 s88, s88, 0
s_add_u32 s87, s88, 2
v_mov_b32_e32 v105, 0x55555556
v_mul_hi_u32 v105, v105, s87
s_nop 0
v_readfirstlane_b32 s87, v105
s_sub_u32 s57, 0, s87
s_sub_u32 s56, 0, s86
s_add_u32 s1, s24, 1
v_mov_b32_e32 v105, 0x80000000
v_mul_hi_u32 v105, v105, s1
s_nop 0
v_readfirstlane_b32 s1, v105
s_add_u32 s5, s25, 1
v_mov_b32_e32 v105, 0x80000000
v_mul_hi_u32 v105, v105, s5
s_nop 0
v_readfirstlane_b32 s5, v105
v_mad_i32_i24 v104, 2, s1, -1
v_sub_co_u32_e64 v104, vcc, v104, s24
v_addc_co_u32_e64 v104, vcc, 0, 0, vcc
s_nop 0
v_readfirstlane_b32 s88, v104
s_and_b32 s88, s88, 0
s_and_b32 s88, s88, s1
s_add_u32 s1, s1, s88
v_readfirstlane_b32 s89, v0
s_and_b32 s90, s89, 64
s_cselect_b32 s90, 0x80000, 0
s_or_b32 s14, s14, s90
s_lshl_b32 s51, s47, 1
s_mov_b64 s[52:53], 0
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
v_bfe_u32 v105, v0, 2, 6
v_lshrrev_b32_e32 v98, 1, v105
s_bitcmp0_b32 s89, 8
s_cselect_b32 s90, 0x1000000, 0
s_or_b32 s90, s90, 0x100000
s_and_b32 s90, s14, s90
s_cselect_b32 s90, 0, 15
v_bfi_b32 v98, s90, v105, v98
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
v_mov_b32_e32 v105, s4
v_mov_b32_e32 v106, s13
v_and_b32_e32 v107, 3, v0
v_cmp_eq_u32_e32 vcc, 2, v107
v_cndmask_b32_e32 v105, v105, v106, vcc
v_cmp_eq_u32_e32 vcc, 1, v107
v_cndmask_b32_e32 v108, 0, v98, vcc
s_bitcmp1_b32 s14, 20
s_cbranch_scc0 4
v_add_co_u32_e64 v106, vcc, v98, 8
v_cmp_eq_u32_e32 vcc, 0, v107
v_cndmask_b32_e32 v108, v108, v106, vcc
v_cmp_eq_u32_e64 s[90:91], 3, v107
v_bfe_u32 v96, v108, 0, 5
v_mad_u32_u24 v96, v105, 32, v96
v_ffbh_u32_e32 v110, s87
v_lshlrev_b32_e64 v111, v110, s87
v_and_b32_e32 v112, 0xffffff00, v111
v_cmp_eq_u32_e32 vcc, 0x80000000, v111
v_cvt_f32_u32_e32 v112, v112
v_rcp_f32_e32 v97, v112
v_subb_co_u32_e32 v109, vcc, 32, v110, vcc
v_cvt_f32_ubyte0_e32 v110, v111
v_fma_f32 v112, v112, v97, -1.0
v_fma_f32 v112, v110, v97, v112
v_madak_f32 v112, v112, v97, 0x9f000000
v_mul_f32_e32 v112, 0x5f800000, v112
v_mov_b32_e32 v110, 0
v_cvt_flr_i32_f32_e64 v112, -v112
v_lshl_add_u32 v97, v97, 9, v112
v_mad_u64_u32 v[110:111], vcc, v111, v97, v[110:111]
v_subb_co_u32_e64 v97, vcc, v97, -1, vcc
v_mul_hi_u32 v110, v96, v97
v_add_co_u32_e32 v97, vcc, v110, v96
v_addc_co_u32_e64 v110, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v109
v_cndmask_b32_e32 v97, v97, v110, vcc
v_alignbit_b32 v97, v110, v97, v109
v_mad_i32_i24 v95, v97, s57, v96
v_lshrrev_b32_e32 v96, 5, v108
v_mad_u32_u24 v96, v97, 1, v96
v_cndmask_b32_e64 v96, v96, 1, s[90:91]
v_ffbh_u32_e32 v110, s86
v_lshlrev_b32_e64 v111, v110, s86
v_and_b32_e32 v112, 0xffffff00, v111
v_cmp_eq_u32_e32 vcc, 0x80000000, v111
v_cvt_f32_u32_e32 v112, v112
v_rcp_f32_e32 v97, v112
v_subb_co_u32_e32 v109, vcc, 32, v110, vcc
v_cvt_f32_ubyte0_e32 v110, v111
v_fma_f32 v112, v112, v97, -1.0
v_fma_f32 v112, v110, v97, v112
v_madak_f32 v112, v112, v97, 0x9f000000
v_mul_f32_e32 v112, 0x5f800000, v112
v_mov_b32_e32 v110, 0
v_cvt_flr_i32_f32_e64 v112, -v112
v_lshl_add_u32 v97, v97, 9, v112
v_mad_u64_u32 v[110:111], vcc, v111, v97, v[110:111]
v_subb_co_u32_e64 v97, vcc, v97, -1, vcc
v_mul_hi_u32 v110, v96, v97
v_add_co_u32_e32 v97, vcc, v110, v96
v_addc_co_u32_e64 v110, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v109
v_cndmask_b32_e32 v97, v97, v110, vcc
v_alignbit_b32 v97, v110, v97, v109
v_mad_i32_i24 v96, v97, s56, v96
v_readlane_b32 s58, v95, 2
v_readlane_b32 s59, v96, 2
v_readlane_b32 s60, v97, 2
v_readlane_b32 s61, v96, 3
v_readlane_b32 s62, v97, 3
v_add_co_u32_e64 v95, vcc, v95, s57
v_add_co_u32_e64 v96, vcc, v96, s56
v_mov_b32_dpp v97, v97 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v95, v95 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v96, v96 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x20000
s_mov_b32 s86, 0x80000000
s_mov_b32 s87, 0x20000
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 5
v_xor_b32_dpp v99, v0, v0 quad_perm:[2,3,2,1] row_mask:0xf bank_mask:0xf
v_subrev_co_u32_e32 v99, vcc, 1, v99
v_cvt_f32_i32_e32 v99, v99
s_branch 4
v_xor_b32_dpp v99, v0, v0 quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32_e32 v99, vcc, 1, v99
v_cvt_f32_i32_e32 v99, v99
v_mov_b32_e32 v100, 1
v_xor_b32_dpp v100, v0, v0 quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v100, v0, v0 quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32_e32 v100, vcc, 1, v100
v_mov_b32_e32 v101, 1
v_xor_b32_dpp v101, v0, v0 quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v101, v0, v0 quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32_e32 v101, vcc, 1, v101
v_cvt_f32_i32_e32 v100, v100
v_cvt_f32_i32_e32 v101, v101
v_lshrrev_b32_e64 v104, 2, s74
v_and_b32_e32 v105, 3, v0
v_bfe_u32 v106, v0, 4, 3
v_mad_u32_u24 v94, v106, 4, v105
v_lshlrev_b32_e32 v94, 4, v94
v_mad_u32_u24 v89, v104, 4, v105
v_lshlrev_b32_e32 v89, 4, v89
v_bfe_u32 v104, v0, 2, 2
v_and_b32_e32 v105, 1, v104
v_mad_u32_u24 v107, v104, 16, v105
v_lshlrev_b32_e32 v107, 6, v107
v_xor_b32_e32 v89, v89, v107
v_mul_u32_u24_e32 v107, 0x400, v104
v_xor_b32_e32 v94, v94, v107
s_lshr_b32 s74, s74, 0
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 47
s_and_b32 s3, s14, 0x1100000
s_addc_u32 s3, 0, 0
v_lshrrev_b32_e32 v107, 1, v0
s_mul_i32 s2, 60, s3
s_sub_u32 s2, 63, s2
v_bfi_b32 v107, s2, v0, v107
v_and_b32_e32 v104, 1, v107
v_bfe_u32 v105, v107, 1, 1
v_xor_b32_e32 v104, v104, v105
v_bfe_u32 v106, v107, 3, 1
v_mad_u32_u24 v105, v105, 2, v106
v_mul_u32_u24_e32 v104, 0x118, v104
v_bfe_u32 v106, v107, 2, 1
v_mad_u32_u24 v105, v105, 2, v104
v_xor_b32_e32 v105, v105, v106
v_and_b32_e32 v106, 0xf0, v107
v_xor_b32_e32 v105, v105, v106
s_mul_i32 s2, 4, s3
s_sub_u32 s2, 6, s2
v_bfe_u32 v107, v0, s2, 1
v_mul_u32_u24_e32 v107, 0x1040, v107
v_xor_b32_e32 v91, 0x314, v105
v_xor_b32_e32 v92, 0x31c, v105
v_xor_b32_e32 v93, 8, v105
v_mov_b32_e32 v90, v105
v_mad_u32_u24 v90, 4, v90, v107
v_mad_u32_u24 v91, 4, v91, v107
v_mad_u32_u24 v92, 4, v92, v107
v_mad_u32_u24 v93, 4, v93, v107
s_branch 44
s_bfe_u32 s3, s14, 0x10014
v_lshrrev_b32_e32 v107, 1, v0
s_mul_i32 s2, 60, s3
s_sub_u32 s2, 63, s2
v_bfi_b32 v107, s2, v0, v107
v_and_b32_e32 v104, 1, v107
v_bfe_u32 v105, v107, 1, 1
v_bfe_u32 v106, v107, 3, 1
v_xor_b32_e32 v104, v104, v105
v_mad_u32_u24 v105, v105, 2, v106
v_mul_u32_u24_e32 v104, 0x109, v104
v_bfe_u32 v106, v107, 2, 1
v_mad_u32_u24 v105, v105, 2, v104
v_xor_b32_e32 v105, v105, v106
v_and_b32_e32 v106, 0xf0, v107
v_or_b32_e32 v105, v105, v106
s_mul_i32 s2, 4, s3
s_sub_u32 s2, 6, s2
v_bfe_u32 v107, v0, s2, 1
v_mul_u32_u24_e32 v107, 0x1040, v107
v_mad_u32_u24 v90, 4, v105, v107
v_xor_b32_e32 v91, 0x307, v105
v_mad_u32_u24 v91, 4, v91, v107
v_xor_b32_e32 v92, 0x30f, v105
v_mad_u32_u24 v92, 4, v92, v107
v_xor_b32_e32 v93, 8, v105
v_mad_u32_u24 v93, 4, v93, v107
v_subrev_co_u32_e32 v95, vcc, s58, v95
v_mov_b32_e32 v105, s57
v_cmp_lt_i32_e32 vcc, v95, v105
v_subb_co_u32_e64 v104, vcc, 0, 0, vcc
v_mad_i32_i24 v95, v104, s57, v95
v_mad_i32_i24 v97, v104, s62, v97
v_mad_i32_i24 v96, v104, s61, v96
v_mov_b32_e32 v105, s56
v_cmp_lt_i32_e32 vcc, v96, v105
v_subb_co_u32_e64 v104, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v104
v_mad_i32_i24 v96, v104, v105, v96
v_subrev_co_u32_e32 v96, vcc, s59, v96
v_cmp_lt_i32_e32 vcc, v96, v105
v_subb_co_u32_e64 v104, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v104
v_mad_i32_i24 v96, v104, s56, v96
v_subrev_co_u32_e32 v97, vcc, s60, v97
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
s_mov_b32 s76, 17
s_mov_b32 s64, 0
s_bitset1_b32 s14, 26
s_call_b64 s[36:37], 1517
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 1
s_branch 787
s_nop 0
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v72, v74, v72 div:2
v_subrev_f32_e64 v75, v73, v75 div:2
v_add_f32_e64 v73, v74, v73 div:2
v_mad_f32 v74, v74, 1.0, -v73
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v64, v1, s[40:43], 0 idxen
buffer_load_dword v66, v3, s[40:43], 0 idxen
buffer_load_dword v65, v2, s[40:43], 0 idxen
buffer_load_dword v67, v36, s[40:43], 0 idxen
ds_write_b32 v90, v68
ds_write_b32 v91, v69
ds_read_b128 v[38:41], v94 offset:29440
ds_read_b128 v[46:49], v89 offset:28928
ds_read_b128 v[50:53], v89 offset:29056
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 1446
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_mac_f32_e32 v33, v43, v61
v_mac_f32_dpp v72, v72, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v73, v73, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v74, v74, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v75, v75, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_setprio 1
ds_write_b32 v92, v74 offset:8256
ds_write_b32 v93, v75 offset:8256
ds_read_b128 v[42:45], v94 offset:33536
ds_read_b128 v[54:57], v89 offset:33024
ds_read_b128 v[58:61], v89 offset:33152
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 1
s_call_b64 s[36:37], 1384
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v76, v78, v76 div:2
v_subrev_f32_e64 v79, v77, v79 div:2
v_add_f32_e64 v77, v78, v77 div:2
v_mad_f32 v78, v78, 1.0, -v77
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v68, v1, s[40:43], 0 idxen
buffer_load_dword v70, v3, s[40:43], 0 idxen
buffer_load_dword v69, v2, s[40:43], 0 idxen
buffer_load_dword v71, v36, s[40:43], 0 idxen
ds_write_b32 v90, v72 offset:8256
ds_write_b32 v91, v73 offset:8256
ds_read_b128 v[38:41], v94 offset:37696
ds_read_b128 v[46:49], v89 offset:37184
ds_read_b128 v[50:53], v89 offset:37312
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 1318
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_mac_f32_dpp v76, v76, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v77, v77, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v78, v78, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v79, v79, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v33, v43, v61
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_barrier
s_setprio 1
ds_write_b32 v92, v78 offset:16512
ds_write_b32 v93, v79 offset:16512
ds_read_b128 v[42:45], v94 offset:41792
ds_read_b128 v[54:57], v89 offset:41280
ds_read_b128 v[58:61], v89 offset:41408
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 8
s_call_b64 s[36:37], 1255
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v80, v82, v80 div:2
v_subrev_f32_e64 v83, v81, v83 div:2
v_add_f32_e64 v81, v82, v81 div:2
v_mad_f32 v82, v82, 1.0, -v81
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v72, v1, s[40:43], 0 idxen
buffer_load_dword v74, v3, s[40:43], 0 idxen
buffer_load_dword v73, v2, s[40:43], 0 idxen
buffer_load_dword v75, v36, s[40:43], 0 idxen
ds_write_b32 v90, v76 offset:16512
ds_write_b32 v91, v77 offset:16512
ds_read_b128 v[38:41], v94 offset:45952
ds_read_b128 v[46:49], v89 offset:45440
ds_read_b128 v[50:53], v89 offset:45568
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 1182
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_mac_f32_e32 v33, v43, v61
v_mac_f32_dpp v80, v80, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v81, v81, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v82, v82, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v83, v83, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_setprio 1
ds_write_b32 v92, v82 offset:24768
ds_write_b32 v93, v83 offset:24768
ds_read_b128 v[42:45], v94 offset:512
ds_read_b128 v[54:57], v89
ds_read_b128 v[58:61], v89 offset:128
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 1
s_call_b64 s[36:37], 1120
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v84, v86, v84 div:2
v_subrev_f32_e64 v87, v85, v87 div:2
v_add_f32_e64 v85, v86, v85 div:2
v_mad_f32 v86, v86, 1.0, -v85
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v76, v1, s[40:43], 0 idxen
buffer_load_dword v78, v3, s[40:43], 0 idxen
buffer_load_dword v77, v2, s[40:43], 0 idxen
buffer_load_dword v79, v36, s[40:43], 0 idxen
ds_write_b32 v90, v80 offset:24768
ds_write_b32 v91, v81 offset:24768
ds_read_b128 v[38:41], v94 offset:4672
ds_read_b128 v[46:49], v89 offset:4160
ds_read_b128 v[50:53], v89 offset:4288
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 1054
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_mac_f32_dpp v84, v84, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v85, v85, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v86, v86, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v87, v87, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v33, v43, v61
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_barrier
s_setprio 1
ds_write_b32 v92, v86 offset:33024
ds_write_b32 v93, v87 offset:33024
ds_read_b128 v[42:45], v94 offset:8768
ds_read_b128 v[54:57], v89 offset:8256
ds_read_b128 v[58:61], v89 offset:8384
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 8
s_call_b64 s[36:37], 991
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v64, v66, v64 div:2
v_subrev_f32_e64 v67, v65, v67 div:2
v_add_f32_e64 v65, v66, v65 div:2
v_mad_f32 v66, v66, 1.0, -v65
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v80, v1, s[40:43], 0 idxen
buffer_load_dword v82, v3, s[40:43], 0 idxen
buffer_load_dword v81, v2, s[40:43], 0 idxen
buffer_load_dword v83, v36, s[40:43], 0 idxen
ds_write_b32 v90, v84 offset:33024
ds_write_b32 v91, v85 offset:33024
ds_read_b128 v[38:41], v94 offset:12928
ds_read_b128 v[46:49], v89 offset:12416
ds_read_b128 v[50:53], v89 offset:12544
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 918
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_mac_f32_e32 v33, v43, v61
v_mac_f32_dpp v64, v64, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v65, v65, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v66, v66, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v67, v67, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_setprio 1
ds_write_b32 v92, v66 offset:41280
ds_write_b32 v93, v67 offset:41280
ds_read_b128 v[42:45], v94 offset:17024
ds_read_b128 v[54:57], v89 offset:16512
ds_read_b128 v[58:61], v89 offset:16640
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 1
s_call_b64 s[36:37], 856
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_subrev_f32_e64 v68, v70, v68 div:2
v_subrev_f32_e64 v71, v69, v71 div:2
v_add_f32_e64 v69, v70, v69 div:2
v_mad_f32 v70, v70, 1.0, -v69
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v84, v1, s[40:43], 0 idxen
buffer_load_dword v86, v3, s[40:43], 0 idxen
buffer_load_dword v85, v2, s[40:43], 0 idxen
buffer_load_dword v87, v36, s[40:43], 0 idxen
ds_write_b32 v90, v64 offset:41280
ds_write_b32 v91, v65 offset:41280
ds_read_b128 v[38:41], v94 offset:21184
ds_read_b128 v[46:49], v89 offset:20672
ds_read_b128 v[50:53], v89 offset:20800
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 7
s_call_b64 s[36:37], 790
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_mac_f32_dpp v68, v68, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v69, v69, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v70, v70, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v71, v71, v99 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v33, v43, v61
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_barrier
s_setprio 1
ds_write_b32 v92, v70
ds_write_b32 v93, v71
ds_read_b128 v[42:45], v94 offset:25280
ds_read_b128 v[54:57], v89 offset:24768
ds_read_b128 v[58:61], v89 offset:24896
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 64752
s_call_b64 s[36:37], 727
s_branch 64750
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
s_setprio 1
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_mac_f32_dpp v72, v72, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v75, v75, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v64, v1, s[40:43], 0 idxen
buffer_load_dword v67, v36, s[40:43], 0 idxen
ds_write_b32 v90, v68
ds_write_b32 v91, v69
ds_read_b128 v[38:41], v94 offset:29440
ds_read_b128 v[46:49], v89 offset:28928
ds_read_b128 v[50:53], v89 offset:29056
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 6
s_call_b64 s[36:37], 661
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
s_setprio 1
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_mac_f32_e32 v33, v43, v61
v_add_f32_e64 v73, v72, v75 div:2
v_add_f32_e64 v74, v72, -v75 div:2
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v92, v74 offset:8256
ds_write_b32 v93, v75 offset:8256
ds_read_b128 v[42:45], v94 offset:33536
ds_read_b128 v[54:57], v89 offset:33024
ds_read_b128 v[58:61], v89 offset:33152
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 4
s_call_b64 s[36:37], 603
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
s_setprio 1
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_mac_f32_dpp v76, v76, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v79, v79, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v68, v1, s[40:43], 0 idxen
buffer_load_dword v71, v36, s[40:43], 0 idxen
ds_write_b32 v90, v72 offset:8256
ds_write_b32 v91, v73 offset:8256
ds_read_b128 v[38:41], v94 offset:37696
ds_read_b128 v[46:49], v89 offset:37184
ds_read_b128 v[50:53], v89 offset:37312
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 6
s_call_b64 s[36:37], 541
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_barrier
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
s_setprio 1
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_add_f32_e64 v77, v76, v79 div:2
v_add_f32_e64 v78, v76, -v79 div:2
v_mac_f32_e32 v33, v43, v61
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v92, v78 offset:16512
ds_write_b32 v93, v79 offset:16512
ds_read_b128 v[42:45], v94 offset:41792
ds_read_b128 v[54:57], v89 offset:41280
ds_read_b128 v[58:61], v89 offset:41408
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 3
s_call_b64 s[36:37], 482
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
s_setprio 1
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_mac_f32_dpp v80, v80, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v83, v83, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v72, v1, s[40:43], 0 idxen
buffer_load_dword v75, v36, s[40:43], 0 idxen
ds_write_b32 v90, v76 offset:16512
ds_write_b32 v91, v77 offset:16512
ds_read_b128 v[38:41], v94 offset:45952
ds_read_b128 v[46:49], v89 offset:45440
ds_read_b128 v[50:53], v89 offset:45568
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 6
s_call_b64 s[36:37], 421
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
s_setprio 1
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_mac_f32_e32 v33, v43, v61
v_add_f32_e64 v81, v80, v83 div:2
v_add_f32_e64 v82, v80, -v83 div:2
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v92, v82 offset:24768
ds_write_b32 v93, v83 offset:24768
ds_read_b128 v[42:45], v94 offset:512
ds_read_b128 v[54:57], v89
ds_read_b128 v[58:61], v89 offset:128
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 4
s_call_b64 s[36:37], 363
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
s_setprio 1
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_mac_f32_dpp v84, v84, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v87, v87, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v76, v1, s[40:43], 0 idxen
buffer_load_dword v79, v36, s[40:43], 0 idxen
ds_write_b32 v90, v80 offset:24768
ds_write_b32 v91, v81 offset:24768
ds_read_b128 v[38:41], v94 offset:4672
ds_read_b128 v[46:49], v89 offset:4160
ds_read_b128 v[50:53], v89 offset:4288
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 6
s_call_b64 s[36:37], 301
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_barrier
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
s_setprio 1
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_add_f32_e64 v85, v84, v87 div:2
v_add_f32_e64 v86, v84, -v87 div:2
v_mac_f32_e32 v33, v43, v61
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v92, v86 offset:33024
ds_write_b32 v93, v87 offset:33024
ds_read_b128 v[42:45], v94 offset:8768
ds_read_b128 v[54:57], v89 offset:8256
ds_read_b128 v[58:61], v89 offset:8384
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 3
s_call_b64 s[36:37], 242
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
s_setprio 1
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_mac_f32_dpp v64, v64, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v67, v67, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v80, v1, s[40:43], 0 idxen
buffer_load_dword v83, v36, s[40:43], 0 idxen
ds_write_b32 v90, v84 offset:33024
ds_write_b32 v91, v85 offset:33024
ds_read_b128 v[38:41], v94 offset:12928
ds_read_b128 v[46:49], v89 offset:12416
ds_read_b128 v[50:53], v89 offset:12544
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 6
s_call_b64 s[36:37], 181
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
s_setprio 1
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_mac_f32_e32 v33, v43, v61
v_add_f32_e64 v65, v64, v67 div:2
v_add_f32_e64 v66, v64, -v67 div:2
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v92, v66 offset:41280
ds_write_b32 v93, v67 offset:41280
ds_read_b128 v[42:45], v94 offset:17024
ds_read_b128 v[54:57], v89 offset:16512
ds_read_b128 v[58:61], v89 offset:16640
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 4
s_call_b64 s[36:37], 123
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v4, v38, v46
v_mac_f32_e32 v5, v39, v46
v_mac_f32_e32 v6, v40, v46
s_setprio 1
v_mac_f32_e32 v7, v41, v46
v_mac_f32_e32 v8, v38, v47
v_mac_f32_e32 v9, v39, v47
v_mac_f32_e32 v10, v40, v47
v_mac_f32_e32 v11, v41, v47
v_mac_f32_e32 v12, v38, v48
v_mac_f32_e32 v13, v39, v48
v_mac_f32_e32 v14, v40, v48
v_mac_f32_e32 v15, v41, v48
v_mac_f32_e32 v16, v38, v49
v_mac_f32_e32 v17, v39, v49
v_mac_f32_e32 v18, v40, v49
v_mac_f32_e32 v19, v41, v49
v_mac_f32_e32 v20, v38, v50
v_mac_f32_e32 v21, v39, v50
v_mac_f32_e32 v22, v40, v50
v_mac_f32_e32 v23, v41, v50
v_mac_f32_e32 v24, v38, v51
v_mac_f32_e32 v25, v39, v51
v_mac_f32_e32 v26, v40, v51
v_mac_f32_e32 v27, v41, v51
v_mac_f32_e32 v28, v38, v52
v_mac_f32_e32 v29, v39, v52
v_mac_f32_e32 v30, v40, v52
v_mac_f32_e32 v31, v41, v52
v_mac_f32_e32 v32, v38, v53
v_mac_f32_e32 v33, v39, v53
v_mac_f32_dpp v68, v68, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v71, v71, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
buffer_load_dword v84, v1, s[40:43], 0 idxen
buffer_load_dword v87, v36, s[40:43], 0 idxen
ds_write_b32 v90, v64 offset:41280
ds_write_b32 v91, v65 offset:41280
ds_read_b128 v[38:41], v94 offset:21184
ds_read_b128 v[46:49], v89 offset:20672
ds_read_b128 v[50:53], v89 offset:20800
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 6
s_call_b64 s[36:37], 61
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_barrier
v_mac_f32_e32 v4, v42, v54
v_mac_f32_e32 v5, v43, v54
v_mac_f32_e32 v6, v44, v54
s_setprio 1
v_mac_f32_e32 v7, v45, v54
v_mac_f32_e32 v8, v42, v55
v_mac_f32_e32 v9, v43, v55
v_mac_f32_e32 v10, v44, v55
v_mac_f32_e32 v11, v45, v55
v_mac_f32_e32 v12, v42, v56
v_mac_f32_e32 v13, v43, v56
v_mac_f32_e32 v14, v44, v56
v_mac_f32_e32 v15, v45, v56
v_mac_f32_e32 v16, v42, v57
v_mac_f32_e32 v17, v43, v57
v_mac_f32_e32 v18, v44, v57
v_mac_f32_e32 v19, v45, v57
v_mac_f32_e32 v20, v42, v58
v_mac_f32_e32 v21, v43, v58
v_mac_f32_e32 v22, v44, v58
v_mac_f32_e32 v23, v45, v58
v_mac_f32_e32 v24, v42, v59
v_mac_f32_e32 v25, v43, v59
v_mac_f32_e32 v26, v44, v59
v_mac_f32_e32 v27, v45, v59
v_mac_f32_e32 v28, v42, v60
v_mac_f32_e32 v29, v43, v60
v_mac_f32_e32 v30, v44, v60
v_mac_f32_e32 v31, v45, v60
v_mac_f32_e32 v32, v42, v61
v_add_f32_e64 v69, v68, v71 div:2
v_add_f32_e64 v70, v68, -v71 div:2
v_mac_f32_e32 v33, v43, v61
v_mac_f32_e32 v34, v44, v61
v_mac_f32_e32 v35, v45, v61
s_setprio 0
ds_write_b32 v92, v70
ds_write_b32 v93, v71
ds_read_b128 v[42:45], v94 offset:25280
ds_read_b128 v[54:57], v89 offset:24768
ds_read_b128 v[58:61], v89 offset:24896
s_waitcnt vmcnt(6) lgkmcnt(5)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -1
s_cbranch_scc1 64819
s_call_b64 s[36:37], 2
s_branch 64817
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
s_cbranch_scc1 602
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
s_add_u32 s69, s12, 31
s_lshr_b32 s69, s69, 5
v_mov_b32_e32 v105, s70
v_mul_u32_u24_e32 v105, s69, v105
v_add_co_u32_e32 v105, vcc, s13, v105
v_sub_co_u32_e64 v105, vcc, v105, 1
v_ffbh_u32_e32 v108, s13
v_lshlrev_b32_e64 v109, v108, s13
v_and_b32_e32 v110, 0xffffff00, v109
v_cmp_eq_u32_e32 vcc, 0x80000000, v109
v_cvt_f32_u32_e32 v110, v110
v_rcp_f32_e32 v104, v110
v_subb_co_u32_e32 v107, vcc, 32, v108, vcc
v_cvt_f32_ubyte0_e32 v108, v109
v_fma_f32 v110, v110, v104, -1.0
v_fma_f32 v110, v108, v104, v110
v_madak_f32 v110, v110, v104, 0x9f000000
v_mul_f32_e32 v110, 0x5f800000, v110
v_mov_b32_e32 v108, 0
v_cvt_flr_i32_f32_e64 v110, -v110
v_lshl_add_u32 v104, v104, 9, v110
v_mad_u64_u32 v[108:109], vcc, v109, v104, v[108:109]
v_subb_co_u32_e64 v104, vcc, v104, -1, vcc
v_mul_hi_u32 v108, v105, v104
v_add_co_u32_e32 v104, vcc, v108, v105
v_addc_co_u32_e64 v108, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v107
v_cndmask_b32_e32 v104, v104, v108, vcc
v_alignbit_b32 v104, v108, v104, v107
s_nop 0
v_readfirstlane_b32 s68, v104
v_mul_u32_u24_e64 v104, v104, s4
v_ffbh_u32_e32 v108, s69
v_lshlrev_b32_e64 v109, v108, s69
v_and_b32_e32 v110, 0xffffff00, v109
v_cmp_eq_u32_e32 vcc, 0x80000000, v109
v_cvt_f32_u32_e32 v110, v110
v_rcp_f32_e32 v105, v110
v_subb_co_u32_e32 v107, vcc, 32, v108, vcc
v_cvt_f32_ubyte0_e32 v108, v109
v_fma_f32 v110, v110, v105, -1.0
v_fma_f32 v110, v108, v105, v110
v_madak_f32 v110, v110, v105, 0x9f000000
v_mul_f32_e32 v110, 0x5f800000, v110
v_mov_b32_e32 v108, 0
v_cvt_flr_i32_f32_e64 v110, -v110
v_lshl_add_u32 v105, v105, 9, v110
v_mad_u64_u32 v[108:109], vcc, v109, v105, v[108:109]
v_subb_co_u32_e64 v105, vcc, v105, -1, vcc
v_mul_hi_u32 v108, v104, v105
v_add_co_u32_e32 v105, vcc, v108, v104
v_addc_co_u32_e64 v108, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v107
v_cndmask_b32_e32 v105, v105, v108, vcc
v_alignbit_b32 v105, v108, v105, v107
v_readfirstlane_b32 s2, v104
v_readfirstlane_b32 s66, v105
s_mul_i32 s66, s66, s69
s_sub_u32 s66, s2, s66
v_sub_co_u32_e32 v105, vcc, s4, v105
v_sub_co_u32_e32 v105, vcc, s13, v105
v_and_b32_e64 v107, v0, 63
v_cmp_eq_u32_e64 vcc, v107, 0
v_cndmask_b32_e32 v105, 1, v105, vcc
s_sub_u32 s3, 0, s57
s_sub_u32 s15, 0, s56
v_mul_u32_u24_e64 v109, v105, 32
v_ffbh_u32_e32 v112, s3
v_lshlrev_b32_e64 v113, v112, s3
v_and_b32_e32 v111, 0xffffff00, v113
v_cmp_eq_u32_e32 vcc, 0x80000000, v113
v_cvt_f32_u32_e32 v111, v111
v_rcp_f32_e32 v107, v111
v_subb_co_u32_e32 v110, vcc, 32, v112, vcc
v_cvt_f32_ubyte0_e32 v112, v113
v_fma_f32 v111, v111, v107, -1.0
v_fma_f32 v111, v112, v107, v111
v_madak_f32 v111, v111, v107, 0x9f000000
v_mul_f32_e32 v111, 0x5f800000, v111
v_mov_b32_e32 v112, 0
v_cvt_flr_i32_f32_e64 v111, -v111
v_lshl_add_u32 v107, v107, 9, v111
v_mad_u64_u32 v[112:113], vcc, v113, v107, v[112:113]
v_subb_co_u32_e64 v107, vcc, v107, -1, vcc
v_mul_hi_u32 v111, v109, v107
v_add_co_u32_e32 v107, vcc, v111, v109
v_addc_co_u32_e64 v111, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v110
v_cndmask_b32_e32 v107, v107, v111, vcc
v_alignbit_b32 v107, v111, v107, v110
v_mad_i32_i24 v108, v107, s57, v109
v_mul_u32_u24_e64 v109, v107, 1
v_ffbh_u32_e32 v112, s15
v_lshlrev_b32_e64 v113, v112, s15
v_and_b32_e32 v111, 0xffffff00, v113
v_cmp_eq_u32_e32 vcc, 0x80000000, v113
v_cvt_f32_u32_e32 v111, v111
v_rcp_f32_e32 v107, v111
v_subb_co_u32_e32 v110, vcc, 32, v112, vcc
v_cvt_f32_ubyte0_e32 v112, v113
v_fma_f32 v111, v111, v107, -1.0
v_fma_f32 v111, v112, v107, v111
v_madak_f32 v111, v111, v107, 0x9f000000
v_mul_f32_e32 v111, 0x5f800000, v111
v_mov_b32_e32 v112, 0
v_cvt_flr_i32_f32_e64 v111, -v111
v_lshl_add_u32 v107, v107, 9, v111
v_mad_u64_u32 v[112:113], vcc, v113, v107, v[112:113]
v_subb_co_u32_e64 v107, vcc, v107, -1, vcc
v_mul_hi_u32 v111, v109, v107
v_add_co_u32_e32 v107, vcc, v111, v109
v_addc_co_u32_e64 v111, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v110
v_cndmask_b32_e32 v107, v107, v111, vcc
v_alignbit_b32 v107, v111, v107, v110
v_mad_i32_i24 v109, v107, s56, v109
v_readfirstlane_b32 s58, v108
v_readfirstlane_b32 s59, v109
v_readfirstlane_b32 s60, v107
v_add_co_u32_e32 v95, vcc, s58, v95
v_addc_co_u32_e64 v110, vcc, 0, 0, vcc
v_mad_i32_i24 v95, v110, s57, v95
v_mad_i32_i24 v97, v110, s62, v97
v_mad_i32_i24 v96, v110, s61, v96
v_cmp_ge_i32_e64 vcc, v96, 0
v_addc_co_u32_e64 v110, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v110
v_mad_i32_i24 v96, v110, s56, v96
v_add_co_u32_e32 v96, vcc, s59, v96
v_addc_co_u32_e64 v110, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v110
v_mad_i32_i24 v96, v110, s56, v96
v_add_co_u32_e32 v97, vcc, s60, v97
v_readlane_b32 s58, v108, 1
v_readlane_b32 s59, v109, 1
v_readlane_b32 s60, v107, 1
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
v_add_co_u32_e32 v95, vcc, s58, v95
v_addc_co_u32_e64 v104, vcc, 0, 0, vcc
v_mad_i32_i24 v95, v104, s57, v95
v_mad_i32_i24 v97, v104, s62, v97
v_mad_i32_i24 v96, v104, s61, v96
v_cmp_ge_i32_e64 vcc, v96, 0
v_addc_co_u32_e64 v104, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v104
v_mad_i32_i24 v96, v104, s56, v96
v_add_co_u32_e32 v96, vcc, s59, v96
v_addc_co_u32_e64 v104, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v104
v_mad_i32_i24 v96, v104, s56, v96
v_add_co_u32_e32 v97, vcc, s60, v97
s_mov_b32 s65, s66
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccz 168
v_subrev_co_u32_e32 v104, vcc, s57, v95
v_subrev_co_u32_e32 v105, vcc, s56, v96
s_bitcmp1_b32 s14, 22
s_cbranch_scc0 64
s_bitset0_b32 s14, 22
s_bfe_u32 s2, s14, 0x10014
v_mul_u32_u24_e32 v107, 3, v104
v_mul_u32_u24_e32 v108, 3, v105
v_cvt_pk_u16_u32 v110, v107, v108
v_and_b32_e64 v107, v0, 1
v_cmp_eq_u32_e64 vcc, v107, 1
v_cndmask_b32_e32 v110, v97, v110, vcc
v_lshrrev_b32_e32 v106, 1, v0
v_bfe_u32 v111, v106, s2, 1
v_lshrrev_b32_e32 v106, 1, v0
v_bfi_b32 v106, 1, v0, v106
v_lshrrev_b32_e32 v107, 2, v0
v_bfi_b32 v107, 1, v0, v107
v_cmp_eq_u32_e64 vcc, s2, 0
v_cndmask_b32_e32 v106, v107, v106, vcc
s_sub_u32 s2, 1, s2
v_lshrrev_b32_e32 v107, s2, v106
v_bfi_b32 v106, 32, v107, v106
v_and_b32_e32 v106, 63, v106
v_add_co_u32_e32 v107, vcc, 16, v106
v_and_b32_e64 v108, v0, 2
v_cmp_eq_u32_e64 vcc, v108, 0
v_cndmask_b32_e32 v107, v107, v106, vcc
v_lshlrev_b32_e32 v108, 14, v111
v_mad_u32_u24 v107, 4, v107, v108
v_add_co_u32_e32 v106, vcc, s78, v107
ds_write_b32 v106, v110
v_writelane_b32 v108, s14, 0
v_writelane_b32 v108, s67, 1
v_writelane_b32 v108, s66, 2
v_and_b32_e64 v106, v0, 63
v_cmp_ge_u32_e64 vcc, v106, 3
v_mov_b32_e32 v109, 0x4000
v_cndmask_b32_e32 v106, v106, v109, vcc
v_mad_i32_i24 v106, v106, 4, s78
ds_write_b32 v106, v108 offset:256
s_add_u32 s78, s78, 0x18c
s_cmp_eq_u32 s78, 0x10000
s_cselect_b32 s78, 0xc220, s78
v_mov_b32_dpp v106, v97 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v104, v104 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v105, v105 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_readfirstlane_b32 s63, v106
v_sub_co_u32_e64 v107, vcc, v106, s63
v_mul_lo_u32 v107, v107, s44
v_and_b32_e64 v111, v0, 3
v_ashrrev_i32_e64 v112, 0, s27
v_subrev_co_u32_e32 v111, vcc, v112, v111
v_ashrrev_i32_e64 v112, 0, s35
v_mad_i32_i24 v108, v112, 2, v111
s_bfe_u32 s2, s14, 0x10014
v_lshrrev_b32_e32 v110, 2, v0
v_and_b32_e32 v110, s2, v110
v_mad_i32_i24 v108, v110, 2, v108
v_add_co_u32_e64 v109, vcc, 0, s38
v_ashrrev_i32_e32 v109, 0, v109
v_add_co_u32_e64 v110, vcc, 0, s26
v_ashrrev_i32_e32 v110, 0, v110
v_sub_i32 v109, v109, v110
v_cmp_ge_u32_e64 s[2:3], v106, s8
v_mad_i32_i24 v104, v104, 3, v108
v_cmp_ge_u32_e64 s[22:23], v104, s11
v_add_co_u32_e32 v104, vcc, v104, v107
s_or_b64 s[22:23], s[22:23], s[2:3]
v_mad_i32_i24 v105, v105, 3, v109
v_cmp_ge_u32_e64 s[96:97], v105, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v1, v105, s49, v104
v_cndmask_b32_e64 v1, v1, -1, s[96:97]
v_add_co_u32_e32 v105, vcc, 1, v105
v_cmp_ge_u32_e64 s[96:97], v105, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v2, v105, s49, v104
v_cndmask_b32_e64 v2, v2, -1, s[96:97]
v_add_co_u32_e32 v105, vcc, 1, v105
v_cmp_ge_u32_e64 s[96:97], v105, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v3, v105, s49, v104
v_cndmask_b32_e64 v3, v3, -1, s[96:97]
v_add_co_u32_e32 v105, vcc, 1, v105
v_cmp_ge_u32_e64 s[96:97], v105, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v36, v105, s49, v104
v_cndmask_b32_e64 v36, v36, -1, s[96:97]
s_bitcmp1_b32 s14, 18
s_cbranch_scc1 144
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
s_branch 97
s_bitcmp1_b32 s14, 18
s_cbranch_scc1 119
s_bfe_u32 s2, s14, 0x10014
v_xor_b32_dpp v104, v0, v0 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_bfe_u32 v106, v0, 2, s2
v_mad_u32_u24 v104, v106, 2, v104
v_mad_u32_u24 v104, s35, 2, v104
v_sub_co_u32_e32 v106, vcc, s25, v104
v_sub_co_u32_e64 v106, vcc, v106, 1
s_bfe_u32 s2, s14, 0x10001
v_cmp_eq_u32_e64 vcc, s2, 1
v_cndmask_b32_e32 v104, v104, v106, vcc
v_cmp_ge_u32_e64 s[2:3], v104, s25
s_bfe_u32 s15, s14, 0x10018
v_bfe_u32 v107, v0, 2, s15
v_mul_lo_u32 v107, s48, v107
v_add_co_u32_e32 v104, vcc, v104, v107
v_mul_lo_u32 v105, s72, v98
v_add_co_u32_e32 v105, vcc, v105, v104
s_sub_u32 s15, s24, s38
s_sub_u32 s15, s15, 2
s_bitcmp1_b32 s14, 0
s_cselect_b32 s15, s15, s38
v_mov_b32_e32 v107, s15
v_cmp_ge_u32_e64 s[22:23], v107, s24
v_mad_i32_i24 v1, v107, s73, v105
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v1, v1, -1, s[22:23]
v_mov_b32_e32 v2, v1
v_add_co_u32_e64 v107, vcc, v107, 1
v_cmp_ge_u32_e64 s[22:23], v107, s24
v_mad_i32_i24 v36, v107, s73, v105
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v36, v36, -1, s[22:23]
v_add_co_u32_e64 v107, vcc, v107, 1
v_cmp_ge_u32_e64 s[22:23], v107, s24
v_mad_i32_i24 v3, v107, s73, v105
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v3, v3, -1, s[22:23]
s_bitcmp1_b32 s14, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v1, v2, v36, vcc
v_cndmask_b32_e32 v36, v36, v2, vcc
v_add_co_u32_e64 v104, vcc, v98, s65
v_cmp_lt_u32_e64 vcc, v104, s12
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
s_cbranch_scc1 5
s_branch 64944
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_and_b32 s92, 0x900000, s14
s_subb_u32 s35, s35, 1
s_cbranch_scc0 65235
s_and_b32 s92, 0x900000, s14
s_subb_u32 s35, s5, 1
s_add_u32 s38, s38, 2
s_cmp_ge_u32 s38, s24
s_cbranch_scc0 65229
s_mov_b32 s38, 0
s_branch 65196
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s2, 0x3c3c3c3c
s_mov_b32 s3, s2
v_mov_b32_e32 v104, v4
v_mov_b32_e32 v105, v5
v_mov_b32_e32 v106, v6
v_mov_b32_e32 v107, v7
v_add_f32_dpp v104, v4, v4 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v105, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v6, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v7, v7 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v6, v6, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v7, v7, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v4, v4, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v5, v5, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v5, v6 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v4, v7 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v105, v106, v105 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v104, v107, v104 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v107, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v5, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v106, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v4, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v7, v105
v_add_f32_dpp v7, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v107 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v106, v107 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v6, v104
v_add_f32_dpp v6, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v106 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v107, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v5, v5, v4 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v4, v7, v6 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v6, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v6, v107 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v106, v106 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v5, v106, v5, s[2:3]
v_mov_b32_dpp v6, v6 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v6, v6 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v104, v8
v_mov_b32_e32 v105, v9
v_mov_b32_e32 v106, v10
v_mov_b32_e32 v107, v11
v_add_f32_dpp v104, v8, v8 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v105, v9, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v10, v10 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v11, v11 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v10, v10, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v11, v11, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v8, v8, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v9, v9, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v9, v10 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v8, v11 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v105, v106, v105 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v104, v107, v104 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v107, v9, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v9, v9, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v106, v8, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v8, v8, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v11, v105
v_add_f32_dpp v11, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v107 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v106, v107 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v10, v104
v_add_f32_dpp v10, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v106 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v107, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v9, v9, v8 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v7, v11, v10 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v10, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v10, v107 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v106, v106 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v8, v106, v9, s[2:3]
v_mov_b32_dpp v9, v10 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v9, v10 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v104, v12
v_mov_b32_e32 v105, v13
v_mov_b32_e32 v106, v14
v_mov_b32_e32 v107, v15
v_add_f32_dpp v104, v12, v12 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v105, v13, v13 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v14, v14 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v15, v15 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v14, v14, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v15, v15, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v12, v12, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v13, v13, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v13, v14 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v12, v15 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v105, v106, v105 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v104, v107, v104 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v107, v13, v13 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v13, v13, v13 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v106, v12, v12 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v12, v12, v12 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v15, v105
v_add_f32_dpp v15, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v107 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v106, v107 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v14, v104
v_add_f32_dpp v14, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v106 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v107, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v13, v13, v12 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v10, v15, v14 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v14, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v14, v107 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v106, v106 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v11, v106, v13, s[2:3]
v_mov_b32_dpp v12, v14 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v12, v14 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v104, v16
v_mov_b32_e32 v105, v17
v_mov_b32_e32 v106, v18
v_mov_b32_e32 v107, v19
v_add_f32_dpp v104, v16, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v105, v17, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v18, v18 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v19, v19 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v18, v18, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v19, v19, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v16, v16, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v17, v17, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v17, v18 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v16, v19 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v105, v106, v105 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v104, v107, v104 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v107, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v17, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v106, v16, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v16, v16, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v19, v105
v_add_f32_dpp v19, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v107 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v106, v107 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v18, v104
v_add_f32_dpp v18, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v106 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v107, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v17, v17, v16 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v13, v19, v18 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v18, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v18, v107 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v106, v106 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v14, v106, v17, s[2:3]
v_mov_b32_dpp v15, v18 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v15, v18 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v104, v20
v_mov_b32_e32 v105, v21
v_mov_b32_e32 v106, v22
v_mov_b32_e32 v107, v23
v_add_f32_dpp v104, v20, v20 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v105, v21, v21 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v22, v22 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v23, v23 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v22, v22, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v23, v23, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v20, v20, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v21, v21, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v21, v22 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v20, v23 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v105, v106, v105 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v104, v107, v104 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v107, v21, v21 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v21, v21, v21 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v106, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v20, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v23, v105
v_add_f32_dpp v23, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v107 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v106, v107 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v22, v104
v_add_f32_dpp v22, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v106 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v107, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v21, v21, v20 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v16, v23, v22 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v22, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v22, v107 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v106, v106 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v17, v106, v21, s[2:3]
v_mov_b32_dpp v18, v22 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v18, v22 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v104, v24
v_mov_b32_e32 v105, v25
v_mov_b32_e32 v106, v26
v_mov_b32_e32 v107, v27
v_add_f32_dpp v104, v24, v24 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v105, v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v26, v26 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v27, v27 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v26, v26, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v27, v27, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v24, v24, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v25, v25, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v25, v26 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v24, v27 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v105, v106, v105 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v104, v107, v104 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v107, v25, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v25, v25, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v106, v24, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v24, v24, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v27, v105
v_add_f32_dpp v27, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v107 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v106, v107 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v26, v104
v_add_f32_dpp v26, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v106 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v107, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v25, v25, v24 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v19, v27, v26 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v26, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v26, v107 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v106, v106 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v20, v106, v25, s[2:3]
v_mov_b32_dpp v21, v26 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v21, v26 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v104, v28
v_mov_b32_e32 v105, v29
v_mov_b32_e32 v106, v30
v_mov_b32_e32 v107, v31
v_add_f32_dpp v104, v28, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v105, v29, v29 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v30, v30 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v31, v31 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v30, v30, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v31, v31, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v28, v28, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v29, v29, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v29, v30 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v28, v31 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v105, v106, v105 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v104, v107, v104 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v107, v29, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v29, v29, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v106, v28, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v28, v28, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v31, v105
v_add_f32_dpp v31, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v107 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v106, v107 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v30, v104
v_add_f32_dpp v30, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v106 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v107, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v29, v29, v28 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v22, v31, v30 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v30, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v30, v107 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v106, v106 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v23, v106, v29, s[2:3]
v_mov_b32_dpp v24, v30 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v24, v30 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
s_setprio 1
v_mov_b32_e32 v104, v32
v_mov_b32_e32 v105, v33
v_mov_b32_e32 v106, v34
v_mov_b32_e32 v107, v35
v_add_f32_dpp v104, v32, v32 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v105, v33, v33 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v34, v34 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v34, v34, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v35, v35, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v32, v32, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v33, v33, v100 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v33, v34 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v32, v35 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v105, v106, v105 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v104, v107, v104 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v107, v33, v33 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v33, v33, v33 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v106, v32, v32 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v32, v32, v32 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v35, v105
v_add_f32_dpp v35, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v107 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v106, v107 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v34, v104
v_add_f32_dpp v34, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v106, v106 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v107, v105, v105 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v33, v33, v32 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v25, v35, v34 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v34, v104, v104 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v34, v107 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v106, v106 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v26, v106, v33, s[2:3]
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
s_mul_i32 s76, s1, s5
s_mul_i32 s76, s76, s9
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
ds_read_b32 v104, v62 offset:256
ds_read_b32 v37, v37
s_add_u32 s79, s79, 0x18c
s_cmp_eq_u32 s79, 0x10000
s_cselect_b32 s79, 0xc220, s79
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s77, v37
v_readlane_b32 s94, v104, 0
s_bitcmp1_b32 s94, 18
s_cbranch_scc1 1391
v_readlane_b32 s92, v104, 1
v_readlane_b32 s93, v104, 2
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
v_bfe_u32 v104, v37, 16, 16
v_bfe_u32 v105, v37, 0, 16
v_and_b32_e64 v106, v0, 7
v_sub_co_u32_e32 v107, vcc, 7, v106
v_min_u32_e32 v106, v106, v107
v_bfe_u32 v107, v106, 1, 1
v_bfe_u32 v106, v106, 0, 1
v_mov_b32_dpp v104, v104 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v105, v105 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_add_co_u32_e32 v104, vcc, v104, v107
v_add_co_u32_e32 v105, vcc, v105, v106
v_mov_b32_dpp v106, v37 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[2:3], v106, s8
v_sub_co_u32_e64 v106, vcc, v106, s77
v_mul_lo_u32 v106, v106, s45
v_xor_b32_dpp v107, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v107, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v108, v0, v0 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v108, v0, v0 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_add_co_u32_e32 v108, vcc, v105, v108
v_add_co_u32_e32 v107, vcc, v104, v107
v_mad_i32_i24 v37, v107, s50, v108
v_add_co_u32_e32 v37, vcc, v37, v106
v_cmp_ge_u32_e64 s[96:97], v108, s29
s_or_b64 s[96:97], s[96:97], s[2:3]
v_cmp_ge_u32_e64 s[22:23], v107, s28
s_or_b64 s[96:97], s[96:97], s[22:23]
v_cndmask_b32_e64 v37, v37, -1, s[96:97]
v_xor_b32_dpp v107, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v107, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v108, v0, v0 quad_perm:[1,1,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32_e32 v108, vcc, v105, v108
v_add_co_u32_e32 v107, vcc, v104, v107
v_mad_i32_i24 v62, v107, s50, v108
v_add_co_u32_e32 v62, vcc, v62, v106
v_cmp_ge_u32_e64 s[96:97], v108, s29
s_or_b64 s[96:97], s[96:97], s[2:3]
v_cmp_ge_u32_e64 s[22:23], v107, s28
s_or_b64 s[96:97], s[96:97], s[22:23]
v_cndmask_b32_e64 v62, v62, -1, s[96:97]
v_xor_b32_dpp v107, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v107, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v108, v0, v0 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v108, v0, v0 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xa
v_add_co_u32_e32 v108, vcc, v105, v108
v_add_co_u32_e32 v107, vcc, v104, v107
v_mad_i32_i24 v63, v107, s50, v108
v_add_co_u32_e32 v63, vcc, v63, v106
v_cmp_ge_u32_e64 s[96:97], v108, s29
s_or_b64 s[96:97], s[96:97], s[2:3]
v_cmp_ge_u32_e64 s[22:23], v107, s28
s_or_b64 s[96:97], s[96:97], s[22:23]
v_cndmask_b32_e64 v63, v63, -1, s[96:97]
v_and_b32_e64 v103, v0, 63
buffer_load_dword v103, v103, s[88:91], 0 idxen
s_branch 64192
v_readlane_b32 s99, v103, 0
v_add_f32_e64 v4, v4, s99
v_add_f32_e64 v5, v5, s99
v_add_f32_e64 v6, v6, s99
buffer_store_dword v4, v37, s[84:87], 0 idxen
buffer_store_dword v5, v62, s[84:87], 0 idxen
buffer_store_dword v6, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 1
v_add_f32_e64 v7, v7, s99
v_add_f32_e64 v8, v8, s99
v_add_f32_e64 v9, v9, s99
buffer_store_dword v7, v37, s[84:87], 0 idxen
buffer_store_dword v8, v62, s[84:87], 0 idxen
buffer_store_dword v9, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 2
v_add_f32_e64 v10, v10, s99
v_add_f32_e64 v11, v11, s99
v_add_f32_e64 v12, v12, s99
buffer_store_dword v10, v37, s[84:87], 0 idxen
buffer_store_dword v11, v62, s[84:87], 0 idxen
buffer_store_dword v12, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 3
v_add_f32_e64 v13, v13, s99
v_add_f32_e64 v14, v14, s99
v_add_f32_e64 v15, v15, s99
buffer_store_dword v13, v37, s[84:87], 0 idxen
buffer_store_dword v14, v62, s[84:87], 0 idxen
buffer_store_dword v15, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s99, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 4
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 8
v_add_f32_e64 v16, v16, s99
v_add_f32_e64 v17, v17, s99
v_add_f32_e64 v18, v18, s99
buffer_store_dword v16, v37, s[84:87], 0 idxen
buffer_store_dword v17, v62, s[84:87], 0 idxen
buffer_store_dword v18, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 9
v_add_f32_e64 v19, v19, s99
v_add_f32_e64 v20, v20, s99
v_add_f32_e64 v21, v21, s99
buffer_store_dword v19, v37, s[84:87], 0 idxen
buffer_store_dword v20, v62, s[84:87], 0 idxen
buffer_store_dword v21, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 10
v_add_f32_e64 v22, v22, s99
v_add_f32_e64 v23, v23, s99
v_add_f32_e64 v24, v24, s99
buffer_store_dword v22, v37, s[84:87], 0 idxen
buffer_store_dword v23, v62, s[84:87], 0 idxen
buffer_store_dword v24, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 11
v_add_f32_e64 v25, v25, s99
v_add_f32_e64 v26, v26, s99
v_add_f32_e64 v27, v27, s99
buffer_store_dword v25, v37, s[84:87], 0 idxen
buffer_store_dword v26, v62, s[84:87], 0 idxen
buffer_store_dword v27, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
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
s_add_u32 s88, s88, 0x80
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, 32
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
v_readlane_b32 s99, v103, 0
v_add_f32_e64 v4, v4, s99
v_mul_f32_e64 v105, v4, s32
v_cmp_lt_f32_e64 vcc, v4, 0
v_cndmask_b32_e32 v4, v4, v105, vcc
v_add_f32_e64 v5, v5, s99
v_mul_f32_e64 v105, v5, s32
v_cmp_lt_f32_e64 vcc, v5, 0
v_cndmask_b32_e32 v5, v5, v105, vcc
v_add_f32_e64 v6, v6, s99
v_mul_f32_e64 v105, v6, s32
v_cmp_lt_f32_e64 vcc, v6, 0
v_cndmask_b32_e32 v6, v6, v105, vcc
buffer_store_dword v4, v37, s[84:87], 0 idxen
buffer_store_dword v5, v62, s[84:87], 0 idxen
buffer_store_dword v6, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 1
v_add_f32_e64 v7, v7, s99
v_mul_f32_e64 v105, v7, s32
v_cmp_lt_f32_e64 vcc, v7, 0
v_cndmask_b32_e32 v7, v7, v105, vcc
v_add_f32_e64 v8, v8, s99
v_mul_f32_e64 v105, v8, s32
v_cmp_lt_f32_e64 vcc, v8, 0
v_cndmask_b32_e32 v8, v8, v105, vcc
v_add_f32_e64 v9, v9, s99
v_mul_f32_e64 v105, v9, s32
v_cmp_lt_f32_e64 vcc, v9, 0
v_cndmask_b32_e32 v9, v9, v105, vcc
buffer_store_dword v7, v37, s[84:87], 0 idxen
buffer_store_dword v8, v62, s[84:87], 0 idxen
buffer_store_dword v9, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 2
v_add_f32_e64 v10, v10, s99
v_mul_f32_e64 v105, v10, s32
v_cmp_lt_f32_e64 vcc, v10, 0
v_cndmask_b32_e32 v10, v10, v105, vcc
v_add_f32_e64 v11, v11, s99
v_mul_f32_e64 v105, v11, s32
v_cmp_lt_f32_e64 vcc, v11, 0
v_cndmask_b32_e32 v11, v11, v105, vcc
v_add_f32_e64 v12, v12, s99
v_mul_f32_e64 v105, v12, s32
v_cmp_lt_f32_e64 vcc, v12, 0
v_cndmask_b32_e32 v12, v12, v105, vcc
buffer_store_dword v10, v37, s[84:87], 0 idxen
buffer_store_dword v11, v62, s[84:87], 0 idxen
buffer_store_dword v12, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 3
v_add_f32_e64 v13, v13, s99
v_mul_f32_e64 v105, v13, s32
v_cmp_lt_f32_e64 vcc, v13, 0
v_cndmask_b32_e32 v13, v13, v105, vcc
v_add_f32_e64 v14, v14, s99
v_mul_f32_e64 v105, v14, s32
v_cmp_lt_f32_e64 vcc, v14, 0
v_cndmask_b32_e32 v14, v14, v105, vcc
v_add_f32_e64 v15, v15, s99
v_mul_f32_e64 v105, v15, s32
v_cmp_lt_f32_e64 vcc, v15, 0
v_cndmask_b32_e32 v15, v15, v105, vcc
buffer_store_dword v13, v37, s[84:87], 0 idxen
buffer_store_dword v14, v62, s[84:87], 0 idxen
buffer_store_dword v15, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s99, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 4
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 8
v_add_f32_e64 v16, v16, s99
v_mul_f32_e64 v105, v16, s32
v_cmp_lt_f32_e64 vcc, v16, 0
v_cndmask_b32_e32 v16, v16, v105, vcc
v_add_f32_e64 v17, v17, s99
v_mul_f32_e64 v105, v17, s32
v_cmp_lt_f32_e64 vcc, v17, 0
v_cndmask_b32_e32 v17, v17, v105, vcc
v_add_f32_e64 v18, v18, s99
v_mul_f32_e64 v105, v18, s32
v_cmp_lt_f32_e64 vcc, v18, 0
v_cndmask_b32_e32 v18, v18, v105, vcc
buffer_store_dword v16, v37, s[84:87], 0 idxen
buffer_store_dword v17, v62, s[84:87], 0 idxen
buffer_store_dword v18, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 9
v_add_f32_e64 v19, v19, s99
v_mul_f32_e64 v105, v19, s32
v_cmp_lt_f32_e64 vcc, v19, 0
v_cndmask_b32_e32 v19, v19, v105, vcc
v_add_f32_e64 v20, v20, s99
v_mul_f32_e64 v105, v20, s32
v_cmp_lt_f32_e64 vcc, v20, 0
v_cndmask_b32_e32 v20, v20, v105, vcc
v_add_f32_e64 v21, v21, s99
v_mul_f32_e64 v105, v21, s32
v_cmp_lt_f32_e64 vcc, v21, 0
v_cndmask_b32_e32 v21, v21, v105, vcc
buffer_store_dword v19, v37, s[84:87], 0 idxen
buffer_store_dword v20, v62, s[84:87], 0 idxen
buffer_store_dword v21, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 10
v_add_f32_e64 v22, v22, s99
v_mul_f32_e64 v105, v22, s32
v_cmp_lt_f32_e64 vcc, v22, 0
v_cndmask_b32_e32 v22, v22, v105, vcc
v_add_f32_e64 v23, v23, s99
v_mul_f32_e64 v105, v23, s32
v_cmp_lt_f32_e64 vcc, v23, 0
v_cndmask_b32_e32 v23, v23, v105, vcc
v_add_f32_e64 v24, v24, s99
v_mul_f32_e64 v105, v24, s32
v_cmp_lt_f32_e64 vcc, v24, 0
v_cndmask_b32_e32 v24, v24, v105, vcc
buffer_store_dword v22, v37, s[84:87], 0 idxen
buffer_store_dword v23, v62, s[84:87], 0 idxen
buffer_store_dword v24, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 11
v_add_f32_e64 v25, v25, s99
v_mul_f32_e64 v105, v25, s32
v_cmp_lt_f32_e64 vcc, v25, 0
v_cndmask_b32_e32 v25, v25, v105, vcc
v_add_f32_e64 v26, v26, s99
v_mul_f32_e64 v105, v26, s32
v_cmp_lt_f32_e64 vcc, v26, 0
v_cndmask_b32_e32 v26, v26, v105, vcc
v_add_f32_e64 v27, v27, s99
v_mul_f32_e64 v105, v27, s32
v_cmp_lt_f32_e64 vcc, v27, 0
v_cndmask_b32_e32 v27, v27, v105, vcc
buffer_store_dword v25, v37, s[84:87], 0 idxen
buffer_store_dword v26, v62, s[84:87], 0 idxen
buffer_store_dword v27, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
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
s_add_u32 s88, s88, 0x80
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, 32
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
v_readlane_b32 s99, v103, 0
v_add_f32_e64 v4, v4, s99
v_mul_f32_e32 v4, 0xbfb8aa3b, v4
v_exp_f32_e32 v4, v4
v_add_f32_e32 v4, 1.0, v4
v_rcp_f32_e32 v4, v4
v_add_f32_e64 v5, v5, s99
v_mul_f32_e32 v5, 0xbfb8aa3b, v5
v_exp_f32_e32 v5, v5
v_add_f32_e32 v5, 1.0, v5
v_rcp_f32_e32 v5, v5
v_add_f32_e64 v6, v6, s99
v_mul_f32_e32 v6, 0xbfb8aa3b, v6
v_exp_f32_e32 v6, v6
v_add_f32_e32 v6, 1.0, v6
v_rcp_f32_e32 v6, v6
buffer_store_dword v4, v37, s[84:87], 0 idxen
buffer_store_dword v5, v62, s[84:87], 0 idxen
buffer_store_dword v6, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 1
v_add_f32_e64 v7, v7, s99
v_mul_f32_e32 v7, 0xbfb8aa3b, v7
v_exp_f32_e32 v7, v7
v_add_f32_e32 v7, 1.0, v7
v_rcp_f32_e32 v7, v7
v_add_f32_e64 v8, v8, s99
v_mul_f32_e32 v8, 0xbfb8aa3b, v8
v_exp_f32_e32 v8, v8
v_add_f32_e32 v8, 1.0, v8
v_rcp_f32_e32 v8, v8
v_add_f32_e64 v9, v9, s99
v_mul_f32_e32 v9, 0xbfb8aa3b, v9
v_exp_f32_e32 v9, v9
v_add_f32_e32 v9, 1.0, v9
v_rcp_f32_e32 v9, v9
buffer_store_dword v7, v37, s[84:87], 0 idxen
buffer_store_dword v8, v62, s[84:87], 0 idxen
buffer_store_dword v9, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 2
v_add_f32_e64 v10, v10, s99
v_mul_f32_e32 v10, 0xbfb8aa3b, v10
v_exp_f32_e32 v10, v10
v_add_f32_e32 v10, 1.0, v10
v_rcp_f32_e32 v10, v10
v_add_f32_e64 v11, v11, s99
v_mul_f32_e32 v11, 0xbfb8aa3b, v11
v_exp_f32_e32 v11, v11
v_add_f32_e32 v11, 1.0, v11
v_rcp_f32_e32 v11, v11
v_add_f32_e64 v12, v12, s99
v_mul_f32_e32 v12, 0xbfb8aa3b, v12
v_exp_f32_e32 v12, v12
v_add_f32_e32 v12, 1.0, v12
v_rcp_f32_e32 v12, v12
buffer_store_dword v10, v37, s[84:87], 0 idxen
buffer_store_dword v11, v62, s[84:87], 0 idxen
buffer_store_dword v12, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 3
v_add_f32_e64 v13, v13, s99
v_mul_f32_e32 v13, 0xbfb8aa3b, v13
v_exp_f32_e32 v13, v13
v_add_f32_e32 v13, 1.0, v13
v_rcp_f32_e32 v13, v13
v_add_f32_e64 v14, v14, s99
v_mul_f32_e32 v14, 0xbfb8aa3b, v14
v_exp_f32_e32 v14, v14
v_add_f32_e32 v14, 1.0, v14
v_rcp_f32_e32 v14, v14
v_add_f32_e64 v15, v15, s99
v_mul_f32_e32 v15, 0xbfb8aa3b, v15
v_exp_f32_e32 v15, v15
v_add_f32_e32 v15, 1.0, v15
v_rcp_f32_e32 v15, v15
buffer_store_dword v13, v37, s[84:87], 0 idxen
buffer_store_dword v14, v62, s[84:87], 0 idxen
buffer_store_dword v15, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s99, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 4
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 8
v_add_f32_e64 v16, v16, s99
v_mul_f32_e32 v16, 0xbfb8aa3b, v16
v_exp_f32_e32 v16, v16
v_add_f32_e32 v16, 1.0, v16
v_rcp_f32_e32 v16, v16
v_add_f32_e64 v17, v17, s99
v_mul_f32_e32 v17, 0xbfb8aa3b, v17
v_exp_f32_e32 v17, v17
v_add_f32_e32 v17, 1.0, v17
v_rcp_f32_e32 v17, v17
v_add_f32_e64 v18, v18, s99
v_mul_f32_e32 v18, 0xbfb8aa3b, v18
v_exp_f32_e32 v18, v18
v_add_f32_e32 v18, 1.0, v18
v_rcp_f32_e32 v18, v18
buffer_store_dword v16, v37, s[84:87], 0 idxen
buffer_store_dword v17, v62, s[84:87], 0 idxen
buffer_store_dword v18, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 9
v_add_f32_e64 v19, v19, s99
v_mul_f32_e32 v19, 0xbfb8aa3b, v19
v_exp_f32_e32 v19, v19
v_add_f32_e32 v19, 1.0, v19
v_rcp_f32_e32 v19, v19
v_add_f32_e64 v20, v20, s99
v_mul_f32_e32 v20, 0xbfb8aa3b, v20
v_exp_f32_e32 v20, v20
v_add_f32_e32 v20, 1.0, v20
v_rcp_f32_e32 v20, v20
v_add_f32_e64 v21, v21, s99
v_mul_f32_e32 v21, 0xbfb8aa3b, v21
v_exp_f32_e32 v21, v21
v_add_f32_e32 v21, 1.0, v21
v_rcp_f32_e32 v21, v21
buffer_store_dword v19, v37, s[84:87], 0 idxen
buffer_store_dword v20, v62, s[84:87], 0 idxen
buffer_store_dword v21, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 10
v_add_f32_e64 v22, v22, s99
v_mul_f32_e32 v22, 0xbfb8aa3b, v22
v_exp_f32_e32 v22, v22
v_add_f32_e32 v22, 1.0, v22
v_rcp_f32_e32 v22, v22
v_add_f32_e64 v23, v23, s99
v_mul_f32_e32 v23, 0xbfb8aa3b, v23
v_exp_f32_e32 v23, v23
v_add_f32_e32 v23, 1.0, v23
v_rcp_f32_e32 v23, v23
v_add_f32_e64 v24, v24, s99
v_mul_f32_e32 v24, 0xbfb8aa3b, v24
v_exp_f32_e32 v24, v24
v_add_f32_e32 v24, 1.0, v24
v_rcp_f32_e32 v24, v24
buffer_store_dword v22, v37, s[84:87], 0 idxen
buffer_store_dword v23, v62, s[84:87], 0 idxen
buffer_store_dword v24, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 11
v_add_f32_e64 v25, v25, s99
v_mul_f32_e32 v25, 0xbfb8aa3b, v25
v_exp_f32_e32 v25, v25
v_add_f32_e32 v25, 1.0, v25
v_rcp_f32_e32 v25, v25
v_add_f32_e64 v26, v26, s99
v_mul_f32_e32 v26, 0xbfb8aa3b, v26
v_exp_f32_e32 v26, v26
v_add_f32_e32 v26, 1.0, v26
v_rcp_f32_e32 v26, v26
v_add_f32_e64 v27, v27, s99
v_mul_f32_e32 v27, 0xbfb8aa3b, v27
v_exp_f32_e32 v27, v27
v_add_f32_e32 v27, 1.0, v27
v_rcp_f32_e32 v27, v27
buffer_store_dword v25, v37, s[84:87], 0 idxen
buffer_store_dword v26, v62, s[84:87], 0 idxen
buffer_store_dword v27, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
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
s_add_u32 s88, s88, 0x80
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, 32
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
v_readlane_b32 s99, v103, 0
v_add_f32_e64 v4, v4, s99
v_mul_f32_e64 v4, v4, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v4
v_mul_f32_e32 v106, 0xbfb8aa3b, v4
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v4, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v4, v4
v_mul_f32_e32 v4, v4, v105
v_mul_f32_e64 v4, v4, s32
v_add_f32_e64 v5, v5, s99
v_mul_f32_e64 v5, v5, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v5
v_mul_f32_e32 v106, 0xbfb8aa3b, v5
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v5, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v5, v5
v_mul_f32_e32 v5, v5, v105
v_mul_f32_e64 v5, v5, s32
v_add_f32_e64 v6, v6, s99
v_mul_f32_e64 v6, v6, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v6
v_mul_f32_e32 v106, 0xbfb8aa3b, v6
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v6, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v6, v6
v_mul_f32_e32 v6, v6, v105
v_mul_f32_e64 v6, v6, s32
buffer_store_dword v4, v37, s[84:87], 0 idxen
buffer_store_dword v5, v62, s[84:87], 0 idxen
buffer_store_dword v6, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 1
v_add_f32_e64 v7, v7, s99
v_mul_f32_e64 v7, v7, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v7
v_mul_f32_e32 v106, 0xbfb8aa3b, v7
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v7, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v7, v7
v_mul_f32_e32 v7, v7, v105
v_mul_f32_e64 v7, v7, s32
v_add_f32_e64 v8, v8, s99
v_mul_f32_e64 v8, v8, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v8
v_mul_f32_e32 v106, 0xbfb8aa3b, v8
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v8, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v8, v8
v_mul_f32_e32 v8, v8, v105
v_mul_f32_e64 v8, v8, s32
v_add_f32_e64 v9, v9, s99
v_mul_f32_e64 v9, v9, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v9
v_mul_f32_e32 v106, 0xbfb8aa3b, v9
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v9, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v9, v9
v_mul_f32_e32 v9, v9, v105
v_mul_f32_e64 v9, v9, s32
buffer_store_dword v7, v37, s[84:87], 0 idxen
buffer_store_dword v8, v62, s[84:87], 0 idxen
buffer_store_dword v9, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 2
v_add_f32_e64 v10, v10, s99
v_mul_f32_e64 v10, v10, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v10
v_mul_f32_e32 v106, 0xbfb8aa3b, v10
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v10, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v10, v10
v_mul_f32_e32 v10, v10, v105
v_mul_f32_e64 v10, v10, s32
v_add_f32_e64 v11, v11, s99
v_mul_f32_e64 v11, v11, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v11
v_mul_f32_e32 v106, 0xbfb8aa3b, v11
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v11, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v11, v11
v_mul_f32_e32 v11, v11, v105
v_mul_f32_e64 v11, v11, s32
v_add_f32_e64 v12, v12, s99
v_mul_f32_e64 v12, v12, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v12
v_mul_f32_e32 v106, 0xbfb8aa3b, v12
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v12, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v12, v12
v_mul_f32_e32 v12, v12, v105
v_mul_f32_e64 v12, v12, s32
buffer_store_dword v10, v37, s[84:87], 0 idxen
buffer_store_dword v11, v62, s[84:87], 0 idxen
buffer_store_dword v12, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 3
v_add_f32_e64 v13, v13, s99
v_mul_f32_e64 v13, v13, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v13
v_mul_f32_e32 v106, 0xbfb8aa3b, v13
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v13, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v13, v13
v_mul_f32_e32 v13, v13, v105
v_mul_f32_e64 v13, v13, s32
v_add_f32_e64 v14, v14, s99
v_mul_f32_e64 v14, v14, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v14
v_mul_f32_e32 v106, 0xbfb8aa3b, v14
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v14, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v14, v14
v_mul_f32_e32 v14, v14, v105
v_mul_f32_e64 v14, v14, s32
v_add_f32_e64 v15, v15, s99
v_mul_f32_e64 v15, v15, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v15
v_mul_f32_e32 v106, 0xbfb8aa3b, v15
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v15, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v15, v15
v_mul_f32_e32 v15, v15, v105
v_mul_f32_e64 v15, v15, s32
buffer_store_dword v13, v37, s[84:87], 0 idxen
buffer_store_dword v14, v62, s[84:87], 0 idxen
buffer_store_dword v15, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s99, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 4
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 8
v_add_f32_e64 v16, v16, s99
v_mul_f32_e64 v16, v16, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v16
v_mul_f32_e32 v106, 0xbfb8aa3b, v16
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v16, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v16, v16
v_mul_f32_e32 v16, v16, v105
v_mul_f32_e64 v16, v16, s32
v_add_f32_e64 v17, v17, s99
v_mul_f32_e64 v17, v17, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v17
v_mul_f32_e32 v106, 0xbfb8aa3b, v17
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v17, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v17, v17
v_mul_f32_e32 v17, v17, v105
v_mul_f32_e64 v17, v17, s32
v_add_f32_e64 v18, v18, s99
v_mul_f32_e64 v18, v18, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v18
v_mul_f32_e32 v106, 0xbfb8aa3b, v18
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v18, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v18, v18
v_mul_f32_e32 v18, v18, v105
v_mul_f32_e64 v18, v18, s32
buffer_store_dword v16, v37, s[84:87], 0 idxen
buffer_store_dword v17, v62, s[84:87], 0 idxen
buffer_store_dword v18, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 9
v_add_f32_e64 v19, v19, s99
v_mul_f32_e64 v19, v19, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v19
v_mul_f32_e32 v106, 0xbfb8aa3b, v19
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v19, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v19, v19
v_mul_f32_e32 v19, v19, v105
v_mul_f32_e64 v19, v19, s32
v_add_f32_e64 v20, v20, s99
v_mul_f32_e64 v20, v20, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v20
v_mul_f32_e32 v106, 0xbfb8aa3b, v20
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v20, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v20, v20
v_mul_f32_e32 v20, v20, v105
v_mul_f32_e64 v20, v20, s32
v_add_f32_e64 v21, v21, s99
v_mul_f32_e64 v21, v21, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v21
v_mul_f32_e32 v106, 0xbfb8aa3b, v21
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v21, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v21, v21
v_mul_f32_e32 v21, v21, v105
v_mul_f32_e64 v21, v21, s32
buffer_store_dword v19, v37, s[84:87], 0 idxen
buffer_store_dword v20, v62, s[84:87], 0 idxen
buffer_store_dword v21, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 10
v_add_f32_e64 v22, v22, s99
v_mul_f32_e64 v22, v22, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v22
v_mul_f32_e32 v106, 0xbfb8aa3b, v22
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v22, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v22, v22
v_mul_f32_e32 v22, v22, v105
v_mul_f32_e64 v22, v22, s32
v_add_f32_e64 v23, v23, s99
v_mul_f32_e64 v23, v23, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v23
v_mul_f32_e32 v106, 0xbfb8aa3b, v23
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v23, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v23, v23
v_mul_f32_e32 v23, v23, v105
v_mul_f32_e64 v23, v23, s32
v_add_f32_e64 v24, v24, s99
v_mul_f32_e64 v24, v24, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v24
v_mul_f32_e32 v106, 0xbfb8aa3b, v24
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v24, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v24, v24
v_mul_f32_e32 v24, v24, v105
v_mul_f32_e64 v24, v24, s32
buffer_store_dword v22, v37, s[84:87], 0 idxen
buffer_store_dword v23, v62, s[84:87], 0 idxen
buffer_store_dword v24, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
s_add_u32 s84, s84, s99
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s99, v103, 11
v_add_f32_e64 v25, v25, s99
v_mul_f32_e64 v25, v25, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v25
v_mul_f32_e32 v106, 0xbfb8aa3b, v25
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v25, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v25, v25
v_mul_f32_e32 v25, v25, v105
v_mul_f32_e64 v25, v25, s32
v_add_f32_e64 v26, v26, s99
v_mul_f32_e64 v26, v26, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v26
v_mul_f32_e32 v106, 0xbfb8aa3b, v26
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v26, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v26, v26
v_mul_f32_e32 v26, v26, v105
v_mul_f32_e64 v26, v26, s32
v_add_f32_e64 v27, v27, s99
v_mul_f32_e64 v27, v27, s33
v_mul_f32_e32 v105, 0x3fb8aa3b, v27
v_mul_f32_e32 v106, 0xbfb8aa3b, v27
v_exp_f32_e32 v105, v105
v_exp_f32_e32 v106, v106
v_add_f32_e32 v27, v105, v106
v_sub_f32_e32 v105, v105, v106
v_rcp_f32_e32 v27, v27
v_mul_f32_e32 v27, v27, v105
v_mul_f32_e64 v27, v27, s32
buffer_store_dword v25, v37, s[84:87], 0 idxen
buffer_store_dword v26, v62, s[84:87], 0 idxen
buffer_store_dword v27, v63, s[84:87], 0 idxen
s_lshl_b32 s99, s46, 2
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
s_add_u32 s88, s88, 0x80
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
