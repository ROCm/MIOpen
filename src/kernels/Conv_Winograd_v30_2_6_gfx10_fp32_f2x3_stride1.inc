/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

s_version 0x2004
s_inst_prefetch 0x3
s_inst_prefetch 0x3
v_mov_b32_e32 v1, v0
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_mov_b32_e32 v180, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s76, 0xc220
s_mov_b32 s75, 0xc220
v_and_b32_e32 v181, 0xc0, v0
v_add_co_u32 v1, vcc, v0, v181
v_readfirstlane_b32 s77, v1
s_lshr_b32 s77, s77, 5
s_add_u32 s77, s77, 8
s_and_b32 s71, s77, 20
s_mov_b64 s[78:79], s[6:7]
s_load_dwordx16 s[12:27], s[78:79], null
s_load_dwordx4 s[28:31], s[78:79], 0x40
s_load_dwordx2 s[32:33], s[78:79], 0x50
s_waitcnt lgkmcnt(0)
s_and_b32 s18, s18, 0xffff
s_bitcmp1_b32 s18, 6
s_cbranch_scc0 16
s_and_b32 s21, s21, 0xffff
s_and_b32 s23, s23, 0xffff
s_and_b32 s25, s25, 0xffff
s_and_b32 s27, s27, 0xffff
s_load_dwordx2 s[20:21], s[20:21], null
s_load_dwordx2 s[22:23], s[22:23], null
s_load_dwordx2 s[24:25], s[24:25], null
s_load_dwordx2 s[26:27], s[26:27], null
s_bitcmp1_b32 s18, 7
s_cbranch_scc0 2
s_load_dwordx2 s[34:35], s[78:79], 0x58
s_mov_b32 s36, 1.0
s_bitcmp1_b32 s18, 8
s_cbranch_scc0 2
s_load_dword s36, s[78:79], 0x60
s_bitcmp1_b32 s18, 7
s_cbranch_scc0 7
s_bitcmp1_b32 s18, 6
s_cbranch_scc0 5
s_waitcnt lgkmcnt(0)
s_and_b32 s35, s35, 0xffff
s_load_dwordx2 s[34:35], s[34:35], null
s_bitcmp1_b32 s18, 9
s_cbranch_scc0 77
s_mov_b32 s77, 0x8c
s_mov_b32 s80, 0x9c
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cmp_eq_u64 0, vcc
s_cselect_b32 s77, s80, s77
s_load_dword s44, s[78:79], 0x88
s_load_dword s70, s[78:79], 0x98
s_load_dword s48, s[78:79], s77
s_load_dword s45, s[78:79], 0xa8
s_load_dword s46, s[78:79], 0xac
s_bitcmp1_b32 s18, 10
s_cbranch_scc0 76
s_load_dwordx4 s[84:87], s[78:79], 0xb8
v_ffbh_u32_e32 v184, s17
v_lshlrev_b32_e64 v185, v184, s17
v_and_b32_e32 v183, 0xffffff00, v185
v_cmp_eq_u32_e32 vcc, 0x80000000, v185
v_cvt_f32_u32_e32 v183, v183
v_rcp_f32_e32 v181, v183
v_sub_co_ci_u32_e32 v182, vcc, 32, v184, vcc
v_cvt_f32_ubyte0_e32 v184, v185
v_fma_f32 v183, v183, v181, -1.0
v_fma_f32 v183, v184, v181, v183
v_fmaak_f32 v183, v183, v181, 0x9f000000
v_mul_f32_e32 v183, 0x5f800000, v183
v_mov_b32_e32 v184, 0
v_cvt_flr_i32_f32_e64 v183, -v183
v_lshl_add_u32 v181, v181, 9, v183
v_mad_u64_u32 v[184:185], vcc, v185, v181, v[184:185]
v_sub_co_ci_u32_e64 v181, vcc, v181, -1, vcc
v_mul_hi_u32 v183, s8, v181
v_add_co_u32 v181, vcc, v183, s8
v_add_co_ci_u32_e64 v183, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v182
v_cndmask_b32_e32 v181, v181, v183, vcc
v_alignbit_b32 v181, v183, v181, v182
s_nop 0
v_readfirstlane_b32 s81, v181
s_mul_i32 s82, s81, s17
s_sub_u32 s8, s8, s82
s_waitcnt lgkmcnt(0)
s_lshl_b32 s85, s85, 2
s_lshl_b64 s[86:87], s[86:87], 2
s_mul_i32 s82, s85, s81
s_add_u32 s20, s20, s82
s_addc_u32 s21, s21, 0
s_mul_i32 s82, s86, s81
s_add_u32 s22, s22, s82
s_addc_u32 s23, s23, 0
s_mul_i32 s82, s87, s81
s_add_u32 s24, s24, s82
s_addc_u32 s25, s25, 0
s_branch 19
s_mul_i32 s48, s14, s15
s_mul_i32 s44, s48, s13
s_mul_i32 s46, s32, s33
s_mul_i32 s45, s46, s16
s_bitcmp1_b32 s18, 13
s_cbranch_scc0 2
s_load_dwordx8 s[88:95], s[78:79], 0x68
s_mul_i32 s77, s28, s29
s_bitcmp1_b32 s18, 2
s_cselect_b32 s80, s16, s13
s_mul_i32 s80, s77, s80
s_bitcmp1_b32 s18, 2
s_cselect_b32 s85, s80, s77
s_cselect_b32 s70, s77, s80
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cmp_eq_u64 0, vcc
s_cselect_b32 s48, s85, s48
s_waitcnt lgkmcnt(0)
s_lshl_b32 s47, s48, 2
s_and_b32 s21, s21, 0xffff
s_and_b32 s23, s23, 0xffff
s_and_b32 s25, s25, 0xffff
s_and_b32 s27, s27, 0xffff
s_and_b32 s35, s35, 0xffff
s_bitcmp1_b32 s18, 13
s_cbranch_scc0 8
s_add_u32 s20, s20, s88
s_addc_u32 s21, s21, s89
s_add_u32 s22, s22, s90
s_addc_u32 s23, s23, s91
s_add_u32 s24, s24, s92
s_addc_u32 s25, s25, s93
s_add_u32 s34, s34, s94
s_addc_u32 s35, s35, s95
s_and_b32 s81, 0, s30
s_addc_u32 s81, s32, 0
s_ashr_i32 s81, s81, 0
s_add_u32 s77, s81, 1
v_mov_b32_e32 v182, 0x80000000
v_mul_hi_u32 v182, v182, s77
s_nop 0
v_readfirstlane_b32 s77, v182
s_andn2_b32 s81, 0, s31
s_addc_u32 s81, s33, 0
s_ashr_i32 s81, s81, 0
s_add_u32 s80, s81, 1
v_mov_b32_e32 v182, 0x80000000
v_mul_hi_u32 v182, v182, s80
s_nop 0
v_readfirstlane_b32 s80, v182
s_sub_u32 s55, 0, s80
s_sub_u32 s54, 0, s77
s_add_u32 s9, s28, 2
v_mov_b32_e32 v182, 0x55555556
v_mul_hi_u32 v182, v182, s9
s_nop 0
v_readfirstlane_b32 s9, v182
s_add_u32 s10, s29, 2
v_mov_b32_e32 v182, 0x55555556
v_mul_hi_u32 v182, v182, s10
s_nop 0
v_readfirstlane_b32 s10, v182
v_mad_i32_i24 v181, 3, s9, -2
v_sub_co_u32 v181, vcc, v181, s28
v_add_co_ci_u32_e64 v181, vcc, 0, 0, vcc
s_nop 0
v_readfirstlane_b32 s81, v181
s_and_b32 s81, s81, 0
s_and_b32 s81, s81, s9
s_add_u32 s9, s9, s81
v_readfirstlane_b32 s82, v1
s_and_b32 s83, s82, 64
s_cselect_b32 s83, 0x80000, 0
s_or_b32 s18, s18, s83
s_lshl_b32 s49, s47, 1
s_mov_b64 s[50:51], 0
s_bitcmp1_b32 s18, 12
s_cselect_b32 s81, 0, -1
s_bitcmp1_b32 s18, 11
s_cselect_b32 s81, s81, 1
s_cmp_gt_u32 s10, s81
s_cbranch_scc0 8
s_bitset1_b32 s18, 23
s_bitset1_b32 s18, 20
s_bitset0_b32 s18, 19
s_ashr_i32 s49, s49, 1
s_ashr_i64 s[50:51], s[50:51], 1
s_add_u32 s10, s10, 1
s_and_b32 s10, s10, -2
s_branch 16
s_and_b32 s83, s13, 1
s_cselect_b32 s83, 0, 0x1000000
s_bitcmp1_b32 s18, 2
s_cselect_b32 s83, 0, s83
s_or_b32 s18, s18, s83
s_cmp_eq_u32 s83, 0
s_cselect_b32 s49, s47, s49
s_cselect_b32 s50, s47, s50
s_cselect_b32 s51, 0, s51
s_bitcmp0_b32 s82, 8
s_cselect_b32 s83, s83, 0
s_cmp_eq_u32 s83, 0
s_cselect_b32 s83, 0, 0x80000
s_andn2_b32 s18, s18, s83
v_bfe_u32 v182, v1, 2, 6
v_lshrrev_b32_e32 v175, 1, v182
s_bitcmp0_b32 s82, 8
s_cselect_b32 s83, 0x1000000, 0
s_or_b32 s83, s83, 0x100000
s_and_b32 s83, s18, s83
s_cselect_b32 s83, 0, 15
v_bfi_b32 v175, s83, v182, v175
s_mul_i32 s68, s12, s77
s_sub_u32 s68, s68, 1
s_lshr_b32 s68, s68, 0
s_add_u32 s68, s68, 1
s_lshr_b32 s82, -1, 16
s_and_b32 s82, s82, s68
s_lshr_b32 s83, s68, 16
s_mul_i32 s83, s83, s80
s_mul_i32 s68, s82, s80
s_lshl_b32 s82, s83, 16
s_lshr_b32 s83, s83, 16
s_add_u32 s68, s82, s68
s_addc_u32 s69, s83, 0
s_sub_u32 s68, s68, 1
s_subb_u32 s69, s69, 0
s_lshr_b64 s[68:69], s[68:69], 5
s_add_u32 s68, s68, 1
s_addc_u32 s69, s69, 0
v_mov_b32_e32 v182, s8
v_mov_b32_e32 v183, s17
v_and_b32_e32 v184, 3, v1
v_cmp_eq_u32_e32 vcc, 2, v184
v_cndmask_b32_e32 v182, v182, v183, vcc
v_cmp_eq_u32_e32 vcc, 1, v184
v_cndmask_b32_e32 v185, 0, v175, vcc
s_bitcmp1_b32 s18, 20
s_cbranch_scc0 4
v_add_co_u32 v183, vcc, v175, 8
v_cmp_eq_u32_e32 vcc, 0, v184
v_cndmask_b32_e32 v185, v185, v183, vcc
v_cmp_eq_u32_e64 s[82:83], 3, v184
v_bfe_u32 v173, v185, 0, 5
v_mad_u32_u24 v173, v182, 32, v173
v_ffbh_u32_e32 v188, s80
v_lshlrev_b32_e64 v189, v188, s80
v_and_b32_e32 v187, 0xffffff00, v189
v_cmp_eq_u32_e32 vcc, 0x80000000, v189
v_cvt_f32_u32_e32 v187, v187
v_rcp_f32_e32 v174, v187
v_sub_co_ci_u32_e32 v186, vcc, 32, v188, vcc
v_cvt_f32_ubyte0_e32 v188, v189
v_fma_f32 v187, v187, v174, -1.0
v_fma_f32 v187, v188, v174, v187
v_fmaak_f32 v187, v187, v174, 0x9f000000
v_mul_f32_e32 v187, 0x5f800000, v187
v_mov_b32_e32 v188, 0
v_cvt_flr_i32_f32_e64 v187, -v187
v_lshl_add_u32 v174, v174, 9, v187
v_mad_u64_u32 v[188:189], vcc, v189, v174, v[188:189]
v_sub_co_ci_u32_e64 v174, vcc, v174, -1, vcc
v_mul_hi_u32 v187, v173, v174
v_add_co_u32 v174, vcc, v187, v173
v_add_co_ci_u32_e64 v187, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v186
v_cndmask_b32_e32 v174, v174, v187, vcc
v_alignbit_b32 v174, v187, v174, v186
v_mad_i32_i24 v172, v174, s55, v173
v_lshrrev_b32_e32 v173, 5, v185
v_mad_u32_u24 v173, v174, 1, v173
v_cndmask_b32_e64 v173, v173, 1, s[82:83]
v_ffbh_u32_e32 v188, s77
v_lshlrev_b32_e64 v189, v188, s77
v_and_b32_e32 v187, 0xffffff00, v189
v_cmp_eq_u32_e32 vcc, 0x80000000, v189
v_cvt_f32_u32_e32 v187, v187
v_rcp_f32_e32 v174, v187
v_sub_co_ci_u32_e32 v186, vcc, 32, v188, vcc
v_cvt_f32_ubyte0_e32 v188, v189
v_fma_f32 v187, v187, v174, -1.0
v_fma_f32 v187, v188, v174, v187
v_fmaak_f32 v187, v187, v174, 0x9f000000
v_mul_f32_e32 v187, 0x5f800000, v187
v_mov_b32_e32 v188, 0
v_cvt_flr_i32_f32_e64 v187, -v187
v_lshl_add_u32 v174, v174, 9, v187
v_mad_u64_u32 v[188:189], vcc, v189, v174, v[188:189]
v_sub_co_ci_u32_e64 v174, vcc, v174, -1, vcc
v_mul_hi_u32 v187, v173, v174
v_add_co_u32 v174, vcc, v187, v173
v_add_co_ci_u32_e64 v187, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v186
v_cndmask_b32_e32 v174, v174, v187, vcc
v_alignbit_b32 v174, v187, v174, v186
v_mad_i32_i24 v173, v174, s54, v173
v_readlane_b32 s56, v172, 2
v_readlane_b32 s57, v173, 2
v_readlane_b32 s58, v174, 2
v_readlane_b32 s59, v173, 3
v_readlane_b32 s60, v174, 3
v_add_co_u32 v172, vcc, v172, s55
v_add_co_u32 v173, vcc, v173, s54
v_mov_b32_dpp v174, v174 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v172, v172 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v173, v173 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x11014000
s_mov_b32 s82, 0x80000000
s_mov_b32 s83, 0x11014000
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 6
v_xor_b32_dpp v176, v1, v1 quad_perm:[1,3,2,2] row_mask:0xf bank_mask:0xf
v_subrev_co_u32 v176, vcc, 1, v176
v_cvt_f32_i32_e32 v176, v176
s_branch 5
v_xor_b32_dpp v176, v1, v1 quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32 v176, vcc, 1, v176
v_cvt_f32_i32_e32 v176, v176
v_mov_b32_e32 v177, 1
v_xor_b32_dpp v177, v1, v1 quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v177, v1, v1 quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32 v177, vcc, 1, v177
v_mov_b32_e32 v178, 1
v_xor_b32_dpp v178, v1, v1 quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v178, v1, v1 quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32 v178, vcc, 1, v178
v_cvt_f32_i32_e32 v177, v177
v_cvt_f32_i32_e32 v178, v178
v_lshrrev_b32_e64 v181, 2, s71
v_and_b32_e32 v182, 3, v1
v_bfe_u32 v183, v1, 4, 3
v_mad_u32_u24 v171, v183, 4, v182
v_lshlrev_b32_e32 v171, 4, v171
v_mad_u32_u24 v162, v181, 4, v182
v_lshlrev_b32_e32 v162, 4, v162
v_bfe_u32 v181, v1, 2, 2
v_and_b32_e32 v182, 1, v181
v_mad_u32_u24 v184, v181, 16, v182
v_lshlrev_b32_e32 v184, 6, v184
v_xor_b32_e32 v162, v162, v184
v_mul_u32_u24_e32 v184, 0x400, v181
v_xor_b32_e32 v171, v171, v184
s_lshr_b32 s71, s71, 0
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 64
s_and_b32 s78, s18, 0x1100000
s_addc_u32 s78, 0, 0
v_lshrrev_b32_e32 v184, 1, v1
s_mul_i32 s77, 60, s78
s_sub_u32 s77, 63, s77
v_bfi_b32 v184, s77, v1, v184
v_and_b32_e32 v181, 1, v184
v_bfe_u32 v182, v184, 1, 1
v_xor_b32_e32 v181, v181, v182
v_bfe_u32 v183, v184, 3, 1
v_mad_u32_u24 v182, v182, 2, v183
v_mul_u32_u24_e32 v181, 0x118, v181
v_bfe_u32 v183, v184, 2, 1
v_mad_u32_u24 v182, v182, 2, v181
v_xor_b32_e32 v182, v182, v183
v_and_b32_e32 v183, 0xf0, v184
v_xor_b32_e32 v182, v182, v183
s_mul_i32 s77, 4, s78
s_sub_u32 s77, 6, s77
v_bfe_u32 v184, v1, s77, 1
v_mul_u32_u24_e32 v184, 0x1040, v184
v_xor_b32_e32 v164, 0x314, v182
v_xor_b32_e32 v165, 0x31c, v182
v_xor_b32_e32 v166, 8, v182
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v163, v182, v166, vcc
v_cndmask_b32_e32 v166, v166, v182, vcc
v_mad_u32_u24 v163, 4, v163, v184
v_mad_u32_u24 v164, 4, v164, v184
v_mad_u32_u24 v165, 4, v165, v184
v_mad_u32_u24 v166, 4, v166, v184
s_mov_b32 s77, 0x1040
s_and_b32 s78, s18, 0x1100000
s_cselect_b32 s77, 0x80, s77
v_add_co_u32 v167, vcc, v163, s77
v_add_co_u32 v168, vcc, v164, s77
v_add_co_u32 v169, vcc, v165, s77
v_add_co_u32 v170, vcc, v166, s77
s_branch 57
s_bfe_u32 s78, s18, 0x10014
v_lshrrev_b32_e32 v184, 1, v1
s_mul_i32 s77, 60, s78
s_sub_u32 s77, 63, s77
v_bfi_b32 v184, s77, v1, v184
v_and_b32_e32 v181, 1, v184
v_bfe_u32 v182, v184, 1, 1
v_bfe_u32 v183, v184, 3, 1
v_xor_b32_e32 v181, v181, v182
v_mad_u32_u24 v182, v182, 2, v183
v_mul_u32_u24_e32 v181, 0x109, v181
v_bfe_u32 v183, v184, 2, 1
v_mad_u32_u24 v182, v182, 2, v181
v_xor_b32_e32 v182, v182, v183
v_and_b32_e32 v183, 0xf0, v184
v_or_b32_e32 v182, v182, v183
s_mul_i32 s77, 4, s78
s_sub_u32 s77, 6, s77
v_bfe_u32 v184, v1, s77, 1
v_mul_u32_u24_e32 v184, 0x1040, v184
v_mad_u32_u24 v163, 4, v182, v184
v_xor_b32_e32 v164, 0x307, v182
v_mad_u32_u24 v164, 4, v164, v184
v_xor_b32_e32 v165, 0x30f, v182
v_mad_u32_u24 v165, 4, v165, v184
v_xor_b32_e32 v166, 8, v182
v_mad_u32_u24 v166, 4, v166, v184
s_mov_b32 s77, 0x1040
s_bitcmp1_b32 s18, 20
s_cselect_b32 s77, 0x80, s77
v_add_co_u32 v167, vcc, v163, s77
v_add_co_u32 v168, vcc, v164, s77
v_add_co_u32 v169, vcc, v165, s77
v_add_co_u32 v170, vcc, v166, s77
v_subrev_co_u32 v172, vcc, s56, v172
v_mov_b32_e32 v182, s55
v_cmp_lt_i32_e32 vcc, v172, v182
v_sub_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_mad_i32_i24 v172, v181, s55, v172
v_mad_i32_i24 v174, v181, s60, v174
v_mad_i32_i24 v173, v181, s59, v173
v_mov_b32_e32 v182, s54
v_cmp_lt_i32_e32 vcc, v173, v182
v_sub_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v181
v_mad_i32_i24 v173, v181, v182, v173
v_subrev_co_u32 v173, vcc, s57, v173
v_cmp_lt_i32_e32 vcc, v173, v182
v_sub_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v181
v_mad_i32_i24 v173, v181, s54, v173
v_subrev_co_u32 v174, vcc, s58, v174
s_mov_b32 s11, 0
s_mov_b32 s38, s28
s_mov_b32 s39, 1
s_mov_b32 s64, 0
s_mov_b32 s65, s16
s_mov_b32 s63, s65
s_sub_u32 s72, -1, s71
s_sub_u32 s72, s72, 32
s_bitset1_b32 s18, 21
s_mov_b32 s83, 0
s_mov_b32 s87, 0
s_mov_b32 s73, 19
s_mov_b32 s62, 0
s_bitset1_b32 s18, 26
s_call_b64 s[4:5], 2887
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 1
s_branch 1439
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v116, v118, v116 div:2
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_subrev_f32_e64 v119, v117, v119 div:2
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_e64 v117, v118, v117 div:2
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
v_fma_f32 v118, v118, 1.0, -v117
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_fmac_f32_dpp v116, v116, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v117, v117, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_fmac_f32_dpp v118, v118, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_fmac_f32_dpp v119, v119, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x3
buffer_load_dword v104, v2, s[40:43], 0 idxen
buffer_load_dword v106, v68, s[40:43], 0 idxen
buffer_load_dword v105, v3, s[40:43], 0 idxen
buffer_load_dword v107, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v108
ds_read_b128 v[70:73], v171 offset:29440
ds_write_b32 v168, v109
ds_read_b128 v[74:77], v171 offset:29696
ds_write_b32 v169, v110
ds_read_b128 v[86:89], v162 offset:28928
ds_write_b32 v170, v111
ds_read_b128 v[90:93], v162 offset:29056
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 2766
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v120, v122, v120 div:2
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_subrev_f32_e64 v123, v121, v123 div:2
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_e64 v121, v122, v121 div:2
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
v_fma_f32 v122, v122, 1.0, -v121
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_fmac_f32_dpp v120, v120, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v121, v121, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_fmac_f32_dpp v122, v122, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_fmac_f32_dpp v123, v123, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x3
buffer_load_dword v108, v102, s[40:43], 0 idxen
buffer_load_dword v110, v152, s[40:43], 0 idxen
buffer_load_dword v109, v103, s[40:43], 0 idxen
buffer_load_dword v111, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v112 offset:8256
ds_read_b128 v[78:81], v171 offset:33536
ds_write_b32 v164, v113 offset:8256
ds_read_b128 v[82:85], v171 offset:33792
ds_write_b32 v165, v114 offset:8256
ds_read_b128 v[94:97], v162 offset:33024
ds_write_b32 v166, v115 offset:8256
ds_read_b128 v[98:101], v162 offset:33152
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 2646
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v124, v126, v124 div:2
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_subrev_f32_e64 v127, v125, v127 div:2
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_e64 v125, v126, v125 div:2
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
v_fma_f32 v126, v126, 1.0, -v125
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_fmac_f32_dpp v124, v124, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v125, v125, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_fmac_f32_dpp v126, v126, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_fmac_f32_dpp v127, v127, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x3
buffer_load_dword v112, v2, s[40:43], 0 idxen
buffer_load_dword v114, v68, s[40:43], 0 idxen
buffer_load_dword v113, v3, s[40:43], 0 idxen
buffer_load_dword v115, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v116 offset:8256
ds_read_b128 v[70:73], v171 offset:37696
ds_write_b32 v168, v117 offset:8256
ds_read_b128 v[74:77], v171 offset:37952
ds_write_b32 v169, v118 offset:8256
ds_read_b128 v[86:89], v162 offset:37184
ds_write_b32 v170, v119 offset:8256
ds_read_b128 v[90:93], v162 offset:37312
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 2526
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v128, v130, v128 div:2
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_subrev_f32_e64 v131, v129, v131 div:2
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_e64 v129, v130, v129 div:2
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
v_fma_f32 v130, v130, 1.0, -v129
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_fmac_f32_dpp v128, v128, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v129, v129, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_fmac_f32_dpp v130, v130, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_fmac_f32_dpp v131, v131, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
s_barrier
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x3
buffer_load_dword v116, v102, s[40:43], 0 idxen
buffer_load_dword v118, v152, s[40:43], 0 idxen
buffer_load_dword v117, v103, s[40:43], 0 idxen
buffer_load_dword v119, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v120 offset:16512
ds_read_b128 v[78:81], v171 offset:41792
ds_write_b32 v164, v121 offset:16512
ds_read_b128 v[82:85], v171 offset:42048
ds_write_b32 v165, v122 offset:16512
ds_read_b128 v[94:97], v162 offset:41280
ds_write_b32 v166, v123 offset:16512
ds_read_b128 v[98:101], v162 offset:41408
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 6
s_call_b64 s[4:5], 2405
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v132, v134, v132 div:2
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_subrev_f32_e64 v135, v133, v135 div:2
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_e64 v133, v134, v133 div:2
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
v_fma_f32 v134, v134, 1.0, -v133
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_fmac_f32_dpp v132, v132, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v133, v133, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_fmac_f32_dpp v134, v134, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_fmac_f32_dpp v135, v135, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x3
buffer_load_dword v120, v2, s[40:43], 0 idxen
buffer_load_dword v122, v68, s[40:43], 0 idxen
buffer_load_dword v121, v3, s[40:43], 0 idxen
buffer_load_dword v123, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v124 offset:16512
ds_read_b128 v[70:73], v171 offset:45952
ds_write_b32 v168, v125 offset:16512
ds_read_b128 v[74:77], v171 offset:46208
ds_write_b32 v169, v126 offset:16512
ds_read_b128 v[86:89], v162 offset:45440
ds_write_b32 v170, v127 offset:16512
ds_read_b128 v[90:93], v162 offset:45568
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 2286
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v136, v138, v136 div:2
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_subrev_f32_e64 v139, v137, v139 div:2
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_e64 v137, v138, v137 div:2
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
v_fma_f32 v138, v138, 1.0, -v137
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_fmac_f32_dpp v136, v136, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v137, v137, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_fmac_f32_dpp v138, v138, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_fmac_f32_dpp v139, v139, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x3
buffer_load_dword v124, v102, s[40:43], 0 idxen
buffer_load_dword v126, v152, s[40:43], 0 idxen
buffer_load_dword v125, v103, s[40:43], 0 idxen
buffer_load_dword v127, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v128 offset:24768
ds_read_b128 v[78:81], v171 offset:512
ds_write_b32 v164, v129 offset:24768
ds_read_b128 v[82:85], v171 offset:768
ds_write_b32 v165, v130 offset:24768
ds_read_b128 v[94:97], v162
ds_write_b32 v166, v131 offset:24768
ds_read_b128 v[98:101], v162 offset:128
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 2166
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v140, v142, v140 div:2
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_subrev_f32_e64 v143, v141, v143 div:2
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_e64 v141, v142, v141 div:2
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
v_fma_f32 v142, v142, 1.0, -v141
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_fmac_f32_dpp v140, v140, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v141, v141, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_fmac_f32_dpp v142, v142, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_fmac_f32_dpp v143, v143, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x3
buffer_load_dword v128, v2, s[40:43], 0 idxen
buffer_load_dword v130, v68, s[40:43], 0 idxen
buffer_load_dword v129, v3, s[40:43], 0 idxen
buffer_load_dword v131, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v132 offset:24768
ds_read_b128 v[70:73], v171 offset:4672
ds_write_b32 v168, v133 offset:24768
ds_read_b128 v[74:77], v171 offset:4928
ds_write_b32 v169, v134 offset:24768
ds_read_b128 v[86:89], v162 offset:4160
ds_write_b32 v170, v135 offset:24768
ds_read_b128 v[90:93], v162 offset:4288
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 2046
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v144, v146, v144 div:2
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_subrev_f32_e64 v147, v145, v147 div:2
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_e64 v145, v146, v145 div:2
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
v_fma_f32 v146, v146, 1.0, -v145
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_fmac_f32_dpp v144, v144, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v145, v145, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_fmac_f32_dpp v146, v146, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_fmac_f32_dpp v147, v147, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
s_barrier
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x3
buffer_load_dword v132, v102, s[40:43], 0 idxen
buffer_load_dword v134, v152, s[40:43], 0 idxen
buffer_load_dword v133, v103, s[40:43], 0 idxen
buffer_load_dword v135, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v136 offset:33024
ds_read_b128 v[78:81], v171 offset:8768
ds_write_b32 v164, v137 offset:33024
ds_read_b128 v[82:85], v171 offset:9024
ds_write_b32 v165, v138 offset:33024
ds_read_b128 v[94:97], v162 offset:8256
ds_write_b32 v166, v139 offset:33024
ds_read_b128 v[98:101], v162 offset:8384
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 6
s_call_b64 s[4:5], 1925
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v148, v150, v148 div:2
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_subrev_f32_e64 v151, v149, v151 div:2
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_e64 v149, v150, v149 div:2
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
v_fma_f32 v150, v150, 1.0, -v149
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_fmac_f32_dpp v148, v148, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v149, v149, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_fmac_f32_dpp v150, v150, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_fmac_f32_dpp v151, v151, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x3
buffer_load_dword v136, v2, s[40:43], 0 idxen
buffer_load_dword v138, v68, s[40:43], 0 idxen
buffer_load_dword v137, v3, s[40:43], 0 idxen
buffer_load_dword v139, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v140 offset:33024
ds_read_b128 v[70:73], v171 offset:12928
ds_write_b32 v168, v141 offset:33024
ds_read_b128 v[74:77], v171 offset:13184
ds_write_b32 v169, v142 offset:33024
ds_read_b128 v[86:89], v162 offset:12416
ds_write_b32 v170, v143 offset:33024
ds_read_b128 v[90:93], v162 offset:12544
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 1806
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v104, v106, v104 div:2
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_subrev_f32_e64 v107, v105, v107 div:2
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_e64 v105, v106, v105 div:2
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
v_fma_f32 v106, v106, 1.0, -v105
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_fmac_f32_dpp v104, v104, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v105, v105, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_fmac_f32_dpp v106, v106, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_fmac_f32_dpp v107, v107, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x3
buffer_load_dword v140, v102, s[40:43], 0 idxen
buffer_load_dword v142, v152, s[40:43], 0 idxen
buffer_load_dword v141, v103, s[40:43], 0 idxen
buffer_load_dword v143, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v144 offset:41280
ds_read_b128 v[78:81], v171 offset:17024
ds_write_b32 v164, v145 offset:41280
ds_read_b128 v[82:85], v171 offset:17280
ds_write_b32 v165, v146 offset:41280
ds_read_b128 v[94:97], v162 offset:16512
ds_write_b32 v166, v147 offset:41280
ds_read_b128 v[98:101], v162 offset:16640
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 1686
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v108, v110, v108 div:2
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_subrev_f32_e64 v111, v109, v111 div:2
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_e64 v109, v110, v109 div:2
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
v_fma_f32 v110, v110, 1.0, -v109
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_fmac_f32_dpp v108, v108, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v109, v109, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_fmac_f32_dpp v110, v110, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_fmac_f32_dpp v111, v111, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x3
buffer_load_dword v144, v2, s[40:43], 0 idxen
buffer_load_dword v146, v68, s[40:43], 0 idxen
buffer_load_dword v145, v3, s[40:43], 0 idxen
buffer_load_dword v147, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v148 offset:41280
ds_read_b128 v[70:73], v171 offset:21184
ds_write_b32 v168, v149 offset:41280
ds_read_b128 v[74:77], v171 offset:21440
ds_write_b32 v169, v150 offset:41280
ds_read_b128 v[86:89], v162 offset:20672
ds_write_b32 v170, v151 offset:41280
ds_read_b128 v[90:93], v162 offset:20800
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 1566
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v112, v114, v112 div:2
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_subrev_f32_e64 v115, v113, v115 div:2
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_e64 v113, v114, v113 div:2
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
v_fma_f32 v114, v114, 1.0, -v113
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_fmac_f32_dpp v112, v112, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v113, v113, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_fmac_f32_dpp v114, v114, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_fmac_f32_dpp v115, v115, v176 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
s_barrier
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x3
buffer_load_dword v148, v102, s[40:43], 0 idxen
buffer_load_dword v150, v152, s[40:43], 0 idxen
buffer_load_dword v149, v103, s[40:43], 0 idxen
buffer_load_dword v151, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v104
ds_read_b128 v[78:81], v171 offset:25280
ds_write_b32 v164, v105
ds_read_b128 v[82:85], v171 offset:25536
ds_write_b32 v165, v106
ds_read_b128 v[94:97], v162 offset:24768
ds_write_b32 v166, v107
ds_read_b128 v[98:101], v162 offset:24896
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 64102
s_call_b64 s[4:5], 1445
s_branch 64100
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_dpp v116, v117, v117 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_fmac_f32_dpp v116, v117, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_dpp v179, v119, v119 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
s_setprio 1
v_fmac_f32_dpp v179, v119, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_add_f32_dpp v119, v118, v118 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v119, v118, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_add_f32_e32 v118, v116, v119
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_add_f32_e64 v117, v179, v118 div:2
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
v_add_f32_e64 v118, -v179, v118 div:2
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x2
buffer_load_dword v106, v68, s[40:43], 0 idxen
buffer_load_dword v105, v3, s[40:43], 0 idxen
buffer_load_dword v107, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v108
ds_read_b128 v[70:73], v171 offset:29440
ds_write_b32 v168, v109
ds_read_b128 v[74:77], v171 offset:29696
ds_write_b32 v169, v110
ds_read_b128 v[86:89], v162 offset:28928
ds_write_b32 v170, v111
ds_read_b128 v[90:93], v162 offset:29056
s_waitcnt vmcnt(21) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 1326
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_dpp v120, v121, v121 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_fmac_f32_dpp v120, v121, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_dpp v179, v123, v123 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
s_setprio 1
v_fmac_f32_dpp v179, v123, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_add_f32_dpp v123, v122, v122 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v123, v122, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_add_f32_e32 v122, v120, v123
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_add_f32_e64 v121, v179, v122 div:2
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
v_add_f32_e64 v122, -v179, v122 div:2
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x2
buffer_load_dword v110, v152, s[40:43], 0 idxen
buffer_load_dword v109, v103, s[40:43], 0 idxen
buffer_load_dword v111, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v112 offset:8256
ds_read_b128 v[78:81], v171 offset:33536
ds_write_b32 v164, v113 offset:8256
ds_read_b128 v[82:85], v171 offset:33792
ds_write_b32 v165, v114 offset:8256
ds_read_b128 v[94:97], v162 offset:33024
ds_write_b32 v166, v115 offset:8256
ds_read_b128 v[98:101], v162 offset:33152
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 1206
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_dpp v124, v125, v125 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_fmac_f32_dpp v124, v125, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_dpp v179, v127, v127 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
s_setprio 1
v_fmac_f32_dpp v179, v127, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_add_f32_dpp v127, v126, v126 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v127, v126, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_add_f32_e32 v126, v124, v127
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_add_f32_e64 v125, v179, v126 div:2
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
v_add_f32_e64 v126, -v179, v126 div:2
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x2
buffer_load_dword v114, v68, s[40:43], 0 idxen
buffer_load_dword v113, v3, s[40:43], 0 idxen
buffer_load_dword v115, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v116 offset:8256
ds_read_b128 v[70:73], v171 offset:37696
ds_write_b32 v168, v117 offset:8256
ds_read_b128 v[74:77], v171 offset:37952
ds_write_b32 v169, v118 offset:8256
ds_read_b128 v[86:89], v162 offset:37184
ds_write_b32 v170, v119 offset:8256
ds_read_b128 v[90:93], v162 offset:37312
s_waitcnt vmcnt(21) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 1086
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_barrier
v_add_f32_dpp v128, v129, v129 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_fmac_f32_dpp v128, v129, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_dpp v179, v131, v131 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
s_setprio 1
v_fmac_f32_dpp v179, v131, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_add_f32_dpp v131, v130, v130 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v131, v130, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_add_f32_e32 v130, v128, v131
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_add_f32_e64 v129, v179, v130 div:2
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
v_add_f32_e64 v130, -v179, v130 div:2
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x2
buffer_load_dword v118, v152, s[40:43], 0 idxen
buffer_load_dword v117, v103, s[40:43], 0 idxen
buffer_load_dword v119, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v120 offset:16512
ds_read_b128 v[78:81], v171 offset:41792
ds_write_b32 v164, v121 offset:16512
ds_read_b128 v[82:85], v171 offset:42048
ds_write_b32 v165, v122 offset:16512
ds_read_b128 v[94:97], v162 offset:41280
ds_write_b32 v166, v123 offset:16512
ds_read_b128 v[98:101], v162 offset:41408
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 6
s_call_b64 s[4:5], 965
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_dpp v132, v133, v133 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_fmac_f32_dpp v132, v133, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_dpp v179, v135, v135 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
s_setprio 1
v_fmac_f32_dpp v179, v135, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_add_f32_dpp v135, v134, v134 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v135, v134, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_add_f32_e32 v134, v132, v135
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_add_f32_e64 v133, v179, v134 div:2
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
v_add_f32_e64 v134, -v179, v134 div:2
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x2
buffer_load_dword v122, v68, s[40:43], 0 idxen
buffer_load_dword v121, v3, s[40:43], 0 idxen
buffer_load_dword v123, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v124 offset:16512
ds_read_b128 v[70:73], v171 offset:45952
ds_write_b32 v168, v125 offset:16512
ds_read_b128 v[74:77], v171 offset:46208
ds_write_b32 v169, v126 offset:16512
ds_read_b128 v[86:89], v162 offset:45440
ds_write_b32 v170, v127 offset:16512
ds_read_b128 v[90:93], v162 offset:45568
s_waitcnt vmcnt(21) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 846
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_dpp v136, v137, v137 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_fmac_f32_dpp v136, v137, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_dpp v179, v139, v139 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
s_setprio 1
v_fmac_f32_dpp v179, v139, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_add_f32_dpp v139, v138, v138 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v139, v138, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_add_f32_e32 v138, v136, v139
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_add_f32_e64 v137, v179, v138 div:2
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
v_add_f32_e64 v138, -v179, v138 div:2
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x2
buffer_load_dword v126, v152, s[40:43], 0 idxen
buffer_load_dword v125, v103, s[40:43], 0 idxen
buffer_load_dword v127, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v128 offset:24768
ds_read_b128 v[78:81], v171 offset:512
ds_write_b32 v164, v129 offset:24768
ds_read_b128 v[82:85], v171 offset:768
ds_write_b32 v165, v130 offset:24768
ds_read_b128 v[94:97], v162
ds_write_b32 v166, v131 offset:24768
ds_read_b128 v[98:101], v162 offset:128
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 726
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_dpp v140, v141, v141 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_fmac_f32_dpp v140, v141, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_dpp v179, v143, v143 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
s_setprio 1
v_fmac_f32_dpp v179, v143, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_add_f32_dpp v143, v142, v142 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v143, v142, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_add_f32_e32 v142, v140, v143
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_add_f32_e64 v141, v179, v142 div:2
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
v_add_f32_e64 v142, -v179, v142 div:2
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x2
buffer_load_dword v130, v68, s[40:43], 0 idxen
buffer_load_dword v129, v3, s[40:43], 0 idxen
buffer_load_dword v131, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v132 offset:24768
ds_read_b128 v[70:73], v171 offset:4672
ds_write_b32 v168, v133 offset:24768
ds_read_b128 v[74:77], v171 offset:4928
ds_write_b32 v169, v134 offset:24768
ds_read_b128 v[86:89], v162 offset:4160
ds_write_b32 v170, v135 offset:24768
ds_read_b128 v[90:93], v162 offset:4288
s_waitcnt vmcnt(21) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 606
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_barrier
v_add_f32_dpp v144, v145, v145 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_fmac_f32_dpp v144, v145, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_dpp v179, v147, v147 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
s_setprio 1
v_fmac_f32_dpp v179, v147, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_add_f32_dpp v147, v146, v146 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v147, v146, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_add_f32_e32 v146, v144, v147
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_add_f32_e64 v145, v179, v146 div:2
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
v_add_f32_e64 v146, -v179, v146 div:2
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x2
buffer_load_dword v134, v152, s[40:43], 0 idxen
buffer_load_dword v133, v103, s[40:43], 0 idxen
buffer_load_dword v135, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v136 offset:33024
ds_read_b128 v[78:81], v171 offset:8768
ds_write_b32 v164, v137 offset:33024
ds_read_b128 v[82:85], v171 offset:9024
ds_write_b32 v165, v138 offset:33024
ds_read_b128 v[94:97], v162 offset:8256
ds_write_b32 v166, v139 offset:33024
ds_read_b128 v[98:101], v162 offset:8384
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 6
s_call_b64 s[4:5], 485
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_dpp v148, v149, v149 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_fmac_f32_dpp v148, v149, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_dpp v179, v151, v151 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
s_setprio 1
v_fmac_f32_dpp v179, v151, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_add_f32_dpp v151, v150, v150 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v151, v150, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_add_f32_e32 v150, v148, v151
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_add_f32_e64 v149, v179, v150 div:2
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
v_add_f32_e64 v150, -v179, v150 div:2
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x2
buffer_load_dword v138, v68, s[40:43], 0 idxen
buffer_load_dword v137, v3, s[40:43], 0 idxen
buffer_load_dword v139, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v140 offset:33024
ds_read_b128 v[70:73], v171 offset:12928
ds_write_b32 v168, v141 offset:33024
ds_read_b128 v[74:77], v171 offset:13184
ds_write_b32 v169, v142 offset:33024
ds_read_b128 v[86:89], v162 offset:12416
ds_write_b32 v170, v143 offset:33024
ds_read_b128 v[90:93], v162 offset:12544
s_waitcnt vmcnt(21) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 366
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_dpp v104, v105, v105 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_fmac_f32_dpp v104, v105, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_dpp v179, v107, v107 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
s_setprio 1
v_fmac_f32_dpp v179, v107, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_add_f32_dpp v107, v106, v106 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v107, v106, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_add_f32_e32 v106, v104, v107
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_add_f32_e64 v105, v179, v106 div:2
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
v_add_f32_e64 v106, -v179, v106 div:2
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x2
buffer_load_dword v142, v152, s[40:43], 0 idxen
buffer_load_dword v141, v103, s[40:43], 0 idxen
buffer_load_dword v143, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v144 offset:41280
ds_read_b128 v[78:81], v171 offset:17024
ds_write_b32 v164, v145 offset:41280
ds_read_b128 v[82:85], v171 offset:17280
ds_write_b32 v165, v146 offset:41280
ds_read_b128 v[94:97], v162 offset:16512
ds_write_b32 v166, v147 offset:41280
ds_read_b128 v[98:101], v162 offset:16640
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 246
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_dpp v108, v109, v109 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v70, v86
v_fmac_f32_e32 v5, v71, v86
v_fmac_f32_e32 v6, v72, v86
v_fmac_f32_e32 v7, v73, v86
v_fmac_f32_e32 v8, v74, v86
v_fmac_f32_e32 v9, v75, v86
v_fmac_f32_e32 v10, v76, v86
v_fmac_f32_e32 v11, v77, v86
v_fmac_f32_dpp v108, v109, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v70, v87
v_fmac_f32_e32 v13, v71, v87
v_fmac_f32_e32 v14, v72, v87
v_fmac_f32_e32 v15, v73, v87
v_fmac_f32_e32 v16, v74, v87
v_fmac_f32_e32 v17, v75, v87
v_fmac_f32_e32 v18, v76, v87
v_fmac_f32_e32 v19, v77, v87
v_add_f32_dpp v179, v111, v111 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v70, v88
v_fmac_f32_e32 v21, v71, v88
v_fmac_f32_e32 v22, v72, v88
v_fmac_f32_e32 v23, v73, v88
v_fmac_f32_e32 v24, v74, v88
v_fmac_f32_e32 v25, v75, v88
v_fmac_f32_e32 v26, v76, v88
v_fmac_f32_e32 v27, v77, v88
s_setprio 1
v_fmac_f32_dpp v179, v111, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v70, v89
v_fmac_f32_e32 v29, v71, v89
v_fmac_f32_e32 v30, v72, v89
v_fmac_f32_e32 v31, v73, v89
v_fmac_f32_e32 v32, v74, v89
v_fmac_f32_e32 v33, v75, v89
v_fmac_f32_e32 v34, v76, v89
v_fmac_f32_e32 v35, v77, v89
v_add_f32_dpp v111, v110, v110 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v70, v90
v_fmac_f32_e32 v37, v71, v90
v_fmac_f32_e32 v38, v72, v90
v_fmac_f32_e32 v39, v73, v90
v_fmac_f32_e32 v40, v74, v90
v_fmac_f32_e32 v41, v75, v90
v_fmac_f32_e32 v42, v76, v90
v_fmac_f32_e32 v43, v77, v90
v_fmac_f32_dpp v111, v110, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v70, v91
v_fmac_f32_e32 v45, v71, v91
v_fmac_f32_e32 v46, v72, v91
v_fmac_f32_e32 v47, v73, v91
v_fmac_f32_e32 v48, v74, v91
v_fmac_f32_e32 v49, v75, v91
v_fmac_f32_e32 v50, v76, v91
v_fmac_f32_e32 v51, v77, v91
v_add_f32_e32 v110, v108, v111
v_fmac_f32_e32 v52, v70, v92
v_fmac_f32_e32 v53, v71, v92
v_fmac_f32_e32 v54, v72, v92
v_fmac_f32_e32 v55, v73, v92
v_fmac_f32_e32 v56, v74, v92
v_fmac_f32_e32 v57, v75, v92
v_fmac_f32_e32 v58, v76, v92
v_fmac_f32_e32 v59, v77, v92
v_add_f32_e64 v109, v179, v110 div:2
v_fmac_f32_e32 v60, v70, v93
v_fmac_f32_e32 v61, v71, v93
v_fmac_f32_e32 v62, v72, v93
v_fmac_f32_e32 v63, v73, v93
v_fmac_f32_e32 v64, v74, v93
v_fmac_f32_e32 v65, v75, v93
v_fmac_f32_e32 v66, v76, v93
v_fmac_f32_e32 v67, v77, v93
v_add_f32_e64 v110, -v179, v110 div:2
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_clause 0x2
buffer_load_dword v146, v68, s[40:43], 0 idxen
buffer_load_dword v145, v3, s[40:43], 0 idxen
buffer_load_dword v147, v69, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v167, v148 offset:41280
ds_read_b128 v[70:73], v171 offset:21184
ds_write_b32 v168, v149 offset:41280
ds_read_b128 v[74:77], v171 offset:21440
ds_write_b32 v169, v150 offset:41280
ds_read_b128 v[86:89], v162 offset:20672
ds_write_b32 v170, v151 offset:41280
ds_read_b128 v[90:93], v162 offset:20800
s_waitcnt vmcnt(21) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 126
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_barrier
v_add_f32_dpp v112, v113, v113 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v4, v78, v94
v_fmac_f32_e32 v5, v79, v94
v_fmac_f32_e32 v6, v80, v94
v_fmac_f32_e32 v7, v81, v94
v_fmac_f32_e32 v8, v82, v94
v_fmac_f32_e32 v9, v83, v94
v_fmac_f32_e32 v10, v84, v94
v_fmac_f32_e32 v11, v85, v94
v_fmac_f32_dpp v112, v113, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v12, v78, v95
v_fmac_f32_e32 v13, v79, v95
v_fmac_f32_e32 v14, v80, v95
v_fmac_f32_e32 v15, v81, v95
v_fmac_f32_e32 v16, v82, v95
v_fmac_f32_e32 v17, v83, v95
v_fmac_f32_e32 v18, v84, v95
v_fmac_f32_e32 v19, v85, v95
v_add_f32_dpp v179, v115, v115 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v20, v78, v96
v_fmac_f32_e32 v21, v79, v96
v_fmac_f32_e32 v22, v80, v96
v_fmac_f32_e32 v23, v81, v96
v_fmac_f32_e32 v24, v82, v96
v_fmac_f32_e32 v25, v83, v96
v_fmac_f32_e32 v26, v84, v96
v_fmac_f32_e32 v27, v85, v96
s_setprio 1
v_fmac_f32_dpp v179, v115, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v28, v78, v97
v_fmac_f32_e32 v29, v79, v97
v_fmac_f32_e32 v30, v80, v97
v_fmac_f32_e32 v31, v81, v97
v_fmac_f32_e32 v32, v82, v97
v_fmac_f32_e32 v33, v83, v97
v_fmac_f32_e32 v34, v84, v97
v_fmac_f32_e32 v35, v85, v97
v_add_f32_dpp v115, v114, v114 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v36, v78, v98
v_fmac_f32_e32 v37, v79, v98
v_fmac_f32_e32 v38, v80, v98
v_fmac_f32_e32 v39, v81, v98
v_fmac_f32_e32 v40, v82, v98
v_fmac_f32_e32 v41, v83, v98
v_fmac_f32_e32 v42, v84, v98
v_fmac_f32_e32 v43, v85, v98
v_fmac_f32_dpp v115, v114, v176 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v44, v78, v99
v_fmac_f32_e32 v45, v79, v99
v_fmac_f32_e32 v46, v80, v99
v_fmac_f32_e32 v47, v81, v99
v_fmac_f32_e32 v48, v82, v99
v_fmac_f32_e32 v49, v83, v99
v_fmac_f32_e32 v50, v84, v99
v_fmac_f32_e32 v51, v85, v99
v_add_f32_e32 v114, v112, v115
v_fmac_f32_e32 v52, v78, v100
v_fmac_f32_e32 v53, v79, v100
v_fmac_f32_e32 v54, v80, v100
v_fmac_f32_e32 v55, v81, v100
v_fmac_f32_e32 v56, v82, v100
v_fmac_f32_e32 v57, v83, v100
v_fmac_f32_e32 v58, v84, v100
v_fmac_f32_e32 v59, v85, v100
v_add_f32_e64 v113, v179, v114 div:2
v_fmac_f32_e32 v60, v78, v101
v_fmac_f32_e32 v61, v79, v101
v_fmac_f32_e32 v62, v80, v101
v_fmac_f32_e32 v63, v81, v101
v_fmac_f32_e32 v64, v82, v101
v_fmac_f32_e32 v65, v83, v101
v_fmac_f32_e32 v66, v84, v101
v_fmac_f32_e32 v67, v85, v101
v_add_f32_e64 v114, -v179, v114 div:2
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_clause 0x2
buffer_load_dword v150, v152, s[40:43], 0 idxen
buffer_load_dword v149, v103, s[40:43], 0 idxen
buffer_load_dword v151, v153, s[40:43], 0 idxen
s_clause 0x7
ds_write_b32 v163, v104
ds_read_b128 v[78:81], v171 offset:25280
ds_write_b32 v164, v105
ds_read_b128 v[82:85], v171 offset:25536
ds_write_b32 v165, v106
ds_read_b128 v[94:97], v162 offset:24768
ds_write_b32 v166, v107
ds_read_b128 v[98:101], v162 offset:24896
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 64102
s_call_b64 s[4:5], 5
s_branch 64100
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_nop
s_cmp_eq_u32 s62, 0
s_cbranch_scc0 6
s_branch 724
s_bitcmp1_b32 s18, 26
s_cselect_b32 s88, s49, s50
s_cselect_b32 s89, 0, s51
s_sub_u32 s40, s40, s88
s_subb_u32 s41, s41, s89
s_cmp_eq_u32 s73, 0
s_cbranch_scc0 5
s_cbranch_scc1 732
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_min_u32 s52, s62, s73
s_sub_u32 s62, s62, s52
s_sub_u32 s73, s73, s52
s_sub_u32 s52, s52, 1
s_setpc_b64 s[4:5]
s_nop 0
s_nop 0
s_nop 0
s_bitcmp1_b32 s18, 17
s_cbranch_scc1 253
s_add_u32 s68, s68, s17
s_cmp_eq_u32 s68, 0
s_cbranch_scc1 250
s_mov_b32 s69, 0
s_bitcmp1_b32 s18, 16
s_cbranch_scc1 239
s_add_u32 s67, s16, 31
s_lshr_b32 s67, s67, 5
v_mov_b32_e32 v182, s68
v_mul_u32_u24_e32 v182, s67, v182
v_add_co_u32 v182, vcc, s17, v182
v_sub_co_u32 v182, vcc, v182, 1
v_ffbh_u32_e32 v186, s17
v_lshlrev_b32_e64 v187, v186, s17
v_and_b32_e32 v185, 0xffffff00, v187
v_cmp_eq_u32_e32 vcc, 0x80000000, v187
v_cvt_f32_u32_e32 v185, v185
v_rcp_f32_e32 v181, v185
v_sub_co_ci_u32_e32 v184, vcc, 32, v186, vcc
v_cvt_f32_ubyte0_e32 v186, v187
v_fma_f32 v185, v185, v181, -1.0
v_fma_f32 v185, v186, v181, v185
v_fmaak_f32 v185, v185, v181, 0x9f000000
v_mul_f32_e32 v185, 0x5f800000, v185
v_mov_b32_e32 v186, 0
v_cvt_flr_i32_f32_e64 v185, -v185
v_lshl_add_u32 v181, v181, 9, v185
v_mad_u64_u32 v[186:187], vcc, v187, v181, v[186:187]
v_sub_co_ci_u32_e64 v181, vcc, v181, -1, vcc
v_mul_hi_u32 v185, v182, v181
v_add_co_u32 v181, vcc, v185, v182
v_add_co_ci_u32_e64 v185, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v184
v_cndmask_b32_e32 v181, v181, v185, vcc
v_alignbit_b32 v181, v185, v181, v184
s_nop 0
v_readfirstlane_b32 s66, v181
v_mul_u32_u24_e64 v181, v181, s8
v_ffbh_u32_e32 v186, s67
v_lshlrev_b32_e64 v187, v186, s67
v_and_b32_e32 v185, 0xffffff00, v187
v_cmp_eq_u32_e32 vcc, 0x80000000, v187
v_cvt_f32_u32_e32 v185, v185
v_rcp_f32_e32 v182, v185
v_sub_co_ci_u32_e32 v184, vcc, 32, v186, vcc
v_cvt_f32_ubyte0_e32 v186, v187
v_fma_f32 v185, v185, v182, -1.0
v_fma_f32 v185, v186, v182, v185
v_fmaak_f32 v185, v185, v182, 0x9f000000
v_mul_f32_e32 v185, 0x5f800000, v185
v_mov_b32_e32 v186, 0
v_cvt_flr_i32_f32_e64 v185, -v185
v_lshl_add_u32 v182, v182, 9, v185
v_mad_u64_u32 v[186:187], vcc, v187, v182, v[186:187]
v_sub_co_ci_u32_e64 v182, vcc, v182, -1, vcc
v_mul_hi_u32 v185, v181, v182
v_add_co_u32 v182, vcc, v185, v181
v_add_co_ci_u32_e64 v185, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v184
v_cndmask_b32_e32 v182, v182, v185, vcc
v_alignbit_b32 v182, v185, v182, v184
v_readfirstlane_b32 s77, v181
v_readfirstlane_b32 s64, v182
s_mul_i32 s64, s64, s67
s_sub_u32 s64, s77, s64
v_sub_co_u32 v182, vcc, s8, v182
v_sub_co_u32 v182, vcc, s17, v182
v_and_b32_e64 v184, v1, 63
v_cmp_eq_u32_e64 vcc, v184, 0
v_cndmask_b32_e32 v182, 1, v182, vcc
s_sub_u32 s78, 0, s55
s_sub_u32 s79, 0, s54
v_mul_u32_u24_e64 v186, v182, 32
v_ffbh_u32_e32 v188, s78
v_lshlrev_b32_e64 v189, v188, s78
v_and_b32_e32 v190, 0xffffff00, v189
v_cmp_eq_u32_e32 vcc, 0x80000000, v189
v_cvt_f32_u32_e32 v190, v190
v_rcp_f32_e32 v184, v190
v_sub_co_ci_u32_e32 v187, vcc, 32, v188, vcc
v_cvt_f32_ubyte0_e32 v188, v189
v_fma_f32 v190, v190, v184, -1.0
v_fma_f32 v190, v188, v184, v190
v_fmaak_f32 v190, v190, v184, 0x9f000000
v_mul_f32_e32 v190, 0x5f800000, v190
v_mov_b32_e32 v188, 0
v_cvt_flr_i32_f32_e64 v190, -v190
v_lshl_add_u32 v184, v184, 9, v190
v_mad_u64_u32 v[188:189], vcc, v189, v184, v[188:189]
v_sub_co_ci_u32_e64 v184, vcc, v184, -1, vcc
v_mul_hi_u32 v188, v186, v184
v_add_co_u32 v184, vcc, v188, v186
v_add_co_ci_u32_e64 v188, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v187
v_cndmask_b32_e32 v184, v184, v188, vcc
v_alignbit_b32 v184, v188, v184, v187
v_mad_i32_i24 v185, v184, s55, v186
v_mul_u32_u24_e64 v186, v184, 1
v_ffbh_u32_e32 v188, s79
v_lshlrev_b32_e64 v189, v188, s79
v_and_b32_e32 v190, 0xffffff00, v189
v_cmp_eq_u32_e32 vcc, 0x80000000, v189
v_cvt_f32_u32_e32 v190, v190
v_rcp_f32_e32 v184, v190
v_sub_co_ci_u32_e32 v187, vcc, 32, v188, vcc
v_cvt_f32_ubyte0_e32 v188, v189
v_fma_f32 v190, v190, v184, -1.0
v_fma_f32 v190, v188, v184, v190
v_fmaak_f32 v190, v190, v184, 0x9f000000
v_mul_f32_e32 v190, 0x5f800000, v190
v_mov_b32_e32 v188, 0
v_cvt_flr_i32_f32_e64 v190, -v190
v_lshl_add_u32 v184, v184, 9, v190
v_mad_u64_u32 v[188:189], vcc, v189, v184, v[188:189]
v_sub_co_ci_u32_e64 v184, vcc, v184, -1, vcc
v_mul_hi_u32 v188, v186, v184
v_add_co_u32 v184, vcc, v188, v186
v_add_co_ci_u32_e64 v188, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v187
v_cndmask_b32_e32 v184, v184, v188, vcc
v_alignbit_b32 v184, v188, v184, v187
v_mad_i32_i24 v186, v184, s54, v186
v_readfirstlane_b32 s56, v185
v_readfirstlane_b32 s57, v186
v_readfirstlane_b32 s58, v184
v_add_co_u32 v172, vcc, s56, v172
v_add_co_ci_u32_e64 v187, vcc, 0, 0, vcc
v_mad_i32_i24 v172, v187, s55, v172
v_mad_i32_i24 v174, v187, s60, v174
v_mad_i32_i24 v173, v187, s59, v173
v_cmp_ge_i32_e64 vcc, v173, 0
v_add_co_ci_u32_e64 v187, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v187
v_mad_i32_i24 v173, v187, s54, v173
v_add_co_u32 v173, vcc, s57, v173
v_add_co_ci_u32_e64 v187, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v187
v_mad_i32_i24 v173, v187, s54, v173
v_add_co_u32 v174, vcc, s58, v174
v_readlane_b32 s56, v185, 1
v_readlane_b32 s57, v186, 1
v_readlane_b32 s58, v184, 1
s_add_u32 s65, s64, s66
s_cmp_le_u32 s65, s67
s_cselect_b32 s88, 0x20000, 0
s_cselect_b32 s65, s65, s67
s_or_b32 s18, s18, s88
s_lshl_b32 s64, s64, 5
s_lshl_b32 s65, s65, 5
s_min_u32 s65, s65, s16
s_cmp_eq_u32 s8, s17
s_cselect_b32 s88, 0x20000, 0
s_or_b32 s18, s18, s88
s_bitset1_b32 s18, 16
s_branch 48
s_lshr_b32 s64, s64, 5
s_add_u32 s65, s64, s66
s_sub_u32 s65, s65, s67
s_mov_b32 s64, 0
s_lshl_b32 s65, s65, 5
s_min_u32 s65, s65, s16
s_bitset1_b32 s18, 17
s_branch 12
s_bitset1_b32 s18, 18
s_mov_b32 s43, 0
s_mov_b32 s53, -1
s_mov_b32 s62, 40
s_branch 36
s_add_u32 s63, s63, 32
s_cmp_ge_u32 s63, s65
s_cbranch_scc0 33
s_bitset1_b32 s18, 22
s_sub_u32 s68, s68, s17
s_subb_u32 s69, s69, 0
s_cbranch_scc1 65269
v_add_co_u32 v172, vcc, s56, v172
v_add_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_mad_i32_i24 v172, v181, s55, v172
v_mad_i32_i24 v174, v181, s60, v174
v_mad_i32_i24 v173, v181, s59, v173
v_cmp_ge_i32_e64 vcc, v173, 0
v_add_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v181
v_mad_i32_i24 v173, v181, s54, v173
v_add_co_u32 v173, vcc, s57, v173
v_add_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v181
v_mad_i32_i24 v173, v181, s54, v173
v_add_co_u32 v174, vcc, s58, v174
s_mov_b32 s63, s64
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccz 257
v_subrev_co_u32 v181, vcc, s55, v172
v_subrev_co_u32 v182, vcc, s54, v173
s_bitcmp1_b32 s18, 22
s_cbranch_scc0 66
s_bitset0_b32 s18, 22
s_bfe_u32 s77, s18, 0x10014
v_mul_u32_u24_e32 v184, 2, v181
v_mul_u32_u24_e32 v185, 2, v182
v_cvt_pk_u16_u32 v187, v184, v185
v_and_b32_e64 v184, v1, 1
v_cmp_eq_u32_e64 vcc, v184, 1
v_cndmask_b32_e32 v187, v174, v187, vcc
v_lshrrev_b32_e32 v183, 1, v1
v_bfe_u32 v188, v183, s77, 1
v_lshrrev_b32_e32 v183, 1, v1
v_bfi_b32 v183, 1, v1, v183
v_lshrrev_b32_e32 v184, 2, v1
v_bfi_b32 v184, 1, v1, v184
v_cmp_eq_u32_e64 vcc, s77, 0
v_cndmask_b32_e32 v183, v184, v183, vcc
s_sub_u32 s77, 1, s77
v_lshrrev_b32_e32 v184, s77, v183
v_bfi_b32 v183, 32, v184, v183
v_and_b32_e32 v183, 63, v183
v_add_co_u32 v184, vcc, 16, v183
v_and_b32_e64 v185, v1, 2
v_cmp_eq_u32_e64 vcc, v185, 0
v_cndmask_b32_e32 v184, v184, v183, vcc
v_lshlrev_b32_e32 v185, 14, v188
v_mad_u32_u24 v184, 4, v184, v185
v_add_co_u32 v183, vcc, s75, v184
ds_write_b32 v183, v187
v_writelane_b32 v185, s18, 0
v_writelane_b32 v185, s65, 1
v_writelane_b32 v185, s64, 2
v_and_b32_e64 v183, v1, 63
v_cmp_ge_u32_e64 vcc, v183, 3
v_mov_b32_e32 v186, 0x4000
v_cndmask_b32_e32 v183, v183, v186, vcc
v_mad_i32_i24 v183, v183, 4, s75
ds_write_b32 v183, v185 offset:256
s_add_u32 s75, s75, 0x18c
s_cmp_eq_u32 s75, 0x10000
s_cselect_b32 s75, 0xc220, s75
v_mov_b32_dpp v183, v174 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v181, v181 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v182, v182 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_readfirstlane_b32 s61, v183
v_sub_co_u32 v184, vcc, v183, s61
v_mul_lo_u32 v184, v184, s44
v_and_b32_e64 v188, v1, 3
v_ashrrev_i32_e64 v189, 0, s31
v_subrev_co_u32 v188, vcc, v189, v188
v_ashrrev_i32_e64 v189, 0, s11
v_mad_i32_i24 v185, v189, 3, v188
s_bfe_u32 s77, s18, 0x10014
v_lshrrev_b32_e32 v187, 2, v1
v_and_b32_e32 v187, s77, v187
v_mad_i32_i24 v185, v187, 3, v185
v_add_co_u32 v186, vcc, 0, s38
v_ashrrev_i32_e32 v186, 0, v186
v_add_co_u32 v187, vcc, 0, s30
v_ashrrev_i32_e32 v187, 0, v187
v_sub_nc_i32 v186, v186, v187
v_cmp_ge_u32_e64 s[78:79], v183, s12
v_mad_i32_i24 v181, v181, 2, v185
v_cmp_ge_u32_e64 s[92:93], v181, s15
v_add_co_u32 v181, vcc, v181, v184
s_or_b64 s[92:93], s[92:93], s[78:79]
v_mad_i32_i24 v182, v182, 2, v186
v_cmp_ge_u32_e64 s[94:95], v182, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v2, v182, s15, v181
v_cndmask_b32_e64 v2, v2, -1, s[94:95]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[94:95], v182, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v3, v182, s15, v181
v_cndmask_b32_e64 v3, v3, -1, s[94:95]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[94:95], v182, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v68, v182, s15, v181
v_cndmask_b32_e64 v68, v68, -1, s[94:95]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[94:95], v182, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v69, v182, s15, v181
v_cndmask_b32_e64 v69, v69, -1, s[94:95]
s_bitcmp1_b32 s18, 20
s_cbranch_scc0 60
v_mov_b32_dpp v183, v174 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v181, v172 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v182, v173 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[78:79], v183, s12
v_sub_co_u32 v184, vcc, v183, s61
v_mul_lo_u32 v184, v184, s44
v_sub_co_u32 v181, vcc, v181, s55
v_sub_co_u32 v182, vcc, v182, s54
v_mad_i32_i24 v181, v181, 2, v185
v_cmp_ge_u32_e64 s[92:93], v181, s15
v_add_co_u32 v181, vcc, v181, v184
s_or_b64 s[92:93], s[92:93], s[78:79]
v_mad_i32_i24 v182, v182, 2, v186
v_cmp_ge_u32_e64 s[94:95], v182, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v102, v182, s15, v181
v_cndmask_b32_e64 v102, v102, -1, s[94:95]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[94:95], v182, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v103, v182, s15, v181
v_cndmask_b32_e64 v103, v103, -1, s[94:95]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[94:95], v182, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v152, v182, s15, v181
v_cndmask_b32_e64 v152, v152, -1, s[94:95]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[94:95], v182, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v153, v182, s15, v181
v_cndmask_b32_e64 v153, v153, -1, s[94:95]
s_branch 26
s_bitcmp1_b32 s18, 24
s_cselect_b32 s77, s48, 0
v_add_co_u32 v187, vcc, v2, s77
v_cmp_eq_u32_e64 vcc, v2, -1
v_cndmask_b32_e64 v102, v187, -1, vcc
v_add_co_u32 v187, vcc, v3, s77
v_cmp_eq_u32_e64 vcc, v3, -1
v_cndmask_b32_e64 v103, v187, -1, vcc
v_add_co_u32 v187, vcc, v68, s77
v_cmp_eq_u32_e64 vcc, v68, -1
v_cndmask_b32_e64 v152, v187, -1, vcc
v_add_co_u32 v187, vcc, v69, s77
v_cmp_eq_u32_e64 vcc, v69, -1
v_cndmask_b32_e64 v153, v187, -1, vcc
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 155
s_lshr_b32 s79, -1, 16
s_and_b32 s79, s79, s44
s_lshr_b32 s92, s44, 16
s_mul_i32 s92, s92, s61
s_mul_i32 s40, s79, s61
s_lshl_b32 s79, s92, 16
s_lshr_b32 s92, s92, 16
s_add_u32 s40, s79, s40
s_addc_u32 s41, s92, 0
s_lshl_b64 s[40:41], s[40:41], 2
s_add_u32 s40, s40, s20
s_addc_u32 s41, s41, s21
s_add_u32 s41, s41, 0x40000
s_branch 128
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 138
s_bfe_u32 s77, s18, 0x10014
v_bfe_u32 v181, v1, 0, 2
v_min_u32_e32 v181, 2, v181
v_bfe_u32 v183, v1, 2, s77
v_mad_u32_u24 v181, v183, 3, v181
v_mad_u32_u24 v181, s11, 3, v181
v_sub_co_u32 v183, vcc, s29, v181
v_sub_co_u32 v183, vcc, v183, 1
s_bfe_u32 s77, s18, 0x10001
v_cmp_eq_u32_e64 vcc, s77, 1
v_cndmask_b32_e32 v181, v181, v183, vcc
v_cmp_ge_u32_e64 s[78:79], v181, s29
s_bfe_u32 s77, s18, 0x10018
v_bfe_u32 v184, v1, 2, s77
v_mul_lo_u32 v184, s48, v184
v_add_co_u32 v181, vcc, v181, v184
v_mul_lo_u32 v182, s70, v175
v_add_co_u32 v182, vcc, v182, v181
s_sub_u32 s77, s28, s38
s_sub_u32 s77, s77, 3
s_bitcmp1_b32 s18, 0
s_cselect_b32 s77, s77, s38
v_mov_b32_e32 v184, s77
v_cmp_ge_u32_e64 s[92:93], v184, s28
v_mad_i32_i24 v2, v184, s29, v182
s_or_b64 s[92:93], s[92:93], s[78:79]
v_cndmask_b32_e64 v2, v2, -1, s[92:93]
v_mov_b32_e32 v3, v2
v_add_co_u32 v184, vcc, v184, 1
v_cmp_ge_u32_e64 s[92:93], v184, s28
v_mad_i32_i24 v69, v184, s29, v182
s_or_b64 s[92:93], s[92:93], s[78:79]
v_cndmask_b32_e64 v69, v69, -1, s[92:93]
v_add_co_u32 v184, vcc, v184, 1
v_cmp_ge_u32_e64 s[92:93], v184, s28
v_mad_i32_i24 v68, v184, s29, v182
s_or_b64 s[92:93], s[92:93], s[78:79]
v_cndmask_b32_e64 v68, v68, -1, s[92:93]
s_lshl_b32 s92, s70, 3
s_and_b32 s93, s18, 0x1100000
s_cselect_b32 s92, s92, 0
v_add_co_u32 v181, vcc, v2, s92
v_cmp_eq_u32_e64 vcc, v2, -1
v_cndmask_b32_e64 v102, v181, -1, vcc
v_add_co_u32 v181, vcc, v3, s92
v_cmp_eq_u32_e64 vcc, v3, -1
v_cndmask_b32_e64 v103, v181, -1, vcc
v_add_co_u32 v181, vcc, v68, s92
v_cmp_eq_u32_e64 vcc, v68, -1
v_cndmask_b32_e64 v152, v181, -1, vcc
v_add_co_u32 v181, vcc, v69, s92
v_cmp_eq_u32_e64 vcc, v69, -1
v_cndmask_b32_e64 v153, v181, -1, vcc
v_add_co_u32 v181, vcc, v175, s63
v_cmp_lt_u32_e64 vcc, v181, s16
v_cndmask_b32_e32 v2, -1, v2, vcc
v_cndmask_b32_e32 v3, -1, v3, vcc
v_cndmask_b32_e32 v68, -1, v68, vcc
v_cndmask_b32_e32 v69, -1, v69, vcc
s_and_b32 s77, s18, 0x1100000
s_cbranch_scc0 4
v_add_co_u32 v181, vcc, v181, 8
v_cmp_lt_u32_e64 vcc, v181, s16
v_cndmask_b32_e32 v102, -1, v102, vcc
v_cndmask_b32_e32 v103, -1, v103, vcc
v_cndmask_b32_e32 v152, -1, v152, vcc
v_cndmask_b32_e32 v153, -1, v153, vcc
s_lshr_b32 s79, -1, 16
s_and_b32 s79, s79, s70
s_lshr_b32 s92, s70, 16
s_mul_i32 s92, s92, s63
s_mul_i32 s40, s79, s63
s_lshl_b32 s79, s92, 16
s_lshr_b32 s92, s92, 16
s_add_u32 s40, s79, s40
s_addc_u32 s41, s92, 0
s_lshl_b64 s[40:41], s[40:41], 2
s_add_u32 s40, s40, s22
s_addc_u32 s41, s41, s23
s_add_u32 s41, s41, 0x40000
s_mov_b32 s43, 0x11014000
s_mov_b32 s53, -1
s_bfe_u32 s88, s18, 0x10014
s_lshl_b32 s62, s13, s88
s_bitcmp1_b32 s18, 20
s_cselect_b32 s88, 0, 0x2000000
s_bitcmp1_b32 s13, 0
s_cselect_b32 s88, s88, 0
s_xor_b32 s18, s18, s88
s_branch 64819
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_and_b32 s88, 0x900000, s18
s_subb_u32 s11, s11, 1
s_cbranch_scc0 65124
s_and_b32 s88, 0x900000, s18
s_subb_u32 s11, s10, 1
s_add_u32 s38, s38, 3
s_cmp_ge_u32 s38, s28
s_cbranch_scc0 65118
s_mov_b32 s38, 0
s_branch 65080
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_dpp v6, v6, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v7, v7, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v4, v4, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v5, v5, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v5, v6, v5 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v4, v7, v4 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v5, v5, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v4, v4, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v4, v5, v4 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v10, v10, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v11, v11, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v8, v8, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v9, v9, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v9, v10, v9 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v8, v11, v8 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v9, v9, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v8, v8, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v5, v9, v8 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v14, v14, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v15, v15, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v12, v12, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v13, v13, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v13, v14, v13 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v12, v15, v12 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v13, v13, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v12, v12, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v6, v13, v12 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v18, v18, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v19, v19, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v16, v16, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v17, v17, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v17, v18, v17 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v16, v19, v16 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v17, v17, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v16, v16, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v7, v17, v16 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v22, v22, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v23, v23, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v20, v20, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v21, v21, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v21, v22, v21 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v20, v23, v20 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v21, v21, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v20, v20, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v8, v21, v20 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v26, v26, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v27, v27, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v24, v24, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v25, v25, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v25, v26, v25 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v24, v27, v24 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v25, v25, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v24, v24, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v9, v25, v24 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v30, v30, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v31, v31, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v28, v28, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v29, v29, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v29, v30, v29 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v28, v31, v28 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v29, v29, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v28, v28, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v10, v29, v28 row_half_mirror row_mask:0xf bank_mask:0xf
s_setprio 1
v_fmac_f32_dpp v34, v34, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v35, v35, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v32, v32, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v33, v33, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v33, v34, v33 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v32, v35, v32 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v33, v33, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v32, v32, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v11, v33, v32 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v38, v38, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v39, v39, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v36, v36, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v37, v37, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v37, v38, v37 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v36, v39, v36 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v37, v37, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v36, v36, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v12, v37, v36 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v42, v42, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v43, v43, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v40, v40, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v41, v41, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v41, v42, v41 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v40, v43, v40 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v41, v41, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v40, v40, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v13, v41, v40 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v46, v46, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v47, v47, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v44, v44, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v45, v45, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v45, v46, v45 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v44, v47, v44 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v45, v45, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v44, v44, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v14, v45, v44 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v50, v50, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v51, v51, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v48, v48, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v49, v49, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v49, v50, v49 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v48, v51, v48 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v49, v49, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v48, v48, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v15, v49, v48 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v54, v54, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v55, v55, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v52, v52, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v53, v53, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v53, v54, v53 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v52, v55, v52 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v53, v53, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v52, v52, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v16, v53, v52 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v58, v58, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v59, v59, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v56, v56, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v57, v57, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v57, v58, v57 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v56, v59, v56 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v57, v57, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v56, v56, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v17, v57, v56 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v62, v62, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v63, v63, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v60, v60, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v61, v61, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v61, v62, v61 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v60, v63, v60 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v61, v61, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v60, v60, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v18, v61, v60 row_half_mirror row_mask:0xf bank_mask:0xf
v_fmac_f32_dpp v66, v66, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v67, v67, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v64, v64, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v65, v65, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v65, v66, v65 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v64, v67, v64 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v65, v65, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v64, v64, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v19, v65, v64 row_half_mirror row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(0)
v_readlane_b32 s95, v180, 0
v_add_f32_e64 v4, v4, s95
v_mul_f32_e64 v182, v4, s36
v_cmp_lt_f32_e64 vcc, v4, 0
v_cndmask_b32_e32 v4, v4, v182, vcc
v_add_f32_e64 v5, v5, s95
v_mul_f32_e64 v182, v5, s36
v_cmp_lt_f32_e64 vcc, v5, 0
v_cndmask_b32_e32 v5, v5, v182, vcc
buffer_store_dword v4, v154, s[80:83], 0 idxen
buffer_store_dword v5, v158, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v180, 1
v_add_f32_e64 v6, v6, s95
v_mul_f32_e64 v182, v6, s36
v_cmp_lt_f32_e64 vcc, v6, 0
v_cndmask_b32_e32 v6, v6, v182, vcc
v_add_f32_e64 v7, v7, s95
v_mul_f32_e64 v182, v7, s36
v_cmp_lt_f32_e64 vcc, v7, 0
v_cndmask_b32_e32 v7, v7, v182, vcc
buffer_store_dword v6, v154, s[80:83], 0 idxen
buffer_store_dword v7, v158, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v180, 2
v_add_f32_e64 v8, v8, s95
v_mul_f32_e64 v182, v8, s36
v_cmp_lt_f32_e64 vcc, v8, 0
v_cndmask_b32_e32 v8, v8, v182, vcc
v_add_f32_e64 v9, v9, s95
v_mul_f32_e64 v182, v9, s36
v_cmp_lt_f32_e64 vcc, v9, 0
v_cndmask_b32_e32 v9, v9, v182, vcc
buffer_store_dword v8, v154, s[80:83], 0 idxen
buffer_store_dword v9, v158, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v180, 3
v_add_f32_e64 v10, v10, s95
v_mul_f32_e64 v182, v10, s36
v_cmp_lt_f32_e64 vcc, v10, 0
v_cndmask_b32_e32 v10, v10, v182, vcc
v_add_f32_e64 v11, v11, s95
v_mul_f32_e64 v182, v11, s36
v_cmp_lt_f32_e64 vcc, v11, 0
v_cndmask_b32_e32 v11, v11, v182, vcc
buffer_store_dword v10, v154, s[80:83], 0 idxen
buffer_store_dword v11, v158, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
s_lshl_b32 s92, s95, 2
s_add_u32 s80, s80, s92
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 4
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v180, 8
v_add_f32_e64 v12, v12, s95
v_mul_f32_e64 v182, v12, s36
v_cmp_lt_f32_e64 vcc, v12, 0
v_cndmask_b32_e32 v12, v12, v182, vcc
v_add_f32_e64 v13, v13, s95
v_mul_f32_e64 v182, v13, s36
v_cmp_lt_f32_e64 vcc, v13, 0
v_cndmask_b32_e32 v13, v13, v182, vcc
buffer_store_dword v12, v154, s[80:83], 0 idxen
buffer_store_dword v13, v158, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v180, 9
v_add_f32_e64 v14, v14, s95
v_mul_f32_e64 v182, v14, s36
v_cmp_lt_f32_e64 vcc, v14, 0
v_cndmask_b32_e32 v14, v14, v182, vcc
v_add_f32_e64 v15, v15, s95
v_mul_f32_e64 v182, v15, s36
v_cmp_lt_f32_e64 vcc, v15, 0
v_cndmask_b32_e32 v15, v15, v182, vcc
buffer_store_dword v14, v154, s[80:83], 0 idxen
buffer_store_dword v15, v158, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v180, 10
v_add_f32_e64 v16, v16, s95
v_mul_f32_e64 v182, v16, s36
v_cmp_lt_f32_e64 vcc, v16, 0
v_cndmask_b32_e32 v16, v16, v182, vcc
v_add_f32_e64 v17, v17, s95
v_mul_f32_e64 v182, v17, s36
v_cmp_lt_f32_e64 vcc, v17, 0
v_cndmask_b32_e32 v17, v17, v182, vcc
buffer_store_dword v16, v154, s[80:83], 0 idxen
buffer_store_dword v17, v158, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v180, 11
v_add_f32_e64 v18, v18, s95
v_mul_f32_e64 v182, v18, s36
v_cmp_lt_f32_e64 vcc, v18, 0
v_cndmask_b32_e32 v18, v18, v182, vcc
v_add_f32_e64 v19, v19, s95
v_mul_f32_e64 v182, v19, s36
v_cmp_lt_f32_e64 vcc, v19, 0
v_cndmask_b32_e32 v19, v19, v182, vcc
buffer_store_dword v18, v154, s[80:83], 0 idxen
buffer_store_dword v19, v158, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
s_add_u32 s80, s80, s92
s_addc_u32 s81, s81, 0
s_lshl_b32 s92, s92, 2
s_add_u32 s80, s80, s92
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 20
s_cselect_b32 s83, 0, s83
s_cselect_b32 s87, 0, s87
s_add_u32 s84, s84, 0x80
s_addc_u32 s85, s85, 0
s_sub_u32 s86, s86, 32
s_cselect_b32 s87, 0, s87
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
v_mov_b32_e32 v36, 0
v_mov_b32_e32 v37, 0
v_mov_b32_e32 v38, 0
v_mov_b32_e32 v39, 0
v_mov_b32_e32 v40, 0
v_mov_b32_e32 v41, 0
v_mov_b32_e32 v42, 0
v_mov_b32_e32 v43, 0
v_mov_b32_e32 v44, 0
v_mov_b32_e32 v45, 0
v_mov_b32_e32 v46, 0
v_mov_b32_e32 v47, 0
v_mov_b32_e32 v48, 0
v_mov_b32_e32 v49, 0
v_mov_b32_e32 v50, 0
v_mov_b32_e32 v51, 0
v_mov_b32_e32 v52, 0
v_mov_b32_e32 v53, 0
v_mov_b32_e32 v54, 0
v_mov_b32_e32 v55, 0
v_mov_b32_e32 v56, 0
v_mov_b32_e32 v57, 0
v_mov_b32_e32 v58, 0
v_mov_b32_e32 v59, 0
v_mov_b32_e32 v60, 0
v_mov_b32_e32 v61, 0
v_mov_b32_e32 v62, 0
v_mov_b32_e32 v63, 0
v_mov_b32_e32 v64, 0
v_mov_b32_e32 v65, 0
v_mov_b32_e32 v66, 0
v_mov_b32_e32 v67, 0
s_xor_b32 s18, s18, 0x200000
s_mul_i32 s73, s9, s10
s_mul_i32 s73, s73, s13
s_add_u32 s88, s72, s71
s_cmp_lt_i32 s88, 0
s_cbranch_scc0 153
v_and_b32_e32 v154, 0x7f, v1
v_lshrrev_b32_e32 v154, 1, v154
v_bfi_b32 v154, 1, v1, v154
v_and_b32_e64 v155, v1, 2
v_mad_u32_u24 v154, v155, 16, v154
v_lshlrev_b32_e32 v154, 2, v154
v_add_co_u32 v154, vcc, v154, s76
v_and_b32_e32 v155, 3, v1
v_lshlrev_b32_e32 v155, 2, v155
v_add_co_u32 v155, vcc, v155, s76
ds_read_b32 v181, v155 offset:256
ds_read_b32 v154, v154
s_add_u32 s76, s76, 0x18c
s_cmp_eq_u32 s76, 0x10000
s_cselect_b32 s76, 0xc220, s76
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s74, v154
v_readlane_b32 s90, v181, 0
s_bitcmp1_b32 s90, 18
s_cbranch_scc1 126
v_readlane_b32 s88, v181, 1
v_readlane_b32 s89, v181, 2
s_add_u32 s72, s71, s89
s_lshr_b32 s92, -1, 16
s_and_b32 s92, s92, s45
s_lshr_b32 s93, s45, 16
s_mul_i32 s93, s93, s74
s_mul_i32 s80, s92, s74
s_lshl_b32 s92, s93, 16
s_lshr_b32 s93, s93, 16
s_add_u32 s80, s92, s80
s_addc_u32 s81, s93, 0
s_lshl_b64 s[80:81], s[80:81], 2
s_add_u32 s80, s80, s24
s_addc_u32 s81, s81, s25
s_mul_i32 s78, s46, s72
s_lshl_b32 s78, s78, 2
s_add_u32 s80, s80, s78
s_addc_u32 s81, s81, 0
s_add_u32 s81, s81, 0x40000
s_mov_b32 s83, 0x11014000
s_bitcmp1_b32 s18, 7
s_cselect_b32 s87, 0x11014000, 0
s_lshl_b32 s77, s72, 2
s_add_u32 s84, s34, s77
s_addc_u32 s85, s35, 0
s_add_u32 s85, s85, 0x40000
s_sub_u32 s86, s88, s72
s_cselect_b32 s87, 0, s87
s_sub_u32 s72, s88, s89
s_sub_u32 s72, s72, 1
s_sub_u32 s72, s72, s71
s_cselect_b32 s83, 0, s83
v_bfe_u32 v181, v154, 16, 16
v_bfe_u32 v182, v154, 0, 16
v_and_b32_e64 v183, v1, 7
v_sub_co_u32 v184, vcc, 7, v183
v_min_u32_e32 v183, v183, v184
v_bfe_u32 v184, v183, 1, 1
v_bfe_u32 v183, v183, 0, 1
v_mov_b32_dpp v181, v181 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v182, v182 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
v_add_co_u32 v181, vcc, v181, v184
v_add_co_u32 v182, vcc, v182, v183
v_mov_b32_dpp v183, v154 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[78:79], v183, s12
v_sub_co_u32 v183, vcc, v183, s74
v_mul_lo_u32 v183, v183, s45
v_mad_i32_i24 v158, v181, s33, v182
v_add_co_u32 v158, vcc, v158, v183
v_cmp_ge_u32_e64 s[96:97], v182, s33
s_or_b64 s[94:95], s[96:97], s[78:79]
v_cmp_ge_u32_e64 s[92:93], v181, s32
s_or_b64 s[78:79], s[94:95], s[92:93]
v_cndmask_b32_e64 v158, v158, -1, s[78:79]
v_bfe_u32 v181, v154, 16, 16
v_bfe_u32 v182, v154, 0, 16
v_and_b32_e64 v183, v1, 7
v_sub_co_u32 v184, vcc, 7, v183
v_min_u32_e32 v183, v183, v184
v_bfe_u32 v184, v183, 1, 1
v_bfe_u32 v183, v183, 0, 1
v_mov_b32_dpp v181, v181 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v182, v182 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_add_co_u32 v181, vcc, v181, v184
v_add_co_u32 v182, vcc, v182, v183
v_mov_b32_dpp v183, v154 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[78:79], v183, s12
v_sub_co_u32 v183, vcc, v183, s74
v_mul_lo_u32 v183, v183, s45
v_mad_i32_i24 v154, v181, s33, v182
v_add_co_u32 v154, vcc, v154, v183
v_cmp_ge_u32_e64 s[96:97], v182, s33
s_or_b64 s[94:95], s[96:97], s[78:79]
v_cmp_ge_u32_e64 s[92:93], v181, s32
s_or_b64 s[78:79], s[94:95], s[92:93]
v_cndmask_b32_e64 v154, v154, -1, s[78:79]
v_and_b32_e64 v180, v1, 63
buffer_load_dword v180, v180, s[84:87], 0 idxen
s_branch 64039
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
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_code_end
s_code_end
s_code_end
s_code_end
