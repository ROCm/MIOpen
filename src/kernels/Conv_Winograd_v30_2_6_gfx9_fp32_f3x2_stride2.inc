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

v_mov_b32_e32 v0, v0
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_mov_b32_e32 v103, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s76, 0xc220
s_mov_b32 s75, 0xc220
v_readfirstlane_b32 s77, v0
s_lshr_b32 s77, s77, 5
s_add_u32 s77, s77, 8
s_and_b32 s71, s77, 20
s_mov_b64 s[78:79], s[6:7]
s_load_dwordx16 s[12:27], s[78:79], 0x0
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
s_load_dwordx2 s[20:21], s[20:21], 0x0
s_load_dwordx2 s[22:23], s[22:23], 0x0
s_load_dwordx2 s[24:25], s[24:25], 0x0
s_load_dwordx2 s[26:27], s[26:27], 0x0
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
s_load_dwordx2 s[34:35], s[34:35], 0x0
s_bitcmp1_b32 s18, 9
s_cbranch_scc0 77
s_mov_b32 s77, 0x8c
s_mov_b32 s80, 0x9c
v_cmp_le_u32_e32 vcc, 0x100, v0
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
v_ffbh_u32_e32 v106, s17
v_lshlrev_b32_e64 v107, v106, s17
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
v_mul_hi_u32 v106, s8, v104
v_add_co_u32_e64 v104, vcc, v106, s8
v_addc_co_u32_e64 v106, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v105
v_cndmask_b32_e32 v104, v104, v106, vcc
v_alignbit_b32 v104, v106, v104, v105
s_nop 0
v_readfirstlane_b32 s81, v104
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
v_cmp_le_u32_e32 vcc, 0x100, v0
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
s_add_u32 s77, s81, 2
v_mov_b32_e32 v105, 0x55555556
v_mul_hi_u32 v105, v105, s77
s_nop 0
v_readfirstlane_b32 s77, v105
s_andn2_b32 s81, 0, s31
s_addc_u32 s81, s33, 0
s_ashr_i32 s81, s81, 0
s_add_u32 s80, s81, 2
v_mov_b32_e32 v105, 0x55555556
v_mul_hi_u32 v105, v105, s80
s_nop 0
v_readfirstlane_b32 s80, v105
s_sub_u32 s55, 0, s80
s_sub_u32 s54, 0, s77
s_add_u32 s9, s28, 1
v_mov_b32_e32 v105, 0x80000000
v_mul_hi_u32 v105, v105, s9
s_nop 0
v_readfirstlane_b32 s9, v105
s_add_u32 s10, s29, 1
v_mov_b32_e32 v105, 0x80000000
v_mul_hi_u32 v105, v105, s10
s_nop 0
v_readfirstlane_b32 s10, v105
v_mad_i32_i24 v104, 2, s9, -1
v_sub_co_u32_e64 v104, vcc, v104, s28
v_addc_co_u32_e64 v104, vcc, 0, 0, vcc
s_nop 0
v_readfirstlane_b32 s81, v104
s_and_b32 s81, s81, 1
s_and_b32 s81, s81, s9
s_add_u32 s9, s9, s81
v_readfirstlane_b32 s82, v0
s_and_b32 s83, s82, 64
s_cselect_b32 s83, 0x80000, 0
s_or_b32 s18, s18, s83
s_lshl_b32 s49, s47, 1
s_mov_b64 s[50:51], 0
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
s_add_u32 s50, s50, s49
s_addc_u32 s51, s51, 0
v_bfe_u32 v105, v0, 2, 6
v_lshrrev_b32_e32 v98, 1, v105
s_bitcmp0_b32 s82, 8
s_cselect_b32 s83, 0x1000000, 0
s_or_b32 s83, s83, 0x100000
s_and_b32 s83, s18, s83
s_cselect_b32 s83, 0, 15
v_bfi_b32 v98, s83, v105, v98
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
v_mov_b32_e32 v105, s8
v_mov_b32_e32 v106, s17
v_and_b32_e32 v107, 3, v0
v_cmp_eq_u32_e32 vcc, 2, v107
v_cndmask_b32_e32 v105, v105, v106, vcc
v_cmp_eq_u32_e32 vcc, 1, v107
v_cndmask_b32_e32 v108, 0, v98, vcc
s_bitcmp1_b32 s18, 20
s_cbranch_scc0 4
v_add_co_u32_e64 v106, vcc, v98, 8
v_cmp_eq_u32_e32 vcc, 0, v107
v_cndmask_b32_e32 v108, v108, v106, vcc
v_cmp_eq_u32_e64 s[82:83], 3, v107
v_bfe_u32 v96, v108, 0, 5
v_mad_u32_u24 v96, v105, 32, v96
v_ffbh_u32_e32 v110, s80
v_lshlrev_b32_e64 v111, v110, s80
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
v_mad_i32_i24 v95, v97, s55, v96
v_lshrrev_b32_e32 v96, 5, v108
v_mad_u32_u24 v96, v97, 1, v96
v_cndmask_b32_e64 v96, v96, 1, s[82:83]
v_ffbh_u32_e32 v110, s77
v_lshlrev_b32_e64 v111, v110, s77
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
v_mad_i32_i24 v96, v97, s54, v96
v_readlane_b32 s56, v95, 2
v_readlane_b32 s57, v96, 2
v_readlane_b32 s58, v97, 2
v_readlane_b32 s59, v96, 3
v_readlane_b32 s60, v97, 3
v_add_co_u32_e64 v95, vcc, v95, s55
v_add_co_u32_e64 v96, vcc, v96, s54
v_mov_b32_dpp v97, v97 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v95, v95 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v96, v96 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x20000
s_mov_b32 s82, 0x80000000
s_mov_b32 s83, 0x20000
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
v_lshrrev_b32_e64 v104, 2, s71
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
s_lshr_b32 s71, s71, 0
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 47
s_and_b32 s78, s18, 0x1100000
s_addc_u32 s78, 0, 0
v_lshrrev_b32_e32 v107, 1, v0
s_mul_i32 s77, 60, s78
s_sub_u32 s77, 63, s77
v_bfi_b32 v107, s77, v0, v107
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
s_mul_i32 s77, 4, s78
s_sub_u32 s77, 6, s77
v_bfe_u32 v107, v0, s77, 1
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
s_bfe_u32 s78, s18, 0x10014
v_lshrrev_b32_e32 v107, 1, v0
s_mul_i32 s77, 60, s78
s_sub_u32 s77, 63, s77
v_bfi_b32 v107, s77, v0, v107
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
s_mul_i32 s77, 4, s78
s_sub_u32 s77, 6, s77
v_bfe_u32 v107, v0, s77, 1
v_mul_u32_u24_e32 v107, 0x1040, v107
v_mad_u32_u24 v90, 4, v105, v107
v_xor_b32_e32 v91, 0x307, v105
v_mad_u32_u24 v91, 4, v91, v107
v_xor_b32_e32 v92, 0x30f, v105
v_mad_u32_u24 v92, 4, v92, v107
v_xor_b32_e32 v93, 8, v105
v_mad_u32_u24 v93, 4, v93, v107
v_subrev_co_u32_e32 v95, vcc, s56, v95
v_mov_b32_e32 v105, s55
v_cmp_lt_i32_e32 vcc, v95, v105
v_subb_co_u32_e64 v104, vcc, 0, 0, vcc
v_mad_i32_i24 v95, v104, s55, v95
v_mad_i32_i24 v97, v104, s60, v97
v_mad_i32_i24 v96, v104, s59, v96
v_mov_b32_e32 v105, s54
v_cmp_lt_i32_e32 vcc, v96, v105
v_subb_co_u32_e64 v104, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v104
v_mad_i32_i24 v96, v104, v105, v96
v_subrev_co_u32_e32 v96, vcc, s57, v96
v_cmp_lt_i32_e32 vcc, v96, v105
v_subb_co_u32_e64 v104, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v104
v_mad_i32_i24 v96, v104, s54, v96
v_subrev_co_u32_e32 v97, vcc, s58, v97
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
v_add_co_u32_e32 v104, vcc, 2, v0
v_bfe_u32 v104, v104, 2, 1
v_cmp_ne_u32_e64 vcc, v104, 1
s_mov_b64 s[2:3], vcc
s_mov_b32 s73, 17
s_mov_b32 s62, 0
s_bitset1_b32 s18, 26
s_call_b64 s[4:5], 1589
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 1
s_branch 815
s_mov_b64 vcc, s[2:3]
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
v_cndmask_b32_dpp v72, v72, v72, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v73, v73, v73, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v74, v74, v74, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v75, v75, v75, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v72, v74, v72 div:2
v_subrev_f32_e64 v75, v73, v75 div:2
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 3
s_call_b64 s[4:5], 1514
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
v_add_f32_e64 v73, v74, v73 div:2
v_mad_f32 v74, v74, 1.0, -v73
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 5
s_call_b64 s[4:5], 1452
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
v_cndmask_b32_dpp v76, v76, v76, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v77, v77, v77, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v78, v78, v78, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v79, v79, v79, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v76, v78, v76 div:2
v_subrev_f32_e64 v79, v77, v79 div:2
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 3
s_call_b64 s[4:5], 1378
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
v_add_f32_e64 v77, v78, v77 div:2
v_mad_f32 v78, v78, 1.0, -v77
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 4
s_call_b64 s[4:5], 1315
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
v_cndmask_b32_dpp v80, v80, v80, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v81, v81, v81, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v82, v82, v82, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v83, v83, v83, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v80, v82, v80 div:2
v_subrev_f32_e64 v83, v81, v83 div:2
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 3
s_call_b64 s[4:5], 1242
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
v_add_f32_e64 v81, v82, v81 div:2
v_mad_f32 v82, v82, 1.0, -v81
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 5
s_call_b64 s[4:5], 1180
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
v_cndmask_b32_dpp v84, v84, v84, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v85, v85, v85, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v86, v86, v86, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v87, v87, v87, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v84, v86, v84 div:2
v_subrev_f32_e64 v87, v85, v87 div:2
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 3
s_call_b64 s[4:5], 1106
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
v_add_f32_e64 v85, v86, v85 div:2
v_mad_f32 v86, v86, 1.0, -v85
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 4
s_call_b64 s[4:5], 1043
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
v_cndmask_b32_dpp v64, v64, v64, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v65, v65, v65, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v66, v66, v66, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v67, v67, v67, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v64, v66, v64 div:2
v_subrev_f32_e64 v67, v65, v67 div:2
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 3
s_call_b64 s[4:5], 970
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
v_add_f32_e64 v65, v66, v65 div:2
v_mad_f32 v66, v66, 1.0, -v65
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 5
s_call_b64 s[4:5], 908
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
v_cndmask_b32_dpp v68, v68, v68, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v69, v69, v69, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v70, v70, v70, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v71, v71, v71, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v68, v70, v68 div:2
v_subrev_f32_e64 v71, v69, v71 div:2
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 3
s_call_b64 s[4:5], 834
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
v_add_f32_e64 v69, v70, v69 div:2
v_mad_f32 v70, v70, 1.0, -v69
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 64724
s_call_b64 s[4:5], 771
s_branch 64722
s_mov_b64 vcc, s[2:3]
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
v_cndmask_b32_dpp v72, v72, v72, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v73, v73, v73, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v74, v74, v74, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v75, v75, v75, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
buffer_load_dword v64, v1, s[40:43], 0 idxen
buffer_load_dword v67, v36, s[40:43], 0 idxen
ds_write_b32 v90, v68
ds_write_b32 v91, v69
ds_read_b128 v[38:41], v94 offset:29440
ds_read_b128 v[46:49], v89 offset:28928
ds_read_b128 v[50:53], v89 offset:29056
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 2
s_call_b64 s[4:5], 705
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
v_mac_f32_dpp v72, v72, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v75, v75, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 8
s_call_b64 s[4:5], 647
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
v_cndmask_b32_dpp v76, v76, v76, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v77, v77, v77, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v78, v78, v78, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v79, v79, v79, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
buffer_load_dword v68, v1, s[40:43], 0 idxen
buffer_load_dword v71, v36, s[40:43], 0 idxen
ds_write_b32 v90, v72 offset:8256
ds_write_b32 v91, v73 offset:8256
ds_read_b128 v[38:41], v94 offset:37696
ds_read_b128 v[46:49], v89 offset:37184
ds_read_b128 v[50:53], v89 offset:37312
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 2
s_call_b64 s[4:5], 577
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
v_mac_f32_dpp v76, v76, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v79, v79, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 518
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
v_cndmask_b32_dpp v80, v80, v80, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v81, v81, v81, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v82, v82, v82, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v83, v83, v83, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
buffer_load_dword v72, v1, s[40:43], 0 idxen
buffer_load_dword v75, v36, s[40:43], 0 idxen
ds_write_b32 v90, v76 offset:16512
ds_write_b32 v91, v77 offset:16512
ds_read_b128 v[38:41], v94 offset:45952
ds_read_b128 v[46:49], v89 offset:45440
ds_read_b128 v[50:53], v89 offset:45568
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 2
s_call_b64 s[4:5], 449
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
v_mac_f32_dpp v80, v80, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v83, v83, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 8
s_call_b64 s[4:5], 391
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
v_cndmask_b32_dpp v84, v84, v84, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v85, v85, v85, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v86, v86, v86, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v87, v87, v87, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
buffer_load_dword v76, v1, s[40:43], 0 idxen
buffer_load_dword v79, v36, s[40:43], 0 idxen
ds_write_b32 v90, v80 offset:24768
ds_write_b32 v91, v81 offset:24768
ds_read_b128 v[38:41], v94 offset:4672
ds_read_b128 v[46:49], v89 offset:4160
ds_read_b128 v[50:53], v89 offset:4288
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 2
s_call_b64 s[4:5], 321
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
v_mac_f32_dpp v84, v84, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v87, v87, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 7
s_call_b64 s[4:5], 262
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
v_cndmask_b32_dpp v64, v64, v64, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v65, v65, v65, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v66, v66, v66, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v67, v67, v67, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
buffer_load_dword v80, v1, s[40:43], 0 idxen
buffer_load_dword v83, v36, s[40:43], 0 idxen
ds_write_b32 v90, v84 offset:33024
ds_write_b32 v91, v85 offset:33024
ds_read_b128 v[38:41], v94 offset:12928
ds_read_b128 v[46:49], v89 offset:12416
ds_read_b128 v[50:53], v89 offset:12544
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 2
s_call_b64 s[4:5], 193
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
v_mac_f32_dpp v64, v64, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v67, v67, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 8
s_call_b64 s[4:5], 135
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
v_cndmask_b32_dpp v68, v68, v68, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v69, v69, v69, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v70, v70, v70, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v71, v71, v71, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v34, v40, v53
v_mac_f32_e32 v35, v41, v53
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
buffer_load_dword v84, v1, s[40:43], 0 idxen
buffer_load_dword v87, v36, s[40:43], 0 idxen
ds_write_b32 v90, v64 offset:41280
ds_write_b32 v91, v65 offset:41280
ds_read_b128 v[38:41], v94 offset:21184
ds_read_b128 v[46:49], v89 offset:20672
ds_read_b128 v[50:53], v89 offset:20800
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 2
s_call_b64 s[4:5], 65
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
v_mac_f32_dpp v68, v68, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v71, v71, v99 quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
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
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -1
s_cbranch_scc1 64775
s_call_b64 s[4:5], 6
s_branch 64773
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_nop
s_cmp_eq_u32 s62, 0
s_cbranch_scc0 8
s_branch 596
s_add_u32 s62, s62, 1
s_andn2_b32 s62, s62, 1
s_bitcmp1_b32 0, 26
s_cselect_b32 s88, s49, s50
s_cselect_b32 s89, 0, s51
s_sub_u32 s40, s40, s88
s_subb_u32 s41, s41, s89
s_cmp_eq_u32 s73, 0
s_cbranch_scc0 3
s_cbranch_scc1 610
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
s_cbranch_scc1 241
s_add_u32 s68, s68, s17
s_cmp_eq_u32 s68, 0
s_cbranch_scc1 238
s_mov_b32 s69, 0
s_bitcmp1_b32 s18, 16
s_cbranch_scc1 227
s_add_u32 s67, s16, 31
s_lshr_b32 s67, s67, 5
v_mov_b32_e32 v105, s68
v_mul_u32_u24_e32 v105, s67, v105
v_add_co_u32_e32 v105, vcc, s17, v105
v_sub_co_u32_e64 v105, vcc, v105, 1
v_ffbh_u32_e32 v108, s17
v_lshlrev_b32_e64 v109, v108, s17
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
v_readfirstlane_b32 s66, v104
v_mul_u32_u24_e64 v104, v104, s8
v_ffbh_u32_e32 v108, s67
v_lshlrev_b32_e64 v109, v108, s67
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
v_readfirstlane_b32 s77, v104
v_readfirstlane_b32 s64, v105
s_mul_i32 s64, s64, s67
s_sub_u32 s64, s77, s64
v_sub_co_u32_e32 v105, vcc, s8, v105
v_sub_co_u32_e32 v105, vcc, s17, v105
v_and_b32_e64 v107, v0, 63
v_cmp_eq_u32_e64 vcc, v107, 0
v_cndmask_b32_e32 v105, 1, v105, vcc
s_sub_u32 s78, 0, s55
s_sub_u32 s79, 0, s54
v_mul_u32_u24_e64 v109, v105, 32
v_ffbh_u32_e32 v112, s78
v_lshlrev_b32_e64 v113, v112, s78
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
v_mad_i32_i24 v108, v107, s55, v109
v_mul_u32_u24_e64 v109, v107, 1
v_ffbh_u32_e32 v112, s79
v_lshlrev_b32_e64 v113, v112, s79
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
v_mad_i32_i24 v109, v107, s54, v109
v_readfirstlane_b32 s56, v108
v_readfirstlane_b32 s57, v109
v_readfirstlane_b32 s58, v107
v_add_co_u32_e32 v95, vcc, s56, v95
v_addc_co_u32_e64 v110, vcc, 0, 0, vcc
v_mad_i32_i24 v95, v110, s55, v95
v_mad_i32_i24 v97, v110, s60, v97
v_mad_i32_i24 v96, v110, s59, v96
v_cmp_ge_i32_e64 vcc, v96, 0
v_addc_co_u32_e64 v110, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v110
v_mad_i32_i24 v96, v110, s54, v96
v_add_co_u32_e32 v96, vcc, s57, v96
v_addc_co_u32_e64 v110, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v110
v_mad_i32_i24 v96, v110, s54, v96
v_add_co_u32_e32 v97, vcc, s58, v97
v_readlane_b32 s56, v108, 1
v_readlane_b32 s57, v109, 1
v_readlane_b32 s58, v107, 1
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
s_branch 43
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
s_branch 31
s_add_u32 s63, s63, 32
s_cmp_ge_u32 s63, s65
s_cbranch_scc0 28
s_bitset1_b32 s18, 22
s_sub_u32 s68, s68, s17
s_subb_u32 s69, s69, 0
s_cbranch_scc1 65281
v_add_co_u32_e32 v95, vcc, s56, v95
v_addc_co_u32_e64 v104, vcc, 0, 0, vcc
v_mad_i32_i24 v95, v104, s55, v95
v_mad_i32_i24 v97, v104, s60, v97
v_mad_i32_i24 v96, v104, s59, v96
v_cmp_ge_i32_e64 vcc, v96, 0
v_addc_co_u32_e64 v104, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v104
v_mad_i32_i24 v96, v104, s54, v96
v_add_co_u32_e32 v96, vcc, s57, v96
v_addc_co_u32_e64 v104, vcc, 0, 0, vcc
v_add_co_u32_e32 v97, vcc, v97, v104
v_mad_i32_i24 v96, v104, s54, v96
v_add_co_u32_e32 v97, vcc, s58, v97
s_mov_b32 s63, s64
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccz 168
v_subrev_co_u32_e32 v104, vcc, s55, v95
v_subrev_co_u32_e32 v105, vcc, s54, v96
s_bitcmp1_b32 s18, 22
s_cbranch_scc0 64
s_bitset0_b32 s18, 22
s_bfe_u32 s77, s18, 0x10014
v_mul_u32_u24_e32 v107, 3, v104
v_mul_u32_u24_e32 v108, 3, v105
v_cvt_pk_u16_u32 v110, v107, v108
v_and_b32_e64 v107, v0, 1
v_cmp_eq_u32_e64 vcc, v107, 1
v_cndmask_b32_e32 v110, v97, v110, vcc
v_lshrrev_b32_e32 v106, 1, v0
v_bfe_u32 v111, v106, s77, 1
v_lshrrev_b32_e32 v106, 1, v0
v_bfi_b32 v106, 1, v0, v106
v_lshrrev_b32_e32 v107, 2, v0
v_bfi_b32 v107, 1, v0, v107
v_cmp_eq_u32_e64 vcc, s77, 0
v_cndmask_b32_e32 v106, v107, v106, vcc
s_sub_u32 s77, 1, s77
v_lshrrev_b32_e32 v107, s77, v106
v_bfi_b32 v106, 32, v107, v106
v_and_b32_e32 v106, 63, v106
v_add_co_u32_e32 v107, vcc, 16, v106
v_and_b32_e64 v108, v0, 2
v_cmp_eq_u32_e64 vcc, v108, 0
v_cndmask_b32_e32 v107, v107, v106, vcc
v_lshlrev_b32_e32 v108, 14, v111
v_mad_u32_u24 v107, 4, v107, v108
v_add_co_u32_e32 v106, vcc, s75, v107
ds_write_b32 v106, v110
v_writelane_b32 v108, s18, 0
v_writelane_b32 v108, s65, 1
v_writelane_b32 v108, s64, 2
v_and_b32_e64 v106, v0, 63
v_cmp_ge_u32_e64 vcc, v106, 3
v_mov_b32_e32 v109, 0x4000
v_cndmask_b32_e32 v106, v106, v109, vcc
v_mad_i32_i24 v106, v106, 4, s75
ds_write_b32 v106, v108 offset:256
s_add_u32 s75, s75, 0x18c
s_cmp_eq_u32 s75, 0x10000
s_cselect_b32 s75, 0xc220, s75
v_mov_b32_dpp v106, v97 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v104, v104 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v105, v105 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_readfirstlane_b32 s61, v106
v_sub_co_u32_e64 v107, vcc, v106, s61
v_mul_lo_u32 v107, v107, s44
v_mad_u32_u24 v110, 5, v0, 2
v_and_b32_e32 v111, 6, v0
v_add_co_u32_e32 v111, vcc, 4, v111
v_bfi_b32 v111, 4, v110, v111
v_bfe_u32 v111, v111, 1, 3
v_ashrrev_i32_e64 v112, 0, s31
v_subrev_co_u32_e32 v111, vcc, v112, v111
v_ashrrev_i32_e64 v112, 0, s11
v_mad_i32_i24 v108, v112, 2, v111
v_add_co_u32_e64 v109, vcc, 0, s38
v_ashrrev_i32_e32 v109, 0, v109
v_add_co_u32_e64 v110, vcc, 0, s30
v_ashrrev_i32_e32 v110, 0, v110
v_sub_i32 v109, v109, v110
v_cmp_ge_u32_e64 s[78:79], v106, s12
v_mad_i32_i24 v104, v104, 6, v108
v_cmp_ge_u32_e64 s[92:93], v104, s15
v_add_co_u32_e32 v104, vcc, v104, v107
s_or_b64 s[92:93], s[92:93], s[78:79]
v_mad_i32_i24 v105, v105, 6, v109
v_cmp_ge_u32_e64 s[94:95], v105, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v1, v105, s15, v104
v_cndmask_b32_e64 v1, v1, -1, s[94:95]
v_add_co_u32_e32 v105, vcc, 2, v105
v_cmp_ge_u32_e64 s[94:95], v105, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v2, v105, s15, v104
v_cndmask_b32_e64 v2, v2, -1, s[94:95]
v_add_co_u32_e32 v105, vcc, 2, v105
v_cmp_ge_u32_e64 s[94:95], v105, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v3, v105, s15, v104
v_cndmask_b32_e64 v3, v3, -1, s[94:95]
v_add_co_u32_e32 v105, vcc, 2, v105
v_cmp_ge_u32_e64 s[94:95], v105, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v36, v105, s15, v104
v_cndmask_b32_e64 v36, v36, -1, s[94:95]
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 147
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
s_lshr_b32 s77, s18, 6
s_xor_b32 s77, s77, s18
s_and_b32 s77, s77, 0x80000
s_cselect_b32 s77, s48, 0
s_lshl_b32 s77, s77, 2
s_add_u32 s40, s40, s77
s_addc_u32 s41, s41, 0
s_add_u32 s41, s41, 0x40000
s_branch 100
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 122
v_mad_u32_u24 v106, 5, v0, 2
v_lshlrev_b32_e32 v104, 1, v0
v_bfi_b32 v106, 4, v106, v104
v_bfe_u32 v104, v106, 2, 1
v_lshlrev_b32_e32 v106, 1, v104
v_bfe_u32 v104, v0, 1, 1
v_add_co_u32_e32 v104, vcc, v104, v106
v_mad_u32_u24 v104, s11, 2, v104
v_sub_co_u32_e32 v106, vcc, s29, v104
v_sub_co_u32_e64 v106, vcc, v106, 1
s_bfe_u32 s77, s18, 0x10001
v_cmp_eq_u32_e64 vcc, s77, 1
v_cndmask_b32_e32 v104, v104, v106, vcc
v_cmp_ge_u32_e64 s[78:79], v104, s29
s_bfe_u32 s77, s18, 0x10018
v_bfe_u32 v107, v0, 2, s77
v_mul_lo_u32 v107, s48, v107
v_add_co_u32_e32 v104, vcc, v104, v107
v_mul_lo_u32 v105, s70, v98
v_add_co_u32_e32 v105, vcc, v105, v104
s_sub_u32 s77, s28, s38
s_sub_u32 s77, s77, 3
s_bitcmp1_b32 s18, 0
s_cselect_b32 s77, s77, s38
v_mov_b32_e32 v107, s77
v_cmp_ge_u32_e64 s[92:93], v107, s28
v_mad_i32_i24 v1, v107, s29, v105
s_or_b64 s[92:93], s[92:93], s[78:79]
v_cndmask_b32_e64 v1, v1, -1, s[92:93]
v_mov_b32_e32 v2, v1
v_add_co_u32_e64 v107, vcc, v107, 2
v_cmp_ge_u32_e64 s[92:93], v107, s28
v_mad_i32_i24 v36, v107, s29, v105
s_or_b64 s[92:93], s[92:93], s[78:79]
v_cndmask_b32_e64 v36, v36, -1, s[92:93]
v_add_co_u32_e64 v107, vcc, v107, 2
v_cmp_ge_u32_e64 s[92:93], v107, s28
v_mad_i32_i24 v3, v107, s29, v105
s_or_b64 s[92:93], s[92:93], s[78:79]
v_cndmask_b32_e64 v3, v3, -1, s[92:93]
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v1, v2, v36, vcc
v_cndmask_b32_e32 v36, v36, v2, vcc
v_add_co_u32_e64 v104, vcc, v98, s63
v_cmp_lt_u32_e64 vcc, v104, s16
v_cndmask_b32_e32 v1, -1, v1, vcc
v_cndmask_b32_e32 v2, -1, v2, vcc
v_cndmask_b32_e32 v3, -1, v3, vcc
v_cndmask_b32_e32 v36, -1, v36, vcc
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
s_lshr_b32 s77, s18, 6
s_xor_b32 s77, s77, s18
s_and_b32 s77, s77, 0x80000
s_cselect_b32 s77, s48, 0
s_lshl_b32 s77, s77, 2
s_add_u32 s40, s40, s77
s_addc_u32 s41, s41, 0
s_add_u32 s41, s41, 0x40000
s_mov_b32 s43, 0x20000
s_mov_b32 s53, -1
s_bfe_u32 s88, s18, 0x10014
s_lshl_b32 s62, s13, s88
s_bfe_u32 s88, s18, 0x10013
s_bfe_u32 s90, s18, 0x10019
s_xor_b32 s88, s88, s90
s_cselect_b32 s88, 1, 0
s_cselect_b32 s43, 0x20000, s43
s_and_b32 s88, s88, s62
s_sub_u32 s62, s62, s88
s_bitcmp1_b32 s18, 20
s_cselect_b32 s88, 0, 0x2000000
s_bitcmp1_b32 s13, 0
s_cselect_b32 s88, s88, 0
s_xor_b32 s18, s18, s88
s_cmp_eq_u32 s62, 0
s_cbranch_scc1 2
s_mov_b64 vcc, s[2:3]
s_branch 64940
s_and_b32 s88, 0x900000, s18
s_subb_u32 s11, s11, 1
s_cbranch_scc0 65235
s_and_b32 s88, 0x900000, s18
s_subb_u32 s11, s10, 1
s_add_u32 s38, s38, 4
s_cmp_ge_u32 s38, s28
s_cbranch_scc0 65229
s_mov_b32 s38, 1
s_cmp_ge_u32 s38, s28
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
s_mov_b32 s78, 0x3c3c3c3c
s_mov_b32 s79, s78
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
v_cndmask_b32_e64 v5, v106, v5, s[78:79]
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
v_cndmask_b32_e64 v8, v106, v9, s[78:79]
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
v_cndmask_b32_e64 v11, v106, v13, s[78:79]
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
v_cndmask_b32_e64 v14, v106, v17, s[78:79]
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
v_cndmask_b32_e64 v17, v106, v21, s[78:79]
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
v_cndmask_b32_e64 v20, v106, v25, s[78:79]
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
v_cndmask_b32_e64 v23, v106, v29, s[78:79]
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
v_cndmask_b32_e64 v26, v106, v33, s[78:79]
v_mov_b32_dpp v27, v34 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v27, v34 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
s_waitcnt vmcnt(0)
v_readlane_b32 s95, v103, 0
v_add_f32_e64 v4, v4, s95
v_mul_f32_e64 v105, v4, s36
v_cmp_lt_f32_e64 vcc, v4, 0
v_cndmask_b32_e32 v4, v4, v105, vcc
v_add_f32_e64 v5, v5, s95
v_mul_f32_e64 v105, v5, s36
v_cmp_lt_f32_e64 vcc, v5, 0
v_cndmask_b32_e32 v5, v5, v105, vcc
v_add_f32_e64 v6, v6, s95
v_mul_f32_e64 v105, v6, s36
v_cmp_lt_f32_e64 vcc, v6, 0
v_cndmask_b32_e32 v6, v6, v105, vcc
buffer_store_dword v4, v37, s[80:83], 0 idxen
buffer_store_dword v5, v62, s[80:83], 0 idxen
buffer_store_dword v6, v63, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v103, 1
v_add_f32_e64 v7, v7, s95
v_mul_f32_e64 v105, v7, s36
v_cmp_lt_f32_e64 vcc, v7, 0
v_cndmask_b32_e32 v7, v7, v105, vcc
v_add_f32_e64 v8, v8, s95
v_mul_f32_e64 v105, v8, s36
v_cmp_lt_f32_e64 vcc, v8, 0
v_cndmask_b32_e32 v8, v8, v105, vcc
v_add_f32_e64 v9, v9, s95
v_mul_f32_e64 v105, v9, s36
v_cmp_lt_f32_e64 vcc, v9, 0
v_cndmask_b32_e32 v9, v9, v105, vcc
buffer_store_dword v7, v37, s[80:83], 0 idxen
buffer_store_dword v8, v62, s[80:83], 0 idxen
buffer_store_dword v9, v63, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v103, 2
v_add_f32_e64 v10, v10, s95
v_mul_f32_e64 v105, v10, s36
v_cmp_lt_f32_e64 vcc, v10, 0
v_cndmask_b32_e32 v10, v10, v105, vcc
v_add_f32_e64 v11, v11, s95
v_mul_f32_e64 v105, v11, s36
v_cmp_lt_f32_e64 vcc, v11, 0
v_cndmask_b32_e32 v11, v11, v105, vcc
v_add_f32_e64 v12, v12, s95
v_mul_f32_e64 v105, v12, s36
v_cmp_lt_f32_e64 vcc, v12, 0
v_cndmask_b32_e32 v12, v12, v105, vcc
buffer_store_dword v10, v37, s[80:83], 0 idxen
buffer_store_dword v11, v62, s[80:83], 0 idxen
buffer_store_dword v12, v63, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v103, 3
v_add_f32_e64 v13, v13, s95
v_mul_f32_e64 v105, v13, s36
v_cmp_lt_f32_e64 vcc, v13, 0
v_cndmask_b32_e32 v13, v13, v105, vcc
v_add_f32_e64 v14, v14, s95
v_mul_f32_e64 v105, v14, s36
v_cmp_lt_f32_e64 vcc, v14, 0
v_cndmask_b32_e32 v14, v14, v105, vcc
v_add_f32_e64 v15, v15, s95
v_mul_f32_e64 v105, v15, s36
v_cmp_lt_f32_e64 vcc, v15, 0
v_cndmask_b32_e32 v15, v15, v105, vcc
buffer_store_dword v13, v37, s[80:83], 0 idxen
buffer_store_dword v14, v62, s[80:83], 0 idxen
buffer_store_dword v15, v63, s[80:83], 0 idxen
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
v_readlane_b32 s95, v103, 8
v_add_f32_e64 v16, v16, s95
v_mul_f32_e64 v105, v16, s36
v_cmp_lt_f32_e64 vcc, v16, 0
v_cndmask_b32_e32 v16, v16, v105, vcc
v_add_f32_e64 v17, v17, s95
v_mul_f32_e64 v105, v17, s36
v_cmp_lt_f32_e64 vcc, v17, 0
v_cndmask_b32_e32 v17, v17, v105, vcc
v_add_f32_e64 v18, v18, s95
v_mul_f32_e64 v105, v18, s36
v_cmp_lt_f32_e64 vcc, v18, 0
v_cndmask_b32_e32 v18, v18, v105, vcc
buffer_store_dword v16, v37, s[80:83], 0 idxen
buffer_store_dword v17, v62, s[80:83], 0 idxen
buffer_store_dword v18, v63, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v103, 9
v_add_f32_e64 v19, v19, s95
v_mul_f32_e64 v105, v19, s36
v_cmp_lt_f32_e64 vcc, v19, 0
v_cndmask_b32_e32 v19, v19, v105, vcc
v_add_f32_e64 v20, v20, s95
v_mul_f32_e64 v105, v20, s36
v_cmp_lt_f32_e64 vcc, v20, 0
v_cndmask_b32_e32 v20, v20, v105, vcc
v_add_f32_e64 v21, v21, s95
v_mul_f32_e64 v105, v21, s36
v_cmp_lt_f32_e64 vcc, v21, 0
v_cndmask_b32_e32 v21, v21, v105, vcc
buffer_store_dword v19, v37, s[80:83], 0 idxen
buffer_store_dword v20, v62, s[80:83], 0 idxen
buffer_store_dword v21, v63, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v103, 10
v_add_f32_e64 v22, v22, s95
v_mul_f32_e64 v105, v22, s36
v_cmp_lt_f32_e64 vcc, v22, 0
v_cndmask_b32_e32 v22, v22, v105, vcc
v_add_f32_e64 v23, v23, s95
v_mul_f32_e64 v105, v23, s36
v_cmp_lt_f32_e64 vcc, v23, 0
v_cndmask_b32_e32 v23, v23, v105, vcc
v_add_f32_e64 v24, v24, s95
v_mul_f32_e64 v105, v24, s36
v_cmp_lt_f32_e64 vcc, v24, 0
v_cndmask_b32_e32 v24, v24, v105, vcc
buffer_store_dword v22, v37, s[80:83], 0 idxen
buffer_store_dword v23, v62, s[80:83], 0 idxen
buffer_store_dword v24, v63, s[80:83], 0 idxen
s_lshl_b32 s95, s46, 2
s_add_u32 s80, s80, s95
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s95, v103, 11
v_add_f32_e64 v25, v25, s95
v_mul_f32_e64 v105, v25, s36
v_cmp_lt_f32_e64 vcc, v25, 0
v_cndmask_b32_e32 v25, v25, v105, vcc
v_add_f32_e64 v26, v26, s95
v_mul_f32_e64 v105, v26, s36
v_cmp_lt_f32_e64 vcc, v26, 0
v_cndmask_b32_e32 v26, v26, v105, vcc
v_add_f32_e64 v27, v27, s95
v_mul_f32_e64 v105, v27, s36
v_cmp_lt_f32_e64 vcc, v27, 0
v_cndmask_b32_e32 v27, v27, v105, vcc
buffer_store_dword v25, v37, s[80:83], 0 idxen
buffer_store_dword v26, v62, s[80:83], 0 idxen
buffer_store_dword v27, v63, s[80:83], 0 idxen
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
s_xor_b32 s18, s18, 0x200000
s_mul_i32 s73, s9, s10
s_mul_i32 s73, s73, s13
s_add_u32 s88, s72, s71
s_cmp_lt_i32 s88, 0
s_cbranch_scc0 158
v_and_b32_e32 v37, 0x7f, v0
v_lshrrev_b32_e32 v37, 1, v37
v_bfi_b32 v37, 1, v0, v37
v_and_b32_e64 v62, v0, 2
v_mad_u32_u24 v37, v62, 16, v37
v_lshlrev_b32_e32 v37, 2, v37
v_add_co_u32_e64 v37, vcc, v37, s76
v_and_b32_e32 v62, 3, v0
v_lshlrev_b32_e32 v62, 2, v62
v_add_co_u32_e64 v62, vcc, v62, s76
ds_read_b32 v104, v62 offset:256
ds_read_b32 v37, v37
s_add_u32 s76, s76, 0x18c
s_cmp_eq_u32 s76, 0x10000
s_cselect_b32 s76, 0xc220, s76
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s74, v37
v_readlane_b32 s90, v104, 0
s_bitcmp1_b32 s90, 18
s_cbranch_scc1 132
v_readlane_b32 s88, v104, 1
v_readlane_b32 s89, v104, 2
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
s_mov_b32 s83, 0x20000
s_bitcmp1_b32 s18, 7
s_cselect_b32 s87, 0x20000, 0
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
v_cmp_ge_u32_e64 s[78:79], v106, s12
v_sub_co_u32_e64 v106, vcc, v106, s74
v_mul_lo_u32 v106, v106, s45
v_xor_b32_dpp v107, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v107, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v108, v0, v0 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v108, v0, v0 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_add_co_u32_e32 v108, vcc, v105, v108
v_add_co_u32_e32 v107, vcc, v104, v107
v_mad_i32_i24 v37, v107, s33, v108
v_add_co_u32_e32 v37, vcc, v37, v106
v_cmp_ge_u32_e64 s[94:95], v108, s33
s_or_b64 s[94:95], s[94:95], s[78:79]
v_cmp_ge_u32_e64 s[92:93], v107, s32
s_or_b64 s[94:95], s[94:95], s[92:93]
v_cndmask_b32_e64 v37, v37, -1, s[94:95]
v_xor_b32_dpp v107, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v107, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v108, v0, v0 quad_perm:[1,1,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32_e32 v108, vcc, v105, v108
v_add_co_u32_e32 v107, vcc, v104, v107
v_mad_i32_i24 v62, v107, s33, v108
v_add_co_u32_e32 v62, vcc, v62, v106
v_cmp_ge_u32_e64 s[94:95], v108, s33
s_or_b64 s[94:95], s[94:95], s[78:79]
v_cmp_ge_u32_e64 s[92:93], v107, s32
s_or_b64 s[94:95], s[94:95], s[92:93]
v_cndmask_b32_e64 v62, v62, -1, s[94:95]
v_xor_b32_dpp v107, v0, v0 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v107, v0, v0 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v108, v0, v0 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v108, v0, v0 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xa
v_add_co_u32_e32 v108, vcc, v105, v108
v_add_co_u32_e32 v107, vcc, v104, v107
v_mad_i32_i24 v63, v107, s33, v108
v_add_co_u32_e32 v63, vcc, v63, v106
v_cmp_ge_u32_e64 s[94:95], v108, s33
s_or_b64 s[94:95], s[94:95], s[78:79]
v_cmp_ge_u32_e64 s[92:93], v107, s32
s_or_b64 s[94:95], s[94:95], s[92:93]
v_cndmask_b32_e64 v63, v63, -1, s[94:95]
v_and_b32_e64 v103, v0, 63
buffer_load_dword v103, v103, s[84:87], 0 idxen
s_mov_b64 vcc, s[2:3]
s_branch 63894
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
