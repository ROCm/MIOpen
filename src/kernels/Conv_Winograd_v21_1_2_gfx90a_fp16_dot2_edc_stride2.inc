/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
v_mov_b32_e32 v116, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s97, 0xc1e0
s_mov_b32 s96, 0xc1e0
s_mov_b32 s91, 0
v_lshlrev_b32_e32 v119, 2, v0
v_add_co_u32_e32 v119, vcc, 0xffc0, v119
v_cmp_ge_u32_e32 vcc, 12, v0
s_cbranch_vccz 5
v_mov_b32_e32 v118, 0
v_cndmask_b32_e32 v119, -1, v119, vcc
ds_write_b32 v119, v118
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
s_barrier
v_readfirstlane_b32 s52, v0
s_lshr_b32 s52, s52, 5
s_add_u32 s52, s52, 8
s_and_b32 s92, s52, 20
s_mov_b64 s[40:41], s[6:7]
s_load_dwordx16 s[12:27], s[40:41], 0x0
s_load_dwordx4 s[28:31], s[40:41], 0x40
s_load_dwordx2 s[32:33], s[40:41], 0x50
s_waitcnt lgkmcnt(0)
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
s_load_dwordx2 s[34:35], s[40:41], 0x58
s_mov_b32 s36, 1.0
s_bitcmp1_b32 s18, 8
s_cbranch_scc0 2
s_load_dword s36, s[40:41], 0x60
s_mul_i32 s42, s14, s15
s_lshr_b32 s46, -1, 16
s_and_b32 s46, s46, s42
s_lshr_b32 s47, s42, 16
s_mul_i32 s47, s47, s13
s_mul_i32 s44, s46, s13
s_lshl_b32 s46, s47, 16
s_lshr_b32 s47, s47, 16
s_add_u32 s44, s46, s44
s_addc_u32 s45, s47, 0
s_lshl_b32 s65, s44, 1
s_lshl_b32 s68, s42, 1
s_mul_i32 s43, s32, s33
s_lshr_b32 s46, -1, 16
s_and_b32 s46, s46, s43
s_lshr_b32 s47, s43, 16
s_mul_i32 s47, s47, s16
s_mul_i32 s44, s46, s16
s_lshl_b32 s46, s47, 16
s_lshr_b32 s47, s47, 16
s_add_u32 s44, s46, s44
s_addc_u32 s45, s47, 0
s_lshl_b32 s66, s44, 1
s_lshl_b32 s67, s43, 1
s_bitcmp1_b32 s18, 7
s_cbranch_scc0 7
s_bitcmp1_b32 s18, 6
s_cbranch_scc0 5
s_waitcnt lgkmcnt(0)
s_and_b32 s35, s35, 0xffff
s_load_dwordx2 s[34:35], s[34:35], 0x0
s_and_b32 s18, s18, 0xffff
s_bitcmp1_b32 s18, 13
s_cbranch_scc0 2
s_load_dwordx8 s[48:55], s[40:41], 0x68
s_mul_i32 s42, s28, s29
s_lshl_b32 s42, s42, 1
s_bitcmp1_b32 s18, 2
s_cselect_b32 s43, s16, s13
s_lshr_b32 s44, -1, 16
s_and_b32 s44, s44, s42
s_lshr_b32 s45, s42, 16
s_mul_i32 s45, s45, s43
s_mul_i32 s56, s44, s43
s_lshl_b32 s44, s45, 16
s_lshr_b32 s45, s45, 16
s_add_u32 s56, s44, s56
s_addc_u32 s57, s45, 0
s_mov_b32 s43, s56
s_bitcmp1_b32 s18, 2
s_cselect_b32 s44, s43, s42
s_cselect_b32 s90, s42, s43
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cmp_eq_u32 1, src_vccz
s_cselect_b32 s68, s44, s68
s_waitcnt lgkmcnt(0)
s_and_b32 s21, s21, 0xffff
s_and_b32 s23, s23, 0xffff
s_and_b32 s25, s25, 0xffff
s_and_b32 s27, s27, 0xffff
s_and_b32 s35, s35, 0xffff
s_bitcmp1_b32 s18, 13
s_cbranch_scc0 8
s_add_u32 s20, s20, s48
s_addc_u32 s21, s21, s49
s_add_u32 s22, s22, s50
s_addc_u32 s23, s23, s51
s_add_u32 s24, s24, s52
s_addc_u32 s25, s25, s53
s_add_u32 s34, s34, s54
s_addc_u32 s35, s35, s55
v_cvt_f16_f32_e32 v2, s36
v_readfirstlane_b32 s36, v2
s_and_b32 s44, 0, s30
s_addc_u32 s44, s32, 0
s_ashr_i32 s44, s44, 0
s_add_u32 s42, s44, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s42
v_readfirstlane_b32 s42, v2
s_andn2_b32 s44, 0, s31
s_addc_u32 s44, s33, 0
s_ashr_i32 s44, s44, 0
s_add_u32 s43, s44, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s43
v_readfirstlane_b32 s43, v2
s_sub_u32 s75, 0, s43
s_sub_u32 s74, 0, s42
s_add_u32 s60, s28, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s60
v_readfirstlane_b32 s60, v2
s_add_u32 s61, s29, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s61
v_readfirstlane_b32 s61, v2
v_mad_i32_i24 v2, 3, s60, -2
v_sub_co_u32_e64 v2, vcc, v2, s28
v_addc_co_u32_e64 v2, vcc, 0, 0, vcc
v_readfirstlane_b32 s44, v2
s_and_b32 s44, s44, 1
s_and_b32 s44, s44, s60
s_add_u32 s60, s60, s44
v_readfirstlane_b32 s45, v0
s_and_b32 s48, s45, 64
s_cselect_b32 s48, 0x80000, 0
s_or_b32 s18, s18, s48
s_lshl_b32 s69, s68, 1
s_sub_u32 s70, 0, s69
s_subb_u32 s71, 0, 0
s_bitset1_b32 s18, 23
s_bitset1_b32 s18, 20
s_bitset0_b32 s18, 19
s_ashr_i32 s69, s69, 1
s_ashr_i64 s[70:71], s[70:71], 1
s_add_u32 s61, s61, 1
s_and_b32 s61, s61, -2
s_branch 16
s_and_b32 s48, s13, 3
s_cselect_b32 s48, 0, 0x1000000
s_bitcmp1_b32 s18, 2
s_cselect_b32 s48, 0, s48
s_or_b32 s18, s18, s48
s_cmp_eq_u32 s48, 0
s_cselect_b32 s69, s68, s69
s_cselect_b32 s70, s68, s70
s_cselect_b32 s71, 0, s71
s_bitcmp0_b32 s45, 8
s_cselect_b32 s48, s48, 0
s_cmp_eq_u32 s48, 0
s_cselect_b32 s48, 0, 0x80000
s_andn2_b32 s18, s18, s48
s_add_u32 s70, s70, s69
s_addc_u32 s71, s71, 0
s_add_u32 s70, s70, s69
s_addc_u32 s71, s71, 0
v_bfe_u32 v3, v0, 2, 6
v_lshrrev_b32_e32 v111, 1, v3
s_bitcmp0_b32 s45, 8
s_cselect_b32 s48, 0x1000000, 0
s_or_b32 s48, s48, 0x100000
s_and_b32 s48, s18, s48
s_cselect_b32 s48, 0, 15
v_bfi_b32 v111, s48, v3, v111
s_mul_i32 s88, s12, s42
s_sub_u32 s88, s88, 1
s_lshr_b32 s88, s88, 0
s_add_u32 s88, s88, 1
s_lshr_b32 s46, -1, 16
s_and_b32 s46, s46, s88
s_lshr_b32 s47, s88, 16
s_mul_i32 s47, s47, s43
s_mul_i32 s88, s46, s43
s_lshl_b32 s46, s47, 16
s_lshr_b32 s47, s47, 16
s_add_u32 s88, s46, s88
s_addc_u32 s89, s47, 0
s_sub_u32 s88, s88, 1
s_subb_u32 s89, s89, 0
s_lshr_b64 s[88:89], s[88:89], 5
s_add_u32 s88, s88, 1
s_addc_u32 s89, s89, 0
v_mov_b32_e32 v4, s8
v_mov_b32_e32 v5, s17
v_and_b32_e32 v6, 3, v0
v_cmp_eq_u32_e32 vcc, 2, v6
v_cndmask_b32_e32 v4, v4, v5, vcc
v_cmp_eq_u32_e32 vcc, 1, v6
v_cndmask_b32_e32 v7, 0, v111, vcc
s_bitcmp1_b32 s18, 20
s_cbranch_scc0 4
v_add_co_u32_e64 v5, vcc, v111, 8
v_cmp_eq_u32_e32 vcc, 0, v6
v_cndmask_b32_e32 v7, v7, v5, vcc
v_cmp_eq_u32_e64 s[46:47], 3, v6
v_bfe_u32 v109, v7, 0, 5
v_mad_u32_u24 v109, v4, 32, v109
v_ffbh_u32_e32 v10, s43
v_lshlrev_b32_e64 v11, v10, s43
v_and_b32_e32 v9, 0xffffff00, v11
v_cmp_eq_u32_e32 vcc, 0x80000000, v11
v_cvt_f32_u32_e32 v9, v9
v_rcp_f32_e32 v110, v9
v_subb_co_u32_e32 v8, vcc, 32, v10, vcc
v_cvt_f32_ubyte0_e32 v10, v11
v_fma_f32 v9, v9, v110, -1.0
v_fma_f32 v9, v10, v110, v9
v_madak_f32 v9, v9, v110, 0x9f000000
v_mul_f32_e32 v9, 0x5f800000, v9
v_mov_b32_e32 v10, 0
v_cvt_flr_i32_f32_e64 v9, -v9
v_lshl_add_u32 v110, v110, 9, v9
v_mad_u64_u32 v[10:11], vcc, v11, v110, v[10:11]
v_subb_co_u32_e64 v110, vcc, v110, -1, vcc
v_mul_hi_u32 v9, v109, v110
v_add_co_u32_e32 v110, vcc, v9, v109
v_addc_co_u32_e64 v9, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v8
v_cndmask_b32_e32 v110, v110, v9, vcc
v_alignbit_b32 v110, v9, v110, v8
v_mad_i32_i24 v108, v110, s75, v109
v_lshrrev_b32_e32 v109, 5, v7
v_mad_u32_u24 v109, v110, 1, v109
v_cndmask_b32_e64 v109, v109, 1, s[46:47]
v_ffbh_u32_e32 v10, s42
v_lshlrev_b32_e64 v11, v10, s42
v_and_b32_e32 v9, 0xffffff00, v11
v_cmp_eq_u32_e32 vcc, 0x80000000, v11
v_cvt_f32_u32_e32 v9, v9
v_rcp_f32_e32 v110, v9
v_subb_co_u32_e32 v8, vcc, 32, v10, vcc
v_cvt_f32_ubyte0_e32 v10, v11
v_fma_f32 v9, v9, v110, -1.0
v_fma_f32 v9, v10, v110, v9
v_madak_f32 v9, v9, v110, 0x9f000000
v_mul_f32_e32 v9, 0x5f800000, v9
v_mov_b32_e32 v10, 0
v_cvt_flr_i32_f32_e64 v9, -v9
v_lshl_add_u32 v110, v110, 9, v9
v_mad_u64_u32 v[10:11], vcc, v11, v110, v[10:11]
v_subb_co_u32_e64 v110, vcc, v110, -1, vcc
v_mul_hi_u32 v9, v109, v110
v_add_co_u32_e32 v110, vcc, v9, v109
v_addc_co_u32_e64 v9, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v8
v_cndmask_b32_e32 v110, v110, v9, vcc
v_alignbit_b32 v110, v9, v110, v8
v_mad_i32_i24 v109, v110, s74, v109
v_readlane_b32 s76, v108, 2
v_readlane_b32 s77, v109, 2
v_readlane_b32 s78, v110, 2
v_readlane_b32 s79, v109, 3
v_readlane_b32 s80, v110, 3
v_add_co_u32_e64 v108, vcc, v108, s75
v_add_co_u32_e64 v109, vcc, v109, s74
v_mov_b32_dpp v110, v110 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v108, v108 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v109, v109 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x20000
s_mov_b32 s46, 0x80000000
s_mov_b32 s47, 0x20000
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 7
v_xor_b32_dpp v112, v0, v0 quad_perm:[1,3,2,2] row_mask:0xf bank_mask:0xf
v_subrev_co_u32_e32 v112, vcc, 1, v112
v_cvt_f16_i16_e32 v112, v112
v_pk_add_f16 v112, v112, 0 op_sel_hi:[0,0]
s_branch 6
v_xor_b32_dpp v112, v0, v0 quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32_e32 v112, vcc, 1, v112
v_cvt_f16_i16_e32 v112, v112
v_pk_add_f16 v112, v112, 0 op_sel_hi:[0,0]
v_mov_b32_e32 v113, 1
v_xor_b32_dpp v113, v0, v0 quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v113, v0, v0 quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32_e32 v113, vcc, 1, v113
v_mov_b32_e32 v114, 1
v_xor_b32_dpp v114, v0, v0 quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v114, v0, v0 quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32_e32 v114, vcc, 1, v114
v_cvt_f32_i32_e32 v113, v113
v_cvt_f32_i32_e32 v114, v114
v_lshrrev_b32_e64 v118, 2, s92
v_and_b32_e32 v119, 3, v0
v_bfe_u32 v120, v0, 4, 3
v_mad_u32_u24 v107, v120, 4, v119
v_lshlrev_b32_e32 v107, 4, v107
v_mad_u32_u24 v102, v118, 4, v119
v_lshlrev_b32_e32 v102, 4, v102
v_bfe_u32 v118, v0, 2, 2
v_and_b32_e32 v119, 1, v118
v_mad_u32_u24 v121, v118, 16, v119
v_lshlrev_b32_e32 v121, 6, v121
v_xor_b32_e32 v102, v102, v121
v_mul_u32_u24_e32 v121, 0x400, v118
v_xor_b32_e32 v107, v107, v121
s_lshr_b32 s92, s92, 0
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 50
s_and_b32 s53, s18, 0x1100000
s_addc_u32 s53, 0, 0
v_lshrrev_b32_e32 v121, 1, v0
s_mul_i32 s52, 60, s53
s_sub_u32 s52, 63, s52
v_bfi_b32 v121, s52, v0, v121
v_and_b32_e32 v118, 1, v121
v_bfe_u32 v119, v121, 1, 1
v_xor_b32_e32 v118, v118, v119
v_bfe_u32 v120, v121, 3, 1
v_mad_u32_u24 v119, v119, 2, v120
v_mul_u32_u24_e32 v118, 0x118, v118
v_bfe_u32 v120, v121, 2, 1
v_mad_u32_u24 v119, v119, 2, v118
v_xor_b32_e32 v119, v119, v120
v_and_b32_e32 v120, 0xf0, v121
v_xor_b32_e32 v119, v119, v120
s_mul_i32 s52, 4, s53
s_sub_u32 s52, 6, s52
v_bfe_u32 v121, v0, s52, 1
v_mul_u32_u24_e32 v121, 0x1040, v121
v_xor_b32_e32 v104, 0x314, v119
v_xor_b32_e32 v105, 0x31c, v119
v_xor_b32_e32 v106, 8, v119
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v103, v119, v106, vcc
v_cndmask_b32_e32 v106, v106, v119, vcc
v_mad_u32_u24 v103, 4, v103, v121
v_mad_u32_u24 v104, 4, v104, v121
v_mad_u32_u24 v105, 4, v105, v121
v_mad_u32_u24 v106, 4, v106, v121
s_branch 44
s_bfe_u32 s53, s18, 0x10014
v_lshrrev_b32_e32 v121, 1, v0
s_mul_i32 s52, 60, s53
s_sub_u32 s52, 63, s52
v_bfi_b32 v121, s52, v0, v121
v_and_b32_e32 v118, 1, v121
v_bfe_u32 v119, v121, 1, 1
v_bfe_u32 v120, v121, 3, 1
v_xor_b32_e32 v118, v118, v119
v_mad_u32_u24 v119, v119, 2, v120
v_mul_u32_u24_e32 v118, 0x109, v118
v_bfe_u32 v120, v121, 2, 1
v_mad_u32_u24 v119, v119, 2, v118
v_xor_b32_e32 v119, v119, v120
v_and_b32_e32 v120, 0xf0, v121
v_or_b32_e32 v119, v119, v120
s_mul_i32 s52, 4, s53
s_sub_u32 s52, 6, s52
v_bfe_u32 v121, v0, s52, 1
v_mul_u32_u24_e32 v121, 0x1040, v121
v_mad_u32_u24 v103, 4, v119, v121
v_xor_b32_e32 v104, 0x307, v119
v_mad_u32_u24 v104, 4, v104, v121
v_xor_b32_e32 v105, 0x30f, v119
v_mad_u32_u24 v105, 4, v105, v121
v_xor_b32_e32 v106, 8, v119
v_mad_u32_u24 v106, 4, v106, v121
v_subrev_co_u32_e32 v108, vcc, s76, v108
v_mov_b32_e32 v119, s75
v_cmp_lt_i32_e32 vcc, v108, v119
v_subb_co_u32_e64 v118, vcc, 0, 0, vcc
v_mad_i32_i24 v108, v118, s75, v108
v_mad_i32_i24 v110, v118, s80, v110
v_mad_i32_i24 v109, v118, s79, v109
v_mov_b32_e32 v119, s74
v_cmp_lt_i32_e32 vcc, v109, v119
v_subb_co_u32_e64 v118, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v118
v_mad_i32_i24 v109, v118, v119, v109
v_subrev_co_u32_e32 v109, vcc, s77, v109
v_cmp_lt_i32_e32 vcc, v109, v119
v_subb_co_u32_e64 v118, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v118
v_mad_i32_i24 v109, v118, s74, v109
v_subrev_co_u32_e32 v110, vcc, s78, v110
s_mov_b32 s62, 0
s_mov_b32 s63, s28
s_mov_b32 s64, 1
s_mov_b32 s84, 0
s_mov_b32 s85, s16
s_mov_b32 s83, s85
s_sub_u32 s93, -1, s92
s_sub_u32 s93, s93, 32
s_bitset1_b32 s18, 21
s_mov_b32 s47, 0
s_mov_b32 s51, 0
v_add_co_u32_e32 v118, vcc, 2, v0
v_bfe_u32 v118, v118, 2, 1
v_cmp_ne_u32_e64 vcc, v118, 1
s_mov_b64 s[10:11], vcc
s_mov_b32 s94, 34
s_mov_b32 s82, 0
s_bitset1_b32 s18, 26
s_call_b64 s[38:39], 3124
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 65
s_branch 1594
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v70, v82, v70
v_pack_b32_f16 v71, v83, v71
v_pack_b32_f16 v72, v84, v72
v_pack_b32_f16 v73, v85, v73
v_cndmask_b32_dpp v70, v70, v70, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v71, v71, v71, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v72, v72, v72, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v73, v73, v73, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v70, v72, -1.0, v70 op_sel_hi:[1,0,1]
v_pk_mul_f16 v70, v70, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v73, v71, -1.0, v73 op_sel_hi:[1,0,1]
v_pk_mul_f16 v73, v73, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:29440
ds_read_b128 v[42:45], v102 offset:28928
ds_read_b128 v[46:49], v102 offset:29056
ds_write_b32 v103, v62
ds_write_b32 v104, v63
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v58, v94, s[40:43], 0 offen
buffer_load_short_d16 v60, v96, s[40:43], 0 offen
buffer_load_short_d16 v59, v95, s[40:43], 0 offen
buffer_load_short_d16 v61, v97, s[40:43], 0 offen
s_add_u32 s91, s91, 0x200
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 6
s_call_b64 s[38:39], 2933
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_pk_add_f16 v71, v72, v71
v_pk_mul_f16 v71, v71, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v72, v71, -1.0, v72 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v70 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v70, v115, v112, v70
v_mov_b32_dpp v115, v71 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v71, v115, v112, v71
v_mov_b32_dpp v115, v72 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v72, v115, v112, v72
v_mov_b32_dpp v115, v73 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v73, v115, v112, v73
s_setprio 0
s_nop 0
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_nop 0
ds_read_b128 v[38:41], v107 offset:33536
ds_read_b128 v[50:53], v102 offset:33024
ds_read_b128 v[54:57], v102 offset:33152
ds_write_b32 v105, v68 offset:8256
ds_write_b32 v106, v69 offset:8256
s_setprio 1
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v82, v94, s[40:43], 0 offen
buffer_load_short_d16 v84, v96, s[40:43], 0 offen
buffer_load_short_d16 v83, v95, s[40:43], 0 offen
buffer_load_short_d16 v85, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(16) lgkmcnt(5)
ds_append v117 offset:65472
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 2807
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v74, v86, v74
v_pack_b32_f16 v75, v87, v75
v_pack_b32_f16 v76, v88, v76
v_pack_b32_f16 v77, v89, v77
v_cndmask_b32_dpp v74, v74, v74, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v75, v75, v75, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v76, v76, v76, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v77, v77, v77, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v74, v76, -1.0, v74 op_sel_hi:[1,0,1]
v_pk_mul_f16 v74, v74, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v77, v75, -1.0, v77 op_sel_hi:[1,0,1]
v_pk_mul_f16 v77, v77, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:37696
ds_read_b128 v[42:45], v102 offset:37184
ds_read_b128 v[46:49], v102 offset:37312
ds_write_b32 v103, v66 offset:8256
ds_write_b32 v104, v67 offset:8256
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v62, v94, s[40:43], 0 offen
buffer_load_short_d16 v64, v96, s[40:43], 0 offen
buffer_load_short_d16 v63, v95, s[40:43], 0 offen
buffer_load_short_d16 v65, v97, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc0
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 6
s_call_b64 s[38:39], 2677
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_pk_add_f16 v75, v76, v75
v_pk_mul_f16 v75, v75, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v76, v75, -1.0, v76 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v74 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v74, v115, v112, v74
v_mov_b32_dpp v115, v75 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v75, v115, v112, v75
v_mov_b32_dpp v115, v76 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v76, v115, v112, v76
v_mov_b32_dpp v115, v77 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v77, v115, v112, v77
s_setprio 0
s_nop 0
v_dot2_f32_f16 v33, v41, v57, v33
s_barrier
s_mov_b64 vcc, s[10:11]
s_nop 0
ds_read_b128 v[38:41], v107 offset:41792
ds_read_b128 v[50:53], v102 offset:41280
ds_read_b128 v[54:57], v102 offset:41408
ds_write_b32 v105, v72 offset:16512
ds_write_b32 v106, v73 offset:16512
s_setprio 1
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v86, v94, s[40:43], 0 offen
buffer_load_short_d16 v88, v96, s[40:43], 0 offen
buffer_load_short_d16 v87, v95, s[40:43], 0 offen
buffer_load_short_d16 v89, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(16) lgkmcnt(5)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 2551
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v78, v90, v78
v_pack_b32_f16 v79, v91, v79
v_pack_b32_f16 v80, v92, v80
v_pack_b32_f16 v81, v93, v81
v_cndmask_b32_dpp v78, v78, v78, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v79, v79, v79, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v80, v80, v80, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v81, v81, v81, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v78, v80, -1.0, v78 op_sel_hi:[1,0,1]
v_pk_mul_f16 v78, v78, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v81, v79, -1.0, v81 op_sel_hi:[1,0,1]
v_pk_mul_f16 v81, v81, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:45952
ds_read_b128 v[42:45], v102 offset:45440
ds_read_b128 v[46:49], v102 offset:45568
ds_write_b32 v103, v70 offset:16512
ds_write_b32 v104, v71 offset:16512
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v66, v94, s[40:43], 0 offen
buffer_load_short_d16 v68, v96, s[40:43], 0 offen
buffer_load_short_d16 v67, v95, s[40:43], 0 offen
buffer_load_short_d16 v69, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 2423
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_pk_add_f16 v79, v80, v79
v_pk_mul_f16 v79, v79, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v80, v79, -1.0, v80 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v78 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v78, v115, v112, v78
v_mov_b32_dpp v115, v79 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v79, v115, v112, v79
v_mov_b32_dpp v115, v80 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v80, v115, v112, v80
v_mov_b32_dpp v115, v81 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v81, v115, v112, v81
s_setprio 0
s_nop 0
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_nop 0
ds_read_b128 v[38:41], v107 offset:512
ds_read_b128 v[50:53], v102
ds_read_b128 v[54:57], v102 offset:128
ds_write_b32 v105, v76 offset:24768
ds_write_b32 v106, v77 offset:24768
s_setprio 1
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v90, v94, s[40:43], 0 offen
buffer_load_short_d16 v92, v96, s[40:43], 0 offen
buffer_load_short_d16 v91, v95, s[40:43], 0 offen
buffer_load_short_d16 v93, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(16) lgkmcnt(5)
ds_append v117 offset:65476
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 2295
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v58, v82, v58
v_pack_b32_f16 v59, v83, v59
v_pack_b32_f16 v60, v84, v60
v_pack_b32_f16 v61, v85, v61
v_cndmask_b32_dpp v58, v58, v58, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v59, v59, v59, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v60, v60, v60, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v61, v61, v61, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v58, v60, -1.0, v58 op_sel_hi:[1,0,1]
v_pk_mul_f16 v58, v58, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v61, v59, -1.0, v61 op_sel_hi:[1,0,1]
v_pk_mul_f16 v61, v61, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:4672
ds_read_b128 v[42:45], v102 offset:4160
ds_read_b128 v[46:49], v102 offset:4288
ds_write_b32 v103, v74 offset:24768
ds_write_b32 v104, v75 offset:24768
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v70, v94, s[40:43], 0 offen
buffer_load_short_d16 v72, v96, s[40:43], 0 offen
buffer_load_short_d16 v71, v95, s[40:43], 0 offen
buffer_load_short_d16 v73, v97, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc4
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 6
s_call_b64 s[38:39], 2165
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_pk_add_f16 v59, v60, v59
v_pk_mul_f16 v59, v59, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v60, v59, -1.0, v60 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v58 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v58, v115, v112, v58
v_mov_b32_dpp v115, v59 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v59, v115, v112, v59
v_mov_b32_dpp v115, v60 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v60, v115, v112, v60
v_mov_b32_dpp v115, v61 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v61, v115, v112, v61
s_setprio 0
s_nop 0
v_dot2_f32_f16 v33, v41, v57, v33
s_barrier
s_mov_b64 vcc, s[10:11]
s_nop 0
ds_read_b128 v[38:41], v107 offset:8768
ds_read_b128 v[50:53], v102 offset:8256
ds_read_b128 v[54:57], v102 offset:8384
ds_write_b32 v105, v80 offset:33024
ds_write_b32 v106, v81 offset:33024
s_setprio 1
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v82, v94, s[40:43], 0 offen
buffer_load_short_d16 v84, v96, s[40:43], 0 offen
buffer_load_short_d16 v83, v95, s[40:43], 0 offen
buffer_load_short_d16 v85, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(16) lgkmcnt(5)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 2039
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v62, v86, v62
v_pack_b32_f16 v63, v87, v63
v_pack_b32_f16 v64, v88, v64
v_pack_b32_f16 v65, v89, v65
v_cndmask_b32_dpp v62, v62, v62, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v63, v63, v63, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v64, v64, v64, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v65, v65, v65, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v62, v64, -1.0, v62 op_sel_hi:[1,0,1]
v_pk_mul_f16 v62, v62, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v65, v63, -1.0, v65 op_sel_hi:[1,0,1]
v_pk_mul_f16 v65, v65, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:12928
ds_read_b128 v[42:45], v102 offset:12416
ds_read_b128 v[46:49], v102 offset:12544
ds_write_b32 v103, v78 offset:33024
ds_write_b32 v104, v79 offset:33024
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v74, v94, s[40:43], 0 offen
buffer_load_short_d16 v76, v96, s[40:43], 0 offen
buffer_load_short_d16 v75, v95, s[40:43], 0 offen
buffer_load_short_d16 v77, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 1911
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_pk_add_f16 v63, v64, v63
v_pk_mul_f16 v63, v63, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v64, v63, -1.0, v64 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v62 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v62, v115, v112, v62
v_mov_b32_dpp v115, v63 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v63, v115, v112, v63
v_mov_b32_dpp v115, v64 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v64, v115, v112, v64
v_mov_b32_dpp v115, v65 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v65, v115, v112, v65
s_setprio 0
s_nop 0
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_nop 0
ds_read_b128 v[38:41], v107 offset:17024
ds_read_b128 v[50:53], v102 offset:16512
ds_read_b128 v[54:57], v102 offset:16640
ds_write_b32 v105, v60 offset:41280
ds_write_b32 v106, v61 offset:41280
s_setprio 1
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v86, v94, s[40:43], 0 offen
buffer_load_short_d16 v88, v96, s[40:43], 0 offen
buffer_load_short_d16 v87, v95, s[40:43], 0 offen
buffer_load_short_d16 v89, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(16) lgkmcnt(5)
ds_append v117 offset:65480
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 1783
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v66, v90, v66
v_pack_b32_f16 v67, v91, v67
v_pack_b32_f16 v68, v92, v68
v_pack_b32_f16 v69, v93, v69
v_cndmask_b32_dpp v66, v66, v66, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v67, v67, v67, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v68, v68, v68, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v69, v69, v69, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v66, v68, -1.0, v66 op_sel_hi:[1,0,1]
v_pk_mul_f16 v66, v66, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v69, v67, -1.0, v69 op_sel_hi:[1,0,1]
v_pk_mul_f16 v69, v69, 0.5 op_sel_hi:[1,0]
s_setprio 0
s_nop 0
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:21184
ds_read_b128 v[42:45], v102 offset:20672
ds_read_b128 v[46:49], v102 offset:20800
ds_write_b32 v103, v58 offset:41280
ds_write_b32 v104, v59 offset:41280
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v78, v94, s[40:43], 0 offen
buffer_load_short_d16 v80, v96, s[40:43], 0 offen
buffer_load_short_d16 v79, v95, s[40:43], 0 offen
buffer_load_short_d16 v81, v97, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc8
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 6
s_call_b64 s[38:39], 1653
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_pk_add_f16 v67, v68, v67
v_pk_mul_f16 v67, v67, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v68, v67, -1.0, v68 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v66 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v66, v115, v112, v66
v_mov_b32_dpp v115, v67 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v67, v115, v112, v67
v_mov_b32_dpp v115, v68 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v68, v115, v112, v68
v_mov_b32_dpp v115, v69 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v69, v115, v112, v69
s_setprio 0
s_nop 0
v_dot2_f32_f16 v33, v41, v57, v33
s_barrier
s_mov_b64 vcc, s[10:11]
s_nop 0
ds_read_b128 v[38:41], v107 offset:25280
ds_read_b128 v[50:53], v102 offset:24768
ds_read_b128 v[54:57], v102 offset:24896
ds_write_b32 v105, v64
ds_write_b32 v106, v65
s_setprio 1
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v90, v94, s[40:43], 0 offen
buffer_load_short_d16 v92, v96, s[40:43], 0 offen
buffer_load_short_d16 v91, v95, s[40:43], 0 offen
buffer_load_short_d16 v93, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(16) lgkmcnt(5)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 64008
s_call_b64 s[38:39], 1527
s_branch 64006
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v70, v82, v70
v_pack_b32_f16 v71, v83, v71
v_pack_b32_f16 v72, v84, v72
v_pack_b32_f16 v73, v85, v73
v_cndmask_b32_dpp v70, v70, v70, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v71, v71, v71, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v72, v72, v72, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v73, v73, v73, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v70, v71 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v70, v70, v71
v_mov_b32_dpp v71, v71 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v70, v71, v112, v70
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:29440
ds_read_b128 v[42:45], v102 offset:28928
ds_read_b128 v[46:49], v102 offset:29056
ds_write_b32 v103, v62
ds_write_b32 v104, v63
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v60, v96, s[40:43], 0 offen
buffer_load_short_d16 v59, v95, s[40:43], 0 offen
buffer_load_short_d16 v61, v97, s[40:43], 0 offen
s_add_u32 s91, s91, 0x200
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 1399
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_mov_b32_dpp v71, v73 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v71, v71, v73
v_mov_b32_dpp v73, v73 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v71, v73, v112, v71
v_mov_b32_dpp v73, v72 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v73, v73, v72
v_mov_b32_dpp v72, v72 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v73, v72, v112, v73
v_pk_add_f16 v72, v70, v73
v_pk_add_f16 v71, v71, v72
v_pk_mul_f16 v71, v71, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v72, -1.0, v71, v72 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_nop 0
ds_read_b128 v[38:41], v107 offset:33536
ds_read_b128 v[50:53], v102 offset:33024
ds_read_b128 v[54:57], v102 offset:33152
ds_write_b32 v105, v68 offset:8256
ds_write_b32 v106, v69 offset:8256
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v84, v96, s[40:43], 0 offen
buffer_load_short_d16 v83, v95, s[40:43], 0 offen
buffer_load_short_d16 v85, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
ds_append v117 offset:65472
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 1271
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v74, v86, v74
v_pack_b32_f16 v75, v87, v75
v_pack_b32_f16 v76, v88, v76
v_pack_b32_f16 v77, v89, v77
v_cndmask_b32_dpp v74, v74, v74, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v75, v75, v75, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v76, v76, v76, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v77, v77, v77, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v74, v75 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v74, v74, v75
v_mov_b32_dpp v75, v75 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v74, v75, v112, v74
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:37696
ds_read_b128 v[42:45], v102 offset:37184
ds_read_b128 v[46:49], v102 offset:37312
ds_write_b32 v103, v66 offset:8256
ds_write_b32 v104, v67 offset:8256
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v64, v96, s[40:43], 0 offen
buffer_load_short_d16 v63, v95, s[40:43], 0 offen
buffer_load_short_d16 v65, v97, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc0
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 1143
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_mov_b32_dpp v75, v77 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v75, v75, v77
v_mov_b32_dpp v77, v77 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v75, v77, v112, v75
v_mov_b32_dpp v77, v76 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v77, v77, v76
v_mov_b32_dpp v76, v76 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v77, v76, v112, v77
v_pk_add_f16 v76, v74, v77
v_pk_add_f16 v75, v75, v76
v_pk_mul_f16 v75, v75, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v76, -1.0, v75, v76 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v33, v41, v57, v33
s_barrier
s_mov_b64 vcc, s[10:11]
s_nop 0
ds_read_b128 v[38:41], v107 offset:41792
ds_read_b128 v[50:53], v102 offset:41280
ds_read_b128 v[54:57], v102 offset:41408
ds_write_b32 v105, v72 offset:16512
ds_write_b32 v106, v73 offset:16512
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v88, v96, s[40:43], 0 offen
buffer_load_short_d16 v87, v95, s[40:43], 0 offen
buffer_load_short_d16 v89, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 1015
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v78, v90, v78
v_pack_b32_f16 v79, v91, v79
v_pack_b32_f16 v80, v92, v80
v_pack_b32_f16 v81, v93, v81
v_cndmask_b32_dpp v78, v78, v78, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v79, v79, v79, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v80, v80, v80, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v81, v81, v81, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v78, v79 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v78, v78, v79
v_mov_b32_dpp v79, v79 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v78, v79, v112, v78
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:45952
ds_read_b128 v[42:45], v102 offset:45440
ds_read_b128 v[46:49], v102 offset:45568
ds_write_b32 v103, v70 offset:16512
ds_write_b32 v104, v71 offset:16512
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v68, v96, s[40:43], 0 offen
buffer_load_short_d16 v67, v95, s[40:43], 0 offen
buffer_load_short_d16 v69, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 2
s_call_b64 s[38:39], 889
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_mov_b32_dpp v79, v81 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v79, v79, v81
v_mov_b32_dpp v81, v81 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v79, v81, v112, v79
v_mov_b32_dpp v81, v80 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v81, v81, v80
v_mov_b32_dpp v80, v80 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v81, v80, v112, v81
v_pk_add_f16 v80, v78, v81
v_pk_add_f16 v79, v79, v80
v_pk_mul_f16 v79, v79, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v80, -1.0, v79, v80 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_nop 0
ds_read_b128 v[38:41], v107 offset:512
ds_read_b128 v[50:53], v102
ds_read_b128 v[54:57], v102 offset:128
ds_write_b32 v105, v76 offset:24768
ds_write_b32 v106, v77 offset:24768
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v92, v96, s[40:43], 0 offen
buffer_load_short_d16 v91, v95, s[40:43], 0 offen
buffer_load_short_d16 v93, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
ds_append v117 offset:65476
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 767
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v58, v82, v58
v_pack_b32_f16 v59, v83, v59
v_pack_b32_f16 v60, v84, v60
v_pack_b32_f16 v61, v85, v61
v_cndmask_b32_dpp v58, v58, v58, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v59, v59, v59, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v60, v60, v60, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v61, v61, v61, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v58, v59 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v58, v58, v59
v_mov_b32_dpp v59, v59 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v58, v59, v112, v58
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:4672
ds_read_b128 v[42:45], v102 offset:4160
ds_read_b128 v[46:49], v102 offset:4288
ds_write_b32 v103, v74 offset:24768
ds_write_b32 v104, v75 offset:24768
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v72, v96, s[40:43], 0 offen
buffer_load_short_d16 v71, v95, s[40:43], 0 offen
buffer_load_short_d16 v73, v97, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc4
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 639
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_mov_b32_dpp v59, v61 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v59, v59, v61
v_mov_b32_dpp v61, v61 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v59, v61, v112, v59
v_mov_b32_dpp v61, v60 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v61, v61, v60
v_mov_b32_dpp v60, v60 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v61, v60, v112, v61
v_pk_add_f16 v60, v58, v61
v_pk_add_f16 v59, v59, v60
v_pk_mul_f16 v59, v59, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v60, -1.0, v59, v60 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v33, v41, v57, v33
s_barrier
s_mov_b64 vcc, s[10:11]
s_nop 0
ds_read_b128 v[38:41], v107 offset:8768
ds_read_b128 v[50:53], v102 offset:8256
ds_read_b128 v[54:57], v102 offset:8384
ds_write_b32 v105, v80 offset:33024
ds_write_b32 v106, v81 offset:33024
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v84, v96, s[40:43], 0 offen
buffer_load_short_d16 v83, v95, s[40:43], 0 offen
buffer_load_short_d16 v85, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 511
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v62, v86, v62
v_pack_b32_f16 v63, v87, v63
v_pack_b32_f16 v64, v88, v64
v_pack_b32_f16 v65, v89, v65
v_cndmask_b32_dpp v62, v62, v62, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v63, v63, v63, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v64, v64, v64, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v65, v65, v65, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v62, v63 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v62, v62, v63
v_mov_b32_dpp v63, v63 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v62, v63, v112, v62
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:12928
ds_read_b128 v[42:45], v102 offset:12416
ds_read_b128 v[46:49], v102 offset:12544
ds_write_b32 v103, v78 offset:33024
ds_write_b32 v104, v79 offset:33024
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v76, v96, s[40:43], 0 offen
buffer_load_short_d16 v75, v95, s[40:43], 0 offen
buffer_load_short_d16 v77, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 2
s_call_b64 s[38:39], 385
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_mov_b32_dpp v63, v65 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v63, v63, v65
v_mov_b32_dpp v65, v65 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v63, v65, v112, v63
v_mov_b32_dpp v65, v64 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v65, v65, v64
v_mov_b32_dpp v64, v64 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v65, v64, v112, v65
v_pk_add_f16 v64, v62, v65
v_pk_add_f16 v63, v63, v64
v_pk_mul_f16 v63, v63, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v64, -1.0, v63, v64 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_nop 0
ds_read_b128 v[38:41], v107 offset:17024
ds_read_b128 v[50:53], v102 offset:16512
ds_read_b128 v[54:57], v102 offset:16640
ds_write_b32 v105, v60 offset:41280
ds_write_b32 v106, v61 offset:41280
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v88, v96, s[40:43], 0 offen
buffer_load_short_d16 v87, v95, s[40:43], 0 offen
buffer_load_short_d16 v89, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
ds_append v117 offset:65480
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 263
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v34, v43, v6
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_pack_b32_f16 v66, v90, v66
v_pack_b32_f16 v67, v91, v67
v_pack_b32_f16 v68, v92, v68
v_pack_b32_f16 v69, v93, v69
v_cndmask_b32_dpp v66, v66, v66, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v67, v67, v67, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v68, v68, v68, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v69, v69, v69, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v66, v67 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v66, v66, v67
v_mov_b32_dpp v67, v67 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v66, v67, v112, v66
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
ds_read_b128 v[34:37], v107 offset:21184
ds_read_b128 v[42:45], v102 offset:20672
ds_read_b128 v[46:49], v102 offset:20800
ds_write_b32 v103, v58 offset:41280
ds_write_b32 v104, v59 offset:41280
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v80, v96, s[40:43], 0 offen
buffer_load_short_d16 v79, v95, s[40:43], 0 offen
buffer_load_short_d16 v81, v97, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc8
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 8
s_call_b64 s[38:39], 135
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
s_setprio 1
s_nop 0
v_dot2_f32_f16 v6, v38, v51, v6
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_mov_b32_dpp v67, v69 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v67, v67, v69
v_mov_b32_dpp v69, v69 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v67, v69, v112, v67
v_mov_b32_dpp v69, v68 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v69, v69, v68
v_mov_b32_dpp v68, v68 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v69, v68, v112, v69
v_pk_add_f16 v68, v66, v69
v_pk_add_f16 v67, v67, v68
v_pk_mul_f16 v67, v67, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v68, -1.0, v67, v68 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v33, v41, v57, v33
s_barrier
s_mov_b64 vcc, s[10:11]
s_nop 0
ds_read_b128 v[38:41], v107 offset:25280
ds_read_b128 v[50:53], v102 offset:24768
ds_read_b128 v[54:57], v102 offset:24896
ds_write_b32 v105, v64
ds_write_b32 v106, v65
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x20000, s43
buffer_load_short_d16 v92, v96, s[40:43], 0 offen
buffer_load_short_d16 v91, v95, s[40:43], 0 offen
buffer_load_short_d16 v93, v97, s[40:43], 0 offen
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 64024
s_call_b64 s[38:39], 7
s_branch 64022
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_nop
s_cmp_eq_u32 s82, 0
s_cbranch_scc0 8
s_branch 612
s_add_u32 s82, s82, 3
s_andn2_b32 s82, s82, 3
s_bitcmp0_b32 s18, 26
s_cselect_b32 s52, s69, s70
s_cselect_b32 s53, 0, s71
s_sub_u32 s40, s40, s52
s_subb_u32 s41, s41, s53
s_cmp_eq_u32 s94, 0
s_cbranch_scc0 5
s_cbranch_scc1 642
s_nop 0
s_nop 0
s_add_u32 s94, s94, 1
s_andn2_b32 s94, s94, 1
s_min_u32 s72, s82, s94
s_sub_u32 s82, s82, s72
s_sub_u32 s94, s94, s72
s_sub_u32 s72, s72, 2
s_setpc_b64 s[38:39]
s_nop 0
s_bitcmp1_b32 s18, 17
s_cbranch_scc1 241
s_add_u32 s88, s88, s17
s_cmp_eq_u32 s88, 0
s_cbranch_scc1 238
s_mov_b32 s89, 0
s_bitcmp1_b32 s18, 16
s_cbranch_scc1 227
s_add_u32 s87, s16, 31
s_lshr_b32 s87, s87, 5
v_mov_b32_e32 v119, s88
v_mul_u32_u24_e32 v119, s87, v119
v_add_co_u32_e32 v119, vcc, s17, v119
v_sub_co_u32_e64 v119, vcc, v119, 1
v_ffbh_u32_e32 v122, s17
v_lshlrev_b32_e64 v123, v122, s17
v_and_b32_e32 v124, 0xffffff00, v123
v_cmp_eq_u32_e32 vcc, 0x80000000, v123
v_cvt_f32_u32_e32 v124, v124
v_rcp_f32_e32 v118, v124
v_subb_co_u32_e32 v121, vcc, 32, v122, vcc
v_cvt_f32_ubyte0_e32 v122, v123
v_fma_f32 v124, v124, v118, -1.0
v_fma_f32 v124, v122, v118, v124
v_madak_f32 v124, v124, v118, 0x9f000000
v_mul_f32_e32 v124, 0x5f800000, v124
v_mov_b32_e32 v122, 0
v_cvt_flr_i32_f32_e64 v124, -v124
v_lshl_add_u32 v118, v118, 9, v124
v_mad_u64_u32 v[122:123], vcc, v123, v118, v[122:123]
v_subb_co_u32_e64 v118, vcc, v118, -1, vcc
v_mul_hi_u32 v122, v119, v118
v_add_co_u32_e32 v118, vcc, v122, v119
v_addc_co_u32_e64 v122, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v121
v_cndmask_b32_e32 v118, v118, v122, vcc
v_alignbit_b32 v118, v122, v118, v121
v_readfirstlane_b32 s86, v118
v_mul_u32_u24_e64 v118, v118, s8
v_ffbh_u32_e32 v122, s87
v_lshlrev_b32_e64 v123, v122, s87
v_and_b32_e32 v124, 0xffffff00, v123
v_cmp_eq_u32_e32 vcc, 0x80000000, v123
v_cvt_f32_u32_e32 v124, v124
v_rcp_f32_e32 v119, v124
v_subb_co_u32_e32 v121, vcc, 32, v122, vcc
v_cvt_f32_ubyte0_e32 v122, v123
v_fma_f32 v124, v124, v119, -1.0
v_fma_f32 v124, v122, v119, v124
v_madak_f32 v124, v124, v119, 0x9f000000
v_mul_f32_e32 v124, 0x5f800000, v124
v_mov_b32_e32 v122, 0
v_cvt_flr_i32_f32_e64 v124, -v124
v_lshl_add_u32 v119, v119, 9, v124
v_mad_u64_u32 v[122:123], vcc, v123, v119, v[122:123]
v_subb_co_u32_e64 v119, vcc, v119, -1, vcc
v_mul_hi_u32 v122, v118, v119
v_add_co_u32_e32 v119, vcc, v122, v118
v_addc_co_u32_e64 v122, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v121
v_cndmask_b32_e32 v119, v119, v122, vcc
v_alignbit_b32 v119, v122, v119, v121
v_readfirstlane_b32 s52, v118
v_readfirstlane_b32 s84, v119
s_mul_i32 s84, s84, s87
s_sub_u32 s84, s52, s84
v_sub_co_u32_e32 v119, vcc, s8, v119
v_sub_co_u32_e32 v119, vcc, s17, v119
v_and_b32_e64 v121, v0, 63
v_cmp_eq_u32_e64 vcc, v121, 0
v_cndmask_b32_e32 v119, 1, v119, vcc
s_sub_u32 s58, 0, s75
s_sub_u32 s59, 0, s74
v_mul_u32_u24_e64 v123, v119, 32
v_ffbh_u32_e32 v126, s58
v_lshlrev_b32_e64 v127, v126, s58
v_and_b32_e32 v125, 0xffffff00, v127
v_cmp_eq_u32_e32 vcc, 0x80000000, v127
v_cvt_f32_u32_e32 v125, v125
v_rcp_f32_e32 v121, v125
v_subb_co_u32_e32 v124, vcc, 32, v126, vcc
v_cvt_f32_ubyte0_e32 v126, v127
v_fma_f32 v125, v125, v121, -1.0
v_fma_f32 v125, v126, v121, v125
v_madak_f32 v125, v125, v121, 0x9f000000
v_mul_f32_e32 v125, 0x5f800000, v125
v_mov_b32_e32 v126, 0
v_cvt_flr_i32_f32_e64 v125, -v125
v_lshl_add_u32 v121, v121, 9, v125
v_mad_u64_u32 v[126:127], vcc, v127, v121, v[126:127]
v_subb_co_u32_e64 v121, vcc, v121, -1, vcc
v_mul_hi_u32 v125, v123, v121
v_add_co_u32_e32 v121, vcc, v125, v123
v_addc_co_u32_e64 v125, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v124
v_cndmask_b32_e32 v121, v121, v125, vcc
v_alignbit_b32 v121, v125, v121, v124
v_mad_i32_i24 v122, v121, s75, v123
v_mul_u32_u24_e64 v123, v121, 1
v_ffbh_u32_e32 v126, s59
v_lshlrev_b32_e64 v127, v126, s59
v_and_b32_e32 v125, 0xffffff00, v127
v_cmp_eq_u32_e32 vcc, 0x80000000, v127
v_cvt_f32_u32_e32 v125, v125
v_rcp_f32_e32 v121, v125
v_subb_co_u32_e32 v124, vcc, 32, v126, vcc
v_cvt_f32_ubyte0_e32 v126, v127
v_fma_f32 v125, v125, v121, -1.0
v_fma_f32 v125, v126, v121, v125
v_madak_f32 v125, v125, v121, 0x9f000000
v_mul_f32_e32 v125, 0x5f800000, v125
v_mov_b32_e32 v126, 0
v_cvt_flr_i32_f32_e64 v125, -v125
v_lshl_add_u32 v121, v121, 9, v125
v_mad_u64_u32 v[126:127], vcc, v127, v121, v[126:127]
v_subb_co_u32_e64 v121, vcc, v121, -1, vcc
v_mul_hi_u32 v125, v123, v121
v_add_co_u32_e32 v121, vcc, v125, v123
v_addc_co_u32_e64 v125, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v124
v_cndmask_b32_e32 v121, v121, v125, vcc
v_alignbit_b32 v121, v125, v121, v124
v_mad_i32_i24 v123, v121, s74, v123
v_readfirstlane_b32 s76, v122
v_readfirstlane_b32 s77, v123
v_readfirstlane_b32 s78, v121
v_add_co_u32_e32 v108, vcc, s76, v108
v_addc_co_u32_e64 v124, vcc, 0, 0, vcc
v_mad_i32_i24 v108, v124, s75, v108
v_mad_i32_i24 v110, v124, s80, v110
v_mad_i32_i24 v109, v124, s79, v109
v_cmp_ge_i32_e64 vcc, v109, 0
v_addc_co_u32_e64 v124, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v124
v_mad_i32_i24 v109, v124, s74, v109
v_add_co_u32_e32 v109, vcc, s77, v109
v_addc_co_u32_e64 v124, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v124
v_mad_i32_i24 v109, v124, s74, v109
v_add_co_u32_e32 v110, vcc, s78, v110
v_readlane_b32 s76, v122, 1
v_readlane_b32 s77, v123, 1
v_readlane_b32 s78, v121, 1
s_add_u32 s85, s84, s86
s_cmp_le_u32 s85, s87
s_cselect_b32 s52, 0x20000, 0
s_cselect_b32 s85, s85, s87
s_or_b32 s18, s18, s52
s_lshl_b32 s84, s84, 5
s_lshl_b32 s85, s85, 5
s_min_u32 s85, s85, s16
s_cmp_eq_u32 s8, s17
s_cselect_b32 s52, 0x20000, 0
s_or_b32 s18, s18, s52
s_or_b32 s18, s18, s52
s_bitset1_b32 s18, 16
s_branch 43
s_lshr_b32 s84, s84, 5
s_add_u32 s85, s84, s86
s_sub_u32 s85, s85, s87
s_mov_b32 s84, 0
s_lshl_b32 s85, s85, 5
s_min_u32 s85, s85, s16
s_bitset1_b32 s18, 17
s_branch 12
s_bitset1_b32 s18, 18
s_mov_b32 s43, 0
s_mov_b32 s73, -1
s_mov_b32 s82, 40
s_branch 31
s_add_u32 s83, s83, 32
s_cmp_ge_u32 s83, s85
s_cbranch_scc0 28
s_bitset1_b32 s18, 22
s_sub_u32 s88, s88, s17
s_subb_u32 s89, s89, 0
s_cbranch_scc1 65281
v_add_co_u32_e32 v108, vcc, s76, v108
v_addc_co_u32_e64 v118, vcc, 0, 0, vcc
v_mad_i32_i24 v108, v118, s75, v108
v_mad_i32_i24 v110, v118, s80, v110
v_mad_i32_i24 v109, v118, s79, v109
v_cmp_ge_i32_e64 vcc, v109, 0
v_addc_co_u32_e64 v118, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v118
v_mad_i32_i24 v109, v118, s74, v109
v_add_co_u32_e32 v109, vcc, s77, v109
v_addc_co_u32_e64 v118, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v118
v_mad_i32_i24 v109, v118, s74, v109
v_add_co_u32_e32 v110, vcc, s78, v110
s_mov_b32 s83, s84
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccz 170
v_subrev_co_u32_e32 v118, vcc, s75, v108
v_subrev_co_u32_e32 v119, vcc, s74, v109
s_bitcmp1_b32 s18, 22
s_cbranch_scc0 64
s_bitset0_b32 s18, 22
s_bfe_u32 s52, s18, 0x10014
v_mul_u32_u24_e32 v123, 2, v118
v_mul_u32_u24_e32 v124, 2, v119
v_cvt_pk_u16_u32 v126, v123, v124
v_and_b32_e64 v123, v0, 1
v_cmp_eq_u32_e64 vcc, v123, 1
v_cndmask_b32_e32 v126, v110, v126, vcc
v_lshrrev_b32_e32 v122, 1, v0
v_bfe_u32 v127, v122, s52, 1
v_lshrrev_b32_e32 v122, 1, v0
v_bfi_b32 v122, 1, v0, v122
v_lshrrev_b32_e32 v123, 2, v0
v_bfi_b32 v123, 1, v0, v123
v_cmp_eq_u32_e64 vcc, s52, 0
v_cndmask_b32_e32 v122, v123, v122, vcc
s_sub_u32 s52, 1, s52
v_lshrrev_b32_e32 v123, s52, v122
v_bfi_b32 v122, 32, v123, v122
v_and_b32_e32 v122, 63, v122
v_add_co_u32_e32 v123, vcc, 16, v122
v_and_b32_e64 v124, v0, 2
v_cmp_eq_u32_e64 vcc, v124, 0
v_cndmask_b32_e32 v123, v123, v122, vcc
v_lshlrev_b32_e32 v124, 14, v127
v_mad_u32_u24 v123, 4, v123, v124
v_add_co_u32_e32 v122, vcc, s96, v123
ds_write_b32 v122, v126
v_writelane_b32 v124, s18, 0
v_writelane_b32 v124, s85, 1
v_writelane_b32 v124, s84, 2
v_and_b32_e64 v122, v0, 63
v_cmp_ge_u32_e64 vcc, v122, 3
v_mov_b32_e32 v125, 0x4000
v_cndmask_b32_e32 v122, v122, v125, vcc
v_mad_i32_i24 v122, v122, 4, s96
ds_write_b32 v122, v124 offset:256
s_add_u32 s96, s96, 0x18c
s_cmp_eq_u32 s96, 0xffc0
s_cselect_b32 s96, 0xc1e0, s96
v_mov_b32_dpp v120, v110 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v118, v118 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v119, v119 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_readfirstlane_b32 s81, v120
v_sub_co_u32_e64 v121, vcc, v120, s81
v_mul_lo_u32 v121, v121, s65
v_mad_u32_u24 v124, 5, v0, 2
v_and_b32_e32 v125, 6, v0
v_add_co_u32_e32 v125, vcc, 4, v125
v_bfi_b32 v125, 4, v124, v125
v_bfe_u32 v125, v125, 1, 3
v_ashrrev_i32_e64 v126, 0, s31
v_subrev_co_u32_e32 v125, vcc, v126, v125
v_ashrrev_i32_e64 v126, 0, s62
v_mad_i32_i24 v122, v126, 3, v125
v_add_co_u32_e64 v123, vcc, 0, s63
v_ashrrev_i32_e32 v123, 0, v123
v_add_co_u32_e64 v124, vcc, 0, s30
v_ashrrev_i32_e32 v124, 0, v124
v_sub_i32 v123, v123, v124
s_lshl_b32 s54, s15, 1
v_cmp_ge_u32_e64 s[52:53], v120, s12
v_mad_i32_i24 v118, v118, 4, v122
v_cmp_ge_u32_e64 s[56:57], v118, s15
v_mad_i32_i24 v118, 2, v118, v121
s_or_b64 s[56:57], s[56:57], s[52:53]
v_mad_i32_i24 v119, v119, 4, v123
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v94, v119, s54, v118
v_cndmask_b32_e64 v94, v94, -1, s[58:59]
v_add_co_u32_e32 v119, vcc, 2, v119
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v95, v119, s54, v118
v_cndmask_b32_e64 v95, v95, -1, s[58:59]
v_add_co_u32_e32 v119, vcc, 2, v119
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v96, v119, s54, v118
v_cndmask_b32_e64 v96, v96, -1, s[58:59]
v_add_co_u32_e32 v119, vcc, 2, v119
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v97, v119, s54, v118
v_cndmask_b32_e64 v97, v97, -1, s[58:59]
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 154
s_lshr_b32 s52, -1, 16
s_and_b32 s52, s52, s65
s_lshr_b32 s53, s65, 16
s_mul_i32 s53, s53, s81
s_mul_i32 s40, s52, s81
s_lshl_b32 s52, s53, 16
s_lshr_b32 s53, s53, 16
s_add_u32 s40, s52, s40
s_addc_u32 s41, s53, 0
s_add_u32 s40, s40, s20
s_addc_u32 s41, s41, s21
s_lshr_b32 s52, s18, 6
s_xor_b32 s52, s52, s18
s_and_b32 s52, s52, 0x80000
s_cselect_b32 s52, s68, 0
s_and_b32 s53, s18, 0x1100000
s_cselect_b32 s53, 0, 1
s_lshl_b32 s52, s52, s53
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, 0
s_branch 99
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 129
v_mad_u32_u24 v120, 5, v0, 2
v_lshlrev_b32_e32 v118, 1, v0
v_bfi_b32 v120, 4, v120, v118
v_bfe_u32 v118, v120, 2, 2
v_min_u32_e32 v118, 2, v118
v_bfe_u32 v120, v0, 1, 1
v_mad_u32_u24 v118, 2, v118, v120
v_mad_u32_u24 v118, s62, 3, v118
v_sub_co_u32_e32 v120, vcc, s29, v118
v_sub_co_u32_e64 v120, vcc, v120, 1
s_bfe_u32 s54, s18, 0x10001
v_cmp_eq_u32_e64 vcc, s54, 1
v_cndmask_b32_e32 v118, v118, v120, vcc
v_cmp_ge_u32_e64 s[52:53], v118, s29
v_lshlrev_b32_e32 v118, 1, v118
s_bfe_u32 s54, s18, 0x10018
v_bfe_u32 v121, v0, 2, s54
v_mul_lo_u32 v121, s68, v121
v_add_co_u32_e32 v118, vcc, v118, v121
v_mul_lo_u32 v119, s90, v111
v_add_co_u32_e32 v119, vcc, v119, v118
s_sub_u32 s54, s28, s63
s_sub_u32 s54, s54, 5
s_bitcmp1_b32 s18, 0
s_cselect_b32 s54, s54, s63
v_mov_b32_e32 v121, s54
s_lshl_b32 s57, s29, 1
v_cmp_ge_u32_e64 s[54:55], v121, s28
v_mad_i32_i24 v94, v121, s57, v119
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v94, v94, -1, s[54:55]
v_mov_b32_e32 v95, v94
v_add_co_u32_e64 v121, vcc, v121, 2
v_cmp_ge_u32_e64 s[54:55], v121, s28
v_mad_i32_i24 v97, v121, s57, v119
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v97, v97, -1, s[54:55]
v_add_co_u32_e64 v121, vcc, v121, 2
v_cmp_ge_u32_e64 s[54:55], v121, s28
v_mad_i32_i24 v96, v121, s57, v119
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v96, v96, -1, s[54:55]
v_add_co_u32_e64 v118, vcc, v111, s83
v_cmp_lt_u32_e64 vcc, v118, s16
v_cndmask_b32_e32 v94, -1, v94, vcc
v_cndmask_b32_e32 v95, -1, v95, vcc
v_cndmask_b32_e32 v96, -1, v96, vcc
v_cndmask_b32_e32 v97, -1, v97, vcc
s_lshr_b32 s52, -1, 16
s_and_b32 s52, s52, s90
s_lshr_b32 s53, s90, 16
s_mul_i32 s53, s53, s83
s_mul_i32 s40, s52, s83
s_lshl_b32 s52, s53, 16
s_lshr_b32 s53, s53, 16
s_add_u32 s40, s52, s40
s_addc_u32 s41, s53, 0
s_add_u32 s40, s40, s22
s_addc_u32 s41, s41, s23
s_lshr_b32 s52, s18, 6
s_xor_b32 s52, s52, s18
s_and_b32 s52, s52, 0x80000
s_cselect_b32 s52, s68, 0
s_and_b32 s53, s18, 0x1100000
s_cselect_b32 s53, 0, 1
s_lshl_b32 s52, s52, s53
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, 0
s_mov_b32 s43, 0x20000
s_mov_b32 s73, -1
s_bitcmp0_b32 s13, 0
s_cbranch_scc1 4
s_mov_b32 s43, 0
s_mov_b32 s73, 1
s_sub_u32 s40, s40, s68
s_subb_u32 s41, s41, 0
s_add_u32 s53, s13, 1
s_and_b32 s53, s53, -2
s_bfe_u32 s52, s18, 0x10014
s_lshl_b32 s82, s53, s52
s_bfe_u32 s52, s18, 0x10013
s_bfe_u32 s54, s18, 0x10019
s_xor_b32 s52, s52, s54
s_cselect_b32 s52, 2, 0
s_cselect_b32 s43, 0x20000, s43
s_and_b32 s52, s52, s82
s_sub_u32 s82, s82, s52
s_bitcmp1_b32 s18, 20
s_cselect_b32 s52, 0, 0x2000000
s_bitcmp1_b32 s53, 1
s_cselect_b32 s52, s52, 0
s_xor_b32 s18, s18, s52
s_cmp_eq_u32 s82, 0
s_cbranch_scc1 9
s_mov_b64 vcc, s[10:11]
s_branch 64931
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_and_b32 s52, 0x900000, s18
s_subb_u32 s62, s62, 1
s_cbranch_scc0 65219
s_and_b32 s52, 0x900000, s18
s_subb_u32 s62, s61, 1
s_add_u32 s63, s63, 6
s_cmp_ge_u32 s63, s28
s_cbranch_scc0 65213
s_mov_b32 s63, 1
s_cmp_ge_u32 s63, s28
s_addc_u32 s64, s64, 1
s_cmp_gt_u32 s64, 1
s_cbranch_scc0 65208
s_mov_b32 s64, 0
s_mov_b32 s63, 0
s_branch 65174
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_dpp v4, v4, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v5, v5, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v2, v2, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v3, v3, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v3, v4, v3 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v2, v5, v2 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v3, v3, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v2, v2, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v2, v3, v2 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v2, v2
v_mac_f32_dpp v8, v8, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v9, v9, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v6, v6, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v7, v7, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v7, v8, v7 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v6, v9, v6 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v7, v7, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v6, v6, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v3, v7, v6 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v3, v3
v_mac_f32_dpp v12, v12, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v13, v13, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v10, v10, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v11, v11, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v11, v12, v11 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v10, v13, v10 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v11, v11, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v10, v10, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v4, v11, v10 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v4, v4
v_mac_f32_dpp v16, v16, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v17, v17, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v14, v14, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v15, v15, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v15, v16, v15 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v14, v17, v14 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v15, v15, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v14, v14, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v5, v15, v14 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v5, v5
v_mac_f32_dpp v20, v20, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v21, v21, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v18, v18, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v19, v19, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v19, v20, v19 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v18, v21, v18 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v19, v19, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v18, v18, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v6, v19, v18 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v6, v6
v_mac_f32_dpp v24, v24, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v25, v25, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v22, v22, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v23, v23, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v23, v24, v23 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v22, v25, v22 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v23, v23, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v22, v22, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v7, v23, v22 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v7, v7
v_mac_f32_dpp v28, v28, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v29, v29, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v26, v26, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v27, v27, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v27, v28, v27 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v26, v29, v26 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v27, v27, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v26, v26, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v8, v27, v26 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v8, v8
v_mac_f32_dpp v32, v32, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v33, v33, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v30, v30, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v31, v31, v113 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v31, v32, v31 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v30, v33, v30 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v31, v31, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v30, v30, v114 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v9, v31, v30 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v9, v9
s_waitcnt vmcnt(0)
v_readlane_b32 s55, v116, 0
v_add_f16_e64 v2, v2, s55
v_mul_f16_e64 v118, v2, s36
v_cmp_lt_f16_e64 vcc, v2, 0
v_cndmask_b32_e32 v2, v2, v118, vcc
buffer_store_short v2, v98, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v116, 1
v_add_f16_e64 v3, v3, s55
v_mul_f16_e64 v118, v3, s36
v_cmp_lt_f16_e64 vcc, v3, 0
v_cndmask_b32_e32 v3, v3, v118, vcc
buffer_store_short v3, v98, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v116, 2
v_add_f16_e64 v4, v4, s55
v_mul_f16_e64 v118, v4, s36
v_cmp_lt_f16_e64 vcc, v4, 0
v_cndmask_b32_e32 v4, v4, v118, vcc
buffer_store_short v4, v98, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v116, 3
v_add_f16_e64 v5, v5, s55
v_mul_f16_e64 v118, v5, s36
v_cmp_lt_f16_e64 vcc, v5, 0
v_cndmask_b32_e32 v5, v5, v118, vcc
buffer_store_short v5, v98, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
s_lshl_b32 s52, s67, 2
s_add_u32 s44, s44, s52
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 4
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v116, 8
v_add_f16_e64 v6, v6, s55
v_mul_f16_e64 v118, v6, s36
v_cmp_lt_f16_e64 vcc, v6, 0
v_cndmask_b32_e32 v6, v6, v118, vcc
buffer_store_short v6, v98, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v116, 9
v_add_f16_e64 v7, v7, s55
v_mul_f16_e64 v118, v7, s36
v_cmp_lt_f16_e64 vcc, v7, 0
v_cndmask_b32_e32 v7, v7, v118, vcc
buffer_store_short v7, v98, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v116, 10
v_add_f16_e64 v8, v8, s55
v_mul_f16_e64 v118, v8, s36
v_cmp_lt_f16_e64 vcc, v8, 0
v_cndmask_b32_e32 v8, v8, v118, vcc
buffer_store_short v8, v98, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v116, 11
v_add_f16_e64 v9, v9, s55
v_mul_f16_e64 v118, v9, s36
v_cmp_lt_f16_e64 vcc, v9, 0
v_cndmask_b32_e32 v9, v9, v118, vcc
buffer_store_short v9, v98, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
s_add_u32 s44, s44, s52
s_addc_u32 s45, s45, 0
s_lshl_b32 s52, s52, 2
s_add_u32 s44, s44, s52
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 20
s_cselect_b32 s47, 0, s47
s_cselect_b32 s51, 0, s51
s_add_u32 s48, s48, 64
s_addc_u32 s49, s49, 0
s_sub_u32 s50, s50, 64
s_cselect_b32 s51, 0, s51
v_mov_b32_e32 v2, 0
v_mov_b32_e32 v3, 0
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
s_xor_b32 s18, s18, 0x200000
s_bitcmp1_b32 s13, 0
s_addc_u32 s52, s13, 0
s_mul_i32 s94, s60, s61
s_mul_i32 s94, s94, s52
s_add_u32 s52, s93, s92
s_cmp_lt_i32 s52, 0
s_cbranch_scc0 104
v_and_b32_e32 v98, 0x7f, v0
v_lshrrev_b32_e32 v98, 1, v98
v_bfi_b32 v98, 1, v0, v98
v_and_b32_e64 v99, v0, 2
v_mad_u32_u24 v98, v99, 16, v98
v_lshlrev_b32_e32 v98, 2, v98
v_add_co_u32_e64 v98, vcc, v98, s97
v_and_b32_e32 v99, 3, v0
v_lshlrev_b32_e32 v99, 2, v99
v_add_co_u32_e64 v99, vcc, v99, s97
ds_read_b32 v120, v99 offset:256
ds_read_b32 v98, v98
s_add_u32 s97, s97, 0x18c
s_cmp_eq_u32 s97, 0xffc0
s_cselect_b32 s97, 0xc1e0, s97
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s95, v98
v_readlane_b32 s54, v120, 0
s_bitcmp1_b32 s54, 18
s_cbranch_scc1 80
v_readlane_b32 s52, v120, 1
v_readlane_b32 s53, v120, 2
s_add_u32 s93, s92, s53
s_lshr_b32 s55, -1, 16
s_and_b32 s55, s55, s66
s_lshr_b32 s56, s66, 16
s_mul_i32 s56, s56, s95
s_mul_i32 s44, s55, s95
s_lshl_b32 s55, s56, 16
s_lshr_b32 s56, s56, 16
s_add_u32 s44, s55, s44
s_addc_u32 s45, s56, 0
s_add_u32 s44, s44, s24
s_addc_u32 s45, s45, s25
s_mul_i32 s55, s67, s93
s_add_u32 s44, s44, s55
s_addc_u32 s45, s45, 0
s_mov_b32 s47, 0x20000
s_bitcmp1_b32 s18, 7
s_cselect_b32 s51, 0x20000, 0
s_lshl_b32 s55, s93, 1
s_add_u32 s48, s34, s55
s_addc_u32 s49, s35, 0
s_lshl_b32 s56, s52, 1
s_sub_u32 s50, s56, s55
s_cselect_b32 s51, 0, s51
s_sub_u32 s93, s52, s53
s_sub_u32 s93, s93, 1
s_sub_u32 s93, s93, s92
s_cselect_b32 s47, 0, s47
v_bfe_u32 v118, v98, 16, 16
v_bfe_u32 v119, v98, 0, 16
v_and_b32_e64 v120, v0, 7
v_sub_co_u32_e32 v121, vcc, 7, v120
v_min_u32_e32 v120, v120, v121
v_bfe_u32 v121, v120, 1, 1
v_bfe_u32 v120, v120, 0, 1
v_mov_b32_dpp v118, v118 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v119, v119 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_add_co_u32_e32 v118, vcc, v118, v121
v_add_co_u32_e32 v119, vcc, v119, v120
v_mov_b32_dpp v120, v98 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[52:53], v120, s12
v_sub_co_u32_e64 v120, vcc, v120, s95
v_mul_lo_u32 v120, v120, s66
v_mad_i32_i24 v98, v118, s33, v119
v_lshlrev_b32_e32 v98, 1, v98
v_add_co_u32_e32 v98, vcc, v98, v120
v_cmp_ge_u32_e64 s[58:59], v119, s33
s_or_b64 s[56:57], s[58:59], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v118, s32
s_or_b64 s[52:53], s[56:57], s[54:55]
v_cndmask_b32_e64 v98, v98, -1, s[52:53]
v_and_b32_e64 v116, v0, 63
v_lshlrev_b32_e32 v116, 1, v116
s_barrier
buffer_load_ushort v116, v116, s[48:51], 0 offen
s_mov_b64 vcc, s[10:11]
s_branch 64437
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
