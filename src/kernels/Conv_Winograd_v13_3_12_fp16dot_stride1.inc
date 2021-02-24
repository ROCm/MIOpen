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

s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_mov_b32_e32 v115, 0
v_mov_b32_e32 v116, 0
v_mov_b32_e32 v113, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s99, 0xdfc0
s_mov_b32 s98, 0xdfc0
s_nop 0
s_nop 0
v_readfirstlane_b32 s56, v0
s_cmp_eq_u32 s56, 0
s_cmovk_i32 s57, 0xfeef
s_cmovk_i32 s87, 0x10
s_cmp_eq_u32 s56, 64
s_cmovk_i32 s57, 0xfddf
s_cmovk_i32 s87, 0x10
s_cmp_eq_u32 s56, 0x80
s_cmovk_i32 s57, 0xeffe
s_cmovk_i32 s87, 0x14
s_cmp_eq_u32 s56, 0xc0
s_cmovk_i32 s57, 0xdffd
s_cmovk_i32 s87, 0x14
s_cmp_eq_u32 s56, 0x100
s_cmovk_i32 s57, 0xfbfb
s_cmovk_i32 s87, 0x18
s_cmp_eq_u32 s56, 0x140
s_cmovk_i32 s57, 0xf7f7
s_cmovk_i32 s87, 0x18
s_cmp_eq_u32 s56, 0x180
s_cmovk_i32 s57, 0xbfbf
s_cmovk_i32 s87, 0x1c
s_cmp_eq_u32 s56, 0x1c0
s_cmovk_i32 s57, 0x7f7f
s_cmovk_i32 s87, 0x1c
s_mov_b32 s58, 0x1010101
s_mul_i32 s59, s87, s58
s_mov_b32 s58, 0x3020100
s_add_i32 s59, s59, s58
v_mov_b32_e32 v114, s59
v_and_b32_e64 v111, v0, 63
v_and_b32_e64 v117, v0, 15
v_bfe_u32 v117, s57, v117, 1
v_bfe_u32 v118, v0, 4, 2
v_cmp_eq_u32_e64 vcc, v118, 2
v_addc_co_u32_e64 v118, vcc, 0, 0, vcc
v_or_b32_e32 v117, v117, v118
v_mad_u32_u24 v111, 64, v117, v111
s_or_b32 s87, s87, 0x20000
s_add_u32 s87, 0xffc0, s87
ds_write_b8 v111, v111 offset:65472
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
s_barrier
v_readfirstlane_b32 s56, v0
s_lshr_b32 s56, s56, 4
s_and_b32 s94, s56, 8
s_subb_u32 s94, 0, 0
s_xnor_b32 s94, s94, s56
s_and_b32 s94, s94, 20
s_mov_b64 s[40:41], s[6:7]
s_load_dwordx16 s[12:27], s[40:41], 0x0
s_load_dwordx4 s[28:31], s[40:41], 0x40
s_load_dwordx2 s[32:33], s[40:41], 0x50
s_waitcnt lgkmcnt(0)
s_bitcmp1_b32 s18, 6
s_cbranch_scc0 16
s_and_b32 s23, s23, 0xffff
s_and_b32 s25, s25, 0xffff
s_and_b32 s21, s21, 0xffff
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
s_mov_b32 s42, 0
s_cmp_gt_u32 s12, 0xffff
s_addc_u32 s42, s42, 0
s_cmp_lt_u32 s12, 1
s_addc_u32 s42, s42, 0
s_cmp_gt_u32 s13, 0xffff
s_addc_u32 s42, s42, 0
s_cmp_lt_u32 s13, 1
s_addc_u32 s42, s42, 0
s_cmp_gt_u32 s14, 0xffff
s_addc_u32 s42, s42, 0
s_cmp_lt_u32 s14, 1
s_addc_u32 s42, s42, 0
s_cmp_gt_u32 s15, 0xffff
s_addc_u32 s42, s42, 0
s_cmp_lt_u32 s15, 1
s_addc_u32 s42, s42, 0
s_cmp_gt_u32 s16, 0xffff
s_addc_u32 s42, s42, 0
s_cmp_lt_u32 s16, 1
s_addc_u32 s42, s42, 0
s_cmp_gt_u32 s17, 0xffff
s_addc_u32 s42, s42, 0
s_cmp_lt_u32 s17, 0
s_addc_u32 s42, s42, 0
s_cmp_gt_u32 s28, 0xffff
s_addc_u32 s42, s42, 0
s_cmp_lt_u32 s28, 1
s_addc_u32 s42, s42, 0
s_cmp_gt_u32 s29, 0xffff
s_addc_u32 s42, s42, 0
s_cmp_lt_u32 s29, 1
s_addc_u32 s42, s42, 0
s_cmp_gt_u32 s32, 0xffff
s_addc_u32 s42, s42, 0
s_cmp_lt_u32 s32, 0
s_addc_u32 s42, s42, 0
s_cmp_gt_u32 s33, 0xffff
s_addc_u32 s42, s42, 0
s_cmp_lt_u32 s33, 0
s_addc_u32 s42, s42, 0
s_mul_i32 s44, s14, s15
s_lshr_b32 s48, -1, 16
s_and_b32 s48, s48, s44
s_lshr_b32 s49, s44, 16
s_mul_i32 s49, s49, s13
s_mul_i32 s46, s48, s13
s_lshl_b32 s48, s49, 16
s_lshr_b32 s49, s49, 16
s_add_u32 s46, s48, s46
s_addc_u32 s47, s49, 0
s_cmp_gt_u32 s46, 0x10000000
s_addc_u32 s42, s47, s42
s_lshl_b32 s69, s46, 1
s_lshl_b32 s72, s44, 1
s_mul_i32 s45, s32, s33
s_lshr_b32 s48, -1, 16
s_and_b32 s48, s48, s45
s_lshr_b32 s49, s45, 16
s_mul_i32 s49, s49, s16
s_mul_i32 s46, s48, s16
s_lshl_b32 s48, s49, 16
s_lshr_b32 s49, s49, 16
s_add_u32 s46, s48, s46
s_addc_u32 s47, s49, 0
s_cmp_gt_u32 s46, 0x10000000
s_addc_u32 s42, s47, s42
s_lshl_b32 s70, s46, 1
s_lshl_b32 s71, s45, 1
s_cmp_eq_u32 s42, 0
s_cbranch_scc0 4530
s_bitcmp1_b32 s18, 7
s_cbranch_scc0 7
s_bitcmp1_b32 s18, 6
s_cbranch_scc0 5
s_waitcnt lgkmcnt(0)
s_and_b32 s35, s35, 0xffff
s_load_dwordx2 s[34:35], s[34:35], 0x0
s_and_b32 s18, s18, 0x1c7
s_mul_i32 s44, s28, s29
s_lshl_b32 s44, s44, 1
s_bitcmp1_b32 s18, 2
s_cselect_b32 s45, s16, s13
s_lshr_b32 s48, -1, 16
s_and_b32 s48, s48, s44
s_lshr_b32 s49, s44, 16
s_mul_i32 s49, s49, s45
s_mul_i32 s46, s48, s45
s_lshl_b32 s48, s49, 16
s_lshr_b32 s49, s49, 16
s_add_u32 s46, s48, s46
s_addc_u32 s47, s49, 0
s_cmp_gt_u32 s46, 2.0
s_addc_u32 s42, s47, s42
s_mov_b32 s45, s46
s_bitcmp1_b32 s18, 2
s_cselect_b32 s73, s45, s44
s_cselect_b32 s74, s44, s45
s_lshl_b32 s76, s73, 1
s_waitcnt lgkmcnt(0)
s_and_b32 s23, s23, 0xffff
s_and_b32 s25, s25, 0xffff
s_and_b32 s21, s21, 0xffff
s_and_b32 s27, s27, 0xffff
s_and_b32 s35, s35, 0xffff
v_cvt_f16_f32_e32 v2, s36
v_readfirstlane_b32 s36, v2
s_nop 0
s_nop 0
s_and_b32 s46, 0, s30
s_addc_u32 s46, s32, 0
s_ashr_i32 s46, s46, 0
s_add_u32 s44, s46, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s44
v_readfirstlane_b32 s44, v2
s_andn2_b32 s46, 0, s31
s_addc_u32 s46, s33, 0
s_ashr_i32 s46, s46, 0
s_add_u32 s45, s46, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s45
v_readfirstlane_b32 s45, v2
s_sub_u32 s78, 0, s45
s_sub_u32 s77, 0, s44
s_add_u32 s64, s28, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s64
v_readfirstlane_b32 s64, v2
s_add_u32 s65, s29, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s65
v_readfirstlane_b32 s65, v2
v_mad_i32_i24 v2, 3, s64, -2
v_sub_co_u32_e64 v2, vcc, v2, s28
v_addc_co_u32_e64 v2, vcc, 0, 0, vcc
v_readfirstlane_b32 s46, v2
s_and_b32 s46, s46, 0
s_and_b32 s46, s46, s64
s_add_u32 s64, s64, s46
v_readfirstlane_b32 s47, v0
s_and_b32 s50, s47, 64
s_cselect_b32 s50, 0x80000, 0
s_or_b32 s18, s18, s50
s_lshl_b32 s75, s72, 1
s_lshl_b32 s46, s65, 0
s_cmp_eq_u32 s46, 1
s_cbranch_scc0 5
s_bitcmp1_b32 s18, 2
s_cselect_b32 s50, 0, 0x1000000
s_or_b32 s18, s18, s50
s_branch 6
s_bitset1_b32 s18, 23
s_bitset1_b32 s18, 20
s_lshr_b32 s75, s75, 1
s_lshr_b32 s76, s76, 1
s_add_u32 s65, s65, 1
s_and_b32 s65, s65, -2
v_bfe_u32 v3, v0, 2, 6
v_lshrrev_b32_e32 v107, 1, v3
v_readfirstlane_b32 s50, v0
s_bitcmp1_b32 s18, 24
s_cselect_b32 s50, s50, -1
s_bitcmp0_b32 s50, 8
s_cselect_b32 s50, 0x80000, 0
s_bitcmp1_b32 s18, 20
s_cselect_b32 s50, 0x80000, s50
s_andn2_b32 s18, s18, s50
s_cmp_eq_u32 s50, 0
s_cselect_b32 s50, 15, 0
v_bfi_b32 v107, s50, v3, v107
s_mul_i32 s92, s12, s44
s_sub_u32 s92, s92, 1
s_lshr_b32 s92, s92, 0
s_add_u32 s92, s92, 1
s_lshr_b32 s48, -1, 16
s_and_b32 s48, s48, s92
s_lshr_b32 s49, s92, 16
s_mul_i32 s49, s49, s45
s_mul_i32 s92, s48, s45
s_lshl_b32 s48, s49, 16
s_lshr_b32 s49, s49, 16
s_add_u32 s92, s48, s92
s_addc_u32 s93, s49, 0
s_sub_u32 s92, s92, 1
s_subb_u32 s93, s93, 0
s_lshr_b64 s[92:93], s[92:93], 5
s_add_u32 s92, s92, 1
s_addc_u32 s93, s93, 0
v_mov_b32_e32 v4, s8
v_mov_b32_e32 v5, s17
v_and_b32_e32 v6, 3, v0
v_cmp_eq_u32_e32 vcc, 2, v6
v_cndmask_b32_e32 v4, v4, v5, vcc
v_cmp_eq_u32_e32 vcc, 1, v6
v_cndmask_b32_e32 v7, 0, v107, vcc
v_cmp_eq_u32_e64 s[48:49], 3, v6
v_bfe_u32 v105, v7, 0, 5
v_mad_u32_u24 v105, v4, 32, v105
v_cvt_f32_u32_e32 v8, s45
v_rcp_f32_e32 v8, v8
v_mul_f32_e32 v8, 0x4f800000, v8
v_cvt_u32_f32_e32 v8, v8
v_mul_lo_u32 v9, s45, v8
v_mul_hi_u32 v10, s45, v8
v_sub_co_u32_e32 v11, vcc, 0, v9
v_cmp_ne_i32_e64 s[50:51], 0, v10
v_cndmask_b32_e64 v9, v11, v9, s[50:51]
v_mul_hi_u32 v9, v9, v8
v_sub_co_u32_e32 v10, vcc, v8, v9
v_add_co_u32_e32 v8, vcc, v8, v9
v_cndmask_b32_e64 v8, v8, v10, s[50:51]
v_mul_hi_u32 v8, v8, v105
v_mul_lo_u32 v9, v8, s45
v_sub_co_u32_e32 v10, vcc, v105, v9
v_cmp_ge_u32_e64 s[50:51], v105, v9
v_cmp_ge_u32_e64 s[52:53], v10, s45
v_add_co_u32_e32 v10, vcc, 1, v8
s_and_b64 s[52:53], s[50:51], s[52:53]
v_add_co_u32_e32 v9, vcc, -1, v8
v_cndmask_b32_e64 v10, v8, v10, s[52:53]
v_cndmask_b32_e64 v10, v9, v10, s[50:51]
v_cmp_ne_i32_e64 vcc, 0, s45
v_cndmask_b32_e32 v106, -1, v10, vcc
v_mad_i32_i24 v104, v106, s78, v105
v_lshrrev_b32_e32 v105, 5, v7
v_mad_u32_u24 v105, v106, 1, v105
v_cndmask_b32_e64 v105, v105, 1, s[48:49]
v_cvt_f32_u32_e32 v8, s44
v_rcp_f32_e32 v8, v8
v_mul_f32_e32 v8, 0x4f800000, v8
v_cvt_u32_f32_e32 v8, v8
v_mul_lo_u32 v9, s44, v8
v_mul_hi_u32 v10, s44, v8
v_sub_co_u32_e32 v11, vcc, 0, v9
v_cmp_ne_i32_e64 s[50:51], 0, v10
v_cndmask_b32_e64 v9, v11, v9, s[50:51]
v_mul_hi_u32 v9, v9, v8
v_sub_co_u32_e32 v10, vcc, v8, v9
v_add_co_u32_e32 v8, vcc, v8, v9
v_cndmask_b32_e64 v8, v8, v10, s[50:51]
v_mul_hi_u32 v8, v8, v105
v_mul_lo_u32 v9, v8, s44
v_sub_co_u32_e32 v10, vcc, v105, v9
v_cmp_ge_u32_e64 s[50:51], v105, v9
v_cmp_ge_u32_e64 s[52:53], v10, s44
v_add_co_u32_e32 v10, vcc, 1, v8
s_and_b64 s[52:53], s[50:51], s[52:53]
v_add_co_u32_e32 v9, vcc, -1, v8
v_cndmask_b32_e64 v10, v8, v10, s[52:53]
v_cndmask_b32_e64 v10, v9, v10, s[50:51]
v_cmp_ne_i32_e64 vcc, 0, s44
v_cndmask_b32_e32 v106, -1, v10, vcc
v_mad_i32_i24 v105, v106, s77, v105
v_readlane_b32 s79, v104, 2
v_readlane_b32 s80, v105, 2
v_readlane_b32 s81, v106, 2
v_readlane_b32 s82, v105, 3
v_readlane_b32 s83, v106, 3
v_add_co_u32_e64 v104, vcc, v104, s78
v_add_co_u32_e64 v105, vcc, v105, s77
v_mov_b32_dpp v106, v106  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v104, v104  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v105, v105  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x20000
s_mov_b32 s46, 0x80000000
s_mov_b32 s47, 0x20000
s_mov_b32 s50, 0x80000000
s_mov_b32 s51, 0x20000
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 7
v_xor_b32_dpp v108, v0, v0  quad_perm:[1,3,2,2] row_mask:0xf bank_mask:0xf
v_subrev_co_u32_e32 v108, vcc, 1, v108
v_cvt_f16_i16_e32 v108, v108
v_pk_add_f16 v108, v108, 0 op_sel_hi:[0,0]
s_branch 6
v_xor_b32_dpp v108, v0, v0  quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32_e32 v108, vcc, 1, v108
v_cvt_f16_i16_e32 v108, v108
v_pk_add_f16 v108, v108, 0 op_sel_hi:[0,0]
v_mov_b32_e32 v109, 1
v_xor_b32_dpp v109, v0, v0  quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v109, v0, v0  quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32_e32 v109, vcc, 1, v109
v_mov_b32_e32 v110, 1
v_xor_b32_dpp v110, v0, v0  quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v110, v0, v0  quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32_e32 v110, vcc, 1, v110
v_cvt_f32_i32_e32 v109, v109
v_cvt_f32_i32_e32 v110, v110
v_lshrrev_b32_e64 v117, 2, s94
v_and_b32_e32 v118, 3, v0
v_lshrrev_b32_e32 v119, 1, v0
v_bfi_b32 v119, 64, v119, v0
v_bfe_u32 v119, v119, 4, 3
v_mad_u32_u24 v102, v119, 4, v118
v_lshlrev_b32_e32 v102, 4, v102
v_mad_u32_u24 v103, v117, 4, v118
v_lshlrev_b32_e32 v103, 4, v103
v_bfe_u32 v117, v0, 2, 2
v_and_b32_e32 v118, 1, v117
v_mad_u32_u24 v120, v117, 16, v118
v_lshlrev_b32_e32 v120, 6, v120
v_xor_b32_e32 v103, v103, v120
v_mul_u32_u24_e32 v120, 0x400, v117
v_xor_b32_e32 v102, v102, v120
s_lshr_b32 s94, s94, 0
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 50
s_and_b32 s57, s18, 0x1100000
s_addc_u32 s57, 0, 0
v_lshrrev_b32_e32 v120, 1, v0
s_mul_i32 s56, 60, s57
s_sub_u32 s56, 63, s56
v_bfi_b32 v120, s56, v0, v120
v_and_b32_e32 v117, 1, v120
v_bfe_u32 v118, v120, 1, 1
v_xor_b32_e32 v117, v117, v118
v_bfe_u32 v119, v120, 3, 1
v_mad_u32_u24 v118, v118, 2, v119
v_mul_u32_u24_e32 v117, 0x118, v117
v_bfe_u32 v119, v120, 2, 1
v_mad_u32_u24 v118, v118, 2, v117
v_xor_b32_e32 v118, v118, v119
v_and_b32_e32 v119, 0xf0, v120
v_xor_b32_e32 v118, v118, v119
s_mul_i32 s56, 4, s57
s_sub_u32 s56, 6, s56
v_bfe_u32 v120, v0, s56, 1
v_mul_u32_u24_e32 v120, 0x1040, v120
v_xor_b32_e32 v99, 0x314, v118
v_xor_b32_e32 v100, 0x31c, v118
v_xor_b32_e32 v101, 8, v118
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v98, v118, v101, vcc
v_cndmask_b32_e32 v101, v101, v118, vcc
v_mad_u32_u24 v98, 4, v98, v120
v_mad_u32_u24 v99, 4, v99, v120
v_mad_u32_u24 v100, 4, v100, v120
v_mad_u32_u24 v101, 4, v101, v120
s_branch 44
s_bfe_u32 s57, s18, 0x10014
v_lshrrev_b32_e32 v120, 1, v0
s_mul_i32 s56, 60, s57
s_sub_u32 s56, 63, s56
v_bfi_b32 v120, s56, v0, v120
v_and_b32_e32 v117, 1, v120
v_bfe_u32 v118, v120, 1, 1
v_bfe_u32 v119, v120, 3, 1
v_xor_b32_e32 v117, v117, v118
v_mad_u32_u24 v118, v118, 2, v119
v_mul_u32_u24_e32 v117, 0x109, v117
v_bfe_u32 v119, v120, 2, 1
v_mad_u32_u24 v118, v118, 2, v117
v_xor_b32_e32 v118, v118, v119
v_and_b32_e32 v119, 0xf0, v120
v_or_b32_e32 v118, v118, v119
s_mul_i32 s56, 4, s57
s_sub_u32 s56, 6, s56
v_bfe_u32 v120, v0, s56, 1
v_mul_u32_u24_e32 v120, 0x1040, v120
v_mad_u32_u24 v98, 4, v118, v120
v_xor_b32_e32 v99, 0x307, v118
v_mad_u32_u24 v99, 4, v99, v120
v_xor_b32_e32 v100, 0x30f, v118
v_mad_u32_u24 v100, 4, v100, v120
v_xor_b32_e32 v101, 8, v118
v_mad_u32_u24 v101, 4, v101, v120
v_subrev_co_u32_e32 v104, vcc, s79, v104
v_mov_b32_e32 v118, s78
v_cmp_lt_i32_e32 vcc, v104, v118
v_subb_co_u32_e32 v117, vcc, 0, v113, vcc
v_mad_i32_i24 v104, v117, s78, v104
v_mad_i32_i24 v106, v117, s83, v106
v_mad_i32_i24 v105, v117, s82, v105
v_mov_b32_e32 v118, s77
v_cmp_lt_i32_e32 vcc, v105, v118
v_subb_co_u32_e32 v117, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v117
v_mad_i32_i24 v105, v117, v118, v105
v_subrev_co_u32_e32 v105, vcc, s80, v105
v_cmp_lt_i32_e32 vcc, v105, v118
v_subb_co_u32_e32 v117, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v117
v_mad_i32_i24 v105, v117, s77, v105
v_subrev_co_u32_e32 v106, vcc, s81, v106
s_mov_b32 s66, 0
s_mov_b32 s67, s28
s_mov_b32 s68, 1
s_mov_b32 s88, 0
s_mov_b32 s89, s16
s_mov_b32 s86, s89
s_sub_u32 s95, -1, s94
s_sub_u32 s95, s95, 32
s_bitset1_b32 s18, 21
s_mov_b32 s51, 0
s_mov_b32 s55, 0
s_mov_b32 s96, 32
s_mov_b32 s100, 0
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 954
s_branch 2497
s_nop 0
s_nop 0
s_bitcmp1_b32 s18, 17
s_cbranch_scc1 246
s_add_u32 s92, s92, s17
s_cmp_eq_u32 s92, 0
s_cbranch_scc1 243
s_mov_b32 s93, 0
s_bitcmp1_b32 s18, 16
s_cbranch_scc1 232
s_add_u32 s91, s16, 31
s_lshr_b32 s91, s91, 5
v_mov_b32_e32 v118, s92
v_mul_u32_u24_e32 v118, s91, v118
v_add_co_u32_e32 v118, vcc, s17, v118
v_sub_co_u32_e64 v118, vcc, v118, 1
v_cvt_f32_u32_e32 v120, s17
v_rcp_f32_e32 v120, v120
v_mul_f32_e32 v120, 0x4f800000, v120
v_cvt_u32_f32_e32 v120, v120
v_mul_lo_u32 v121, s17, v120
v_mul_hi_u32 v122, s17, v120
v_sub_co_u32_e32 v123, vcc, 0, v121
v_cmp_ne_i32_e64 s[58:59], 0, v122
v_cndmask_b32_e64 v121, v123, v121, s[58:59]
v_mul_hi_u32 v121, v121, v120
v_sub_co_u32_e32 v122, vcc, v120, v121
v_add_co_u32_e32 v120, vcc, v120, v121
v_cndmask_b32_e64 v120, v120, v122, s[58:59]
v_mul_hi_u32 v120, v120, v118
v_mul_lo_u32 v121, v120, s17
v_sub_co_u32_e32 v122, vcc, v118, v121
v_cmp_ge_u32_e64 s[58:59], v118, v121
v_cmp_ge_u32_e64 s[60:61], v122, s17
v_add_co_u32_e32 v122, vcc, 1, v120
s_and_b64 s[60:61], s[58:59], s[60:61]
v_add_co_u32_e32 v121, vcc, -1, v120
v_cndmask_b32_e64 v122, v120, v122, s[60:61]
v_cndmask_b32_e64 v122, v121, v122, s[58:59]
v_cmp_ne_i32_e64 vcc, 0, s17
v_cndmask_b32_e32 v117, -1, v122, vcc
v_readfirstlane_b32 s90, v117
v_mul_u32_u24_e64 v117, v117, s8
v_cvt_f32_u32_e32 v120, s91
v_rcp_f32_e32 v120, v120
v_mul_f32_e32 v120, 0x4f800000, v120
v_cvt_u32_f32_e32 v120, v120
v_mul_lo_u32 v121, s91, v120
v_mul_hi_u32 v122, s91, v120
v_sub_co_u32_e32 v123, vcc, 0, v121
v_cmp_ne_i32_e64 s[58:59], 0, v122
v_cndmask_b32_e64 v121, v123, v121, s[58:59]
v_mul_hi_u32 v121, v121, v120
v_sub_co_u32_e32 v122, vcc, v120, v121
v_add_co_u32_e32 v120, vcc, v120, v121
v_cndmask_b32_e64 v120, v120, v122, s[58:59]
v_mul_hi_u32 v120, v120, v117
v_mul_lo_u32 v121, v120, s91
v_sub_co_u32_e32 v122, vcc, v117, v121
v_cmp_ge_u32_e64 s[58:59], v117, v121
v_cmp_ge_u32_e64 s[60:61], v122, s91
v_add_co_u32_e32 v122, vcc, 1, v120
s_and_b64 s[60:61], s[58:59], s[60:61]
v_add_co_u32_e32 v121, vcc, -1, v120
v_cndmask_b32_e64 v122, v120, v122, s[60:61]
v_cndmask_b32_e64 v122, v121, v122, s[58:59]
v_cmp_ne_i32_e64 vcc, 0, s91
v_cndmask_b32_e32 v118, -1, v122, vcc
v_readfirstlane_b32 s56, v117
v_readfirstlane_b32 s88, v118
s_mul_i32 s88, s88, s91
s_sub_u32 s88, s56, s88
v_sub_co_u32_e32 v118, vcc, s8, v118
v_sub_co_u32_e32 v118, vcc, s17, v118
v_and_b32_e64 v120, v0, 63
v_cmp_eq_u32_e64 vcc, v120, 0
v_cndmask_b32_e32 v118, 1, v118, vcc
s_sub_u32 s62, 0, s78
s_sub_u32 s63, 0, s77
v_mul_u32_u24_e64 v122, v118, 32
v_cvt_f32_u32_e32 v123, s62
v_rcp_f32_e32 v123, v123
v_mul_f32_e32 v123, 0x4f800000, v123
v_cvt_u32_f32_e32 v123, v123
v_mul_lo_u32 v124, s62, v123
v_mul_hi_u32 v125, s62, v123
v_sub_co_u32_e32 v126, vcc, 0, v124
v_cmp_ne_i32_e64 s[58:59], 0, v125
v_cndmask_b32_e64 v124, v126, v124, s[58:59]
v_mul_hi_u32 v124, v124, v123
v_sub_co_u32_e32 v125, vcc, v123, v124
v_add_co_u32_e32 v123, vcc, v123, v124
v_cndmask_b32_e64 v123, v123, v125, s[58:59]
v_mul_hi_u32 v123, v123, v122
v_mul_lo_u32 v124, v123, s62
v_sub_co_u32_e32 v125, vcc, v122, v124
v_cmp_ge_u32_e64 s[58:59], v122, v124
v_cmp_ge_u32_e64 s[60:61], v125, s62
v_add_co_u32_e32 v125, vcc, 1, v123
s_and_b64 s[60:61], s[58:59], s[60:61]
v_add_co_u32_e32 v124, vcc, -1, v123
v_cndmask_b32_e64 v125, v123, v125, s[60:61]
v_cndmask_b32_e64 v125, v124, v125, s[58:59]
v_cmp_ne_i32_e64 vcc, 0, s62
v_cndmask_b32_e32 v120, -1, v125, vcc
v_mad_i32_i24 v121, v120, s78, v122
v_mul_u32_u24_e64 v122, v120, 1
v_cvt_f32_u32_e32 v123, s63
v_rcp_f32_e32 v123, v123
v_mul_f32_e32 v123, 0x4f800000, v123
v_cvt_u32_f32_e32 v123, v123
v_mul_lo_u32 v124, s63, v123
v_mul_hi_u32 v125, s63, v123
v_sub_co_u32_e32 v126, vcc, 0, v124
v_cmp_ne_i32_e64 s[58:59], 0, v125
v_cndmask_b32_e64 v124, v126, v124, s[58:59]
v_mul_hi_u32 v124, v124, v123
v_sub_co_u32_e32 v125, vcc, v123, v124
v_add_co_u32_e32 v123, vcc, v123, v124
v_cndmask_b32_e64 v123, v123, v125, s[58:59]
v_mul_hi_u32 v123, v123, v122
v_mul_lo_u32 v124, v123, s63
v_sub_co_u32_e32 v125, vcc, v122, v124
v_cmp_ge_u32_e64 s[58:59], v122, v124
v_cmp_ge_u32_e64 s[60:61], v125, s63
v_add_co_u32_e32 v125, vcc, 1, v123
s_and_b64 s[60:61], s[58:59], s[60:61]
v_add_co_u32_e32 v124, vcc, -1, v123
v_cndmask_b32_e64 v125, v123, v125, s[60:61]
v_cndmask_b32_e64 v125, v124, v125, s[58:59]
v_cmp_ne_i32_e64 vcc, 0, s63
v_cndmask_b32_e32 v120, -1, v125, vcc
v_mad_i32_i24 v122, v120, s77, v122
v_readfirstlane_b32 s79, v121
v_readfirstlane_b32 s80, v122
v_readfirstlane_b32 s81, v120
v_add_co_u32_e32 v104, vcc, s79, v104
v_addc_co_u32_e32 v123, vcc, 0, v113, vcc
v_mad_i32_i24 v104, v123, s78, v104
v_mad_i32_i24 v106, v123, s83, v106
v_mad_i32_i24 v105, v123, s82, v105
v_cmp_ge_i32_e64 vcc, v105, 0
v_addc_co_u32_e32 v123, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v123
v_mad_i32_i24 v105, v123, s77, v105
v_add_co_u32_e32 v105, vcc, s80, v105
v_addc_co_u32_e32 v123, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v123
v_mad_i32_i24 v105, v123, s77, v105
v_add_co_u32_e32 v106, vcc, s81, v106
v_readlane_b32 s79, v121, 1
v_readlane_b32 s80, v122, 1
v_readlane_b32 s81, v120, 1
s_add_u32 s89, s88, s90
s_cmp_le_u32 s89, s91
s_cselect_b32 s56, 0x20000, 0
s_cselect_b32 s89, s89, s91
s_or_b32 s18, s18, s56
s_lshl_b32 s88, s88, 5
s_lshl_b32 s89, s89, 5
s_min_u32 s89, s89, s16
s_cmp_eq_u32 s8, s17
s_cselect_b32 s56, 0x20000, 0
s_or_b32 s18, s18, s56
s_or_b32 s18, s18, s56
s_bitset1_b32 s18, 16
s_branch 40
s_lshr_b32 s88, s88, 5
s_add_u32 s89, s88, s90
s_sub_u32 s89, s89, s91
s_mov_b32 s88, 0
s_lshl_b32 s89, s89, 5
s_min_u32 s89, s89, s16
s_bitset1_b32 s18, 17
s_branch 12
s_bitset1_b32 s18, 18
s_mov_b32 s43, 0
s_mov_b32 s47, 0
s_mov_b32 s85, 16
s_branch 254
s_add_u32 s86, s86, 32
s_cmp_ge_u32 s86, s89
s_cbranch_scc0 25
s_bitset1_b32 s18, 22
s_sub_u32 s92, s92, s17
s_subb_u32 s93, s93, 0
s_cbranch_scc1 65276
v_add_co_u32_e32 v104, vcc, s79, v104
v_addc_co_u32_e32 v117, vcc, 0, v113, vcc
v_mad_i32_i24 v104, v117, s78, v104
v_mad_i32_i24 v106, v117, s83, v106
v_mad_i32_i24 v105, v117, s82, v105
v_cmp_ge_i32_e64 vcc, v105, 0
v_addc_co_u32_e32 v117, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v117
v_mad_i32_i24 v105, v117, s77, v105
v_add_co_u32_e32 v105, vcc, s80, v105
v_addc_co_u32_e32 v117, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v117
v_mad_i32_i24 v105, v117, s77, v105
v_add_co_u32_e32 v106, vcc, s81, v106
s_mov_b32 s86, s88
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccz 131
v_readfirstlane_b32 s84, v106
v_cmp_ge_u32_e64 s[60:61], v106, s12
v_subrev_co_u32_e32 v118, vcc, s78, v104
v_subrev_co_u32_e32 v119, vcc, s77, v105
s_bfe_u32 s62, s18, 0x10014
v_lshrrev_b32_e32 v90, 2, v0
v_and_b32_e32 v120, s62, v90
s_bitcmp1_b32 s18, 22
s_cbranch_scc0 38
s_bitset0_b32 s18, 22
v_and_b32_e64 v92, v0, 1
v_mad_i32_i24 v91, v118, 2, v92
v_bfe_u32 v92, v0, 1, 1
v_mad_i32_i24 v92, v119, 2, v92
v_cvt_pk_u16_u32 v93, v91, v92
v_mul_u32_u24_e32 v91, s62, v90
v_mad_i32_i24 v91, -2, v91, v0
v_bfe_u32 v90, v90, s62, 1
v_mul_u32_u24_e32 v90, 3, v90
s_sub_u32 s62, 1, s62
v_lshrrev_b32_e32 v92, s62, v91
v_bfi_b32 v91, 64, v92, v91
v_and_b32_e32 v91, 0x7f, v91
v_xor_b32_e32 v91, v91, v90
v_lshlrev_b32_e32 v92, 14, v120
v_mad_u32_u24 v91, 4, v91, v92
v_add_co_u32_e32 v91, vcc, s98, v91
ds_write_b32 v91, v106
ds_write_b32 v91, v93 offset:512
s_add_u32 s98, s98, 0x800
s_cmp_eq_u32 s98, 0xffc0
s_cselect_b32 s98, 0xdfc0, s98
v_sub_co_u32_e64 v117, vcc, v106, s84
v_and_b32_e64 v91, v0, 3
v_ashrrev_i32_e64 v92, 0, s31
v_subrev_co_u32_e32 v91, vcc, v92, v91
v_ashrrev_i32_e64 v92, 0, s66
v_mad_i32_i24 v91, v92, 3, v91
v_mad_i32_i24 v91, v120, 3, v91
v_mad_i32_i24 v118, v118, 2, v91
v_cmp_ge_u32_e64 s[56:57], v118, s15
v_mul_lo_u32 v120, v117, s69
v_mad_i32_i24 v118, 2, v118, v120
s_or_b64 s[56:57], s[56:57], s[60:61]
v_add_co_u32_e64 v92, vcc, 0, s67
v_ashrrev_i32_e32 v92, 0, v92
v_mad_i32_i24 v119, v119, 2, v92
v_add_co_u32_e64 v92, vcc, 0, s30
v_ashrrev_i32_e32 v92, 0, v92
v_subrev_co_u32_e32 v119, vcc, v92, v119
s_lshl_b32 s63, s15, 1
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v90, v119, s63, v118
v_cndmask_b32_e64 v90, v90, -1, s[58:59]
v_add_co_u32_e32 v119, vcc, 1, v119
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v91, v119, s63, v118
v_cndmask_b32_e64 v91, v91, -1, s[58:59]
v_add_co_u32_e32 v119, vcc, 1, v119
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v92, v119, s63, v118
v_cndmask_b32_e64 v92, v92, -1, s[58:59]
v_add_co_u32_e32 v119, vcc, 1, v119
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v93, v119, s63, v118
v_cndmask_b32_e64 v93, v93, -1, s[58:59]
s_lshr_b32 s56, -1, 16
s_and_b32 s56, s56, s69
s_lshr_b32 s57, s69, 16
s_mul_i32 s57, s57, s84
s_mul_i32 s40, s56, s84
s_lshl_b32 s56, s57, 16
s_lshr_b32 s57, s57, 16
s_add_u32 s40, s56, s40
s_addc_u32 s41, s57, 0
s_add_u32 s40, s40, s20
s_addc_u32 s41, s41, s21
s_and_b32 s56, s18, 0x80000
s_cselect_b32 s56, s72, 0
s_add_u32 s40, s40, s56
s_addc_u32 s41, s41, 0
s_mov_b32 s43, 0x20000
s_branch 88
s_bfe_u32 s56, s18, 0x10014
v_bfe_u32 v117, v0, 0, 2
v_min_u32_e32 v117, 2, v117
v_bfe_u32 v119, v0, 2, s56
v_mad_u32_u24 v117, v119, 3, v117
v_mad_u32_u24 v117, s66, 3, v117
v_sub_co_u32_e32 v119, vcc, s29, v117
v_sub_co_u32_e64 v119, vcc, v119, 1
s_bfe_u32 s58, s18, 0x10001
v_cmp_eq_u32_e64 vcc, s58, 1
v_cndmask_b32_e32 v117, v117, v119, vcc
v_cmp_ge_u32_e64 s[56:57], v117, s29
v_lshlrev_b32_e32 v117, 1, v117
s_bfe_u32 s58, s18, 0x10018
v_bfe_u32 v120, v0, 2, s58
v_mad_u32_u24 v117, s73, v120, v117
v_mad_u32_u24 v118, s74, v107, v117
s_sub_u32 s58, s28, s67
s_sub_u32 s58, s58, 3
s_bitcmp1_b32 s18, 0
s_cselect_b32 s58, s58, s67
v_mov_b32_e32 v120, s58
s_lshl_b32 s61, s29, 1
v_cmp_ge_u32_e64 s[58:59], v120, s28
v_mad_i32_i24 v90, v120, s61, v118
s_or_b64 s[58:59], s[58:59], s[56:57]
v_cndmask_b32_e64 v90, v90, -1, s[58:59]
v_mov_b32_e32 v91, v90
v_add_co_u32_e64 v120, vcc, v120, 1
v_cmp_ge_u32_e64 s[58:59], v120, s28
v_mad_i32_i24 v93, v120, s61, v118
s_or_b64 s[58:59], s[58:59], s[56:57]
v_cndmask_b32_e64 v93, v93, -1, s[58:59]
v_add_co_u32_e64 v120, vcc, v120, 1
v_cmp_ge_u32_e64 s[58:59], v120, s28
v_mad_i32_i24 v92, v120, s61, v118
s_or_b64 s[58:59], s[58:59], s[56:57]
v_cndmask_b32_e64 v92, v92, -1, s[58:59]
v_add_co_u32_e64 v117, vcc, v107, s86
v_cmp_lt_u32_e64 vcc, v117, s16
v_cndmask_b32_e32 v90, -1, v90, vcc
v_cndmask_b32_e32 v91, -1, v91, vcc
v_cndmask_b32_e32 v92, -1, v92, vcc
v_cndmask_b32_e32 v93, -1, v93, vcc
s_lshr_b32 s56, -1, 16
s_and_b32 s56, s56, s74
s_lshr_b32 s57, s74, 16
s_mul_i32 s57, s57, s86
s_mul_i32 s44, s56, s86
s_lshl_b32 s56, s57, 16
s_lshr_b32 s57, s57, 16
s_add_u32 s44, s56, s44
s_addc_u32 s45, s57, 0
s_add_u32 s44, s44, s22
s_addc_u32 s45, s45, s23
s_and_b32 s56, s18, 0x80000
s_cselect_b32 s56, s73, 0
s_add_u32 s44, s44, s56
s_addc_u32 s45, s45, 0
s_mov_b32 s47, 0x20000
s_bfe_u32 s56, s18, 0x10014
s_sub_u32 s85, s13, 1
s_lshl_b32 s85, s85, s56
s_add_u32 s56, s38, 0x16c
s_addc_u32 s57, s39, 0
s_sub_u32 s56, s56, s100
s_subb_u32 s57, s57, 0
s_nop 0
s_setpc_b64 s[56:57]
s_and_b32 s56, 0x900000, s18
s_subb_u32 s66, s66, 1
s_cbranch_scc0 65299
s_and_b32 s56, 0x900000, s18
s_subb_u32 s66, s65, 1
s_add_u32 s67, s67, 3
s_cmp_ge_u32 s67, s28
s_cbranch_scc0 65293
s_mov_b32 s67, 0
s_branch 65263
v_mac_f32_dpp v4, v4, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v5, v5, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v2, v2, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v3, v3, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v3, v4, v3  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v2, v5, v2  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v3, v3, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v2, v2, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v34, v3, v2  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v34, v34
v_mac_f32_dpp v8, v8, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v9, v9, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v6, v6, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v7, v7, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v7, v8, v7  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v6, v9, v6  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v7, v7, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v6, v6, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v35, v7, v6  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v35, v35
v_mac_f32_dpp v12, v12, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v13, v13, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v10, v10, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v11, v11, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v11, v12, v11  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v10, v13, v10  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v11, v11, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v10, v10, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v36, v11, v10  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v36, v36
v_mac_f32_dpp v16, v16, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v17, v17, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v14, v14, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v15, v15, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v15, v16, v15  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v14, v17, v14  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v15, v15, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v14, v14, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v37, v15, v14  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v37, v37
v_mac_f32_dpp v20, v20, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v21, v21, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v18, v18, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v19, v19, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v19, v20, v19  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v18, v21, v18  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v19, v19, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v18, v18, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v38, v19, v18  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v38, v38
v_mac_f32_dpp v24, v24, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v25, v25, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v22, v22, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v23, v23, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v23, v24, v23  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v22, v25, v22  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v23, v23, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v22, v22, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v39, v23, v22  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v39, v39
v_mac_f32_dpp v28, v28, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v29, v29, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v26, v26, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v27, v27, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v27, v28, v27  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v26, v29, v26  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v27, v27, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v26, v26, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v40, v27, v26  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v40, v40
v_mac_f32_dpp v32, v32, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v33, v33, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v30, v30, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v31, v31, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v31, v32, v31  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v30, v33, v30  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v31, v31, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v30, v30, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v41, v31, v30  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v41, v41
v_readlane_b32 s59, v116, 0
v_add_f16_e64 v34, v34, s59
v_mul_f16_e64 v119, v34, s36
v_max_f16_e32 v34, v34, v119
buffer_store_short v34, v94, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
v_readlane_b32 s59, v116, 1
v_add_f16_e64 v35, v35, s59
v_mul_f16_e64 v119, v35, s36
v_max_f16_e32 v35, v35, v119
buffer_store_short v35, v94, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
v_readlane_b32 s59, v116, 2
v_add_f16_e64 v36, v36, s59
v_mul_f16_e64 v119, v36, s36
v_max_f16_e32 v36, v36, v119
buffer_store_short v36, v94, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
v_readlane_b32 s59, v116, 3
v_add_f16_e64 v37, v37, s59
v_mul_f16_e64 v119, v37, s36
v_max_f16_e32 v37, v37, v119
buffer_store_short v37, v94, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
s_lshl_b32 s56, s71, 2
s_add_u32 s48, s48, s56
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 4
s_cselect_b32 s51, 0, s51
v_readlane_b32 s59, v116, 8
v_add_f16_e64 v38, v38, s59
v_mul_f16_e64 v119, v38, s36
v_max_f16_e32 v38, v38, v119
buffer_store_short v38, v94, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
v_readlane_b32 s59, v116, 9
v_add_f16_e64 v39, v39, s59
v_mul_f16_e64 v119, v39, s36
v_max_f16_e32 v39, v39, v119
buffer_store_short v39, v94, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
v_readlane_b32 s59, v116, 10
v_add_f16_e64 v40, v40, s59
v_mul_f16_e64 v119, v40, s36
v_max_f16_e32 v40, v40, v119
buffer_store_short v40, v94, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
v_readlane_b32 s59, v116, 11
v_add_f16_e64 v41, v41, s59
v_mul_f16_e64 v119, v41, s36
v_max_f16_e32 v41, v41, v119
buffer_store_short v41, v94, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
s_add_u32 s48, s48, s56
s_addc_u32 s49, s49, 0
s_lshl_b32 s56, s56, 2
s_add_u32 s48, s48, s56
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 20
s_cselect_b32 s51, 0, s51
s_cselect_b32 s55, 0, s55
s_add_u32 s52, s52, 64
s_addc_u32 s53, s53, 0
s_sub_u32 s54, s54, 64
s_cselect_b32 s55, 0, s55
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
s_mul_i32 s96, s64, s65
s_mul_i32 s96, s96, s13
s_sub_u32 s96, s96, 1
s_add_u32 s56, s95, s94
s_cmp_lt_i32 s56, 0
s_cbranch_scc0 74
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 3204
v_lshrrev_b32_e32 v119, 1, v0
v_bfi_b32 v119, 64, v119, v0
v_and_b32_e32 v94, 0x7f, v119
v_lshlrev_b32_e32 v94, 2, v94
v_add_co_u32_e64 v94, vcc, v94, s99
ds_read_b32 v95, v94 offset:512
ds_read_b32 v94, v94
s_add_u32 s99, s99, 0x800
s_cmp_eq_u32 s99, 0xffc0
s_cselect_b32 s99, 0xdfc0, s99
s_waitcnt lgkmcnt(0)
v_bfe_u32 v117, v95, 16, 16
v_bfe_u32 v118, v95, 0, 16
v_readfirstlane_b32 s97, v94
v_sub_co_u32_e64 v119, vcc, v94, s97
v_mul_lo_u32 v119, v119, s70
v_cmp_ge_u32_e64 s[56:57], v94, s12
v_mad_i32_i24 v94, v117, s33, v118
v_mad_i32_i24 v94, 2, v94, v119
v_cmp_ge_u32_e64 s[62:63], v118, s33
s_or_b64 s[60:61], s[62:63], s[56:57]
v_cmp_ge_u32_e64 s[58:59], v117, s32
s_or_b64 s[56:57], s[60:61], s[58:59]
v_cndmask_b32_e64 v94, v94, -1, s[56:57]
s_add_u32 s95, s94, s88
s_lshr_b32 s56, -1, 16
s_and_b32 s56, s56, s70
s_lshr_b32 s57, s70, 16
s_mul_i32 s57, s57, s97
s_mul_i32 s48, s56, s97
s_lshl_b32 s56, s57, 16
s_lshr_b32 s57, s57, 16
s_add_u32 s48, s56, s48
s_addc_u32 s49, s57, 0
s_add_u32 s48, s48, s24
s_addc_u32 s49, s49, s25
s_mul_i32 s56, s71, s95
s_add_u32 s48, s48, s56
s_addc_u32 s49, s49, 0
s_mov_b32 s51, 0x20000
s_bitcmp1_b32 s18, 7
s_cselect_b32 s55, 0x20000, 0
s_lshl_b32 s56, s95, 1
s_add_u32 s52, s34, s56
s_addc_u32 s53, s35, 0
s_lshl_b32 s57, s89, 1
s_sub_u32 s54, s57, s56
s_cselect_b32 s55, 0, s55
s_sub_u32 s95, s89, s88
s_sub_u32 s95, s95, 1
s_sub_u32 s95, s95, s94
s_cselect_b32 s51, 0, s51
v_and_b32_e64 v116, v0, 63
v_lshlrev_b32_e32 v116, 1, v116
buffer_load_ushort v116, v116, s[52:55], 0 offen
s_add_u32 s56, s38, 0x12c
s_addc_u32 s57, s39, 0
s_setpc_b64 s[56:57]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_getpc_b64 s[38:39]
s_sub_u32 s38, s38, 0x114
s_subb_u32 s39, s39, 0
s_branch 65094
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
s_cbranch_scc0 65076
buffer_load_short_d16 v66, v90, s[40:43], 0 offen
ds_read_b128 v[42:45], v102 offset:25280
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v47, v59, v7
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v70
s_mov_b32 m0, -1
v_pk_fma_f16 v78, v80, -1.0, v78 op_sel_hi:[1,0,1]
v_pk_mul_f16 v78, v78, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v12, v48, v60, v12
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v68, v92, s[40:43], 0 offen
ds_read_b128 v[50:53], v103 offset:24768
s_getpc_b64 s[38:39]
v_pk_fma_f16 v81, v79, -1.0, v81 op_sel_hi:[1,0,1]
v_pk_mul_f16 v81, v81, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v79, v80, v79
v_pk_mul_f16 v79, v79, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v80, v79, -1.0, v80 op_sel_hi:[1,0,1]
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v71
s_add_u32 s56, s40, s75
v_mov_b32_dpp v112, v78  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v67, v91, s[40:43], 0 offen
ds_read_b128 v[54:57], v103 offset:24896
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v69, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 64976
s_setprio 0
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v66, v90, s[40:43], 0 offen
ds_read_b128 v[46:49], v102 offset:29440
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v72
s_mov_b32 m0, -1
v_pk_fma_f16 v78, v112, v108, v78
v_mov_b32_dpp v112, v79  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16_hi v68, v92, s[40:43], 0 offen
ds_read_b128 v[58:61], v103 offset:28928
s_getpc_b64 s[38:39]
v_pk_fma_f16 v79, v112, v108, v79
v_mov_b32_dpp v112, v80  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v80, v112, v108, v80
v_mov_b32_dpp v112, v81  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v81, v112, v108, v81
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v73
s_add_u32 s56, s40, s75
v_nop_e64
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v67, v91, s[40:43], 0 offen
ds_read_b128 v[62:65], v103 offset:29056
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16_hi v69, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
s_cbranch_scc0 64820
buffer_load_short_d16 v70, v90, s[40:43], 0 offen
ds_read_b128 v[42:45], v102 offset:33536
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v47, v59, v7
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v74 offset:8256
s_mov_b32 m0, -1
v_pk_fma_f16 v82, v84, -1.0, v82 op_sel_hi:[1,0,1]
v_pk_mul_f16 v82, v82, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v12, v48, v60, v12
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v72, v92, s[40:43], 0 offen
ds_read_b128 v[50:53], v103 offset:33024
s_getpc_b64 s[38:39]
v_pk_fma_f16 v85, v83, -1.0, v85 op_sel_hi:[1,0,1]
v_pk_mul_f16 v85, v85, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v83, v84, v83
v_pk_mul_f16 v83, v83, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v84, v83, -1.0, v84 op_sel_hi:[1,0,1]
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v75 offset:8256
s_add_u32 s56, s40, s75
v_mov_b32_dpp v112, v82  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v71, v91, s[40:43], 0 offen
ds_read_b128 v[54:57], v103 offset:33152
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
ds_write_b8 v111, v111 offset:65504
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v73, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 64720
s_setprio 1
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v70, v90, s[40:43], 0 offen
ds_read_b128 v[46:49], v102 offset:37696
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v76 offset:8256
s_add_u32 m0, s87, 32
v_pk_fma_f16 v82, v112, v108, v82
v_mov_b32_dpp v112, v83  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(3)
buffer_load_short_d16_hi v72, v92, s[40:43], 0 offen
ds_read_b128 v[58:61], v103 offset:37184
s_getpc_b64 s[38:39]
v_pk_fma_f16 v83, v112, v108, v83
v_mov_b32_dpp v112, v84  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v84, v112, v108, v84
v_mov_b32_dpp v112, v85  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v85, v112, v108, v85
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v77 offset:8256
s_add_u32 s56, s40, s75
v_cmp_eq_u32_e64 vcc, src_lds_direct, v114
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v71, v91, s[40:43], 0 offen
ds_read_b128 v[62:65], v103 offset:37312
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_cbranch_vccz 2605
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16_hi v73, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
s_cbranch_scc0 64564
buffer_load_short_d16 v74, v90, s[40:43], 0 offen
ds_read_b128 v[42:45], v102 offset:41792
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v47, v59, v7
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v78 offset:16512
s_mov_b32 m0, -1
v_pk_fma_f16 v86, v88, -1.0, v86 op_sel_hi:[1,0,1]
v_pk_mul_f16 v86, v86, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v12, v48, v60, v12
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v76, v92, s[40:43], 0 offen
ds_read_b128 v[50:53], v103 offset:41280
s_getpc_b64 s[38:39]
v_pk_fma_f16 v89, v87, -1.0, v89 op_sel_hi:[1,0,1]
v_pk_mul_f16 v89, v89, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v87, v88, v87
v_pk_mul_f16 v87, v87, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v88, v87, -1.0, v88 op_sel_hi:[1,0,1]
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v79 offset:16512
s_add_u32 s56, s40, s75
v_mov_b32_dpp v112, v86  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v75, v91, s[40:43], 0 offen
ds_read_b128 v[54:57], v103 offset:41408
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v77, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 64464
s_setprio 0
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v74, v90, s[40:43], 0 offen
ds_read_b128 v[46:49], v102 offset:45952
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v80 offset:16512
s_mov_b32 m0, -1
v_pk_fma_f16 v86, v112, v108, v86
v_mov_b32_dpp v112, v87  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16_hi v76, v92, s[40:43], 0 offen
ds_read_b128 v[58:61], v103 offset:45440
s_getpc_b64 s[38:39]
v_pk_fma_f16 v87, v112, v108, v87
v_mov_b32_dpp v112, v88  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v88, v112, v108, v88
v_mov_b32_dpp v112, v89  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v89, v112, v108, v89
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v81 offset:16512
s_add_u32 s56, s40, s75
v_nop_e64
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v75, v91, s[40:43], 0 offen
ds_read_b128 v[62:65], v103 offset:45568
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16_hi v77, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
s_cbranch_scc0 64308
buffer_load_short_d16 v78, v90, s[40:43], 0 offen
ds_read_b128 v[42:45], v102 offset:512
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v47, v59, v7
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v82 offset:24768
s_mov_b32 m0, -1
v_pk_fma_f16 v66, v68, -1.0, v66 op_sel_hi:[1,0,1]
v_pk_mul_f16 v66, v66, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v12, v48, v60, v12
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v80, v92, s[40:43], 0 offen
ds_read_b128 v[50:53], v103
s_getpc_b64 s[38:39]
v_pk_fma_f16 v69, v67, -1.0, v69 op_sel_hi:[1,0,1]
v_pk_mul_f16 v69, v69, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v67, v68, v67
v_pk_mul_f16 v67, v67, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v68, v67, -1.0, v68 op_sel_hi:[1,0,1]
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v83 offset:24768
s_add_u32 s56, s40, s75
v_mov_b32_dpp v112, v66  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v79, v91, s[40:43], 0 offen
ds_read_b128 v[54:57], v103 offset:128
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
ds_write_b8 v111, v111 offset:65488
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v81, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 64208
s_setprio 1
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v78, v90, s[40:43], 0 offen
ds_read_b128 v[46:49], v102 offset:4672
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v84 offset:24768
s_add_u32 m0, s87, 16
v_pk_fma_f16 v66, v112, v108, v66
v_mov_b32_dpp v112, v67  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(3)
buffer_load_short_d16_hi v80, v92, s[40:43], 0 offen
ds_read_b128 v[58:61], v103 offset:4160
s_getpc_b64 s[38:39]
v_pk_fma_f16 v67, v112, v108, v67
v_mov_b32_dpp v112, v68  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v68, v112, v108, v68
v_mov_b32_dpp v112, v69  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v69, v112, v108, v69
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v85 offset:24768
s_add_u32 s56, s40, s75
v_cmp_eq_u32_e64 vcc, src_lds_direct, v114
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v79, v91, s[40:43], 0 offen
ds_read_b128 v[62:65], v103 offset:4288
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_cbranch_vccz 2093
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16_hi v81, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
s_cbranch_scc0 64052
buffer_load_short_d16 v82, v90, s[40:43], 0 offen
ds_read_b128 v[42:45], v102 offset:8768
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v47, v59, v7
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v86 offset:33024
s_mov_b32 m0, -1
v_pk_fma_f16 v70, v72, -1.0, v70 op_sel_hi:[1,0,1]
v_pk_mul_f16 v70, v70, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v12, v48, v60, v12
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v84, v92, s[40:43], 0 offen
ds_read_b128 v[50:53], v103 offset:8256
s_getpc_b64 s[38:39]
v_pk_fma_f16 v73, v71, -1.0, v73 op_sel_hi:[1,0,1]
v_pk_mul_f16 v73, v73, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v71, v72, v71
v_pk_mul_f16 v71, v71, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v72, v71, -1.0, v72 op_sel_hi:[1,0,1]
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v87 offset:33024
s_add_u32 s56, s40, s75
v_mov_b32_dpp v112, v70  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v83, v91, s[40:43], 0 offen
ds_read_b128 v[54:57], v103 offset:8384
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v85, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 63952
s_setprio 0
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v82, v90, s[40:43], 0 offen
ds_read_b128 v[46:49], v102 offset:12928
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v88 offset:33024
s_mov_b32 m0, -1
v_pk_fma_f16 v70, v112, v108, v70
v_mov_b32_dpp v112, v71  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16_hi v84, v92, s[40:43], 0 offen
ds_read_b128 v[58:61], v103 offset:12416
s_getpc_b64 s[38:39]
v_pk_fma_f16 v71, v112, v108, v71
v_mov_b32_dpp v112, v72  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v72, v112, v108, v72
v_mov_b32_dpp v112, v73  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v73, v112, v108, v73
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v89 offset:33024
s_add_u32 s56, s40, s75
v_nop_e64
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v83, v91, s[40:43], 0 offen
ds_read_b128 v[62:65], v103 offset:12544
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16_hi v85, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
s_cbranch_scc0 63796
buffer_load_short_d16 v86, v90, s[40:43], 0 offen
ds_read_b128 v[42:45], v102 offset:17024
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v47, v59, v7
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v66 offset:41280
s_mov_b32 m0, -1
v_pk_fma_f16 v74, v76, -1.0, v74 op_sel_hi:[1,0,1]
v_pk_mul_f16 v74, v74, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v12, v48, v60, v12
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v88, v92, s[40:43], 0 offen
ds_read_b128 v[50:53], v103 offset:16512
s_getpc_b64 s[38:39]
v_pk_fma_f16 v77, v75, -1.0, v77 op_sel_hi:[1,0,1]
v_pk_mul_f16 v77, v77, 0.5 op_sel_hi:[1,0]
v_pk_add_f16 v75, v76, v75
v_pk_mul_f16 v75, v75, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v76, v75, -1.0, v76 op_sel_hi:[1,0,1]
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v67 offset:41280
s_add_u32 s56, s40, s75
v_mov_b32_dpp v112, v74  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v87, v91, s[40:43], 0 offen
ds_read_b128 v[54:57], v103 offset:16640
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
ds_write_b8 v111, v111 offset:65472
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v89, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 63696
s_setprio 1
s_addk_i32 s100, 0x1800
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v86, v90, s[40:43], 0 offen
ds_read_b128 v[46:49], v102 offset:21184
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v68 offset:41280
s_add_u32 m0, s87, 0
v_pk_fma_f16 v74, v112, v108, v74
v_mov_b32_dpp v112, v75  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(3)
buffer_load_short_d16_hi v88, v92, s[40:43], 0 offen
ds_read_b128 v[58:61], v103 offset:20672
s_getpc_b64 s[38:39]
v_pk_fma_f16 v75, v112, v108, v75
v_mov_b32_dpp v112, v76  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v76, v112, v108, v76
v_mov_b32_dpp v112, v77  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v77, v112, v108, v77
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v69 offset:41280
s_add_u32 s56, s40, s75
v_cmp_eq_u32_e64 vcc, src_lds_direct, v114
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v87, v91, s[40:43], 0 offen
ds_read_b128 v[62:65], v103 offset:20800
s_addc_u32 s57, s41, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_cbranch_vccz 1581
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16_hi v89, v93, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_branch 64005
s_nop 0
s_getpc_b64 s[38:39]
s_sub_u32 s38, s38, 0x114
s_subb_u32 s39, s39, 0
s_branch 63550
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
s_cbranch_scc0 63532
buffer_load_short_d16 v68, v92, s[44:47], 0 offen
ds_read_b128 v[42:45], v102 offset:25280
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v47, v59, v7
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v70
s_mov_b32 m0, -1
v_mov_b32_dpp v78, v79  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v78, v78, v79
v_dot2_f32_f16 v12, v48, v60, v12
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v67, v91, s[44:47], 0 offen
ds_read_b128 v[50:53], v103 offset:24768
s_getpc_b64 s[38:39]
v_mov_b32_dpp v79, v79  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v78, v79, v108, v78
v_mov_b32_dpp v79, v81  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v79, v79, v81
v_mov_b32_dpp v81, v81  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v71
s_add_u32 s56, s44, s76
v_pk_fma_f16 v79, v81, v108, v79
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v69, v93, s[44:47], 0 offen
ds_read_b128 v[54:57], v103 offset:24896
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 63432
s_setprio 1
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v68, v92, s[44:47], 0 offen
ds_read_b128 v[46:49], v102 offset:29440
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v72
s_mov_b32 m0, -1
v_mov_b32_dpp v81, v80  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v81, v81, v80
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16_hi v67, v91, s[44:47], 0 offen
ds_read_b128 v[58:61], v103 offset:28928
s_getpc_b64 s[38:39]
v_mov_b32_dpp v80, v80  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v81, v80, v108, v81
v_pk_add_f16 v80, v78, v81
v_pk_add_f16 v79, v79, v80
v_pk_mul_f16 v79, v79, 0.5 op_sel_hi:[1,0]
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v73
s_add_u32 s56, s44, s76
v_pk_fma_f16 v80, -1.0, v79, v80 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v69, v93, s[44:47], 0 offen
ds_read_b128 v[62:65], v103 offset:29056
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_nop 0
s_nop 0
s_nop 0
s_nop 1
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
v_dot2_f32_f16 v7, v47, v59, v7
s_cbranch_scc0 63276
buffer_load_short_d16 v72, v92, s[44:47], 0 offen
ds_read_b128 v[42:45], v102 offset:33536
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
v_dot2_f32_f16 v12, v48, v60, v12
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v74 offset:8256
s_mov_b32 m0, -1
v_mov_b32_dpp v82, v83  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v82, v82, v83
v_mov_b32_dpp v83, v83  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v71, v91, s[44:47], 0 offen
ds_read_b128 v[50:53], v103 offset:33024
s_getpc_b64 s[38:39]
v_pk_fma_f16 v82, v83, v108, v82
v_mov_b32_dpp v83, v85  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v83, v83, v85
v_mov_b32_dpp v85, v85  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v83, v85, v108, v83
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v75 offset:8256
s_add_u32 s56, s44, s76
v_mov_b32_dpp v85, v84  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v73, v93, s[44:47], 0 offen
ds_read_b128 v[54:57], v103 offset:33152
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
ds_write_b8 v111, v111 offset:65504
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 63176
s_setprio 0
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v72, v92, s[44:47], 0 offen
ds_read_b128 v[46:49], v102 offset:37696
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v76 offset:8256
s_add_u32 m0, s87, 32
v_pk_add_f16 v85, v85, v84
v_mov_b32_dpp v84, v84  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(3)
buffer_load_short_d16_hi v71, v91, s[44:47], 0 offen
ds_read_b128 v[58:61], v103 offset:37184
s_getpc_b64 s[38:39]
v_pk_fma_f16 v85, v84, v108, v85
v_pk_add_f16 v84, v82, v85
v_pk_add_f16 v83, v83, v84
v_pk_mul_f16 v83, v83, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v84, -1.0, v83, v84 op_sel_hi:[0,1,1]
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v77 offset:8256
s_add_u32 s56, s44, s76
v_cmp_eq_u32_e64 vcc, src_lds_direct, v114
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v73, v93, s[44:47], 0 offen
ds_read_b128 v[62:65], v103 offset:37312
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_cbranch_vccz 1061
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
s_cbranch_scc0 63020
buffer_load_short_d16 v76, v92, s[44:47], 0 offen
ds_read_b128 v[42:45], v102 offset:41792
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v47, v59, v7
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v78 offset:16512
s_mov_b32 m0, -1
v_mov_b32_dpp v86, v87  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v86, v86, v87
v_dot2_f32_f16 v12, v48, v60, v12
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v75, v91, s[44:47], 0 offen
ds_read_b128 v[50:53], v103 offset:41280
s_getpc_b64 s[38:39]
v_mov_b32_dpp v87, v87  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v86, v87, v108, v86
v_mov_b32_dpp v87, v89  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v87, v87, v89
v_mov_b32_dpp v89, v89  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v79 offset:16512
s_add_u32 s56, s44, s76
v_pk_fma_f16 v87, v89, v108, v87
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v77, v93, s[44:47], 0 offen
ds_read_b128 v[54:57], v103 offset:41408
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 62920
s_setprio 1
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v76, v92, s[44:47], 0 offen
ds_read_b128 v[46:49], v102 offset:45952
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v80 offset:16512
s_mov_b32 m0, -1
v_mov_b32_dpp v89, v88  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v89, v89, v88
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16_hi v75, v91, s[44:47], 0 offen
ds_read_b128 v[58:61], v103 offset:45440
s_getpc_b64 s[38:39]
v_mov_b32_dpp v88, v88  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v89, v88, v108, v89
v_pk_add_f16 v88, v86, v89
v_pk_add_f16 v87, v87, v88
v_pk_mul_f16 v87, v87, 0.5 op_sel_hi:[1,0]
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v81 offset:16512
s_add_u32 s56, s44, s76
v_pk_fma_f16 v88, -1.0, v87, v88 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v77, v93, s[44:47], 0 offen
ds_read_b128 v[62:65], v103 offset:45568
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_nop 0
s_nop 0
s_nop 0
s_nop 1
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
v_dot2_f32_f16 v7, v47, v59, v7
s_cbranch_scc0 62764
buffer_load_short_d16 v80, v92, s[44:47], 0 offen
ds_read_b128 v[42:45], v102 offset:512
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
v_dot2_f32_f16 v12, v48, v60, v12
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v82 offset:24768
s_mov_b32 m0, -1
v_mov_b32_dpp v66, v67  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v66, v66, v67
v_mov_b32_dpp v67, v67  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v79, v91, s[44:47], 0 offen
ds_read_b128 v[50:53], v103
s_getpc_b64 s[38:39]
v_pk_fma_f16 v66, v67, v108, v66
v_mov_b32_dpp v67, v69  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v67, v67, v69
v_mov_b32_dpp v69, v69  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v67, v69, v108, v67
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v83 offset:24768
s_add_u32 s56, s44, s76
v_mov_b32_dpp v69, v68  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v81, v93, s[44:47], 0 offen
ds_read_b128 v[54:57], v103 offset:128
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
ds_write_b8 v111, v111 offset:65488
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 62664
s_setprio 0
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v80, v92, s[44:47], 0 offen
ds_read_b128 v[46:49], v102 offset:4672
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v84 offset:24768
s_add_u32 m0, s87, 16
v_pk_add_f16 v69, v69, v68
v_mov_b32_dpp v68, v68  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(3)
buffer_load_short_d16_hi v79, v91, s[44:47], 0 offen
ds_read_b128 v[58:61], v103 offset:4160
s_getpc_b64 s[38:39]
v_pk_fma_f16 v69, v68, v108, v69
v_pk_add_f16 v68, v66, v69
v_pk_add_f16 v67, v67, v68
v_pk_mul_f16 v67, v67, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v68, -1.0, v67, v68 op_sel_hi:[0,1,1]
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v85 offset:24768
s_add_u32 s56, s44, s76
v_cmp_eq_u32_e64 vcc, src_lds_direct, v114
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v81, v93, s[44:47], 0 offen
ds_read_b128 v[62:65], v103 offset:4288
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_cbranch_vccz 549
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
s_cbranch_scc0 62508
buffer_load_short_d16 v84, v92, s[44:47], 0 offen
ds_read_b128 v[42:45], v102 offset:8768
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v47, v59, v7
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v86 offset:33024
s_mov_b32 m0, -1
v_mov_b32_dpp v70, v71  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v70, v70, v71
v_dot2_f32_f16 v12, v48, v60, v12
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v83, v91, s[44:47], 0 offen
ds_read_b128 v[50:53], v103 offset:8256
s_getpc_b64 s[38:39]
v_mov_b32_dpp v71, v71  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v70, v71, v108, v70
v_mov_b32_dpp v71, v73  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v71, v71, v73
v_mov_b32_dpp v73, v73  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v87 offset:33024
s_add_u32 s56, s44, s76
v_pk_fma_f16 v71, v73, v108, v71
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v85, v93, s[44:47], 0 offen
ds_read_b128 v[54:57], v103 offset:8384
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 62408
s_setprio 1
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v84, v92, s[44:47], 0 offen
ds_read_b128 v[46:49], v102 offset:12928
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v88 offset:33024
s_mov_b32 m0, -1
v_mov_b32_dpp v73, v72  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v73, v73, v72
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16_hi v83, v91, s[44:47], 0 offen
ds_read_b128 v[58:61], v103 offset:12416
s_getpc_b64 s[38:39]
v_mov_b32_dpp v72, v72  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v73, v72, v108, v73
v_pk_add_f16 v72, v70, v73
v_pk_add_f16 v71, v71, v72
v_pk_mul_f16 v71, v71, 0.5 op_sel_hi:[1,0]
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v89 offset:33024
s_add_u32 s56, s44, s76
v_pk_fma_f16 v72, -1.0, v71, v72 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v85, v93, s[44:47], 0 offen
ds_read_b128 v[62:65], v103 offset:12544
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_nop 0
s_nop 0
s_nop 0
s_nop 1
v_dot2_f32_f16 v2, v46, v58, v2
v_dot2_f32_f16 v3, v47, v58, v3
v_dot2_f32_f16 v4, v48, v58, v4
v_dot2_f32_f16 v5, v49, v58, v5
v_dot2_f32_f16 v6, v46, v59, v6
v_dot2_f32_f16 v7, v47, v59, v7
s_cbranch_scc0 62252
buffer_load_short_d16 v88, v92, s[44:47], 0 offen
ds_read_b128 v[42:45], v102 offset:17024
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v8, v48, v59, v8
v_dot2_f32_f16 v9, v49, v59, v9
v_dot2_f32_f16 v10, v46, v60, v10
v_dot2_f32_f16 v11, v47, v60, v11
v_dot2_f32_f16 v12, v48, v60, v12
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v98, v66 offset:41280
s_mov_b32 m0, -1
v_mov_b32_dpp v74, v75  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v74, v74, v75
v_mov_b32_dpp v75, v75  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v13, v49, v60, v13
v_dot2_f32_f16 v14, v46, v61, v14
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v87, v91, s[44:47], 0 offen
ds_read_b128 v[50:53], v103 offset:16512
s_getpc_b64 s[38:39]
v_pk_fma_f16 v74, v75, v108, v74
v_mov_b32_dpp v75, v77  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v75, v75, v77
v_mov_b32_dpp v77, v77  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v75, v77, v108, v75
s_nop 0
s_add_u32 s96, s96, 0xf0f0f0c
ds_write_b32 v99, v67 offset:41280
s_add_u32 s56, s44, s76
v_mov_b32_dpp v77, v76  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v15, v47, v61, v15
v_dot2_f32_f16 v16, v48, v61, v16
v_dot2_f32_f16 v17, v49, v61, v17
v_dot2_f32_f16 v18, v46, v62, v18
s_nop 0
buffer_load_short_d16 v89, v93, s[44:47], 0 offen
ds_read_b128 v[54:57], v103 offset:16640
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v47, v62, v19
v_dot2_f32_f16 v20, v48, v62, v20
v_dot2_f32_f16 v21, v49, v62, v21
v_dot2_f32_f16 v22, v46, v63, v22
v_dot2_f32_f16 v23, v47, v63, v23
s_nop 0
s_nop 0
ds_write_b8 v111, v111 offset:65472
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v48, v63, v24
v_dot2_f32_f16 v25, v49, v63, v25
v_dot2_f32_f16 v26, v46, v64, v26
v_dot2_f32_f16 v27, v47, v64, v27
v_dot2_f32_f16 v28, v48, v64, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s96, s96, 0xf0f0f0f0
v_dot2_f32_f16 v29, v49, v64, v29
v_dot2_f32_f16 v30, v46, v65, v30
v_dot2_f32_f16 v31, v47, v65, v31
v_dot2_f32_f16 v32, v48, v65, v32
v_dot2_f32_f16 v33, v49, v65, v33
s_cbranch_scc0 62152
s_setprio 0
s_addk_i32 s100, 0x1800
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v42, v50, v2
v_dot2_f32_f16 v3, v43, v50, v3
v_dot2_f32_f16 v4, v44, v50, v4
v_dot2_f32_f16 v5, v45, v50, v5
v_dot2_f32_f16 v6, v42, v51, v6
s_nop 0
buffer_load_short_d16_hi v88, v92, s[44:47], 0 offen
ds_read_b128 v[46:49], v102 offset:21184
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v43, v51, v7
v_dot2_f32_f16 v8, v44, v51, v8
v_dot2_f32_f16 v9, v45, v51, v9
v_dot2_f32_f16 v10, v42, v52, v10
v_dot2_f32_f16 v11, v43, v52, v11
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v100, v68 offset:41280
s_add_u32 m0, s87, 0
v_pk_add_f16 v77, v77, v76
v_mov_b32_dpp v76, v76  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v44, v52, v12
v_dot2_f32_f16 v13, v45, v52, v13
v_dot2_f32_f16 v14, v42, v53, v14
s_waitcnt lgkmcnt(3)
buffer_load_short_d16_hi v87, v91, s[44:47], 0 offen
ds_read_b128 v[58:61], v103 offset:20672
s_getpc_b64 s[38:39]
v_pk_fma_f16 v77, v76, v108, v77
v_pk_add_f16 v76, v74, v77
v_pk_add_f16 v75, v75, v76
v_pk_mul_f16 v75, v75, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v76, -1.0, v75, v76 op_sel_hi:[0,1,1]
s_nop 0
s_add_u32 s85, s85, 0xf0f0f0c
ds_write_b32 v101, v69 offset:41280
s_add_u32 s56, s44, s76
v_cmp_eq_u32_e64 vcc, src_lds_direct, v114
v_dot2_f32_f16 v15, v43, v53, v15
v_dot2_f32_f16 v16, v44, v53, v16
v_dot2_f32_f16 v17, v45, v53, v17
v_dot2_f32_f16 v18, v42, v54, v18
s_nop 0
buffer_load_short_d16_hi v89, v93, s[44:47], 0 offen
ds_read_b128 v[62:65], v103 offset:20800
s_addc_u32 s57, s45, 0
v_dot2_f32_f16 v19, v43, v54, v19
v_dot2_f32_f16 v20, v44, v54, v20
v_dot2_f32_f16 v21, v45, v54, v21
v_dot2_f32_f16 v22, v42, v55, v22
v_dot2_f32_f16 v23, v43, v55, v23
s_cbranch_vccz 37
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v44, v55, v24
v_dot2_f32_f16 v25, v45, v55, v25
v_dot2_f32_f16 v26, v42, v56, v26
v_dot2_f32_f16 v27, v43, v56, v27
v_dot2_f32_f16 v28, v44, v56, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
s_add_u32 s85, s85, 0xf0f0f0f0
v_dot2_f32_f16 v29, v45, v56, v29
v_dot2_f32_f16 v30, v42, v57, v30
v_dot2_f32_f16 v31, v43, v57, v31
v_dot2_f32_f16 v32, v44, v57, v32
v_dot2_f32_f16 v33, v45, v57, v33
s_branch 64005
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 2
s_nop 1
s_subb_u32 s60, 0, 0
s_add_u32 s58, s38, 0xc0
s_addc_u32 s59, s39, 0
s_add_u32 s60, s60, 1
s_nop 0
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
s_nop 3
s_cbranch_vccz 65533
s_nop 1
s_nop 3
s_nop 9
s_nop 9
s_nop 4
s_nop 4
s_setpc_b64 s[58:59]
s_endpgm
