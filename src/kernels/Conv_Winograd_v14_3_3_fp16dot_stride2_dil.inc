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

.macro _s_add_u32_lit_gfx9 dst, src, lit
    .long  0x8000FF00 + ((\dst) << 16) + ((\src) << 0)
    .long \lit
.endm

s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_mov_b32_e32 v116, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s99, 0xdfc0
s_mov_b32 s98, 0xdfc0
s_mov_b32 s87, 0
v_lshlrev_b32_e32 v118, 2, v0
v_add_co_u32_e32 v118, vcc, 0xffc0, v118
v_cmp_ge_u32_e32 vcc, 3, v0
s_cbranch_vccz 5
v_mov_b32_e32 v117, 0
v_cndmask_b32_e32 v118, -1, v118, vcc
ds_write_b32 v118, v117
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
s_cbranch_scc0 4706
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
s_and_b32 s46, 1, s30
s_addc_u32 s46, s32, 1
s_ashr_i32 s46, s46, 1
s_add_u32 s44, s46, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s44
v_readfirstlane_b32 s44, v2
s_andn2_b32 s46, 1, s31
s_addc_u32 s46, s33, 1
s_ashr_i32 s46, s46, 1
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
s_and_b32 s46, s46, 1
s_and_b32 s46, s46, s64
s_add_u32 s64, s64, s46
v_readfirstlane_b32 s47, v0
s_and_b32 s50, s47, 64
s_cselect_b32 s50, 0x80000, 0
s_or_b32 s18, s18, s50
s_lshl_b32 s75, s72, 1
s_lshl_b32 s46, s65, 1
s_cmp_eq_u32 s46, 1
s_cbranch_scc0 5
s_bitcmp1_b32 s18, 2
s_cselect_b32 s50, 0, 0x1000000
s_or_b32 s18, s18, s50
s_branch 3
s_bitset1_b32 s18, 23
s_add_u32 s65, s65, 1
s_and_b32 s65, s65, -2
v_bfe_u32 v3, v0, 2, 6
v_lshrrev_b32_e32 v111, 1, v3
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
v_bfi_b32 v111, s50, v3, v111
v_bfe_u32 v3, s47, 8, 1
v_xor_b32_e64 v3, v3, 1
v_lshrrev_b32_e32 v111, v3, v111
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
v_cndmask_b32_e32 v7, 0, v111, vcc
v_cmp_eq_u32_e64 s[48:49], 3, v6
v_bfe_u32 v109, v7, 0, 5
v_mad_u32_u24 v109, v4, 32, v109
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
v_mul_hi_u32 v8, v8, v109
v_mul_lo_u32 v9, v8, s45
v_sub_co_u32_e32 v10, vcc, v109, v9
v_cmp_ge_u32_e64 s[50:51], v109, v9
v_cmp_ge_u32_e64 s[52:53], v10, s45
v_add_co_u32_e32 v10, vcc, 1, v8
s_and_b64 s[52:53], s[50:51], s[52:53]
v_add_co_u32_e32 v9, vcc, -1, v8
v_cndmask_b32_e64 v10, v8, v10, s[52:53]
v_cndmask_b32_e64 v10, v9, v10, s[50:51]
v_cmp_ne_i32_e64 vcc, 0, s45
v_cndmask_b32_e32 v110, -1, v10, vcc
v_mad_i32_i24 v108, v110, s78, v109
v_lshrrev_b32_e32 v109, 5, v7
v_mad_u32_u24 v109, v110, 1, v109
v_cndmask_b32_e64 v109, v109, 1, s[48:49]
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
v_mul_hi_u32 v8, v8, v109
v_mul_lo_u32 v9, v8, s44
v_sub_co_u32_e32 v10, vcc, v109, v9
v_cmp_ge_u32_e64 s[50:51], v109, v9
v_cmp_ge_u32_e64 s[52:53], v10, s44
v_add_co_u32_e32 v10, vcc, 1, v8
s_and_b64 s[52:53], s[50:51], s[52:53]
v_add_co_u32_e32 v9, vcc, -1, v8
v_cndmask_b32_e64 v10, v8, v10, s[52:53]
v_cndmask_b32_e64 v10, v9, v10, s[50:51]
v_cmp_ne_i32_e64 vcc, 0, s44
v_cndmask_b32_e32 v110, -1, v10, vcc
v_mad_i32_i24 v109, v110, s77, v109
v_readlane_b32 s79, v108, 2
v_readlane_b32 s80, v109, 2
v_readlane_b32 s81, v110, 2
v_readlane_b32 s82, v109, 3
v_readlane_b32 s83, v110, 3
v_add_co_u32_e64 v108, vcc, v108, s78
v_add_co_u32_e64 v109, vcc, v109, s77
v_mov_b32_dpp v110, v110  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v108, v108  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v109, v109  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x20000
s_mov_b32 s46, 0x80000000
s_mov_b32 s47, 0x20000
s_mov_b32 s50, 0x80000000
s_mov_b32 s51, 0x20000
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 7
v_xor_b32_dpp v112, v0, v0  quad_perm:[1,3,2,2] row_mask:0xf bank_mask:0xf
v_subrev_co_u32_e32 v112, vcc, 1, v112
v_cvt_f16_i16_e32 v112, v112
v_pk_add_f16 v112, v112, 0 op_sel_hi:[0,0]
s_branch 6
v_xor_b32_dpp v112, v0, v0  quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32_e32 v112, vcc, 1, v112
v_cvt_f16_i16_e32 v112, v112
v_pk_add_f16 v112, v112, 0 op_sel_hi:[0,0]
v_mov_b32_e32 v113, 1
v_xor_b32_dpp v113, v0, v0  quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v113, v0, v0  quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32_e32 v113, vcc, 1, v113
v_mov_b32_e32 v114, 1
v_xor_b32_dpp v114, v0, v0  quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v114, v0, v0  quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32_e32 v114, vcc, 1, v114
v_cvt_f32_i32_e32 v113, v113
v_cvt_f32_i32_e32 v114, v114
v_lshrrev_b32_e64 v117, 2, s94
v_and_b32_e32 v118, 3, v0
v_lshrrev_b32_e32 v119, 1, v0
v_bfi_b32 v119, 64, v119, v0
v_bfe_u32 v119, v119, 4, 3
v_mad_u32_u24 v106, v119, 4, v118
v_lshlrev_b32_e32 v106, 4, v106
v_mad_u32_u24 v107, v117, 4, v118
v_lshlrev_b32_e32 v107, 4, v107
v_bfe_u32 v117, v0, 2, 2
v_and_b32_e32 v118, 1, v117
v_mad_u32_u24 v120, v117, 16, v118
v_lshlrev_b32_e32 v120, 6, v120
v_xor_b32_e32 v107, v107, v120
v_mul_u32_u24_e32 v120, 0x400, v117
v_xor_b32_e32 v106, v106, v120
s_lshr_b32 s94, s94, 1
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
v_xor_b32_e32 v103, 0x314, v118
v_xor_b32_e32 v104, 0x31c, v118
v_xor_b32_e32 v105, 8, v118
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v102, v118, v105, vcc
v_cndmask_b32_e32 v105, v105, v118, vcc
v_mad_u32_u24 v102, 4, v102, v120
v_mad_u32_u24 v103, 4, v103, v120
v_mad_u32_u24 v104, 4, v104, v120
v_mad_u32_u24 v105, 4, v105, v120
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
v_mad_u32_u24 v102, 4, v118, v120
v_xor_b32_e32 v103, 0x307, v118
v_mad_u32_u24 v103, 4, v103, v120
v_xor_b32_e32 v104, 0x30f, v118
v_mad_u32_u24 v104, 4, v104, v120
v_xor_b32_e32 v105, 8, v118
v_mad_u32_u24 v105, 4, v105, v120
v_subrev_co_u32_e32 v108, vcc, s79, v108
v_mov_b32_e32 v118, s78
v_cmp_lt_i32_e32 vcc, v108, v118
v_subb_co_u32_e64 v117, vcc, 0, 0, vcc
v_mad_i32_i24 v108, v117, s78, v108
v_mad_i32_i24 v110, v117, s83, v110
v_mad_i32_i24 v109, v117, s82, v109
v_mov_b32_e32 v118, s77
v_cmp_lt_i32_e32 vcc, v109, v118
v_subb_co_u32_e64 v117, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v117
v_mad_i32_i24 v109, v117, v118, v109
v_subrev_co_u32_e32 v109, vcc, s80, v109
v_cmp_lt_i32_e32 vcc, v109, v118
v_subb_co_u32_e64 v117, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v117
v_mad_i32_i24 v109, v117, s77, v109
v_subrev_co_u32_e32 v110, vcc, s81, v110
s_mov_b32 s66, 0
s_mov_b32 s67, s28
s_mov_b32 s68, 1
s_mov_b32 s88, 0
s_mov_b32 s89, s16
s_mov_b32 s86, s89
s_sub_u32 s95, -1, s94
s_sub_u32 s95, s95, 16
s_bitset1_b32 s18, 21
s_mov_b32 s51, 0
s_mov_b32 s55, 0
v_add_co_u32_e32 v117, vcc, 2, v0
v_bfe_u32 v117, v117, 2, 1
v_cmp_ne_u32_e64 vcc, v117, 1
s_mov_b64 s[10:11], vcc
s_mov_b32 s96, 32
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 1590
s_branch 3229
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 3738
buffer_load_short_d16 v58, v94, s[40:43], 0 offen
ds_read_b128 v[34:37], v106 offset:25280
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v62
s_add_u32 s87, s87, 0x200
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v60, v96, s[40:43], 0 offen
ds_read_b128 v[42:45], v107 offset:24768
s_add_u32 s56, s40, s75
v_pack_b32_f16 v70, v82, v70
v_pack_b32_f16 v71, v83, v71
v_pack_b32_f16 v72, v84, v72
v_pack_b32_f16 v73, v85, v73
v_pk_fma_f16 v70, v72, -1.0, v70 op_sel_hi:[1,0,1]
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v63
s_addc_u32 s57, s41, 0
v_pk_mul_f16 v70, v70, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v73, v71, -1.0, v73 op_sel_hi:[1,0,1]
v_pk_mul_f16 v73, v73, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_pk_add_f16 v71, v72, v71
v_pk_mul_f16 v71, v71, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v72, v71, -1.0, v72 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v70  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_short_d16 v59, v95, s[40:43], 0 offen
ds_read_b128 v[46:49], v107 offset:24896
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v61, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 3636
s_setprio 0
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v82, v94, s[40:43], 0 offen
ds_read_b128 v[38:41], v106 offset:29440
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v64
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v84, v96, s[40:43], 0 offen
ds_read_b128 v[50:53], v107 offset:28928
s_add_u32 s56, s40, s75
v_nop_e64
v_pk_fma_f16 v70, v115, v112, v70
v_mov_b32_dpp v115, v71  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v71, v115, v112, v71
v_mov_b32_dpp v115, v72  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v65
s_addc_u32 s57, s41, 0
v_pk_fma_f16 v72, v115, v112, v72
v_mov_b32_dpp v115, v73  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v73, v115, v112, v73
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
s_nop 0
buffer_load_short_d16 v83, v95, s[40:43], 0 offen
ds_read_b128 v[54:57], v107 offset:29056
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16 v85, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 3474
buffer_load_short_d16 v62, v94, s[40:43], 0 offen
ds_read_b128 v[34:37], v106 offset:33536
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v66 offset:8256
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v64, v96, s[40:43], 0 offen
ds_read_b128 v[42:45], v107 offset:33024
s_add_u32 s56, s40, s75
v_pack_b32_f16 v74, v86, v74
v_pack_b32_f16 v75, v87, v75
v_pack_b32_f16 v76, v88, v76
v_pack_b32_f16 v77, v89, v77
v_pk_fma_f16 v74, v76, -1.0, v74 op_sel_hi:[1,0,1]
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v67 offset:8256
s_addc_u32 s57, s41, 0
v_pk_mul_f16 v74, v74, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v77, v75, -1.0, v77 op_sel_hi:[1,0,1]
v_pk_mul_f16 v77, v77, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
s_nop 0
buffer_load_short_d16 v63, v95, s[40:43], 0 offen
ds_read_b128 v[46:49], v107 offset:33152
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 1
ds_append v117 offset:65472
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v65, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 3380
s_setprio 1
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v86, v94, s[40:43], 0 offen
ds_read_b128 v[38:41], v106 offset:37696
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v68 offset:8256
s_mov_b32 m0, 0x2ffc0
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(3)
buffer_load_short_d16 v88, v96, s[40:43], 0 offen
ds_read_b128 v[50:53], v107 offset:37184
s_add_u32 s56, s40, s75
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
v_pk_add_f16 v75, v76, v75
v_pk_mul_f16 v75, v75, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v76, v75, -1.0, v76 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v74  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v69 offset:8256
s_addc_u32 s57, s41, 0
v_pk_fma_f16 v74, v115, v112, v74
v_mov_b32_dpp v115, v75  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v75, v115, v112, v75
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_mov_b32_dpp v115, v76  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v76, v115, v112, v76
v_mov_b32_dpp v115, v77  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v77, v115, v112, v77
s_nop 0
buffer_load_short_d16 v87, v95, s[40:43], 0 offen
ds_read_b128 v[54:57], v107 offset:37312
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_cbranch_vccz 3717
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16 v89, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 3210
buffer_load_short_d16 v66, v94, s[40:43], 0 offen
ds_read_b128 v[34:37], v106 offset:41792
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v70 offset:16512
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v68, v96, s[40:43], 0 offen
ds_read_b128 v[42:45], v107 offset:41280
s_add_u32 s56, s40, s75
v_pack_b32_f16 v78, v90, v78
v_pack_b32_f16 v79, v91, v79
v_pack_b32_f16 v80, v92, v80
v_pack_b32_f16 v81, v93, v81
v_pk_fma_f16 v78, v80, -1.0, v78 op_sel_hi:[1,0,1]
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v71 offset:16512
s_addc_u32 s57, s41, 0
v_pk_mul_f16 v78, v78, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v81, v79, -1.0, v81 op_sel_hi:[1,0,1]
v_pk_mul_f16 v81, v81, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_pk_add_f16 v79, v80, v79
v_pk_mul_f16 v79, v79, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v80, v79, -1.0, v80 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v78  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_short_d16 v67, v95, s[40:43], 0 offen
ds_read_b128 v[46:49], v107 offset:41408
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v69, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 3108
s_setprio 0
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v90, v94, s[40:43], 0 offen
ds_read_b128 v[38:41], v106 offset:45952
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v72 offset:16512
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v92, v96, s[40:43], 0 offen
ds_read_b128 v[50:53], v107 offset:45440
s_add_u32 s56, s40, s75
v_nop_e64
v_pk_fma_f16 v78, v115, v112, v78
v_mov_b32_dpp v115, v79  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v79, v115, v112, v79
v_mov_b32_dpp v115, v80  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v73 offset:16512
s_addc_u32 s57, s41, 0
v_pk_fma_f16 v80, v115, v112, v80
v_mov_b32_dpp v115, v81  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v81, v115, v112, v81
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
s_nop 0
buffer_load_short_d16 v91, v95, s[40:43], 0 offen
ds_read_b128 v[54:57], v107 offset:45568
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16 v93, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 2946
buffer_load_short_d16 v70, v94, s[40:43], 0 offen
ds_read_b128 v[34:37], v106 offset:512
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v74 offset:24768
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v72, v96, s[40:43], 0 offen
ds_read_b128 v[42:45], v107
s_add_u32 s56, s40, s75
v_pack_b32_f16 v58, v82, v58
v_pack_b32_f16 v59, v83, v59
v_pack_b32_f16 v60, v84, v60
v_pack_b32_f16 v61, v85, v61
v_pk_fma_f16 v58, v60, -1.0, v58 op_sel_hi:[1,0,1]
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v75 offset:24768
s_addc_u32 s57, s41, 0
v_pk_mul_f16 v58, v58, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v61, v59, -1.0, v61 op_sel_hi:[1,0,1]
v_pk_mul_f16 v61, v61, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
s_nop 0
buffer_load_short_d16 v71, v95, s[40:43], 0 offen
ds_read_b128 v[46:49], v107 offset:128
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 1
ds_append v117 offset:65476
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v73, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 2852
s_setprio 1
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v82, v94, s[40:43], 0 offen
ds_read_b128 v[38:41], v106 offset:4672
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v76 offset:24768
s_mov_b32 m0, 0x2ffc4
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(3)
buffer_load_short_d16 v84, v96, s[40:43], 0 offen
ds_read_b128 v[50:53], v107 offset:4160
s_add_u32 s56, s40, s75
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
v_pk_add_f16 v59, v60, v59
v_pk_mul_f16 v59, v59, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v60, v59, -1.0, v60 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v58  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v77 offset:24768
s_addc_u32 s57, s41, 0
v_pk_fma_f16 v58, v115, v112, v58
v_mov_b32_dpp v115, v59  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v59, v115, v112, v59
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_mov_b32_dpp v115, v60  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v60, v115, v112, v60
v_mov_b32_dpp v115, v61  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v61, v115, v112, v61
s_nop 0
buffer_load_short_d16 v83, v95, s[40:43], 0 offen
ds_read_b128 v[54:57], v107 offset:4288
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_cbranch_vccz 3189
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16 v85, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 2682
buffer_load_short_d16 v74, v94, s[40:43], 0 offen
ds_read_b128 v[34:37], v106 offset:8768
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v78 offset:33024
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v76, v96, s[40:43], 0 offen
ds_read_b128 v[42:45], v107 offset:8256
s_add_u32 s56, s40, s75
v_pack_b32_f16 v62, v86, v62
v_pack_b32_f16 v63, v87, v63
v_pack_b32_f16 v64, v88, v64
v_pack_b32_f16 v65, v89, v65
v_pk_fma_f16 v62, v64, -1.0, v62 op_sel_hi:[1,0,1]
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v79 offset:33024
s_addc_u32 s57, s41, 0
v_pk_mul_f16 v62, v62, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v65, v63, -1.0, v65 op_sel_hi:[1,0,1]
v_pk_mul_f16 v65, v65, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_pk_add_f16 v63, v64, v63
v_pk_mul_f16 v63, v63, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v64, v63, -1.0, v64 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v62  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_short_d16 v75, v95, s[40:43], 0 offen
ds_read_b128 v[46:49], v107 offset:8384
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v77, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 2580
s_setprio 0
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v86, v94, s[40:43], 0 offen
ds_read_b128 v[38:41], v106 offset:12928
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v80 offset:33024
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v88, v96, s[40:43], 0 offen
ds_read_b128 v[50:53], v107 offset:12416
s_add_u32 s56, s40, s75
v_nop_e64
v_pk_fma_f16 v62, v115, v112, v62
v_mov_b32_dpp v115, v63  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v63, v115, v112, v63
v_mov_b32_dpp v115, v64  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v81 offset:33024
s_addc_u32 s57, s41, 0
v_pk_fma_f16 v64, v115, v112, v64
v_mov_b32_dpp v115, v65  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v65, v115, v112, v65
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
s_nop 0
buffer_load_short_d16 v87, v95, s[40:43], 0 offen
ds_read_b128 v[54:57], v107 offset:12544
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16 v89, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 2418
buffer_load_short_d16 v78, v94, s[40:43], 0 offen
ds_read_b128 v[34:37], v106 offset:17024
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v58 offset:41280
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v80, v96, s[40:43], 0 offen
ds_read_b128 v[42:45], v107 offset:16512
s_add_u32 s56, s40, s75
v_pack_b32_f16 v66, v90, v66
v_pack_b32_f16 v67, v91, v67
v_pack_b32_f16 v68, v92, v68
v_pack_b32_f16 v69, v93, v69
v_pk_fma_f16 v66, v68, -1.0, v66 op_sel_hi:[1,0,1]
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v59 offset:41280
s_addc_u32 s57, s41, 0
v_pk_mul_f16 v66, v66, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v69, v67, -1.0, v69 op_sel_hi:[1,0,1]
v_pk_mul_f16 v69, v69, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
s_nop 0
buffer_load_short_d16 v79, v95, s[40:43], 0 offen
ds_read_b128 v[46:49], v107 offset:16640
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 1
ds_append v117 offset:65480
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v81, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 2324
s_setprio 1
s_addk_i32 s100, 0x18c0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v90, v94, s[40:43], 0 offen
ds_read_b128 v[38:41], v106 offset:21184
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v60 offset:41280
s_mov_b32 m0, 0x2ffc8
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(3)
buffer_load_short_d16 v92, v96, s[40:43], 0 offen
ds_read_b128 v[50:53], v107 offset:20672
s_add_u32 s56, s40, s75
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
v_pk_add_f16 v67, v68, v67
v_pk_mul_f16 v67, v67, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v68, v67, -1.0, v68 op_sel_hi:[1,0,1]
v_mov_b32_dpp v115, v66  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v61 offset:41280
s_addc_u32 s57, s41, 0
v_pk_fma_f16 v66, v115, v112, v66
v_mov_b32_dpp v115, v67  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v67, v115, v112, v67
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_mov_b32_dpp v115, v68  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v68, v115, v112, v68
v_mov_b32_dpp v115, v69  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v69, v115, v112, v69
s_nop 0
buffer_load_short_d16 v91, v95, s[40:43], 0 offen
ds_read_b128 v[54:57], v107 offset:20800
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_cbranch_vccz 2661
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(15) lgkmcnt(2)
buffer_load_short_d16 v93, v97, s[40:43], 0 offen
s_mov_b64 s[40:41], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_branch 63957
s_getpc_b64 s[38:39]
s_mov_b32 s100, 0x18c0
s_sub_u32 s38, s38, 0xb0
s_subb_u32 s39, s39, 0
s_branch 2163
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 2146
buffer_load_short_d16 v60, v96, s[44:47], 0 offen
ds_read_b128 v[34:37], v106 offset:25280
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v62
s_add_u32 s87, s87, 0x200
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v59, v95, s[44:47], 0 offen
ds_read_b128 v[42:45], v107 offset:24768
s_add_u32 s56, s44, s76
v_pack_b32_f16 v70, v82, v70
v_pack_b32_f16 v71, v83, v71
v_pack_b32_f16 v72, v84, v72
v_pack_b32_f16 v73, v85, v73
v_cndmask_b32_dpp v70, v70, v70, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v63
s_addc_u32 s57, s45, 0
v_cndmask_b32_dpp v71, v71, v71, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v72, v72, v72, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v73, v73, v73, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
s_nop 0
buffer_load_short_d16 v61, v97, s[44:47], 0 offen
ds_read_b128 v[46:49], v107 offset:24896
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 2052
s_setprio 1
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v84, v96, s[44:47], 0 offen
ds_read_b128 v[38:41], v106 offset:29440
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v64
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v83, v95, s[44:47], 0 offen
ds_read_b128 v[50:53], v107 offset:28928
s_add_u32 s56, s44, s76
v_mov_b32_dpp v70, v71  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v70, v70, v71
v_mov_b32_dpp v71, v71  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v70, v71, v112, v70
v_mov_b32_dpp v71, v73  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v65
s_addc_u32 s57, s45, 0
v_pk_add_f16 v71, v71, v73
v_mov_b32_dpp v73, v73  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v71, v73, v112, v71
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_mov_b32_dpp v73, v72  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v73, v73, v72
v_mov_b32_dpp v72, v72  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v73, v72, v112, v73
v_pk_add_f16 v72, v70, v73
v_pk_add_f16 v71, v71, v72
v_pk_mul_f16 v71, v71, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v72, -1.0, v71, v72 op_sel_hi:[0,1,1]
s_nop 0
buffer_load_short_d16 v85, v97, s[44:47], 0 offen
ds_read_b128 v[54:57], v107 offset:29056
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
s_nop 0
s_nop 0
s_nop 1
v_pack_b32_f16 v74, v86, v74
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 1874
buffer_load_short_d16 v64, v96, s[44:47], 0 offen
ds_read_b128 v[34:37], v106 offset:33536
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v66 offset:8256
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v63, v95, s[44:47], 0 offen
ds_read_b128 v[42:45], v107 offset:33024
s_add_u32 s56, s44, s76
v_pack_b32_f16 v75, v87, v75
v_pack_b32_f16 v76, v88, v76
v_pack_b32_f16 v77, v89, v77
v_cndmask_b32_dpp v74, v74, v74, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v75, v75, v75, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v67 offset:8256
s_addc_u32 s57, s45, 0
v_cndmask_b32_dpp v76, v76, v76, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v77, v77, v77, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v74, v75  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_pk_add_f16 v74, v74, v75
v_mov_b32_dpp v75, v75  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v74, v75, v112, v74
v_mov_b32_dpp v75, v77  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v75, v75, v77
v_mov_b32_dpp v77, v77  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v75, v77, v112, v75
v_mov_b32_dpp v77, v76  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_short_d16 v65, v97, s[44:47], 0 offen
ds_read_b128 v[46:49], v107 offset:33152
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 1
ds_append v117 offset:65472
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 1764
s_setprio 0
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v88, v96, s[44:47], 0 offen
ds_read_b128 v[38:41], v106 offset:37696
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v68 offset:8256
s_mov_b32 m0, 0x2ffc0
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(3)
buffer_load_short_d16 v87, v95, s[44:47], 0 offen
ds_read_b128 v[50:53], v107 offset:37184
s_add_u32 s56, s44, s76
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
v_pk_add_f16 v77, v77, v76
v_mov_b32_dpp v76, v76  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v77, v76, v112, v77
v_pk_add_f16 v76, v74, v77
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v69 offset:8256
s_addc_u32 s57, s45, 0
v_pk_add_f16 v75, v75, v76
v_pk_mul_f16 v75, v75, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v76, -1.0, v75, v76 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
s_nop 0
buffer_load_short_d16 v89, v97, s[44:47], 0 offen
ds_read_b128 v[54:57], v107 offset:37312
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_cbranch_vccz 2109
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 1602
buffer_load_short_d16 v68, v96, s[44:47], 0 offen
ds_read_b128 v[34:37], v106 offset:41792
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v70 offset:16512
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v67, v95, s[44:47], 0 offen
ds_read_b128 v[42:45], v107 offset:41280
s_add_u32 s56, s44, s76
v_pack_b32_f16 v78, v90, v78
v_pack_b32_f16 v79, v91, v79
v_pack_b32_f16 v80, v92, v80
v_pack_b32_f16 v81, v93, v81
v_cndmask_b32_dpp v78, v78, v78, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v71 offset:16512
s_addc_u32 s57, s45, 0
v_cndmask_b32_dpp v79, v79, v79, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v80, v80, v80, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v81, v81, v81, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
s_nop 0
buffer_load_short_d16 v69, v97, s[44:47], 0 offen
ds_read_b128 v[46:49], v107 offset:41408
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 1508
s_setprio 1
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v92, v96, s[44:47], 0 offen
ds_read_b128 v[38:41], v106 offset:45952
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v72 offset:16512
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v91, v95, s[44:47], 0 offen
ds_read_b128 v[50:53], v107 offset:45440
s_add_u32 s56, s44, s76
v_mov_b32_dpp v78, v79  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v78, v78, v79
v_mov_b32_dpp v79, v79  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v78, v79, v112, v78
v_mov_b32_dpp v79, v81  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v73 offset:16512
s_addc_u32 s57, s45, 0
v_pk_add_f16 v79, v79, v81
v_mov_b32_dpp v81, v81  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v79, v81, v112, v79
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_mov_b32_dpp v81, v80  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v81, v81, v80
v_mov_b32_dpp v80, v80  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v81, v80, v112, v81
v_pk_add_f16 v80, v78, v81
v_pk_add_f16 v79, v79, v80
v_pk_mul_f16 v79, v79, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v80, -1.0, v79, v80 op_sel_hi:[0,1,1]
s_nop 0
buffer_load_short_d16 v93, v97, s[44:47], 0 offen
ds_read_b128 v[54:57], v107 offset:45568
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
s_nop 0
s_nop 0
s_nop 1
v_pack_b32_f16 v58, v82, v58
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 1330
buffer_load_short_d16 v72, v96, s[44:47], 0 offen
ds_read_b128 v[34:37], v106 offset:512
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v74 offset:24768
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v71, v95, s[44:47], 0 offen
ds_read_b128 v[42:45], v107
s_add_u32 s56, s44, s76
v_pack_b32_f16 v59, v83, v59
v_pack_b32_f16 v60, v84, v60
v_pack_b32_f16 v61, v85, v61
v_cndmask_b32_dpp v58, v58, v58, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v59, v59, v59, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v75 offset:24768
s_addc_u32 s57, s45, 0
v_cndmask_b32_dpp v60, v60, v60, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v61, v61, v61, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v58, v59  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_pk_add_f16 v58, v58, v59
v_mov_b32_dpp v59, v59  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v58, v59, v112, v58
v_mov_b32_dpp v59, v61  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v59, v59, v61
v_mov_b32_dpp v61, v61  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v59, v61, v112, v59
v_mov_b32_dpp v61, v60  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_short_d16 v73, v97, s[44:47], 0 offen
ds_read_b128 v[46:49], v107 offset:128
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 1
ds_append v117 offset:65476
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 1220
s_setprio 0
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v84, v96, s[44:47], 0 offen
ds_read_b128 v[38:41], v106 offset:4672
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v76 offset:24768
s_mov_b32 m0, 0x2ffc4
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(3)
buffer_load_short_d16 v83, v95, s[44:47], 0 offen
ds_read_b128 v[50:53], v107 offset:4160
s_add_u32 s56, s44, s76
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
v_pk_add_f16 v61, v61, v60
v_mov_b32_dpp v60, v60  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v61, v60, v112, v61
v_pk_add_f16 v60, v58, v61
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v77 offset:24768
s_addc_u32 s57, s45, 0
v_pk_add_f16 v59, v59, v60
v_pk_mul_f16 v59, v59, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v60, -1.0, v59, v60 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
s_nop 0
buffer_load_short_d16 v85, v97, s[44:47], 0 offen
ds_read_b128 v[54:57], v107 offset:4288
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_cbranch_vccz 1565
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 1058
buffer_load_short_d16 v76, v96, s[44:47], 0 offen
ds_read_b128 v[34:37], v106 offset:8768
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v78 offset:33024
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v75, v95, s[44:47], 0 offen
ds_read_b128 v[42:45], v107 offset:8256
s_add_u32 s56, s44, s76
v_pack_b32_f16 v62, v86, v62
v_pack_b32_f16 v63, v87, v63
v_pack_b32_f16 v64, v88, v64
v_pack_b32_f16 v65, v89, v65
v_cndmask_b32_dpp v62, v62, v62, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v79 offset:33024
s_addc_u32 s57, s45, 0
v_cndmask_b32_dpp v63, v63, v63, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v64, v64, v64, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v65, v65, v65, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
s_nop 0
buffer_load_short_d16 v77, v97, s[44:47], 0 offen
ds_read_b128 v[46:49], v107 offset:8384
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 964
s_setprio 1
s_mov_b32 s100, 0
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v88, v96, s[44:47], 0 offen
ds_read_b128 v[38:41], v106 offset:12928
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v80 offset:33024
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v87, v95, s[44:47], 0 offen
ds_read_b128 v[50:53], v107 offset:12416
s_add_u32 s56, s44, s76
v_mov_b32_dpp v62, v63  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v62, v62, v63
v_mov_b32_dpp v63, v63  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v62, v63, v112, v62
v_mov_b32_dpp v63, v65  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v81 offset:33024
s_addc_u32 s57, s45, 0
v_pk_add_f16 v63, v63, v65
v_mov_b32_dpp v65, v65  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v63, v65, v112, v63
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
v_mov_b32_dpp v65, v64  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v65, v65, v64
v_mov_b32_dpp v64, v64  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v65, v64, v112, v65
v_pk_add_f16 v64, v62, v65
v_pk_add_f16 v63, v63, v64
v_pk_mul_f16 v63, v63, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v64, -1.0, v63, v64 op_sel_hi:[0,1,1]
s_nop 0
buffer_load_short_d16 v89, v97, s[44:47], 0 offen
ds_read_b128 v[54:57], v107 offset:12544
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_nop 0
s_nop 0
s_nop 0
s_nop 1
v_pack_b32_f16 v66, v90, v66
v_dot2_f32_f16 v2, v38, v50, v2
v_dot2_f32_f16 v3, v39, v50, v3
v_dot2_f32_f16 v4, v40, v50, v4
v_dot2_f32_f16 v5, v41, v50, v5
v_dot2_f32_f16 v6, v38, v51, v6
s_cbranch_scc0 786
buffer_load_short_d16 v80, v96, s[44:47], 0 offen
ds_read_b128 v[34:37], v106 offset:17024
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v39, v51, v7
v_dot2_f32_f16 v8, v40, v51, v8
v_dot2_f32_f16 v9, v41, v51, v9
v_dot2_f32_f16 v10, v38, v52, v10
v_dot2_f32_f16 v11, v39, v52, v11
s_nop 0
s_nop 0
ds_write_b32 v102, v58 offset:41280
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v12, v40, v52, v12
v_dot2_f32_f16 v13, v41, v52, v13
v_dot2_f32_f16 v14, v38, v53, v14
v_dot2_f32_f16 v15, v39, v53, v15
v_dot2_f32_f16 v16, v40, v53, v16
s_waitcnt lgkmcnt(2)
buffer_load_short_d16 v79, v95, s[44:47], 0 offen
ds_read_b128 v[42:45], v107 offset:16512
s_add_u32 s56, s44, s76
v_pack_b32_f16 v67, v91, v67
v_pack_b32_f16 v68, v92, v68
v_pack_b32_f16 v69, v93, v69
v_cndmask_b32_dpp v66, v66, v66, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v67, v67, v67, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v103, v59 offset:41280
s_addc_u32 s57, s45, 0
v_cndmask_b32_dpp v68, v68, v68, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v69, v69, v69, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v66, v67  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v17, v41, v53, v17
v_dot2_f32_f16 v18, v38, v54, v18
v_pk_add_f16 v66, v66, v67
v_mov_b32_dpp v67, v67  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v66, v67, v112, v66
v_mov_b32_dpp v67, v69  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_pk_add_f16 v67, v67, v69
v_mov_b32_dpp v69, v69  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v67, v69, v112, v67
v_mov_b32_dpp v69, v68  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_short_d16 v81, v97, s[44:47], 0 offen
ds_read_b128 v[46:49], v107 offset:16640
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v39, v54, v19
v_dot2_f32_f16 v20, v40, v54, v20
v_dot2_f32_f16 v21, v41, v54, v21
v_dot2_f32_f16 v22, v38, v55, v22
v_dot2_f32_f16 v23, v39, v55, v23
s_nop 0
s_nop 1
ds_append v117 offset:65480
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v40, v55, v24
v_dot2_f32_f16 v25, v41, v55, v25
v_dot2_f32_f16 v26, v38, v56, v26
v_dot2_f32_f16 v27, v39, v56, v27
v_dot2_f32_f16 v28, v40, v56, v28
s_waitcnt lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 96, 96, -4
v_dot2_f32_f16 v29, v41, v56, v29
v_dot2_f32_f16 v30, v38, v57, v30
v_dot2_f32_f16 v31, v39, v57, v31
v_dot2_f32_f16 v32, v40, v57, v32
v_dot2_f32_f16 v33, v41, v57, v33
s_cbranch_scc0 676
s_setprio 0
s_addk_i32 s100, 0x1980
s_mov_b32 s1, 0xf0f0f0f0
s_nop 0
v_dot2_f32_f16 v2, v34, v42, v2
v_dot2_f32_f16 v3, v35, v42, v3
v_dot2_f32_f16 v4, v36, v42, v4
v_dot2_f32_f16 v5, v37, v42, v5
v_dot2_f32_f16 v6, v34, v43, v6
s_nop 0
buffer_load_short_d16 v92, v96, s[44:47], 0 offen
ds_read_b128 v[38:41], v106 offset:21184
s_mov_b64 vcc, s[10:11]
v_dot2_f32_f16 v7, v35, v43, v7
v_dot2_f32_f16 v8, v36, v43, v8
v_dot2_f32_f16 v9, v37, v43, v9
v_dot2_f32_f16 v10, v34, v44, v10
v_dot2_f32_f16 v11, v35, v44, v11
s_nop 0
s_nop 0
ds_write_b32 v104, v60 offset:41280
s_mov_b32 m0, 0x2ffc8
v_dot2_f32_f16 v12, v36, v44, v12
v_dot2_f32_f16 v13, v37, v44, v13
v_dot2_f32_f16 v14, v34, v45, v14
v_dot2_f32_f16 v15, v35, v45, v15
v_dot2_f32_f16 v16, v36, v45, v16
s_waitcnt lgkmcnt(3)
buffer_load_short_d16 v91, v95, s[44:47], 0 offen
ds_read_b128 v[50:53], v107 offset:20672
s_add_u32 s56, s44, s76
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
v_pk_add_f16 v69, v69, v68
v_mov_b32_dpp v68, v68  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_pk_fma_f16 v69, v68, v112, v69
v_pk_add_f16 v68, v66, v69
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
ds_write_b32 v105, v61 offset:41280
s_addc_u32 s57, s45, 0
v_pk_add_f16 v67, v67, v68
v_pk_mul_f16 v67, v67, 0.5 op_sel_hi:[1,0]
v_pk_fma_f16 v68, -1.0, v67, v68 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v17, v37, v45, v17
v_dot2_f32_f16 v18, v34, v46, v18
s_nop 0
buffer_load_short_d16 v93, v97, s[44:47], 0 offen
ds_read_b128 v[54:57], v107 offset:20800
s_getpc_b64 s[38:39]
v_dot2_f32_f16 v19, v35, v46, v19
v_dot2_f32_f16 v20, v36, v46, v20
v_dot2_f32_f16 v21, v37, v46, v21
v_dot2_f32_f16 v22, v34, v47, v22
v_dot2_f32_f16 v23, v35, v47, v23
s_cbranch_vccz 1021
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s1, 0xf0f0f0f0
v_dot2_f32_f16 v24, v36, v47, v24
v_dot2_f32_f16 v25, v37, v47, v25
v_dot2_f32_f16 v26, v34, v48, v26
v_dot2_f32_f16 v27, v35, v48, v27
v_dot2_f32_f16 v28, v36, v48, v28
s_waitcnt vmcnt(12) lgkmcnt(2)
s_nop 0
s_nop 0
s_mov_b64 s[44:45], s[56:57]
_s_add_u32_lit_gfx9 85, 85, -4
v_dot2_f32_f16 v29, v37, v48, v29
v_dot2_f32_f16 v30, v34, v49, v30
v_dot2_f32_f16 v31, v35, v49, v31
v_dot2_f32_f16 v32, v36, v49, v32
v_dot2_f32_f16 v33, v37, v49, v33
s_branch 63909
s_getpc_b64 s[38:39]
s_mov_b32 s100, 0x1980
s_sub_u32 s38, s38, 0xb0
s_subb_u32 s39, s39, 0
s_branch 523
s_bitcmp1_b32 s18, 17
s_cbranch_scc1 249
s_add_u32 s92, s92, s17
s_cmp_eq_u32 s92, 0
s_cbranch_scc1 246
s_mov_b32 s93, 0
s_bitcmp1_b32 s18, 16
s_cbranch_scc1 235
s_add_u32 s91, s16, 15
s_lshr_b32 s91, s91, 4
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
v_add_co_u32_e32 v108, vcc, s79, v108
v_addc_co_u32_e64 v123, vcc, 0, 0, vcc
v_mad_i32_i24 v108, v123, s78, v108
v_mad_i32_i24 v110, v123, s83, v110
v_mad_i32_i24 v109, v123, s82, v109
v_cmp_ge_i32_e64 vcc, v109, 0
v_addc_co_u32_e64 v123, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v123
v_mad_i32_i24 v109, v123, s77, v109
v_add_co_u32_e32 v109, vcc, s80, v109
v_addc_co_u32_e64 v123, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v123
v_mad_i32_i24 v109, v123, s77, v109
v_add_co_u32_e32 v110, vcc, s81, v110
v_readlane_b32 s79, v121, 1
v_readlane_b32 s80, v122, 1
v_readlane_b32 s81, v120, 1
s_add_u32 s89, s88, s90
s_cmp_le_u32 s89, s91
s_cselect_b32 s56, 0x20000, 0
s_cselect_b32 s89, s89, s91
s_or_b32 s18, s18, s56
s_lshl_b32 s88, s88, 4
s_lshl_b32 s89, s89, 4
s_min_u32 s89, s89, s16
s_cmp_eq_u32 s8, s17
s_cselect_b32 s56, 0x20000, 0
s_or_b32 s18, s18, s56
s_or_b32 s18, s18, s56
s_bitset1_b32 s18, 16
s_branch 43
s_lshr_b32 s88, s88, 4
s_add_u32 s89, s88, s90
s_sub_u32 s89, s89, s91
s_mov_b32 s88, 0
s_lshl_b32 s89, s89, 4
s_min_u32 s89, s89, s16
s_bitset1_b32 s18, 17
s_branch 12
s_bitset1_b32 s18, 18
s_mov_b32 s43, 0
s_mov_b32 s47, 0
s_mov_b32 s85, 16
s_branch 260
s_add_u32 s86, s86, 16
s_cmp_ge_u32 s86, s89
s_cbranch_scc0 28
s_bitset1_b32 s18, 22
s_sub_u32 s92, s92, s17
s_subb_u32 s93, s93, 0
s_cbranch_scc1 65273
v_add_co_u32_e32 v108, vcc, s79, v108
v_addc_co_u32_e64 v117, vcc, 0, 0, vcc
v_mad_i32_i24 v108, v117, s78, v108
v_mad_i32_i24 v110, v117, s83, v110
v_mad_i32_i24 v109, v117, s82, v109
v_cmp_ge_i32_e64 vcc, v109, 0
v_addc_co_u32_e64 v117, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v117
v_mad_i32_i24 v109, v117, s77, v109
v_add_co_u32_e32 v109, vcc, s80, v109
v_addc_co_u32_e64 v117, vcc, 0, 0, vcc
v_add_co_u32_e32 v110, vcc, v110, v117
v_mad_i32_i24 v109, v117, s77, v109
v_add_co_u32_e32 v110, vcc, s81, v110
s_mov_b32 s86, s88
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccz 131
v_readfirstlane_b32 s84, v110
v_cmp_ge_u32_e64 s[60:61], v110, s12
v_subrev_co_u32_e32 v118, vcc, s78, v108
v_subrev_co_u32_e32 v119, vcc, s77, v109
s_bfe_u32 s62, s18, 0x10014
v_lshrrev_b32_e32 v94, 2, v0
v_and_b32_e32 v120, s62, v94
s_bitcmp1_b32 s18, 22
s_cbranch_scc0 38
s_bitset0_b32 s18, 22
v_and_b32_e64 v96, v0, 1
v_mad_i32_i24 v95, v118, 2, v96
v_bfe_u32 v96, v0, 1, 1
v_mad_i32_i24 v96, v119, 2, v96
v_cvt_pk_u16_u32 v97, v95, v96
v_mul_u32_u24_e32 v95, s62, v94
v_mad_i32_i24 v95, -2, v95, v0
v_bfe_u32 v94, v94, s62, 1
v_mul_u32_u24_e32 v94, 3, v94
s_sub_u32 s62, 1, s62
v_lshrrev_b32_e32 v96, s62, v95
v_bfi_b32 v95, 64, v96, v95
v_and_b32_e32 v95, 0x7f, v95
v_xor_b32_e32 v95, v95, v94
v_lshlrev_b32_e32 v96, 14, v120
v_mad_u32_u24 v95, 4, v95, v96
v_add_co_u32_e32 v95, vcc, s98, v95
ds_write_b32 v95, v110
ds_write_b32 v95, v97 offset:512
s_add_u32 s98, s98, 0x800
s_cmp_eq_u32 s98, 0xffc0
s_cselect_b32 s98, 0xdfc0, s98
v_sub_co_u32_e64 v117, vcc, v110, s84
v_and_b32_e64 v95, v0, 3
v_ashrrev_i32_e64 v96, 1, s31
v_subrev_co_u32_e32 v95, vcc, v96, v95
v_ashrrev_i32_e64 v96, 1, s66
v_mad_i32_i24 v95, v96, 3, v95
v_mad_i32_i24 v95, v120, 3, v95
v_mad_i32_i24 v118, v118, 2, v95
v_cmp_ge_u32_e64 s[56:57], v118, s15
v_mul_lo_u32 v120, v117, s69
v_mad_i32_i24 v118, 2, v118, v120
s_or_b64 s[56:57], s[56:57], s[60:61]
v_add_co_u32_e64 v96, vcc, 1, s67
v_ashrrev_i32_e32 v96, 1, v96
v_mad_i32_i24 v119, v119, 2, v96
v_add_co_u32_e64 v96, vcc, 1, s30
v_ashrrev_i32_e32 v96, 1, v96
v_subrev_co_u32_e32 v119, vcc, v96, v119
s_lshl_b32 s63, s15, 1
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v94, v119, s63, v118
v_cndmask_b32_e64 v94, v94, -1, s[58:59]
v_add_co_u32_e32 v119, vcc, 1, v119
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v95, v119, s63, v118
v_cndmask_b32_e64 v95, v95, -1, s[58:59]
v_add_co_u32_e32 v119, vcc, 1, v119
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v96, v119, s63, v118
v_cndmask_b32_e64 v96, v96, -1, s[58:59]
v_add_co_u32_e32 v119, vcc, 1, v119
v_cmp_ge_u32_e64 s[58:59], v119, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v97, v119, s63, v118
v_cndmask_b32_e64 v97, v97, -1, s[58:59]
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
s_branch 91
v_mad_u32_u24 v119, 5, v0, 2
v_lshlrev_b32_e32 v117, 1, v0
v_bfi_b32 v119, 4, v119, v117
v_bfe_u32 v117, v119, 2, 2
v_min_u32_e32 v117, 2, v117
v_bfe_u32 v119, v0, 1, 1
v_mad_u32_u24 v117, 2, v117, v119
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
v_mad_u32_u24 v118, s74, v111, v117
s_sub_u32 s58, s28, s67
s_sub_u32 s58, s58, 5
s_bitcmp1_b32 s18, 0
s_cselect_b32 s58, s58, s67
v_mov_b32_e32 v120, s58
s_lshl_b32 s61, s29, 1
v_cmp_ge_u32_e64 s[58:59], v120, s28
v_mad_i32_i24 v94, v120, s61, v118
s_or_b64 s[58:59], s[58:59], s[56:57]
v_cndmask_b32_e64 v94, v94, -1, s[58:59]
v_mov_b32_e32 v95, v94
v_add_co_u32_e64 v120, vcc, v120, 2
v_cmp_ge_u32_e64 s[58:59], v120, s28
v_mad_i32_i24 v97, v120, s61, v118
s_or_b64 s[58:59], s[58:59], s[56:57]
v_cndmask_b32_e64 v97, v97, -1, s[58:59]
v_add_co_u32_e64 v120, vcc, v120, 2
v_cmp_ge_u32_e64 s[58:59], v120, s28
v_mad_i32_i24 v96, v120, s61, v118
s_or_b64 s[58:59], s[58:59], s[56:57]
v_cndmask_b32_e64 v96, v96, -1, s[58:59]
v_add_co_u32_e64 v117, vcc, v111, s86
v_cmp_lt_u32_e64 vcc, v117, s16
v_cndmask_b32_e32 v94, -1, v94, vcc
v_cndmask_b32_e32 v95, -1, v95, vcc
v_cndmask_b32_e32 v96, -1, v96, vcc
v_cndmask_b32_e32 v97, -1, v97, vcc
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
s_mov_b64 vcc, s[10:11]
s_add_u32 s56, s38, 0xec
s_addc_u32 s57, s39, 0
s_sub_u32 s56, s56, s100
s_subb_u32 s57, s57, 0
s_setpc_b64 s[56:57]
s_and_b32 s56, 0x900000, s18
s_subb_u32 s66, s66, 1
s_cbranch_scc0 65296
s_and_b32 s56, 0x900000, s18
s_subb_u32 s66, s65, 1
s_add_u32 s67, s67, 6
s_cmp_ge_u32 s67, s28
s_cbranch_scc0 65290
s_mov_b32 s67, 1
s_cmp_ge_u32 s67, s28
s_addc_u32 s68, s68, 1
s_cmp_gt_u32 s68, 1
s_cbranch_scc0 65285
s_mov_b32 s68, 0
s_mov_b32 s67, 0
s_branch 65251
v_mac_f32_dpp v4, v4, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v5, v5, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v2, v2, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v3, v3, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v3, v4, v3  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v2, v5, v2  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v3, v3, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v2, v2, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v2, v3, v2  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v2, v2
v_mac_f32_dpp v8, v8, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v9, v9, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v6, v6, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v7, v7, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v7, v8, v7  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v6, v9, v6  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v7, v7, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v6, v6, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v3, v7, v6  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v3, v3
v_mac_f32_dpp v12, v12, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v13, v13, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v10, v10, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v11, v11, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v11, v12, v11  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v10, v13, v10  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v11, v11, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v10, v10, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v4, v11, v10  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v4, v4
v_mac_f32_dpp v16, v16, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v17, v17, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v14, v14, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v15, v15, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v15, v16, v15  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v14, v17, v14  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v15, v15, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v14, v14, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v5, v15, v14  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v5, v5
v_mac_f32_dpp v20, v20, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v21, v21, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v18, v18, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v19, v19, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v19, v20, v19  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v18, v21, v18  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v19, v19, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v18, v18, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v6, v19, v18  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v6, v6
v_mac_f32_dpp v24, v24, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v25, v25, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v22, v22, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v23, v23, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v23, v24, v23  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v22, v25, v22  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v23, v23, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v22, v22, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v7, v23, v22  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v7, v7
v_mac_f32_dpp v28, v28, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v29, v29, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v26, v26, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v27, v27, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v27, v28, v27  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v26, v29, v26  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v27, v27, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v26, v26, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v8, v27, v26  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v8, v8
v_mac_f32_dpp v32, v32, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v33, v33, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v30, v30, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v31, v31, v113  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v31, v32, v31  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v30, v33, v30  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v31, v31, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v30, v30, v114  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v9, v31, v30  row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v9, v9
s_mov_b64 s[58:59], s[48:49]
s_mov_b32 s57, s51
v_bfe_u32 v117, s18, 21, 1
v_sub_co_u32_e64 v117, vcc, v117, 1
v_cndmask_b32_e32 v117, v100, v98, vcc
v_cndmask_b32_e32 v118, v101, v99, vcc
v_readlane_b32 s56, v116, 0
v_add_f16_e64 v2, v2, s56
v_mul_f16_e64 v119, v2, s36
v_max_f16_e32 v2, v2, v119
buffer_store_short v2, v117, s[48:51], 0 offen
v_add_f16_e64 v3, v3, s56
v_mul_f16_e64 v119, v3, s36
v_max_f16_e32 v3, v3, v119
buffer_store_short v3, v118, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
v_readlane_b32 s56, v116, 1
v_add_f16_e64 v4, v4, s56
v_mul_f16_e64 v119, v4, s36
v_max_f16_e32 v4, v4, v119
buffer_store_short v4, v117, s[48:51], 0 offen
v_add_f16_e64 v5, v5, s56
v_mul_f16_e64 v119, v5, s36
v_max_f16_e32 v5, v5, v119
buffer_store_short v5, v118, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
s_lshl_b32 s56, s71, 1
s_add_u32 s48, s48, s56
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 2
s_cselect_b32 s51, 0, s51
v_readlane_b32 s56, v116, 4
v_add_f16_e64 v6, v6, s56
v_mul_f16_e64 v119, v6, s36
v_max_f16_e32 v6, v6, v119
buffer_store_short v6, v117, s[48:51], 0 offen
v_add_f16_e64 v7, v7, s56
v_mul_f16_e64 v119, v7, s36
v_max_f16_e32 v7, v7, v119
buffer_store_short v7, v118, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
v_readlane_b32 s56, v116, 5
v_add_f16_e64 v8, v8, s56
v_mul_f16_e64 v119, v8, s36
v_max_f16_e32 v8, v8, v119
buffer_store_short v8, v117, s[48:51], 0 offen
v_add_f16_e64 v9, v9, s56
v_mul_f16_e64 v119, v9, s36
v_max_f16_e32 v9, v9, v119
buffer_store_short v9, v118, s[48:51], 0 offen
s_add_u32 s48, s48, s71
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s51, 0, s51
s_lshl_b32 s56, s71, 1
s_add_u32 s48, s48, s56
s_addc_u32 s49, s49, 0
s_lshl_b32 s56, s56, 2
s_add_u32 s48, s48, s56
s_addc_u32 s49, s49, 0
s_sub_u32 s95, s95, 10
s_cselect_b32 s51, 0, s51
s_bitcmp1_b32 s18, 21
s_cselect_b32 s51, s51, s57
s_cselect_b32 s48, s48, s58
s_cselect_b32 s49, s49, s59
s_cselect_b32 s57, 0, 16
s_cselect_b32 s58, 32, 0
s_add_u32 s95, s95, s57
s_add_u32 s52, s52, s58
s_addc_u32 s53, s53, 0
s_sub_u32 s54, s54, s58
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
s_bitcmp0_b32 s18, 21
s_addc_u32 s96, s64, 0
s_lshr_b32 s96, s96, 1
s_mul_i32 s96, s96, s65
s_lshr_b32 s96, s96, 1
s_mul_i32 s96, s96, s13
s_sub_u32 s96, s96, 1
s_cbranch_scc1 65214
s_add_u32 s56, s95, s94
s_cmp_lt_i32 s56, 0
s_cbranch_scc0 101
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 127
v_lshrrev_b32_e32 v119, 1, v0
v_bfi_b32 v119, 64, v119, v0
v_and_b32_e32 v98, 0x7f, v119
v_lshlrev_b32_e32 v98, 2, v98
v_add_co_u32_e64 v98, vcc, v98, s99
ds_read_b32 v99, v98 offset:512
ds_read_b32 v98, v98
s_add_u32 s99, s99, 0x800
s_cmp_eq_u32 s99, 0xffc0
s_cselect_b32 s99, 0xdfc0, s99
s_waitcnt lgkmcnt(0)
v_bfe_u32 v117, v99, 16, 16
v_bfe_u32 v118, v99, 0, 16
v_readfirstlane_b32 s97, v98
v_sub_co_u32_e64 v119, vcc, v98, s97
v_mul_lo_u32 v119, v119, s70
v_cmp_ge_u32_e64 s[56:57], v98, s12
v_lshlrev_b32_e32 v118, 1, v118
s_and_b32 s58, 1, s31
v_add_co_u32_e32 v118, vcc, s58, v118
v_lshlrev_b32_e32 v117, 1, v117
s_and_b32 s58, 1, s30
v_subrev_co_u32_e32 v117, vcc, s58, v117
v_mad_i32_i24 v98, v117, s33, v118
v_mad_i32_i24 v98, 2, v98, v119
v_subrev_co_u32_e32 v99, vcc, 2, v98
v_mad_i32_i24 v100, 2, s33, v98
v_mad_i32_i24 v101, 2, s33, v99
v_cmp_ge_u32_e64 s[62:63], v118, s33
s_or_b64 s[60:61], s[62:63], s[56:57]
v_subrev_co_u32_e32 v118, vcc, 1, v118
v_cmp_ge_u32_e64 s[62:63], v118, s33
s_or_b64 s[62:63], s[62:63], s[56:57]
v_cmp_ge_u32_e64 s[58:59], v117, s32
s_or_b64 s[56:57], s[60:61], s[58:59]
v_cndmask_b32_e64 v98, v98, -1, s[56:57]
s_or_b64 s[56:57], s[62:63], s[58:59]
v_cndmask_b32_e64 v99, v99, -1, s[56:57]
v_add_co_u32_e32 v117, vcc, 1, v117
v_cmp_ge_u32_e64 s[58:59], v117, s32
s_or_b64 s[56:57], s[60:61], s[58:59]
v_cndmask_b32_e64 v100, v100, -1, s[56:57]
s_or_b64 s[56:57], s[62:63], s[58:59]
v_cndmask_b32_e64 v101, v101, -1, s[56:57]
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
s_mov_b64 vcc, s[10:11]
s_add_u32 s56, s38, 0xac
s_addc_u32 s57, s39, 0
s_setpc_b64 s[56:57]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 2
s_nop 1
s_subb_u32 s60, 0, 0
_s_add_u32_lit_gfx9 58, 38, 64
s_addc_u32 s59, s39, 0
s_add_u32 s60, s60, 1
s_nop 0
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
s_nop 3
s_cbranch_vccz 65532
s_setpc_b64 s[58:59]
s_endpgm
