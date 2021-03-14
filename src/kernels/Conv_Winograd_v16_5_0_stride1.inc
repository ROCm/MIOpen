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

.macro _s_add_u32_lit_gfx9 sdst, ssrc, lit
    .long  0x8000FF00 + ((\sdst) << 16) + ((\ssrc) << 0)
    .long \lit
.endm

s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_mov_b32_e32 v104, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s99, 0xdfc0
s_mov_b32 s98, 0xdfc0
s_mov_b32 s87, 0x200
v_lshlrev_b32_e32 v106, 2, v0
v_add_co_u32_e32 v106, vcc, 0xffc0, v106
v_cmp_ge_u32_e32 vcc, 3, v0
s_cbranch_vccz 5
v_mov_b32_e32 v105, 0
v_cndmask_b32_e32 v106, -1, v106, vcc
ds_write_b32 v106, v105
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
s_barrier
v_readfirstlane_b32 s64, v0
s_lshr_b32 s64, s64, 4
s_and_b32 s94, s64, 8
s_subb_u32 s94, 0, 0
s_xnor_b32 s94, s94, s64
s_and_b32 s94, s94, 20
s_mov_b64 s[48:49], s[6:7]
s_load_dwordx16 s[12:27], s[48:49], 0x0
s_load_dwordx4 s[28:31], s[48:49], 0x40
s_load_dwordx2 s[32:33], s[48:49], 0x50
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
s_load_dwordx2 s[34:35], s[48:49], 0x58
s_mov_b32 s36, 1.0
s_bitcmp1_b32 s18, 8
s_cbranch_scc0 2
s_load_dword s36, s[48:49], 0x60
s_mov_b32 s50, 0
s_cmp_gt_u32 s12, 0xffff
s_addc_u32 s50, s50, 0
s_cmp_lt_u32 s12, 1
s_addc_u32 s50, s50, 0
s_cmp_gt_u32 s13, 0xffff
s_addc_u32 s50, s50, 0
s_cmp_lt_u32 s13, 1
s_addc_u32 s50, s50, 0
s_cmp_gt_u32 s14, 0xffff
s_addc_u32 s50, s50, 0
s_cmp_lt_u32 s14, 1
s_addc_u32 s50, s50, 0
s_cmp_gt_u32 s15, 0xffff
s_addc_u32 s50, s50, 0
s_cmp_lt_u32 s15, 1
s_addc_u32 s50, s50, 0
s_cmp_gt_u32 s16, 0xffff
s_addc_u32 s50, s50, 0
s_cmp_lt_u32 s16, 1
s_addc_u32 s50, s50, 0
s_cmp_gt_u32 s17, 0xffff
s_addc_u32 s50, s50, 0
s_cmp_lt_u32 s17, 0
s_addc_u32 s50, s50, 0
s_cmp_gt_u32 s28, 0xffff
s_addc_u32 s50, s50, 0
s_cmp_lt_u32 s28, 1
s_addc_u32 s50, s50, 0
s_cmp_gt_u32 s29, 0xffff
s_addc_u32 s50, s50, 0
s_cmp_lt_u32 s29, 1
s_addc_u32 s50, s50, 0
s_bitcmp1_b32 s18, 7
s_cbranch_scc0 7
s_bitcmp1_b32 s18, 6
s_cbranch_scc0 5
s_waitcnt lgkmcnt(0)
s_and_b32 s35, s35, 0xffff
s_load_dwordx2 s[34:35], s[34:35], 0x0
s_bitcmp1_b32 s18, 9
s_cbranch_scc0 5
s_load_dwordx4 s[40:43], s[48:49], 0x64
s_load_dwordx2 s[44:45], s[48:49], 0x74
s_branch 61
s_cmp_gt_u32 s32, 0xffff
s_addc_u32 s50, s50, 0
s_cmp_lt_u32 s32, 0
s_addc_u32 s50, s50, 0
s_cmp_gt_u32 s33, 0xffff
s_addc_u32 s50, s50, 0
s_cmp_lt_u32 s33, 0
s_addc_u32 s50, s50, 0
s_mul_i32 s52, s14, s15
s_lshr_b32 s56, -1, 16
s_and_b32 s56, s56, s52
s_lshr_b32 s57, s52, 16
s_mul_i32 s57, s57, s13
s_mul_i32 s54, s56, s13
s_lshl_b32 s56, s57, 16
s_lshr_b32 s57, s57, 16
s_add_u32 s54, s56, s54
s_addc_u32 s55, s57, 0
s_cmp_gt_u32 s54, 0x10000000
s_addc_u32 s50, s55, s50
s_lshl_b32 s40, s54, 2
s_lshl_b32 s41, s52, 2
s_mul_i32 s53, s32, s33
s_lshr_b32 s56, -1, 16
s_and_b32 s56, s56, s53
s_lshr_b32 s57, s53, 16
s_mul_i32 s57, s57, s16
s_mul_i32 s54, s56, s16
s_lshl_b32 s56, s57, 16
s_lshr_b32 s57, s57, 16
s_add_u32 s54, s56, s54
s_addc_u32 s55, s57, 0
s_cmp_gt_u32 s54, 0x10000000
s_addc_u32 s50, s55, s50
s_lshl_b32 s44, s54, 2
s_lshl_b32 s45, s53, 2
s_cmp_eq_u32 s50, 0
s_cbranch_scc0 3728
s_mul_i32 s52, s28, s29
s_lshl_b32 s52, s52, 2
s_bitcmp1_b32 s18, 2
s_cselect_b32 s53, s16, s13
s_lshr_b32 s56, -1, 16
s_and_b32 s56, s56, s52
s_lshr_b32 s57, s52, 16
s_mul_i32 s57, s57, s53
s_mul_i32 s54, s56, s53
s_lshl_b32 s56, s57, 16
s_lshr_b32 s57, s57, 16
s_add_u32 s54, s56, s54
s_addc_u32 s55, s57, 0
s_cmp_gt_u32 s54, 2.0
s_addc_u32 s50, s55, s50
s_mov_b32 s53, s54
s_bitcmp1_b32 s18, 2
s_cselect_b32 s43, s53, s52
s_cselect_b32 s42, s52, s53
s_and_b32 s18, s18, 0x3c7
s_waitcnt lgkmcnt(0)
s_and_b32 s23, s23, 0xffff
s_and_b32 s25, s25, 0xffff
s_and_b32 s21, s21, 0xffff
s_and_b32 s27, s27, 0xffff
s_and_b32 s35, s35, 0xffff
s_lshl_b32 s47, s43, 1
s_and_b32 s54, 0, s30
s_addc_u32 s54, s32, 0
s_ashr_i32 s54, s54, 0
s_add_u32 s52, s54, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s52
v_readfirstlane_b32 s52, v2
s_andn2_b32 s54, 0, s31
s_addc_u32 s54, s33, 0
s_ashr_i32 s54, s54, 0
s_add_u32 s53, s54, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s53
v_readfirstlane_b32 s53, v2
s_sub_u32 s78, 0, s53
s_sub_u32 s77, 0, s52
s_add_u32 s72, s28, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s72
v_readfirstlane_b32 s72, v2
s_add_u32 s73, s29, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s73
v_readfirstlane_b32 s73, v2
v_mad_i32_i24 v2, 2, s72, -1
v_sub_co_u32_e64 v2, vcc, v2, s28
v_addc_co_u32_e64 v2, vcc, 0, 0, vcc
v_readfirstlane_b32 s54, v2
s_and_b32 s54, s54, 0
s_and_b32 s54, s54, s72
s_add_u32 s72, s72, s54
v_readfirstlane_b32 s55, v0
s_and_b32 s58, s55, 64
s_cselect_b32 s58, 0x80000, 0
s_or_b32 s18, s18, s58
s_lshl_b32 s46, s41, 1
s_lshl_b32 s54, s73, 0
s_cmp_eq_u32 s54, 1
s_cbranch_scc0 5
s_bitcmp1_b32 s18, 2
s_cselect_b32 s58, 0, 0x1000000
s_or_b32 s18, s18, s58
s_branch 6
s_bitset1_b32 s18, 23
s_bitset1_b32 s18, 20
s_lshr_b32 s46, s46, 1
s_lshr_b32 s47, s47, 1
s_add_u32 s73, s73, 1
s_and_b32 s73, s73, -2
v_bfe_u32 v3, v0, 2, 6
v_lshrrev_b32_e32 v99, 1, v3
v_readfirstlane_b32 s58, v0
s_bitcmp1_b32 s18, 24
s_cselect_b32 s58, s58, -1
s_bitcmp0_b32 s58, 8
s_cselect_b32 s58, 0x80000, 0
s_bitcmp1_b32 s18, 20
s_cselect_b32 s58, 0x80000, s58
s_andn2_b32 s18, s18, s58
s_cmp_eq_u32 s58, 0
s_cselect_b32 s58, 15, 0
v_bfi_b32 v99, s58, v3, v99
s_mul_i32 s92, s12, s52
s_sub_u32 s92, s92, 1
s_lshr_b32 s92, s92, 0
s_add_u32 s92, s92, 1
s_lshr_b32 s56, -1, 16
s_and_b32 s56, s56, s92
s_lshr_b32 s57, s92, 16
s_mul_i32 s57, s57, s53
s_mul_i32 s92, s56, s53
s_lshl_b32 s56, s57, 16
s_lshr_b32 s57, s57, 16
s_add_u32 s92, s56, s92
s_addc_u32 s93, s57, 0
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
v_cndmask_b32_e32 v7, 0, v99, vcc
v_cmp_eq_u32_e64 s[56:57], 3, v6
v_bfe_u32 v97, v7, 0, 5
v_mad_u32_u24 v97, v4, 32, v97
v_cvt_f32_u32_e32 v8, s53
v_rcp_f32_e32 v8, v8
v_mul_f32_e32 v8, 0x4f800000, v8
v_cvt_u32_f32_e32 v8, v8
v_mul_lo_u32 v9, s53, v8
v_mul_hi_u32 v10, s53, v8
v_sub_co_u32_e32 v11, vcc, 0, v9
v_cmp_ne_i32_e64 s[58:59], 0, v10
v_cndmask_b32_e64 v9, v11, v9, s[58:59]
v_mul_hi_u32 v9, v9, v8
v_sub_co_u32_e32 v10, vcc, v8, v9
v_add_co_u32_e32 v8, vcc, v8, v9
v_cndmask_b32_e64 v8, v8, v10, s[58:59]
v_mul_hi_u32 v8, v8, v97
v_mul_lo_u32 v9, v8, s53
v_sub_co_u32_e32 v10, vcc, v97, v9
v_cmp_ge_u32_e64 s[58:59], v97, v9
v_cmp_ge_u32_e64 s[60:61], v10, s53
v_add_co_u32_e32 v10, vcc, 1, v8
s_and_b64 s[60:61], s[58:59], s[60:61]
v_add_co_u32_e32 v9, vcc, -1, v8
v_cndmask_b32_e64 v10, v8, v10, s[60:61]
v_cndmask_b32_e64 v10, v9, v10, s[58:59]
v_cmp_ne_i32_e64 vcc, 0, s53
v_cndmask_b32_e32 v98, -1, v10, vcc
v_mad_i32_i24 v96, v98, s78, v97
v_lshrrev_b32_e32 v97, 5, v7
v_mad_u32_u24 v97, v98, 1, v97
v_cndmask_b32_e64 v97, v97, 1, s[56:57]
v_cvt_f32_u32_e32 v8, s52
v_rcp_f32_e32 v8, v8
v_mul_f32_e32 v8, 0x4f800000, v8
v_cvt_u32_f32_e32 v8, v8
v_mul_lo_u32 v9, s52, v8
v_mul_hi_u32 v10, s52, v8
v_sub_co_u32_e32 v11, vcc, 0, v9
v_cmp_ne_i32_e64 s[58:59], 0, v10
v_cndmask_b32_e64 v9, v11, v9, s[58:59]
v_mul_hi_u32 v9, v9, v8
v_sub_co_u32_e32 v10, vcc, v8, v9
v_add_co_u32_e32 v8, vcc, v8, v9
v_cndmask_b32_e64 v8, v8, v10, s[58:59]
v_mul_hi_u32 v8, v8, v97
v_mul_lo_u32 v9, v8, s52
v_sub_co_u32_e32 v10, vcc, v97, v9
v_cmp_ge_u32_e64 s[58:59], v97, v9
v_cmp_ge_u32_e64 s[60:61], v10, s52
v_add_co_u32_e32 v10, vcc, 1, v8
s_and_b64 s[60:61], s[58:59], s[60:61]
v_add_co_u32_e32 v9, vcc, -1, v8
v_cndmask_b32_e64 v10, v8, v10, s[60:61]
v_cndmask_b32_e64 v10, v9, v10, s[58:59]
v_cmp_ne_i32_e64 vcc, 0, s52
v_cndmask_b32_e32 v98, -1, v10, vcc
v_mad_i32_i24 v97, v98, s77, v97
v_readlane_b32 s79, v96, 2
v_readlane_b32 s80, v97, 2
v_readlane_b32 s81, v98, 2
v_readlane_b32 s82, v97, 3
v_readlane_b32 s83, v98, 3
v_add_co_u32_e64 v96, vcc, v96, s78
v_add_co_u32_e64 v97, vcc, v97, s77
v_mov_b32_dpp v98, v98  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v96, v96  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v97, v97  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
s_mov_b32 s50, 0x80000000
s_mov_b32 s51, 0x20000
s_mov_b32 s54, 0x80000000
s_mov_b32 s55, 0x20000
s_mov_b32 s58, 0x80000000
s_mov_b32 s59, 0x20000
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 5
v_xor_b32_dpp v100, v0, v0  quad_perm:[2,3,2,1] row_mask:0xf bank_mask:0xf
v_subrev_co_u32_e32 v100, vcc, 1, v100
v_cvt_f32_i32_e32 v100, v100
s_branch 4
v_xor_b32_dpp v100, v0, v0  quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32_e32 v100, vcc, 1, v100
v_cvt_f32_i32_e32 v100, v100
v_mov_b32_e32 v101, 1
v_xor_b32_dpp v101, v0, v0  quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v101, v0, v0  quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32_e32 v101, vcc, 1, v101
v_mov_b32_e32 v102, 1
v_xor_b32_dpp v102, v0, v0  quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v102, v0, v0  quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32_e32 v102, vcc, 1, v102
v_cvt_f32_i32_e32 v101, v101
v_cvt_f32_i32_e32 v102, v102
v_lshrrev_b32_e64 v105, 2, s94
v_and_b32_e32 v106, 3, v0
v_lshrrev_b32_e32 v107, 1, v0
v_bfi_b32 v107, 64, v107, v0
v_bfe_u32 v107, v107, 4, 3
v_mad_u32_u24 v94, v107, 4, v106
v_lshlrev_b32_e32 v94, 4, v94
v_mad_u32_u24 v95, v105, 4, v106
v_lshlrev_b32_e32 v95, 4, v95
v_bfe_u32 v105, v0, 2, 2
v_and_b32_e32 v106, 1, v105
v_mad_u32_u24 v108, v105, 16, v106
v_lshlrev_b32_e32 v108, 6, v108
v_xor_b32_e32 v95, v95, v108
v_mul_u32_u24_e32 v108, 0x400, v105
v_xor_b32_e32 v94, v94, v108
s_lshr_b32 s94, s94, 0
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 47
s_and_b32 s65, s18, 0x1100000
s_addc_u32 s65, 0, 0
v_lshrrev_b32_e32 v108, 1, v0
s_mul_i32 s64, 60, s65
s_sub_u32 s64, 63, s64
v_bfi_b32 v108, s64, v0, v108
v_and_b32_e32 v105, 1, v108
v_bfe_u32 v106, v108, 1, 1
v_xor_b32_e32 v105, v105, v106
v_bfe_u32 v107, v108, 3, 1
v_mad_u32_u24 v106, v106, 2, v107
v_mul_u32_u24_e32 v105, 0x118, v105
v_bfe_u32 v107, v108, 2, 1
v_mad_u32_u24 v106, v106, 2, v105
v_xor_b32_e32 v106, v106, v107
v_and_b32_e32 v107, 0xf0, v108
v_xor_b32_e32 v106, v106, v107
s_mul_i32 s64, 4, s65
s_sub_u32 s64, 6, s64
v_bfe_u32 v108, v0, s64, 1
v_mul_u32_u24_e32 v108, 0x1040, v108
v_xor_b32_e32 v91, 0x314, v106
v_xor_b32_e32 v92, 0x31c, v106
v_xor_b32_e32 v93, 8, v106
v_mov_b32_e32 v90, v106
v_mad_u32_u24 v90, 4, v90, v108
v_mad_u32_u24 v91, 4, v91, v108
v_mad_u32_u24 v92, 4, v92, v108
v_mad_u32_u24 v93, 4, v93, v108
s_branch 44
s_bfe_u32 s65, s18, 0x10014
v_lshrrev_b32_e32 v108, 1, v0
s_mul_i32 s64, 60, s65
s_sub_u32 s64, 63, s64
v_bfi_b32 v108, s64, v0, v108
v_and_b32_e32 v105, 1, v108
v_bfe_u32 v106, v108, 1, 1
v_bfe_u32 v107, v108, 3, 1
v_xor_b32_e32 v105, v105, v106
v_mad_u32_u24 v106, v106, 2, v107
v_mul_u32_u24_e32 v105, 0x109, v105
v_bfe_u32 v107, v108, 2, 1
v_mad_u32_u24 v106, v106, 2, v105
v_xor_b32_e32 v106, v106, v107
v_and_b32_e32 v107, 0xf0, v108
v_or_b32_e32 v106, v106, v107
s_mul_i32 s64, 4, s65
s_sub_u32 s64, 6, s64
v_bfe_u32 v108, v0, s64, 1
v_mul_u32_u24_e32 v108, 0x1040, v108
v_mad_u32_u24 v90, 4, v106, v108
v_xor_b32_e32 v91, 0x307, v106
v_mad_u32_u24 v91, 4, v91, v108
v_xor_b32_e32 v92, 0x30f, v106
v_mad_u32_u24 v92, 4, v92, v108
v_xor_b32_e32 v93, 8, v106
v_mad_u32_u24 v93, 4, v93, v108
v_subrev_co_u32_e32 v96, vcc, s79, v96
v_mov_b32_e32 v106, s78
v_cmp_lt_i32_e32 vcc, v96, v106
v_subb_co_u32_e64 v105, vcc, 0, 0, vcc
v_mad_i32_i24 v96, v105, s78, v96
v_mad_i32_i24 v98, v105, s83, v98
v_mad_i32_i24 v97, v105, s82, v97
v_mov_b32_e32 v106, s77
v_cmp_lt_i32_e32 vcc, v97, v106
v_subb_co_u32_e64 v105, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v105
v_mad_i32_i24 v97, v105, v106, v97
v_subrev_co_u32_e32 v97, vcc, s80, v97
v_cmp_lt_i32_e32 vcc, v97, v106
v_subb_co_u32_e64 v105, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v105
v_mad_i32_i24 v97, v105, s77, v97
v_subrev_co_u32_e32 v98, vcc, s81, v98
s_mov_b32 s74, 0
s_mov_b32 s75, s28
s_mov_b32 s76, 1
s_mov_b32 s88, 0
s_mov_b32 s89, s16
s_mov_b32 s86, s89
s_sub_u32 s95, -1, s94
s_sub_u32 s95, s95, 32
s_bitset1_b32 s18, 21
s_mov_b32 s59, 0
s_mov_b32 s63, 0
s_mov_b32 s96, 16
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 875
s_branch 1770
s_nop 0
s_nop 0
s_nop 0
s_nop 0
ds_read_b128 v[34:37], v94 offset:25280
s_setprio 1
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v58, v82, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
ds_write_b32 v90, v62
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_subrev_f32_e64 v70, v72, v70 div:2
v_subrev_f32_e64 v73, v71, v73 div:2
ds_read_b128 v[42:45], v95 offset:24768
s_add_u32 s87, s87, 0x200
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
buffer_load_dword v60, v84, s[48:51], 0 offen
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v63
v_add_f32_e64 v71, v72, v71 div:2
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:24896
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_mad_f32 v72, v72, 1.0, -v71
s_cbranch_scc0 2230
ds_read_b128 v[38:41], v94 offset:29440
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v59, v83, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v64
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_dpp v70, v70, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v71, v71, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v95 offset:28928
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
s_nop 0
buffer_load_dword v61, v85, s[48:51], 0 offen
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
ds_write_b32 v93, v65
v_mac_f32_dpp v72, v72, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:29056
s_nop 0
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s48, s48, s46
s_addc_u32 s49, s49, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 2154
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_mac_f32_dpp v73, v73, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[34:37], v94 offset:33536
s_setprio 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v62, v82, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
ds_write_b32 v90, v66 offset:8256
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_subrev_f32_e64 v74, v76, v74 div:2
v_subrev_f32_e64 v77, v75, v77 div:2
ds_read_b128 v[42:45], v95 offset:33024
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
buffer_load_dword v64, v84, s[48:51], 0 offen
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v67 offset:8256
v_add_f32_e64 v75, v76, v75 div:2
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:33152
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 2
ds_append v105 offset:65472
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_mad_f32 v76, v76, 1.0, -v75
s_cbranch_scc0 2086
ds_read_b128 v[38:41], v94 offset:37696
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v63, v83, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v68 offset:8256
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_dpp v74, v74, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v75, v75, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v95 offset:37184
s_mov_b32 m0, 0x2ffc0
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
buffer_load_dword v65, v85, s[48:51], 0 offen
s_waitcnt lgkmcnt(3)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
ds_write_b32 v93, v69 offset:8256
v_mac_f32_e64 v19, v35, v46
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:37312
s_cbranch_vccz 2996
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s48, s48, s46
s_addc_u32 s49, s49, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 2010
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_mac_f32_dpp v76, v76, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[34:37], v94 offset:41792
s_setprio 1
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v66, v82, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
ds_write_b32 v90, v70 offset:16512
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_dpp v77, v77, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v78, v80, v78 div:2
ds_read_b128 v[42:45], v95 offset:41280
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
buffer_load_dword v68, v84, s[48:51], 0 offen
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v71 offset:16512
v_subrev_f32_e64 v81, v79, v81 div:2
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:41408
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_add_f32_e64 v79, v80, v79 div:2
s_cbranch_scc0 1942
ds_read_b128 v[38:41], v94 offset:45952
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v67, v83, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v72 offset:16512
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mad_f32 v80, v80, 1.0, -v79
v_mac_f32_dpp v78, v78, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v95 offset:45440
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
s_nop 0
buffer_load_dword v69, v85, s[48:51], 0 offen
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
ds_write_b32 v93, v73 offset:16512
v_mac_f32_dpp v79, v79, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:45568
s_nop 0
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s48, s48, s46
s_addc_u32 s49, s49, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 1866
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_mac_f32_dpp v80, v80, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[34:37], v94 offset:512
s_setprio 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v70, v82, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
ds_write_b32 v90, v74 offset:24768
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_dpp v81, v81, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v58, v60, v58 div:2
ds_read_b128 v[42:45], v95
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
buffer_load_dword v72, v84, s[48:51], 0 offen
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v75 offset:24768
v_subrev_f32_e64 v61, v59, v61 div:2
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:128
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 2
ds_append v105 offset:65476
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_add_f32_e64 v59, v60, v59 div:2
s_cbranch_scc0 1798
ds_read_b128 v[38:41], v94 offset:4672
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v71, v83, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v76 offset:24768
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mad_f32 v60, v60, 1.0, -v59
v_mac_f32_dpp v58, v58, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v95 offset:4160
s_mov_b32 m0, 0x2ffc4
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
buffer_load_dword v73, v85, s[48:51], 0 offen
s_waitcnt lgkmcnt(3)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
ds_write_b32 v93, v77 offset:24768
v_mac_f32_e64 v19, v35, v46
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:4288
s_cbranch_vccz 2708
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s48, s48, s46
s_addc_u32 s49, s49, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 1722
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_mac_f32_dpp v59, v59, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[34:37], v94 offset:8768
s_setprio 1
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v74, v82, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
ds_write_b32 v90, v78 offset:33024
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_dpp v60, v60, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v61, v61, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[42:45], v95 offset:8256
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
buffer_load_dword v76, v84, s[48:51], 0 offen
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v79 offset:33024
v_subrev_f32_e64 v62, v64, v62 div:2
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:8384
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_subrev_f32_e64 v65, v63, v65 div:2
s_cbranch_scc0 1654
ds_read_b128 v[38:41], v94 offset:12928
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v75, v83, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v80 offset:33024
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_add_f32_e64 v63, v64, v63 div:2
v_mad_f32 v64, v64, 1.0, -v63
ds_read_b128 v[50:53], v95 offset:12416
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
s_nop 0
buffer_load_dword v77, v85, s[48:51], 0 offen
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
ds_write_b32 v93, v81 offset:33024
v_mac_f32_dpp v62, v62, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:12544
s_nop 0
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s48, s48, s46
s_addc_u32 s49, s49, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 1578
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_mac_f32_dpp v63, v63, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[34:37], v94 offset:17024
s_setprio 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v78, v82, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
ds_write_b32 v90, v58 offset:41280
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_dpp v64, v64, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v65, v65, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[42:45], v95 offset:16512
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
buffer_load_dword v80, v84, s[48:51], 0 offen
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v59 offset:41280
v_subrev_f32_e64 v66, v68, v66 div:2
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:16640
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 2
ds_append v105 offset:65480
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_subrev_f32_e64 v69, v67, v69 div:2
s_cbranch_scc0 1510
ds_read_b128 v[38:41], v94 offset:21184
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v79, v83, s[48:51], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v60 offset:41280
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_add_f32_e64 v67, v68, v67 div:2
v_mad_f32 v68, v68, 1.0, -v67
ds_read_b128 v[50:53], v95 offset:20672
s_mov_b32 m0, 0x2ffc8
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
buffer_load_dword v81, v85, s[48:51], 0 offen
s_waitcnt lgkmcnt(3)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
ds_write_b32 v93, v61 offset:41280
v_mac_f32_e64 v19, v35, v46
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:20800
s_cbranch_vccz 2420
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s48, s48, s46
s_addc_u32 s49, s49, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 1434
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_mac_f32_dpp v66, v66, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v67, v67, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v68, v68, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v69, v69, v100  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_branch 64666
s_getpc_b64 s[38:39]
s_mov_b32 s100, 0
s_sub_u32 s38, s38, 0xbbc
s_subb_u32 s39, s39, 0
s_branch 1414
s_nop 0
s_nop 0
s_nop 0
s_nop 0
ds_read_b128 v[34:37], v94 offset:25280
s_setprio 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v58, v82, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
ds_write_b32 v90, v62
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_dpp v70, v70, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v73, v73, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[42:45], v95 offset:24768
s_add_u32 s87, s87, 0x200
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v63
v_add_f32_e64 v71, v70, v73 div:2
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:24896
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_add_f32_e64 v72, v70, -v73 div:2
s_cbranch_scc0 1350
ds_read_b128 v[38:41], v94 offset:29440
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v61, v85, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v64
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_or_b32_e64 v73, v73, v73
v_or_b32_e64 v73, v73, v73
ds_read_b128 v[50:53], v95 offset:28928
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
s_nop 0
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
ds_write_b32 v93, v65
v_or_b32_e64 v73, v73, v73
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:29056
s_nop 0
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s52, s52, s47
s_addc_u32 s53, s53, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 1274
s_nop 1
s_waitcnt vmcnt(4) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_or_b32_e64 v73, v73, v73
s_nop 0
ds_read_b128 v[34:37], v94 offset:33536
s_setprio 1
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v62, v82, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
ds_write_b32 v90, v66 offset:8256
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_or_b32_e64 v73, v73, v73
v_mac_f32_dpp v74, v74, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[42:45], v95 offset:33024
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v67 offset:8256
v_mac_f32_dpp v77, v77, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:33152
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 2
ds_append v105 offset:65472
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_add_f32_e64 v75, v74, v77 div:2
s_cbranch_scc0 1206
ds_read_b128 v[38:41], v94 offset:37696
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v65, v85, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v68 offset:8256
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_add_f32_e64 v76, v74, -v77 div:2
v_or_b32_e64 v77, v77, v77
ds_read_b128 v[50:53], v95 offset:37184
s_mov_b32 m0, 0x2ffc0
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(3)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
ds_write_b32 v93, v69 offset:8256
v_mac_f32_e64 v19, v35, v46
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:37312
s_cbranch_vccz 2116
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s52, s52, s47
s_addc_u32 s53, s53, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 1130
s_nop 1
s_waitcnt vmcnt(4) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_or_b32_e64 v77, v77, v77
s_nop 0
ds_read_b128 v[34:37], v94 offset:41792
s_setprio 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v66, v82, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
ds_write_b32 v90, v70 offset:16512
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_or_b32_e64 v77, v77, v77
v_or_b32_e64 v77, v77, v77
ds_read_b128 v[42:45], v95 offset:41280
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v71 offset:16512
v_or_b32_e64 v77, v77, v77
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:41408
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_mac_f32_dpp v78, v78, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
s_cbranch_scc0 1062
ds_read_b128 v[38:41], v94 offset:45952
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v69, v85, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v72 offset:16512
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_dpp v81, v81, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v79, v78, v81 div:2
ds_read_b128 v[50:53], v95 offset:45440
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
s_nop 0
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
ds_write_b32 v93, v73 offset:16512
v_add_f32_e64 v80, v78, -v81 div:2
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:45568
s_nop 0
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s52, s52, s47
s_addc_u32 s53, s53, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 986
s_nop 1
s_waitcnt vmcnt(4) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_or_b32_e64 v81, v81, v81
s_nop 0
ds_read_b128 v[34:37], v94 offset:512
s_setprio 1
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v70, v82, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_or_b32_e64 v81, v81, v81
v_or_b32_e64 v81, v81, v81
v_or_b32_e64 v81, v81, v81
v_or_b32_e64 v81, v81, v81
ds_write_b32 v90, v74 offset:24768
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_dpp v58, v58, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v61, v61, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[42:45], v95
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v75 offset:24768
v_add_f32_e64 v59, v58, v61 div:2
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:128
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 2
ds_append v105 offset:65476
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_add_f32_e64 v60, v58, -v61 div:2
s_cbranch_scc0 910
ds_read_b128 v[38:41], v94 offset:4672
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v73, v85, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v76 offset:24768
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_or_b32_e64 v61, v61, v61
v_or_b32_e64 v61, v61, v61
ds_read_b128 v[50:53], v95 offset:4160
s_mov_b32 m0, 0x2ffc4
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(3)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
ds_write_b32 v93, v77 offset:24768
v_mac_f32_e64 v19, v35, v46
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:4288
s_cbranch_vccz 1820
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s52, s52, s47
s_addc_u32 s53, s53, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 834
s_nop 1
s_waitcnt vmcnt(4) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_or_b32_e64 v61, v61, v61
s_nop 0
ds_read_b128 v[34:37], v94 offset:8768
s_setprio 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v74, v82, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
ds_write_b32 v90, v78 offset:33024
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_or_b32_e64 v61, v61, v61
v_or_b32_e64 v61, v61, v61
ds_read_b128 v[42:45], v95 offset:8256
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v79 offset:33024
v_mac_f32_dpp v62, v62, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:8384
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_mac_f32_dpp v65, v65, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
s_cbranch_scc0 766
ds_read_b128 v[38:41], v94 offset:12928
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v77, v85, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v80 offset:33024
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_add_f32_e64 v63, v62, v65 div:2
v_add_f32_e64 v64, v62, -v65 div:2
ds_read_b128 v[50:53], v95 offset:12416
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
s_nop 0
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
ds_write_b32 v93, v81 offset:33024
v_or_b32_e64 v65, v65, v65
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:12544
s_nop 0
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s52, s52, s47
s_addc_u32 s53, s53, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 690
s_nop 1
s_waitcnt vmcnt(4) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_or_b32_e64 v65, v65, v65
s_nop 0
ds_read_b128 v[34:37], v94 offset:17024
s_setprio 1
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
s_nop 0
buffer_load_dword v78, v82, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_or_b32_e64 v65, v65, v65
v_or_b32_e64 v65, v65, v65
v_or_b32_e64 v65, v65, v65
v_mac_f32_dpp v66, v66, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
ds_write_b32 v90, v58 offset:41280
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_dpp v69, v69, v100  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v67, v66, v69 div:2
ds_read_b128 v[42:45], v95 offset:16512
s_mov_b32 s1, 0xf0f0f0f0
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
s_nop 0
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
ds_write_b32 v91, v59 offset:41280
v_add_f32_e64 v68, v66, -v69 div:2
v_mac_f32_e64 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
s_getpc_b64 s[38:39]
ds_read_b128 v[46:49], v95 offset:16640
s_nop 0
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
s_mov_b64 vcc, s[10:11]
s_mov_b32 m0, -1
s_add_u32 s96, s96, -2
s_waitcnt lgkmcnt(2)
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
s_nop 2
ds_append v105 offset:65480
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
v_or_b32_e64 v69, v69, v69
s_cbranch_scc0 614
ds_read_b128 v[38:41], v94 offset:21184
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
s_nop 0
buffer_load_dword v81, v85, s[52:55], 0 offen
s_nop 0
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
ds_write_b32 v92, v60 offset:41280
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_or_b32_e64 v69, v69, v69
v_or_b32_e64 v69, v69, v69
ds_read_b128 v[50:53], v95 offset:20672
s_mov_b32 m0, 0x2ffc8
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
s_nop 0
s_nop 0
s_waitcnt lgkmcnt(3)
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
ds_write_b32 v93, v61 offset:41280
v_mac_f32_e64 v19, v35, v46
v_mac_f32_e64 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
s_getpc_b64 s[38:39]
ds_read_b128 v[54:57], v95 offset:20800
s_cbranch_vccz 1524
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
s_mov_b64 vcc, s[10:11]
s_add_u32 s52, s52, s47
s_addc_u32 s53, s53, 0
s_add_u32 s85, s85, -2
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
s_cbranch_scc0 538
s_nop 1
s_waitcnt vmcnt(4) lgkmcnt(2)
v_mac_f32_e32 v31, v35, v49
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
v_or_b32_e64 v69, v69, v69
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_or_b32_e64 v69, v69, v69
s_branch 64650
s_getpc_b64 s[38:39]
s_mov_b32 s100, 0
s_sub_u32 s38, s38, 0xbfc
s_subb_u32 s39, s39, 0
s_branch 518
s_bitcmp1_b32 s18, 17
s_cbranch_scc1 247
s_add_u32 s92, s92, s17
s_cmp_eq_u32 s92, 0
s_cbranch_scc1 244
s_mov_b32 s93, 0
s_bitcmp1_b32 s18, 16
s_cbranch_scc1 233
s_add_u32 s91, s16, 31
s_lshr_b32 s91, s91, 5
v_mov_b32_e32 v106, s92
v_mul_u32_u24_e32 v106, s91, v106
v_add_co_u32_e32 v106, vcc, s17, v106
v_sub_co_u32_e64 v106, vcc, v106, 1
v_cvt_f32_u32_e32 v108, s17
v_rcp_f32_e32 v108, v108
v_mul_f32_e32 v108, 0x4f800000, v108
v_cvt_u32_f32_e32 v108, v108
v_mul_lo_u32 v109, s17, v108
v_mul_hi_u32 v110, s17, v108
v_sub_co_u32_e32 v111, vcc, 0, v109
v_cmp_ne_i32_e64 s[66:67], 0, v110
v_cndmask_b32_e64 v109, v111, v109, s[66:67]
v_mul_hi_u32 v109, v109, v108
v_sub_co_u32_e32 v110, vcc, v108, v109
v_add_co_u32_e32 v108, vcc, v108, v109
v_cndmask_b32_e64 v108, v108, v110, s[66:67]
v_mul_hi_u32 v108, v108, v106
v_mul_lo_u32 v109, v108, s17
v_sub_co_u32_e32 v110, vcc, v106, v109
v_cmp_ge_u32_e64 s[66:67], v106, v109
v_cmp_ge_u32_e64 s[68:69], v110, s17
v_add_co_u32_e32 v110, vcc, 1, v108
s_and_b64 s[68:69], s[66:67], s[68:69]
v_add_co_u32_e32 v109, vcc, -1, v108
v_cndmask_b32_e64 v110, v108, v110, s[68:69]
v_cndmask_b32_e64 v110, v109, v110, s[66:67]
v_cmp_ne_i32_e64 vcc, 0, s17
v_cndmask_b32_e32 v105, -1, v110, vcc
v_readfirstlane_b32 s90, v105
v_mul_u32_u24_e64 v105, v105, s8
v_cvt_f32_u32_e32 v108, s91
v_rcp_f32_e32 v108, v108
v_mul_f32_e32 v108, 0x4f800000, v108
v_cvt_u32_f32_e32 v108, v108
v_mul_lo_u32 v109, s91, v108
v_mul_hi_u32 v110, s91, v108
v_sub_co_u32_e32 v111, vcc, 0, v109
v_cmp_ne_i32_e64 s[66:67], 0, v110
v_cndmask_b32_e64 v109, v111, v109, s[66:67]
v_mul_hi_u32 v109, v109, v108
v_sub_co_u32_e32 v110, vcc, v108, v109
v_add_co_u32_e32 v108, vcc, v108, v109
v_cndmask_b32_e64 v108, v108, v110, s[66:67]
v_mul_hi_u32 v108, v108, v105
v_mul_lo_u32 v109, v108, s91
v_sub_co_u32_e32 v110, vcc, v105, v109
v_cmp_ge_u32_e64 s[66:67], v105, v109
v_cmp_ge_u32_e64 s[68:69], v110, s91
v_add_co_u32_e32 v110, vcc, 1, v108
s_and_b64 s[68:69], s[66:67], s[68:69]
v_add_co_u32_e32 v109, vcc, -1, v108
v_cndmask_b32_e64 v110, v108, v110, s[68:69]
v_cndmask_b32_e64 v110, v109, v110, s[66:67]
v_cmp_ne_i32_e64 vcc, 0, s91
v_cndmask_b32_e32 v106, -1, v110, vcc
v_readfirstlane_b32 s64, v105
v_readfirstlane_b32 s88, v106
s_mul_i32 s88, s88, s91
s_sub_u32 s88, s64, s88
v_sub_co_u32_e32 v106, vcc, s8, v106
v_sub_co_u32_e32 v106, vcc, s17, v106
v_readfirstlane_b32 s66, v0
v_cmp_eq_u32_e32 vcc, s66, v0
v_cndmask_b32_e32 v106, 1, v106, vcc
s_sub_u32 s70, 0, s78
s_sub_u32 s71, 0, s77
v_mul_u32_u24_e64 v110, v106, 32
v_cvt_f32_u32_e32 v111, s70
v_rcp_f32_e32 v111, v111
v_mul_f32_e32 v111, 0x4f800000, v111
v_cvt_u32_f32_e32 v111, v111
v_mul_lo_u32 v112, s70, v111
v_mul_hi_u32 v113, s70, v111
v_sub_co_u32_e32 v114, vcc, 0, v112
v_cmp_ne_i32_e64 s[66:67], 0, v113
v_cndmask_b32_e64 v112, v114, v112, s[66:67]
v_mul_hi_u32 v112, v112, v111
v_sub_co_u32_e32 v113, vcc, v111, v112
v_add_co_u32_e32 v111, vcc, v111, v112
v_cndmask_b32_e64 v111, v111, v113, s[66:67]
v_mul_hi_u32 v111, v111, v110
v_mul_lo_u32 v112, v111, s70
v_sub_co_u32_e32 v113, vcc, v110, v112
v_cmp_ge_u32_e64 s[66:67], v110, v112
v_cmp_ge_u32_e64 s[68:69], v113, s70
v_add_co_u32_e32 v113, vcc, 1, v111
s_and_b64 s[68:69], s[66:67], s[68:69]
v_add_co_u32_e32 v112, vcc, -1, v111
v_cndmask_b32_e64 v113, v111, v113, s[68:69]
v_cndmask_b32_e64 v113, v112, v113, s[66:67]
v_cmp_ne_i32_e64 vcc, 0, s70
v_cndmask_b32_e32 v108, -1, v113, vcc
v_mad_i32_i24 v109, v108, s78, v110
v_mul_u32_u24_e64 v110, v108, 1
v_cvt_f32_u32_e32 v111, s71
v_rcp_f32_e32 v111, v111
v_mul_f32_e32 v111, 0x4f800000, v111
v_cvt_u32_f32_e32 v111, v111
v_mul_lo_u32 v112, s71, v111
v_mul_hi_u32 v113, s71, v111
v_sub_co_u32_e32 v114, vcc, 0, v112
v_cmp_ne_i32_e64 s[66:67], 0, v113
v_cndmask_b32_e64 v112, v114, v112, s[66:67]
v_mul_hi_u32 v112, v112, v111
v_sub_co_u32_e32 v113, vcc, v111, v112
v_add_co_u32_e32 v111, vcc, v111, v112
v_cndmask_b32_e64 v111, v111, v113, s[66:67]
v_mul_hi_u32 v111, v111, v110
v_mul_lo_u32 v112, v111, s71
v_sub_co_u32_e32 v113, vcc, v110, v112
v_cmp_ge_u32_e64 s[66:67], v110, v112
v_cmp_ge_u32_e64 s[68:69], v113, s71
v_add_co_u32_e32 v113, vcc, 1, v111
s_and_b64 s[68:69], s[66:67], s[68:69]
v_add_co_u32_e32 v112, vcc, -1, v111
v_cndmask_b32_e64 v113, v111, v113, s[68:69]
v_cndmask_b32_e64 v113, v112, v113, s[66:67]
v_cmp_ne_i32_e64 vcc, 0, s71
v_cndmask_b32_e32 v108, -1, v113, vcc
v_mad_i32_i24 v110, v108, s77, v110
v_readfirstlane_b32 s79, v109
v_readfirstlane_b32 s80, v110
v_readfirstlane_b32 s81, v108
v_add_co_u32_e32 v96, vcc, s79, v96
v_addc_co_u32_e64 v111, vcc, 0, 0, vcc
v_mad_i32_i24 v96, v111, s78, v96
v_mad_i32_i24 v98, v111, s83, v98
v_mad_i32_i24 v97, v111, s82, v97
v_cmp_ge_i32_e64 vcc, v97, 0
v_addc_co_u32_e64 v111, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v111
v_mad_i32_i24 v97, v111, s77, v97
v_add_co_u32_e32 v97, vcc, s80, v97
v_addc_co_u32_e64 v111, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v111
v_mad_i32_i24 v97, v111, s77, v97
v_add_co_u32_e32 v98, vcc, s81, v98
v_readlane_b32 s79, v109, 1
v_readlane_b32 s80, v110, 1
v_readlane_b32 s81, v108, 1
s_add_u32 s89, s88, s90
s_cmp_le_u32 s89, s91
s_cselect_b32 s64, 0x20000, 0
s_cselect_b32 s89, s89, s91
s_or_b32 s18, s18, s64
s_lshl_b32 s88, s88, 5
s_lshl_b32 s89, s89, 5
s_min_u32 s89, s89, s16
s_cmp_eq_u32 s8, s17
s_cselect_b32 s64, 0x20000, 0
s_or_b32 s18, s18, s64
s_or_b32 s18, s18, s64
s_bitset1_b32 s18, 16
s_branch 43
s_lshr_b32 s88, s88, 5
s_add_u32 s89, s88, s90
s_sub_u32 s89, s89, s91
s_mov_b32 s88, 0
s_lshl_b32 s89, s89, 5
s_min_u32 s89, s89, s16
s_bitset1_b32 s18, 17
s_branch 12
s_bitset1_b32 s18, 18
s_mov_b32 s51, 0
s_mov_b32 s55, 0
s_mov_b32 s85, 16
s_branch 260
s_add_u32 s86, s86, 32
s_cmp_ge_u32 s86, s89
s_cbranch_scc0 28
s_bitset1_b32 s18, 22
s_sub_u32 s92, s92, s17
s_subb_u32 s93, s93, 0
s_cbranch_scc1 65275
v_add_co_u32_e32 v96, vcc, s79, v96
v_addc_co_u32_e64 v105, vcc, 0, 0, vcc
v_mad_i32_i24 v96, v105, s78, v96
v_mad_i32_i24 v98, v105, s83, v98
v_mad_i32_i24 v97, v105, s82, v97
v_cmp_ge_i32_e64 vcc, v97, 0
v_addc_co_u32_e64 v105, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v105
v_mad_i32_i24 v97, v105, s77, v97
v_add_co_u32_e32 v97, vcc, s80, v97
v_addc_co_u32_e64 v105, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v105
v_mad_i32_i24 v97, v105, s77, v97
v_add_co_u32_e32 v98, vcc, s81, v98
s_mov_b32 s86, s88
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccz 131
v_readfirstlane_b32 s84, v98
v_cmp_ge_u32_e64 s[68:69], v98, s12
v_subrev_co_u32_e32 v106, vcc, s78, v96
v_subrev_co_u32_e32 v107, vcc, s77, v97
s_bfe_u32 s70, s18, 0x10014
v_lshrrev_b32_e32 v82, 2, v0
v_and_b32_e32 v108, s70, v82
s_bitcmp1_b32 s18, 22
s_cbranch_scc0 38
s_bitset0_b32 s18, 22
v_and_b32_e64 v84, v0, 1
v_mad_i32_i24 v83, v106, 3, v84
v_bfe_u32 v84, v0, 1, 1
v_mad_i32_i24 v84, v107, 3, v84
v_cvt_pk_u16_u32 v85, v83, v84
v_mul_u32_u24_e32 v83, s70, v82
v_mad_i32_i24 v83, -2, v83, v0
v_bfe_u32 v82, v82, s70, 1
v_mul_u32_u24_e32 v82, 3, v82
s_sub_u32 s70, 1, s70
v_lshrrev_b32_e32 v84, s70, v83
v_bfi_b32 v83, 64, v84, v83
v_and_b32_e32 v83, 0x7f, v83
v_xor_b32_e32 v83, v83, v82
v_lshlrev_b32_e32 v84, 14, v108
v_mad_u32_u24 v83, 4, v83, v84
v_add_co_u32_e32 v83, vcc, s98, v83
ds_write_b32 v83, v98
ds_write_b32 v83, v85 offset:512
s_add_u32 s98, s98, 0x800
s_cmp_eq_u32 s98, 0xffc0
s_cselect_b32 s98, 0xdfc0, s98
v_sub_co_u32_e64 v105, vcc, v98, s84
v_and_b32_e64 v83, v0, 3
v_ashrrev_i32_e64 v84, 0, s31
v_subrev_co_u32_e32 v83, vcc, v84, v83
v_ashrrev_i32_e64 v84, 0, s74
v_mad_i32_i24 v83, v84, 2, v83
v_mad_i32_i24 v83, v108, 2, v83
v_mad_i32_i24 v106, v106, 3, v83
v_cmp_ge_u32_e64 s[64:65], v106, s15
v_mul_lo_u32 v108, v105, s40
v_mad_i32_i24 v106, 4, v106, v108
s_or_b64 s[64:65], s[64:65], s[68:69]
v_add_co_u32_e64 v84, vcc, 0, s75
v_ashrrev_i32_e32 v84, 0, v84
v_mad_i32_i24 v107, v107, 3, v84
v_add_co_u32_e64 v84, vcc, 0, s30
v_ashrrev_i32_e32 v84, 0, v84
v_subrev_co_u32_e32 v107, vcc, v84, v107
s_lshl_b32 s71, s15, 2
v_cmp_ge_u32_e64 s[66:67], v107, s14
s_or_b64 s[66:67], s[64:65], s[66:67]
v_mad_u32_u24 v82, v107, s71, v106
v_cndmask_b32_e64 v82, v82, -1, s[66:67]
v_add_co_u32_e32 v107, vcc, 1, v107
v_cmp_ge_u32_e64 s[66:67], v107, s14
s_or_b64 s[66:67], s[64:65], s[66:67]
v_mad_u32_u24 v83, v107, s71, v106
v_cndmask_b32_e64 v83, v83, -1, s[66:67]
v_add_co_u32_e32 v107, vcc, 1, v107
v_cmp_ge_u32_e64 s[66:67], v107, s14
s_or_b64 s[66:67], s[64:65], s[66:67]
v_mad_u32_u24 v84, v107, s71, v106
v_cndmask_b32_e64 v84, v84, -1, s[66:67]
v_add_co_u32_e32 v107, vcc, 1, v107
v_cmp_ge_u32_e64 s[66:67], v107, s14
s_or_b64 s[66:67], s[64:65], s[66:67]
v_mad_u32_u24 v85, v107, s71, v106
v_cndmask_b32_e64 v85, v85, -1, s[66:67]
s_lshr_b32 s64, -1, 16
s_and_b32 s64, s64, s40
s_lshr_b32 s65, s40, 16
s_mul_i32 s65, s65, s84
s_mul_i32 s48, s64, s84
s_lshl_b32 s64, s65, 16
s_lshr_b32 s65, s65, 16
s_add_u32 s48, s64, s48
s_addc_u32 s49, s65, 0
s_add_u32 s48, s48, s20
s_addc_u32 s49, s49, s21
s_and_b32 s64, s18, 0x80000
s_cselect_b32 s64, s41, 0
s_add_u32 s48, s48, s64
s_addc_u32 s49, s49, 0
s_mov_b32 s51, 0x20000
s_branch 91
s_bfe_u32 s64, s18, 0x10014
v_xor_b32_dpp v105, v0, v0  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_bfe_u32 v107, v0, 2, s64
v_mad_u32_u24 v105, v107, 2, v105
v_mad_u32_u24 v105, s74, 2, v105
v_sub_co_u32_e32 v107, vcc, s29, v105
v_sub_co_u32_e64 v107, vcc, v107, 1
s_bfe_u32 s66, s18, 0x10001
v_cmp_eq_u32_e64 vcc, s66, 1
v_cndmask_b32_e32 v105, v105, v107, vcc
v_cmp_ge_u32_e64 s[64:65], v105, s29
v_lshlrev_b32_e32 v105, 2, v105
s_bfe_u32 s66, s18, 0x10018
v_bfe_u32 v108, v0, 2, s66
v_mad_u32_u24 v105, s43, v108, v105
v_mad_u32_u24 v106, s42, v99, v105
s_sub_u32 s66, s28, s75
s_sub_u32 s66, s66, 2
s_bitcmp1_b32 s18, 0
s_cselect_b32 s66, s66, s75
v_mov_b32_e32 v108, s66
s_lshl_b32 s69, s29, 2
v_cmp_ge_u32_e64 s[66:67], v108, s28
v_mad_i32_i24 v82, v108, s69, v106
s_or_b64 s[66:67], s[66:67], s[64:65]
v_cndmask_b32_e64 v82, v82, -1, s[66:67]
v_mov_b32_e32 v83, v82
v_add_co_u32_e64 v108, vcc, v108, 1
v_cmp_ge_u32_e64 s[66:67], v108, s28
v_mad_i32_i24 v85, v108, s69, v106
s_or_b64 s[66:67], s[66:67], s[64:65]
v_cndmask_b32_e64 v85, v85, -1, s[66:67]
v_add_co_u32_e64 v108, vcc, v108, 1
v_cmp_ge_u32_e64 s[66:67], v108, s28
v_mad_i32_i24 v84, v108, s69, v106
s_or_b64 s[66:67], s[66:67], s[64:65]
v_cndmask_b32_e64 v84, v84, -1, s[66:67]
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v82, v83, v85, vcc
v_cndmask_b32_e32 v85, v85, v83, vcc
v_add_co_u32_e64 v105, vcc, v99, s86
v_cmp_lt_u32_e64 vcc, v105, s16
v_cndmask_b32_e32 v82, -1, v82, vcc
v_cndmask_b32_e32 v83, -1, v83, vcc
v_cndmask_b32_e32 v84, -1, v84, vcc
v_cndmask_b32_e32 v85, -1, v85, vcc
s_lshr_b32 s64, -1, 16
s_and_b32 s64, s64, s42
s_lshr_b32 s65, s42, 16
s_mul_i32 s65, s65, s86
s_mul_i32 s52, s64, s86
s_lshl_b32 s64, s65, 16
s_lshr_b32 s65, s65, 16
s_add_u32 s52, s64, s52
s_addc_u32 s53, s65, 0
s_add_u32 s52, s52, s22
s_addc_u32 s53, s53, s23
s_and_b32 s64, s18, 0x80000
s_cselect_b32 s64, s43, 0
s_add_u32 s52, s52, s64
s_addc_u32 s53, s53, 0
s_mov_b32 s55, 0x20000
s_bfe_u32 s64, s18, 0x10014
s_sub_u32 s85, s13, 1
s_lshl_b32 s85, s85, s64
_s_add_u32_lit_gfx9 64, 38, 64
s_addc_u32 s65, s39, 0
s_setpc_b64 s[64:65]
s_and_b32 s64, 0x900000, s18
s_subb_u32 s74, s74, 1
s_cbranch_scc0 65299
s_and_b32 s64, 0x900000, s18
s_subb_u32 s74, s73, 1
s_add_u32 s75, s75, 2
s_cmp_ge_u32 s75, s28
s_cbranch_scc0 65293
s_mov_b32 s75, 0
s_branch 65260
s_mov_b32 s64, 0x3c3c3c3c
s_mov_b32 s65, s64
v_mov_b32_e32 v106, v3
v_mov_b32_e32 v107, v4
v_mov_b32_e32 v108, v5
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v105, v2
v_add_f32_dpp v105, v2, v2  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v3, v3  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v4, v4  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v108, v5, v5  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v4, v4, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v5, v5, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v2, v2, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v3, v3, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v3, v4  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v2, v5  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v106, v107, v106  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v105, v108, v105  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v108, v3, v3  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v3, v3, v3  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v107, v2, v2  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v2, v2, v2  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v5, v106
v_add_f32_dpp v5, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v108  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v107, v108  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v4, v105
v_add_f32_dpp v4, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v107  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v108, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v3, v3, v2  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v2, v5, v4  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v4, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v4, v108  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v107, v107  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v3, v107, v3, s[64:65]
v_mov_b32_dpp v4, v4  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v4, v4  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v106, v7
v_mov_b32_e32 v107, v8
v_mov_b32_e32 v108, v9
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v105, v6
v_add_f32_dpp v105, v6, v6  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v7, v7  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v8, v8  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v108, v9, v9  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v8, v8, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v9, v9, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v6, v6, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v7, v7, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v7, v8  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v6, v9  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v106, v107, v106  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v105, v108, v105  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v108, v7, v7  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v7, v7, v7  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v107, v6, v6  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v6, v6, v6  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v9, v106
v_add_f32_dpp v9, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v108  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v107, v108  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v8, v105
v_add_f32_dpp v8, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v107  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v108, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v7, v7, v6  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v5, v9, v8  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v8, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v8, v108  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v107, v107  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v6, v107, v7, s[64:65]
v_mov_b32_dpp v7, v8  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v7, v8  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v106, v11
v_mov_b32_e32 v107, v12
v_mov_b32_e32 v108, v13
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v105, v10
v_add_f32_dpp v105, v10, v10  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v11, v11  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v12, v12  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v108, v13, v13  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v12, v12, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v13, v13, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v10, v10, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v11, v11, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v11, v12  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v10, v13  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v106, v107, v106  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v105, v108, v105  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v108, v11, v11  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v11, v11, v11  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v107, v10, v10  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v10, v10, v10  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v13, v106
v_add_f32_dpp v13, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v108  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v107, v108  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v12, v105
v_add_f32_dpp v12, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v107  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v108, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v11, v11, v10  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v8, v13, v12  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v12, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v12, v108  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v107, v107  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v9, v107, v11, s[64:65]
v_mov_b32_dpp v10, v12  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v10, v12  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v106, v15
v_mov_b32_e32 v107, v16
v_mov_b32_e32 v108, v17
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v105, v14
v_add_f32_dpp v105, v14, v14  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v15, v15  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v16, v16  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v108, v17, v17  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v16, v16, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v17, v17, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v14, v14, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v15, v15, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v15, v16  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v14, v17  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v106, v107, v106  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v105, v108, v105  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v108, v15, v15  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v15, v15, v15  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v107, v14, v14  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v14, v14, v14  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v17, v106
v_add_f32_dpp v17, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v108  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v107, v108  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v16, v105
v_add_f32_dpp v16, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v107  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v108, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v15, v15, v14  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v11, v17, v16  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v16, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v16, v108  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v107, v107  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v12, v107, v15, s[64:65]
v_mov_b32_dpp v13, v16  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v13, v16  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v106, v19
v_mov_b32_e32 v107, v20
v_mov_b32_e32 v108, v21
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v105, v18
v_add_f32_dpp v105, v18, v18  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v19, v19  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v20, v20  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v108, v21, v21  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v20, v20, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v21, v21, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v18, v18, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v19, v19, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v19, v20  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v18, v21  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v106, v107, v106  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v105, v108, v105  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v108, v19, v19  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v19, v19, v19  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v107, v18, v18  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v18, v18, v18  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v21, v106
v_add_f32_dpp v21, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v108  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v107, v108  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v20, v105
v_add_f32_dpp v20, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v107  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v108, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v19, v19, v18  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v14, v21, v20  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v20, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v20, v108  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v107, v107  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v15, v107, v19, s[64:65]
v_mov_b32_dpp v16, v20  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v16, v20  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v106, v23
v_mov_b32_e32 v107, v24
v_mov_b32_e32 v108, v25
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v105, v22
v_add_f32_dpp v105, v22, v22  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v23, v23  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v24, v24  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v108, v25, v25  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v24, v24, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v25, v25, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v22, v22, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v23, v23, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v23, v24  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v22, v25  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v106, v107, v106  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v105, v108, v105  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v108, v23, v23  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v23, v23, v23  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v107, v22, v22  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v22, v22, v22  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v25, v106
v_add_f32_dpp v25, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v108  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v107, v108  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v24, v105
v_add_f32_dpp v24, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v107  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v108, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v23, v23, v22  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v17, v25, v24  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v24, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v24, v108  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v107, v107  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v18, v107, v23, s[64:65]
v_mov_b32_dpp v19, v24  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v19, v24  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v106, v27
v_mov_b32_e32 v107, v28
v_mov_b32_e32 v108, v29
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v105, v26
v_add_f32_dpp v105, v26, v26  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v27, v27  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v28, v28  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v108, v29, v29  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v28, v28, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v29, v29, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v26, v26, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v27, v27, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v27, v28  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v26, v29  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v106, v107, v106  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v105, v108, v105  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v108, v27, v27  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v27, v27, v27  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v107, v26, v26  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v26, v26, v26  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v29, v106
v_add_f32_dpp v29, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v108  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v107, v108  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v28, v105
v_add_f32_dpp v28, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v107  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v108, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v27, v27, v26  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v20, v29, v28  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v28, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v28, v108  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v107, v107  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v21, v107, v27, s[64:65]
v_mov_b32_dpp v22, v28  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v22, v28  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v106, v31
v_mov_b32_e32 v107, v32
v_mov_b32_e32 v108, v33
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v105, v30
v_add_f32_dpp v105, v30, v30  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v106, v31, v31  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v107, v32, v32  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v108, v33, v33  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v32, v32, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v33, v33, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v30, v30, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v31, v31, v101  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v31, v32  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v30, v33  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v106, v107, v106  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v105, v108, v105  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v108, v31, v31  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v31, v31, v31  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v107, v30, v30  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v30, v30, v30  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v33, v106
v_add_f32_dpp v33, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v108  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v107, v108  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v32, v105
v_add_f32_dpp v32, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v107, v107  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v108, v106, v106  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v31, v31, v30  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v23, v33, v32  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v32, v105, v105  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v32, v108  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v107, v107  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v24, v107, v31, s[64:65]
v_mov_b32_dpp v25, v32  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v25, v32  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_readlane_b32 s67, v104, 0
v_add_f32_e64 v2, v2, s67
v_mul_f32_e64 v107, v2, s36
v_max_f32_e32 v2, v2, v107
buffer_store_dword v2, v86, s[56:59], 0 offen
v_add_f32_e64 v3, v3, s67
v_mul_f32_e64 v107, v3, s36
v_max_f32_e32 v3, v3, v107
buffer_store_dword v3, v87, s[56:59], 0 offen
v_add_f32_e64 v4, v4, s67
v_mul_f32_e64 v107, v4, s36
v_max_f32_e32 v4, v4, v107
buffer_store_dword v4, v88, s[56:59], 0 offen
s_add_u32 s56, s56, s45
s_addc_u32 s57, s57, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s59, 0, s59
v_readlane_b32 s67, v104, 1
v_add_f32_e64 v3, v3, s67
v_mul_f32_e64 v107, v3, s36
v_max_f32_e32 v3, v3, v107
buffer_store_dword v5, v86, s[56:59], 0 offen
v_add_f32_e64 v4, v4, s67
v_mul_f32_e64 v107, v4, s36
v_max_f32_e32 v4, v4, v107
buffer_store_dword v6, v87, s[56:59], 0 offen
v_add_f32_e64 v5, v5, s67
v_mul_f32_e64 v107, v5, s36
v_max_f32_e32 v5, v5, v107
buffer_store_dword v7, v88, s[56:59], 0 offen
s_add_u32 s56, s56, s45
s_addc_u32 s57, s57, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s59, 0, s59
v_readlane_b32 s67, v104, 2
v_add_f32_e64 v4, v4, s67
v_mul_f32_e64 v107, v4, s36
v_max_f32_e32 v4, v4, v107
buffer_store_dword v8, v86, s[56:59], 0 offen
v_add_f32_e64 v5, v5, s67
v_mul_f32_e64 v107, v5, s36
v_max_f32_e32 v5, v5, v107
buffer_store_dword v9, v87, s[56:59], 0 offen
v_add_f32_e64 v6, v6, s67
v_mul_f32_e64 v107, v6, s36
v_max_f32_e32 v6, v6, v107
buffer_store_dword v10, v88, s[56:59], 0 offen
s_add_u32 s56, s56, s45
s_addc_u32 s57, s57, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s59, 0, s59
v_readlane_b32 s67, v104, 3
v_add_f32_e64 v5, v5, s67
v_mul_f32_e64 v107, v5, s36
v_max_f32_e32 v5, v5, v107
buffer_store_dword v11, v86, s[56:59], 0 offen
v_add_f32_e64 v6, v6, s67
v_mul_f32_e64 v107, v6, s36
v_max_f32_e32 v6, v6, v107
buffer_store_dword v12, v87, s[56:59], 0 offen
v_add_f32_e64 v7, v7, s67
v_mul_f32_e64 v107, v7, s36
v_max_f32_e32 v7, v7, v107
buffer_store_dword v13, v88, s[56:59], 0 offen
s_add_u32 s56, s56, s45
s_addc_u32 s57, s57, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s59, 0, s59
s_lshl_b32 s64, s45, 2
s_add_u32 s56, s56, s64
s_addc_u32 s57, s57, 0
s_sub_u32 s95, s95, 4
s_cselect_b32 s59, 0, s59
v_readlane_b32 s67, v104, 8
v_add_f32_e64 v6, v6, s67
v_mul_f32_e64 v107, v6, s36
v_max_f32_e32 v6, v6, v107
buffer_store_dword v14, v86, s[56:59], 0 offen
v_add_f32_e64 v7, v7, s67
v_mul_f32_e64 v107, v7, s36
v_max_f32_e32 v7, v7, v107
buffer_store_dword v15, v87, s[56:59], 0 offen
v_add_f32_e64 v8, v8, s67
v_mul_f32_e64 v107, v8, s36
v_max_f32_e32 v8, v8, v107
buffer_store_dword v16, v88, s[56:59], 0 offen
s_add_u32 s56, s56, s45
s_addc_u32 s57, s57, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s59, 0, s59
v_readlane_b32 s67, v104, 9
v_add_f32_e64 v7, v7, s67
v_mul_f32_e64 v107, v7, s36
v_max_f32_e32 v7, v7, v107
buffer_store_dword v17, v86, s[56:59], 0 offen
v_add_f32_e64 v8, v8, s67
v_mul_f32_e64 v107, v8, s36
v_max_f32_e32 v8, v8, v107
buffer_store_dword v18, v87, s[56:59], 0 offen
v_add_f32_e64 v9, v9, s67
v_mul_f32_e64 v107, v9, s36
v_max_f32_e32 v9, v9, v107
buffer_store_dword v19, v88, s[56:59], 0 offen
s_add_u32 s56, s56, s45
s_addc_u32 s57, s57, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s59, 0, s59
v_readlane_b32 s67, v104, 10
v_add_f32_e64 v8, v8, s67
v_mul_f32_e64 v107, v8, s36
v_max_f32_e32 v8, v8, v107
buffer_store_dword v20, v86, s[56:59], 0 offen
v_add_f32_e64 v9, v9, s67
v_mul_f32_e64 v107, v9, s36
v_max_f32_e32 v9, v9, v107
buffer_store_dword v21, v87, s[56:59], 0 offen
v_add_f32_e64 v10, v10, s67
v_mul_f32_e64 v107, v10, s36
v_max_f32_e32 v10, v10, v107
buffer_store_dword v22, v88, s[56:59], 0 offen
s_add_u32 s56, s56, s45
s_addc_u32 s57, s57, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s59, 0, s59
v_readlane_b32 s67, v104, 11
v_add_f32_e64 v9, v9, s67
v_mul_f32_e64 v107, v9, s36
v_max_f32_e32 v9, v9, v107
buffer_store_dword v23, v86, s[56:59], 0 offen
v_add_f32_e64 v10, v10, s67
v_mul_f32_e64 v107, v10, s36
v_max_f32_e32 v10, v10, v107
buffer_store_dword v24, v87, s[56:59], 0 offen
v_add_f32_e64 v11, v11, s67
v_mul_f32_e64 v107, v11, s36
v_max_f32_e32 v11, v11, v107
buffer_store_dword v25, v88, s[56:59], 0 offen
s_add_u32 s56, s56, s45
s_addc_u32 s57, s57, 0
s_sub_u32 s95, s95, 1
s_cselect_b32 s59, 0, s59
s_add_u32 s56, s56, s64
s_addc_u32 s57, s57, 0
s_lshl_b32 s64, s64, 2
s_add_u32 s56, s56, s64
s_addc_u32 s57, s57, 0
s_sub_u32 s95, s95, 20
s_cselect_b32 s59, 0, s59
s_cselect_b32 s63, 0, s63
s_add_u32 s60, s60, 0x80
s_addc_u32 s61, s61, 0
s_sub_u32 s62, s62, 0x80
s_cselect_b32 s63, 0, s63
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
s_mul_i32 s96, s72, s73
s_mul_i32 s96, s96, s13
s_sub_u32 s96, s96, 1
s_add_u32 s64, s95, s94
s_cmp_lt_i32 s64, 0
s_cbranch_scc0 126
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 149
v_lshrrev_b32_e32 v107, 1, v0
v_bfi_b32 v107, 64, v107, v0
v_and_b32_e32 v86, 0x7f, v107
v_lshlrev_b32_e32 v86, 2, v86
v_add_co_u32_e64 v86, vcc, v86, s99
ds_read_b32 v87, v86 offset:512
ds_read_b32 v86, v86
s_add_u32 s99, s99, 0x800
s_cmp_eq_u32 s99, 0xffc0
s_cselect_b32 s99, 0xdfc0, s99
s_waitcnt lgkmcnt(0)
v_bfe_u32 v105, v87, 16, 16
v_bfe_u32 v106, v87, 0, 16
v_readfirstlane_b32 s97, v86
v_sub_co_u32_e64 v107, vcc, v86, s97
v_mul_lo_u32 v107, v107, s44
v_cmp_ge_u32_e64 s[64:65], v86, s12
v_xor_b32_dpp v108, v0, v0  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v108, v0, v0  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v89, v0, v0  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v89, v0, v0  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_add_co_u32_e32 v89, vcc, v106, v89
v_add_co_u32_e32 v108, vcc, v105, v108
v_mad_i32_i24 v86, v108, s33, v89
v_mad_i32_i24 v86, 4, v86, v107
v_cmp_ge_u32_e64 s[68:69], v89, s33
s_or_b64 s[68:69], s[68:69], s[64:65]
v_cmp_ge_u32_e64 s[66:67], v108, s32
s_or_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32_e64 v86, v86, -1, s[68:69]
v_xor_b32_dpp v108, v0, v0  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v108, v0, v0  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v89, v0, v0  quad_perm:[1,1,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32_e32 v89, vcc, v106, v89
v_add_co_u32_e32 v108, vcc, v105, v108
v_mad_i32_i24 v87, v108, s33, v89
v_mad_i32_i24 v87, 4, v87, v107
v_cmp_ge_u32_e64 s[68:69], v89, s33
s_or_b64 s[68:69], s[68:69], s[64:65]
v_cmp_ge_u32_e64 s[66:67], v108, s32
s_or_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32_e64 v87, v87, -1, s[68:69]
v_xor_b32_dpp v108, v0, v0  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v108, v0, v0  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v89, v0, v0  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v89, v0, v0  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xa
v_add_co_u32_e32 v89, vcc, v106, v89
v_add_co_u32_e32 v108, vcc, v105, v108
v_mad_i32_i24 v88, v108, s33, v89
v_mad_i32_i24 v88, 4, v88, v107
v_cmp_ge_u32_e64 s[68:69], v89, s33
s_or_b64 s[68:69], s[68:69], s[64:65]
v_cmp_ge_u32_e64 s[66:67], v108, s32
s_or_b64 s[68:69], s[68:69], s[66:67]
v_cndmask_b32_e64 v88, v88, -1, s[68:69]
s_add_u32 s95, s94, s88
s_lshr_b32 s64, -1, 16
s_and_b32 s64, s64, s44
s_lshr_b32 s65, s44, 16
s_mul_i32 s65, s65, s97
s_mul_i32 s56, s64, s97
s_lshl_b32 s64, s65, 16
s_lshr_b32 s65, s65, 16
s_add_u32 s56, s64, s56
s_addc_u32 s57, s65, 0
s_add_u32 s56, s56, s24
s_addc_u32 s57, s57, s25
s_mul_i32 s64, s45, s95
s_add_u32 s56, s56, s64
s_addc_u32 s57, s57, 0
s_mov_b32 s59, 0x20000
s_bitcmp1_b32 s18, 7
s_cselect_b32 s63, 0x20000, 0
s_lshl_b32 s64, s95, 2
s_add_u32 s60, s34, s64
s_addc_u32 s61, s35, 0
s_lshl_b32 s65, s89, 2
s_sub_u32 s62, s65, s64
s_cselect_b32 s63, 0, s63
s_sub_u32 s95, s89, s88
s_sub_u32 s95, s95, 1
s_sub_u32 s95, s95, s94
s_cselect_b32 s59, 0, s59
v_and_b32_e64 v104, v0, 63
v_lshlrev_b32_e32 v104, 2, v104
s_barrier
buffer_load_dword v104, v104, s[60:63], 0 offen
s_add_u32 s64, s38, 0x60
s_addc_u32 s65, s39, 0
s_setpc_b64 s[64:65]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 1
s_subb_u32 s68, 0, 0
_s_add_u32_lit_gfx9 66, 38, 12
s_addc_u32 s67, s39, 0
s_add_u32 s68, s68, 1
v_cmp_eq_u32_e64 vcc, src_lds_direct, s87
s_cbranch_vccz 65533
s_setpc_b64 s[66:67]
s_endpgm
