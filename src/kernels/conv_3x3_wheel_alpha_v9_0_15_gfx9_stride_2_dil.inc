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
s_mov_b32 s4, 0
s_mov_b32 s5, 0
v_mov_b32_e32 v115, 0
v_mov_b32_e32 v112, 1.0
v_mov_b32_e32 v113, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s96, 0xdfc0
s_mov_b32 s95, 0xdfc0
s_nop 0
s_nop 0
v_readfirstlane_b32 s48, v0
s_cmp_eq_u32 s48, 0
s_cmovk_i32 s49, 0xfeef
s_cmovk_i32 s92, 0x10
s_cmp_eq_u32 s48, 64
s_cmovk_i32 s49, 0xfddf
s_cmovk_i32 s92, 0x10
s_cmp_eq_u32 s48, 0x80
s_cmovk_i32 s49, 0xeffe
s_cmovk_i32 s92, 0x14
s_cmp_eq_u32 s48, 0xc0
s_cmovk_i32 s49, 0xdffd
s_cmovk_i32 s92, 0x14
s_cmp_eq_u32 s48, 0x100
s_cmovk_i32 s49, 0xfbfb
s_cmovk_i32 s92, 0x18
s_cmp_eq_u32 s48, 0x140
s_cmovk_i32 s49, 0xf7f7
s_cmovk_i32 s92, 0x18
s_cmp_eq_u32 s48, 0x180
s_cmovk_i32 s49, 0xbfbf
s_cmovk_i32 s92, 0x1c
s_cmp_eq_u32 s48, 0x1c0
s_cmovk_i32 s49, 0x7f7f
s_cmovk_i32 s92, 0x1c
s_mov_b32 s50, 0x1010101
s_mul_i32 s51, s92, s50
s_mov_b32 s50, 0x3020100
s_add_i32 s51, s51, s50
v_mov_b32_e32 v114, s51
v_and_b32_e64 v111, v0, 63
v_and_b32_e64 v116, v0, 15
v_bfe_u32 v116, s49, v116, 1
v_bfe_u32 v117, v0, 4, 2
v_cmp_eq_u32_e64 vcc, v117, 2
v_addc_co_u32_e64 v117, vcc, 0, 0, vcc
v_or_b32_e32 v116, v116, v117
v_mad_u32_u24 v111, 64, v116, v111
s_or_b32 s92, s92, 0x20000
s_add_u32 s92, 0xffc0, s92
ds_write_b8 v111, v111 offset:65472
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
s_barrier
v_readfirstlane_b32 s48, v0
s_lshr_b32 s48, s48, 4
s_and_b32 s88, s48, 8
s_subb_u32 s88, 0, 0
s_xnor_b32 s88, s88, s48
s_and_b32 s88, s88, 20
s_load_dwordx16 s[12:27], s[6:7], 0x0
s_load_dwordx4 s[28:31], s[6:7], 0x40
s_load_dwordx2 s[32:33], s[6:7], 0x50
s_waitcnt vmcnt(15) lgkmcnt(0)
s_mov_b32 s36, 0
s_cmp_gt_u32 s12, 0xffff
s_addc_u32 s36, s36, 0
s_cmp_lt_u32 s12, 1
s_addc_u32 s36, s36, 0
s_cmp_gt_u32 s13, 0xffff
s_addc_u32 s36, s36, 0
s_cmp_lt_u32 s13, 1
s_addc_u32 s36, s36, 0
s_cmp_gt_u32 s14, 0xffff
s_addc_u32 s36, s36, 0
s_cmp_lt_u32 s14, 1
s_addc_u32 s36, s36, 0
s_cmp_gt_u32 s15, 0xffff
s_addc_u32 s36, s36, 0
s_cmp_lt_u32 s15, 1
s_addc_u32 s36, s36, 0
s_cmp_gt_u32 s16, 0xffff
s_addc_u32 s36, s36, 0
s_cmp_lt_u32 s16, 1
s_addc_u32 s36, s36, 0
s_cmp_gt_u32 s17, 0xffff
s_addc_u32 s36, s36, 0
s_cmp_lt_u32 s17, 0
s_addc_u32 s36, s36, 0
s_cmp_gt_u32 s28, 0xffff
s_addc_u32 s36, s36, 0
s_cmp_lt_u32 s28, 2
s_addc_u32 s36, s36, 0
s_cmp_gt_u32 s29, 0xffff
s_addc_u32 s36, s36, 0
s_cmp_lt_u32 s29, 1
s_addc_u32 s36, s36, 0
s_and_b32 s23, s23, 0xffff
s_and_b32 s25, s25, 0xffff
s_and_b32 s21, s21, 0xffff
s_and_b32 s27, s27, 0xffff
s_cmp_gt_u32 s32, 0xffff
s_addc_u32 s36, s36, 0
s_cmp_lt_u32 s32, 0
s_addc_u32 s36, s36, 0
s_cmp_gt_u32 s33, 0xffff
s_addc_u32 s36, s36, 0
s_cmp_lt_u32 s33, 0
s_addc_u32 s36, s36, 0
s_mul_i32 s38, s14, s15
s_lshr_b32 s45, -1, 16
s_and_b32 s42, s45, s38
s_lshr_b32 s43, s38, 16
s_mul_i32 s44, s43, s13
s_mul_i32 s40, s42, s13
s_lshl_b32 s42, s44, 16
s_lshr_b32 s45, s44, 16
s_add_u32 s40, s42, s40
s_addc_u32 s41, s45, 0
s_cmp_gt_u32 s40, 0x10000000
s_addc_u32 s36, s41, s36
s_lshl_b32 s63, s40, 2
s_lshl_b32 s67, s38, 2
s_mul_i32 s39, s32, s33
s_lshr_b32 s45, -1, 16
s_and_b32 s42, s45, s39
s_lshr_b32 s43, s39, 16
s_mul_i32 s44, s43, s16
s_mul_i32 s40, s42, s16
s_lshl_b32 s42, s44, 16
s_lshr_b32 s45, s44, 16
s_add_u32 s40, s42, s40
s_addc_u32 s41, s45, 0
s_cmp_gt_u32 s40, 0x10000000
s_addc_u32 s36, s41, s36
s_lshl_b32 s64, s40, 2
s_lshl_b32 s66, s39, 2
s_cmp_eq_u32 s36, 0
s_cbranch_scc0 3259
s_and_b32 s18, s18, 7
s_mul_i32 s38, s28, s29
s_lshl_b32 s38, s38, 2
s_bitcmp1_b32 s18, 2
s_cselect_b32 s39, s16, s13
s_lshr_b32 s45, -1, 16
s_and_b32 s42, s45, s38
s_lshr_b32 s43, s38, 16
s_mul_i32 s44, s43, s39
s_mul_i32 s40, s42, s39
s_lshl_b32 s42, s44, 16
s_lshr_b32 s45, s44, 16
s_add_u32 s40, s42, s40
s_addc_u32 s41, s45, 0
s_cmp_gt_u32 s40, 2.0
s_addc_u32 s36, s41, s36
s_mov_b32 s39, s40
s_bitcmp1_b32 s18, 2
s_cselect_b32 s68, s39, s38
s_cselect_b32 s69, s38, s39
s_lshl_b32 s71, s68, 1
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_and_b32 s40, 1, s30
s_addc_u32 s40, s32, 1
s_ashr_i32 s40, s40, 1
s_add_u32 s38, s40, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s38
v_readfirstlane_b32 s38, v2
s_andn2_b32 s40, 1, s31
s_addc_u32 s40, s33, 1
s_ashr_i32 s40, s40, 1
s_add_u32 s39, s40, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s39
v_readfirstlane_b32 s39, v2
s_sub_u32 s73, 0, s39
s_sub_u32 s72, 0, s38
s_add_u32 s58, s28, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s58
v_readfirstlane_b32 s58, v2
s_add_u32 s59, s29, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s59
v_readfirstlane_b32 s59, v2
v_mad_i32_i24 v2, 3, s58, -2
v_sub_co_u32_e64 v2, vcc, v2, s28
v_addc_co_u32_e64 v2, vcc, 0, 0, vcc
v_readfirstlane_b32 s40, v2
s_and_b32 s40, s40, 1
s_and_b32 s40, s40, s58
s_add_u32 s58, s58, s40
v_readfirstlane_b32 s41, v0
s_and_b32 s44, s41, 64
s_cselect_b32 s44, 0x80000, 0
s_or_b32 s18, s18, s44
s_lshl_b32 s70, s67, 1
s_lshl_b32 s40, s59, 1
s_cmp_eq_u32 s40, 1
s_cbranch_scc1 3
s_bitset1_b32 s18, 23
s_add_u32 s59, s59, 1
s_and_b32 s59, s59, -2
v_bfe_u32 v3, v0, 2, 6
v_lshrrev_b32_e32 v107, 1, v3
s_and_b32 s44, s18, 0x100000
s_cselect_b32 s44, 0, 15
v_bfi_b32 v107, s44, v3, v107
v_bfe_u32 v3, s41, 8, 1
v_xor_b32_e64 v3, v3, 1
v_lshrrev_b32_e32 v107, v3, v107
s_mul_i32 s86, s12, s38
s_sub_u32 s86, s86, 1
s_lshr_b32 s86, s86, 0
s_add_u32 s86, s86, 1
s_lshr_b32 s45, -1, 16
s_and_b32 s42, s45, s86
s_lshr_b32 s43, s86, 16
s_mul_i32 s44, s43, s39
s_mul_i32 s86, s42, s39
s_lshl_b32 s42, s44, 16
s_lshr_b32 s45, s44, 16
s_add_u32 s86, s42, s86
s_addc_u32 s87, s45, 0
s_sub_u32 s86, s86, 1
s_subb_u32 s87, s87, 0
s_lshr_b64 s[86:87], s[86:87], 5
s_add_u32 s86, s86, 1
s_addc_u32 s87, s87, 0
v_mov_b32_e32 v4, s8
v_mov_b32_e32 v5, s17
v_and_b32_e32 v6, 3, v0
v_cmp_eq_u32_e32 vcc, 2, v6
v_cndmask_b32_e32 v4, v4, v5, vcc
v_cmp_eq_u32_e32 vcc, 1, v6
v_cndmask_b32_e32 v7, 0, v107, vcc
v_cmp_eq_u32_e64 s[42:43], 3, v6
v_bfe_u32 v105, v7, 0, 5
v_mad_u32_u24 v105, v4, 32, v105
v_cvt_f32_u32_e32 v8, s39
v_rcp_f32_e32 v8, v8
v_mul_f32_e32 v8, 0x4f800000, v8
v_cvt_u32_f32_e32 v8, v8
v_mul_lo_u32 v9, s39, v8
v_mul_hi_u32 v10, s39, v8
v_sub_co_u32_e32 v11, vcc, 0, v9
v_cmp_ne_i32_e64 s[44:45], 0, v10
v_cndmask_b32_e64 v9, v11, v9, s[44:45]
v_mul_hi_u32 v9, v9, v8
v_sub_co_u32_e32 v10, vcc, v8, v9
v_add_co_u32_e32 v8, vcc, v8, v9
v_cndmask_b32_e64 v8, v8, v10, s[44:45]
v_mul_hi_u32 v8, v8, v105
v_mul_lo_u32 v9, v8, s39
v_sub_co_u32_e32 v10, vcc, v105, v9
v_cmp_ge_u32_e64 s[44:45], v105, v9
v_cmp_ge_u32_e64 s[46:47], v10, s39
v_add_co_u32_e32 v10, vcc, 1, v8
s_and_b64 s[46:47], s[44:45], s[46:47]
v_add_co_u32_e32 v9, vcc, -1, v8
v_cndmask_b32_e64 v10, v8, v10, s[46:47]
v_cndmask_b32_e64 v10, v9, v10, s[44:45]
v_cmp_ne_i32_e64 vcc, 0, s39
v_cndmask_b32_e32 v106, -1, v10, vcc
v_mad_i32_i24 v104, v106, s73, v105
v_lshrrev_b32_e32 v105, 5, v7
v_mad_u32_u24 v105, v106, 1, v105
v_cndmask_b32_e64 v105, v105, 1, s[42:43]
v_cvt_f32_u32_e32 v8, s38
v_rcp_f32_e32 v8, v8
v_mul_f32_e32 v8, 0x4f800000, v8
v_cvt_u32_f32_e32 v8, v8
v_mul_lo_u32 v9, s38, v8
v_mul_hi_u32 v10, s38, v8
v_sub_co_u32_e32 v11, vcc, 0, v9
v_cmp_ne_i32_e64 s[44:45], 0, v10
v_cndmask_b32_e64 v9, v11, v9, s[44:45]
v_mul_hi_u32 v9, v9, v8
v_sub_co_u32_e32 v10, vcc, v8, v9
v_add_co_u32_e32 v8, vcc, v8, v9
v_cndmask_b32_e64 v8, v8, v10, s[44:45]
v_mul_hi_u32 v8, v8, v105
v_mul_lo_u32 v9, v8, s38
v_sub_co_u32_e32 v10, vcc, v105, v9
v_cmp_ge_u32_e64 s[44:45], v105, v9
v_cmp_ge_u32_e64 s[46:47], v10, s38
v_add_co_u32_e32 v10, vcc, 1, v8
s_and_b64 s[46:47], s[44:45], s[46:47]
v_add_co_u32_e32 v9, vcc, -1, v8
v_cndmask_b32_e64 v10, v8, v10, s[46:47]
v_cndmask_b32_e64 v10, v9, v10, s[44:45]
v_cmp_ne_i32_e64 vcc, 0, s38
v_cndmask_b32_e32 v106, -1, v10, vcc
v_mad_i32_i24 v105, v106, s72, v105
v_readlane_b32 s74, v104, 2
v_readlane_b32 s75, v105, 2
v_readlane_b32 s76, v106, 2
v_readlane_b32 s77, v105, 3
v_readlane_b32 s78, v106, 3
v_add_co_u32_e64 v104, vcc, v104, s73
v_add_co_u32_e64 v105, vcc, v105, s72
v_mov_b32_dpp v106, v106  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v104, v104  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v105, v105  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
s_mov_b32 s38, 0x80000000
s_mov_b32 s39, 0x20000
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x20000
s_mov_b32 s46, 0x80000000
s_mov_b32 s47, 0x20000
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 10
v_and_b32_e32 v108, 3, v0
v_cmp_eq_u32_e32 vcc, 1, v108
v_cndmask_b32_e32 v108, v108, v112, vcc
v_cmp_eq_u32_e32 vcc, 2, v108
v_cndmask_b32_e64 v108, v108, -v112, vcc
v_cmp_eq_u32_e32 vcc, 3, v108
v_cndmask_b32_e64 v108, v108, 0, vcc
s_branch 3
v_and_b32_e32 v108, 3, v0
v_cmp_ne_u32_e32 vcc, 1, v108
v_cndmask_b32_e32 v108, -1.0, v112, vcc
v_and_b32_e32 v116, 1, v0
v_bfe_u32 v117, v0, 1, 1
v_bfe_u32 v118, v0, 2, 1
v_bfe_u32 v119, v0, 3, 1
v_xor_b32_e32 v116, v116, v119
v_xor_b32_e32 v117, v117, v118
v_xor_b32_e32 v118, v118, v119
v_cmp_eq_u32_e32 vcc, 0, v117
v_cndmask_b32_e32 v109, -1.0, v112, vcc
v_cmp_eq_u32_e32 vcc, 1, v119
v_cndmask_b32_e32 v109, 0, v109, vcc
v_cmp_eq_u32_e32 vcc, 1, v116
v_cndmask_b32_e32 v110, -1.0, v112, vcc
v_cmp_eq_u32_e32 vcc, 1, v118
v_cndmask_b32_e32 v110, 0, v110, vcc
v_lshrrev_b32_e64 v116, 2, s88
v_and_b32_e32 v117, 3, v0
v_lshrrev_b32_e32 v118, 1, v0
v_bfi_b32 v118, 64, v118, v0
v_bfe_u32 v118, v118, 4, 3
v_mad_u32_u24 v102, v118, 4, v117
v_lshlrev_b32_e32 v102, 4, v102
v_mad_u32_u24 v103, v116, 4, v117
v_lshlrev_b32_e32 v103, 4, v103
v_bfe_u32 v116, v0, 2, 2
v_and_b32_e32 v117, 1, v116
v_mad_u32_u24 v119, v116, 16, v117
v_lshlrev_b32_e32 v119, 6, v119
v_xor_b32_e32 v103, v103, v119
v_mul_u32_u24_e32 v119, 0x400, v116
v_xor_b32_e32 v102, v102, v119
s_lshr_b32 s88, s88, 1
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 49
s_bfe_u32 s49, s18, 0x10014
v_lshrrev_b32_e32 v119, 1, v0
s_mul_i32 s48, 60, s49
s_sub_u32 s48, 63, s48
v_bfi_b32 v119, s48, v0, v119
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
s_mul_i32 s48, 4, s49
s_sub_u32 s48, 6, s48
v_bfe_u32 v119, v0, s48, 1
v_mul_u32_u24_e32 v119, 0x1040, v119
v_xor_b32_e32 v99, 0x314, v117
v_xor_b32_e32 v100, 0x31c, v117
v_xor_b32_e32 v101, 8, v117
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v98, v117, v101, vcc
v_cndmask_b32_e32 v101, v101, v117, vcc
v_mad_u32_u24 v98, 4, v98, v119
v_mad_u32_u24 v99, 4, v99, v119
v_mad_u32_u24 v100, 4, v100, v119
v_mad_u32_u24 v101, 4, v101, v119
s_branch 44
s_bfe_u32 s49, s18, 0x10014
v_lshrrev_b32_e32 v119, 1, v0
s_mul_i32 s48, 60, s49
s_sub_u32 s48, 63, s48
v_bfi_b32 v119, s48, v0, v119
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
s_mul_i32 s48, 4, s49
s_sub_u32 s48, 6, s48
v_bfe_u32 v119, v0, s48, 1
v_mul_u32_u24_e32 v119, 0x1040, v119
v_mad_u32_u24 v98, 4, v117, v119
v_xor_b32_e32 v99, 0x307, v117
v_mad_u32_u24 v99, 4, v99, v119
v_xor_b32_e32 v100, 0x30f, v117
v_mad_u32_u24 v100, 4, v100, v119
v_xor_b32_e32 v101, 8, v117
v_mad_u32_u24 v101, 4, v101, v119
v_subrev_co_u32_e32 v104, vcc, s74, v104
v_mov_b32_e32 v117, s73
v_cmp_lt_i32_e32 vcc, v104, v117
v_subb_co_u32_e32 v116, vcc, 0, v113, vcc
v_mad_i32_i24 v104, v116, s73, v104
v_mad_i32_i24 v106, v116, s78, v106
v_mad_i32_i24 v105, v116, s77, v105
v_mov_b32_e32 v117, s72
v_cmp_lt_i32_e32 vcc, v105, v117
v_subb_co_u32_e32 v116, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v116
v_mad_i32_i24 v105, v116, v117, v105
v_subrev_co_u32_e32 v105, vcc, s75, v105
v_cmp_lt_i32_e32 vcc, v105, v117
v_subb_co_u32_e32 v116, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v116
v_mad_i32_i24 v105, v116, s72, v105
v_subrev_co_u32_e32 v106, vcc, s76, v106
s_mov_b32 s60, 0
s_mov_b32 s61, s28
s_mov_b32 s62, 1
s_mov_b32 s82, 0
s_mov_b32 s83, s16
s_mov_b32 s81, s83
s_sub_u32 s89, -1, s88
s_sub_u32 s89, s89, 16
s_bitset1_b32 s18, 21
s_mov_b32 s47, 0
v_add_co_u32_e32 v116, vcc, 2, v0
v_bfe_u32 v116, v116, 2, 1
v_cmp_ne_u32_e64 vcc, v116, 1
s_mov_b64 s[56:57], vcc
s_mov_b32 s90, 16
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 917
s_branch 1796
s_nop 0
s_nop 0
s_bitcmp1_b32 s18, 17
s_cbranch_scc1 246
s_add_u32 s86, s86, s17
s_cmp_eq_u32 s86, 0
s_cbranch_scc1 243
s_mov_b32 s87, 0
s_bitcmp1_b32 s18, 16
s_cbranch_scc1 232
s_add_u32 s85, s16, 15
s_lshr_b32 s85, s85, 4
v_mov_b32_e32 v117, s86
v_mul_u32_u24_e32 v117, s85, v117
v_add_co_u32_e32 v117, vcc, s17, v117
v_sub_co_u32_e64 v117, vcc, v117, 1
v_cvt_f32_u32_e32 v119, s17
v_rcp_f32_e32 v119, v119
v_mul_f32_e32 v119, 0x4f800000, v119
v_cvt_u32_f32_e32 v119, v119
v_mul_lo_u32 v120, s17, v119
v_mul_hi_u32 v121, s17, v119
v_sub_co_u32_e32 v122, vcc, 0, v120
v_cmp_ne_i32_e64 s[50:51], 0, v121
v_cndmask_b32_e64 v120, v122, v120, s[50:51]
v_mul_hi_u32 v120, v120, v119
v_sub_co_u32_e32 v121, vcc, v119, v120
v_add_co_u32_e32 v119, vcc, v119, v120
v_cndmask_b32_e64 v119, v119, v121, s[50:51]
v_mul_hi_u32 v119, v119, v117
v_mul_lo_u32 v120, v119, s17
v_sub_co_u32_e32 v121, vcc, v117, v120
v_cmp_ge_u32_e64 s[50:51], v117, v120
v_cmp_ge_u32_e64 s[52:53], v121, s17
v_add_co_u32_e32 v121, vcc, 1, v119
s_and_b64 s[52:53], s[50:51], s[52:53]
v_add_co_u32_e32 v120, vcc, -1, v119
v_cndmask_b32_e64 v121, v119, v121, s[52:53]
v_cndmask_b32_e64 v121, v120, v121, s[50:51]
v_cmp_ne_i32_e64 vcc, 0, s17
v_cndmask_b32_e32 v116, -1, v121, vcc
v_readfirstlane_b32 s84, v116
v_mul_u32_u24_e64 v116, v116, s8
v_cvt_f32_u32_e32 v119, s85
v_rcp_f32_e32 v119, v119
v_mul_f32_e32 v119, 0x4f800000, v119
v_cvt_u32_f32_e32 v119, v119
v_mul_lo_u32 v120, s85, v119
v_mul_hi_u32 v121, s85, v119
v_sub_co_u32_e32 v122, vcc, 0, v120
v_cmp_ne_i32_e64 s[50:51], 0, v121
v_cndmask_b32_e64 v120, v122, v120, s[50:51]
v_mul_hi_u32 v120, v120, v119
v_sub_co_u32_e32 v121, vcc, v119, v120
v_add_co_u32_e32 v119, vcc, v119, v120
v_cndmask_b32_e64 v119, v119, v121, s[50:51]
v_mul_hi_u32 v119, v119, v116
v_mul_lo_u32 v120, v119, s85
v_sub_co_u32_e32 v121, vcc, v116, v120
v_cmp_ge_u32_e64 s[50:51], v116, v120
v_cmp_ge_u32_e64 s[52:53], v121, s85
v_add_co_u32_e32 v121, vcc, 1, v119
s_and_b64 s[52:53], s[50:51], s[52:53]
v_add_co_u32_e32 v120, vcc, -1, v119
v_cndmask_b32_e64 v121, v119, v121, s[52:53]
v_cndmask_b32_e64 v121, v120, v121, s[50:51]
v_cmp_ne_i32_e64 vcc, 0, s85
v_cndmask_b32_e32 v117, -1, v121, vcc
v_readfirstlane_b32 s48, v116
v_readfirstlane_b32 s82, v117
s_mul_i32 s82, s82, s85
s_sub_u32 s82, s48, s82
v_sub_co_u32_e32 v117, vcc, s8, v117
v_sub_co_u32_e32 v117, vcc, s17, v117
v_and_b32_e64 v119, v0, 63
v_cmp_eq_u32_e64 vcc, v119, 0
v_cndmask_b32_e32 v117, 1, v117, vcc
s_sub_u32 s54, 0, s73
s_sub_u32 s55, 0, s72
v_mul_u32_u24_e64 v121, v117, 32
v_cvt_f32_u32_e32 v122, s54
v_rcp_f32_e32 v122, v122
v_mul_f32_e32 v122, 0x4f800000, v122
v_cvt_u32_f32_e32 v122, v122
v_mul_lo_u32 v123, s54, v122
v_mul_hi_u32 v124, s54, v122
v_sub_co_u32_e32 v125, vcc, 0, v123
v_cmp_ne_i32_e64 s[50:51], 0, v124
v_cndmask_b32_e64 v123, v125, v123, s[50:51]
v_mul_hi_u32 v123, v123, v122
v_sub_co_u32_e32 v124, vcc, v122, v123
v_add_co_u32_e32 v122, vcc, v122, v123
v_cndmask_b32_e64 v122, v122, v124, s[50:51]
v_mul_hi_u32 v122, v122, v121
v_mul_lo_u32 v123, v122, s54
v_sub_co_u32_e32 v124, vcc, v121, v123
v_cmp_ge_u32_e64 s[50:51], v121, v123
v_cmp_ge_u32_e64 s[52:53], v124, s54
v_add_co_u32_e32 v124, vcc, 1, v122
s_and_b64 s[52:53], s[50:51], s[52:53]
v_add_co_u32_e32 v123, vcc, -1, v122
v_cndmask_b32_e64 v124, v122, v124, s[52:53]
v_cndmask_b32_e64 v124, v123, v124, s[50:51]
v_cmp_ne_i32_e64 vcc, 0, s54
v_cndmask_b32_e32 v119, -1, v124, vcc
v_mad_i32_i24 v120, v119, s73, v121
v_mul_u32_u24_e64 v121, v119, 1
v_cvt_f32_u32_e32 v122, s55
v_rcp_f32_e32 v122, v122
v_mul_f32_e32 v122, 0x4f800000, v122
v_cvt_u32_f32_e32 v122, v122
v_mul_lo_u32 v123, s55, v122
v_mul_hi_u32 v124, s55, v122
v_sub_co_u32_e32 v125, vcc, 0, v123
v_cmp_ne_i32_e64 s[50:51], 0, v124
v_cndmask_b32_e64 v123, v125, v123, s[50:51]
v_mul_hi_u32 v123, v123, v122
v_sub_co_u32_e32 v124, vcc, v122, v123
v_add_co_u32_e32 v122, vcc, v122, v123
v_cndmask_b32_e64 v122, v122, v124, s[50:51]
v_mul_hi_u32 v122, v122, v121
v_mul_lo_u32 v123, v122, s55
v_sub_co_u32_e32 v124, vcc, v121, v123
v_cmp_ge_u32_e64 s[50:51], v121, v123
v_cmp_ge_u32_e64 s[52:53], v124, s55
v_add_co_u32_e32 v124, vcc, 1, v122
s_and_b64 s[52:53], s[50:51], s[52:53]
v_add_co_u32_e32 v123, vcc, -1, v122
v_cndmask_b32_e64 v124, v122, v124, s[52:53]
v_cndmask_b32_e64 v124, v123, v124, s[50:51]
v_cmp_ne_i32_e64 vcc, 0, s55
v_cndmask_b32_e32 v119, -1, v124, vcc
v_mad_i32_i24 v121, v119, s72, v121
v_readfirstlane_b32 s74, v120
v_readfirstlane_b32 s75, v121
v_readfirstlane_b32 s76, v119
v_add_co_u32_e32 v104, vcc, s74, v104
v_addc_co_u32_e32 v122, vcc, 0, v113, vcc
v_mad_i32_i24 v104, v122, s73, v104
v_mad_i32_i24 v106, v122, s78, v106
v_mad_i32_i24 v105, v122, s77, v105
v_cmp_ge_i32_e64 vcc, v105, 0
v_addc_co_u32_e32 v122, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v122
v_mad_i32_i24 v105, v122, s72, v105
v_add_co_u32_e32 v105, vcc, s75, v105
v_addc_co_u32_e32 v122, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v122
v_mad_i32_i24 v105, v122, s72, v105
v_add_co_u32_e32 v106, vcc, s76, v106
v_readlane_b32 s74, v120, 1
v_readlane_b32 s75, v121, 1
v_readlane_b32 s76, v119, 1
s_add_u32 s83, s82, s84
s_cmp_le_u32 s83, s85
s_cselect_b32 s48, 0x20000, 0
s_cselect_b32 s83, s83, s85
s_or_b32 s18, s18, s48
s_lshl_b32 s82, s82, 4
s_lshl_b32 s83, s83, 4
s_min_u32 s83, s83, s16
s_cmp_eq_u32 s8, s17
s_cselect_b32 s48, 0x20000, 0
s_or_b32 s18, s18, s48
s_or_b32 s18, s18, s48
s_bitset1_b32 s18, 16
s_branch 40
s_lshr_b32 s82, s82, 4
s_add_u32 s83, s82, s84
s_sub_u32 s83, s83, s85
s_mov_b32 s82, 0
s_lshl_b32 s83, s83, 4
s_min_u32 s83, s83, s16
s_bitset1_b32 s18, 17
s_branch 12
s_bitset1_b32 s18, 18
s_mov_b32 s39, 0
s_mov_b32 s43, 0
s_mov_b32 s80, 16
s_branch 251
s_add_u32 s81, s81, 16
s_cmp_ge_u32 s81, s83
s_cbranch_scc0 25
s_bitset1_b32 s18, 22
s_sub_u32 s86, s86, s17
s_subb_u32 s87, s87, 0
s_cbranch_scc1 65276
v_add_co_u32_e32 v104, vcc, s74, v104
v_addc_co_u32_e32 v116, vcc, 0, v113, vcc
v_mad_i32_i24 v104, v116, s73, v104
v_mad_i32_i24 v106, v116, s78, v106
v_mad_i32_i24 v105, v116, s77, v105
v_cmp_ge_i32_e64 vcc, v105, 0
v_addc_co_u32_e32 v116, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v116
v_mad_i32_i24 v105, v116, s72, v105
v_add_co_u32_e32 v105, vcc, s75, v105
v_addc_co_u32_e32 v116, vcc, 0, v113, vcc
v_add_co_u32_e32 v106, vcc, v106, v116
v_mad_i32_i24 v105, v116, s72, v105
v_add_co_u32_e32 v106, vcc, s76, v106
s_mov_b32 s81, s82
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccz 131
v_readfirstlane_b32 s79, v106
v_cmp_ge_u32_e64 s[52:53], v106, s12
v_subrev_co_u32_e32 v117, vcc, s73, v104
v_subrev_co_u32_e32 v118, vcc, s72, v105
s_bfe_u32 s54, s18, 0x10014
v_lshrrev_b32_e32 v90, 2, v0
v_and_b32_e32 v119, s54, v90
s_bitcmp1_b32 s18, 22
s_cbranch_scc0 38
s_bitset0_b32 s18, 22
v_and_b32_e64 v92, v0, 1
v_mad_i32_i24 v91, v117, 2, v92
v_bfe_u32 v92, v0, 1, 1
v_mad_i32_i24 v92, v118, 2, v92
v_cvt_pk_u16_u32 v93, v91, v92
v_mul_u32_u24_e32 v91, s54, v90
v_mad_i32_i24 v91, -2, v91, v0
v_bfe_u32 v90, v90, s54, 1
v_mul_u32_u24_e32 v90, 3, v90
s_sub_u32 s54, 1, s54
v_lshrrev_b32_e32 v92, s54, v91
v_bfi_b32 v91, 64, v92, v91
v_and_b32_e32 v91, 0x7f, v91
v_xor_b32_e32 v91, v91, v90
v_lshlrev_b32_e32 v92, 14, v119
v_mad_u32_u24 v91, 4, v91, v92
v_add_co_u32_e32 v91, vcc, s95, v91
ds_write_b32 v91, v106
ds_write_b32 v91, v93 offset:512
s_add_u32 s95, s95, 0x800
s_cmp_eq_u32 s95, 0xffc0
s_cselect_b32 s95, 0xdfc0, s95
v_sub_co_u32_e64 v116, vcc, v106, s79
v_and_b32_e64 v91, v0, 3
v_ashrrev_i32_e64 v92, 1, s31
v_subrev_co_u32_e32 v91, vcc, v92, v91
v_ashrrev_i32_e64 v92, 1, s60
v_mad_i32_i24 v91, v92, 3, v91
v_mad_i32_i24 v91, v119, 3, v91
v_mad_i32_i24 v117, v117, 2, v91
v_cmp_ge_u32_e64 s[48:49], v117, s15
v_mul_lo_u32 v119, v116, s63
v_mad_i32_i24 v117, 4, v117, v119
s_or_b64 s[48:49], s[48:49], s[52:53]
v_add_co_u32_e64 v92, vcc, 1, s61
v_ashrrev_i32_e32 v92, 1, v92
v_mad_i32_i24 v118, v118, 2, v92
v_add_co_u32_e64 v92, vcc, 1, s30
v_ashrrev_i32_e32 v92, 1, v92
v_subrev_co_u32_e32 v118, vcc, v92, v118
s_lshl_b32 s55, s15, 2
v_cmp_ge_u32_e64 s[50:51], v118, s14
s_or_b64 s[50:51], s[48:49], s[50:51]
v_mad_u32_u24 v90, v118, s55, v117
v_cndmask_b32_e64 v90, v90, -1, s[50:51]
v_add_co_u32_e32 v118, vcc, 1, v118
v_cmp_ge_u32_e64 s[50:51], v118, s14
s_or_b64 s[50:51], s[48:49], s[50:51]
v_mad_u32_u24 v91, v118, s55, v117
v_cndmask_b32_e64 v91, v91, -1, s[50:51]
v_add_co_u32_e32 v118, vcc, 1, v118
v_cmp_ge_u32_e64 s[50:51], v118, s14
s_or_b64 s[50:51], s[48:49], s[50:51]
v_mad_u32_u24 v92, v118, s55, v117
v_cndmask_b32_e64 v92, v92, -1, s[50:51]
v_add_co_u32_e32 v118, vcc, 1, v118
v_cmp_ge_u32_e64 s[50:51], v118, s14
s_or_b64 s[50:51], s[48:49], s[50:51]
v_mad_u32_u24 v93, v118, s55, v117
v_cndmask_b32_e64 v93, v93, -1, s[50:51]
s_lshr_b32 s51, -1, 16
s_and_b32 s48, s51, s63
s_lshr_b32 s49, s63, 16
s_mul_i32 s50, s49, s79
s_mul_i32 s36, s48, s79
s_lshl_b32 s48, s50, 16
s_lshr_b32 s51, s50, 16
s_add_u32 s36, s48, s36
s_addc_u32 s37, s51, 0
s_add_u32 s36, s36, s20
s_addc_u32 s37, s37, s21
s_and_b32 s48, s18, 0x80000
s_cselect_b32 s48, s67, 0
s_add_u32 s36, s36, s48
s_addc_u32 s37, s37, 0
s_mov_b32 s39, 0x20000
s_branch 85
v_mad_u32_u24 v118, 5, v0, 2
v_lshlrev_b32_e32 v116, 1, v0
v_bfi_b32 v118, 4, v118, v116
v_bfe_u32 v116, v118, 2, 2
v_min_u32_e32 v116, 2, v116
v_bfe_u32 v118, v0, 1, 1
v_mad_u32_u24 v116, 2, v116, v118
v_mad_u32_u24 v116, s60, 3, v116
v_sub_co_u32_e32 v118, vcc, s29, v116
v_sub_co_u32_e64 v118, vcc, v118, 1
s_bfe_u32 s50, s18, 0x10001
v_cmp_eq_u32_e64 vcc, s50, 1
v_cndmask_b32_e32 v116, v116, v118, vcc
v_cmp_ge_u32_e64 s[48:49], v116, s29
v_lshlrev_b32_e32 v116, 2, v116
v_mad_u32_u24 v117, s69, v107, v116
s_sub_u32 s50, s28, s61
s_sub_u32 s50, s50, 5
s_bitcmp1_b32 s18, 0
s_cselect_b32 s50, s50, s61
v_mov_b32_e32 v119, s50
s_lshl_b32 s53, s29, 2
v_cmp_ge_u32_e64 s[50:51], v119, s28
v_mad_i32_i24 v90, v119, s53, v117
s_or_b64 s[50:51], s[50:51], s[48:49]
v_cndmask_b32_e64 v90, v90, -1, s[50:51]
v_mov_b32_e32 v91, v90
v_add_co_u32_e64 v119, vcc, v119, 2
v_cmp_ge_u32_e64 s[50:51], v119, s28
v_mad_i32_i24 v93, v119, s53, v117
s_or_b64 s[50:51], s[50:51], s[48:49]
v_cndmask_b32_e64 v93, v93, -1, s[50:51]
v_add_co_u32_e64 v119, vcc, v119, 2
v_cmp_ge_u32_e64 s[50:51], v119, s28
v_mad_i32_i24 v92, v119, s53, v117
s_or_b64 s[50:51], s[50:51], s[48:49]
v_cndmask_b32_e64 v92, v92, -1, s[50:51]
v_add_co_u32_e64 v116, vcc, v107, s81
v_cmp_lt_u32_e64 vcc, v116, s16
v_cndmask_b32_e32 v90, -1, v90, vcc
v_cndmask_b32_e32 v91, -1, v91, vcc
v_cndmask_b32_e32 v92, -1, v92, vcc
v_cndmask_b32_e32 v93, -1, v93, vcc
s_lshr_b32 s51, -1, 16
s_and_b32 s48, s51, s69
s_lshr_b32 s49, s69, 16
s_mul_i32 s50, s49, s81
s_mul_i32 s40, s48, s81
s_lshl_b32 s48, s50, 16
s_lshr_b32 s51, s50, 16
s_add_u32 s40, s48, s40
s_addc_u32 s41, s51, 0
s_add_u32 s40, s40, s22
s_addc_u32 s41, s41, s23
s_and_b32 s48, s18, 0x80000
s_cselect_b32 s48, s68, 0
s_add_u32 s40, s40, s48
s_addc_u32 s41, s41, 0
s_mov_b32 s43, 0x20000
s_bfe_u32 s48, s18, 0x10014
s_sub_u32 s80, s13, 1
s_lshl_b32 s80, s80, s48
s_mov_b64 vcc, s[56:57]
s_add_u32 s48, s34, 64
s_addc_u32 s49, s35, 0
s_nop 0
s_setpc_b64 s[48:49]
s_and_b32 s48, 0x900000, s18
s_subb_u32 s60, s60, 1
s_cbranch_scc0 65304
s_and_b32 s48, 0x900000, s18
s_subb_u32 s60, s59, 1
s_add_u32 s61, s61, 6
s_cmp_ge_u32 s61, s28
s_cbranch_scc0 65298
s_mov_b32 s61, 1
s_cmp_ge_u32 s61, s28
s_addc_u32 s62, s62, 1
s_cmp_gt_u32 s62, 1
s_cbranch_scc0 65293
s_mov_b32 s62, 0
s_mov_b32 s61, 0
s_branch 65262
s_nop 0
v_mac_f32_dpp v2, v2, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v5, v5, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v3, v3, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v4, v4, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v2, v5, v2  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v3, v4, v3  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v2, v2, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v3, v3, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v34, v3, v2  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v6, v6, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v9, v9, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v7, v7, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v8, v8, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v6, v9, v6  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v7, v8, v7  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v6, v6, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v7, v7, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v35, v7, v6  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v10, v10, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v13, v13, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v11, v11, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v12, v12, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v10, v13, v10  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v11, v12, v11  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v10, v10, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v11, v11, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v36, v11, v10  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v14, v14, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v17, v17, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v15, v15, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v16, v16, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v14, v17, v14  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v15, v16, v15  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v14, v14, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v15, v15, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v37, v15, v14  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v18, v18, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v21, v21, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v19, v19, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v20, v20, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v18, v21, v18  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v19, v20, v19  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v18, v18, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v19, v19, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v38, v19, v18  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v22, v22, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v25, v25, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v23, v23, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v24, v24, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v22, v25, v22  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v23, v24, v23  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v22, v22, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v23, v23, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v39, v23, v22  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v26, v26, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v29, v29, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v27, v27, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v28, v28, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v26, v29, v26  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v27, v28, v27  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v26, v26, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v27, v27, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v40, v27, v26  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v30, v30, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v33, v33, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v31, v31, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v32, v32, v109  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v30, v33, v30  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v31, v32, v31  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v30, v30, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v31, v31, v110  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v41, v31, v30  row_half_mirror row_mask:0xf bank_mask:0xf
s_mov_b64 s[50:51], s[44:45]
s_mov_b32 s49, s47
v_bfe_u32 v116, s18, 21, 1
v_sub_co_u32_e64 v116, vcc, v116, 1
v_cndmask_b32_e32 v116, v96, v94, vcc
v_cndmask_b32_e32 v117, v97, v95, vcc
s_nop 4
buffer_store_dword v34, v116, s[44:47], 0 offen
buffer_store_dword v35, v117, s[44:47], 0 offen
s_add_u32 s44, s44, s66
s_addc_u32 s45, s45, 0
s_sub_u32 s89, s89, 1
s_cselect_b32 s47, 0, s47
s_nop 4
buffer_store_dword v36, v116, s[44:47], 0 offen
buffer_store_dword v37, v117, s[44:47], 0 offen
s_add_u32 s44, s44, s66
s_addc_u32 s45, s45, 0
s_sub_u32 s89, s89, 1
s_cselect_b32 s47, 0, s47
s_lshl_b32 s48, s66, 1
s_add_u32 s44, s44, s48
s_addc_u32 s45, s45, 0
s_sub_u32 s89, s89, 2
s_cselect_b32 s47, 0, s47
s_nop 4
buffer_store_dword v38, v116, s[44:47], 0 offen
buffer_store_dword v39, v117, s[44:47], 0 offen
s_add_u32 s44, s44, s66
s_addc_u32 s45, s45, 0
s_sub_u32 s89, s89, 1
s_cselect_b32 s47, 0, s47
s_nop 4
buffer_store_dword v40, v116, s[44:47], 0 offen
buffer_store_dword v41, v117, s[44:47], 0 offen
s_add_u32 s44, s44, s66
s_addc_u32 s45, s45, 0
s_sub_u32 s89, s89, 1
s_cselect_b32 s47, 0, s47
s_add_u32 s44, s44, s48
s_addc_u32 s45, s45, 0
s_lshl_b32 s48, s48, 2
s_add_u32 s44, s44, s48
s_addc_u32 s45, s45, 0
s_sub_u32 s89, s89, 10
s_cselect_b32 s47, 0, s47
s_bitcmp1_b32 s18, 21
s_cselect_b32 s47, s47, s49
s_cselect_b32 s44, s44, s50
s_cselect_b32 s45, s45, s51
s_cselect_b32 s49, 0, 16
s_add_u32 s89, s89, s49
s_xor_b32 s18, s18, 0x200000
s_bitcmp0_b32 s18, 21
s_addc_u32 s90, s58, 0
s_lshr_b32 s90, s90, 1
s_mul_i32 s90, s90, s59
s_lshr_b32 s90, s90, 1
s_mul_i32 s90, s90, s13
s_sub_u32 s90, s90, 1
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
s_add_u32 s48, s89, s88
s_cmp_lt_i32 s48, 0
s_cbranch_scc0 92
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 2012
v_lshrrev_b32_e32 v118, 1, v0
v_bfi_b32 v118, 64, v118, v0
v_and_b32_e32 v94, 0x7f, v118
v_lshlrev_b32_e32 v94, 2, v94
v_add_co_u32_e64 v94, vcc, v94, s96
ds_read_b32 v95, v94 offset:512
ds_read_b32 v94, v94
s_add_u32 s96, s96, 0x800
s_cmp_eq_u32 s96, 0xffc0
s_cselect_b32 s96, 0xdfc0, s96
s_waitcnt vmcnt(15) lgkmcnt(0)
v_bfe_u32 v116, v95, 16, 16
v_bfe_u32 v117, v95, 0, 16
v_readfirstlane_b32 s91, v94
v_sub_co_u32_e64 v118, vcc, v94, s91
v_mul_lo_u32 v118, v118, s64
v_cmp_ge_u32_e64 s[48:49], v94, s12
v_lshlrev_b32_e32 v117, 1, v117
s_and_b32 s50, 1, s31
v_add_co_u32_e32 v117, vcc, s50, v117
v_lshlrev_b32_e32 v116, 1, v116
s_and_b32 s50, 1, s30
v_subrev_co_u32_e32 v116, vcc, s50, v116
v_mad_i32_i24 v94, v116, s33, v117
v_mad_i32_i24 v94, 4, v94, v118
v_subrev_co_u32_e32 v95, vcc, 4, v94
v_mad_i32_i24 v96, 4, s33, v94
v_mad_i32_i24 v97, 4, s33, v95
v_cmp_ge_u32_e64 s[54:55], v117, s33
s_or_b64 s[52:53], s[54:55], s[48:49]
v_subrev_co_u32_e32 v117, vcc, 1, v117
v_cmp_ge_u32_e64 s[54:55], v117, s33
s_or_b64 s[54:55], s[54:55], s[48:49]
v_cmp_ge_u32_e64 s[50:51], v116, s32
s_or_b64 s[48:49], s[52:53], s[50:51]
v_cndmask_b32_e64 v94, v94, -1, s[48:49]
s_or_b64 s[48:49], s[54:55], s[50:51]
v_cndmask_b32_e64 v95, v95, -1, s[48:49]
v_add_co_u32_e32 v116, vcc, 1, v116
v_cmp_ge_u32_e64 s[50:51], v116, s32
s_or_b64 s[48:49], s[52:53], s[50:51]
v_cndmask_b32_e64 v96, v96, -1, s[48:49]
s_or_b64 s[48:49], s[54:55], s[50:51]
v_cndmask_b32_e64 v97, v97, -1, s[48:49]
s_add_u32 s89, s88, s82
s_lshr_b32 s51, -1, 16
s_and_b32 s48, s51, s64
s_lshr_b32 s49, s64, 16
s_mul_i32 s50, s49, s91
s_mul_i32 s44, s48, s91
s_lshl_b32 s48, s50, 16
s_lshr_b32 s51, s50, 16
s_add_u32 s44, s48, s44
s_addc_u32 s45, s51, 0
s_add_u32 s44, s44, s24
s_addc_u32 s45, s45, s25
s_mul_i32 s48, s66, s89
s_add_u32 s44, s44, s48
s_addc_u32 s45, s45, 0
s_mov_b32 s47, 0x20000
s_sub_u32 s89, s83, s82
s_sub_u32 s89, s89, 1
s_sub_u32 s89, s89, s88
s_cselect_b32 s47, 0, s47
s_barrier
s_mov_b64 vcc, s[56:57]
s_add_u32 s48, s34, 0x60
s_addc_u32 s49, s35, 0
s_setpc_b64 s[48:49]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_getpc_b64 s[34:35]
s_sub_u32 s34, s34, 48
s_nop 0
s_subb_u32 s35, s35, 0
s_branch 65126
s_nop 0
ds_read_b128 v[42:45], v102 offset:25280
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v66, v90, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v98, v70
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_subrev_f32_e64 v78, v80, v78 div:2
v_subrev_f32_e64 v81, v79, v81 div:2
ds_read_b128 v[50:53], v103 offset:24768
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v79, v80, v79 div:2
s_nop 0
buffer_load_dword v68, v92, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_write_b32 v99, v71
v_mad_f32 v80, v80, 1.0, -v79
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:24896
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 65072
ds_read_b128 v[46:49], v102 offset:29440
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v67, v91, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v72
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v78, v78, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v79, v79, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v103 offset:28928
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v80, v80, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v69, v93, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_write_b32 v101, v73
v_mac_f32_dpp v81, v81, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:29056
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s36, s36, s70
s_addc_u32 s37, s37, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 64989
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v102 offset:33536
s_setprio 0
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v70, v90, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v98, v74 offset:8256
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_subrev_f32_e64 v82, v84, v82 div:2
v_subrev_f32_e64 v85, v83, v85 div:2
ds_read_b128 v[50:53], v103 offset:33024
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v83, v84, v83 div:2
s_nop 0
buffer_load_dword v72, v92, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_write_b32 v99, v75 offset:8256
v_mad_f32 v84, v84, 1.0, -v83
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:33152
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
ds_write_b8 v111, v111 offset:65504
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 64928
ds_read_b128 v[46:49], v102 offset:37696
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v71, v91, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v76 offset:8256
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v82, v82, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v83, v83, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v103 offset:37184
s_add_u32 m0, s92, 32
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v84, v84, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v73, v93, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(3)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
ds_write_b32 v101, v77 offset:8256
v_mac_f32_e64 v18, v42, v54
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:37312
s_cbranch_vccz 1550
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s36, s36, s70
s_addc_u32 s37, s37, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 64845
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v102 offset:41792
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v74, v90, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v98, v78 offset:16512
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_mac_f32_dpp v85, v85, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v86, v88, v86 div:2
ds_read_b128 v[50:53], v103 offset:41280
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_subrev_f32_e64 v89, v87, v89 div:2
s_nop 0
buffer_load_dword v76, v92, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_write_b32 v99, v79 offset:16512
v_add_f32_e64 v87, v88, v87 div:2
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:41408
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 64784
ds_read_b128 v[46:49], v102 offset:45952
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v75, v91, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v80 offset:16512
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mad_f32 v88, v88, 1.0, -v87
v_mac_f32_dpp v86, v86, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v103 offset:45440
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v87, v87, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v77, v93, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_write_b32 v101, v81 offset:16512
v_mac_f32_dpp v88, v88, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:45568
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s36, s36, s70
s_addc_u32 s37, s37, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 64701
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v102 offset:512
s_setprio 0
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v78, v90, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v98, v82 offset:24768
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_mac_f32_dpp v89, v89, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v66, v68, v66 div:2
ds_read_b128 v[50:53], v103
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_subrev_f32_e64 v69, v67, v69 div:2
s_nop 0
buffer_load_dword v80, v92, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_write_b32 v99, v83 offset:24768
v_add_f32_e64 v67, v68, v67 div:2
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:128
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
ds_write_b8 v111, v111 offset:65488
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 64640
ds_read_b128 v[46:49], v102 offset:4672
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v79, v91, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v84 offset:24768
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mad_f32 v68, v68, 1.0, -v67
v_mac_f32_dpp v66, v66, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v103 offset:4160
s_add_u32 m0, s92, 16
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v67, v67, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v81, v93, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(3)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
ds_write_b32 v101, v85 offset:24768
v_mac_f32_e64 v18, v42, v54
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:4288
s_cbranch_vccz 1262
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s36, s36, s70
s_addc_u32 s37, s37, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 64557
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v102 offset:8768
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v82, v90, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v98, v86 offset:33024
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_mac_f32_dpp v68, v68, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v69, v69, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v103 offset:8256
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_subrev_f32_e64 v70, v72, v70 div:2
s_nop 0
buffer_load_dword v84, v92, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_write_b32 v99, v87 offset:33024
v_subrev_f32_e64 v73, v71, v73 div:2
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:8384
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 64496
ds_read_b128 v[46:49], v102 offset:12928
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v83, v91, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v88 offset:33024
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_e64 v71, v72, v71 div:2
v_mad_f32 v72, v72, 1.0, -v71
ds_read_b128 v[58:61], v103 offset:12416
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v70, v70, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v85, v93, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_write_b32 v101, v89 offset:33024
v_mac_f32_dpp v71, v71, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:12544
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s36, s36, s70
s_addc_u32 s37, s37, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 64413
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v102 offset:17024
s_setprio 0
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v86, v90, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v98, v66 offset:41280
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_mac_f32_dpp v72, v72, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v73, v73, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v103 offset:16512
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_subrev_f32_e64 v74, v76, v74 div:2
s_nop 0
buffer_load_dword v88, v92, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_write_b32 v99, v67 offset:41280
v_subrev_f32_e64 v77, v75, v77 div:2
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:16640
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
ds_write_b8 v111, v111 offset:65472
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 64352
ds_read_b128 v[46:49], v102 offset:21184
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v87, v91, s[36:39], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v68 offset:41280
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_e64 v75, v76, v75 div:2
v_mad_f32 v76, v76, 1.0, -v75
ds_read_b128 v[58:61], v103 offset:20672
s_add_u32 m0, s92, 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v74, v74, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v89, v93, s[36:39], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(3)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
ds_write_b32 v101, v69 offset:41280
v_mac_f32_e64 v18, v42, v54
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:20800
s_cbranch_vccz 974
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s36, s36, s70
s_addc_u32 s37, s37, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 64269
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
v_mac_f32_dpp v75, v75, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v76, v76, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v77, v77, -v108  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_branch 64666
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_getpc_b64 s[34:35]
s_sub_u32 s34, s34, 48
s_nop 0
s_subb_u32 s35, s35, 0
s_branch 64246
s_nop 0
ds_read_b128 v[42:45], v102 offset:25280
s_setprio 0
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v66, v90, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v98, v70
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_cndmask_b32_dpp v78, v78, v78, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v79, v79, v79, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v103 offset:24768
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_cndmask_b32_dpp v80, v80, v80, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v68, v92, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_write_b32 v99, v71
v_cndmask_b32_dpp v81, v81, v81, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:24896
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 64192
ds_read_b128 v[46:49], v102 offset:29440
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v67, v91, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v72
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_dpp v78, v79, v79  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v78, v79, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v103 offset:28928
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_add_f32_dpp v79, v81, v81  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v69, v93, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
v_mac_f32_dpp v79, v81, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v81, v80, v80  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v81, v80, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v80, v78, v81
ds_write_b32 v101, v73
v_add_f32_e64 v79, v79, v80 div:2
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:29056
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s40, s40, s71
s_addc_u32 s41, s41, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 64101
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v102 offset:33536
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v70, v90, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v98, v74 offset:8256
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_e64 v80, -v79, v80
v_cndmask_b32_dpp v82, v82, v82, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v103 offset:33024
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_cndmask_b32_dpp v83, v83, v83, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v72, v92, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
v_cndmask_b32_dpp v84, v84, v84, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v85, v85, v85, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v82, v83, v83  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v82, v83, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_write_b32 v99, v75 offset:8256
v_add_f32_dpp v83, v85, v85  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:33152
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
ds_write_b8 v111, v111 offset:65504
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 64032
ds_read_b128 v[46:49], v102 offset:37696
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v71, v91, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v76 offset:8256
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v83, v85, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v85, v84, v84  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v103 offset:37184
s_add_u32 m0, s92, 32
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v85, v84, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v73, v93, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(3)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
ds_write_b32 v101, v77 offset:8256
v_mac_f32_e64 v18, v42, v54
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:37312
s_cbranch_vccz 654
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s40, s40, s71
s_addc_u32 s41, s41, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 63949
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v102 offset:41792
s_setprio 0
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v74, v90, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v98, v78 offset:16512
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_e64 v84, v82, v85
v_add_f32_e64 v83, v83, v84 div:2
ds_read_b128 v[50:53], v103 offset:41280
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v84, -v83, v84
s_nop 0
buffer_load_dword v76, v92, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_write_b32 v99, v79 offset:16512
v_cndmask_b32_dpp v86, v86, v86, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:41408
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 63888
ds_read_b128 v[46:49], v102 offset:45952
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v75, v91, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v80 offset:16512
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_cndmask_b32_dpp v87, v87, v87, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v88, v88, v88, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v103 offset:45440
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_cndmask_b32_dpp v89, v89, v89, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v77, v93, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
v_add_f32_dpp v86, v87, v87  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v86, v87, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v87, v89, v89  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v87, v89, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_write_b32 v101, v81 offset:16512
v_add_f32_dpp v89, v88, v88  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:45568
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s40, s40, s71
s_addc_u32 s41, s41, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 63797
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v102 offset:512
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v78, v90, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
v_mac_f32_dpp v89, v88, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v88, v86, v89
v_add_f32_e64 v87, v87, v88 div:2
v_add_f32_e64 v88, -v87, v88
ds_write_b32 v98, v82 offset:24768
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_cndmask_b32_dpp v66, v66, v66, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v67, v67, v67, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v103
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_cndmask_b32_dpp v68, v68, v68, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v80, v92, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
v_cndmask_b32_dpp v69, v69, v69, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v66, v67, v67  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v66, v67, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v67, v69, v69  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
ds_write_b32 v99, v83 offset:24768
v_mac_f32_dpp v67, v69, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:128
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
ds_write_b8 v111, v111 offset:65488
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 63720
ds_read_b128 v[46:49], v102 offset:4672
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v79, v91, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v84 offset:24768
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_dpp v69, v68, v68  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v69, v68, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v103 offset:4160
s_add_u32 m0, s92, 16
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_add_f32_e64 v68, v66, v69
s_nop 0
buffer_load_dword v81, v93, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(3)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
ds_write_b32 v101, v85 offset:24768
v_mac_f32_e64 v18, v42, v54
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:4288
s_cbranch_vccz 342
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s40, s40, s71
s_addc_u32 s41, s41, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 63637
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v102 offset:8768
s_setprio 0
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v82, v90, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v98, v86 offset:33024
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_e64 v67, v67, v68 div:2
v_add_f32_e64 v68, -v67, v68
ds_read_b128 v[50:53], v103 offset:8256
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_cndmask_b32_dpp v70, v70, v70, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v84, v92, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_write_b32 v99, v87 offset:33024
v_cndmask_b32_dpp v71, v71, v71, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:8384
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
s_nop 0
s_nop 0
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 63576
ds_read_b128 v[46:49], v102 offset:12928
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v83, v91, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v88 offset:33024
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_cndmask_b32_dpp v72, v72, v72, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v73, v73, v73, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v103 offset:12416
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_add_f32_dpp v70, v71, v71  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v85, v93, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
v_mac_f32_dpp v70, v71, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v71, v73, v73  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v71, v73, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v73, v72, v72  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
ds_write_b32 v101, v89 offset:33024
v_mac_f32_dpp v73, v72, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:12544
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s40, s40, s71
s_addc_u32 s41, s41, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 63485
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v102 offset:17024
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v86, v90, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
v_add_f32_e64 v72, v70, v73
v_add_f32_e64 v71, v71, v72 div:2
v_add_f32_e64 v72, -v71, v72
v_cndmask_b32_dpp v74, v74, v74, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
ds_write_b32 v98, v66 offset:41280
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_cndmask_b32_dpp v75, v75, v75, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v76, v76, v76, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v103 offset:16512
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_cndmask_b32_dpp v77, v77, v77, vcc  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 0
buffer_load_dword v88, v92, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
v_add_f32_dpp v74, v75, v75  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v74, v75, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v75, v77, v77  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v75, v77, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_write_b32 v99, v67 offset:41280
v_add_f32_dpp v77, v76, v76  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_e64 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_getpc_b64 s[34:35]
ds_read_b128 v[54:57], v103 offset:16640
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_mov_b64 vcc, s[56:57]
s_mov_b32 m0, -1
s_add_u32 s90, s90, -2
s_waitcnt vmcnt(15) lgkmcnt(2)
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_nop 1
ds_write_b8 v111, v111 offset:65472
v_mac_f32_e64 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc0 63408
ds_read_b128 v[46:49], v102 offset:21184
s_nop 0
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v87, v91, s[40:43], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v100, v68 offset:41280
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v77, v76, v108  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v76, v74, v77
ds_read_b128 v[58:61], v103 offset:20672
s_add_u32 m0, s92, 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_add_f32_e64 v75, v75, v76 div:2
s_nop 0
buffer_load_dword v89, v93, s[40:43], 0 offen
s_waitcnt vmcnt(15) lgkmcnt(3)
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
ds_write_b32 v101, v69 offset:41280
v_mac_f32_e64 v18, v42, v54
v_mac_f32_e64 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_getpc_b64 s[34:35]
ds_read_b128 v[62:65], v103 offset:20800
s_cbranch_vccz 30
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_mov_b64 vcc, s[56:57]
s_add_u32 s40, s40, s71
s_addc_u32 s41, s41, 0
s_add_u32 s80, s80, -2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_cbranch_scc0 63325
s_nop 1
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e64 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_e64 v76, -v75, v76
s_branch 64602
s_nop 2
s_nop 0
s_nop 0
s_add_u32 s48, s34, 12
s_addc_u32 s49, s35, 0
s_nop 1
s_cbranch_vccnz 67
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 60
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 53
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 46
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 39
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 32
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 25
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 18
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 11
v_cmp_eq_u32_e32 vcc, src_lds_direct, v114
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_branch 65470
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 1
s_nop 1
s_setpc_b64 s[48:49]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_endpgm
