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

v_mov_b32_e32 v113, 1.0
v_mov_b32_e32 v114, 0
s_mov_b32 m0, 0x1ffff
s_nop 0
s_nop 0
v_readfirstlane_b32 s44, v0
s_cmp_eq_u32 s44, 0
s_cmovk_i32 s45, 0xfeef
s_cmovk_i32 s86, 0x10
s_cmp_eq_u32 s44, 64
s_cmovk_i32 s45, 0xeffe
s_cmovk_i32 s86, 0x14
s_cmp_eq_u32 s44, 0x80
s_cmovk_i32 s45, 0xfddf
s_cmovk_i32 s86, 0x10
s_cmp_eq_u32 s44, 0xc0
s_cmovk_i32 s45, 0xdffd
s_cmovk_i32 s86, 0x14
s_cmp_eq_u32 s44, 0x100
s_cmovk_i32 s45, 0xfbfb
s_cmovk_i32 s86, 0x18
s_cmp_eq_u32 s44, 0x140
s_cmovk_i32 s45, 0xbfbf
s_cmovk_i32 s86, 0x1c
s_cmp_eq_u32 s44, 0x180
s_cmovk_i32 s45, 0xf7f7
s_cmovk_i32 s86, 0x18
s_cmp_eq_u32 s44, 0x1c0
s_cmovk_i32 s45, 0x7f7f
s_cmovk_i32 s86, 0x1c
s_mov_b32 s46, 0x1010101
s_mul_i32 s47, s86, s46
s_mov_b32 s46, 0x3020100
s_add_i32 s47, s47, s46
v_mov_b32_e32 v115, s47
v_and_b32_e64 v112, v0, 63
v_and_b32_e64 v117, v0, 15
v_bfe_u32 v117, s45, v117, 1
v_bfe_u32 v118, v0, 4, 2
v_cmp_eq_u32_e64 vcc, v118, 2
v_addc_co_u32_e64 v118, vcc, 0, 0, vcc
v_or_b32_e32 v117, v117, v118
v_mad_u32_u24 v112, 64, v117, v112
s_or_b32 s86, s86, 0x20000
s_add_u32 s86, 0xffc0, s86
ds_write_b8 v112, v112 offset:65472
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
s_barrier
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_mov_b32_e32 v116, 0
s_load_dwordx16 s[12:27], s[6:7], 0x0
s_waitcnt vmcnt(15) lgkmcnt(0)
s_mov_b32 s28, 0
s_cmp_gt_u32 s12, 0xffff
s_addc_u32 s28, s28, 0
s_cmp_gt_u32 s13, 0xffff
s_addc_u32 s28, s28, 0
s_cmp_gt_u32 s14, 0xffff
s_addc_u32 s28, s28, 0
s_cmp_gt_u32 s15, 0xffff
s_addc_u32 s28, s28, 0
s_cmp_gt_u32 s16, 0xffff
s_addc_u32 s28, s28, 0
s_cmp_gt_u32 s17, 0xffff
s_addc_u32 s28, s28, 0
s_cmp_eq_u32 s12, 0
s_addc_u32 s28, s28, 0
s_cmp_eq_u32 s13, 0
s_addc_u32 s28, s28, 0
s_cmp_eq_u32 s15, 0
s_addc_u32 s28, s28, 0
s_cmp_eq_u32 s14, 0
s_addc_u32 s28, s28, 0
s_cmp_eq_u32 s16, 0
s_addc_u32 s28, s28, 0
s_cmp_eq_u32 s28, 0
s_cbranch_scc0 2959
s_and_b32 s23, s23, 0xffff
s_and_b32 s25, s25, 0xffff
s_and_b32 s21, s21, 0xffff
s_and_b32 s27, s27, 0xffff
s_mul_i32 s30, s14, s15
s_lshr_b32 s36, -1, 16
s_and_b32 s33, s36, s30
s_lshr_b32 s34, s30, 16
s_mul_i32 s35, s34, s13
s_mul_i32 s31, s33, s13
s_lshl_b32 s33, s35, 16
s_lshr_b32 s36, s35, 16
s_add_u32 s31, s33, s31
s_addc_u32 s32, s36, 0
s_cmp_gt_u32 s31, 0x10000000
s_addc_u32 s28, s32, 0
s_lshl_b32 s56, s31, 2
s_mul_i32 s30, s14, s15
s_lshr_b32 s36, -1, 16
s_and_b32 s33, s36, s30
s_lshr_b32 s34, s30, 16
s_mul_i32 s35, s34, s16
s_mul_i32 s31, s33, s16
s_lshl_b32 s33, s35, 16
s_lshr_b32 s36, s35, 16
s_add_u32 s31, s33, s31
s_addc_u32 s32, s36, 0
s_cmp_gt_u32 s31, 0x10000000
s_addc_u32 s28, s32, s28
s_lshl_b32 s57, s31, 2
s_sub_u32 s58, s57, s56
s_cmp_eq_u32 s28, 0
s_cbranch_scc0 2920
s_lshl_b32 s60, s30, 2
s_lshl_b32 s63, s60, 1
s_mul_i32 s64, s60, 12
s_and_b32 s18, s18, 7
s_bitcmp1_b32 s18, 2
s_mov_b32 s30, 36
s_cselect_b32 s31, s16, s13
s_mul_i32 s31, s31, s30
s_cselect_b32 s61, s31, s30
s_cselect_b32 s62, s30, s31
s_lshl_b32 s65, s61, 1
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_readfirstlane_b32 s66, v0
s_add_u32 s30, s66, 0x100
s_lshr_b32 s30, s30, 5
s_and_b32 s83, s66, 64
s_subb_u32 s83, 0, 0
s_xnor_b32 s83, s83, s30
s_and_b32 s83, s83, 20
s_bfe_u32 s66, s66, 0x10007
s_add_u32 s30, s14, 1
s_lshr_b32 s30, s30, 1
s_add_u32 s31, s15, 1
s_lshr_b32 s31, s31, 1
s_sub_u32 s68, 0, s31
s_sub_u32 s67, 0, s30
s_mul_i32 s80, s12, s30
s_sub_u32 s80, s80, 1
s_lshr_b32 s80, s80, 0
s_add_u32 s80, s80, 1
s_lshr_b32 s35, -1, 16
s_and_b32 s32, s35, s80
s_lshr_b32 s33, s80, 16
s_mul_i32 s34, s33, s31
s_mul_i32 s80, s32, s31
s_lshl_b32 s32, s34, 16
s_lshr_b32 s35, s34, 16
s_add_u32 s80, s32, s80
s_addc_u32 s81, s35, 0
s_sub_u32 s80, s80, 1
s_subb_u32 s81, s81, 0
s_lshr_b64 s[80:81], s[80:81], 5
s_add_u32 s80, s80, 1
s_addc_u32 s81, s81, 0
v_mov_b32_e32 v3, s8
v_mov_b32_e32 v4, s17
v_bfe_u32 v2, v0, 2, 5
v_and_b32_e32 v5, 3, v0
v_cmp_eq_u32_e32 vcc, 2, v5
v_cndmask_b32_e32 v3, v3, v4, vcc
v_cmp_eq_u32_e32 vcc, 1, v5
v_cndmask_b32_e32 v2, 0, v2, vcc
v_cmp_eq_u32_e64 s[32:33], 3, v5
v_bfe_u32 v107, v2, 0, 5
v_mad_u32_u24 v107, v3, 32, v107
v_cvt_f32_u32_e32 v6, s31
v_rcp_f32_e32 v6, v6
v_mul_f32_e32 v6, 0x4f800000, v6
v_cvt_u32_f32_e32 v6, v6
v_mul_lo_u32 v7, s31, v6
v_mul_hi_u32 v8, s31, v6
v_sub_co_u32_e32 v9, vcc, 0, v7
v_cmp_ne_i32_e64 s[34:35], 0, v8
v_cndmask_b32_e64 v7, v9, v7, s[34:35]
v_mul_hi_u32 v7, v7, v6
v_sub_co_u32_e32 v8, vcc, v6, v7
v_add_co_u32_e32 v6, vcc, v6, v7
v_cndmask_b32_e64 v6, v6, v8, s[34:35]
v_mul_hi_u32 v6, v6, v107
v_mul_lo_u32 v7, v6, s31
v_sub_co_u32_e32 v8, vcc, v107, v7
v_cmp_ge_u32_e64 s[34:35], v107, v7
v_cmp_ge_u32_e64 s[36:37], v8, s31
v_add_co_u32_e32 v8, vcc, 1, v6
s_and_b64 s[36:37], s[34:35], s[36:37]
v_add_co_u32_e32 v7, vcc, -1, v6
v_cndmask_b32_e64 v8, v6, v8, s[36:37]
v_cndmask_b32_e64 v8, v7, v8, s[34:35]
v_cmp_ne_i32_e64 vcc, 0, s31
v_cndmask_b32_e32 v108, -1, v8, vcc
v_mad_i32_i24 v106, v108, s68, v107
v_lshrrev_b32_e32 v107, 5, v2
v_mad_u32_u24 v107, v108, 1, v107
v_cndmask_b32_e64 v107, v107, 1, s[32:33]
v_cvt_f32_u32_e32 v6, s30
v_rcp_f32_e32 v6, v6
v_mul_f32_e32 v6, 0x4f800000, v6
v_cvt_u32_f32_e32 v6, v6
v_mul_lo_u32 v7, s30, v6
v_mul_hi_u32 v8, s30, v6
v_sub_co_u32_e32 v9, vcc, 0, v7
v_cmp_ne_i32_e64 s[34:35], 0, v8
v_cndmask_b32_e64 v7, v9, v7, s[34:35]
v_mul_hi_u32 v7, v7, v6
v_sub_co_u32_e32 v8, vcc, v6, v7
v_add_co_u32_e32 v6, vcc, v6, v7
v_cndmask_b32_e64 v6, v6, v8, s[34:35]
v_mul_hi_u32 v6, v6, v107
v_mul_lo_u32 v7, v6, s30
v_sub_co_u32_e32 v8, vcc, v107, v7
v_cmp_ge_u32_e64 s[34:35], v107, v7
v_cmp_ge_u32_e64 s[36:37], v8, s30
v_add_co_u32_e32 v8, vcc, 1, v6
s_and_b64 s[36:37], s[34:35], s[36:37]
v_add_co_u32_e32 v7, vcc, -1, v6
v_cndmask_b32_e64 v8, v6, v8, s[36:37]
v_cndmask_b32_e64 v8, v7, v8, s[34:35]
v_cmp_ne_i32_e64 vcc, 0, s30
v_cndmask_b32_e32 v108, -1, v8, vcc
v_mad_i32_i24 v107, v108, s67, v107
v_readlane_b32 s69, v106, 2
v_readlane_b32 s70, v107, 2
v_readlane_b32 s71, v108, 2
v_readlane_b32 s72, v107, 3
v_readlane_b32 s73, v108, 3
v_add_co_u32_e64 v106, vcc, v106, s68
v_add_co_u32_e64 v107, vcc, v107, s67
v_mov_b32_dpp v108, v108  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v106, v106  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v107, v107  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
s_mov_b32 s30, 0x80000000
s_mov_b32 s31, 0x20000
s_mov_b32 s34, 0x80000000
s_mov_b32 s35, 0x20000
s_mov_b32 s38, 0x80000000
s_mov_b32 s39, 0x20000
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x20000
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 10
v_and_b32_e32 v109, 3, v0
v_cmp_eq_u32_e32 vcc, 1, v109
v_cndmask_b32_e32 v109, v109, v113, vcc
v_cmp_eq_u32_e32 vcc, 2, v109
v_cndmask_b32_e64 v109, v109, -v113, vcc
v_cmp_eq_u32_e32 vcc, 3, v109
v_cndmask_b32_e64 v109, v109, 0, vcc
s_branch 3
v_and_b32_e32 v109, 3, v0
v_cmp_ne_u32_e32 vcc, 1, v109
v_cndmask_b32_e32 v109, -1.0, v113, vcc
v_and_b32_e32 v117, 1, v0
v_bfe_u32 v118, v0, 1, 1
v_bfe_u32 v119, v0, 2, 1
v_bfe_u32 v120, v0, 3, 1
v_xor_b32_e32 v117, v117, v120
v_xor_b32_e32 v118, v118, v119
v_xor_b32_e32 v119, v119, v120
v_cmp_eq_u32_e32 vcc, 0, v118
v_cndmask_b32_e32 v110, -1.0, v113, vcc
v_cmp_eq_u32_e32 vcc, 1, v120
v_cndmask_b32_e32 v110, 0, v110, vcc
v_cmp_eq_u32_e32 vcc, 1, v117
v_cndmask_b32_e32 v111, -1.0, v113, vcc
v_cmp_eq_u32_e32 vcc, 1, v119
v_cndmask_b32_e32 v111, 0, v111, vcc
v_lshrrev_b32_e64 v117, 2, s83
v_and_b32_e32 v118, 3, v0
v_bfe_u32 v119, v0, 4, 3
v_mad_u32_u24 v104, v119, 4, v118
v_lshlrev_b32_e32 v104, 4, v104
v_mad_u32_u24 v105, v117, 4, v118
v_lshlrev_b32_e32 v105, 4, v105
v_bfe_u32 v117, v0, 2, 2
v_and_b32_e32 v118, 1, v117
v_mad_u32_u24 v120, v117, 32, v118
v_lshlrev_b32_e32 v120, 6, v120
v_xor_b32_e32 v105, v105, v120
v_mad_u32_u24 v120, v117, 32, 0
v_lshlrev_b32_e32 v120, 6, v120
v_xor_b32_e32 v104, v104, v120
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 35
v_and_b32_e32 v117, 1, v0
v_bfe_u32 v118, v0, 1, 1
v_xor_b32_e32 v117, v117, v118
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v119, 1, v117, vcc
v_xor_b32_e64 v119, v119, 1
v_xor_b32_e32 v118, v118, v119
v_bfe_u32 v119, v0, 2, 1
v_bfe_u32 v120, v0, 3, 1
v_mad_u32_u24 v118, v118, 2, v120
v_mul_u32_u24_e32 v117, 0x218, v117
v_mad_u32_u24 v118, v118, 2, v117
v_xor_b32_e32 v118, v118, v119
v_and_b32_e32 v119, 0x1f0, v0
v_xor_b32_e32 v118, v118, v119
v_lshlrev_b32_e32 v118, 2, v118
v_xor_b32_e32 v101, 0x1850, v118
v_xor_b32_e32 v102, 0x1870, v118
v_xor_b32_e32 v103, 32, v118
s_bitcmp1_b32 s18, 1
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v100, v118, v103, vcc
v_cndmask_b32_e32 v103, v103, v118, vcc
s_branch 24
v_and_b32_e32 v117, 1, v0
v_bfe_u32 v118, v0, 1, 1
v_bfe_u32 v119, v0, 2, 1
v_bfe_u32 v120, v0, 3, 1
v_xor_b32_e32 v117, v117, v118
v_mad_u32_u24 v118, v118, 2, v120
v_mul_u32_u24_e32 v117, 0x209, v117
v_mad_u32_u24 v118, v118, 2, v117
v_xor_b32_e32 v118, v118, v119
v_and_b32_e32 v119, 0x1f0, v0
v_or_b32_e32 v118, v118, v119
v_lshlrev_b32_e32 v100, 2, v118
v_xor_b32_e32 v101, 0x181c, v100
v_xor_b32_e32 v102, 0x183c, v100
v_xor_b32_e32 v103, 32, v100
v_subrev_co_u32_e32 v106, vcc, s69, v106
v_mov_b32_e32 v118, s68
v_cmp_lt_i32_e32 vcc, v106, v118
v_subb_co_u32_e32 v117, vcc, 0, v114, vcc
v_mad_i32_i24 v106, v117, s68, v106
v_mad_i32_i24 v108, v117, s73, v108
v_mad_i32_i24 v107, v117, s72, v107
v_mov_b32_e32 v118, s67
v_cmp_lt_i32_e32 vcc, v107, v118
v_subb_co_u32_e32 v117, vcc, 0, v114, vcc
v_add_co_u32_e32 v108, vcc, v108, v117
v_mad_i32_i24 v107, v117, v118, v107
v_subrev_co_u32_e32 v107, vcc, s70, v107
v_cmp_lt_i32_e32 vcc, v107, v118
v_subb_co_u32_e32 v117, vcc, 0, v114, vcc
v_add_co_u32_e32 v108, vcc, v108, v117
v_mad_i32_i24 v107, v117, s67, v107
v_subrev_co_u32_e32 v108, vcc, s71, v108
s_mov_b32 s77, 0
s_mov_b32 s78, s16
s_mov_b32 s76, s78
s_sub_u32 s84, -1, s83
s_sub_u32 s84, s84, 32
s_mov_b32 s39, 0
s_mov_b32 s85, 16
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 728
s_branch 1607
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_bitcmp1_b32 s18, 17
s_cbranch_scc1 246
s_add_u32 s80, s80, s17
s_cmp_eq_u32 s80, 0
s_cbranch_scc1 243
s_mov_b32 s81, 0
s_bitcmp1_b32 s18, 16
s_cbranch_scc1 232
s_add_u32 s82, s16, 31
s_lshr_b32 s82, s82, 5
v_mov_b32_e32 v118, s80
v_mul_u32_u24_e32 v118, s82, v118
v_add_co_u32_e32 v118, vcc, s17, v118
v_sub_co_u32_e64 v118, vcc, v118, 1
v_cvt_f32_u32_e32 v120, s17
v_rcp_f32_e32 v120, v120
v_mul_f32_e32 v120, 0x4f800000, v120
v_cvt_u32_f32_e32 v120, v120
v_mul_lo_u32 v121, s17, v120
v_mul_hi_u32 v122, s17, v120
v_sub_co_u32_e32 v123, vcc, 0, v121
v_cmp_ne_i32_e64 s[46:47], 0, v122
v_cndmask_b32_e64 v121, v123, v121, s[46:47]
v_mul_hi_u32 v121, v121, v120
v_sub_co_u32_e32 v122, vcc, v120, v121
v_add_co_u32_e32 v120, vcc, v120, v121
v_cndmask_b32_e64 v120, v120, v122, s[46:47]
v_mul_hi_u32 v120, v120, v118
v_mul_lo_u32 v121, v120, s17
v_sub_co_u32_e32 v122, vcc, v118, v121
v_cmp_ge_u32_e64 s[46:47], v118, v121
v_cmp_ge_u32_e64 s[48:49], v122, s17
v_add_co_u32_e32 v122, vcc, 1, v120
s_and_b64 s[48:49], s[46:47], s[48:49]
v_add_co_u32_e32 v121, vcc, -1, v120
v_cndmask_b32_e64 v122, v120, v122, s[48:49]
v_cndmask_b32_e64 v122, v121, v122, s[46:47]
v_cmp_ne_i32_e64 vcc, 0, s17
v_cndmask_b32_e32 v117, -1, v122, vcc
v_readfirstlane_b32 s79, v117
v_mul_u32_u24_e64 v117, v117, s8
v_cvt_f32_u32_e32 v120, s82
v_rcp_f32_e32 v120, v120
v_mul_f32_e32 v120, 0x4f800000, v120
v_cvt_u32_f32_e32 v120, v120
v_mul_lo_u32 v121, s82, v120
v_mul_hi_u32 v122, s82, v120
v_sub_co_u32_e32 v123, vcc, 0, v121
v_cmp_ne_i32_e64 s[46:47], 0, v122
v_cndmask_b32_e64 v121, v123, v121, s[46:47]
v_mul_hi_u32 v121, v121, v120
v_sub_co_u32_e32 v122, vcc, v120, v121
v_add_co_u32_e32 v120, vcc, v120, v121
v_cndmask_b32_e64 v120, v120, v122, s[46:47]
v_mul_hi_u32 v120, v120, v117
v_mul_lo_u32 v121, v120, s82
v_sub_co_u32_e32 v122, vcc, v117, v121
v_cmp_ge_u32_e64 s[46:47], v117, v121
v_cmp_ge_u32_e64 s[48:49], v122, s82
v_add_co_u32_e32 v122, vcc, 1, v120
s_and_b64 s[48:49], s[46:47], s[48:49]
v_add_co_u32_e32 v121, vcc, -1, v120
v_cndmask_b32_e64 v122, v120, v122, s[48:49]
v_cndmask_b32_e64 v122, v121, v122, s[46:47]
v_cmp_ne_i32_e64 vcc, 0, s82
v_cndmask_b32_e32 v118, -1, v122, vcc
v_readfirstlane_b32 s44, v117
v_readfirstlane_b32 s77, v118
s_mul_i32 s77, s77, s82
s_sub_u32 s77, s44, s77
v_sub_co_u32_e32 v118, vcc, s8, v118
v_sub_co_u32_e32 v118, vcc, s17, v118
v_and_b32_e64 v120, v0, 63
v_cmp_eq_u32_e64 vcc, v120, 0
v_cndmask_b32_e32 v118, 1, v118, vcc
s_sub_u32 s50, 0, s68
s_sub_u32 s51, 0, s67
v_mul_u32_u24_e64 v122, v118, 32
v_cvt_f32_u32_e32 v123, s50
v_rcp_f32_e32 v123, v123
v_mul_f32_e32 v123, 0x4f800000, v123
v_cvt_u32_f32_e32 v123, v123
v_mul_lo_u32 v124, s50, v123
v_mul_hi_u32 v125, s50, v123
v_sub_co_u32_e32 v126, vcc, 0, v124
v_cmp_ne_i32_e64 s[46:47], 0, v125
v_cndmask_b32_e64 v124, v126, v124, s[46:47]
v_mul_hi_u32 v124, v124, v123
v_sub_co_u32_e32 v125, vcc, v123, v124
v_add_co_u32_e32 v123, vcc, v123, v124
v_cndmask_b32_e64 v123, v123, v125, s[46:47]
v_mul_hi_u32 v123, v123, v122
v_mul_lo_u32 v124, v123, s50
v_sub_co_u32_e32 v125, vcc, v122, v124
v_cmp_ge_u32_e64 s[46:47], v122, v124
v_cmp_ge_u32_e64 s[48:49], v125, s50
v_add_co_u32_e32 v125, vcc, 1, v123
s_and_b64 s[48:49], s[46:47], s[48:49]
v_add_co_u32_e32 v124, vcc, -1, v123
v_cndmask_b32_e64 v125, v123, v125, s[48:49]
v_cndmask_b32_e64 v125, v124, v125, s[46:47]
v_cmp_ne_i32_e64 vcc, 0, s50
v_cndmask_b32_e32 v120, -1, v125, vcc
v_mad_i32_i24 v121, v120, s68, v122
v_mul_u32_u24_e64 v122, v120, 1
v_cvt_f32_u32_e32 v123, s51
v_rcp_f32_e32 v123, v123
v_mul_f32_e32 v123, 0x4f800000, v123
v_cvt_u32_f32_e32 v123, v123
v_mul_lo_u32 v124, s51, v123
v_mul_hi_u32 v125, s51, v123
v_sub_co_u32_e32 v126, vcc, 0, v124
v_cmp_ne_i32_e64 s[46:47], 0, v125
v_cndmask_b32_e64 v124, v126, v124, s[46:47]
v_mul_hi_u32 v124, v124, v123
v_sub_co_u32_e32 v125, vcc, v123, v124
v_add_co_u32_e32 v123, vcc, v123, v124
v_cndmask_b32_e64 v123, v123, v125, s[46:47]
v_mul_hi_u32 v123, v123, v122
v_mul_lo_u32 v124, v123, s51
v_sub_co_u32_e32 v125, vcc, v122, v124
v_cmp_ge_u32_e64 s[46:47], v122, v124
v_cmp_ge_u32_e64 s[48:49], v125, s51
v_add_co_u32_e32 v125, vcc, 1, v123
s_and_b64 s[48:49], s[46:47], s[48:49]
v_add_co_u32_e32 v124, vcc, -1, v123
v_cndmask_b32_e64 v125, v123, v125, s[48:49]
v_cndmask_b32_e64 v125, v124, v125, s[46:47]
v_cmp_ne_i32_e64 vcc, 0, s51
v_cndmask_b32_e32 v120, -1, v125, vcc
v_mad_i32_i24 v122, v120, s67, v122
v_readfirstlane_b32 s69, v121
v_readfirstlane_b32 s70, v122
v_readfirstlane_b32 s71, v120
v_add_co_u32_e32 v106, vcc, s69, v106
v_addc_co_u32_e32 v123, vcc, 0, v114, vcc
v_mad_i32_i24 v106, v123, s68, v106
v_mad_i32_i24 v108, v123, s73, v108
v_mad_i32_i24 v107, v123, s72, v107
v_cmp_ge_i32_e64 vcc, v107, 0
v_addc_co_u32_e32 v123, vcc, 0, v114, vcc
v_add_co_u32_e32 v108, vcc, v108, v123
v_mad_i32_i24 v107, v123, s67, v107
v_add_co_u32_e32 v107, vcc, s70, v107
v_addc_co_u32_e32 v123, vcc, 0, v114, vcc
v_add_co_u32_e32 v108, vcc, v108, v123
v_mad_i32_i24 v107, v123, s67, v107
v_add_co_u32_e32 v108, vcc, s71, v108
v_readlane_b32 s69, v121, 1
v_readlane_b32 s70, v122, 1
v_readlane_b32 s71, v120, 1
s_add_u32 s78, s77, s79
s_cmp_le_u32 s78, s82
s_cselect_b32 s44, 0x20000, 0
s_cselect_b32 s78, s78, s82
s_or_b32 s18, s18, s44
s_lshl_b32 s77, s77, 5
s_lshl_b32 s78, s78, 5
s_min_u32 s78, s78, s16
s_cmp_eq_u32 s8, s17
s_cselect_b32 s44, 0x20000, 0
s_or_b32 s18, s18, s44
s_or_b32 s18, s18, s44
s_bitset1_b32 s18, 16
s_branch 36
s_lshr_b32 s77, s77, 5
s_add_u32 s78, s77, s79
s_sub_u32 s78, s78, s82
s_mov_b32 s77, 0
s_lshl_b32 s78, s78, 5
s_min_u32 s78, s78, s16
s_bitset1_b32 s18, 17
s_branch 8
s_bitset1_b32 s18, 18
s_mov_b32 s31, 0
s_mov_b32 s35, 0
s_mov_b32 s75, 16
s_branch 155
s_sub_u32 s80, s80, s17
s_subb_u32 s81, s81, 0
s_cbranch_scc1 65280
v_add_co_u32_e32 v106, vcc, s69, v106
v_addc_co_u32_e32 v117, vcc, 0, v114, vcc
v_mad_i32_i24 v106, v117, s68, v106
v_mad_i32_i24 v108, v117, s73, v108
v_mad_i32_i24 v107, v117, s72, v107
v_cmp_ge_i32_e64 vcc, v107, 0
v_addc_co_u32_e32 v117, vcc, 0, v114, vcc
v_add_co_u32_e32 v108, vcc, v108, v117
v_mad_i32_i24 v107, v117, s67, v107
v_add_co_u32_e32 v107, vcc, s70, v107
v_addc_co_u32_e32 v117, vcc, 0, v114, vcc
v_add_co_u32_e32 v108, vcc, v108, v117
v_mad_i32_i24 v107, v117, s67, v107
v_add_co_u32_e32 v108, vcc, s71, v108
v_readfirstlane_b32 s74, v108
v_sub_co_u32_e64 v117, vcc, v108, s74
v_mul_lo_u32 v119, v117, s56
v_cmp_ge_u32_e64 s[48:49], v108, s12
s_lshl_b32 s52, s15, 2
v_subrev_co_u32_e32 v118, vcc, s68, v106
v_and_b32_e64 v120, v0, 3
v_subrev_co_u32_e32 v120, vcc, 1, v120
v_mad_i32_i24 v118, v118, 2, v120
v_cmp_ge_u32_e64 s[44:45], v118, s15
v_mad_u32_u24 v118, 4, v118, v119
s_or_b64 s[44:45], s[44:45], s[48:49]
v_subrev_co_u32_e32 v119, vcc, s67, v107
v_mad_i32_i24 v119, v119, 2, -1
v_cmp_ge_u32_e64 s[46:47], v119, s14
s_or_b64 s[50:51], s[44:45], s[46:47]
v_mad_u32_u24 v90, v119, s52, v118
v_cndmask_b32_e64 v90, v90, -1, s[50:51]
v_add_co_u32_e32 v119, vcc, 1, v119
v_cmp_ge_u32_e64 s[46:47], v119, s14
s_or_b64 s[50:51], s[44:45], s[46:47]
v_mad_u32_u24 v91, v119, s52, v118
v_cndmask_b32_e64 v91, v91, -1, s[50:51]
v_add_co_u32_e32 v119, vcc, 1, v119
v_cmp_ge_u32_e64 s[46:47], v119, s14
s_or_b64 s[50:51], s[44:45], s[46:47]
v_mad_u32_u24 v92, v119, s52, v118
v_cndmask_b32_e64 v92, v92, -1, s[50:51]
v_add_co_u32_e32 v119, vcc, 1, v119
v_cmp_ge_u32_e64 s[46:47], v119, s14
s_or_b64 s[50:51], s[44:45], s[46:47]
v_mad_u32_u24 v93, v119, s52, v118
v_cndmask_b32_e64 v93, v93, -1, s[50:51]
v_mov_b32_dpp v118, v91  quad_perm:[1,2,2,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v98, v92  quad_perm:[2,1,1,2] row_mask:0xf bank_mask:0xf
v_add_co_u32_e64 v119, vcc, v0, 2
v_bfe_u32 v119, v119, 2, 1
v_cmp_eq_u32_e64 vcc, v119, 1
v_cndmask_b32_e32 v98, v118, v98, vcc
v_mul_lo_u32 v117, v117, s58
v_add_co_u32_e32 v117, vcc, v98, v117
v_cmp_eq_u32_e64 vcc, v98, -1
v_cndmask_b32_e32 v98, v117, v98, vcc
v_and_b32_e32 v118, 3, v0
v_min_u32_e32 v118, 2, v118
v_bfe_u32 v119, v0, 2, 5
v_lshlrev_b32_e32 v118, 2, v118
v_mad_u32_u24 v120, s62, v119, v118
v_mad_u32_u24 v94, v120, 1, 0
v_mad_u32_u24 v95, v120, 1, 0
v_mad_u32_u24 v96, v120, 1, 24
v_mad_u32_u24 v97, v120, 1, 12
s_sub_u32 s76, s77, 32
s_add_u32 s76, s76, 32
s_cmp_ge_u32 s76, s78
s_cbranch_scc1 65424
s_lshr_b32 s47, -1, 16
s_and_b32 s44, s47, s56
s_lshr_b32 s45, s56, 16
s_mul_i32 s46, s45, s74
s_mul_i32 s28, s44, s74
s_lshl_b32 s44, s46, 16
s_lshr_b32 s47, s46, 16
s_add_u32 s28, s44, s28
s_addc_u32 s29, s47, 0
s_add_u32 s28, s28, s20
s_addc_u32 s29, s29, s21
s_mul_i32 s44, s60, s66
s_add_u32 s28, s28, s44
s_addc_u32 s29, s29, 0
s_mov_b32 s31, 0x20000
v_bfe_u32 v117, v0, 2, 5
v_add_co_u32_e64 v117, vcc, v117, s76
v_cmp_lt_u32_e64 vcc, v117, s16
v_cndmask_b32_e32 v94, -1, v94, vcc
v_cndmask_b32_e32 v95, -1, v95, vcc
v_cndmask_b32_e32 v96, -1, v96, vcc
v_cndmask_b32_e32 v97, -1, v97, vcc
s_lshr_b32 s47, -1, 16
s_and_b32 s44, s47, s62
s_lshr_b32 s45, s62, 16
s_mul_i32 s46, s45, s76
s_mul_i32 s32, s44, s76
s_lshl_b32 s44, s46, 16
s_lshr_b32 s47, s46, 16
s_add_u32 s32, s44, s32
s_addc_u32 s33, s47, 0
s_add_u32 s32, s32, s22
s_addc_u32 s33, s33, s23
s_mul_i32 s44, s61, s66
s_add_u32 s32, s32, s44
s_addc_u32 s33, s33, 0
s_mov_b32 s35, 0x20000
s_sub_u32 s75, s13, 1
s_add_u32 s44, s54, 0x68
s_addc_u32 s45, s55, 0
s_setpc_b64 s[44:45]
s_nop 0
s_nop 0
v_mac_f32_dpp v2, v2, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v5, v5, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v3, v3, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v4, v4, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v2, v5, v2  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v3, v4, v3  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v2, v2, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v3, v3, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v34, v3, v2  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v6, v6, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v9, v9, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v7, v7, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v8, v8, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v6, v9, v6  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v7, v8, v7  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v6, v6, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v7, v7, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v35, v7, v6  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v10, v10, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v13, v13, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v11, v11, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v12, v12, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v10, v13, v10  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v11, v12, v11  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v10, v10, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v11, v11, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v36, v11, v10  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v14, v14, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v17, v17, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v15, v15, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v16, v16, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v14, v17, v14  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v15, v16, v15  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v14, v14, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v15, v15, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v37, v15, v14  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v18, v18, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v21, v21, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v19, v19, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v20, v20, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v18, v21, v18  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v19, v20, v19  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v18, v18, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v19, v19, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v38, v19, v18  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v22, v22, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v25, v25, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v23, v23, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v24, v24, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v22, v25, v22  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v23, v24, v23  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v22, v22, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v23, v23, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v39, v23, v22  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v26, v26, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v29, v29, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v27, v27, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v28, v28, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v26, v29, v26  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v27, v28, v27  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v26, v26, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v27, v27, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v40, v27, v26  row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v30, v30, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v33, v33, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v31, v31, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v32, v32, v110  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v30, v33, v30  row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_add_f32_dpp v31, v32, v31  row_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v30, v30, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v31, v31, v111  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
s_nop 1
v_add_f32_dpp v41, v31, v30  row_half_mirror row_mask:0xf bank_mask:0xf
s_nop 4
buffer_store_dword v34, v99, s[36:39], 0 offen
s_add_u32 s36, s36, s60
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 1
s_cselect_b32 s39, 0, s39
s_nop 4
buffer_store_dword v35, v99, s[36:39], 0 offen
s_add_u32 s36, s36, s60
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 1
s_cselect_b32 s39, 0, s39
s_nop 4
buffer_store_dword v36, v99, s[36:39], 0 offen
s_add_u32 s36, s36, s60
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 1
s_cselect_b32 s39, 0, s39
s_nop 4
buffer_store_dword v37, v99, s[36:39], 0 offen
s_add_u32 s36, s36, s60
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 1
s_cselect_b32 s39, 0, s39
s_lshl_b32 s44, s60, 2
s_add_u32 s36, s36, s44
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 4
s_cselect_b32 s39, 0, s39
s_nop 4
buffer_store_dword v38, v99, s[36:39], 0 offen
s_add_u32 s36, s36, s60
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 1
s_cselect_b32 s39, 0, s39
s_nop 4
buffer_store_dword v39, v99, s[36:39], 0 offen
s_add_u32 s36, s36, s60
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 1
s_cselect_b32 s39, 0, s39
s_nop 4
buffer_store_dword v40, v99, s[36:39], 0 offen
s_add_u32 s36, s36, s60
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 1
s_cselect_b32 s39, 0, s39
s_nop 4
buffer_store_dword v41, v99, s[36:39], 0 offen
s_add_u32 s36, s36, s60
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 1
s_cselect_b32 s39, 0, s39
s_add_u32 s36, s36, s44
s_addc_u32 s37, s37, 0
s_lshl_b32 s44, s44, 2
s_add_u32 s36, s36, s44
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 20
s_cselect_b32 s39, 0, s39
s_sub_u32 s85, s13, 1
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
s_add_u32 s44, s84, s83
s_cmp_lt_i32 s44, 0
s_cbranch_scc0 24
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 1887
v_mov_b32_e32 v99, v98
s_add_u32 s84, s83, s77
s_lshr_b32 s47, -1, 16
s_and_b32 s44, s47, s57
s_lshr_b32 s45, s57, 16
s_mul_i32 s46, s45, s74
s_mul_i32 s36, s44, s74
s_lshl_b32 s44, s46, 16
s_lshr_b32 s47, s46, 16
s_add_u32 s36, s44, s36
s_addc_u32 s37, s47, 0
s_add_u32 s36, s36, s24
s_addc_u32 s37, s37, s25
s_mul_i32 s44, s60, s84
s_add_u32 s36, s36, s44
s_addc_u32 s37, s37, 0
s_mov_b32 s39, 0x20000
s_sub_u32 s84, s78, s77
s_sub_u32 s84, s84, 1
s_sub_u32 s84, s84, s83
s_cselect_b32 s39, 0, s39
s_barrier
s_add_u32 s44, s54, 0x8c
s_addc_u32 s45, s55, 0
s_setpc_b64 s[44:45]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_getpc_b64 s[54:55]
s_sub_u32 s54, s54, 0x58
s_subb_u32 s55, s55, 0
s_branch 65172
s_nop 0
ds_read_b128 v[42:45], v104 offset:25600
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v66, v90, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v100, v70
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_subrev_f32_e64 v78, v80, v78 div:2
v_subrev_f32_e64 v81, v79, v81 div:2
ds_read_b128 v[50:53], v105 offset:24576
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v79, v80, v79 div:2
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v68, v92, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v71
v_mad_f32 v80, v80, 1.0, -v79
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:24704
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:26112
s_cbranch_scc1 65148
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v67, v91, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v102, v72
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v78, v78, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v79, v79, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v105 offset:25088
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v80, v80, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v69, v93, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
ds_write_b32 v103, v73
v_mac_f32_dpp v81, v81, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
ds_read_b128 v[62:65], v105 offset:25216
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 65033
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v104 offset:33792
s_setprio 0
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v70, v90, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v100, v74 offset:8192
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_subrev_f32_e64 v82, v84, v82 div:2
v_subrev_f32_e64 v85, v83, v85 div:2
ds_read_b128 v[50:53], v105 offset:32768
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v83, v84, v83 div:2
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v72, v92, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v75 offset:8192
v_mad_f32 v84, v84, 1.0, -v83
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:32896
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_mov_b32 m0, -1
ds_write_b8 v112, v112 offset:65504
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:34304
s_cbranch_scc1 65004
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v71, v91, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v102, v76 offset:8192
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v82, v82, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v83, v83, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v105 offset:33280
s_add_u32 m0, s86, 32
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v84, v84, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(4)
buffer_load_dword v73, v93, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
s_nop 0
ds_write_b32 v103, v77 offset:8192
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
s_cbranch_vccz 1504
ds_read_b128 v[62:65], v105 offset:33408
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 64889
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v104 offset:41984
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v74, v90, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v100, v78 offset:16384
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_mac_f32_dpp v85, v85, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v86, v88, v86 div:2
ds_read_b128 v[50:53], v105 offset:40960
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_subrev_f32_e64 v89, v87, v89 div:2
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v76, v92, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v79 offset:16384
v_add_f32_e64 v87, v88, v87 div:2
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:41088
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:42496
s_cbranch_scc1 64860
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v75, v91, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v102, v80 offset:16384
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mad_f32 v88, v88, 1.0, -v87
v_mac_f32_dpp v86, v86, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v105 offset:41472
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v87, v87, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v77, v93, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
ds_write_b32 v103, v81 offset:16384
v_mac_f32_dpp v88, v88, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
ds_read_b128 v[62:65], v105 offset:41600
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 64745
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v104 offset:1024
s_setprio 0
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v78, v90, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v100, v82 offset:24576
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_mac_f32_dpp v89, v89, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_subrev_f32_e64 v66, v68, v66 div:2
ds_read_b128 v[50:53], v105
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_subrev_f32_e64 v69, v67, v69 div:2
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v80, v92, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v83 offset:24576
v_add_f32_e64 v67, v68, v67 div:2
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:128
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_mov_b32 m0, -1
ds_write_b8 v112, v112 offset:65488
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:1536
s_cbranch_scc1 64716
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v79, v91, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v102, v84 offset:24576
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mad_f32 v68, v68, 1.0, -v67
v_mac_f32_dpp v66, v66, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v105 offset:512
s_add_u32 m0, s86, 16
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v67, v67, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(4)
buffer_load_dword v81, v93, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
s_nop 0
ds_write_b32 v103, v85 offset:24576
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
s_cbranch_vccz 1216
ds_read_b128 v[62:65], v105 offset:640
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 64601
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v104 offset:9216
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v82, v90, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v100, v86 offset:32768
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_mac_f32_dpp v68, v68, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v69, v69, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v105 offset:8192
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_subrev_f32_e64 v70, v72, v70 div:2
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v84, v92, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v87 offset:32768
v_subrev_f32_e64 v73, v71, v73 div:2
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:8320
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:9728
s_cbranch_scc1 64572
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v83, v91, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v102, v88 offset:32768
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_e64 v71, v72, v71 div:2
v_mad_f32 v72, v72, 1.0, -v71
ds_read_b128 v[58:61], v105 offset:8704
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v70, v70, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v85, v93, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
ds_write_b32 v103, v89 offset:32768
v_mac_f32_dpp v71, v71, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
ds_read_b128 v[62:65], v105 offset:8832
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 64457
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v104 offset:17408
s_setprio 0
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v86, v90, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v100, v66 offset:40960
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_mac_f32_dpp v72, v72, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v73, v73, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v105 offset:16384
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_subrev_f32_e64 v74, v76, v74 div:2
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v88, v92, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v67 offset:40960
v_subrev_f32_e64 v77, v75, v77 div:2
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:16512
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_mov_b32 m0, -1
ds_write_b8 v112, v112 offset:65472
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:17920
s_cbranch_scc1 64428
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v87, v91, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v102, v68 offset:40960
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_e64 v75, v76, v75 div:2
v_mad_f32 v76, v76, 1.0, -v75
ds_read_b128 v[58:61], v105 offset:16896
s_add_u32 m0, s86, 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v74, v74, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(4)
buffer_load_dword v89, v93, s[28:31], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
s_nop 0
ds_write_b32 v103, v69 offset:40960
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
s_cbranch_vccz 928
ds_read_b128 v[62:65], v105 offset:17024
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 64313
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
v_mac_f32_dpp v75, v75, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v76, v76, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v77, v77, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_branch 64666
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_getpc_b64 s[54:55]
s_sub_u32 s54, s54, 0x58
s_subb_u32 s55, s55, 0
s_branch 64292
s_nop 0
ds_read_b128 v[42:45], v104 offset:25600
s_setprio 0
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v66, v94, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v100, v70
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_dpp v78, v79, v79  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v78, v79, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v105 offset:24576
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_dpp v79, v81, v81  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v68, v96, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v71
v_mac_f32_dpp v79, v81, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:24704
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:26112
s_cbranch_scc1 64268
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v67, v95, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v102, v72
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_dpp v81, v80, v80  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v81, v80, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v105 offset:25088
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_add_f32_e64 v80, v78, v81
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v69, v97, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
ds_write_b32 v103, v73
v_add_f32_e64 v79, v79, v80 div:2
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
ds_read_b128 v[62:65], v105 offset:25216
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 64153
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v104 offset:33792
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v70, v94, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v100, v74 offset:8192
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_e64 v80, -v79, v80
v_add_f32_dpp v82, v83, v83  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v105 offset:32768
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_mac_f32_dpp v82, v83, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v72, v96, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v75 offset:8192
v_add_f32_dpp v83, v85, v85  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:32896
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_mov_b32 m0, -1
ds_write_b8 v112, v112 offset:65504
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:34304
s_cbranch_scc1 64124
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v71, v95, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v102, v76 offset:8192
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v83, v85, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v85, v84, v84  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v105 offset:33280
s_add_u32 m0, s86, 32
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v85, v84, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(4)
buffer_load_dword v73, v97, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
s_nop 0
ds_write_b32 v103, v77 offset:8192
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
s_cbranch_vccz 624
ds_read_b128 v[62:65], v105 offset:33408
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 64009
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v104 offset:41984
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v74, v94, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v100, v78 offset:16384
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_e64 v84, v82, v85
v_add_f32_e64 v83, v83, v84 div:2
ds_read_b128 v[50:53], v105 offset:40960
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v84, -v83, v84
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v76, v96, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v79 offset:16384
v_add_f32_dpp v86, v87, v87  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:41088
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:42496
s_cbranch_scc1 63980
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v75, v95, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v102, v80 offset:16384
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v86, v87, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v87, v89, v89  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v105 offset:41472
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v87, v89, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v77, v97, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
ds_write_b32 v103, v81 offset:16384
v_add_f32_dpp v89, v88, v88  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
ds_read_b128 v[62:65], v105 offset:41600
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 63865
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v104 offset:1024
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v78, v94, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
v_mac_f32_dpp v89, v88, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v88, v86, v89
v_add_f32_e64 v87, v87, v88 div:2
v_add_f32_e64 v88, -v87, v88
ds_write_b32 v100, v82 offset:24576
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_dpp v66, v67, v67  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v66, v67, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[50:53], v105
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_dpp v67, v69, v69  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v80, v96, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v83 offset:24576
v_mac_f32_dpp v67, v69, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:128
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_mov_b32 m0, -1
ds_write_b8 v112, v112 offset:65488
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:1536
s_cbranch_scc1 63828
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v79, v95, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v102, v84 offset:24576
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_dpp v69, v68, v68  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v69, v68, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v105 offset:512
s_add_u32 m0, s86, 16
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_add_f32_e64 v68, v66, v69
s_waitcnt vmcnt(15) lgkmcnt(4)
buffer_load_dword v81, v97, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
s_nop 0
ds_write_b32 v103, v85 offset:24576
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
s_cbranch_vccz 328
ds_read_b128 v[62:65], v105 offset:640
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 63713
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v104 offset:9216
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v82, v94, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v100, v86 offset:32768
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_e64 v67, v67, v68 div:2
v_add_f32_e64 v68, -v67, v68
ds_read_b128 v[50:53], v105 offset:8192
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_dpp v70, v71, v71  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v84, v96, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v87 offset:32768
v_mac_f32_dpp v70, v71, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:8320
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:9728
s_cbranch_scc1 63684
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v83, v95, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
ds_write_b32 v102, v88 offset:32768
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_dpp v71, v73, v73  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v71, v73, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[58:61], v105 offset:8704
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_add_f32_dpp v73, v72, v72  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v85, v97, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
ds_write_b32 v103, v89 offset:32768
v_mac_f32_dpp v73, v72, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
ds_read_b128 v[62:65], v105 offset:8832
s_nop 0
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 63569
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_read_b128 v[42:45], v104 offset:17408
s_setprio 2
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 0
buffer_load_dword v86, v94, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_e32 v9, v49, v59
ds_write_b32 v100, v66 offset:40960
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_e64 v72, v70, v73
v_add_f32_e64 v71, v71, v72 div:2
ds_read_b128 v[50:53], v105 offset:16384
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v72, -v71, v72
s_waitcnt vmcnt(15) lgkmcnt(3)
buffer_load_dword v88, v96, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
s_getpc_b64 s[54:55]
ds_write_b32 v101, v67 offset:40960
v_add_f32_dpp v74, v75, v75  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
s_nop 0
ds_read_b128 v[54:57], v105 offset:16512
s_nop 0
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_sub_u32 s85, s85, 2
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_waitcnt vmcnt(15) lgkmcnt(2)
s_mov_b32 m0, -1
ds_write_b8 v112, v112 offset:65472
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_nop 0
ds_read_b128 v[46:49], v104 offset:17920
s_cbranch_scc1 63540
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 0
buffer_load_dword v87, v95, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_e32 v9, v45, v51
v_mac_f32_dpp v74, v75, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v75, v77, v77  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v75, v77, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v77, v76, v76  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
ds_write_b32 v102, v68 offset:40960
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v77, v76, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v76, v74, v77
ds_read_b128 v[58:61], v105 offset:16896
s_add_u32 m0, s86, 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_add_f32_e64 v75, v75, v76 div:2
s_waitcnt vmcnt(15) lgkmcnt(4)
buffer_load_dword v89, v97, s[32:35], 0 offen
s_nop 0
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
s_getpc_b64 s[54:55]
s_nop 0
ds_write_b32 v103, v69 offset:40960
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
s_nop 0
s_cbranch_vccz 32
ds_read_b128 v[62:65], v105 offset:17024
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_nop 0
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_sub_u32 s75, s75, 2
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_nop 0
s_nop 0
s_cbranch_scc1 63417
s_waitcnt vmcnt(8) lgkmcnt(2)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_add_f32_e64 v76, -v75, v76
s_branch 64650
s_nop 2
s_nop 0
s_nop 0
s_add_u32 s44, s54, 36
s_addc_u32 s45, s55, 0
s_nop 1
s_cbranch_vccnz 67
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 60
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 53
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 46
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 39
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 32
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 25
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 18
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
s_nop 0
s_nop 1
s_nop 0
s_nop 0
s_nop 0
s_cbranch_vccnz 11
v_cmp_eq_u32_e32 vcc, src_lds_direct, v115
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
s_setpc_b64 s[44:45]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_endpgm
