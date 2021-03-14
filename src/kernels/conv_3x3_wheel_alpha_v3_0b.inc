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

v_mov_b32_e32 v112, 1.0
v_mov_b32_e32 v113, 0
s_mov_b32 m0, -1
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_mov_b32_e32 v114, 0
s_load_dwordx16 s[12:27], s[6:7], 0x0
s_waitcnt lgkmcnt(0)
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
s_cbranch_scc0 2885
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
s_cbranch_scc0 2846
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
s_bfe_u32 s83, s66, 0x20007
s_lshl_b32 s83, s83, 2
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
v_sub_u32_e32 v9, vcc, 0, v7
v_cmp_ne_i32_e64 s[34:35], 0, v8
v_cndmask_b32_e64 v7, v9, v7, s[34:35]
v_mul_hi_u32 v7, v7, v6
v_sub_u32_e32 v8, vcc, v6, v7
v_add_u32_e32 v6, vcc, v6, v7
v_cndmask_b32_e64 v6, v6, v8, s[34:35]
v_mul_hi_u32 v6, v6, v107
v_mul_lo_u32 v7, v6, s31
v_sub_u32_e32 v8, vcc, v107, v7
v_cmp_ge_u32_e64 s[34:35], v107, v7
v_cmp_ge_u32_e64 s[36:37], v8, s31
v_add_u32_e32 v8, vcc, 1, v6
s_and_b64 s[36:37], s[34:35], s[36:37]
v_add_u32_e32 v7, vcc, -1, v6
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
v_sub_u32_e32 v9, vcc, 0, v7
v_cmp_ne_i32_e64 s[34:35], 0, v8
v_cndmask_b32_e64 v7, v9, v7, s[34:35]
v_mul_hi_u32 v7, v7, v6
v_sub_u32_e32 v8, vcc, v6, v7
v_add_u32_e32 v6, vcc, v6, v7
v_cndmask_b32_e64 v6, v6, v8, s[34:35]
v_mul_hi_u32 v6, v6, v107
v_mul_lo_u32 v7, v6, s30
v_sub_u32_e32 v8, vcc, v107, v7
v_cmp_ge_u32_e64 s[34:35], v107, v7
v_cmp_ge_u32_e64 s[36:37], v8, s30
v_add_u32_e32 v8, vcc, 1, v6
s_and_b64 s[36:37], s[34:35], s[36:37]
v_add_u32_e32 v7, vcc, -1, v6
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
v_add_u32_e64 v106, vcc, v106, s68
v_add_u32_e64 v107, vcc, v107, s67
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
v_cndmask_b32_e32 v109, v109, v112, vcc
v_cmp_eq_u32_e32 vcc, 2, v109
v_cndmask_b32_e64 v109, v109, -v112, vcc
v_cmp_eq_u32_e32 vcc, 3, v109
v_cndmask_b32_e64 v109, v109, 0, vcc
s_branch 3
v_and_b32_e32 v109, 3, v0
v_cmp_ne_u32_e32 vcc, 1, v109
v_cndmask_b32_e32 v109, -1.0, v112, vcc
v_and_b32_e32 v115, 1, v0
v_bfe_u32 v116, v0, 1, 1
v_bfe_u32 v117, v0, 2, 1
v_bfe_u32 v118, v0, 3, 1
v_xor_b32_e32 v115, v115, v118
v_xor_b32_e32 v116, v116, v117
v_xor_b32_e32 v117, v117, v118
v_cmp_eq_u32_e32 vcc, 0, v116
v_cndmask_b32_e32 v110, -1.0, v112, vcc
v_cmp_eq_u32_e32 vcc, 1, v118
v_cndmask_b32_e32 v110, 0, v110, vcc
v_cmp_eq_u32_e32 vcc, 1, v115
v_cndmask_b32_e32 v111, -1.0, v112, vcc
v_cmp_eq_u32_e32 vcc, 1, v117
v_cndmask_b32_e32 v111, 0, v111, vcc
v_lshrrev_b32_e32 v115, 7, v0
v_and_b32_e32 v116, 3, v0
v_bfe_u32 v117, v0, 4, 3
v_mad_u32_u24 v104, v117, 4, v116
v_lshlrev_b32_e32 v104, 4, v104
v_mad_u32_u24 v105, v115, 4, v116
v_lshlrev_b32_e32 v105, 4, v105
v_bfe_u32 v115, v0, 2, 2
v_and_b32_e32 v116, 1, v115
v_mad_u32_u24 v118, v115, 32, v116
v_lshlrev_b32_e32 v118, 6, v118
v_xor_b32_e32 v105, v105, v118
v_mad_u32_u24 v118, v115, 32, 0
v_lshlrev_b32_e32 v118, 6, v118
v_xor_b32_e32 v104, v104, v118
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 35
v_and_b32_e32 v115, 1, v0
v_bfe_u32 v116, v0, 1, 1
v_xor_b32_e32 v115, v115, v116
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v117, 1, v115, vcc
v_xor_b32_e64 v117, v117, 1
v_xor_b32_e32 v116, v116, v117
v_bfe_u32 v117, v0, 2, 1
v_bfe_u32 v118, v0, 3, 1
v_mad_u32_u24 v116, v116, 2, v118
v_mul_u32_u24_e32 v115, 0x218, v115
v_mad_u32_u24 v116, v116, 2, v115
v_xor_b32_e32 v116, v116, v117
v_and_b32_e32 v117, 0x1f0, v0
v_xor_b32_e32 v116, v116, v117
v_lshlrev_b32_e32 v116, 2, v116
v_xor_b32_e32 v101, 0x1850, v116
v_xor_b32_e32 v102, 0x1870, v116
v_xor_b32_e32 v103, 32, v116
s_bitcmp1_b32 s18, 1
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v100, v116, v103, vcc
v_cndmask_b32_e32 v103, v103, v116, vcc
s_branch 24
v_and_b32_e32 v115, 1, v0
v_bfe_u32 v116, v0, 1, 1
v_bfe_u32 v117, v0, 2, 1
v_bfe_u32 v118, v0, 3, 1
v_xor_b32_e32 v115, v115, v116
v_mad_u32_u24 v116, v116, 2, v118
v_mul_u32_u24_e32 v115, 0x209, v115
v_mad_u32_u24 v116, v116, 2, v115
v_xor_b32_e32 v116, v116, v117
v_and_b32_e32 v117, 0x1f0, v0
v_or_b32_e32 v116, v116, v117
v_lshlrev_b32_e32 v100, 2, v116
v_xor_b32_e32 v101, 0x181c, v100
v_xor_b32_e32 v102, 0x183c, v100
v_xor_b32_e32 v103, 32, v100
v_subrev_u32_e32 v106, vcc, s69, v106
v_mov_b32_e32 v116, s68
v_cmp_lt_i32_e32 vcc, v106, v116
v_subb_u32_e32 v115, vcc, 0, v113, vcc
v_mad_i32_i24 v106, v115, s68, v106
v_mad_i32_i24 v108, v115, s73, v108
v_mad_i32_i24 v107, v115, s72, v107
v_mov_b32_e32 v116, s67
v_cmp_lt_i32_e32 vcc, v107, v116
v_subb_u32_e32 v115, vcc, 0, v113, vcc
v_add_u32_e32 v108, vcc, v108, v115
v_mad_i32_i24 v107, v115, v116, v107
v_subrev_u32_e32 v107, vcc, s70, v107
v_cmp_lt_i32_e32 vcc, v107, v116
v_subb_u32_e32 v115, vcc, 0, v113, vcc
v_add_u32_e32 v108, vcc, v108, v115
v_mad_i32_i24 v107, v115, s67, v107
v_subrev_u32_e32 v108, vcc, s71, v108
s_mov_b32 s77, 0
s_mov_b32 s78, s16
s_mov_b32 s76, s78
s_sub_u32 s84, -1, s83
s_sub_u32 s84, s84, 32
s_mov_b32 s39, 0
s_mov_b32 s85, 12
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 758
s_branch 1629
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_bitcmp1_b32 s18, 17
s_cbranch_scc1 285
s_add_u32 s80, s80, s17
s_cmp_eq_u32 s80, 0
s_cbranch_scc1 282
s_mov_b32 s81, 0
s_bitcmp1_b32 s18, 16
s_cbranch_scc1 271
s_add_u32 s82, s16, 31
s_lshr_b32 s82, s82, 5
v_mov_b32_e32 v116, s80
v_mul_u32_u24_e32 v116, s82, v116
v_add_u32_e32 v116, vcc, s17, v116
v_sub_u32_e64 v116, vcc, v116, 1
v_cvt_f32_u32_e32 v118, s17
v_rcp_f32_e32 v118, v118
v_mul_f32_e32 v118, 0x4f800000, v118
v_cvt_u32_f32_e32 v118, v118
v_mul_lo_u32 v119, s17, v118
v_mul_hi_u32 v120, s17, v118
v_sub_u32_e32 v121, vcc, 0, v119
v_cmp_ne_i32_e64 s[46:47], 0, v120
v_cndmask_b32_e64 v119, v121, v119, s[46:47]
v_mul_hi_u32 v119, v119, v118
v_sub_u32_e32 v120, vcc, v118, v119
v_add_u32_e32 v118, vcc, v118, v119
v_cndmask_b32_e64 v118, v118, v120, s[46:47]
v_mul_hi_u32 v118, v118, v116
v_mul_lo_u32 v119, v118, s17
v_sub_u32_e32 v120, vcc, v116, v119
v_cmp_ge_u32_e64 s[46:47], v116, v119
v_cmp_ge_u32_e64 s[48:49], v120, s17
v_add_u32_e32 v120, vcc, 1, v118
s_and_b64 s[48:49], s[46:47], s[48:49]
v_add_u32_e32 v119, vcc, -1, v118
v_cndmask_b32_e64 v120, v118, v120, s[48:49]
v_cndmask_b32_e64 v120, v119, v120, s[46:47]
v_cmp_ne_i32_e64 vcc, 0, s17
v_cndmask_b32_e32 v115, -1, v120, vcc
v_readfirstlane_b32 s79, v115
v_mul_u32_u24_e64 v115, v115, s8
v_cvt_f32_u32_e32 v118, s82
v_rcp_f32_e32 v118, v118
v_mul_f32_e32 v118, 0x4f800000, v118
v_cvt_u32_f32_e32 v118, v118
v_mul_lo_u32 v119, s82, v118
v_mul_hi_u32 v120, s82, v118
v_sub_u32_e32 v121, vcc, 0, v119
v_cmp_ne_i32_e64 s[46:47], 0, v120
v_cndmask_b32_e64 v119, v121, v119, s[46:47]
v_mul_hi_u32 v119, v119, v118
v_sub_u32_e32 v120, vcc, v118, v119
v_add_u32_e32 v118, vcc, v118, v119
v_cndmask_b32_e64 v118, v118, v120, s[46:47]
v_mul_hi_u32 v118, v118, v115
v_mul_lo_u32 v119, v118, s82
v_sub_u32_e32 v120, vcc, v115, v119
v_cmp_ge_u32_e64 s[46:47], v115, v119
v_cmp_ge_u32_e64 s[48:49], v120, s82
v_add_u32_e32 v120, vcc, 1, v118
s_and_b64 s[48:49], s[46:47], s[48:49]
v_add_u32_e32 v119, vcc, -1, v118
v_cndmask_b32_e64 v120, v118, v120, s[48:49]
v_cndmask_b32_e64 v120, v119, v120, s[46:47]
v_cmp_ne_i32_e64 vcc, 0, s82
v_cndmask_b32_e32 v116, -1, v120, vcc
v_cvt_f32_u32_e32 v118, s82
v_rcp_f32_e32 v118, v118
v_mul_f32_e32 v118, 0x4f800000, v118
v_cvt_u32_f32_e32 v118, v118
v_mul_lo_u32 v119, s82, v118
v_mul_hi_u32 v120, s82, v118
v_sub_u32_e32 v121, vcc, 0, v119
v_cmp_ne_i32_e64 s[46:47], 0, v120
v_cndmask_b32_e64 v119, v121, v119, s[46:47]
v_mul_hi_u32 v119, v119, v118
v_sub_u32_e32 v120, vcc, v118, v119
v_add_u32_e32 v118, vcc, v118, v119
v_cndmask_b32_e64 v118, v118, v120, s[46:47]
v_mul_hi_u32 v118, v118, v115
v_mul_lo_u32 v119, v118, s82
v_sub_u32_e32 v120, vcc, v115, v119
v_cmp_ge_u32_e64 s[46:47], v115, v119
v_cmp_ge_u32_e64 s[48:49], v120, s82
v_add_u32_e32 v120, vcc, 1, v118
s_and_b64 s[48:49], s[46:47], s[48:49]
v_add_u32_e32 v119, vcc, -1, v118
v_cndmask_b32_e64 v120, v118, v120, s[48:49]
v_cndmask_b32_e64 v120, v119, v120, s[46:47]
v_cmp_ne_i32_e64 vcc, 0, s82
v_cndmask_b32_e32 v116, -1, v120, vcc
v_readfirstlane_b32 s44, v115
v_readfirstlane_b32 s77, v116
s_mul_i32 s77, s77, s82
s_sub_u32 s77, s44, s77
v_sub_u32_e32 v116, vcc, s8, v116
v_sub_u32_e32 v116, vcc, s17, v116
v_and_b32_e64 v118, v0, 63
v_cmp_eq_u32_e64 vcc, v118, 0
v_cndmask_b32_e32 v116, 1, v116, vcc
s_sub_u32 s50, 0, s68
s_sub_u32 s51, 0, s67
v_mul_u32_u24_e64 v120, v116, 32
v_cvt_f32_u32_e32 v121, s50
v_rcp_f32_e32 v121, v121
v_mul_f32_e32 v121, 0x4f800000, v121
v_cvt_u32_f32_e32 v121, v121
v_mul_lo_u32 v122, s50, v121
v_mul_hi_u32 v123, s50, v121
v_sub_u32_e32 v124, vcc, 0, v122
v_cmp_ne_i32_e64 s[46:47], 0, v123
v_cndmask_b32_e64 v122, v124, v122, s[46:47]
v_mul_hi_u32 v122, v122, v121
v_sub_u32_e32 v123, vcc, v121, v122
v_add_u32_e32 v121, vcc, v121, v122
v_cndmask_b32_e64 v121, v121, v123, s[46:47]
v_mul_hi_u32 v121, v121, v120
v_mul_lo_u32 v122, v121, s50
v_sub_u32_e32 v123, vcc, v120, v122
v_cmp_ge_u32_e64 s[46:47], v120, v122
v_cmp_ge_u32_e64 s[48:49], v123, s50
v_add_u32_e32 v123, vcc, 1, v121
s_and_b64 s[48:49], s[46:47], s[48:49]
v_add_u32_e32 v122, vcc, -1, v121
v_cndmask_b32_e64 v123, v121, v123, s[48:49]
v_cndmask_b32_e64 v123, v122, v123, s[46:47]
v_cmp_ne_i32_e64 vcc, 0, s50
v_cndmask_b32_e32 v118, -1, v123, vcc
v_mad_i32_i24 v119, v118, s68, v120
v_mul_u32_u24_e64 v120, v118, 1
v_cvt_f32_u32_e32 v121, s51
v_rcp_f32_e32 v121, v121
v_mul_f32_e32 v121, 0x4f800000, v121
v_cvt_u32_f32_e32 v121, v121
v_mul_lo_u32 v122, s51, v121
v_mul_hi_u32 v123, s51, v121
v_sub_u32_e32 v124, vcc, 0, v122
v_cmp_ne_i32_e64 s[46:47], 0, v123
v_cndmask_b32_e64 v122, v124, v122, s[46:47]
v_mul_hi_u32 v122, v122, v121
v_sub_u32_e32 v123, vcc, v121, v122
v_add_u32_e32 v121, vcc, v121, v122
v_cndmask_b32_e64 v121, v121, v123, s[46:47]
v_mul_hi_u32 v121, v121, v120
v_mul_lo_u32 v122, v121, s51
v_sub_u32_e32 v123, vcc, v120, v122
v_cmp_ge_u32_e64 s[46:47], v120, v122
v_cmp_ge_u32_e64 s[48:49], v123, s51
v_add_u32_e32 v123, vcc, 1, v121
s_and_b64 s[48:49], s[46:47], s[48:49]
v_add_u32_e32 v122, vcc, -1, v121
v_cndmask_b32_e64 v123, v121, v123, s[48:49]
v_cndmask_b32_e64 v123, v122, v123, s[46:47]
v_cmp_ne_i32_e64 vcc, 0, s51
v_cndmask_b32_e32 v118, -1, v123, vcc
v_mad_i32_i24 v120, v118, s67, v120
v_readfirstlane_b32 s69, v119
v_readfirstlane_b32 s70, v120
v_readfirstlane_b32 s71, v118
v_add_u32_e32 v106, vcc, s69, v106
v_addc_u32_e32 v121, vcc, 0, v113, vcc
v_mad_i32_i24 v106, v121, s68, v106
v_mad_i32_i24 v108, v121, s73, v108
v_mad_i32_i24 v107, v121, s72, v107
v_cmp_ge_i32_e64 vcc, v107, 0
v_addc_u32_e32 v121, vcc, 0, v113, vcc
v_add_u32_e32 v108, vcc, v108, v121
v_mad_i32_i24 v107, v121, s67, v107
v_add_u32_e32 v107, vcc, s70, v107
v_addc_u32_e32 v121, vcc, 0, v113, vcc
v_add_u32_e32 v108, vcc, v108, v121
v_mad_i32_i24 v107, v121, s67, v107
v_add_u32_e32 v108, vcc, s71, v108
v_readlane_b32 s69, v119, 1
v_readlane_b32 s70, v120, 1
v_readlane_b32 s71, v118, 1
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
s_cbranch_scc1 65241
v_add_u32_e32 v106, vcc, s69, v106
v_addc_u32_e32 v115, vcc, 0, v113, vcc
v_mad_i32_i24 v106, v115, s68, v106
v_mad_i32_i24 v108, v115, s73, v108
v_mad_i32_i24 v107, v115, s72, v107
v_cmp_ge_i32_e64 vcc, v107, 0
v_addc_u32_e32 v115, vcc, 0, v113, vcc
v_add_u32_e32 v108, vcc, v108, v115
v_mad_i32_i24 v107, v115, s67, v107
v_add_u32_e32 v107, vcc, s70, v107
v_addc_u32_e32 v115, vcc, 0, v113, vcc
v_add_u32_e32 v108, vcc, v108, v115
v_mad_i32_i24 v107, v115, s67, v107
v_add_u32_e32 v108, vcc, s71, v108
v_readfirstlane_b32 s74, v108
v_sub_u32_e64 v115, vcc, v108, s74
v_mul_lo_u32 v117, v115, s56
v_cmp_ge_u32_e64 s[48:49], v108, s12
s_lshl_b32 s52, s15, 2
v_subrev_u32_e32 v116, vcc, s68, v106
v_and_b32_e64 v118, v0, 3
v_subrev_u32_e32 v118, vcc, 1, v118
v_mad_i32_i24 v116, v116, 2, v118
v_cmp_ge_u32_e64 s[44:45], v116, s15
v_mad_u32_u24 v116, 4, v116, v117
s_or_b64 s[44:45], s[44:45], s[48:49]
v_subrev_u32_e32 v117, vcc, s67, v107
v_mad_i32_i24 v117, v117, 2, -1
v_cmp_ge_u32_e64 s[46:47], v117, s14
s_or_b64 s[50:51], s[44:45], s[46:47]
v_mad_u32_u24 v90, v117, s52, v116
v_cndmask_b32_e64 v90, v90, -1, s[50:51]
v_add_u32_e32 v117, vcc, 1, v117
v_cmp_ge_u32_e64 s[46:47], v117, s14
s_or_b64 s[50:51], s[44:45], s[46:47]
v_mad_u32_u24 v91, v117, s52, v116
v_cndmask_b32_e64 v91, v91, -1, s[50:51]
v_add_u32_e32 v117, vcc, 1, v117
v_cmp_ge_u32_e64 s[46:47], v117, s14
s_or_b64 s[50:51], s[44:45], s[46:47]
v_mad_u32_u24 v92, v117, s52, v116
v_cndmask_b32_e64 v92, v92, -1, s[50:51]
v_add_u32_e32 v117, vcc, 1, v117
v_cmp_ge_u32_e64 s[46:47], v117, s14
s_or_b64 s[50:51], s[44:45], s[46:47]
v_mad_u32_u24 v93, v117, s52, v116
v_cndmask_b32_e64 v93, v93, -1, s[50:51]
v_mov_b32_dpp v116, v91  quad_perm:[1,2,2,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v98, v92  quad_perm:[2,1,1,2] row_mask:0xf bank_mask:0xf
v_add_u32_e64 v117, vcc, v0, 2
v_bfe_u32 v117, v117, 2, 1
v_cmp_eq_u32_e64 vcc, v117, 1
v_cndmask_b32_e32 v98, v116, v98, vcc
v_mul_lo_u32 v115, v115, s58
v_add_u32_e32 v115, vcc, v98, v115
v_cmp_eq_u32_e64 vcc, v98, -1
v_cndmask_b32_e32 v98, v115, v98, vcc
v_and_b32_e32 v116, 3, v0
v_min_u32_e32 v116, 2, v116
v_bfe_u32 v117, v0, 2, 5
v_lshlrev_b32_e32 v116, 2, v116
v_mad_u32_u24 v118, s62, v117, v116
v_mad_u32_u24 v94, v118, 1, 0
v_mad_u32_u24 v95, v118, 1, 0
v_mad_u32_u24 v96, v118, 1, 24
v_mad_u32_u24 v97, v118, 1, 12
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
v_bfe_u32 v115, v0, 2, 5
v_add_u32_e64 v115, vcc, v115, s76
v_cmp_lt_u32_e64 vcc, v115, s16
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
s_add_u32 s54, s54, 56
s_addc_u32 s55, s55, 0
s_setpc_b64 s[54:55]
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
s_add_u32 s36, s36, s64
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 12
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
s_add_u32 s36, s36, s64
s_addc_u32 s37, s37, 0
s_sub_u32 s84, s84, 12
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
s_cbranch_scc1 1786
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
s_add_u32 s54, s54, 0x58
s_addc_u32 s55, s55, 0
s_setpc_b64 s[54:55]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_getpc_b64 s[54:55]
s_sub_u32 s54, s54, 44
s_subb_u32 s55, s55, 0
s_branch 65182
s_barrier
s_nop 0
ds_write_b32 v100, v70
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v66, v90, s[28:31], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_subrev_f32_e64 v78, v80, v78 div:2
s_nop 0
ds_write_b32 v101, v71
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_subrev_f32_e64 v81, v79, v81 div:2
ds_read_b128 v[42:45], v104 offset:17408
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v79, v80, v79 div:2
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v68, v92, s[28:31], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105 offset:16384
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_mad_f32 v80, v80, 1.0, -v79
s_nop 0
ds_read_b128 v[54:57], v105 offset:16640
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 65158
s_nop 0
ds_write_b32 v102, v72
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v67, v91, s[28:31], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_dpp v78, v78, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v103, v73
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v79, v79, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:17920
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v80, v80, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v69, v93, s[28:31], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:16896
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_mac_f32_dpp v81, v81, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[62:65], v105 offset:17152
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 65043
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_barrier
s_nop 0
ds_write_b32 v100, v74 offset:8192
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v70, v90, s[28:31], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_subrev_f32_e64 v82, v84, v82 div:2
s_nop 0
ds_write_b32 v101, v75 offset:8192
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_subrev_f32_e64 v85, v83, v85 div:2
ds_read_b128 v[42:45], v104 offset:1024
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v83, v84, v83 div:2
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v72, v92, s[28:31], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_mad_f32 v84, v84, 1.0, -v83
s_nop 0
ds_read_b128 v[54:57], v105 offset:256
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 65014
s_nop 0
ds_write_b32 v102, v76 offset:8192
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v71, v91, s[28:31], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_dpp v82, v82, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v103, v77 offset:8192
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v83, v83, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:1536
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v84, v84, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v73, v93, s[28:31], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:512
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_mac_f32_dpp v85, v85, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[62:65], v105 offset:768
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 64899
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_barrier
s_nop 0
ds_write_b32 v100, v78 offset:16384
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v74, v90, s[28:31], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_subrev_f32_e64 v86, v88, v86 div:2
s_nop 0
ds_write_b32 v101, v79 offset:16384
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_subrev_f32_e64 v89, v87, v89 div:2
ds_read_b128 v[42:45], v104 offset:9216
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v87, v88, v87 div:2
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v76, v92, s[28:31], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105 offset:8192
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_mad_f32 v88, v88, 1.0, -v87
s_nop 0
ds_read_b128 v[54:57], v105 offset:8448
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 64870
s_nop 0
ds_write_b32 v102, v80 offset:16384
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v75, v91, s[28:31], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_dpp v86, v86, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v103, v81 offset:16384
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v87, v87, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:9728
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v88, v88, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v77, v93, s[28:31], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:8704
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_mac_f32_dpp v89, v89, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[62:65], v105 offset:8960
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 64755
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_barrier
s_nop 0
ds_write_b32 v100, v82
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v78, v90, s[28:31], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_subrev_f32_e64 v66, v68, v66 div:2
s_nop 0
ds_write_b32 v101, v83
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_subrev_f32_e64 v69, v67, v69 div:2
ds_read_b128 v[42:45], v104 offset:17408
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v67, v68, v67 div:2
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v80, v92, s[28:31], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105 offset:16384
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_mad_f32 v68, v68, 1.0, -v67
s_nop 0
ds_read_b128 v[54:57], v105 offset:16640
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 64726
s_nop 0
ds_write_b32 v102, v84
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v79, v91, s[28:31], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_dpp v66, v66, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v103, v85
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v67, v67, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:17920
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v68, v68, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v81, v93, s[28:31], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:16896
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_mac_f32_dpp v69, v69, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[62:65], v105 offset:17152
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 64611
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_barrier
s_nop 0
ds_write_b32 v100, v86 offset:8192
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v82, v90, s[28:31], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_subrev_f32_e64 v70, v72, v70 div:2
s_nop 0
ds_write_b32 v101, v87 offset:8192
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_subrev_f32_e64 v73, v71, v73 div:2
ds_read_b128 v[42:45], v104 offset:1024
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v71, v72, v71 div:2
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v84, v92, s[28:31], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_mad_f32 v72, v72, 1.0, -v71
s_nop 0
ds_read_b128 v[54:57], v105 offset:256
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 64582
s_nop 0
ds_write_b32 v102, v88 offset:8192
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v83, v91, s[28:31], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_dpp v70, v70, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v103, v89 offset:8192
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v71, v71, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:1536
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v72, v72, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v85, v93, s[28:31], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:512
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_mac_f32_dpp v73, v73, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[62:65], v105 offset:768
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 64467
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_barrier
s_nop 0
ds_write_b32 v100, v66 offset:16384
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v86, v90, s[28:31], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_subrev_f32_e64 v74, v76, v74 div:2
s_nop 0
ds_write_b32 v101, v67 offset:16384
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_subrev_f32_e64 v77, v75, v77 div:2
ds_read_b128 v[42:45], v104 offset:9216
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v75, v76, v75 div:2
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v88, v92, s[28:31], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105 offset:8192
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_mad_f32 v76, v76, 1.0, -v75
s_nop 0
ds_read_b128 v[54:57], v105 offset:8448
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 64438
s_nop 0
ds_write_b32 v102, v68 offset:16384
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v87, v91, s[28:31], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_dpp v74, v74, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v103, v69 offset:16384
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v75, v75, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:9728
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v76, v76, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v89, v93, s[28:31], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:8704
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_mac_f32_dpp v77, v77, -v109  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[62:65], v105 offset:8960
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s28, s28, s63
s_addc_u32 s29, s29, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 64323
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
buffer_load_dword v114, off, s[0:3], 0
s_branch 64669
s_nop 0
s_getpc_b64 s[54:55]
s_sub_u32 s54, s54, 44
s_subb_u32 s55, s55, 0
s_branch 64310
s_nop 0
ds_write_b32 v100, v70
s_barrier
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v66, v94, s[32:35], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_add_f32_dpp v78, v79, v79  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v101, v71
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_mac_f32_dpp v78, v79, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[42:45], v104 offset:17408
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_dpp v79, v81, v81  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v68, v96, s[32:35], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105 offset:16384
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_mac_f32_dpp v79, v81, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[54:57], v105 offset:16640
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 64286
s_nop 0
ds_write_b32 v102, v72
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v67, v95, s[32:35], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_add_f32_dpp v81, v80, v80  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v103, v73
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v81, v80, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:17920
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_add_f32_e64 v80, v78, v81
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v69, v97, s[32:35], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:16896
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_add_f32_e64 v79, v79, v80 div:2
s_nop 0
ds_read_b128 v[62:65], v105 offset:17152
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 64171
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_write_b32 v100, v74 offset:8192
s_barrier
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v70, v94, s[32:35], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_add_f32_e64 v80, -v79, v80
s_nop 0
ds_write_b32 v101, v75 offset:8192
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_dpp v82, v83, v83  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
ds_read_b128 v[42:45], v104 offset:1024
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_mac_f32_dpp v82, v83, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v72, v96, s[32:35], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_add_f32_dpp v83, v85, v85  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[54:57], v105 offset:256
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 64142
s_nop 0
ds_write_b32 v102, v76 offset:8192
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v71, v95, s[32:35], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_dpp v83, v85, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v103, v77 offset:8192
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_dpp v85, v84, v84  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:1536
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v85, v84, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v73, v97, s[32:35], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:512
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_add_f32_e64 v84, v82, v85
s_nop 0
ds_read_b128 v[62:65], v105 offset:768
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 64027
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_write_b32 v100, v78 offset:16384
s_barrier
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v74, v94, s[32:35], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_add_f32_e64 v83, v83, v84 div:2
s_nop 0
ds_write_b32 v101, v79 offset:16384
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_e64 v84, -v83, v84
ds_read_b128 v[42:45], v104 offset:9216
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_dpp v86, v87, v87  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v76, v96, s[32:35], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105 offset:8192
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_mac_f32_dpp v86, v87, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[54:57], v105 offset:8448
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 63998
s_nop 0
ds_write_b32 v102, v80 offset:16384
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v75, v95, s[32:35], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_add_f32_dpp v87, v89, v89  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v103, v81 offset:16384
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v87, v89, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:9728
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_add_f32_dpp v89, v88, v88  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v77, v97, s[32:35], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:8704
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_mac_f32_dpp v89, v88, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[62:65], v105 offset:8960
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 63883
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_write_b32 v100, v82
s_barrier
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v78, v94, s[32:35], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_add_f32_e64 v88, v86, v89
s_nop 0
ds_write_b32 v101, v83
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_e64 v87, v87, v88 div:2
ds_read_b128 v[42:45], v104 offset:17408
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v88, -v87, v88
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v80, v96, s[32:35], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105 offset:16384
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_add_f32_dpp v66, v67, v67  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[54:57], v105 offset:16640
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 63854
s_nop 0
ds_write_b32 v102, v84
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v79, v95, s[32:35], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_mac_f32_dpp v66, v67, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v103, v85
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_dpp v67, v69, v69  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:17920
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v67, v69, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v81, v97, s[32:35], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:16896
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_add_f32_dpp v69, v68, v68  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[62:65], v105 offset:17152
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 63739
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_write_b32 v100, v86 offset:8192
s_barrier
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v82, v94, s[32:35], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_mac_f32_dpp v69, v68, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v101, v87 offset:8192
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_add_f32_e64 v68, v66, v69
ds_read_b128 v[42:45], v104 offset:1024
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v67, v67, v68 div:2
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v84, v96, s[32:35], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_add_f32_e64 v68, -v67, v68
s_nop 0
ds_read_b128 v[54:57], v105 offset:256
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 63710
s_nop 0
ds_write_b32 v102, v88 offset:8192
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v83, v95, s[32:35], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_add_f32_dpp v70, v71, v71  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v103, v89 offset:8192
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_mac_f32_dpp v70, v71, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:1536
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_add_f32_dpp v71, v73, v73  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v85, v97, s[32:35], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:512
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_mac_f32_dpp v71, v73, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[62:65], v105 offset:768
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 63595
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
ds_write_b32 v100, v66 offset:16384
s_barrier
v_mac_f32_e32 v2, v46, v58
v_mac_f32_e32 v3, v47, v58
v_mac_f32_e32 v4, v48, v58
v_mac_f32_e32 v5, v49, v58
s_nop 2
buffer_load_dword v86, v94, s[32:35], 0 offen
v_mac_f32_e32 v6, v46, v59
v_mac_f32_e32 v7, v47, v59
v_mac_f32_e32 v8, v48, v59
v_add_f32_dpp v73, v72, v72  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
ds_write_b32 v101, v67 offset:16384
v_mac_f32_e32 v9, v49, v59
v_mac_f32_e32 v10, v46, v60
v_mac_f32_e32 v11, v47, v60
v_mac_f32_dpp v73, v72, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
ds_read_b128 v[42:45], v104 offset:9216
s_nop 0
v_mac_f32_e32 v12, v48, v60
v_mac_f32_e32 v13, v49, v60
v_mac_f32_e32 v14, v46, v61
v_add_f32_e64 v72, v70, v73
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v88, v96, s[32:35], 0 offen
v_mac_f32_e32 v15, v47, v61
v_mac_f32_e32 v16, v48, v61
v_mac_f32_e32 v17, v49, v61
v_mac_f32_e32 v18, v46, v62
ds_read_b128 v[50:53], v105 offset:8192
s_nop 0
v_mac_f32_e32 v19, v47, v62
v_mac_f32_e32 v20, v48, v62
v_mac_f32_e32 v21, v49, v62
v_add_f32_e64 v71, v71, v72 div:2
s_nop 0
ds_read_b128 v[54:57], v105 offset:8448
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v46, v63
v_mac_f32_e32 v23, v47, v63
v_mac_f32_e32 v24, v48, v63
v_mac_f32_e32 v25, v49, v63
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v46, v64
v_mac_f32_e32 v27, v47, v64
v_mac_f32_e32 v28, v48, v64
v_mac_f32_e32 v29, v49, v64
s_sub_u32 s85, s85, 2
s_waitcnt vmcnt(8)
s_nop 0
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v46, v65
v_mac_f32_e32 v31, v47, v65
v_mac_f32_e32 v32, v48, v65
v_mac_f32_e32 v33, v49, v65
s_cbranch_scc1 63566
s_nop 0
ds_write_b32 v102, v68 offset:16384
v_mac_f32_e32 v2, v42, v50
v_mac_f32_e32 v3, v43, v50
v_mac_f32_e32 v4, v44, v50
v_mac_f32_e32 v5, v45, v50
s_nop 2
buffer_load_dword v87, v95, s[32:35], 0 offen
v_mac_f32_e32 v6, v42, v51
v_mac_f32_e32 v7, v43, v51
v_mac_f32_e32 v8, v44, v51
v_add_f32_e64 v72, -v71, v72
s_nop 0
ds_write_b32 v103, v69 offset:16384
v_mac_f32_e32 v9, v45, v51
v_mac_f32_e32 v10, v42, v52
v_mac_f32_e32 v11, v43, v52
v_add_f32_dpp v74, v75, v75  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
ds_read_b128 v[46:49], v104 offset:9728
s_nop 0
v_mac_f32_e32 v12, v44, v52
v_mac_f32_e32 v13, v45, v52
v_mac_f32_e32 v14, v42, v53
v_mac_f32_dpp v74, v75, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
s_nop 0
s_waitcnt lgkmcnt(3)
buffer_load_dword v89, v97, s[32:35], 0 offen
v_mac_f32_e32 v15, v43, v53
v_mac_f32_e32 v16, v44, v53
v_mac_f32_e32 v17, v45, v53
v_mac_f32_e32 v18, v42, v54
ds_read_b128 v[58:61], v105 offset:8704
s_nop 0
v_mac_f32_e32 v19, v43, v54
v_mac_f32_e32 v20, v44, v54
v_mac_f32_e32 v21, v45, v54
v_add_f32_dpp v75, v77, v77  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
s_nop 0
ds_read_b128 v[62:65], v105 offset:8960
s_getpc_b64 s[54:55]
v_mac_f32_e32 v22, v42, v55
v_mac_f32_e32 v23, v43, v55
v_mac_f32_e32 v24, v44, v55
v_mac_f32_e32 v25, v45, v55
s_add_u32 s32, s32, s65
s_addc_u32 s33, s33, 0
s_nop 0
s_nop 0
v_mac_f32_e32 v26, v42, v56
v_mac_f32_e32 v27, v43, v56
v_mac_f32_e32 v28, v44, v56
v_mac_f32_e32 v29, v45, v56
s_sub_u32 s75, s75, 2
s_waitcnt vmcnt(8)
s_cbranch_scc1 63451
s_waitcnt lgkmcnt(1)
v_mac_f32_e32 v30, v42, v57
v_mac_f32_e32 v31, v43, v57
v_mac_f32_e32 v32, v44, v57
v_mac_f32_e32 v33, v45, v57
s_nop 0
s_nop 0
v_mac_f32_dpp v75, v77, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v77, v76, v76  quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v77, v76, v109  quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e64 v76, v74, v77
v_add_f32_e64 v75, v75, v76 div:2
v_add_f32_e64 v76, -v75, v76
s_branch 64657
s_endpgm
s_branch 64655
s_branch 63782
