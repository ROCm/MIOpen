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

.macro _v_pk_fmac_f16_gfx10_quad_perm vdst:req, vsrc0:req, vsrc1:req, q0:req, q1:req, q2:req, q3:req
    .long  0x780000FA + ((\vdst << 17) + (\vsrc1 << 9))
    .long  0xFF000000 + (\vsrc0 + (\q0 << 8) + (\q1 << 10) + (\q2 << 12) + (\q3 << 14))
.endm

s_version 0x2004
s_inst_prefetch 0x3
v_mov_b32_e32 v1, v0
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_mov_b32_e32 v235, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s97, 0xc1e0
s_mov_b32 s96, 0xc1e0
v_and_b32_e32 v237, 0xc0, v0
v_add_co_u32_e64 v1, vcc, v0, v237
s_mov_b32 s91, 0
v_lshlrev_b32_e32 v238, 2, v1
v_add_co_u32_e64 v238, vcc, 0xffc0, v238
v_cmp_ge_u32_e32 vcc, 12, v1
s_cbranch_vccz 5
v_mov_b32_e32 v237, 0
v_cndmask_b32_e32 v238, -1, v238, vcc
ds_write_b32 v238, v237
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
s_barrier
v_readfirstlane_b32 s52, v1
s_lshr_b32 s52, s52, 5
s_add_u32 s52, s52, 8
s_and_b32 s92, s52, 20
s_mov_b64 s[40:41], s[6:7]
s_load_dwordx16 s[12:27], s[40:41], 0x0
s_load_dwordx4 s[28:31], s[40:41], 0x40
s_load_dwordx2 s[32:33], s[40:41], 0x50
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
s_load_dwordx2 s[34:35], s[40:41], 0x58
s_mov_b32 s36, 1.0
s_bitcmp1_b32 s18, 8
s_cbranch_scc0 2
s_load_dword s36, s[40:41], 0x60
s_bitcmp1_b32 s18, 7
s_cbranch_scc0 7
s_bitcmp1_b32 s18, 6
s_cbranch_scc0 5
s_waitcnt lgkmcnt(0)
s_and_b32 s35, s35, 0xffff
s_load_dwordx2 s[34:35], s[34:35], 0x0
s_bitcmp1_b32 s18, 9
s_cbranch_scc0 72
s_mov_b32 s42, 0x8c
s_mov_b32 s43, 0x9c
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cmp_eq_u32 1, src_vccz
s_cselect_b32 s42, s43, s42
s_load_dword s65, s[40:41], 0x88
s_load_dword s90, s[40:41], 0x98
s_load_dword s68, s[40:41], s42
s_load_dwordx2 s[66:67], s[40:41], 0xa8
s_bitcmp1_b32 s18, 10
s_cbranch_scc0 103
s_load_dwordx4 s[44:47], s[40:41], 0xb8
v_ffbh_u32_e32 v4, s17
v_lshlrev_b32_e64 v5, v4, s17
v_and_b32_e32 v6, 0xffffff00, v5
v_cmp_eq_u32_e32 vcc, 0x80000000, v5
v_cvt_f32_u32_e32 v6, v6
v_rcp_f32_e32 v2, v6
v_sub_co_ci_u32_e32 v3, vcc, 32, v4, vcc
v_cvt_f32_ubyte0_e32 v4, v5
v_fma_f32 v6, v6, v2, -1.0
v_fma_f32 v6, v4, v2, v6
v_fmaak_f32 v6, v6, v2, 0x9f000000
v_mul_f32_e32 v6, 0x5f800000, v6
v_mov_b32_e32 v4, 0
v_cvt_flr_i32_f32_e64 v6, -v6
v_lshl_add_u32 v2, v2, 9, v6
v_mad_u64_u32 v[4:5], vcc, v5, v2, v[4:5]
v_sub_co_ci_u32_e64 v2, vcc, v2, -1, vcc
v_mul_hi_u32 v4, s8, v2
v_add_co_u32_e64 v2, vcc, v4, s8
v_add_co_ci_u32_e64 v4, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v3
v_cndmask_b32_e32 v2, v2, v4, vcc
v_alignbit_b32 v2, v4, v2, v3
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s48, v2
s_mul_i32 s49, s48, s17
s_sub_u32 s8, s8, s49
s_mul_i32 s49, s45, s48
s_add_u32 s20, s20, s49
s_addc_u32 s21, s21, 0
s_mul_i32 s49, s46, s48
s_add_u32 s22, s22, s49
s_addc_u32 s23, s23, 0
s_mul_i32 s49, s47, s48
s_add_u32 s24, s24, s49
s_addc_u32 s25, s25, 0
s_branch 49
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
v_cmp_le_u32_e32 vcc, 0x100, v1
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
s_add_u32 s42, s44, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s42
v_readfirstlane_b32 s42, v2
s_andn2_b32 s44, 0, s31
s_addc_u32 s44, s33, 0
s_ashr_i32 s44, s44, 0
s_add_u32 s43, s44, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s43
v_readfirstlane_b32 s43, v2
s_sub_u32 s75, 0, s43
s_sub_u32 s74, 0, s42
s_add_u32 s60, s28, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s60
v_readfirstlane_b32 s60, v2
s_add_u32 s61, s29, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s61
v_readfirstlane_b32 s61, v2
v_mad_i32_i24 v2, 2, s60, -1
v_sub_co_u32_e64 v2, vcc, v2, s28
v_add_co_ci_u32_e64 v2, vcc, 0, 0, vcc
v_readfirstlane_b32 s44, v2
s_and_b32 s44, s44, 0
s_and_b32 s44, s44, s60
s_add_u32 s60, s60, s44
v_readfirstlane_b32 s45, v1
s_and_b32 s48, s45, 64
s_cselect_b32 s48, 0x80000, 0
s_or_b32 s18, s18, s48
s_lshl_b32 s69, s68, 1
s_sub_u32 s70, 0, s69
s_subb_u32 s71, 0, 0
s_bitcmp1_b32 s18, 12
s_cselect_b32 s44, 0, -1
s_bitcmp1_b32 s18, 11
s_cselect_b32 s44, s44, 1
s_cmp_gt_u32 s61, s44
s_cbranch_scc0 8
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
s_add_u32 s69, s69, s69
s_addc_u32 s70, s70, 0
v_bfe_u32 v3, v1, 2, 6
v_lshrrev_b32_e32 v230, 1, v3
s_bitcmp0_b32 s45, 8
s_cselect_b32 s48, 0x1000000, 0
s_or_b32 s48, s48, 0x100000
s_and_b32 s48, s18, s48
s_cselect_b32 s48, 0, 15
v_bfi_b32 v230, s48, v3, v230
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
v_and_b32_e32 v6, 3, v1
v_cmp_eq_u32_e32 vcc, 2, v6
v_cndmask_b32_e32 v4, v4, v5, vcc
v_cmp_eq_u32_e32 vcc, 1, v6
v_cndmask_b32_e32 v7, 0, v230, vcc
s_bitcmp1_b32 s18, 20
s_cbranch_scc0 4
v_add_co_u32_e64 v5, vcc, v230, 8
v_cmp_eq_u32_e32 vcc, 0, v6
v_cndmask_b32_e32 v7, v7, v5, vcc
v_cmp_eq_u32_e64 s[46:47], 3, v6
v_bfe_u32 v228, v7, 0, 5
v_mad_u32_u24 v228, v4, 32, v228
v_ffbh_u32_e32 v9, s43
v_lshlrev_b32_e64 v10, v9, s43
v_and_b32_e32 v11, 0xffffff00, v10
v_cmp_eq_u32_e32 vcc, 0x80000000, v10
v_cvt_f32_u32_e32 v11, v11
v_rcp_f32_e32 v229, v11
v_sub_co_ci_u32_e32 v8, vcc, 32, v9, vcc
v_cvt_f32_ubyte0_e32 v9, v10
v_fma_f32 v11, v11, v229, -1.0
v_fma_f32 v11, v9, v229, v11
v_fmaak_f32 v11, v11, v229, 0x9f000000
v_mul_f32_e32 v11, 0x5f800000, v11
v_mov_b32_e32 v9, 0
v_cvt_flr_i32_f32_e64 v11, -v11
v_lshl_add_u32 v229, v229, 9, v11
v_mad_u64_u32 v[9:10], vcc, v10, v229, v[9:10]
v_sub_co_ci_u32_e64 v229, vcc, v229, -1, vcc
v_mul_hi_u32 v9, v228, v229
v_add_co_u32_e64 v229, vcc, v9, v228
v_add_co_ci_u32_e64 v9, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v8
v_cndmask_b32_e32 v229, v229, v9, vcc
v_alignbit_b32 v229, v9, v229, v8
v_mad_i32_i24 v227, v229, s75, v228
v_lshrrev_b32_e32 v228, 5, v7
v_mad_u32_u24 v228, v229, 1, v228
v_cndmask_b32_e64 v228, v228, 1, s[46:47]
v_ffbh_u32_e32 v9, s42
v_lshlrev_b32_e64 v10, v9, s42
v_and_b32_e32 v11, 0xffffff00, v10
v_cmp_eq_u32_e32 vcc, 0x80000000, v10
v_cvt_f32_u32_e32 v11, v11
v_rcp_f32_e32 v229, v11
v_sub_co_ci_u32_e32 v8, vcc, 32, v9, vcc
v_cvt_f32_ubyte0_e32 v9, v10
v_fma_f32 v11, v11, v229, -1.0
v_fma_f32 v11, v9, v229, v11
v_fmaak_f32 v11, v11, v229, 0x9f000000
v_mul_f32_e32 v11, 0x5f800000, v11
v_mov_b32_e32 v9, 0
v_cvt_flr_i32_f32_e64 v11, -v11
v_lshl_add_u32 v229, v229, 9, v11
v_mad_u64_u32 v[9:10], vcc, v10, v229, v[9:10]
v_sub_co_ci_u32_e64 v229, vcc, v229, -1, vcc
v_mul_hi_u32 v9, v228, v229
v_add_co_u32_e64 v229, vcc, v9, v228
v_add_co_ci_u32_e64 v9, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v8
v_cndmask_b32_e32 v229, v229, v9, vcc
v_alignbit_b32 v229, v9, v229, v8
v_mad_i32_i24 v228, v229, s74, v228
v_readlane_b32 s76, v227, 2
v_readlane_b32 s77, v228, 2
v_readlane_b32 s78, v229, 2
v_readlane_b32 s79, v228, 3
v_readlane_b32 s80, v229, 3
v_add_co_u32_e64 v227, vcc, v227, s75
v_add_co_u32_e64 v228, vcc, v228, s74
v_mov_b32_dpp v229, v229  quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v227, v227  quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v228, v228  quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x31014000
s_mov_b32 s46, 0x80000000
s_mov_b32 s47, 0x31014000
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 8
v_xor_b32_dpp v231, v1, v1  quad_perm:[2,3,2,1] row_mask:0xf bank_mask:0xf
v_subrev_co_u32_e64 v231, vcc, 1, v231
v_cvt_f16_i16_e32 v231, v231
v_pk_add_f16 v231, v231, 0 op_sel_hi:[0,0]
s_branch 7
v_xor_b32_dpp v231, v1, v1  quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32_e64 v231, vcc, 1, v231
v_cvt_f16_i16_e32 v231, v231
v_pk_add_f16 v231, v231, 0 op_sel_hi:[0,0]
v_mov_b32_e32 v232, 1
v_xor_b32_dpp v232, v1, v1  quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v232, v1, v1  quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32_e64 v232, vcc, 1, v232
v_mov_b32_e32 v233, 1
v_xor_b32_dpp v233, v1, v1  quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v233, v1, v1  quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32_e64 v233, vcc, 1, v233
v_cvt_f32_i32_e32 v232, v232
v_cvt_f32_i32_e32 v233, v233
v_lshrrev_b32_e64 v237, 2, s92
v_and_b32_e32 v238, 3, v1
v_bfe_u32 v239, v1, 4, 3
v_mad_u32_u24 v223, v239, 4, v238
v_lshlrev_b32_e32 v223, 4, v223
v_mad_u32_u24 v213, v237, 4, v238
v_lshlrev_b32_e32 v213, 4, v213
v_bfe_u32 v237, v1, 2, 2
v_and_b32_e32 v238, 1, v237
v_mad_u32_u24 v240, v237, 16, v238
v_lshlrev_b32_e32 v240, 6, v240
v_xor_b32_e32 v213, v213, v240
v_mul_u32_u24_e32 v240, 0x400, v237
v_xor_b32_e32 v223, v223, v240
s_lshr_b32 s92, s92, 0
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 61
s_and_b32 s53, s18, 0x1100000
s_addc_u32 s53, 0, 0
v_lshrrev_b32_e32 v240, 1, v1
s_mul_i32 s52, 60, s53
s_sub_u32 s52, 63, s52
v_bfi_b32 v240, s52, v1, v240
v_and_b32_e32 v237, 1, v240
v_bfe_u32 v238, v240, 1, 1
v_xor_b32_e32 v237, v237, v238
v_bfe_u32 v239, v240, 3, 1
v_mad_u32_u24 v238, v238, 2, v239
v_mul_u32_u24_e32 v237, 0x118, v237
v_bfe_u32 v239, v240, 2, 1
v_mad_u32_u24 v238, v238, 2, v237
v_xor_b32_e32 v238, v238, v239
v_and_b32_e32 v239, 0xf0, v240
v_xor_b32_e32 v238, v238, v239
s_mul_i32 s52, 4, s53
s_sub_u32 s52, 6, s52
v_bfe_u32 v240, v1, s52, 1
v_mul_u32_u24_e32 v240, 0x1040, v240
v_xor_b32_e32 v216, 0x314, v238
v_xor_b32_e32 v217, 0x31c, v238
v_xor_b32_e32 v218, 8, v238
v_mov_b32_e32 v215, v238
v_mad_u32_u24 v215, 4, v215, v240
v_mad_u32_u24 v216, 4, v216, v240
v_mad_u32_u24 v217, 4, v217, v240
v_mad_u32_u24 v218, 4, v218, v240
s_mov_b32 s52, 0x1040
s_and_b32 s53, s18, 0x1100000
s_cselect_b32 s52, 0x80, s52
v_add_co_u32_e64 v219, vcc, v215, s52
v_add_co_u32_e64 v220, vcc, v216, s52
v_add_co_u32_e64 v221, vcc, v217, s52
v_add_co_u32_e64 v222, vcc, v218, s52
s_branch 57
s_bfe_u32 s53, s18, 0x10014
v_lshrrev_b32_e32 v240, 1, v1
s_mul_i32 s52, 60, s53
s_sub_u32 s52, 63, s52
v_bfi_b32 v240, s52, v1, v240
v_and_b32_e32 v237, 1, v240
v_bfe_u32 v238, v240, 1, 1
v_bfe_u32 v239, v240, 3, 1
v_xor_b32_e32 v237, v237, v238
v_mad_u32_u24 v238, v238, 2, v239
v_mul_u32_u24_e32 v237, 0x109, v237
v_bfe_u32 v239, v240, 2, 1
v_mad_u32_u24 v238, v238, 2, v237
v_xor_b32_e32 v238, v238, v239
v_and_b32_e32 v239, 0xf0, v240
v_or_b32_e32 v238, v238, v239
s_mul_i32 s52, 4, s53
s_sub_u32 s52, 6, s52
v_bfe_u32 v240, v1, s52, 1
v_mul_u32_u24_e32 v240, 0x1040, v240
v_mad_u32_u24 v215, 4, v238, v240
v_xor_b32_e32 v216, 0x307, v238
v_mad_u32_u24 v216, 4, v216, v240
v_xor_b32_e32 v217, 0x30f, v238
v_mad_u32_u24 v217, 4, v217, v240
v_xor_b32_e32 v218, 8, v238
v_mad_u32_u24 v218, 4, v218, v240
s_mov_b32 s52, 0x1040
s_bitcmp1_b32 s18, 20
s_cselect_b32 s52, 0x80, s52
v_add_co_u32_e64 v219, vcc, v215, s52
v_add_co_u32_e64 v220, vcc, v216, s52
v_add_co_u32_e64 v221, vcc, v217, s52
v_add_co_u32_e64 v222, vcc, v218, s52
v_subrev_co_u32_e64 v227, vcc, s76, v227
v_mov_b32_e32 v238, s75
v_cmp_lt_i32_e32 vcc, v227, v238
v_sub_co_ci_u32_e64 v237, vcc, 0, 0, vcc
v_mad_i32_i24 v227, v237, s75, v227
v_mad_i32_i24 v229, v237, s80, v229
v_mad_i32_i24 v228, v237, s79, v228
v_mov_b32_e32 v238, s74
v_cmp_lt_i32_e32 vcc, v228, v238
v_sub_co_ci_u32_e64 v237, vcc, 0, 0, vcc
v_add_co_u32_e64 v229, vcc, v229, v237
v_mad_i32_i24 v228, v237, v238, v228
v_subrev_co_u32_e64 v228, vcc, s77, v228
v_cmp_lt_i32_e32 vcc, v228, v238
v_sub_co_ci_u32_e64 v237, vcc, 0, 0, vcc
v_add_co_u32_e64 v229, vcc, v229, v237
v_mad_i32_i24 v228, v237, s74, v228
v_subrev_co_u32_e64 v229, vcc, s78, v229
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
s_mov_b32 s94, 38
s_mov_b32 s82, 0
s_bitset1_b32 s18, 26
s_call_b64 s[38:39], 4872
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 65
s_branch 2564
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v113, v161, v113
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v114, v162, v114
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v115, v163, v115
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v116, v164, v116
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
v_pk_fma_f16 v113, v115, -1.0, v113 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
v_pk_mul_f16 v113, v113, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_fma_f16 v116, v114, -1.0, v116 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v116, v116, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_add_f16 v114, v115, v114
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v114, v114, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_pk_fma_f16 v115, v114, -1.0, v115 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
_v_pk_fmac_f16_gfx10_quad_perm 113, 113, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
_v_pk_fmac_f16_gfx10_quad_perm 114, 114, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
_v_pk_fmac_f16_gfx10_quad_perm 115, 115, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
_v_pk_fmac_f16_gfx10_quad_perm 116, 116, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v105
ds_read_b128 v[69:72], v223 offset:29440
ds_write_b32 v220, v106
ds_read_b128 v[73:76], v223 offset:29696
ds_write_b32 v221, v107
ds_read_b128 v[85:88], v213 offset:28928
ds_write_b32 v222, v108
ds_read_b128 v[89:92], v213 offset:29056
s_setprio 0
s_add_u32 s91, s91, 0x100
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v101, v197, s[40:43], 0 offen
buffer_load_short_d16 v103, v199, s[40:43], 0 offen
buffer_load_short_d16 v102, v198, s[40:43], 0 offen
buffer_load_short_d16 v104, v200, s[40:43], 0 offen
buffer_load_short_d16 v149, v197, s[52:55], 0 offen
buffer_load_short_d16 v151, v199, s[52:55], 0 offen
buffer_load_short_d16 v150, v198, s[52:55], 0 offen
buffer_load_short_d16 v152, v200, s[52:55], 0 offen
s_waitcnt vmcnt(56) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 3
s_call_b64 s[38:39], 4594
s_nop 0
s_nop 0
v_pack_b32_f16 v117, v165, v117
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v118, v166, v118
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v119, v167, v119
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v120, v168, v120
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
v_pk_fma_f16 v117, v119, -1.0, v117 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
v_pk_mul_f16 v117, v117, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_fma_f16 v120, v118, -1.0, v120 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v120, v120, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_add_f16 v118, v119, v118
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v118, v118, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_pk_fma_f16 v119, v118, -1.0, v119 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
_v_pk_fmac_f16_gfx10_quad_perm 117, 117, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
_v_pk_fmac_f16_gfx10_quad_perm 118, 118, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
_v_pk_fmac_f16_gfx10_quad_perm 119, 119, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
_v_pk_fmac_f16_gfx10_quad_perm 120, 120, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
ds_write_b32 v215, v109 offset:8256
ds_read_b128 v[77:80], v223 offset:33536
ds_write_b32 v216, v110 offset:8256
ds_read_b128 v[81:84], v223 offset:33792
ds_write_b32 v217, v111 offset:8256
ds_read_b128 v[93:96], v213 offset:33024
ds_write_b32 v218, v112 offset:8256
ds_read_b128 v[97:100], v213 offset:33152
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v105, v201, s[40:43], 0 offen
buffer_load_short_d16 v107, v203, s[40:43], 0 offen
buffer_load_short_d16 v106, v202, s[40:43], 0 offen
buffer_load_short_d16 v108, v204, s[40:43], 0 offen
buffer_load_short_d16 v153, v201, s[52:55], 0 offen
buffer_load_short_d16 v155, v203, s[52:55], 0 offen
buffer_load_short_d16 v154, v202, s[52:55], 0 offen
buffer_load_short_d16 v156, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v236 offset:65472
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 3
s_call_b64 s[38:39], 4386
s_nop 0
s_nop 0
v_pack_b32_f16 v121, v169, v121
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v122, v170, v122
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v123, v171, v123
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v124, v172, v124
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
v_pk_fma_f16 v121, v123, -1.0, v121 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
v_pk_mul_f16 v121, v121, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_fma_f16 v124, v122, -1.0, v124 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v124, v124, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_add_f16 v122, v123, v122
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v122, v122, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_pk_fma_f16 v123, v122, -1.0, v123 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
_v_pk_fmac_f16_gfx10_quad_perm 121, 121, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
_v_pk_fmac_f16_gfx10_quad_perm 122, 122, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
_v_pk_fmac_f16_gfx10_quad_perm 123, 123, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
_v_pk_fmac_f16_gfx10_quad_perm 124, 124, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v113 offset:8256
ds_read_b128 v[69:72], v223 offset:37696
ds_write_b32 v220, v114 offset:8256
ds_read_b128 v[73:76], v223 offset:37952
ds_write_b32 v221, v115 offset:8256
ds_read_b128 v[85:88], v213 offset:37184
ds_write_b32 v222, v116 offset:8256
ds_read_b128 v[89:92], v213 offset:37312
s_setprio 0
s_mov_b32 m0, 0x2ffc0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v109, v197, s[40:43], 0 offen
buffer_load_short_d16 v111, v199, s[40:43], 0 offen
buffer_load_short_d16 v110, v198, s[40:43], 0 offen
buffer_load_short_d16 v112, v200, s[40:43], 0 offen
buffer_load_short_d16 v157, v197, s[52:55], 0 offen
buffer_load_short_d16 v159, v199, s[52:55], 0 offen
buffer_load_short_d16 v158, v198, s[52:55], 0 offen
buffer_load_short_d16 v160, v200, s[52:55], 0 offen
s_waitcnt vmcnt(56) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 3
s_call_b64 s[38:39], 4178
s_nop 0
s_nop 0
v_pack_b32_f16 v125, v173, v125
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v126, v174, v126
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v127, v175, v127
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v128, v176, v128
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
v_pk_fma_f16 v125, v127, -1.0, v125 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
v_pk_mul_f16 v125, v125, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_fma_f16 v128, v126, -1.0, v128 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v128, v128, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_add_f16 v126, v127, v126
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v126, v126, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_pk_fma_f16 v127, v126, -1.0, v127 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
_v_pk_fmac_f16_gfx10_quad_perm 125, 125, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
_v_pk_fmac_f16_gfx10_quad_perm 126, 126, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
_v_pk_fmac_f16_gfx10_quad_perm 127, 127, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
_v_pk_fmac_f16_gfx10_quad_perm 128, 128, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v215, v117 offset:16512
ds_read_b128 v[77:80], v223 offset:41792
ds_write_b32 v216, v118 offset:16512
ds_read_b128 v[81:84], v223 offset:42048
ds_write_b32 v217, v119 offset:16512
ds_read_b128 v[93:96], v213 offset:41280
ds_write_b32 v218, v120 offset:16512
ds_read_b128 v[97:100], v213 offset:41408
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v113, v201, s[40:43], 0 offen
buffer_load_short_d16 v115, v203, s[40:43], 0 offen
buffer_load_short_d16 v114, v202, s[40:43], 0 offen
buffer_load_short_d16 v116, v204, s[40:43], 0 offen
buffer_load_short_d16 v161, v201, s[52:55], 0 offen
buffer_load_short_d16 v163, v203, s[52:55], 0 offen
buffer_load_short_d16 v162, v202, s[52:55], 0 offen
buffer_load_short_d16 v164, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 2
s_call_b64 s[38:39], 3969
s_nop 0
v_pack_b32_f16 v129, v177, v129
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v130, v178, v130
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v131, v179, v131
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v132, v180, v132
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
v_pk_fma_f16 v129, v131, -1.0, v129 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
v_pk_mul_f16 v129, v129, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_fma_f16 v132, v130, -1.0, v132 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v132, v132, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_add_f16 v130, v131, v130
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v130, v130, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_pk_fma_f16 v131, v130, -1.0, v131 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
_v_pk_fmac_f16_gfx10_quad_perm 129, 129, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
_v_pk_fmac_f16_gfx10_quad_perm 130, 130, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
_v_pk_fmac_f16_gfx10_quad_perm 131, 131, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
_v_pk_fmac_f16_gfx10_quad_perm 132, 132, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v121 offset:16512
ds_read_b128 v[69:72], v223 offset:45952
ds_write_b32 v220, v122 offset:16512
ds_read_b128 v[73:76], v223 offset:46208
ds_write_b32 v221, v123 offset:16512
ds_read_b128 v[85:88], v213 offset:45440
ds_write_b32 v222, v124 offset:16512
ds_read_b128 v[89:92], v213 offset:45568
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v117, v197, s[40:43], 0 offen
buffer_load_short_d16 v119, v199, s[40:43], 0 offen
buffer_load_short_d16 v118, v198, s[40:43], 0 offen
buffer_load_short_d16 v120, v200, s[40:43], 0 offen
buffer_load_short_d16 v165, v197, s[52:55], 0 offen
buffer_load_short_d16 v167, v199, s[52:55], 0 offen
buffer_load_short_d16 v166, v198, s[52:55], 0 offen
buffer_load_short_d16 v168, v200, s[52:55], 0 offen
s_waitcnt vmcnt(56) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 5
s_call_b64 s[38:39], 3764
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v133, v181, v133
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v134, v182, v134
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v135, v183, v135
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v136, v184, v136
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
v_pk_fma_f16 v133, v135, -1.0, v133 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
v_pk_mul_f16 v133, v133, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_fma_f16 v136, v134, -1.0, v136 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v136, v136, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_add_f16 v134, v135, v134
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v134, v134, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_pk_fma_f16 v135, v134, -1.0, v135 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
_v_pk_fmac_f16_gfx10_quad_perm 133, 133, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
_v_pk_fmac_f16_gfx10_quad_perm 134, 134, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
_v_pk_fmac_f16_gfx10_quad_perm 135, 135, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
_v_pk_fmac_f16_gfx10_quad_perm 136, 136, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
ds_write_b32 v215, v125 offset:24768
ds_read_b128 v[77:80], v223 offset:512
ds_write_b32 v216, v126 offset:24768
ds_read_b128 v[81:84], v223 offset:768
ds_write_b32 v217, v127 offset:24768
ds_read_b128 v[93:96], v213
ds_write_b32 v218, v128 offset:24768
ds_read_b128 v[97:100], v213 offset:128
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v121, v201, s[40:43], 0 offen
buffer_load_short_d16 v123, v203, s[40:43], 0 offen
buffer_load_short_d16 v122, v202, s[40:43], 0 offen
buffer_load_short_d16 v124, v204, s[40:43], 0 offen
buffer_load_short_d16 v169, v201, s[52:55], 0 offen
buffer_load_short_d16 v171, v203, s[52:55], 0 offen
buffer_load_short_d16 v170, v202, s[52:55], 0 offen
buffer_load_short_d16 v172, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v236 offset:65476
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 3
s_call_b64 s[38:39], 3554
s_nop 0
s_nop 0
v_pack_b32_f16 v137, v185, v137
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v138, v186, v138
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v139, v187, v139
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v140, v188, v140
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
v_pk_fma_f16 v137, v139, -1.0, v137 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
v_pk_mul_f16 v137, v137, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_fma_f16 v140, v138, -1.0, v140 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v140, v140, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_add_f16 v138, v139, v138
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v138, v138, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_pk_fma_f16 v139, v138, -1.0, v139 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
_v_pk_fmac_f16_gfx10_quad_perm 137, 137, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
_v_pk_fmac_f16_gfx10_quad_perm 138, 138, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
_v_pk_fmac_f16_gfx10_quad_perm 139, 139, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
_v_pk_fmac_f16_gfx10_quad_perm 140, 140, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v129 offset:24768
ds_read_b128 v[69:72], v223 offset:4672
ds_write_b32 v220, v130 offset:24768
ds_read_b128 v[73:76], v223 offset:4928
ds_write_b32 v221, v131 offset:24768
ds_read_b128 v[85:88], v213 offset:4160
ds_write_b32 v222, v132 offset:24768
ds_read_b128 v[89:92], v213 offset:4288
s_setprio 0
s_mov_b32 m0, 0x2ffc4
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v125, v197, s[40:43], 0 offen
buffer_load_short_d16 v127, v199, s[40:43], 0 offen
buffer_load_short_d16 v126, v198, s[40:43], 0 offen
buffer_load_short_d16 v128, v200, s[40:43], 0 offen
buffer_load_short_d16 v173, v197, s[52:55], 0 offen
buffer_load_short_d16 v175, v199, s[52:55], 0 offen
buffer_load_short_d16 v174, v198, s[52:55], 0 offen
buffer_load_short_d16 v176, v200, s[52:55], 0 offen
s_waitcnt vmcnt(56) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 3
s_call_b64 s[38:39], 3346
s_nop 0
s_nop 0
v_pack_b32_f16 v141, v189, v141
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v142, v190, v142
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v143, v191, v143
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v144, v192, v144
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
v_pk_fma_f16 v141, v143, -1.0, v141 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
v_pk_mul_f16 v141, v141, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_fma_f16 v144, v142, -1.0, v144 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v144, v144, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_add_f16 v142, v143, v142
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v142, v142, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_pk_fma_f16 v143, v142, -1.0, v143 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
_v_pk_fmac_f16_gfx10_quad_perm 141, 141, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
_v_pk_fmac_f16_gfx10_quad_perm 142, 142, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
_v_pk_fmac_f16_gfx10_quad_perm 143, 143, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
_v_pk_fmac_f16_gfx10_quad_perm 144, 144, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v215, v133 offset:33024
ds_read_b128 v[77:80], v223 offset:8768
ds_write_b32 v216, v134 offset:33024
ds_read_b128 v[81:84], v223 offset:9024
ds_write_b32 v217, v135 offset:33024
ds_read_b128 v[93:96], v213 offset:8256
ds_write_b32 v218, v136 offset:33024
ds_read_b128 v[97:100], v213 offset:8384
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v129, v201, s[40:43], 0 offen
buffer_load_short_d16 v131, v203, s[40:43], 0 offen
buffer_load_short_d16 v130, v202, s[40:43], 0 offen
buffer_load_short_d16 v132, v204, s[40:43], 0 offen
buffer_load_short_d16 v177, v201, s[52:55], 0 offen
buffer_load_short_d16 v179, v203, s[52:55], 0 offen
buffer_load_short_d16 v178, v202, s[52:55], 0 offen
buffer_load_short_d16 v180, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 2
s_call_b64 s[38:39], 3137
s_nop 0
v_pack_b32_f16 v145, v193, v145
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v146, v194, v146
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v147, v195, v147
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v148, v196, v148
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
v_pk_fma_f16 v145, v147, -1.0, v145 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
v_pk_mul_f16 v145, v145, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_fma_f16 v148, v146, -1.0, v148 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v148, v148, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_add_f16 v146, v147, v146
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v146, v146, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_pk_fma_f16 v147, v146, -1.0, v147 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
_v_pk_fmac_f16_gfx10_quad_perm 145, 145, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
_v_pk_fmac_f16_gfx10_quad_perm 146, 146, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
_v_pk_fmac_f16_gfx10_quad_perm 147, 147, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
_v_pk_fmac_f16_gfx10_quad_perm 148, 148, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v137 offset:33024
ds_read_b128 v[69:72], v223 offset:12928
ds_write_b32 v220, v138 offset:33024
ds_read_b128 v[73:76], v223 offset:13184
ds_write_b32 v221, v139 offset:33024
ds_read_b128 v[85:88], v213 offset:12416
ds_write_b32 v222, v140 offset:33024
ds_read_b128 v[89:92], v213 offset:12544
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v133, v197, s[40:43], 0 offen
buffer_load_short_d16 v135, v199, s[40:43], 0 offen
buffer_load_short_d16 v134, v198, s[40:43], 0 offen
buffer_load_short_d16 v136, v200, s[40:43], 0 offen
buffer_load_short_d16 v181, v197, s[52:55], 0 offen
buffer_load_short_d16 v183, v199, s[52:55], 0 offen
buffer_load_short_d16 v182, v198, s[52:55], 0 offen
buffer_load_short_d16 v184, v200, s[52:55], 0 offen
s_waitcnt vmcnt(56) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 5
s_call_b64 s[38:39], 2932
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v101, v149, v101
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v102, v150, v102
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v103, v151, v103
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v104, v152, v104
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
v_pk_fma_f16 v101, v103, -1.0, v101 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
v_pk_mul_f16 v101, v101, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_fma_f16 v104, v102, -1.0, v104 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v104, v104, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_add_f16 v102, v103, v102
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v102, v102, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_pk_fma_f16 v103, v102, -1.0, v103 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
_v_pk_fmac_f16_gfx10_quad_perm 101, 101, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
_v_pk_fmac_f16_gfx10_quad_perm 102, 102, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
_v_pk_fmac_f16_gfx10_quad_perm 103, 103, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
_v_pk_fmac_f16_gfx10_quad_perm 104, 104, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
ds_write_b32 v215, v141 offset:41280
ds_read_b128 v[77:80], v223 offset:17024
ds_write_b32 v216, v142 offset:41280
ds_read_b128 v[81:84], v223 offset:17280
ds_write_b32 v217, v143 offset:41280
ds_read_b128 v[93:96], v213 offset:16512
ds_write_b32 v218, v144 offset:41280
ds_read_b128 v[97:100], v213 offset:16640
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v137, v201, s[40:43], 0 offen
buffer_load_short_d16 v139, v203, s[40:43], 0 offen
buffer_load_short_d16 v138, v202, s[40:43], 0 offen
buffer_load_short_d16 v140, v204, s[40:43], 0 offen
buffer_load_short_d16 v185, v201, s[52:55], 0 offen
buffer_load_short_d16 v187, v203, s[52:55], 0 offen
buffer_load_short_d16 v186, v202, s[52:55], 0 offen
buffer_load_short_d16 v188, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v236 offset:65480
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 3
s_call_b64 s[38:39], 2722
s_nop 0
s_nop 0
v_pack_b32_f16 v105, v153, v105
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v106, v154, v106
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v107, v155, v107
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v108, v156, v108
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
v_pk_fma_f16 v105, v107, -1.0, v105 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
v_pk_mul_f16 v105, v105, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_fma_f16 v108, v106, -1.0, v108 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v108, v108, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_add_f16 v106, v107, v106
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v106, v106, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_pk_fma_f16 v107, v106, -1.0, v107 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
_v_pk_fmac_f16_gfx10_quad_perm 105, 105, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
_v_pk_fmac_f16_gfx10_quad_perm 106, 106, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
_v_pk_fmac_f16_gfx10_quad_perm 107, 107, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
_v_pk_fmac_f16_gfx10_quad_perm 108, 108, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v145 offset:41280
ds_read_b128 v[69:72], v223 offset:21184
ds_write_b32 v220, v146 offset:41280
ds_read_b128 v[73:76], v223 offset:21440
ds_write_b32 v221, v147 offset:41280
ds_read_b128 v[85:88], v213 offset:20672
ds_write_b32 v222, v148 offset:41280
ds_read_b128 v[89:92], v213 offset:20800
s_setprio 0
s_mov_b32 m0, 0x2ffc8
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v141, v197, s[40:43], 0 offen
buffer_load_short_d16 v143, v199, s[40:43], 0 offen
buffer_load_short_d16 v142, v198, s[40:43], 0 offen
buffer_load_short_d16 v144, v200, s[40:43], 0 offen
buffer_load_short_d16 v189, v197, s[52:55], 0 offen
buffer_load_short_d16 v191, v199, s[52:55], 0 offen
buffer_load_short_d16 v190, v198, s[52:55], 0 offen
buffer_load_short_d16 v192, v200, s[52:55], 0 offen
s_waitcnt vmcnt(56) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 3
s_call_b64 s[38:39], 2514
s_nop 0
s_nop 0
v_pack_b32_f16 v109, v157, v109
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v110, v158, v110
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v111, v159, v111
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v112, v160, v112
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
v_pk_fma_f16 v109, v111, -1.0, v109 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
v_pk_mul_f16 v109, v109, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_fma_f16 v112, v110, -1.0, v112 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v112, v112, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_add_f16 v110, v111, v110
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v110, v110, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_pk_fma_f16 v111, v110, -1.0, v111 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
_v_pk_fmac_f16_gfx10_quad_perm 109, 109, 231, 2, 2, 1, 1
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
_v_pk_fmac_f16_gfx10_quad_perm 110, 110, 231, 2, 2, 1, 1
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
_v_pk_fmac_f16_gfx10_quad_perm 111, 111, 231, 2, 2, 1, 1
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
_v_pk_fmac_f16_gfx10_quad_perm 112, 112, 231, 2, 2, 1, 1
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v215, v101
ds_read_b128 v[77:80], v223 offset:25280
ds_write_b32 v216, v102
ds_read_b128 v[81:84], v223 offset:25536
ds_write_b32 v217, v103
ds_read_b128 v[93:96], v213 offset:24768
ds_write_b32 v218, v104
ds_read_b128 v[97:100], v213 offset:24896
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v145, v201, s[40:43], 0 offen
buffer_load_short_d16 v147, v203, s[40:43], 0 offen
buffer_load_short_d16 v146, v202, s[40:43], 0 offen
buffer_load_short_d16 v148, v204, s[40:43], 0 offen
buffer_load_short_d16 v193, v201, s[52:55], 0 offen
buffer_load_short_d16 v195, v203, s[52:55], 0 offen
buffer_load_short_d16 v194, v202, s[52:55], 0 offen
buffer_load_short_d16 v196, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 63042
s_call_b64 s[38:39], 2305
s_branch 63040
v_pack_b32_f16 v113, v161, v113
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v114, v162, v114
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v115, v163, v115
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
s_setprio 1
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v116, v164, v116
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
_v_pk_fmac_f16_gfx10_quad_perm 113, 113, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
_v_pk_fmac_f16_gfx10_quad_perm 116, 116, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_add_f16 v114, v113, v116
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v114, v114, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_fma_f16 v115, -1.0, v116, v113 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v115, v115, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v105
ds_read_b128 v[69:72], v223 offset:29440
ds_write_b32 v220, v106
ds_read_b128 v[73:76], v223 offset:29696
ds_write_b32 v221, v107
ds_read_b128 v[85:88], v213 offset:28928
ds_write_b32 v222, v108
ds_read_b128 v[89:92], v213 offset:29056
s_setprio 0
s_add_u32 s91, s91, 0x100
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v101, v197, s[40:43], 0 offen
buffer_load_short_d16 v104, v200, s[40:43], 0 offen
buffer_load_short_d16 v149, v197, s[52:55], 0 offen
buffer_load_short_d16 v152, v200, s[52:55], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 5
s_call_b64 s[38:39], 2116
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v117, v165, v117
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v118, v166, v118
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v119, v167, v119
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
s_setprio 1
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v120, v168, v120
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
_v_pk_fmac_f16_gfx10_quad_perm 117, 117, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
_v_pk_fmac_f16_gfx10_quad_perm 120, 120, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_add_f16 v118, v117, v120
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v118, v118, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_fma_f16 v119, -1.0, v120, v117 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v119, v119, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
ds_write_b32 v215, v109 offset:8256
ds_read_b128 v[77:80], v223 offset:33536
ds_write_b32 v216, v110 offset:8256
ds_read_b128 v[81:84], v223 offset:33792
ds_write_b32 v217, v111 offset:8256
ds_read_b128 v[93:96], v213 offset:33024
ds_write_b32 v218, v112 offset:8256
ds_read_b128 v[97:100], v213 offset:33152
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v105, v201, s[40:43], 0 offen
buffer_load_short_d16 v108, v204, s[40:43], 0 offen
buffer_load_short_d16 v153, v201, s[52:55], 0 offen
buffer_load_short_d16 v156, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v236 offset:65472
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 5
s_call_b64 s[38:39], 1924
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v121, v169, v121
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v122, v170, v122
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v123, v171, v123
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
s_setprio 1
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v124, v172, v124
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
_v_pk_fmac_f16_gfx10_quad_perm 121, 121, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
_v_pk_fmac_f16_gfx10_quad_perm 124, 124, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_add_f16 v122, v121, v124
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v122, v122, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_fma_f16 v123, -1.0, v124, v121 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v123, v123, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v113 offset:8256
ds_read_b128 v[69:72], v223 offset:37696
ds_write_b32 v220, v114 offset:8256
ds_read_b128 v[73:76], v223 offset:37952
ds_write_b32 v221, v115 offset:8256
ds_read_b128 v[85:88], v213 offset:37184
ds_write_b32 v222, v116 offset:8256
ds_read_b128 v[89:92], v213 offset:37312
s_setprio 0
s_mov_b32 m0, 0x2ffc0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v109, v197, s[40:43], 0 offen
buffer_load_short_d16 v112, v200, s[40:43], 0 offen
buffer_load_short_d16 v157, v197, s[52:55], 0 offen
buffer_load_short_d16 v160, v200, s[52:55], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 5
s_call_b64 s[38:39], 1732
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v125, v173, v125
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v126, v174, v126
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v127, v175, v127
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
s_setprio 1
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v128, v176, v128
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
_v_pk_fmac_f16_gfx10_quad_perm 125, 125, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
_v_pk_fmac_f16_gfx10_quad_perm 128, 128, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_add_f16 v126, v125, v128
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v126, v126, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_fma_f16 v127, -1.0, v128, v125 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v127, v127, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v215, v117 offset:16512
ds_read_b128 v[77:80], v223 offset:41792
ds_write_b32 v216, v118 offset:16512
ds_read_b128 v[81:84], v223 offset:42048
ds_write_b32 v217, v119 offset:16512
ds_read_b128 v[93:96], v213 offset:41280
ds_write_b32 v218, v120 offset:16512
ds_read_b128 v[97:100], v213 offset:41408
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v113, v201, s[40:43], 0 offen
buffer_load_short_d16 v116, v204, s[40:43], 0 offen
buffer_load_short_d16 v161, v201, s[52:55], 0 offen
buffer_load_short_d16 v164, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 4
s_call_b64 s[38:39], 1539
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v129, v177, v129
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v130, v178, v130
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v131, v179, v131
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
s_setprio 1
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v132, v180, v132
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
_v_pk_fmac_f16_gfx10_quad_perm 129, 129, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
_v_pk_fmac_f16_gfx10_quad_perm 132, 132, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_add_f16 v130, v129, v132
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v130, v130, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_fma_f16 v131, -1.0, v132, v129 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v131, v131, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v121 offset:16512
ds_read_b128 v[69:72], v223 offset:45952
ds_write_b32 v220, v122 offset:16512
ds_read_b128 v[73:76], v223 offset:46208
ds_write_b32 v221, v123 offset:16512
ds_read_b128 v[85:88], v213 offset:45440
ds_write_b32 v222, v124 offset:16512
ds_read_b128 v[89:92], v213 offset:45568
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v117, v197, s[40:43], 0 offen
buffer_load_short_d16 v120, v200, s[40:43], 0 offen
buffer_load_short_d16 v165, v197, s[52:55], 0 offen
buffer_load_short_d16 v168, v200, s[52:55], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 7
s_call_b64 s[38:39], 1350
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v133, v181, v133
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v134, v182, v134
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v135, v183, v135
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
s_setprio 1
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v136, v184, v136
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
_v_pk_fmac_f16_gfx10_quad_perm 133, 133, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
_v_pk_fmac_f16_gfx10_quad_perm 136, 136, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_add_f16 v134, v133, v136
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v134, v134, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_fma_f16 v135, -1.0, v136, v133 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v135, v135, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
ds_write_b32 v215, v125 offset:24768
ds_read_b128 v[77:80], v223 offset:512
ds_write_b32 v216, v126 offset:24768
ds_read_b128 v[81:84], v223 offset:768
ds_write_b32 v217, v127 offset:24768
ds_read_b128 v[93:96], v213
ds_write_b32 v218, v128 offset:24768
ds_read_b128 v[97:100], v213 offset:128
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v121, v201, s[40:43], 0 offen
buffer_load_short_d16 v124, v204, s[40:43], 0 offen
buffer_load_short_d16 v169, v201, s[52:55], 0 offen
buffer_load_short_d16 v172, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v236 offset:65476
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 5
s_call_b64 s[38:39], 1156
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v137, v185, v137
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v138, v186, v138
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v139, v187, v139
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
s_setprio 1
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v140, v188, v140
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
_v_pk_fmac_f16_gfx10_quad_perm 137, 137, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
_v_pk_fmac_f16_gfx10_quad_perm 140, 140, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_add_f16 v138, v137, v140
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v138, v138, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_fma_f16 v139, -1.0, v140, v137 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v139, v139, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v129 offset:24768
ds_read_b128 v[69:72], v223 offset:4672
ds_write_b32 v220, v130 offset:24768
ds_read_b128 v[73:76], v223 offset:4928
ds_write_b32 v221, v131 offset:24768
ds_read_b128 v[85:88], v213 offset:4160
ds_write_b32 v222, v132 offset:24768
ds_read_b128 v[89:92], v213 offset:4288
s_setprio 0
s_mov_b32 m0, 0x2ffc4
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v125, v197, s[40:43], 0 offen
buffer_load_short_d16 v128, v200, s[40:43], 0 offen
buffer_load_short_d16 v173, v197, s[52:55], 0 offen
buffer_load_short_d16 v176, v200, s[52:55], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 5
s_call_b64 s[38:39], 964
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v141, v189, v141
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v142, v190, v142
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v143, v191, v143
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
s_setprio 1
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v144, v192, v144
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
_v_pk_fmac_f16_gfx10_quad_perm 141, 141, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
_v_pk_fmac_f16_gfx10_quad_perm 144, 144, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_add_f16 v142, v141, v144
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v142, v142, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_fma_f16 v143, -1.0, v144, v141 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v143, v143, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v215, v133 offset:33024
ds_read_b128 v[77:80], v223 offset:8768
ds_write_b32 v216, v134 offset:33024
ds_read_b128 v[81:84], v223 offset:9024
ds_write_b32 v217, v135 offset:33024
ds_read_b128 v[93:96], v213 offset:8256
ds_write_b32 v218, v136 offset:33024
ds_read_b128 v[97:100], v213 offset:8384
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v129, v201, s[40:43], 0 offen
buffer_load_short_d16 v132, v204, s[40:43], 0 offen
buffer_load_short_d16 v177, v201, s[52:55], 0 offen
buffer_load_short_d16 v180, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 4
s_call_b64 s[38:39], 771
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v145, v193, v145
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v146, v194, v146
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v147, v195, v147
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
s_setprio 1
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v148, v196, v148
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
_v_pk_fmac_f16_gfx10_quad_perm 145, 145, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
_v_pk_fmac_f16_gfx10_quad_perm 148, 148, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_add_f16 v146, v145, v148
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v146, v146, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_fma_f16 v147, -1.0, v148, v145 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v147, v147, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v137 offset:33024
ds_read_b128 v[69:72], v223 offset:12928
ds_write_b32 v220, v138 offset:33024
ds_read_b128 v[73:76], v223 offset:13184
ds_write_b32 v221, v139 offset:33024
ds_read_b128 v[85:88], v213 offset:12416
ds_write_b32 v222, v140 offset:33024
ds_read_b128 v[89:92], v213 offset:12544
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v133, v197, s[40:43], 0 offen
buffer_load_short_d16 v136, v200, s[40:43], 0 offen
buffer_load_short_d16 v181, v197, s[52:55], 0 offen
buffer_load_short_d16 v184, v200, s[52:55], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 7
s_call_b64 s[38:39], 582
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v101, v149, v101
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v102, v150, v102
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v103, v151, v103
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
s_setprio 1
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v104, v152, v104
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
_v_pk_fmac_f16_gfx10_quad_perm 101, 101, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
_v_pk_fmac_f16_gfx10_quad_perm 104, 104, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_add_f16 v102, v101, v104
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v102, v102, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_fma_f16 v103, -1.0, v104, v101 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v103, v103, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
ds_write_b32 v215, v141 offset:41280
ds_read_b128 v[77:80], v223 offset:17024
ds_write_b32 v216, v142 offset:41280
ds_read_b128 v[81:84], v223 offset:17280
ds_write_b32 v217, v143 offset:41280
ds_read_b128 v[93:96], v213 offset:16512
ds_write_b32 v218, v144 offset:41280
ds_read_b128 v[97:100], v213 offset:16640
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v137, v201, s[40:43], 0 offen
buffer_load_short_d16 v140, v204, s[40:43], 0 offen
buffer_load_short_d16 v185, v201, s[52:55], 0 offen
buffer_load_short_d16 v188, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v236 offset:65480
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 5
s_call_b64 s[38:39], 388
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v105, v153, v105
v_dot2_f32_f16 v2, v69, v85, v2
v_dot2_f32_f16 v3, v70, v85, v3
v_dot2_f32_f16 v4, v71, v85, v4
v_dot2_f32_f16 v5, v72, v85, v5
v_pack_b32_f16 v106, v154, v106
v_dot2_f32_f16 v6, v73, v85, v6
v_dot2_f32_f16 v7, v74, v85, v7
v_dot2_f32_f16 v8, v75, v85, v8
v_dot2_f32_f16 v9, v76, v85, v9
v_pack_b32_f16 v107, v155, v107
v_dot2_f32_f16 v10, v69, v86, v10
v_dot2_f32_f16 v11, v70, v86, v11
v_dot2_f32_f16 v12, v71, v86, v12
s_setprio 1
v_dot2_f32_f16 v13, v72, v86, v13
v_pack_b32_f16 v108, v156, v108
v_dot2_f32_f16 v14, v73, v86, v14
v_dot2_f32_f16 v15, v74, v86, v15
v_dot2_f32_f16 v16, v75, v86, v16
v_dot2_f32_f16 v17, v76, v86, v17
_v_pk_fmac_f16_gfx10_quad_perm 105, 105, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v69, v87, v18
v_dot2_f32_f16 v19, v70, v87, v19
v_dot2_f32_f16 v20, v71, v87, v20
v_dot2_f32_f16 v21, v72, v87, v21
_v_pk_fmac_f16_gfx10_quad_perm 108, 108, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v73, v87, v22
v_dot2_f32_f16 v23, v74, v87, v23
v_dot2_f32_f16 v24, v75, v87, v24
v_dot2_f32_f16 v25, v76, v87, v25
v_pk_add_f16 v106, v105, v108
v_dot2_f32_f16 v26, v69, v88, v26
v_dot2_f32_f16 v27, v70, v88, v27
v_dot2_f32_f16 v28, v71, v88, v28
v_dot2_f32_f16 v29, v72, v88, v29
v_pk_mul_f16 v106, v106, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v73, v88, v30
v_dot2_f32_f16 v31, v74, v88, v31
v_dot2_f32_f16 v32, v75, v88, v32
v_dot2_f32_f16 v33, v76, v88, v33
v_pk_fma_f16 v107, -1.0, v108, v105 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v69, v89, v34
v_dot2_f32_f16 v35, v70, v89, v35
v_dot2_f32_f16 v36, v71, v89, v36
v_dot2_f32_f16 v37, v72, v89, v37
v_pk_mul_f16 v107, v107, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v73, v89, v38
v_dot2_f32_f16 v39, v74, v89, v39
v_dot2_f32_f16 v40, v75, v89, v40
v_dot2_f32_f16 v41, v76, v89, v41
v_dot2_f32_f16 v42, v69, v90, v42
v_dot2_f32_f16 v43, v70, v90, v43
v_dot2_f32_f16 v44, v71, v90, v44
v_dot2_f32_f16 v45, v72, v90, v45
v_dot2_f32_f16 v46, v73, v90, v46
v_dot2_f32_f16 v47, v74, v90, v47
v_dot2_f32_f16 v48, v75, v90, v48
v_dot2_f32_f16 v49, v76, v90, v49
v_dot2_f32_f16 v50, v69, v91, v50
v_dot2_f32_f16 v51, v70, v91, v51
v_dot2_f32_f16 v52, v71, v91, v52
v_dot2_f32_f16 v53, v72, v91, v53
v_dot2_f32_f16 v54, v73, v91, v54
v_dot2_f32_f16 v55, v74, v91, v55
v_dot2_f32_f16 v56, v75, v91, v56
v_dot2_f32_f16 v57, v76, v91, v57
v_dot2_f32_f16 v58, v69, v92, v58
v_dot2_f32_f16 v59, v70, v92, v59
v_dot2_f32_f16 v60, v71, v92, v60
v_dot2_f32_f16 v61, v72, v92, v61
v_dot2_f32_f16 v62, v73, v92, v62
v_dot2_f32_f16 v63, v74, v92, v63
v_dot2_f32_f16 v64, v75, v92, v64
v_dot2_f32_f16 v65, v76, v92, v65
ds_write_b32 v219, v145 offset:41280
ds_read_b128 v[69:72], v223 offset:21184
ds_write_b32 v220, v146 offset:41280
ds_read_b128 v[73:76], v223 offset:21440
ds_write_b32 v221, v147 offset:41280
ds_read_b128 v[85:88], v213 offset:20672
ds_write_b32 v222, v148 offset:41280
ds_read_b128 v[89:92], v213 offset:20800
s_setprio 0
s_mov_b32 m0, 0x2ffc8
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
s_add_u32 s52, s52, s69
s_addc_u32 s53, s53, 0
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v141, v197, s[40:43], 0 offen
buffer_load_short_d16 v144, v200, s[40:43], 0 offen
buffer_load_short_d16 v189, v197, s[52:55], 0 offen
buffer_load_short_d16 v192, v200, s[52:55], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 5
s_call_b64 s[38:39], 196
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pack_b32_f16 v109, v157, v109
v_dot2_f32_f16 v2, v77, v93, v2
v_dot2_f32_f16 v3, v78, v93, v3
v_dot2_f32_f16 v4, v79, v93, v4
v_dot2_f32_f16 v5, v80, v93, v5
v_pack_b32_f16 v110, v158, v110
v_dot2_f32_f16 v6, v81, v93, v6
v_dot2_f32_f16 v7, v82, v93, v7
v_dot2_f32_f16 v8, v83, v93, v8
v_dot2_f32_f16 v9, v84, v93, v9
v_pack_b32_f16 v111, v159, v111
v_dot2_f32_f16 v10, v77, v94, v10
v_dot2_f32_f16 v11, v78, v94, v11
v_dot2_f32_f16 v12, v79, v94, v12
s_setprio 1
v_dot2_f32_f16 v13, v80, v94, v13
v_pack_b32_f16 v112, v160, v112
v_dot2_f32_f16 v14, v81, v94, v14
v_dot2_f32_f16 v15, v82, v94, v15
v_dot2_f32_f16 v16, v83, v94, v16
v_dot2_f32_f16 v17, v84, v94, v17
_v_pk_fmac_f16_gfx10_quad_perm 109, 109, 231, 0, 0, 1, 1
v_dot2_f32_f16 v18, v77, v95, v18
v_dot2_f32_f16 v19, v78, v95, v19
v_dot2_f32_f16 v20, v79, v95, v20
v_dot2_f32_f16 v21, v80, v95, v21
_v_pk_fmac_f16_gfx10_quad_perm 112, 112, 231, 0, 0, 1, 1
v_dot2_f32_f16 v22, v81, v95, v22
v_dot2_f32_f16 v23, v82, v95, v23
v_dot2_f32_f16 v24, v83, v95, v24
v_dot2_f32_f16 v25, v84, v95, v25
v_pk_add_f16 v110, v109, v112
v_dot2_f32_f16 v26, v77, v96, v26
v_dot2_f32_f16 v27, v78, v96, v27
v_dot2_f32_f16 v28, v79, v96, v28
v_dot2_f32_f16 v29, v80, v96, v29
v_pk_mul_f16 v110, v110, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v30, v81, v96, v30
v_dot2_f32_f16 v31, v82, v96, v31
v_dot2_f32_f16 v32, v83, v96, v32
v_dot2_f32_f16 v33, v84, v96, v33
v_pk_fma_f16 v111, -1.0, v112, v109 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v34, v77, v97, v34
v_dot2_f32_f16 v35, v78, v97, v35
v_dot2_f32_f16 v36, v79, v97, v36
v_dot2_f32_f16 v37, v80, v97, v37
v_pk_mul_f16 v111, v111, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v38, v81, v97, v38
v_dot2_f32_f16 v39, v82, v97, v39
v_dot2_f32_f16 v40, v83, v97, v40
v_dot2_f32_f16 v41, v84, v97, v41
v_dot2_f32_f16 v42, v77, v98, v42
v_dot2_f32_f16 v43, v78, v98, v43
v_dot2_f32_f16 v44, v79, v98, v44
v_dot2_f32_f16 v45, v80, v98, v45
v_dot2_f32_f16 v46, v81, v98, v46
v_dot2_f32_f16 v47, v82, v98, v47
v_dot2_f32_f16 v48, v83, v98, v48
v_dot2_f32_f16 v49, v84, v98, v49
v_dot2_f32_f16 v50, v77, v99, v50
v_dot2_f32_f16 v51, v78, v99, v51
v_dot2_f32_f16 v52, v79, v99, v52
v_dot2_f32_f16 v53, v80, v99, v53
v_dot2_f32_f16 v54, v81, v99, v54
v_dot2_f32_f16 v55, v82, v99, v55
v_dot2_f32_f16 v56, v83, v99, v56
v_dot2_f32_f16 v57, v84, v99, v57
v_dot2_f32_f16 v58, v77, v100, v58
v_dot2_f32_f16 v59, v78, v100, v59
v_dot2_f32_f16 v60, v79, v100, v60
v_dot2_f32_f16 v61, v80, v100, v61
v_dot2_f32_f16 v62, v81, v100, v62
v_dot2_f32_f16 v63, v82, v100, v63
v_dot2_f32_f16 v64, v83, v100, v64
v_dot2_f32_f16 v65, v84, v100, v65
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v215, v101
ds_read_b128 v[77:80], v223 offset:25280
ds_write_b32 v216, v102
ds_read_b128 v[81:84], v223 offset:25536
ds_write_b32 v217, v103
ds_read_b128 v[93:96], v213 offset:24768
ds_write_b32 v218, v104
ds_read_b128 v[97:100], v213 offset:24896
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
s_add_u32 s52, s52, s70
s_addc_u32 s53, s53, s71
s_sub_u32 s73, s73, 1
s_cselect_b32 s43, 0x31014000, s43
buffer_load_short_d16 v145, v201, s[40:43], 0 offen
buffer_load_short_d16 v148, v204, s[40:43], 0 offen
buffer_load_short_d16 v193, v201, s[52:55], 0 offen
buffer_load_short_d16 v196, v204, s[52:55], 0 offen
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -2
s_cbranch_scc1 63236
s_call_b64 s[38:39], 3
s_branch 63234
s_nop 0
s_nop 0
v_nop
s_cmp_eq_u32 s82, 0
s_cbranch_scc0 8
s_branch 740
s_add_u32 s82, s82, 1
s_andn2_b32 s82, s82, 1
s_bitcmp1_b32 s18, 26
s_cselect_b32 s52, s69, s70
s_cselect_b32 s53, 0, s71
s_sub_u32 s40, s40, s52
s_subb_u32 s41, s41, s53
s_cmp_eq_u32 s94, 0
s_cbranch_scc0 5
s_cbranch_scc1 762
s_nop 0
s_nop 0
s_add_u32 s94, s94, 1
s_andn2_b32 s94, s94, 1
s_min_u32 s72, s82, s94
s_sub_u32 s82, s82, s72
s_sub_u32 s94, s94, s72
s_sub_u32 s72, s72, 2
s_lshr_b32 s52, s69, 1
s_add_u32 s52, s40, s52
s_addc_u32 s53, s41, 0
s_mov_b64 s[54:55], s[42:43]
s_bitcmp1_b32 s18, 18
s_cselect_b32 s55, 0, 0x31014000
s_setpc_b64 s[38:39]
s_nop 0
s_nop 0
s_bitcmp1_b32 s18, 17
s_cbranch_scc1 253
s_add_u32 s88, s88, s17
s_cmp_eq_u32 s88, 0
s_cbranch_scc1 250
s_mov_b32 s89, 0
s_bitcmp1_b32 s18, 16
s_cbranch_scc1 239
s_add_u32 s87, s16, 31
s_lshr_b32 s87, s87, 5
v_mov_b32_e32 v238, s88
v_mul_u32_u24_e32 v238, s87, v238
v_add_co_u32_e64 v238, vcc, s17, v238
v_sub_co_u32_e64 v238, vcc, v238, 1
v_ffbh_u32_e32 v241, s17
v_lshlrev_b32_e64 v242, v241, s17
v_and_b32_e32 v243, 0xffffff00, v242
v_cmp_eq_u32_e32 vcc, 0x80000000, v242
v_cvt_f32_u32_e32 v243, v243
v_rcp_f32_e32 v237, v243
v_sub_co_ci_u32_e32 v240, vcc, 32, v241, vcc
v_cvt_f32_ubyte0_e32 v241, v242
v_fma_f32 v243, v243, v237, -1.0
v_fma_f32 v243, v241, v237, v243
v_fmaak_f32 v243, v243, v237, 0x9f000000
v_mul_f32_e32 v243, 0x5f800000, v243
v_mov_b32_e32 v241, 0
v_cvt_flr_i32_f32_e64 v243, -v243
v_lshl_add_u32 v237, v237, 9, v243
v_mad_u64_u32 v[241:242], vcc, v242, v237, v[241:242]
v_sub_co_ci_u32_e64 v237, vcc, v237, -1, vcc
v_mul_hi_u32 v241, v238, v237
v_add_co_u32_e64 v237, vcc, v241, v238
v_add_co_ci_u32_e64 v241, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v240
v_cndmask_b32_e32 v237, v237, v241, vcc
v_alignbit_b32 v237, v241, v237, v240
v_readfirstlane_b32 s86, v237
v_mul_u32_u24_e64 v237, v237, s8
v_ffbh_u32_e32 v241, s87
v_lshlrev_b32_e64 v242, v241, s87
v_and_b32_e32 v243, 0xffffff00, v242
v_cmp_eq_u32_e32 vcc, 0x80000000, v242
v_cvt_f32_u32_e32 v243, v243
v_rcp_f32_e32 v238, v243
v_sub_co_ci_u32_e32 v240, vcc, 32, v241, vcc
v_cvt_f32_ubyte0_e32 v241, v242
v_fma_f32 v243, v243, v238, -1.0
v_fma_f32 v243, v241, v238, v243
v_fmaak_f32 v243, v243, v238, 0x9f000000
v_mul_f32_e32 v243, 0x5f800000, v243
v_mov_b32_e32 v241, 0
v_cvt_flr_i32_f32_e64 v243, -v243
v_lshl_add_u32 v238, v238, 9, v243
v_mad_u64_u32 v[241:242], vcc, v242, v238, v[241:242]
v_sub_co_ci_u32_e64 v238, vcc, v238, -1, vcc
v_mul_hi_u32 v241, v237, v238
v_add_co_u32_e64 v238, vcc, v241, v237
v_add_co_ci_u32_e64 v241, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v240
v_cndmask_b32_e32 v238, v238, v241, vcc
v_alignbit_b32 v238, v241, v238, v240
v_readfirstlane_b32 s52, v237
v_readfirstlane_b32 s84, v238
s_mul_i32 s84, s84, s87
s_sub_u32 s84, s52, s84
v_sub_co_u32_e64 v238, vcc, s8, v238
v_sub_co_u32_e64 v238, vcc, s17, v238
v_and_b32_e64 v240, v1, 63
v_cmp_eq_u32_e64 vcc, v240, 0
v_cndmask_b32_e32 v238, 1, v238, vcc
s_sub_u32 s58, 0, s75
s_sub_u32 s59, 0, s74
v_mul_u32_u24_e64 v242, v238, 32
v_ffbh_u32_e32 v244, s58
v_lshlrev_b32_e64 v245, v244, s58
v_and_b32_e32 v246, 0xffffff00, v245
v_cmp_eq_u32_e32 vcc, 0x80000000, v245
v_cvt_f32_u32_e32 v246, v246
v_rcp_f32_e32 v240, v246
v_sub_co_ci_u32_e32 v243, vcc, 32, v244, vcc
v_cvt_f32_ubyte0_e32 v244, v245
v_fma_f32 v246, v246, v240, -1.0
v_fma_f32 v246, v244, v240, v246
v_fmaak_f32 v246, v246, v240, 0x9f000000
v_mul_f32_e32 v246, 0x5f800000, v246
v_mov_b32_e32 v244, 0
v_cvt_flr_i32_f32_e64 v246, -v246
v_lshl_add_u32 v240, v240, 9, v246
v_mad_u64_u32 v[244:245], vcc, v245, v240, v[244:245]
v_sub_co_ci_u32_e64 v240, vcc, v240, -1, vcc
v_mul_hi_u32 v244, v242, v240
v_add_co_u32_e64 v240, vcc, v244, v242
v_add_co_ci_u32_e64 v244, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v243
v_cndmask_b32_e32 v240, v240, v244, vcc
v_alignbit_b32 v240, v244, v240, v243
v_mad_i32_i24 v241, v240, s75, v242
v_mul_u32_u24_e64 v242, v240, 1
v_ffbh_u32_e32 v244, s59
v_lshlrev_b32_e64 v245, v244, s59
v_and_b32_e32 v246, 0xffffff00, v245
v_cmp_eq_u32_e32 vcc, 0x80000000, v245
v_cvt_f32_u32_e32 v246, v246
v_rcp_f32_e32 v240, v246
v_sub_co_ci_u32_e32 v243, vcc, 32, v244, vcc
v_cvt_f32_ubyte0_e32 v244, v245
v_fma_f32 v246, v246, v240, -1.0
v_fma_f32 v246, v244, v240, v246
v_fmaak_f32 v246, v246, v240, 0x9f000000
v_mul_f32_e32 v246, 0x5f800000, v246
v_mov_b32_e32 v244, 0
v_cvt_flr_i32_f32_e64 v246, -v246
v_lshl_add_u32 v240, v240, 9, v246
v_mad_u64_u32 v[244:245], vcc, v245, v240, v[244:245]
v_sub_co_ci_u32_e64 v240, vcc, v240, -1, vcc
v_mul_hi_u32 v244, v242, v240
v_add_co_u32_e64 v240, vcc, v244, v242
v_add_co_ci_u32_e64 v244, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v243
v_cndmask_b32_e32 v240, v240, v244, vcc
v_alignbit_b32 v240, v244, v240, v243
v_mad_i32_i24 v242, v240, s74, v242
v_readfirstlane_b32 s76, v241
v_readfirstlane_b32 s77, v242
v_readfirstlane_b32 s78, v240
v_add_co_u32_e64 v227, vcc, s76, v227
v_add_co_ci_u32_e64 v243, vcc, 0, 0, vcc
v_mad_i32_i24 v227, v243, s75, v227
v_mad_i32_i24 v229, v243, s80, v229
v_mad_i32_i24 v228, v243, s79, v228
v_cmp_ge_i32_e64 vcc, v228, 0
v_add_co_ci_u32_e64 v243, vcc, 0, 0, vcc
v_add_co_u32_e64 v229, vcc, v229, v243
v_mad_i32_i24 v228, v243, s74, v228
v_add_co_u32_e64 v228, vcc, s77, v228
v_add_co_ci_u32_e64 v243, vcc, 0, 0, vcc
v_add_co_u32_e64 v229, vcc, v229, v243
v_mad_i32_i24 v228, v243, s74, v228
v_add_co_u32_e64 v229, vcc, s78, v229
v_readlane_b32 s76, v241, 1
v_readlane_b32 s77, v242, 1
v_readlane_b32 s78, v240, 1
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
s_branch 48
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
s_branch 36
s_add_u32 s83, s83, 32
s_cmp_ge_u32 s83, s85
s_cbranch_scc0 33
s_bitset1_b32 s18, 22
s_sub_u32 s88, s88, s17
s_subb_u32 s89, s89, 0
s_cbranch_scc1 65269
v_add_co_u32_e64 v227, vcc, s76, v227
v_add_co_ci_u32_e64 v237, vcc, 0, 0, vcc
v_mad_i32_i24 v227, v237, s75, v227
v_mad_i32_i24 v229, v237, s80, v229
v_mad_i32_i24 v228, v237, s79, v228
v_cmp_ge_i32_e64 vcc, v228, 0
v_add_co_ci_u32_e64 v237, vcc, 0, 0, vcc
v_add_co_u32_e64 v229, vcc, v229, v237
v_mad_i32_i24 v228, v237, s74, v228
v_add_co_u32_e64 v228, vcc, s77, v228
v_add_co_ci_u32_e64 v237, vcc, 0, 0, vcc
v_add_co_u32_e64 v229, vcc, v229, v237
v_mad_i32_i24 v228, v237, s74, v228
v_add_co_u32_e64 v229, vcc, s78, v229
s_mov_b32 s83, s84
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccz 255
v_subrev_co_u32_e64 v237, vcc, s75, v227
v_subrev_co_u32_e64 v238, vcc, s74, v228
s_bitcmp1_b32 s18, 22
s_cbranch_scc0 66
s_bitset0_b32 s18, 22
s_bfe_u32 s52, s18, 0x10014
v_mul_u32_u24_e32 v242, 3, v237
v_mul_u32_u24_e32 v243, 3, v238
v_cvt_pk_u16_u32_e64 v245, v242, v243
v_and_b32_e64 v242, v1, 1
v_cmp_eq_u32_e64 vcc, v242, 1
v_cndmask_b32_e32 v245, v229, v245, vcc
v_lshrrev_b32_e32 v241, 1, v1
v_bfe_u32 v246, v241, s52, 1
v_lshrrev_b32_e32 v241, 1, v1
v_bfi_b32 v241, 1, v1, v241
v_lshrrev_b32_e32 v242, 2, v1
v_bfi_b32 v242, 1, v1, v242
v_cmp_eq_u32_e64 vcc, s52, 0
v_cndmask_b32_e32 v241, v242, v241, vcc
s_sub_u32 s52, 1, s52
v_lshrrev_b32_e32 v242, s52, v241
v_bfi_b32 v241, 32, v242, v241
v_and_b32_e32 v241, 63, v241
v_add_co_u32_e64 v242, vcc, 16, v241
v_and_b32_e64 v243, v1, 2
v_cmp_eq_u32_e64 vcc, v243, 0
v_cndmask_b32_e32 v242, v242, v241, vcc
v_lshlrev_b32_e32 v243, 14, v246
v_mad_u32_u24 v242, 4, v242, v243
v_add_co_u32_e64 v241, vcc, s96, v242
ds_write_b32 v241, v245
v_writelane_b32 v243, s18, 0
v_writelane_b32 v243, s85, 1
v_writelane_b32 v243, s84, 2
v_and_b32_e64 v241, v1, 63
v_cmp_ge_u32_e64 vcc, v241, 3
v_mov_b32_e32 v244, 0x4000
v_cndmask_b32_e32 v241, v241, v244, vcc
v_mad_i32_i24 v241, v241, 4, s96
ds_write_b32 v241, v243 offset:256
s_add_u32 s96, s96, 0x18c
s_cmp_eq_u32 s96, 0xffc0
s_cselect_b32 s96, 0xc1e0, s96
v_mov_b32_dpp v239, v229  quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v237, v237  quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v238, v238  quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_readfirstlane_b32 s81, v239
v_sub_co_u32_e64 v240, vcc, v239, s81
v_mul_lo_u32 v240, v240, s65
v_and_b32_e64 v244, v1, 3
v_ashrrev_i32_e64 v245, 0, s31
v_subrev_co_u32_e64 v244, vcc, v245, v244
v_ashrrev_i32_e64 v245, 0, s62
v_mad_i32_i24 v241, v245, 2, v244
s_bfe_u32 s52, s18, 0x10014
v_lshrrev_b32_e32 v243, 2, v1
v_and_b32_e32 v243, s52, v243
v_mad_i32_i24 v241, v243, 2, v241
v_add_co_u32_e64 v242, vcc, 0, s63
v_ashrrev_i32_e32 v242, 0, v242
v_add_co_u32_e64 v243, vcc, 0, s30
v_ashrrev_i32_e32 v243, 0, v243
v_sub_nc_i32 v242, v242, v243
s_lshl_b32 s54, s15, 1
v_cmp_ge_u32_e64 s[52:53], v239, s12
v_mad_i32_i24 v237, v237, 3, v241
v_cmp_ge_u32_e64 s[56:57], v237, s15
v_mad_i32_i24 v237, 2, v237, v240
s_or_b64 s[56:57], s[56:57], s[52:53]
v_mad_i32_i24 v238, v238, 3, v242
v_cmp_ge_u32_e64 s[58:59], v238, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v197, v238, s54, v237
v_cndmask_b32_e64 v197, v197, -1, s[58:59]
v_add_co_u32_e64 v238, vcc, 1, v238
v_cmp_ge_u32_e64 s[58:59], v238, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v198, v238, s54, v237
v_cndmask_b32_e64 v198, v198, -1, s[58:59]
v_add_co_u32_e64 v238, vcc, 1, v238
v_cmp_ge_u32_e64 s[58:59], v238, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v199, v238, s54, v237
v_cndmask_b32_e64 v199, v199, -1, s[58:59]
v_add_co_u32_e64 v238, vcc, 1, v238
v_cmp_ge_u32_e64 s[58:59], v238, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v200, v238, s54, v237
v_cndmask_b32_e64 v200, v200, -1, s[58:59]
s_bitcmp1_b32 s18, 20
s_cbranch_scc0 60
v_mov_b32_dpp v239, v229  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v237, v227  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v238, v228  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[52:53], v239, s12
v_sub_co_u32_e64 v240, vcc, v239, s81
v_mul_lo_u32 v240, v240, s65
v_sub_co_u32_e64 v237, vcc, v237, s75
v_sub_co_u32_e64 v238, vcc, v238, s74
v_mad_i32_i24 v237, v237, 3, v241
v_cmp_ge_u32_e64 s[56:57], v237, s15
v_mad_i32_i24 v237, 2, v237, v240
s_or_b64 s[56:57], s[56:57], s[52:53]
v_mad_i32_i24 v238, v238, 3, v242
v_cmp_ge_u32_e64 s[58:59], v238, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v201, v238, s54, v237
v_cndmask_b32_e64 v201, v201, -1, s[58:59]
v_add_co_u32_e64 v238, vcc, 1, v238
v_cmp_ge_u32_e64 s[58:59], v238, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v202, v238, s54, v237
v_cndmask_b32_e64 v202, v202, -1, s[58:59]
v_add_co_u32_e64 v238, vcc, 1, v238
v_cmp_ge_u32_e64 s[58:59], v238, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v203, v238, s54, v237
v_cndmask_b32_e64 v203, v203, -1, s[58:59]
v_add_co_u32_e64 v238, vcc, 1, v238
v_cmp_ge_u32_e64 s[58:59], v238, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v204, v238, s54, v237
v_cndmask_b32_e64 v204, v204, -1, s[58:59]
s_branch 26
s_bitcmp1_b32 s18, 24
s_cselect_b32 s52, s68, 0
v_add_co_u32_e64 v237, vcc, v197, s52
v_cmp_eq_u32_e64 vcc, v197, -1
v_cndmask_b32_e64 v201, v237, -1, vcc
v_add_co_u32_e64 v237, vcc, v198, s52
v_cmp_eq_u32_e64 vcc, v198, -1
v_cndmask_b32_e64 v202, v237, -1, vcc
v_add_co_u32_e64 v237, vcc, v199, s52
v_cmp_eq_u32_e64 vcc, v199, -1
v_cndmask_b32_e64 v203, v237, -1, vcc
v_add_co_u32_e64 v237, vcc, v200, s52
v_cmp_eq_u32_e64 vcc, v200, -1
v_cndmask_b32_e64 v204, v237, -1, vcc
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 163
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
s_branch 130
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 149
s_bfe_u32 s52, s18, 0x10014
v_xor_b32_dpp v237, v1, v1  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_bfe_u32 v239, v1, 2, s52
v_mad_u32_u24 v237, v239, 2, v237
v_mad_u32_u24 v237, s62, 2, v237
v_sub_co_u32_e64 v239, vcc, s29, v237
v_sub_co_u32_e64 v239, vcc, v239, 1
s_bfe_u32 s54, s18, 0x10001
v_cmp_eq_u32_e64 vcc, s54, 1
v_cndmask_b32_e32 v237, v237, v239, vcc
v_cmp_ge_u32_e64 s[52:53], v237, s29
v_lshlrev_b32_e32 v237, 1, v237
s_bfe_u32 s54, s18, 0x10018
v_bfe_u32 v240, v1, 2, s54
v_mul_lo_u32 v240, s68, v240
v_add_co_u32_e64 v237, vcc, v237, v240
v_mul_lo_u32 v238, s90, v230
v_add_co_u32_e64 v238, vcc, v238, v237
s_sub_u32 s54, s28, s63
s_sub_u32 s54, s54, 2
s_bitcmp1_b32 s18, 0
s_cselect_b32 s54, s54, s63
v_mov_b32_e32 v240, s54
s_lshl_b32 s57, s29, 1
v_cmp_ge_u32_e64 s[54:55], v240, s28
v_mad_i32_i24 v197, v240, s57, v238
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v197, v197, -1, s[54:55]
v_mov_b32_e32 v198, v197
v_add_co_u32_e64 v240, vcc, v240, 1
v_cmp_ge_u32_e64 s[54:55], v240, s28
v_mad_i32_i24 v200, v240, s57, v238
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v200, v200, -1, s[54:55]
v_add_co_u32_e64 v240, vcc, v240, 1
v_cmp_ge_u32_e64 s[54:55], v240, s28
v_mad_i32_i24 v199, v240, s57, v238
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v199, v199, -1, s[54:55]
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v197, v198, v200, vcc
v_cndmask_b32_e32 v200, v200, v198, vcc
s_lshl_b32 s52, s90, 3
s_and_b32 s53, s18, 0x1100000
s_cselect_b32 s52, s52, 0
v_add_co_u32_e64 v237, vcc, v197, s52
v_cmp_eq_u32_e64 vcc, v197, -1
v_cndmask_b32_e64 v201, v237, -1, vcc
v_add_co_u32_e64 v237, vcc, v198, s52
v_cmp_eq_u32_e64 vcc, v198, -1
v_cndmask_b32_e64 v202, v237, -1, vcc
v_add_co_u32_e64 v237, vcc, v199, s52
v_cmp_eq_u32_e64 vcc, v199, -1
v_cndmask_b32_e64 v203, v237, -1, vcc
v_add_co_u32_e64 v237, vcc, v200, s52
v_cmp_eq_u32_e64 vcc, v200, -1
v_cndmask_b32_e64 v204, v237, -1, vcc
v_add_co_u32_e64 v237, vcc, v230, s83
v_cmp_lt_u32_e64 vcc, v237, s16
v_cndmask_b32_e32 v197, -1, v197, vcc
v_cndmask_b32_e32 v198, -1, v198, vcc
v_cndmask_b32_e32 v199, -1, v199, vcc
v_cndmask_b32_e32 v200, -1, v200, vcc
s_and_b32 s52, s18, 0x1100000
s_cbranch_scc0 4
v_add_co_u32_e64 v237, vcc, v237, 8
v_cmp_lt_u32_e64 vcc, v237, s16
v_cndmask_b32_e32 v201, -1, v201, vcc
v_cndmask_b32_e32 v202, -1, v202, vcc
v_cndmask_b32_e32 v203, -1, v203, vcc
v_cndmask_b32_e32 v204, -1, v204, vcc
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
s_mov_b32 s43, 0x31014000
s_mov_b32 s73, -1
s_bitcmp0_b32 s13, 0
s_cbranch_scc1 5
s_mov_b32 s43, 0
s_bitcmp1_b32 s18, 20
s_addc_u32 s73, 0, 1
s_sub_u32 s40, s40, s68
s_subb_u32 s41, s41, 0
s_add_u32 s53, s13, 1
s_and_b32 s53, s53, -2
s_bfe_u32 s52, s18, 0x10014
s_lshl_b32 s82, s53, s52
s_bitcmp1_b32 s18, 20
s_cselect_b32 s52, 0, 0x2000000
s_bitcmp1_b32 s53, 1
s_cselect_b32 s52, s52, 0
s_xor_b32 s18, s18, s52
s_branch 64802
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_and_b32 s52, 0x900000, s18
s_subb_u32 s62, s62, 1
s_cbranch_scc0 65116
s_and_b32 s52, 0x900000, s18
s_subb_u32 s62, s61, 1
s_add_u32 s63, s63, 2
s_cmp_ge_u32 s63, s28
s_cbranch_scc0 65110
s_mov_b32 s63, 0
s_branch 65072
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s52, 0x3c3c3c3c
s_mov_b32 s53, s52
v_mov_b32_e32 v238, v3
v_mov_b32_e32 v239, v4
v_mov_b32_e32 v240, v5
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v2
v_add_f32_dpp v237, v2, v2  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v3, v3  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v4, v4  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v5, v5  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v4, v4, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v5, v5, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v2, v2, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v3, v3, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v3, v4  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v2, v5  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v3, v3  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v3, v3, v3  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v2, v2  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v2, v2, v2  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v5, v238
v_add_f32_dpp v5, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v4, v237
v_add_f32_dpp v4, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v3, v3, v2  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v2, v5, v4  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v4, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v4, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v3, v239, v3, s[52:53]
v_mov_b32_dpp v4, v4  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v4, v4  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v2, v2
v_cvt_f16_f32_e32 v3, v3
v_cvt_f16_f32_e32 v4, v4
v_mov_b32_e32 v238, v7
v_mov_b32_e32 v239, v8
v_mov_b32_e32 v240, v9
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v6
v_add_f32_dpp v237, v6, v6  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v7, v7  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v8, v8  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v9, v9  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v8, v8, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v9, v9, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v6, v6, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v7, v7, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v7, v8  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v6, v9  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v7, v7  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v7, v7, v7  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v6, v6  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v6, v6, v6  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v9, v238
v_add_f32_dpp v9, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v8, v237
v_add_f32_dpp v8, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v7, v7, v6  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v5, v9, v8  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v8, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v8, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v6, v239, v7, s[52:53]
v_mov_b32_dpp v7, v8  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v7, v8  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v5, v5
v_cvt_f16_f32_e32 v6, v6
v_cvt_f16_f32_e32 v7, v7
v_mov_b32_e32 v238, v11
v_mov_b32_e32 v239, v12
v_mov_b32_e32 v240, v13
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v10
v_add_f32_dpp v237, v10, v10  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v11, v11  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v12, v12  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v13, v13  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v12, v12, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v13, v13, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v10, v10, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v11, v11, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v11, v12  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v10, v13  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v11, v11  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v11, v11, v11  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v10, v10  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v10, v10, v10  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v13, v238
v_add_f32_dpp v13, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v12, v237
v_add_f32_dpp v12, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v11, v11, v10  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v8, v13, v12  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v12, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v12, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v9, v239, v11, s[52:53]
v_mov_b32_dpp v10, v12  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v10, v12  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v8, v8
v_cvt_f16_f32_e32 v9, v9
v_cvt_f16_f32_e32 v10, v10
v_mov_b32_e32 v238, v15
v_mov_b32_e32 v239, v16
v_mov_b32_e32 v240, v17
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v14
v_add_f32_dpp v237, v14, v14  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v15, v15  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v16, v16  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v17, v17  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v16, v16, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v17, v17, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v14, v14, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v15, v15, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v15, v16  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v14, v17  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v15, v15  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v15, v15, v15  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v14, v14  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v14, v14, v14  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v17, v238
v_add_f32_dpp v17, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v16, v237
v_add_f32_dpp v16, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v15, v15, v14  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v11, v17, v16  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v16, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v16, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v12, v239, v15, s[52:53]
v_mov_b32_dpp v13, v16  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v13, v16  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v11, v11
v_cvt_f16_f32_e32 v12, v12
v_cvt_f16_f32_e32 v13, v13
v_mov_b32_e32 v238, v19
v_mov_b32_e32 v239, v20
v_mov_b32_e32 v240, v21
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v18
v_add_f32_dpp v237, v18, v18  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v19, v19  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v20, v20  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v21, v21  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v20, v20, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v21, v21, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v18, v18, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v19, v19, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v19, v20  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v18, v21  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v19, v19  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v19, v19, v19  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v18, v18  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v18, v18, v18  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v21, v238
v_add_f32_dpp v21, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v20, v237
v_add_f32_dpp v20, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v19, v19, v18  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v14, v21, v20  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v20, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v20, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v15, v239, v19, s[52:53]
v_mov_b32_dpp v16, v20  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v16, v20  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v14, v14
v_cvt_f16_f32_e32 v15, v15
v_cvt_f16_f32_e32 v16, v16
v_mov_b32_e32 v238, v23
v_mov_b32_e32 v239, v24
v_mov_b32_e32 v240, v25
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v22
v_add_f32_dpp v237, v22, v22  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v23, v23  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v24, v24  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v25, v25  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v24, v24, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v25, v25, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v22, v22, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v23, v23, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v23, v24  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v22, v25  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v23, v23  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v23, v23, v23  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v22, v22  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v22, v22, v22  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v25, v238
v_add_f32_dpp v25, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v24, v237
v_add_f32_dpp v24, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v23, v23, v22  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v17, v25, v24  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v24, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v24, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v18, v239, v23, s[52:53]
v_mov_b32_dpp v19, v24  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v19, v24  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v17, v17
v_cvt_f16_f32_e32 v18, v18
v_cvt_f16_f32_e32 v19, v19
v_mov_b32_e32 v238, v27
v_mov_b32_e32 v239, v28
v_mov_b32_e32 v240, v29
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v26
v_add_f32_dpp v237, v26, v26  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v27, v27  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v28, v28  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v29, v29  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v28, v28, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v29, v29, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v26, v26, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v27, v27, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v27, v28  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v26, v29  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v27, v27  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v27, v27, v27  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v26, v26  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v26, v26, v26  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v29, v238
v_add_f32_dpp v29, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v28, v237
v_add_f32_dpp v28, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v27, v27, v26  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v20, v29, v28  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v28, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v28, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v21, v239, v27, s[52:53]
v_mov_b32_dpp v22, v28  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v22, v28  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v20, v20
v_cvt_f16_f32_e32 v21, v21
v_cvt_f16_f32_e32 v22, v22
v_mov_b32_e32 v238, v31
v_mov_b32_e32 v239, v32
v_mov_b32_e32 v240, v33
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v30
v_add_f32_dpp v237, v30, v30  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v31, v31  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v32, v32  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v33, v33  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v32, v32, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v33, v33, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v30, v30, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v31, v31, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v31, v32  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v30, v33  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v31, v31  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v31, v31, v31  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v30, v30  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v30, v30, v30  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v33, v238
v_add_f32_dpp v33, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v32, v237
v_add_f32_dpp v32, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v31, v31, v30  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v23, v33, v32  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v32, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v32, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v24, v239, v31, s[52:53]
v_mov_b32_dpp v25, v32  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v25, v32  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v23, v23
v_cvt_f16_f32_e32 v24, v24
v_cvt_f16_f32_e32 v25, v25
v_mov_b32_e32 v238, v35
v_mov_b32_e32 v239, v36
v_mov_b32_e32 v240, v37
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v34
v_add_f32_dpp v237, v34, v34  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v35, v35  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v36, v36  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v37, v37  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v36, v36, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v37, v37, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v34, v34, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v35, v35, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v35, v36  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v34, v37  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v35, v35  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v35, v35, v35  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v34, v34  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v34, v34, v34  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v37, v238
v_add_f32_dpp v37, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v36, v237
v_add_f32_dpp v36, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v35, v35, v34  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v26, v37, v36  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v36, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v36, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v27, v239, v35, s[52:53]
v_mov_b32_dpp v28, v36  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v28, v36  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v26, v26
v_cvt_f16_f32_e32 v27, v27
v_cvt_f16_f32_e32 v28, v28
v_mov_b32_e32 v238, v39
v_mov_b32_e32 v239, v40
v_mov_b32_e32 v240, v41
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v38
v_add_f32_dpp v237, v38, v38  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v39, v39  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v40, v40  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v41, v41  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v40, v40, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v41, v41, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v38, v38, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v39, v39, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v39, v40  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v38, v41  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v39, v39  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v39, v39, v39  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v38, v38  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v38, v38, v38  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v41, v238
v_add_f32_dpp v41, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v40, v237
v_add_f32_dpp v40, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v39, v39, v38  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v29, v41, v40  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v40, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v40, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v30, v239, v39, s[52:53]
v_mov_b32_dpp v31, v40  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v31, v40  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v29, v29
v_cvt_f16_f32_e32 v30, v30
v_cvt_f16_f32_e32 v31, v31
v_mov_b32_e32 v238, v43
v_mov_b32_e32 v239, v44
v_mov_b32_e32 v240, v45
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v42
v_add_f32_dpp v237, v42, v42  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v43, v43  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v44, v44  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v45, v45  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v44, v44, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v45, v45, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v42, v42, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v43, v43, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v43, v44  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v42, v45  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v43, v43  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v43, v43, v43  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v42, v42  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v42, v42, v42  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v45, v238
v_add_f32_dpp v45, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v44, v237
v_add_f32_dpp v44, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v43, v43, v42  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v32, v45, v44  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v44, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v44, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v33, v239, v43, s[52:53]
v_mov_b32_dpp v34, v44  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v34, v44  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v32, v32
v_cvt_f16_f32_e32 v33, v33
v_cvt_f16_f32_e32 v34, v34
v_mov_b32_e32 v238, v47
v_mov_b32_e32 v239, v48
v_mov_b32_e32 v240, v49
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v46
v_add_f32_dpp v237, v46, v46  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v47, v47  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v48, v48  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v49, v49  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v48, v48, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v49, v49, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v46, v46, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v47, v47, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v47, v48  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v46, v49  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v47, v47  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v47, v47, v47  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v46, v46  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v46, v46, v46  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v49, v238
v_add_f32_dpp v49, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v48, v237
v_add_f32_dpp v48, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v47, v47, v46  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v35, v49, v48  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v48, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v48, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v36, v239, v47, s[52:53]
v_mov_b32_dpp v37, v48  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v37, v48  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v35, v35
v_cvt_f16_f32_e32 v36, v36
v_cvt_f16_f32_e32 v37, v37
v_mov_b32_e32 v238, v51
v_mov_b32_e32 v239, v52
v_mov_b32_e32 v240, v53
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v50
v_add_f32_dpp v237, v50, v50  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v51, v51  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v52, v52  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v53, v53  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v52, v52, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v53, v53, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v50, v50, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v51, v51, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v51, v52  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v50, v53  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v51, v51  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v51, v51, v51  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v50, v50  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v50, v50, v50  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v53, v238
v_add_f32_dpp v53, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v52, v237
v_add_f32_dpp v52, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v51, v51, v50  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v38, v53, v52  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v52, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v52, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v39, v239, v51, s[52:53]
v_mov_b32_dpp v40, v52  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v40, v52  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v38, v38
v_cvt_f16_f32_e32 v39, v39
v_cvt_f16_f32_e32 v40, v40
v_mov_b32_e32 v238, v55
v_mov_b32_e32 v239, v56
v_mov_b32_e32 v240, v57
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v54
v_add_f32_dpp v237, v54, v54  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v55, v55  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v56, v56  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v57, v57  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v56, v56, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v57, v57, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v54, v54, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v55, v55, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v55, v56  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v54, v57  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v55, v55  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v55, v55, v55  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v54, v54  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v54, v54, v54  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v57, v238
v_add_f32_dpp v57, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v56, v237
v_add_f32_dpp v56, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v55, v55, v54  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v41, v57, v56  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v56, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v56, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v42, v239, v55, s[52:53]
v_mov_b32_dpp v43, v56  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v43, v56  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v41, v41
v_cvt_f16_f32_e32 v42, v42
v_cvt_f16_f32_e32 v43, v43
v_mov_b32_e32 v238, v59
v_mov_b32_e32 v239, v60
v_mov_b32_e32 v240, v61
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v58
v_add_f32_dpp v237, v58, v58  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v59, v59  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v60, v60  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v61, v61  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v60, v60, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v61, v61, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v58, v58, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v59, v59, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v59, v60  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v58, v61  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v59, v59  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v59, v59, v59  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v58, v58  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v58, v58, v58  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v61, v238
v_add_f32_dpp v61, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v60, v237
v_add_f32_dpp v60, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v59, v59, v58  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v44, v61, v60  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v60, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v60, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v45, v239, v59, s[52:53]
v_mov_b32_dpp v46, v60  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v46, v60  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v44, v44
v_cvt_f16_f32_e32 v45, v45
v_cvt_f16_f32_e32 v46, v46
v_mov_b32_e32 v238, v63
v_mov_b32_e32 v239, v64
v_mov_b32_e32 v240, v65
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v237, v62
v_add_f32_dpp v237, v62, v62  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v238, v63, v63  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v239, v64, v64  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v240, v65, v65  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v64, v64, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v65, v65, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v62, v62, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v63, v63, v232  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v63, v64  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v62, v65  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v238, v239, v238  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v237, v240, v237  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v240, v63, v63  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v63, v63, v63  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v239, v62, v62  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v62, v62, v62  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v65, v238
v_add_f32_dpp v65, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v240  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v239, v240  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v64, v237
v_add_f32_dpp v64, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v239, v239  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v240, v238, v238  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v63, v63, v62  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v47, v65, v64  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v64, v237, v237  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v64, v240  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v239, v239  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v48, v239, v63, s[52:53]
v_mov_b32_dpp v49, v64  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v49, v64  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v47, v47
v_cvt_f16_f32_e32 v48, v48
v_cvt_f16_f32_e32 v49, v49
s_waitcnt vmcnt(0)
v_readlane_b32 s55, v235, 0
v_add_f16_e64 v2, v2, s55
v_mul_f16_e64 v237, v2, s36
v_cmp_lt_f16_e64 vcc, v2, 0
v_cndmask_b32_e32 v2, v2, v237, vcc
v_add_f16_e64 v5, v5, s55
v_mul_f16_e64 v237, v5, s36
v_cmp_lt_f16_e64 vcc, v5, 0
v_cndmask_b32_e32 v5, v5, v237, vcc
buffer_store_short v2, v205, s[44:47], 0 offen
buffer_store_short v5, v209, s[44:47], 0 offen
v_add_f16_e64 v3, v3, s55
v_mul_f16_e64 v237, v3, s36
v_cmp_lt_f16_e64 vcc, v3, 0
v_cndmask_b32_e32 v3, v3, v237, vcc
v_add_f16_e64 v6, v6, s55
v_mul_f16_e64 v237, v6, s36
v_cmp_lt_f16_e64 vcc, v6, 0
v_cndmask_b32_e32 v6, v6, v237, vcc
buffer_store_short v3, v206, s[44:47], 0 offen
buffer_store_short v6, v210, s[44:47], 0 offen
v_add_f16_e64 v4, v4, s55
v_mul_f16_e64 v237, v4, s36
v_cmp_lt_f16_e64 vcc, v4, 0
v_cndmask_b32_e32 v4, v4, v237, vcc
v_add_f16_e64 v7, v7, s55
v_mul_f16_e64 v237, v7, s36
v_cmp_lt_f16_e64 vcc, v7, 0
v_cndmask_b32_e32 v7, v7, v237, vcc
buffer_store_short v4, v207, s[44:47], 0 offen
buffer_store_short v7, v211, s[44:47], 0 offen
v_nop
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v235, 1
v_add_f16_e64 v8, v8, s55
v_mul_f16_e64 v237, v8, s36
v_cmp_lt_f16_e64 vcc, v8, 0
v_cndmask_b32_e32 v8, v8, v237, vcc
v_add_f16_e64 v11, v11, s55
v_mul_f16_e64 v237, v11, s36
v_cmp_lt_f16_e64 vcc, v11, 0
v_cndmask_b32_e32 v11, v11, v237, vcc
buffer_store_short v8, v205, s[44:47], 0 offen
buffer_store_short v11, v209, s[44:47], 0 offen
v_add_f16_e64 v9, v9, s55
v_mul_f16_e64 v237, v9, s36
v_cmp_lt_f16_e64 vcc, v9, 0
v_cndmask_b32_e32 v9, v9, v237, vcc
v_add_f16_e64 v12, v12, s55
v_mul_f16_e64 v237, v12, s36
v_cmp_lt_f16_e64 vcc, v12, 0
v_cndmask_b32_e32 v12, v12, v237, vcc
buffer_store_short v9, v206, s[44:47], 0 offen
buffer_store_short v12, v210, s[44:47], 0 offen
v_add_f16_e64 v10, v10, s55
v_mul_f16_e64 v237, v10, s36
v_cmp_lt_f16_e64 vcc, v10, 0
v_cndmask_b32_e32 v10, v10, v237, vcc
v_add_f16_e64 v13, v13, s55
v_mul_f16_e64 v237, v13, s36
v_cmp_lt_f16_e64 vcc, v13, 0
v_cndmask_b32_e32 v13, v13, v237, vcc
buffer_store_short v10, v207, s[44:47], 0 offen
buffer_store_short v13, v211, s[44:47], 0 offen
v_nop
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v235, 2
v_add_f16_e64 v14, v14, s55
v_mul_f16_e64 v237, v14, s36
v_cmp_lt_f16_e64 vcc, v14, 0
v_cndmask_b32_e32 v14, v14, v237, vcc
v_add_f16_e64 v17, v17, s55
v_mul_f16_e64 v237, v17, s36
v_cmp_lt_f16_e64 vcc, v17, 0
v_cndmask_b32_e32 v17, v17, v237, vcc
buffer_store_short v14, v205, s[44:47], 0 offen
buffer_store_short v17, v209, s[44:47], 0 offen
v_add_f16_e64 v15, v15, s55
v_mul_f16_e64 v237, v15, s36
v_cmp_lt_f16_e64 vcc, v15, 0
v_cndmask_b32_e32 v15, v15, v237, vcc
v_add_f16_e64 v18, v18, s55
v_mul_f16_e64 v237, v18, s36
v_cmp_lt_f16_e64 vcc, v18, 0
v_cndmask_b32_e32 v18, v18, v237, vcc
buffer_store_short v15, v206, s[44:47], 0 offen
buffer_store_short v18, v210, s[44:47], 0 offen
v_add_f16_e64 v16, v16, s55
v_mul_f16_e64 v237, v16, s36
v_cmp_lt_f16_e64 vcc, v16, 0
v_cndmask_b32_e32 v16, v16, v237, vcc
v_add_f16_e64 v19, v19, s55
v_mul_f16_e64 v237, v19, s36
v_cmp_lt_f16_e64 vcc, v19, 0
v_cndmask_b32_e32 v19, v19, v237, vcc
buffer_store_short v16, v207, s[44:47], 0 offen
buffer_store_short v19, v211, s[44:47], 0 offen
v_nop
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v235, 3
v_add_f16_e64 v20, v20, s55
v_mul_f16_e64 v237, v20, s36
v_cmp_lt_f16_e64 vcc, v20, 0
v_cndmask_b32_e32 v20, v20, v237, vcc
v_add_f16_e64 v23, v23, s55
v_mul_f16_e64 v237, v23, s36
v_cmp_lt_f16_e64 vcc, v23, 0
v_cndmask_b32_e32 v23, v23, v237, vcc
buffer_store_short v20, v205, s[44:47], 0 offen
buffer_store_short v23, v209, s[44:47], 0 offen
v_add_f16_e64 v21, v21, s55
v_mul_f16_e64 v237, v21, s36
v_cmp_lt_f16_e64 vcc, v21, 0
v_cndmask_b32_e32 v21, v21, v237, vcc
v_add_f16_e64 v24, v24, s55
v_mul_f16_e64 v237, v24, s36
v_cmp_lt_f16_e64 vcc, v24, 0
v_cndmask_b32_e32 v24, v24, v237, vcc
buffer_store_short v21, v206, s[44:47], 0 offen
buffer_store_short v24, v210, s[44:47], 0 offen
v_add_f16_e64 v22, v22, s55
v_mul_f16_e64 v237, v22, s36
v_cmp_lt_f16_e64 vcc, v22, 0
v_cndmask_b32_e32 v22, v22, v237, vcc
v_add_f16_e64 v25, v25, s55
v_mul_f16_e64 v237, v25, s36
v_cmp_lt_f16_e64 vcc, v25, 0
v_cndmask_b32_e32 v25, v25, v237, vcc
buffer_store_short v22, v207, s[44:47], 0 offen
buffer_store_short v25, v211, s[44:47], 0 offen
v_nop
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
s_lshl_b32 s52, s67, 2
s_add_u32 s44, s44, s52
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 4
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v235, 8
v_add_f16_e64 v26, v26, s55
v_mul_f16_e64 v237, v26, s36
v_cmp_lt_f16_e64 vcc, v26, 0
v_cndmask_b32_e32 v26, v26, v237, vcc
v_add_f16_e64 v29, v29, s55
v_mul_f16_e64 v237, v29, s36
v_cmp_lt_f16_e64 vcc, v29, 0
v_cndmask_b32_e32 v29, v29, v237, vcc
buffer_store_short v26, v205, s[44:47], 0 offen
buffer_store_short v29, v209, s[44:47], 0 offen
v_add_f16_e64 v27, v27, s55
v_mul_f16_e64 v237, v27, s36
v_cmp_lt_f16_e64 vcc, v27, 0
v_cndmask_b32_e32 v27, v27, v237, vcc
v_add_f16_e64 v30, v30, s55
v_mul_f16_e64 v237, v30, s36
v_cmp_lt_f16_e64 vcc, v30, 0
v_cndmask_b32_e32 v30, v30, v237, vcc
buffer_store_short v27, v206, s[44:47], 0 offen
buffer_store_short v30, v210, s[44:47], 0 offen
v_add_f16_e64 v28, v28, s55
v_mul_f16_e64 v237, v28, s36
v_cmp_lt_f16_e64 vcc, v28, 0
v_cndmask_b32_e32 v28, v28, v237, vcc
v_add_f16_e64 v31, v31, s55
v_mul_f16_e64 v237, v31, s36
v_cmp_lt_f16_e64 vcc, v31, 0
v_cndmask_b32_e32 v31, v31, v237, vcc
buffer_store_short v28, v207, s[44:47], 0 offen
buffer_store_short v31, v211, s[44:47], 0 offen
v_nop
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v235, 9
v_add_f16_e64 v32, v32, s55
v_mul_f16_e64 v237, v32, s36
v_cmp_lt_f16_e64 vcc, v32, 0
v_cndmask_b32_e32 v32, v32, v237, vcc
v_add_f16_e64 v35, v35, s55
v_mul_f16_e64 v237, v35, s36
v_cmp_lt_f16_e64 vcc, v35, 0
v_cndmask_b32_e32 v35, v35, v237, vcc
buffer_store_short v32, v205, s[44:47], 0 offen
buffer_store_short v35, v209, s[44:47], 0 offen
v_add_f16_e64 v33, v33, s55
v_mul_f16_e64 v237, v33, s36
v_cmp_lt_f16_e64 vcc, v33, 0
v_cndmask_b32_e32 v33, v33, v237, vcc
v_add_f16_e64 v36, v36, s55
v_mul_f16_e64 v237, v36, s36
v_cmp_lt_f16_e64 vcc, v36, 0
v_cndmask_b32_e32 v36, v36, v237, vcc
buffer_store_short v33, v206, s[44:47], 0 offen
buffer_store_short v36, v210, s[44:47], 0 offen
v_add_f16_e64 v34, v34, s55
v_mul_f16_e64 v237, v34, s36
v_cmp_lt_f16_e64 vcc, v34, 0
v_cndmask_b32_e32 v34, v34, v237, vcc
v_add_f16_e64 v37, v37, s55
v_mul_f16_e64 v237, v37, s36
v_cmp_lt_f16_e64 vcc, v37, 0
v_cndmask_b32_e32 v37, v37, v237, vcc
buffer_store_short v34, v207, s[44:47], 0 offen
buffer_store_short v37, v211, s[44:47], 0 offen
v_nop
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v235, 10
v_add_f16_e64 v38, v38, s55
v_mul_f16_e64 v237, v38, s36
v_cmp_lt_f16_e64 vcc, v38, 0
v_cndmask_b32_e32 v38, v38, v237, vcc
v_add_f16_e64 v41, v41, s55
v_mul_f16_e64 v237, v41, s36
v_cmp_lt_f16_e64 vcc, v41, 0
v_cndmask_b32_e32 v41, v41, v237, vcc
buffer_store_short v38, v205, s[44:47], 0 offen
buffer_store_short v41, v209, s[44:47], 0 offen
v_add_f16_e64 v39, v39, s55
v_mul_f16_e64 v237, v39, s36
v_cmp_lt_f16_e64 vcc, v39, 0
v_cndmask_b32_e32 v39, v39, v237, vcc
v_add_f16_e64 v42, v42, s55
v_mul_f16_e64 v237, v42, s36
v_cmp_lt_f16_e64 vcc, v42, 0
v_cndmask_b32_e32 v42, v42, v237, vcc
buffer_store_short v39, v206, s[44:47], 0 offen
buffer_store_short v42, v210, s[44:47], 0 offen
v_add_f16_e64 v40, v40, s55
v_mul_f16_e64 v237, v40, s36
v_cmp_lt_f16_e64 vcc, v40, 0
v_cndmask_b32_e32 v40, v40, v237, vcc
v_add_f16_e64 v43, v43, s55
v_mul_f16_e64 v237, v43, s36
v_cmp_lt_f16_e64 vcc, v43, 0
v_cndmask_b32_e32 v43, v43, v237, vcc
buffer_store_short v40, v207, s[44:47], 0 offen
buffer_store_short v43, v211, s[44:47], 0 offen
v_nop
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v235, 11
v_add_f16_e64 v44, v44, s55
v_mul_f16_e64 v237, v44, s36
v_cmp_lt_f16_e64 vcc, v44, 0
v_cndmask_b32_e32 v44, v44, v237, vcc
v_add_f16_e64 v47, v47, s55
v_mul_f16_e64 v237, v47, s36
v_cmp_lt_f16_e64 vcc, v47, 0
v_cndmask_b32_e32 v47, v47, v237, vcc
buffer_store_short v44, v205, s[44:47], 0 offen
buffer_store_short v47, v209, s[44:47], 0 offen
v_add_f16_e64 v45, v45, s55
v_mul_f16_e64 v237, v45, s36
v_cmp_lt_f16_e64 vcc, v45, 0
v_cndmask_b32_e32 v45, v45, v237, vcc
v_add_f16_e64 v48, v48, s55
v_mul_f16_e64 v237, v48, s36
v_cmp_lt_f16_e64 vcc, v48, 0
v_cndmask_b32_e32 v48, v48, v237, vcc
buffer_store_short v45, v206, s[44:47], 0 offen
buffer_store_short v48, v210, s[44:47], 0 offen
v_add_f16_e64 v46, v46, s55
v_mul_f16_e64 v237, v46, s36
v_cmp_lt_f16_e64 vcc, v46, 0
v_cndmask_b32_e32 v46, v46, v237, vcc
v_add_f16_e64 v49, v49, s55
v_mul_f16_e64 v237, v49, s36
v_cmp_lt_f16_e64 vcc, v49, 0
v_cndmask_b32_e32 v49, v49, v237, vcc
buffer_store_short v46, v207, s[44:47], 0 offen
buffer_store_short v49, v211, s[44:47], 0 offen
v_nop
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
s_xor_b32 s18, s18, 0x200000
s_bitcmp1_b32 s13, 0
s_addc_u32 s52, s13, 0
s_mul_i32 s94, s60, s61
s_mul_i32 s94, s94, s52
s_add_u32 s52, s93, s92
s_cmp_lt_i32 s52, 0
s_cbranch_scc0 270
v_and_b32_e32 v205, 0x7f, v1
v_lshrrev_b32_e32 v205, 1, v205
v_bfi_b32 v205, 1, v1, v205
v_and_b32_e64 v206, v1, 2
v_mad_u32_u24 v205, v206, 16, v205
v_lshlrev_b32_e32 v205, 2, v205
v_add_co_u32_e64 v205, vcc, v205, s97
v_and_b32_e32 v206, 3, v1
v_lshlrev_b32_e32 v206, 2, v206
v_add_co_u32_e64 v206, vcc, v206, s97
ds_read_b32 v239, v206 offset:256
ds_read_b32 v205, v205
s_add_u32 s97, s97, 0x18c
s_cmp_eq_u32 s97, 0xffc0
s_cselect_b32 s97, 0xc1e0, s97
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s95, v205
v_readlane_b32 s54, v239, 0
s_bitcmp1_b32 s54, 18
s_cbranch_scc1 245
v_readlane_b32 s52, v239, 1
v_readlane_b32 s53, v239, 2
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
s_mov_b32 s47, 0x31014000
s_bitcmp1_b32 s18, 7
s_cselect_b32 s51, 0x31014000, 0
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
v_bfe_u32 v237, v205, 16, 16
v_bfe_u32 v238, v205, 0, 16
v_and_b32_e64 v239, v1, 7
v_sub_co_u32_e64 v240, vcc, 7, v239
v_min_u32_e32 v239, v239, v240
v_bfe_u32 v240, v239, 1, 1
v_bfe_u32 v239, v239, 0, 1
v_mov_b32_dpp v237, v237  quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v238, v238  quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
v_add_co_u32_e64 v237, vcc, v237, v240
v_add_co_u32_e64 v238, vcc, v238, v239
v_mov_b32_dpp v239, v205  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[52:53], v239, s12
v_sub_co_u32_e64 v239, vcc, v239, s95
v_mul_lo_u32 v239, v239, s66
v_xor_b32_dpp v240, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v240, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v212, v1, v1  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v212, v1, v1  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_add_co_u32_e64 v212, vcc, v238, v212
v_add_co_u32_e64 v240, vcc, v237, v240
v_mad_i32_i24 v209, v240, s33, v212
v_lshlrev_b32_e32 v209, 1, v209
v_add_co_u32_e64 v209, vcc, v209, v239
v_cmp_ge_u32_e64 s[56:57], v212, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v240, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v209, v209, -1, s[56:57]
v_xor_b32_dpp v240, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v240, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v212, v1, v1  quad_perm:[1,1,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32_e64 v212, vcc, v238, v212
v_add_co_u32_e64 v240, vcc, v237, v240
v_mad_i32_i24 v210, v240, s33, v212
v_lshlrev_b32_e32 v210, 1, v210
v_add_co_u32_e64 v210, vcc, v210, v239
v_cmp_ge_u32_e64 s[56:57], v212, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v240, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v210, v210, -1, s[56:57]
v_xor_b32_dpp v240, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v240, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v212, v1, v1  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v212, v1, v1  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xa
v_add_co_u32_e64 v212, vcc, v238, v212
v_add_co_u32_e64 v240, vcc, v237, v240
v_mad_i32_i24 v211, v240, s33, v212
v_lshlrev_b32_e32 v211, 1, v211
v_add_co_u32_e64 v211, vcc, v211, v239
v_cmp_ge_u32_e64 s[56:57], v212, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v240, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v211, v211, -1, s[56:57]
v_bfe_u32 v237, v205, 16, 16
v_bfe_u32 v238, v205, 0, 16
v_and_b32_e64 v239, v1, 7
v_sub_co_u32_e64 v240, vcc, 7, v239
v_min_u32_e32 v239, v239, v240
v_bfe_u32 v240, v239, 1, 1
v_bfe_u32 v239, v239, 0, 1
v_mov_b32_dpp v237, v237  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v238, v238  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_add_co_u32_e64 v237, vcc, v237, v240
v_add_co_u32_e64 v238, vcc, v238, v239
v_mov_b32_dpp v239, v205  quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[52:53], v239, s12
v_sub_co_u32_e64 v239, vcc, v239, s95
v_mul_lo_u32 v239, v239, s66
v_xor_b32_dpp v240, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v240, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v208, v1, v1  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v208, v1, v1  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_add_co_u32_e64 v208, vcc, v238, v208
v_add_co_u32_e64 v240, vcc, v237, v240
v_mad_i32_i24 v205, v240, s33, v208
v_lshlrev_b32_e32 v205, 1, v205
v_add_co_u32_e64 v205, vcc, v205, v239
v_cmp_ge_u32_e64 s[56:57], v208, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v240, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v205, v205, -1, s[56:57]
v_xor_b32_dpp v240, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v240, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v208, v1, v1  quad_perm:[1,1,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32_e64 v208, vcc, v238, v208
v_add_co_u32_e64 v240, vcc, v237, v240
v_mad_i32_i24 v206, v240, s33, v208
v_lshlrev_b32_e32 v206, 1, v206
v_add_co_u32_e64 v206, vcc, v206, v239
v_cmp_ge_u32_e64 s[56:57], v208, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v240, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v206, v206, -1, s[56:57]
v_xor_b32_dpp v240, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v240, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v208, v1, v1  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v208, v1, v1  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xa
v_add_co_u32_e64 v208, vcc, v238, v208
v_add_co_u32_e64 v240, vcc, v237, v240
v_mad_i32_i24 v207, v240, s33, v208
v_lshlrev_b32_e32 v207, 1, v207
v_add_co_u32_e64 v207, vcc, v207, v239
v_cmp_ge_u32_e64 s[56:57], v208, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v240, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v207, v207, -1, s[56:57]
v_and_b32_e64 v235, v1, 63
v_lshlrev_b32_e32 v235, 1, v235
s_barrier
buffer_load_ushort v235, v235, s[48:51], 0 offen
s_branch 62781
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
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
