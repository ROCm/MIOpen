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

.macro _v_pk_fmac_f16_gfx10_quad_perm vdst:req, vsrc0:req, vsrc1:req, q0:req, q1:req, q2:req, q3:req
    .long  0x780000FA + ((\vdst << 17) + (\vsrc1 << 9))
    .long  0xFF000000 + (\vsrc0 + (\q0 << 8) + (\q1 << 10) + (\q2 << 12) + (\q3 << 14))
.endm

s_version 0x2004
s_inst_prefetch 0x3
s_inst_prefetch 0x3
v_mov_b32_e32 v1, v0
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_mov_b32_e32 v196, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s76, 0xc220
s_mov_b32 s75, 0xc220
v_and_b32_e32 v197, 0xc0, v0
v_add_co_u32 v1, vcc, v0, v197
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
v_ffbh_u32_e32 v200, s17
v_lshlrev_b32_e64 v201, v200, s17
v_and_b32_e32 v199, 0xffffff00, v201
v_cmp_eq_u32_e32 vcc, 0x80000000, v201
v_cvt_f32_u32_e32 v199, v199
v_rcp_f32_e32 v197, v199
v_sub_co_ci_u32_e32 v198, vcc, 32, v200, vcc
v_cvt_f32_ubyte0_e32 v200, v201
v_fma_f32 v199, v199, v197, -1.0
v_fma_f32 v199, v200, v197, v199
v_fmaak_f32 v199, v199, v197, 0x9f000000
v_mul_f32_e32 v199, 0x5f800000, v199
v_mov_b32_e32 v200, 0
v_cvt_flr_i32_f32_e64 v199, -v199
v_lshl_add_u32 v197, v197, 9, v199
v_mad_u64_u32 v[200:201], vcc, v201, v197, v[200:201]
v_sub_co_ci_u32_e64 v197, vcc, v197, -1, vcc
v_mul_hi_u32 v199, s8, v197
v_add_co_u32 v197, vcc, v199, s8
v_add_co_ci_u32_e64 v199, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v198
v_cndmask_b32_e32 v197, v197, v199, vcc
v_alignbit_b32 v197, v199, v197, v198
s_nop 0
v_readfirstlane_b32 s81, v197
s_mul_i32 s82, s81, s17
s_sub_u32 s8, s8, s82
s_waitcnt lgkmcnt(0)
s_lshl_b32 s85, s85, 1
s_lshl_b64 s[86:87], s[86:87], 1
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
s_lshl_b32 s47, s48, 1
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
v_cvt_f16_f32_e32 v197, s36
s_nop 0
v_readfirstlane_b32 s36, v197
s_and_b32 s81, 1, s30
s_addc_u32 s81, s32, 1
s_ashr_i32 s81, s81, 1
s_add_u32 s77, s81, 2
v_mov_b32_e32 v198, 0x55555556
v_mul_hi_u32 v198, v198, s77
s_nop 0
v_readfirstlane_b32 s77, v198
s_andn2_b32 s81, 1, s31
s_addc_u32 s81, s33, 1
s_ashr_i32 s81, s81, 1
s_add_u32 s80, s81, 2
v_mov_b32_e32 v198, 0x55555556
v_mul_hi_u32 v198, v198, s80
s_nop 0
v_readfirstlane_b32 s80, v198
s_sub_u32 s55, 0, s80
s_sub_u32 s54, 0, s77
s_add_u32 s9, s28, 1
v_mov_b32_e32 v198, 0x80000000
v_mul_hi_u32 v198, v198, s9
s_nop 0
v_readfirstlane_b32 s9, v198
s_add_u32 s10, s29, 1
v_mov_b32_e32 v198, 0x80000000
v_mul_hi_u32 v198, v198, s10
s_nop 0
v_readfirstlane_b32 s10, v198
v_mad_i32_i24 v197, 2, s9, -1
v_sub_co_u32 v197, vcc, v197, s28
v_add_co_ci_u32_e64 v197, vcc, 0, 0, vcc
s_nop 0
v_readfirstlane_b32 s81, v197
s_and_b32 s81, s81, 1
s_and_b32 s81, s81, s9
s_add_u32 s9, s9, s81
v_readfirstlane_b32 s82, v1
s_and_b32 s83, s82, 64
s_cselect_b32 s83, 0x80000, 0
s_or_b32 s18, s18, s83
s_lshl_b32 s49, s47, 1
s_sub_u32 s50, 0, s49
s_subb_u32 s51, 0, 0
s_bitset1_b32 s18, 23
s_mov_b32 s49, s47
s_mov_b32 s50, s47
s_mov_b32 s51, 0
s_add_u32 s10, s10, 1
s_and_b32 s10, s10, -2
s_branch 16
s_and_b32 s83, s13, 3
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
s_add_u32 s49, s49, s49
v_bfe_u32 v198, v1, 2, 6
v_lshrrev_b32_e32 v191, 1, v198
s_bitcmp0_b32 s82, 8
s_cselect_b32 s83, 0x1000000, 0
s_or_b32 s83, s83, 0x100000
s_and_b32 s83, s18, s83
s_cselect_b32 s83, 0, 15
v_bfi_b32 v191, s83, v198, v191
v_bfe_u32 v198, s82, 8, 1
v_xor_b32_e64 v198, v198, 1
v_lshrrev_b32_e32 v191, v198, v191
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
v_mov_b32_e32 v198, s8
v_mov_b32_e32 v199, s17
v_and_b32_e32 v200, 3, v1
v_cmp_eq_u32_e32 vcc, 2, v200
v_cndmask_b32_e32 v198, v198, v199, vcc
v_cmp_eq_u32_e32 vcc, 1, v200
v_cndmask_b32_e32 v201, 0, v191, vcc
s_bitcmp1_b32 s18, 20
s_cbranch_scc0 4
v_add_co_u32 v199, vcc, v191, 8
v_cmp_eq_u32_e32 vcc, 0, v200
v_cndmask_b32_e32 v201, v201, v199, vcc
v_cmp_eq_u32_e64 s[82:83], 3, v200
v_bfe_u32 v189, v201, 0, 5
v_mad_u32_u24 v189, v198, 32, v189
v_ffbh_u32_e32 v204, s80
v_lshlrev_b32_e64 v205, v204, s80
v_and_b32_e32 v203, 0xffffff00, v205
v_cmp_eq_u32_e32 vcc, 0x80000000, v205
v_cvt_f32_u32_e32 v203, v203
v_rcp_f32_e32 v190, v203
v_sub_co_ci_u32_e32 v202, vcc, 32, v204, vcc
v_cvt_f32_ubyte0_e32 v204, v205
v_fma_f32 v203, v203, v190, -1.0
v_fma_f32 v203, v204, v190, v203
v_fmaak_f32 v203, v203, v190, 0x9f000000
v_mul_f32_e32 v203, 0x5f800000, v203
v_mov_b32_e32 v204, 0
v_cvt_flr_i32_f32_e64 v203, -v203
v_lshl_add_u32 v190, v190, 9, v203
v_mad_u64_u32 v[204:205], vcc, v205, v190, v[204:205]
v_sub_co_ci_u32_e64 v190, vcc, v190, -1, vcc
v_mul_hi_u32 v203, v189, v190
v_add_co_u32 v190, vcc, v203, v189
v_add_co_ci_u32_e64 v203, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v202
v_cndmask_b32_e32 v190, v190, v203, vcc
v_alignbit_b32 v190, v203, v190, v202
v_mad_i32_i24 v188, v190, s55, v189
v_lshrrev_b32_e32 v189, 5, v201
v_mad_u32_u24 v189, v190, 1, v189
v_cndmask_b32_e64 v189, v189, 1, s[82:83]
v_ffbh_u32_e32 v204, s77
v_lshlrev_b32_e64 v205, v204, s77
v_and_b32_e32 v203, 0xffffff00, v205
v_cmp_eq_u32_e32 vcc, 0x80000000, v205
v_cvt_f32_u32_e32 v203, v203
v_rcp_f32_e32 v190, v203
v_sub_co_ci_u32_e32 v202, vcc, 32, v204, vcc
v_cvt_f32_ubyte0_e32 v204, v205
v_fma_f32 v203, v203, v190, -1.0
v_fma_f32 v203, v204, v190, v203
v_fmaak_f32 v203, v203, v190, 0x9f000000
v_mul_f32_e32 v203, 0x5f800000, v203
v_mov_b32_e32 v204, 0
v_cvt_flr_i32_f32_e64 v203, -v203
v_lshl_add_u32 v190, v190, 9, v203
v_mad_u64_u32 v[204:205], vcc, v205, v190, v[204:205]
v_sub_co_ci_u32_e64 v190, vcc, v190, -1, vcc
v_mul_hi_u32 v203, v189, v190
v_add_co_u32 v190, vcc, v203, v189
v_add_co_ci_u32_e64 v203, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v202
v_cndmask_b32_e32 v190, v190, v203, vcc
v_alignbit_b32 v190, v203, v190, v202
v_mad_i32_i24 v189, v190, s54, v189
v_readlane_b32 s56, v188, 2
v_readlane_b32 s57, v189, 2
v_readlane_b32 s58, v190, 2
v_readlane_b32 s59, v189, 3
v_readlane_b32 s60, v190, 3
v_add_co_u32 v188, vcc, v188, s55
v_add_co_u32 v189, vcc, v189, s54
v_mov_b32_dpp v190, v190 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v188, v188 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v189, v189 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x11014000
s_mov_b32 s82, 0x80000000
s_mov_b32 s83, 0x11014000
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 8
v_xor_b32_dpp v192, v1, v1 quad_perm:[2,3,2,1] row_mask:0xf bank_mask:0xf
v_subrev_co_u32 v192, vcc, 1, v192
v_cvt_f16_i16_e32 v192, v192
v_pk_add_f16 v192, v192, 0 op_sel_hi:[0,0]
s_branch 7
v_xor_b32_dpp v192, v1, v1 quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32 v192, vcc, 1, v192
v_cvt_f16_i16_e32 v192, v192
v_pk_add_f16 v192, v192, 0 op_sel_hi:[0,0]
v_mov_b32_e32 v193, 1
v_xor_b32_dpp v193, v1, v1 quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v193, v1, v1 quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32 v193, vcc, 1, v193
v_mov_b32_e32 v194, 1
v_xor_b32_dpp v194, v1, v1 quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v194, v1, v1 quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32 v194, vcc, 1, v194
v_cvt_f32_i32_e32 v193, v193
v_cvt_f32_i32_e32 v194, v194
v_lshrrev_b32_e64 v197, 2, s71
v_and_b32_e32 v198, 3, v1
v_bfe_u32 v199, v1, 4, 3
v_mad_u32_u24 v187, v199, 4, v198
v_lshlrev_b32_e32 v187, 4, v187
v_mad_u32_u24 v178, v197, 4, v198
v_lshlrev_b32_e32 v178, 4, v178
v_bfe_u32 v197, v1, 2, 2
v_and_b32_e32 v198, 1, v197
v_mad_u32_u24 v200, v197, 16, v198
v_lshlrev_b32_e32 v200, 6, v200
v_xor_b32_e32 v178, v178, v200
v_mul_u32_u24_e32 v200, 0x400, v197
v_xor_b32_e32 v187, v187, v200
s_lshr_b32 s71, s71, 1
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 61
s_and_b32 s78, s18, 0x1100000
s_addc_u32 s78, 0, 0
v_lshrrev_b32_e32 v200, 1, v1
s_mul_i32 s77, 60, s78
s_sub_u32 s77, 63, s77
v_bfi_b32 v200, s77, v1, v200
v_and_b32_e32 v197, 1, v200
v_bfe_u32 v198, v200, 1, 1
v_xor_b32_e32 v197, v197, v198
v_bfe_u32 v199, v200, 3, 1
v_mad_u32_u24 v198, v198, 2, v199
v_mul_u32_u24_e32 v197, 0x118, v197
v_bfe_u32 v199, v200, 2, 1
v_mad_u32_u24 v198, v198, 2, v197
v_xor_b32_e32 v198, v198, v199
v_and_b32_e32 v199, 0xf0, v200
v_xor_b32_e32 v198, v198, v199
s_mul_i32 s77, 4, s78
s_sub_u32 s77, 6, s77
v_bfe_u32 v200, v1, s77, 1
v_mul_u32_u24_e32 v200, 0x1040, v200
v_xor_b32_e32 v180, 0x314, v198
v_xor_b32_e32 v181, 0x31c, v198
v_xor_b32_e32 v182, 8, v198
v_mov_b32_e32 v179, v198
v_mad_u32_u24 v179, 4, v179, v200
v_mad_u32_u24 v180, 4, v180, v200
v_mad_u32_u24 v181, 4, v181, v200
v_mad_u32_u24 v182, 4, v182, v200
s_mov_b32 s77, 0x1040
s_and_b32 s78, s18, 0x1100000
s_cselect_b32 s77, 0x80, s77
v_add_co_u32 v183, vcc, v179, s77
v_add_co_u32 v184, vcc, v180, s77
v_add_co_u32 v185, vcc, v181, s77
v_add_co_u32 v186, vcc, v182, s77
s_branch 57
s_bfe_u32 s78, s18, 0x10014
v_lshrrev_b32_e32 v200, 1, v1
s_mul_i32 s77, 60, s78
s_sub_u32 s77, 63, s77
v_bfi_b32 v200, s77, v1, v200
v_and_b32_e32 v197, 1, v200
v_bfe_u32 v198, v200, 1, 1
v_bfe_u32 v199, v200, 3, 1
v_xor_b32_e32 v197, v197, v198
v_mad_u32_u24 v198, v198, 2, v199
v_mul_u32_u24_e32 v197, 0x109, v197
v_bfe_u32 v199, v200, 2, 1
v_mad_u32_u24 v198, v198, 2, v197
v_xor_b32_e32 v198, v198, v199
v_and_b32_e32 v199, 0xf0, v200
v_or_b32_e32 v198, v198, v199
s_mul_i32 s77, 4, s78
s_sub_u32 s77, 6, s77
v_bfe_u32 v200, v1, s77, 1
v_mul_u32_u24_e32 v200, 0x1040, v200
v_mad_u32_u24 v179, 4, v198, v200
v_xor_b32_e32 v180, 0x307, v198
v_mad_u32_u24 v180, 4, v180, v200
v_xor_b32_e32 v181, 0x30f, v198
v_mad_u32_u24 v181, 4, v181, v200
v_xor_b32_e32 v182, 8, v198
v_mad_u32_u24 v182, 4, v182, v200
s_mov_b32 s77, 0x1040
s_bitcmp1_b32 s18, 20
s_cselect_b32 s77, 0x80, s77
v_add_co_u32 v183, vcc, v179, s77
v_add_co_u32 v184, vcc, v180, s77
v_add_co_u32 v185, vcc, v181, s77
v_add_co_u32 v186, vcc, v182, s77
v_subrev_co_u32 v188, vcc, s56, v188
v_mov_b32_e32 v198, s55
v_cmp_lt_i32_e32 vcc, v188, v198
v_sub_co_ci_u32_e64 v197, vcc, 0, 0, vcc
v_mad_i32_i24 v188, v197, s55, v188
v_mad_i32_i24 v190, v197, s60, v190
v_mad_i32_i24 v189, v197, s59, v189
v_mov_b32_e32 v198, s54
v_cmp_lt_i32_e32 vcc, v189, v198
v_sub_co_ci_u32_e64 v197, vcc, 0, 0, vcc
v_add_co_u32 v190, vcc, v190, v197
v_mad_i32_i24 v189, v197, v198, v189
v_subrev_co_u32 v189, vcc, s57, v189
v_cmp_lt_i32_e32 vcc, v189, v198
v_sub_co_ci_u32_e64 v197, vcc, 0, 0, vcc
v_add_co_u32 v190, vcc, v190, v197
v_mad_i32_i24 v189, v197, s54, v189
v_subrev_co_u32 v190, vcc, s58, v190
s_mov_b32 s11, 0
s_mov_b32 s38, s28
s_mov_b32 s39, 1
s_mov_b32 s64, 0
s_mov_b32 s65, s16
s_mov_b32 s63, s65
s_sub_u32 s72, -1, s71
s_sub_u32 s72, s72, 16
s_bitset1_b32 s18, 21
s_mov_b32 s83, 0
s_mov_b32 s87, 0
v_add_co_u32 v197, vcc, 2, v1
v_bfe_u32 v197, v197, 2, 1
v_cmp_ne_u32_e64 vcc, v197, 1
s_mov_b64 s[2:3], vcc
s_mov_b32 s73, 38
s_mov_b32 s62, 0
s_bitset1_b32 s18, 26
s_call_b64 s[4:5], 4714
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 1
s_branch 2405
s_mov_b64 vcc, s[2:3]
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v116, v118, -1.0, v116 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_pk_mul_f16 v116, v116, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v119, v117, -1.0, v119 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
v_pk_mul_f16 v119, v119, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
v_pk_add_f16 v117, v118, v117
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
v_pk_mul_f16 v117, v117, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v118, v117, -1.0, v118 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 116, 116, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 117, 117, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 118, 118, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 119, 119, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v104, v2, s[40:43], 0 idxen
buffer_load_short_d16 v106, v68, s[40:43], 0 idxen
buffer_load_short_d16 v105, v3, s[40:43], 0 idxen
buffer_load_short_d16 v107, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v104, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v106, v68, s[88:91], 0 idxen
buffer_load_short_d16_hi v105, v3, s[88:91], 0 idxen
buffer_load_short_d16_hi v107, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v108
ds_read_b128 v[70:73], v187 offset:29440
ds_write_b32 v184, v109
ds_read_b128 v[74:77], v187 offset:29696
ds_write_b32 v185, v110
ds_read_b128 v[86:89], v178 offset:28928
ds_write_b32 v186, v111
ds_read_b128 v[90:93], v178 offset:29056
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 4
s_call_b64 s[4:5], 4507
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v120, v122, -1.0, v120 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_pk_mul_f16 v120, v120, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v123, v121, -1.0, v123 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
v_pk_mul_f16 v123, v123, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
v_pk_add_f16 v121, v122, v121
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
v_pk_mul_f16 v121, v121, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v122, v121, -1.0, v122 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 120, 120, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 121, 121, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 122, 122, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 123, 123, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v108, v102, s[40:43], 0 idxen
buffer_load_short_d16 v110, v152, s[40:43], 0 idxen
buffer_load_short_d16 v109, v103, s[40:43], 0 idxen
buffer_load_short_d16 v111, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v108, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v110, v152, s[88:91], 0 idxen
buffer_load_short_d16_hi v109, v103, s[88:91], 0 idxen
buffer_load_short_d16_hi v111, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v112 offset:8256
ds_read_b128 v[78:81], v187 offset:33536
ds_write_b32 v180, v113 offset:8256
ds_read_b128 v[82:85], v187 offset:33792
ds_write_b32 v181, v114 offset:8256
ds_read_b128 v[94:97], v178 offset:33024
ds_write_b32 v182, v115 offset:8256
ds_read_b128 v[98:101], v178 offset:33152
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 4
s_call_b64 s[4:5], 4307
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v124, v126, -1.0, v124 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_pk_mul_f16 v124, v124, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v127, v125, -1.0, v127 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
v_pk_mul_f16 v127, v127, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
v_pk_add_f16 v125, v126, v125
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
v_pk_mul_f16 v125, v125, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v126, v125, -1.0, v126 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 124, 124, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 125, 125, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 126, 126, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 127, 127, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v112, v2, s[40:43], 0 idxen
buffer_load_short_d16 v114, v68, s[40:43], 0 idxen
buffer_load_short_d16 v113, v3, s[40:43], 0 idxen
buffer_load_short_d16 v115, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v112, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v114, v68, s[88:91], 0 idxen
buffer_load_short_d16_hi v113, v3, s[88:91], 0 idxen
buffer_load_short_d16_hi v115, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v116 offset:8256
ds_read_b128 v[70:73], v187 offset:37696
ds_write_b32 v184, v117 offset:8256
ds_read_b128 v[74:77], v187 offset:37952
ds_write_b32 v185, v118 offset:8256
ds_read_b128 v[86:89], v178 offset:37184
ds_write_b32 v186, v119 offset:8256
ds_read_b128 v[90:93], v178 offset:37312
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 4
s_call_b64 s[4:5], 4107
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v128, v130, -1.0, v128 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_pk_mul_f16 v128, v128, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v131, v129, -1.0, v131 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
v_pk_mul_f16 v131, v131, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
v_pk_add_f16 v129, v130, v129
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
v_pk_mul_f16 v129, v129, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v130, v129, -1.0, v130 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 128, 128, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 129, 129, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 130, 130, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 131, 131, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_barrier
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v116, v102, s[40:43], 0 idxen
buffer_load_short_d16 v118, v152, s[40:43], 0 idxen
buffer_load_short_d16 v117, v103, s[40:43], 0 idxen
buffer_load_short_d16 v119, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v116, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v118, v152, s[88:91], 0 idxen
buffer_load_short_d16_hi v117, v103, s[88:91], 0 idxen
buffer_load_short_d16_hi v119, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v120 offset:16512
ds_read_b128 v[78:81], v187 offset:41792
ds_write_b32 v180, v121 offset:16512
ds_read_b128 v[82:85], v187 offset:42048
ds_write_b32 v181, v122 offset:16512
ds_read_b128 v[94:97], v178 offset:41280
ds_write_b32 v182, v123 offset:16512
ds_read_b128 v[98:101], v178 offset:41408
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 3
s_call_b64 s[4:5], 3906
s_nop 0
s_nop 0
v_pk_fma_f16 v132, v134, -1.0, v132 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_pk_mul_f16 v132, v132, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v135, v133, -1.0, v135 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
v_pk_mul_f16 v135, v135, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
v_pk_add_f16 v133, v134, v133
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
v_pk_mul_f16 v133, v133, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v134, v133, -1.0, v134 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 132, 132, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 133, 133, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 134, 134, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 135, 135, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v120, v2, s[40:43], 0 idxen
buffer_load_short_d16 v122, v68, s[40:43], 0 idxen
buffer_load_short_d16 v121, v3, s[40:43], 0 idxen
buffer_load_short_d16 v123, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v120, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v122, v68, s[88:91], 0 idxen
buffer_load_short_d16_hi v121, v3, s[88:91], 0 idxen
buffer_load_short_d16_hi v123, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v124 offset:16512
ds_read_b128 v[70:73], v187 offset:45952
ds_write_b32 v184, v125 offset:16512
ds_read_b128 v[74:77], v187 offset:46208
ds_write_b32 v185, v126 offset:16512
ds_read_b128 v[86:89], v178 offset:45440
ds_write_b32 v186, v127 offset:16512
ds_read_b128 v[90:93], v178 offset:45568
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 4
s_call_b64 s[4:5], 3707
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v136, v138, -1.0, v136 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_pk_mul_f16 v136, v136, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v139, v137, -1.0, v139 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
v_pk_mul_f16 v139, v139, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
v_pk_add_f16 v137, v138, v137
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
v_pk_mul_f16 v137, v137, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v138, v137, -1.0, v138 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 136, 136, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 137, 137, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 138, 138, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 139, 139, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v124, v102, s[40:43], 0 idxen
buffer_load_short_d16 v126, v152, s[40:43], 0 idxen
buffer_load_short_d16 v125, v103, s[40:43], 0 idxen
buffer_load_short_d16 v127, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v124, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v126, v152, s[88:91], 0 idxen
buffer_load_short_d16_hi v125, v103, s[88:91], 0 idxen
buffer_load_short_d16_hi v127, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v128 offset:24768
ds_read_b128 v[78:81], v187 offset:512
ds_write_b32 v180, v129 offset:24768
ds_read_b128 v[82:85], v187 offset:768
ds_write_b32 v181, v130 offset:24768
ds_read_b128 v[94:97], v178
ds_write_b32 v182, v131 offset:24768
ds_read_b128 v[98:101], v178 offset:128
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 4
s_call_b64 s[4:5], 3507
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v140, v142, -1.0, v140 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_pk_mul_f16 v140, v140, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v143, v141, -1.0, v143 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
v_pk_mul_f16 v143, v143, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
v_pk_add_f16 v141, v142, v141
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
v_pk_mul_f16 v141, v141, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v142, v141, -1.0, v142 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 140, 140, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 141, 141, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 142, 142, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 143, 143, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v128, v2, s[40:43], 0 idxen
buffer_load_short_d16 v130, v68, s[40:43], 0 idxen
buffer_load_short_d16 v129, v3, s[40:43], 0 idxen
buffer_load_short_d16 v131, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v128, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v130, v68, s[88:91], 0 idxen
buffer_load_short_d16_hi v129, v3, s[88:91], 0 idxen
buffer_load_short_d16_hi v131, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v132 offset:24768
ds_read_b128 v[70:73], v187 offset:4672
ds_write_b32 v184, v133 offset:24768
ds_read_b128 v[74:77], v187 offset:4928
ds_write_b32 v185, v134 offset:24768
ds_read_b128 v[86:89], v178 offset:4160
ds_write_b32 v186, v135 offset:24768
ds_read_b128 v[90:93], v178 offset:4288
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 4
s_call_b64 s[4:5], 3307
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v144, v146, -1.0, v144 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_pk_mul_f16 v144, v144, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v147, v145, -1.0, v147 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
v_pk_mul_f16 v147, v147, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
v_pk_add_f16 v145, v146, v145
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
v_pk_mul_f16 v145, v145, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v146, v145, -1.0, v146 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 144, 144, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 145, 145, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 146, 146, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 147, 147, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_barrier
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v132, v102, s[40:43], 0 idxen
buffer_load_short_d16 v134, v152, s[40:43], 0 idxen
buffer_load_short_d16 v133, v103, s[40:43], 0 idxen
buffer_load_short_d16 v135, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v132, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v134, v152, s[88:91], 0 idxen
buffer_load_short_d16_hi v133, v103, s[88:91], 0 idxen
buffer_load_short_d16_hi v135, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v136 offset:33024
ds_read_b128 v[78:81], v187 offset:8768
ds_write_b32 v180, v137 offset:33024
ds_read_b128 v[82:85], v187 offset:9024
ds_write_b32 v181, v138 offset:33024
ds_read_b128 v[94:97], v178 offset:8256
ds_write_b32 v182, v139 offset:33024
ds_read_b128 v[98:101], v178 offset:8384
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 3
s_call_b64 s[4:5], 3106
s_nop 0
s_nop 0
v_pk_fma_f16 v148, v150, -1.0, v148 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_pk_mul_f16 v148, v148, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v151, v149, -1.0, v151 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
v_pk_mul_f16 v151, v151, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
v_pk_add_f16 v149, v150, v149
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
v_pk_mul_f16 v149, v149, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v150, v149, -1.0, v150 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 148, 148, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 149, 149, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 150, 150, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 151, 151, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v136, v2, s[40:43], 0 idxen
buffer_load_short_d16 v138, v68, s[40:43], 0 idxen
buffer_load_short_d16 v137, v3, s[40:43], 0 idxen
buffer_load_short_d16 v139, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v136, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v138, v68, s[88:91], 0 idxen
buffer_load_short_d16_hi v137, v3, s[88:91], 0 idxen
buffer_load_short_d16_hi v139, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v140 offset:33024
ds_read_b128 v[70:73], v187 offset:12928
ds_write_b32 v184, v141 offset:33024
ds_read_b128 v[74:77], v187 offset:13184
ds_write_b32 v185, v142 offset:33024
ds_read_b128 v[86:89], v178 offset:12416
ds_write_b32 v186, v143 offset:33024
ds_read_b128 v[90:93], v178 offset:12544
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 4
s_call_b64 s[4:5], 2907
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v104, v106, -1.0, v104 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_pk_mul_f16 v104, v104, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v107, v105, -1.0, v107 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
v_pk_mul_f16 v107, v107, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
v_pk_add_f16 v105, v106, v105
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
v_pk_mul_f16 v105, v105, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v106, v105, -1.0, v106 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 104, 104, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 105, 105, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 106, 106, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 107, 107, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v140, v102, s[40:43], 0 idxen
buffer_load_short_d16 v142, v152, s[40:43], 0 idxen
buffer_load_short_d16 v141, v103, s[40:43], 0 idxen
buffer_load_short_d16 v143, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v140, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v142, v152, s[88:91], 0 idxen
buffer_load_short_d16_hi v141, v103, s[88:91], 0 idxen
buffer_load_short_d16_hi v143, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v144 offset:41280
ds_read_b128 v[78:81], v187 offset:17024
ds_write_b32 v180, v145 offset:41280
ds_read_b128 v[82:85], v187 offset:17280
ds_write_b32 v181, v146 offset:41280
ds_read_b128 v[94:97], v178 offset:16512
ds_write_b32 v182, v147 offset:41280
ds_read_b128 v[98:101], v178 offset:16640
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 4
s_call_b64 s[4:5], 2707
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v108, v110, -1.0, v108 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_pk_mul_f16 v108, v108, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v111, v109, -1.0, v111 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
v_pk_mul_f16 v111, v111, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
v_pk_add_f16 v109, v110, v109
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
v_pk_mul_f16 v109, v109, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v110, v109, -1.0, v110 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 108, 108, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 109, 109, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 110, 110, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 111, 111, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 1
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v144, v2, s[40:43], 0 idxen
buffer_load_short_d16 v146, v68, s[40:43], 0 idxen
buffer_load_short_d16 v145, v3, s[40:43], 0 idxen
buffer_load_short_d16 v147, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v144, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v146, v68, s[88:91], 0 idxen
buffer_load_short_d16_hi v145, v3, s[88:91], 0 idxen
buffer_load_short_d16_hi v147, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v148 offset:41280
ds_read_b128 v[70:73], v187 offset:21184
ds_write_b32 v184, v149 offset:41280
ds_read_b128 v[74:77], v187 offset:21440
ds_write_b32 v185, v150 offset:41280
ds_read_b128 v[86:89], v178 offset:20672
ds_write_b32 v186, v151 offset:41280
ds_read_b128 v[90:93], v178 offset:20800
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 4
s_call_b64 s[4:5], 2507
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v112, v114, -1.0, v112 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_pk_mul_f16 v112, v112, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v115, v113, -1.0, v115 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
v_pk_mul_f16 v115, v115, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
v_pk_add_f16 v113, v114, v113
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
v_pk_mul_f16 v113, v113, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v114, v113, -1.0, v114 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 112, 112, 192, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 113, 113, 192, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 114, 114, 192, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 115, 115, 192, 2, 2, 1, 1
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_barrier
s_setprio 1
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v148, v102, s[40:43], 0 idxen
buffer_load_short_d16 v150, v152, s[40:43], 0 idxen
buffer_load_short_d16 v149, v103, s[40:43], 0 idxen
buffer_load_short_d16 v151, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v148, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v150, v152, s[88:91], 0 idxen
buffer_load_short_d16_hi v149, v103, s[88:91], 0 idxen
buffer_load_short_d16_hi v151, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v104
ds_read_b128 v[78:81], v187 offset:25280
ds_write_b32 v180, v105
ds_read_b128 v[82:85], v187 offset:25536
ds_write_b32 v181, v106
ds_read_b128 v[94:97], v178 offset:24768
ds_write_b32 v182, v107
ds_read_b128 v[98:101], v178 offset:24896
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 63139
s_call_b64 s[4:5], 2306
s_branch 63137
s_mov_b64 vcc, s[2:3]
v_cndmask_b32_dpp v116, v116, v116, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_cndmask_b32_dpp v117, v117, v117, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_cndmask_b32_dpp v118, v118, v118, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
s_setprio 1
v_dot2_f32_f16 v15, v73, v87, v15
v_cndmask_b32_dpp v119, v119, v119, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_fmac_f16_gfx10_quad_perm 116, 116, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_fmac_f16_gfx10_quad_perm 119, 119, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_add_f16 v117, v116, v119
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_pk_mul_f16 v117, v117, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
v_pk_fma_f16 v118, -1.0, v119, v116 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
v_pk_mul_f16 v118, v118, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v104, v2, s[40:43], 0 idxen
buffer_load_short_d16 v107, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v104, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v107, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v108
ds_read_b128 v[70:73], v187 offset:29440
ds_write_b32 v184, v109
ds_read_b128 v[74:77], v187 offset:29696
ds_write_b32 v185, v110
ds_read_b128 v[86:89], v178 offset:28928
ds_write_b32 v186, v111
ds_read_b128 v[90:93], v178 offset:29056
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 5
s_call_b64 s[4:5], 2116
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_cndmask_b32_dpp v120, v120, v120, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_cndmask_b32_dpp v121, v121, v121, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_cndmask_b32_dpp v122, v122, v122, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
s_setprio 1
v_dot2_f32_f16 v15, v81, v95, v15
v_cndmask_b32_dpp v123, v123, v123, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_fmac_f16_gfx10_quad_perm 120, 120, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_fmac_f16_gfx10_quad_perm 123, 123, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_add_f16 v121, v120, v123
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_pk_mul_f16 v121, v121, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
v_pk_fma_f16 v122, -1.0, v123, v120 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
v_pk_mul_f16 v122, v122, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v108, v102, s[40:43], 0 idxen
buffer_load_short_d16 v111, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v108, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v111, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v112 offset:8256
ds_read_b128 v[78:81], v187 offset:33536
ds_write_b32 v180, v113 offset:8256
ds_read_b128 v[82:85], v187 offset:33792
ds_write_b32 v181, v114 offset:8256
ds_read_b128 v[94:97], v178 offset:33024
ds_write_b32 v182, v115 offset:8256
ds_read_b128 v[98:101], v178 offset:33152
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 5
s_call_b64 s[4:5], 1924
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_cndmask_b32_dpp v124, v124, v124, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_cndmask_b32_dpp v125, v125, v125, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_cndmask_b32_dpp v126, v126, v126, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
s_setprio 1
v_dot2_f32_f16 v15, v73, v87, v15
v_cndmask_b32_dpp v127, v127, v127, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_fmac_f16_gfx10_quad_perm 124, 124, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_fmac_f16_gfx10_quad_perm 127, 127, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_add_f16 v125, v124, v127
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_pk_mul_f16 v125, v125, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
v_pk_fma_f16 v126, -1.0, v127, v124 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
v_pk_mul_f16 v126, v126, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v112, v2, s[40:43], 0 idxen
buffer_load_short_d16 v115, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v112, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v115, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v116 offset:8256
ds_read_b128 v[70:73], v187 offset:37696
ds_write_b32 v184, v117 offset:8256
ds_read_b128 v[74:77], v187 offset:37952
ds_write_b32 v185, v118 offset:8256
ds_read_b128 v[86:89], v178 offset:37184
ds_write_b32 v186, v119 offset:8256
ds_read_b128 v[90:93], v178 offset:37312
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 5
s_call_b64 s[4:5], 1732
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_barrier
v_cndmask_b32_dpp v128, v128, v128, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_cndmask_b32_dpp v129, v129, v129, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_cndmask_b32_dpp v130, v130, v130, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
s_setprio 1
v_dot2_f32_f16 v15, v81, v95, v15
v_cndmask_b32_dpp v131, v131, v131, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_fmac_f16_gfx10_quad_perm 128, 128, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_fmac_f16_gfx10_quad_perm 131, 131, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_add_f16 v129, v128, v131
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_pk_mul_f16 v129, v129, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
v_pk_fma_f16 v130, -1.0, v131, v128 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
v_pk_mul_f16 v130, v130, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v116, v102, s[40:43], 0 idxen
buffer_load_short_d16 v119, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v116, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v119, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v120 offset:16512
ds_read_b128 v[78:81], v187 offset:41792
ds_write_b32 v180, v121 offset:16512
ds_read_b128 v[82:85], v187 offset:42048
ds_write_b32 v181, v122 offset:16512
ds_read_b128 v[94:97], v178 offset:41280
ds_write_b32 v182, v123 offset:16512
ds_read_b128 v[98:101], v178 offset:41408
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 4
s_call_b64 s[4:5], 1539
s_nop 0
s_nop 0
s_nop 0
v_cndmask_b32_dpp v132, v132, v132, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_cndmask_b32_dpp v133, v133, v133, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_cndmask_b32_dpp v134, v134, v134, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
s_setprio 1
v_dot2_f32_f16 v15, v73, v87, v15
v_cndmask_b32_dpp v135, v135, v135, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_fmac_f16_gfx10_quad_perm 132, 132, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_fmac_f16_gfx10_quad_perm 135, 135, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_add_f16 v133, v132, v135
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_pk_mul_f16 v133, v133, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
v_pk_fma_f16 v134, -1.0, v135, v132 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
v_pk_mul_f16 v134, v134, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v120, v2, s[40:43], 0 idxen
buffer_load_short_d16 v123, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v120, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v123, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v124 offset:16512
ds_read_b128 v[70:73], v187 offset:45952
ds_write_b32 v184, v125 offset:16512
ds_read_b128 v[74:77], v187 offset:46208
ds_write_b32 v185, v126 offset:16512
ds_read_b128 v[86:89], v178 offset:45440
ds_write_b32 v186, v127 offset:16512
ds_read_b128 v[90:93], v178 offset:45568
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 5
s_call_b64 s[4:5], 1348
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_cndmask_b32_dpp v136, v136, v136, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_cndmask_b32_dpp v137, v137, v137, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_cndmask_b32_dpp v138, v138, v138, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
s_setprio 1
v_dot2_f32_f16 v15, v81, v95, v15
v_cndmask_b32_dpp v139, v139, v139, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_fmac_f16_gfx10_quad_perm 136, 136, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_fmac_f16_gfx10_quad_perm 139, 139, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_add_f16 v137, v136, v139
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_pk_mul_f16 v137, v137, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
v_pk_fma_f16 v138, -1.0, v139, v136 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
v_pk_mul_f16 v138, v138, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v124, v102, s[40:43], 0 idxen
buffer_load_short_d16 v127, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v124, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v127, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v128 offset:24768
ds_read_b128 v[78:81], v187 offset:512
ds_write_b32 v180, v129 offset:24768
ds_read_b128 v[82:85], v187 offset:768
ds_write_b32 v181, v130 offset:24768
ds_read_b128 v[94:97], v178
ds_write_b32 v182, v131 offset:24768
ds_read_b128 v[98:101], v178 offset:128
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 5
s_call_b64 s[4:5], 1156
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_cndmask_b32_dpp v140, v140, v140, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_cndmask_b32_dpp v141, v141, v141, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_cndmask_b32_dpp v142, v142, v142, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
s_setprio 1
v_dot2_f32_f16 v15, v73, v87, v15
v_cndmask_b32_dpp v143, v143, v143, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_fmac_f16_gfx10_quad_perm 140, 140, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_fmac_f16_gfx10_quad_perm 143, 143, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_add_f16 v141, v140, v143
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_pk_mul_f16 v141, v141, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
v_pk_fma_f16 v142, -1.0, v143, v140 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
v_pk_mul_f16 v142, v142, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v128, v2, s[40:43], 0 idxen
buffer_load_short_d16 v131, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v128, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v131, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v132 offset:24768
ds_read_b128 v[70:73], v187 offset:4672
ds_write_b32 v184, v133 offset:24768
ds_read_b128 v[74:77], v187 offset:4928
ds_write_b32 v185, v134 offset:24768
ds_read_b128 v[86:89], v178 offset:4160
ds_write_b32 v186, v135 offset:24768
ds_read_b128 v[90:93], v178 offset:4288
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 5
s_call_b64 s[4:5], 964
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_barrier
v_cndmask_b32_dpp v144, v144, v144, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_cndmask_b32_dpp v145, v145, v145, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_cndmask_b32_dpp v146, v146, v146, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
s_setprio 1
v_dot2_f32_f16 v15, v81, v95, v15
v_cndmask_b32_dpp v147, v147, v147, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_fmac_f16_gfx10_quad_perm 144, 144, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_fmac_f16_gfx10_quad_perm 147, 147, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_add_f16 v145, v144, v147
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_pk_mul_f16 v145, v145, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
v_pk_fma_f16 v146, -1.0, v147, v144 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
v_pk_mul_f16 v146, v146, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v132, v102, s[40:43], 0 idxen
buffer_load_short_d16 v135, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v132, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v135, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v136 offset:33024
ds_read_b128 v[78:81], v187 offset:8768
ds_write_b32 v180, v137 offset:33024
ds_read_b128 v[82:85], v187 offset:9024
ds_write_b32 v181, v138 offset:33024
ds_read_b128 v[94:97], v178 offset:8256
ds_write_b32 v182, v139 offset:33024
ds_read_b128 v[98:101], v178 offset:8384
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 4
s_call_b64 s[4:5], 771
s_nop 0
s_nop 0
s_nop 0
v_cndmask_b32_dpp v148, v148, v148, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_cndmask_b32_dpp v149, v149, v149, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_cndmask_b32_dpp v150, v150, v150, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
s_setprio 1
v_dot2_f32_f16 v15, v73, v87, v15
v_cndmask_b32_dpp v151, v151, v151, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_fmac_f16_gfx10_quad_perm 148, 148, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_fmac_f16_gfx10_quad_perm 151, 151, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_add_f16 v149, v148, v151
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_pk_mul_f16 v149, v149, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
v_pk_fma_f16 v150, -1.0, v151, v148 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
v_pk_mul_f16 v150, v150, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v136, v2, s[40:43], 0 idxen
buffer_load_short_d16 v139, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v136, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v139, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v140 offset:33024
ds_read_b128 v[70:73], v187 offset:12928
ds_write_b32 v184, v141 offset:33024
ds_read_b128 v[74:77], v187 offset:13184
ds_write_b32 v185, v142 offset:33024
ds_read_b128 v[86:89], v178 offset:12416
ds_write_b32 v186, v143 offset:33024
ds_read_b128 v[90:93], v178 offset:12544
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 5
s_call_b64 s[4:5], 580
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_cndmask_b32_dpp v104, v104, v104, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_cndmask_b32_dpp v105, v105, v105, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_cndmask_b32_dpp v106, v106, v106, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
s_setprio 1
v_dot2_f32_f16 v15, v81, v95, v15
v_cndmask_b32_dpp v107, v107, v107, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_fmac_f16_gfx10_quad_perm 104, 104, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_fmac_f16_gfx10_quad_perm 107, 107, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_add_f16 v105, v104, v107
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_pk_mul_f16 v105, v105, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
v_pk_fma_f16 v106, -1.0, v107, v104 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
v_pk_mul_f16 v106, v106, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v140, v102, s[40:43], 0 idxen
buffer_load_short_d16 v143, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v140, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v143, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v144 offset:41280
ds_read_b128 v[78:81], v187 offset:17024
ds_write_b32 v180, v145 offset:41280
ds_read_b128 v[82:85], v187 offset:17280
ds_write_b32 v181, v146 offset:41280
ds_read_b128 v[94:97], v178 offset:16512
ds_write_b32 v182, v147 offset:41280
ds_read_b128 v[98:101], v178 offset:16640
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 5
s_call_b64 s[4:5], 388
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_cndmask_b32_dpp v108, v108, v108, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
v_cndmask_b32_dpp v109, v109, v109, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_cndmask_b32_dpp v110, v110, v110, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
s_setprio 1
v_dot2_f32_f16 v15, v73, v87, v15
v_cndmask_b32_dpp v111, v111, v111, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_fmac_f16_gfx10_quad_perm 108, 108, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_fmac_f16_gfx10_quad_perm 111, 111, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_add_f16 v109, v108, v111
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_pk_mul_f16 v109, v109, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
v_pk_fma_f16 v110, -1.0, v111, v108 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
v_pk_mul_f16 v110, v110, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
s_setprio 0
s_add_u32 s40, s40, s49
s_addc_u32 s41, s41, 0
s_add_u32 s88, s88, s49
s_addc_u32 s89, s89, 0
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v144, v2, s[40:43], 0 idxen
buffer_load_short_d16 v147, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v144, v2, s[88:91], 0 idxen
buffer_load_short_d16_hi v147, v69, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v183, v148 offset:41280
ds_read_b128 v[70:73], v187 offset:21184
ds_write_b32 v184, v149 offset:41280
ds_read_b128 v[74:77], v187 offset:21440
ds_write_b32 v185, v150 offset:41280
ds_read_b128 v[86:89], v178 offset:20672
ds_write_b32 v186, v151 offset:41280
ds_read_b128 v[90:93], v178 offset:20800
s_waitcnt vmcnt(28) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 5
s_call_b64 s[4:5], 196
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_barrier
v_cndmask_b32_dpp v112, v112, v112, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
v_cndmask_b32_dpp v113, v113, v113, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_cndmask_b32_dpp v114, v114, v114, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
s_setprio 1
v_dot2_f32_f16 v15, v81, v95, v15
v_cndmask_b32_dpp v115, v115, v115, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_fmac_f16_gfx10_quad_perm 112, 112, 192, 0, 0, 1, 1
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_fmac_f16_gfx10_quad_perm 115, 115, 192, 0, 0, 1, 1
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_add_f16 v113, v112, v115
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_pk_mul_f16 v113, v113, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
v_pk_fma_f16 v114, -1.0, v115, v112 op_sel_hi:[0,1,1]
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
v_pk_mul_f16 v114, v114, 0.5 op_sel_hi:[1,0]
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
s_setprio 0
s_add_u32 s40, s40, s50
s_addc_u32 s41, s41, s51
s_add_u32 s88, s88, s50
s_addc_u32 s89, s89, s51
s_sub_u32 s53, s53, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x3
buffer_load_short_d16 v148, v102, s[40:43], 0 idxen
buffer_load_short_d16 v151, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v148, v102, s[88:91], 0 idxen
buffer_load_short_d16_hi v151, v153, s[88:91], 0 idxen
s_clause 0x7
ds_write_b32 v179, v104
ds_read_b128 v[78:81], v187 offset:25280
ds_write_b32 v180, v105
ds_read_b128 v[82:85], v187 offset:25536
ds_write_b32 v181, v106
ds_read_b128 v[94:97], v178 offset:24768
ds_write_b32 v182, v107
ds_read_b128 v[98:101], v178 offset:24896
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s52, s52, -2
s_cbranch_scc1 63236
s_call_b64 s[4:5], 3
s_branch 63234
s_nop 0
s_nop 0
v_nop
s_cmp_eq_u32 s62, 0
s_cbranch_scc0 8
s_branch 748
s_add_u32 s62, s62, 1
s_andn2_b32 s62, s62, 1
s_bitcmp1_b32 s18, 26
s_cselect_b32 s88, s49, s50
s_cselect_b32 s89, 0, s51
s_sub_u32 s40, s40, s88
s_subb_u32 s41, s41, s89
s_cmp_eq_u32 s73, 0
s_cbranch_scc0 5
s_cbranch_scc1 762
s_nop 0
s_nop 0
s_add_u32 s73, s73, 1
s_andn2_b32 s73, s73, 1
s_min_u32 s52, s62, s73
s_sub_u32 s62, s62, s52
s_sub_u32 s73, s73, s52
s_sub_u32 s52, s52, 2
s_lshr_b32 s88, s49, 1
s_add_u32 s88, s40, s88
s_addc_u32 s89, s41, 0
s_mov_b64 s[90:91], s[42:43]
s_bitcmp1_b32 s18, 18
s_cselect_b32 s91, 0, 0x11014000
s_setpc_b64 s[4:5]
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
s_add_u32 s67, s16, 15
s_lshr_b32 s67, s67, 4
v_mov_b32_e32 v198, s68
v_mul_u32_u24_e32 v198, s67, v198
v_add_co_u32 v198, vcc, s17, v198
v_sub_co_u32 v198, vcc, v198, 1
v_ffbh_u32_e32 v202, s17
v_lshlrev_b32_e64 v203, v202, s17
v_and_b32_e32 v201, 0xffffff00, v203
v_cmp_eq_u32_e32 vcc, 0x80000000, v203
v_cvt_f32_u32_e32 v201, v201
v_rcp_f32_e32 v197, v201
v_sub_co_ci_u32_e32 v200, vcc, 32, v202, vcc
v_cvt_f32_ubyte0_e32 v202, v203
v_fma_f32 v201, v201, v197, -1.0
v_fma_f32 v201, v202, v197, v201
v_fmaak_f32 v201, v201, v197, 0x9f000000
v_mul_f32_e32 v201, 0x5f800000, v201
v_mov_b32_e32 v202, 0
v_cvt_flr_i32_f32_e64 v201, -v201
v_lshl_add_u32 v197, v197, 9, v201
v_mad_u64_u32 v[202:203], vcc, v203, v197, v[202:203]
v_sub_co_ci_u32_e64 v197, vcc, v197, -1, vcc
v_mul_hi_u32 v201, v198, v197
v_add_co_u32 v197, vcc, v201, v198
v_add_co_ci_u32_e64 v201, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v200
v_cndmask_b32_e32 v197, v197, v201, vcc
v_alignbit_b32 v197, v201, v197, v200
s_nop 0
v_readfirstlane_b32 s66, v197
v_mul_u32_u24_e64 v197, v197, s8
v_ffbh_u32_e32 v202, s67
v_lshlrev_b32_e64 v203, v202, s67
v_and_b32_e32 v201, 0xffffff00, v203
v_cmp_eq_u32_e32 vcc, 0x80000000, v203
v_cvt_f32_u32_e32 v201, v201
v_rcp_f32_e32 v198, v201
v_sub_co_ci_u32_e32 v200, vcc, 32, v202, vcc
v_cvt_f32_ubyte0_e32 v202, v203
v_fma_f32 v201, v201, v198, -1.0
v_fma_f32 v201, v202, v198, v201
v_fmaak_f32 v201, v201, v198, 0x9f000000
v_mul_f32_e32 v201, 0x5f800000, v201
v_mov_b32_e32 v202, 0
v_cvt_flr_i32_f32_e64 v201, -v201
v_lshl_add_u32 v198, v198, 9, v201
v_mad_u64_u32 v[202:203], vcc, v203, v198, v[202:203]
v_sub_co_ci_u32_e64 v198, vcc, v198, -1, vcc
v_mul_hi_u32 v201, v197, v198
v_add_co_u32 v198, vcc, v201, v197
v_add_co_ci_u32_e64 v201, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v200
v_cndmask_b32_e32 v198, v198, v201, vcc
v_alignbit_b32 v198, v201, v198, v200
v_readfirstlane_b32 s77, v197
v_readfirstlane_b32 s64, v198
s_mul_i32 s64, s64, s67
s_sub_u32 s64, s77, s64
v_sub_co_u32 v198, vcc, s8, v198
v_sub_co_u32 v198, vcc, s17, v198
v_and_b32_e64 v200, v1, 63
v_cmp_eq_u32_e64 vcc, v200, 0
v_cndmask_b32_e32 v198, 1, v198, vcc
s_sub_u32 s78, 0, s55
s_sub_u32 s79, 0, s54
v_mul_u32_u24_e64 v202, v198, 32
v_ffbh_u32_e32 v204, s78
v_lshlrev_b32_e64 v205, v204, s78
v_and_b32_e32 v206, 0xffffff00, v205
v_cmp_eq_u32_e32 vcc, 0x80000000, v205
v_cvt_f32_u32_e32 v206, v206
v_rcp_f32_e32 v200, v206
v_sub_co_ci_u32_e32 v203, vcc, 32, v204, vcc
v_cvt_f32_ubyte0_e32 v204, v205
v_fma_f32 v206, v206, v200, -1.0
v_fma_f32 v206, v204, v200, v206
v_fmaak_f32 v206, v206, v200, 0x9f000000
v_mul_f32_e32 v206, 0x5f800000, v206
v_mov_b32_e32 v204, 0
v_cvt_flr_i32_f32_e64 v206, -v206
v_lshl_add_u32 v200, v200, 9, v206
v_mad_u64_u32 v[204:205], vcc, v205, v200, v[204:205]
v_sub_co_ci_u32_e64 v200, vcc, v200, -1, vcc
v_mul_hi_u32 v204, v202, v200
v_add_co_u32 v200, vcc, v204, v202
v_add_co_ci_u32_e64 v204, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v203
v_cndmask_b32_e32 v200, v200, v204, vcc
v_alignbit_b32 v200, v204, v200, v203
v_mad_i32_i24 v201, v200, s55, v202
v_mul_u32_u24_e64 v202, v200, 1
v_ffbh_u32_e32 v204, s79
v_lshlrev_b32_e64 v205, v204, s79
v_and_b32_e32 v206, 0xffffff00, v205
v_cmp_eq_u32_e32 vcc, 0x80000000, v205
v_cvt_f32_u32_e32 v206, v206
v_rcp_f32_e32 v200, v206
v_sub_co_ci_u32_e32 v203, vcc, 32, v204, vcc
v_cvt_f32_ubyte0_e32 v204, v205
v_fma_f32 v206, v206, v200, -1.0
v_fma_f32 v206, v204, v200, v206
v_fmaak_f32 v206, v206, v200, 0x9f000000
v_mul_f32_e32 v206, 0x5f800000, v206
v_mov_b32_e32 v204, 0
v_cvt_flr_i32_f32_e64 v206, -v206
v_lshl_add_u32 v200, v200, 9, v206
v_mad_u64_u32 v[204:205], vcc, v205, v200, v[204:205]
v_sub_co_ci_u32_e64 v200, vcc, v200, -1, vcc
v_mul_hi_u32 v204, v202, v200
v_add_co_u32 v200, vcc, v204, v202
v_add_co_ci_u32_e64 v204, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v203
v_cndmask_b32_e32 v200, v200, v204, vcc
v_alignbit_b32 v200, v204, v200, v203
v_mad_i32_i24 v202, v200, s54, v202
v_readfirstlane_b32 s56, v201
v_readfirstlane_b32 s57, v202
v_readfirstlane_b32 s58, v200
v_add_co_u32 v188, vcc, s56, v188
v_add_co_ci_u32_e64 v203, vcc, 0, 0, vcc
v_mad_i32_i24 v188, v203, s55, v188
v_mad_i32_i24 v190, v203, s60, v190
v_mad_i32_i24 v189, v203, s59, v189
v_cmp_ge_i32_e64 vcc, v189, 0
v_add_co_ci_u32_e64 v203, vcc, 0, 0, vcc
v_add_co_u32 v190, vcc, v190, v203
v_mad_i32_i24 v189, v203, s54, v189
v_add_co_u32 v189, vcc, s57, v189
v_add_co_ci_u32_e64 v203, vcc, 0, 0, vcc
v_add_co_u32 v190, vcc, v190, v203
v_mad_i32_i24 v189, v203, s54, v189
v_add_co_u32 v190, vcc, s58, v190
v_readlane_b32 s56, v201, 1
v_readlane_b32 s57, v202, 1
v_readlane_b32 s58, v200, 1
s_add_u32 s65, s64, s66
s_cmp_le_u32 s65, s67
s_cselect_b32 s88, 0x20000, 0
s_cselect_b32 s65, s65, s67
s_or_b32 s18, s18, s88
s_lshl_b32 s64, s64, 4
s_lshl_b32 s65, s65, 4
s_min_u32 s65, s65, s16
s_cmp_eq_u32 s8, s17
s_cselect_b32 s88, 0x20000, 0
s_or_b32 s18, s18, s88
s_bitset1_b32 s18, 16
s_branch 48
s_lshr_b32 s64, s64, 4
s_add_u32 s65, s64, s66
s_sub_u32 s65, s65, s67
s_mov_b32 s64, 0
s_lshl_b32 s65, s65, 4
s_min_u32 s65, s65, s16
s_bitset1_b32 s18, 17
s_branch 12
s_bitset1_b32 s18, 18
s_mov_b32 s43, 0
s_mov_b32 s53, -1
s_mov_b32 s62, 40
s_branch 36
s_add_u32 s63, s63, 16
s_cmp_ge_u32 s63, s65
s_cbranch_scc0 33
s_bitset1_b32 s18, 22
s_sub_u32 s68, s68, s17
s_subb_u32 s69, s69, 0
s_cbranch_scc1 65269
v_add_co_u32 v188, vcc, s56, v188
v_add_co_ci_u32_e64 v197, vcc, 0, 0, vcc
v_mad_i32_i24 v188, v197, s55, v188
v_mad_i32_i24 v190, v197, s60, v190
v_mad_i32_i24 v189, v197, s59, v189
v_cmp_ge_i32_e64 vcc, v189, 0
v_add_co_ci_u32_e64 v197, vcc, 0, 0, vcc
v_add_co_u32 v190, vcc, v190, v197
v_mad_i32_i24 v189, v197, s54, v189
v_add_co_u32 v189, vcc, s57, v189
v_add_co_ci_u32_e64 v197, vcc, 0, 0, vcc
v_add_co_u32 v190, vcc, v190, v197
v_mad_i32_i24 v189, v197, s54, v189
v_add_co_u32 v190, vcc, s58, v190
s_mov_b32 s63, s64
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccz 257
v_subrev_co_u32 v197, vcc, s55, v188
v_subrev_co_u32 v198, vcc, s54, v189
s_bitcmp1_b32 s18, 22
s_cbranch_scc0 66
s_bitset0_b32 s18, 22
s_bfe_u32 s77, s18, 0x10014
v_mul_u32_u24_e32 v200, 3, v197
v_mul_u32_u24_e32 v201, 3, v198
v_cvt_pk_u16_u32 v203, v200, v201
v_and_b32_e64 v200, v1, 1
v_cmp_eq_u32_e64 vcc, v200, 1
v_cndmask_b32_e32 v203, v190, v203, vcc
v_lshrrev_b32_e32 v199, 1, v1
v_bfe_u32 v204, v199, s77, 1
v_lshrrev_b32_e32 v199, 1, v1
v_bfi_b32 v199, 1, v1, v199
v_lshrrev_b32_e32 v200, 2, v1
v_bfi_b32 v200, 1, v1, v200
v_cmp_eq_u32_e64 vcc, s77, 0
v_cndmask_b32_e32 v199, v200, v199, vcc
s_sub_u32 s77, 1, s77
v_lshrrev_b32_e32 v200, s77, v199
v_bfi_b32 v199, 32, v200, v199
v_and_b32_e32 v199, 63, v199
v_add_co_u32 v200, vcc, 16, v199
v_and_b32_e64 v201, v1, 2
v_cmp_eq_u32_e64 vcc, v201, 0
v_cndmask_b32_e32 v200, v200, v199, vcc
v_lshlrev_b32_e32 v201, 14, v204
v_mad_u32_u24 v200, 4, v200, v201
v_add_co_u32 v199, vcc, s75, v200
ds_write_b32 v199, v203
v_writelane_b32 v201, s18, 0
v_writelane_b32 v201, s65, 1
v_writelane_b32 v201, s64, 2
v_and_b32_e64 v199, v1, 63
v_cmp_ge_u32_e64 vcc, v199, 3
v_mov_b32_e32 v202, 0x4000
v_cndmask_b32_e32 v199, v199, v202, vcc
v_mad_i32_i24 v199, v199, 4, s75
ds_write_b32 v199, v201 offset:256
s_add_u32 s75, s75, 0x18c
s_cmp_eq_u32 s75, 0x10000
s_cselect_b32 s75, 0xc220, s75
v_mov_b32_dpp v199, v190 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v197, v197 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v198, v198 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_readfirstlane_b32 s61, v199
v_sub_co_u32 v200, vcc, v199, s61
v_mul_lo_u32 v200, v200, s44
v_and_b32_e64 v204, v1, 3
v_ashrrev_i32_e64 v205, 1, s31
v_subrev_co_u32 v204, vcc, v205, v204
v_ashrrev_i32_e64 v205, 1, s11
v_mad_i32_i24 v201, v205, 2, v204
s_bfe_u32 s77, s18, 0x10014
v_lshrrev_b32_e32 v203, 2, v1
v_and_b32_e32 v203, s77, v203
v_mad_i32_i24 v201, v203, 2, v201
v_add_co_u32 v202, vcc, 1, s38
v_ashrrev_i32_e32 v202, 1, v202
v_add_co_u32 v203, vcc, 1, s30
v_ashrrev_i32_e32 v203, 1, v203
v_sub_nc_i32 v202, v202, v203
v_cmp_ge_u32_e64 s[78:79], v199, s12
v_mad_i32_i24 v197, v197, 3, v201
v_cmp_ge_u32_e64 s[92:93], v197, s15
v_add_co_u32 v197, vcc, v197, v200
s_or_b64 s[92:93], s[92:93], s[78:79]
v_mad_i32_i24 v198, v198, 3, v202
v_cmp_ge_u32_e64 s[94:95], v198, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v2, v198, s15, v197
v_cndmask_b32_e64 v2, v2, -1, s[94:95]
v_add_co_u32 v198, vcc, 1, v198
v_cmp_ge_u32_e64 s[94:95], v198, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v3, v198, s15, v197
v_cndmask_b32_e64 v3, v3, -1, s[94:95]
v_add_co_u32 v198, vcc, 1, v198
v_cmp_ge_u32_e64 s[94:95], v198, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v68, v198, s15, v197
v_cndmask_b32_e64 v68, v68, -1, s[94:95]
v_add_co_u32 v198, vcc, 1, v198
v_cmp_ge_u32_e64 s[94:95], v198, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v69, v198, s15, v197
v_cndmask_b32_e64 v69, v69, -1, s[94:95]
s_bitcmp1_b32 s18, 20
s_cbranch_scc0 60
v_mov_b32_dpp v199, v190 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v197, v188 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v198, v189 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[78:79], v199, s12
v_sub_co_u32 v200, vcc, v199, s61
v_mul_lo_u32 v200, v200, s44
v_sub_co_u32 v197, vcc, v197, s55
v_sub_co_u32 v198, vcc, v198, s54
v_mad_i32_i24 v197, v197, 3, v201
v_cmp_ge_u32_e64 s[92:93], v197, s15
v_add_co_u32 v197, vcc, v197, v200
s_or_b64 s[92:93], s[92:93], s[78:79]
v_mad_i32_i24 v198, v198, 3, v202
v_cmp_ge_u32_e64 s[94:95], v198, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v102, v198, s15, v197
v_cndmask_b32_e64 v102, v102, -1, s[94:95]
v_add_co_u32 v198, vcc, 1, v198
v_cmp_ge_u32_e64 s[94:95], v198, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v103, v198, s15, v197
v_cndmask_b32_e64 v103, v103, -1, s[94:95]
v_add_co_u32 v198, vcc, 1, v198
v_cmp_ge_u32_e64 s[94:95], v198, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v152, v198, s15, v197
v_cndmask_b32_e64 v152, v152, -1, s[94:95]
v_add_co_u32 v198, vcc, 1, v198
v_cmp_ge_u32_e64 s[94:95], v198, s14
s_or_b64 s[94:95], s[92:93], s[94:95]
v_mad_u32_u24 v153, v198, s15, v197
v_cndmask_b32_e64 v153, v153, -1, s[94:95]
s_branch 26
s_bitcmp1_b32 s18, 24
s_cselect_b32 s77, s48, 0
v_add_co_u32 v203, vcc, v2, s77
v_cmp_eq_u32_e64 vcc, v2, -1
v_cndmask_b32_e64 v102, v203, -1, vcc
v_add_co_u32 v203, vcc, v3, s77
v_cmp_eq_u32_e64 vcc, v3, -1
v_cndmask_b32_e64 v103, v203, -1, vcc
v_add_co_u32 v203, vcc, v68, s77
v_cmp_eq_u32_e64 vcc, v68, -1
v_cndmask_b32_e64 v152, v203, -1, vcc
v_add_co_u32 v203, vcc, v69, s77
v_cmp_eq_u32_e64 vcc, v69, -1
v_cndmask_b32_e64 v153, v203, -1, vcc
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 171
s_lshr_b32 s79, -1, 16
s_and_b32 s79, s79, s44
s_lshr_b32 s92, s44, 16
s_mul_i32 s92, s92, s61
s_mul_i32 s40, s79, s61
s_lshl_b32 s79, s92, 16
s_lshr_b32 s92, s92, 16
s_add_u32 s40, s79, s40
s_addc_u32 s41, s92, 0
s_lshl_b64 s[40:41], s[40:41], 1
s_add_u32 s40, s40, s20
s_addc_u32 s41, s41, s21
s_add_u32 s41, s41, 0x20000
s_branch 135
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 154
v_mad_u32_u24 v199, 5, v1, 2
v_lshlrev_b32_e32 v197, 1, v1
v_bfi_b32 v199, 4, v199, v197
v_bfe_u32 v197, v199, 2, 1
v_lshlrev_b32_e32 v199, 1, v197
v_bfe_u32 v197, v1, 1, 1
v_add_co_u32 v197, vcc, v197, v199
v_mad_u32_u24 v197, s11, 2, v197
v_sub_co_u32 v199, vcc, s29, v197
v_sub_co_u32 v199, vcc, v199, 1
s_bfe_u32 s77, s18, 0x10001
v_cmp_eq_u32_e64 vcc, s77, 1
v_cndmask_b32_e32 v197, v197, v199, vcc
v_cmp_ge_u32_e64 s[78:79], v197, s29
s_bfe_u32 s77, s18, 0x10018
v_bfe_u32 v200, v1, 2, s77
v_mul_lo_u32 v200, s48, v200
v_add_co_u32 v197, vcc, v197, v200
v_mul_lo_u32 v198, s70, v191
v_add_co_u32 v198, vcc, v198, v197
s_sub_u32 s77, s28, s38
s_sub_u32 s77, s77, 3
s_bitcmp1_b32 s18, 0
s_cselect_b32 s77, s77, s38
v_mov_b32_e32 v200, s77
v_cmp_ge_u32_e64 s[92:93], v200, s28
v_mad_i32_i24 v2, v200, s29, v198
s_or_b64 s[92:93], s[92:93], s[78:79]
v_cndmask_b32_e64 v2, v2, -1, s[92:93]
v_mov_b32_e32 v3, v2
v_add_co_u32 v200, vcc, v200, 2
v_cmp_ge_u32_e64 s[92:93], v200, s28
v_mad_i32_i24 v69, v200, s29, v198
s_or_b64 s[92:93], s[92:93], s[78:79]
v_cndmask_b32_e64 v69, v69, -1, s[92:93]
v_add_co_u32 v200, vcc, v200, 2
v_cmp_ge_u32_e64 s[92:93], v200, s28
v_mad_i32_i24 v68, v200, s29, v198
s_or_b64 s[92:93], s[92:93], s[78:79]
v_cndmask_b32_e64 v68, v68, -1, s[92:93]
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v2, v3, v69, vcc
v_cndmask_b32_e32 v69, v69, v3, vcc
s_lshl_b32 s92, s70, 3
s_and_b32 s93, s18, 0x1100000
s_cselect_b32 s92, s92, 0
v_add_co_u32 v197, vcc, v2, s92
v_cmp_eq_u32_e64 vcc, v2, -1
v_cndmask_b32_e64 v102, v197, -1, vcc
v_add_co_u32 v197, vcc, v3, s92
v_cmp_eq_u32_e64 vcc, v3, -1
v_cndmask_b32_e64 v103, v197, -1, vcc
v_add_co_u32 v197, vcc, v68, s92
v_cmp_eq_u32_e64 vcc, v68, -1
v_cndmask_b32_e64 v152, v197, -1, vcc
v_add_co_u32 v197, vcc, v69, s92
v_cmp_eq_u32_e64 vcc, v69, -1
v_cndmask_b32_e64 v153, v197, -1, vcc
v_add_co_u32 v197, vcc, v191, s63
v_cmp_lt_u32_e64 vcc, v197, s16
v_cndmask_b32_e32 v2, -1, v2, vcc
v_cndmask_b32_e32 v3, -1, v3, vcc
v_cndmask_b32_e32 v68, -1, v68, vcc
v_cndmask_b32_e32 v69, -1, v69, vcc
s_and_b32 s77, s18, 0x1100000
s_cbranch_scc0 4
v_add_co_u32 v197, vcc, v197, 8
v_cmp_lt_u32_e64 vcc, v197, s16
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
s_lshl_b64 s[40:41], s[40:41], 1
s_add_u32 s40, s40, s22
s_addc_u32 s41, s41, s23
s_add_u32 s41, s41, 0x20000
s_mov_b32 s43, 0x11014000
s_mov_b32 s53, -1
s_bitcmp0_b32 s13, 0
s_cbranch_scc1 5
s_mov_b32 s43, 0
s_bitcmp1_b32 s18, 20
s_addc_u32 s53, 0, 1
s_sub_u32 s40, s40, s47
s_subb_u32 s41, s41, 0
s_add_u32 s89, s13, 1
s_and_b32 s89, s89, -2
s_bfe_u32 s88, s18, 0x10014
s_lshl_b32 s62, s89, s88
s_bitcmp1_b32 s18, 20
s_cselect_b32 s88, 0, 0x2000000
s_bitcmp1_b32 s89, 1
s_cselect_b32 s88, s88, 0
s_xor_b32 s18, s18, s88
s_mov_b64 vcc, s[2:3]
s_branch 64794
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_and_b32 s88, 0x900000, s18
s_subb_u32 s11, s11, 1
s_cbranch_scc0 65108
s_and_b32 s88, 0x900000, s18
s_subb_u32 s11, s10, 1
s_add_u32 s38, s38, 4
s_cmp_ge_u32 s38, s28
s_cbranch_scc0 65102
s_mov_b32 s38, 1
s_cmp_ge_u32 s38, s28
s_addc_u32 s39, s39, 1
s_cmp_gt_u32 s39, 1
s_cbranch_scc0 65097
s_mov_b32 s39, 0
s_mov_b32 s38, 0
s_branch 65058
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_mov_b32 s78, 0x3c3c3c3c
s_mov_b32 s79, s78
v_mov_b32_e32 v197, v4
v_mov_b32_e32 v198, v5
v_mov_b32_e32 v199, v6
v_mov_b32_e32 v200, v7
v_add_f32_dpp v197, v4, v4 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v5, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v6, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v7, v7 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v6, v6, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v7, v7, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v4, v4, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v5, v5, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v5, v6 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v4, v7 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v5, v5, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v4, v4, v4 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v7, v198
v_add_f32_dpp v7, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v6, v197
v_add_f32_dpp v6, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v5, v5, v4 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v4, v7, v6 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v6, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v6, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v5, v199, v5, s[78:79]
v_mov_b32_dpp v6, v6 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v6, v6 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v4, v4
v_cvt_f16_f32_e32 v5, v5
v_cvt_f16_f32_e32 v6, v6
v_mov_b32_e32 v197, v8
v_mov_b32_e32 v198, v9
v_mov_b32_e32 v199, v10
v_mov_b32_e32 v200, v11
v_add_f32_dpp v197, v8, v8 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v9, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v10, v10 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v11, v11 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v10, v10, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v11, v11, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v8, v8, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v9, v9, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v9, v10 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v8, v11 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v9, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v9, v9, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v8, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v8, v8, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v11, v198
v_add_f32_dpp v11, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v10, v197
v_add_f32_dpp v10, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v9, v9, v8 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v7, v11, v10 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v10, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v10, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v8, v199, v9, s[78:79]
v_mov_b32_dpp v9, v10 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v9, v10 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v7, v7
v_cvt_f16_f32_e32 v8, v8
v_cvt_f16_f32_e32 v9, v9
v_mov_b32_e32 v197, v12
v_mov_b32_e32 v198, v13
v_mov_b32_e32 v199, v14
v_mov_b32_e32 v200, v15
v_add_f32_dpp v197, v12, v12 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v13, v13 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v14, v14 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v15, v15 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v14, v14, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v15, v15, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v12, v12, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v13, v13, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v13, v14 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v12, v15 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v13, v13 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v13, v13, v13 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v12, v12 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v12, v12, v12 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v15, v198
v_add_f32_dpp v15, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v14, v197
v_add_f32_dpp v14, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v13, v13, v12 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v10, v15, v14 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v14, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v14, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v11, v199, v13, s[78:79]
v_mov_b32_dpp v12, v14 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v12, v14 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v10, v10
v_cvt_f16_f32_e32 v11, v11
v_cvt_f16_f32_e32 v12, v12
v_mov_b32_e32 v197, v16
v_mov_b32_e32 v198, v17
v_mov_b32_e32 v199, v18
v_mov_b32_e32 v200, v19
v_add_f32_dpp v197, v16, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v17, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v18, v18 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v19, v19 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v18, v18, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v19, v19, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v16, v16, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v17, v17, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v17, v18 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v16, v19 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v17, v17, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v16, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v16, v16, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v19, v198
v_add_f32_dpp v19, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v18, v197
v_add_f32_dpp v18, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v17, v17, v16 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v13, v19, v18 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v18, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v18, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v14, v199, v17, s[78:79]
v_mov_b32_dpp v15, v18 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v15, v18 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v13, v13
v_cvt_f16_f32_e32 v14, v14
v_cvt_f16_f32_e32 v15, v15
v_mov_b32_e32 v197, v20
v_mov_b32_e32 v198, v21
v_mov_b32_e32 v199, v22
v_mov_b32_e32 v200, v23
v_add_f32_dpp v197, v20, v20 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v21, v21 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v22, v22 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v23, v23 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v22, v22, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v23, v23, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v20, v20, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v21, v21, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v21, v22 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v20, v23 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v21, v21 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v21, v21, v21 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v20, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v23, v198
v_add_f32_dpp v23, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v22, v197
v_add_f32_dpp v22, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v21, v21, v20 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v16, v23, v22 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v22, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v22, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v17, v199, v21, s[78:79]
v_mov_b32_dpp v18, v22 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v18, v22 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v16, v16
v_cvt_f16_f32_e32 v17, v17
v_cvt_f16_f32_e32 v18, v18
v_mov_b32_e32 v197, v24
v_mov_b32_e32 v198, v25
v_mov_b32_e32 v199, v26
v_mov_b32_e32 v200, v27
v_add_f32_dpp v197, v24, v24 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v25, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v26, v26 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v27, v27 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v26, v26, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v27, v27, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v24, v24, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v25, v25, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v25, v26 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v24, v27 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v25, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v25, v25, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v24, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v24, v24, v24 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v27, v198
v_add_f32_dpp v27, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v26, v197
v_add_f32_dpp v26, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v25, v25, v24 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v19, v27, v26 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v26, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v26, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v20, v199, v25, s[78:79]
v_mov_b32_dpp v21, v26 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v21, v26 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v19, v19
v_cvt_f16_f32_e32 v20, v20
v_cvt_f16_f32_e32 v21, v21
v_mov_b32_e32 v197, v28
v_mov_b32_e32 v198, v29
v_mov_b32_e32 v199, v30
v_mov_b32_e32 v200, v31
v_add_f32_dpp v197, v28, v28 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v29, v29 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v30, v30 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v31, v31 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v30, v30, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v31, v31, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v28, v28, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v29, v29, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v29, v30 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v28, v31 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v29, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v29, v29, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v28, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v28, v28, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v31, v198
v_add_f32_dpp v31, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v30, v197
v_add_f32_dpp v30, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v29, v29, v28 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v22, v31, v30 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v30, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v30, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v23, v199, v29, s[78:79]
v_mov_b32_dpp v24, v30 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v24, v30 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v22, v22
v_cvt_f16_f32_e32 v23, v23
v_cvt_f16_f32_e32 v24, v24
s_setprio 1
v_mov_b32_e32 v197, v32
v_mov_b32_e32 v198, v33
v_mov_b32_e32 v199, v34
v_mov_b32_e32 v200, v35
v_add_f32_dpp v197, v32, v32 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v33, v33 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v34, v34 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v34, v34, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v35, v35, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v32, v32, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v33, v33, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v33, v34 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v32, v35 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v33, v33 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v33, v33, v33 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v32, v32 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v32, v32, v32 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v35, v198
v_add_f32_dpp v35, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v34, v197
v_add_f32_dpp v34, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v33, v33, v32 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v25, v35, v34 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v34, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v34, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v26, v199, v33, s[78:79]
v_mov_b32_dpp v27, v34 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v27, v34 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v25, v25
v_cvt_f16_f32_e32 v26, v26
v_cvt_f16_f32_e32 v27, v27
v_mov_b32_e32 v197, v36
v_mov_b32_e32 v198, v37
v_mov_b32_e32 v199, v38
v_mov_b32_e32 v200, v39
v_add_f32_dpp v197, v36, v36 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v37, v37 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v38, v38 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v39, v39 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v38, v38, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v39, v39, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v36, v36, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v37, v37, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v37, v38 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v36, v39 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v37, v37 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v37, v37, v37 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v36, v36 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v36, v36, v36 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v39, v198
v_add_f32_dpp v39, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v38, v197
v_add_f32_dpp v38, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v37, v37, v36 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v28, v39, v38 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v38, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v38, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v29, v199, v37, s[78:79]
v_mov_b32_dpp v30, v38 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v30, v38 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v28, v28
v_cvt_f16_f32_e32 v29, v29
v_cvt_f16_f32_e32 v30, v30
v_mov_b32_e32 v197, v40
v_mov_b32_e32 v198, v41
v_mov_b32_e32 v199, v42
v_mov_b32_e32 v200, v43
v_add_f32_dpp v197, v40, v40 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v41, v41 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v42, v42 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v43, v43 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v42, v42, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v43, v43, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v40, v40, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v41, v41, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v41, v42 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v40, v43 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v41, v41 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v41, v41, v41 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v40, v40 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v40, v40, v40 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v43, v198
v_add_f32_dpp v43, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v42, v197
v_add_f32_dpp v42, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v41, v41, v40 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v31, v43, v42 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v42, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v42, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v32, v199, v41, s[78:79]
v_mov_b32_dpp v33, v42 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v33, v42 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v31, v31
v_cvt_f16_f32_e32 v32, v32
v_cvt_f16_f32_e32 v33, v33
v_mov_b32_e32 v197, v44
v_mov_b32_e32 v198, v45
v_mov_b32_e32 v199, v46
v_mov_b32_e32 v200, v47
v_add_f32_dpp v197, v44, v44 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v45, v45 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v46, v46 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v47, v47 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v46, v46, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v47, v47, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v44, v44, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v45, v45, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v45, v46 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v44, v47 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v45, v45 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v45, v45, v45 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v44, v44 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v44, v44, v44 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v47, v198
v_add_f32_dpp v47, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v46, v197
v_add_f32_dpp v46, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v45, v45, v44 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v34, v47, v46 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v46, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v46, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v35, v199, v45, s[78:79]
v_mov_b32_dpp v36, v46 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v36, v46 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v34, v34
v_cvt_f16_f32_e32 v35, v35
v_cvt_f16_f32_e32 v36, v36
v_mov_b32_e32 v197, v48
v_mov_b32_e32 v198, v49
v_mov_b32_e32 v199, v50
v_mov_b32_e32 v200, v51
v_add_f32_dpp v197, v48, v48 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v49, v49 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v50, v50 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v51, v51 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v50, v50, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v51, v51, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v48, v48, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v49, v49, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v49, v50 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v48, v51 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v49, v49 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v49, v49, v49 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v48, v48 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v48, v48, v48 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v51, v198
v_add_f32_dpp v51, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v50, v197
v_add_f32_dpp v50, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v49, v49, v48 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v37, v51, v50 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v50, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v50, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v38, v199, v49, s[78:79]
v_mov_b32_dpp v39, v50 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v39, v50 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v37, v37
v_cvt_f16_f32_e32 v38, v38
v_cvt_f16_f32_e32 v39, v39
v_mov_b32_e32 v197, v52
v_mov_b32_e32 v198, v53
v_mov_b32_e32 v199, v54
v_mov_b32_e32 v200, v55
v_add_f32_dpp v197, v52, v52 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v53, v53 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v54, v54 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v55, v55 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v54, v54, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v55, v55, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v52, v52, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v53, v53, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v53, v54 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v52, v55 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v53, v53 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v53, v53, v53 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v52, v52 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v52, v52, v52 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v55, v198
v_add_f32_dpp v55, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v54, v197
v_add_f32_dpp v54, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v53, v53, v52 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v40, v55, v54 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v54, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v54, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v41, v199, v53, s[78:79]
v_mov_b32_dpp v42, v54 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v42, v54 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v40, v40
v_cvt_f16_f32_e32 v41, v41
v_cvt_f16_f32_e32 v42, v42
v_mov_b32_e32 v197, v56
v_mov_b32_e32 v198, v57
v_mov_b32_e32 v199, v58
v_mov_b32_e32 v200, v59
v_add_f32_dpp v197, v56, v56 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v57, v57 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v58, v58 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v59, v59 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v58, v58, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v59, v59, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v56, v56, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v57, v57, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v57, v58 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v56, v59 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v57, v57 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v57, v57, v57 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v56, v56 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v56, v56, v56 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v59, v198
v_add_f32_dpp v59, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v58, v197
v_add_f32_dpp v58, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v57, v57, v56 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v43, v59, v58 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v58, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v58, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v44, v199, v57, s[78:79]
v_mov_b32_dpp v45, v58 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v45, v58 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v43, v43
v_cvt_f16_f32_e32 v44, v44
v_cvt_f16_f32_e32 v45, v45
v_mov_b32_e32 v197, v60
v_mov_b32_e32 v198, v61
v_mov_b32_e32 v199, v62
v_mov_b32_e32 v200, v63
v_add_f32_dpp v197, v60, v60 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v61, v61 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v62, v62 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v63, v63 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v62, v62, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v63, v63, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v60, v60, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v61, v61, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v61, v62 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v60, v63 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v61, v61 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v61, v61, v61 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v60, v60 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v60, v60, v60 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v63, v198
v_add_f32_dpp v63, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v62, v197
v_add_f32_dpp v62, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v61, v61, v60 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v46, v63, v62 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v62, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v62, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v47, v199, v61, s[78:79]
v_mov_b32_dpp v48, v62 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v48, v62 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v46, v46
v_cvt_f16_f32_e32 v47, v47
v_cvt_f16_f32_e32 v48, v48
v_mov_b32_e32 v197, v64
v_mov_b32_e32 v198, v65
v_mov_b32_e32 v199, v66
v_mov_b32_e32 v200, v67
v_add_f32_dpp v197, v64, v64 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v198, v65, v65 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v199, v66, v66 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v200, v67, v67 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v66, v66, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v67, v67, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v64, v64, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v65, v65, v193 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v65, v66 row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v64, v67 row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v198, v199, v198 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v197, v200, v197 row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v200, v65, v65 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v65, v65, v65 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v199, v64, v64 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v64, v64, v64 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v67, v198
v_add_f32_dpp v67, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v200 row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v199, v200 row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v66, v197
v_add_f32_dpp v66, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v199, v199 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v200, v198, v198 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v65, v65, v64 row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v49, v67, v66 row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v66, v197, v197 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v66, v200 row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v199, v199 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v50, v199, v65, s[78:79]
v_mov_b32_dpp v51, v66 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v51, v66 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_cvt_f16_f32_e32 v49, v49
v_cvt_f16_f32_e32 v50, v50
v_cvt_f16_f32_e32 v51, v51
s_waitcnt vmcnt(0)
s_mov_b64 s[94:95], s[80:81]
s_mov_b32 s93, s83
v_bfe_u32 v197, s18, 21, 1
v_sub_co_u32 v197, vcc, v197, 1
v_cndmask_b32_e32 v198, v160, v154, vcc
v_cndmask_b32_e32 v201, v163, v157, vcc
v_cndmask_b32_e32 v199, v161, v155, vcc
v_cndmask_b32_e32 v202, v164, v158, vcc
v_cndmask_b32_e32 v200, v162, v156, vcc
v_cndmask_b32_e32 v203, v165, v159, vcc
v_cndmask_b32_e32 v204, v172, v166, vcc
v_cndmask_b32_e32 v207, v175, v169, vcc
v_cndmask_b32_e32 v205, v173, v167, vcc
v_cndmask_b32_e32 v208, v176, v170, vcc
v_cndmask_b32_e32 v206, v174, v168, vcc
v_cndmask_b32_e32 v209, v177, v171, vcc
v_readlane_b32 s92, v196, 0
v_add_f16_e64 v4, v4, s92
v_mul_f16_e64 v210, v4, s36
v_cmp_lt_f16_e64 vcc, v4, 0
v_cndmask_b32_e32 v4, v4, v210, vcc
v_add_f16_e64 v7, v7, s92
v_mul_f16_e64 v210, v7, s36
v_cmp_lt_f16_e64 vcc, v7, 0
v_cndmask_b32_e32 v7, v7, v210, vcc
buffer_store_short v4, v198, s[80:83], 0 idxen
buffer_store_short v7, v204, s[80:83], 0 idxen
v_add_f16_e64 v10, v10, s92
v_mul_f16_e64 v210, v10, s36
v_cmp_lt_f16_e64 vcc, v10, 0
v_cndmask_b32_e32 v10, v10, v210, vcc
v_add_f16_e64 v13, v13, s92
v_mul_f16_e64 v210, v13, s36
v_cmp_lt_f16_e64 vcc, v13, 0
v_cndmask_b32_e32 v13, v13, v210, vcc
buffer_store_short v10, v201, s[80:83], 0 idxen
buffer_store_short v13, v207, s[80:83], 0 idxen
v_add_f16_e64 v5, v5, s92
v_mul_f16_e64 v210, v5, s36
v_cmp_lt_f16_e64 vcc, v5, 0
v_cndmask_b32_e32 v5, v5, v210, vcc
v_add_f16_e64 v8, v8, s92
v_mul_f16_e64 v210, v8, s36
v_cmp_lt_f16_e64 vcc, v8, 0
v_cndmask_b32_e32 v8, v8, v210, vcc
buffer_store_short v5, v199, s[80:83], 0 idxen
buffer_store_short v8, v205, s[80:83], 0 idxen
v_add_f16_e64 v11, v11, s92
v_mul_f16_e64 v210, v11, s36
v_cmp_lt_f16_e64 vcc, v11, 0
v_cndmask_b32_e32 v11, v11, v210, vcc
v_add_f16_e64 v14, v14, s92
v_mul_f16_e64 v210, v14, s36
v_cmp_lt_f16_e64 vcc, v14, 0
v_cndmask_b32_e32 v14, v14, v210, vcc
buffer_store_short v11, v202, s[80:83], 0 idxen
buffer_store_short v14, v208, s[80:83], 0 idxen
v_add_f16_e64 v6, v6, s92
v_mul_f16_e64 v210, v6, s36
v_cmp_lt_f16_e64 vcc, v6, 0
v_cndmask_b32_e32 v6, v6, v210, vcc
v_add_f16_e64 v9, v9, s92
v_mul_f16_e64 v210, v9, s36
v_cmp_lt_f16_e64 vcc, v9, 0
v_cndmask_b32_e32 v9, v9, v210, vcc
buffer_store_short v6, v200, s[80:83], 0 idxen
buffer_store_short v9, v206, s[80:83], 0 idxen
v_add_f16_e64 v12, v12, s92
v_mul_f16_e64 v210, v12, s36
v_cmp_lt_f16_e64 vcc, v12, 0
v_cndmask_b32_e32 v12, v12, v210, vcc
v_add_f16_e64 v15, v15, s92
v_mul_f16_e64 v210, v15, s36
v_cmp_lt_f16_e64 vcc, v15, 0
v_cndmask_b32_e32 v15, v15, v210, vcc
buffer_store_short v12, v203, s[80:83], 0 idxen
buffer_store_short v15, v209, s[80:83], 0 idxen
s_lshl_b32 s92, s46, 1
s_add_u32 s80, s80, s92
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s92, v196, 1
v_add_f16_e64 v16, v16, s92
v_mul_f16_e64 v210, v16, s36
v_cmp_lt_f16_e64 vcc, v16, 0
v_cndmask_b32_e32 v16, v16, v210, vcc
v_add_f16_e64 v19, v19, s92
v_mul_f16_e64 v210, v19, s36
v_cmp_lt_f16_e64 vcc, v19, 0
v_cndmask_b32_e32 v19, v19, v210, vcc
buffer_store_short v16, v198, s[80:83], 0 idxen
buffer_store_short v19, v204, s[80:83], 0 idxen
v_add_f16_e64 v22, v22, s92
v_mul_f16_e64 v210, v22, s36
v_cmp_lt_f16_e64 vcc, v22, 0
v_cndmask_b32_e32 v22, v22, v210, vcc
v_add_f16_e64 v25, v25, s92
v_mul_f16_e64 v210, v25, s36
v_cmp_lt_f16_e64 vcc, v25, 0
v_cndmask_b32_e32 v25, v25, v210, vcc
buffer_store_short v22, v201, s[80:83], 0 idxen
buffer_store_short v25, v207, s[80:83], 0 idxen
v_add_f16_e64 v17, v17, s92
v_mul_f16_e64 v210, v17, s36
v_cmp_lt_f16_e64 vcc, v17, 0
v_cndmask_b32_e32 v17, v17, v210, vcc
v_add_f16_e64 v20, v20, s92
v_mul_f16_e64 v210, v20, s36
v_cmp_lt_f16_e64 vcc, v20, 0
v_cndmask_b32_e32 v20, v20, v210, vcc
buffer_store_short v17, v199, s[80:83], 0 idxen
buffer_store_short v20, v205, s[80:83], 0 idxen
v_add_f16_e64 v23, v23, s92
v_mul_f16_e64 v210, v23, s36
v_cmp_lt_f16_e64 vcc, v23, 0
v_cndmask_b32_e32 v23, v23, v210, vcc
v_add_f16_e64 v26, v26, s92
v_mul_f16_e64 v210, v26, s36
v_cmp_lt_f16_e64 vcc, v26, 0
v_cndmask_b32_e32 v26, v26, v210, vcc
buffer_store_short v23, v202, s[80:83], 0 idxen
buffer_store_short v26, v208, s[80:83], 0 idxen
v_add_f16_e64 v18, v18, s92
v_mul_f16_e64 v210, v18, s36
v_cmp_lt_f16_e64 vcc, v18, 0
v_cndmask_b32_e32 v18, v18, v210, vcc
v_add_f16_e64 v21, v21, s92
v_mul_f16_e64 v210, v21, s36
v_cmp_lt_f16_e64 vcc, v21, 0
v_cndmask_b32_e32 v21, v21, v210, vcc
buffer_store_short v18, v200, s[80:83], 0 idxen
buffer_store_short v21, v206, s[80:83], 0 idxen
v_add_f16_e64 v24, v24, s92
v_mul_f16_e64 v210, v24, s36
v_cmp_lt_f16_e64 vcc, v24, 0
v_cndmask_b32_e32 v24, v24, v210, vcc
v_add_f16_e64 v27, v27, s92
v_mul_f16_e64 v210, v27, s36
v_cmp_lt_f16_e64 vcc, v27, 0
v_cndmask_b32_e32 v27, v27, v210, vcc
buffer_store_short v24, v203, s[80:83], 0 idxen
buffer_store_short v27, v209, s[80:83], 0 idxen
s_lshl_b32 s92, s46, 1
s_add_u32 s80, s80, s92
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
s_lshl_b32 s92, s92, 1
s_add_u32 s80, s80, s92
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 2
s_cselect_b32 s83, 0, s83
v_readlane_b32 s92, v196, 4
v_add_f16_e64 v28, v28, s92
v_mul_f16_e64 v210, v28, s36
v_cmp_lt_f16_e64 vcc, v28, 0
v_cndmask_b32_e32 v28, v28, v210, vcc
v_add_f16_e64 v31, v31, s92
v_mul_f16_e64 v210, v31, s36
v_cmp_lt_f16_e64 vcc, v31, 0
v_cndmask_b32_e32 v31, v31, v210, vcc
buffer_store_short v28, v198, s[80:83], 0 idxen
buffer_store_short v31, v204, s[80:83], 0 idxen
v_add_f16_e64 v34, v34, s92
v_mul_f16_e64 v210, v34, s36
v_cmp_lt_f16_e64 vcc, v34, 0
v_cndmask_b32_e32 v34, v34, v210, vcc
v_add_f16_e64 v37, v37, s92
v_mul_f16_e64 v210, v37, s36
v_cmp_lt_f16_e64 vcc, v37, 0
v_cndmask_b32_e32 v37, v37, v210, vcc
buffer_store_short v34, v201, s[80:83], 0 idxen
buffer_store_short v37, v207, s[80:83], 0 idxen
v_add_f16_e64 v29, v29, s92
v_mul_f16_e64 v210, v29, s36
v_cmp_lt_f16_e64 vcc, v29, 0
v_cndmask_b32_e32 v29, v29, v210, vcc
v_add_f16_e64 v32, v32, s92
v_mul_f16_e64 v210, v32, s36
v_cmp_lt_f16_e64 vcc, v32, 0
v_cndmask_b32_e32 v32, v32, v210, vcc
buffer_store_short v29, v199, s[80:83], 0 idxen
buffer_store_short v32, v205, s[80:83], 0 idxen
v_add_f16_e64 v35, v35, s92
v_mul_f16_e64 v210, v35, s36
v_cmp_lt_f16_e64 vcc, v35, 0
v_cndmask_b32_e32 v35, v35, v210, vcc
v_add_f16_e64 v38, v38, s92
v_mul_f16_e64 v210, v38, s36
v_cmp_lt_f16_e64 vcc, v38, 0
v_cndmask_b32_e32 v38, v38, v210, vcc
buffer_store_short v35, v202, s[80:83], 0 idxen
buffer_store_short v38, v208, s[80:83], 0 idxen
v_add_f16_e64 v30, v30, s92
v_mul_f16_e64 v210, v30, s36
v_cmp_lt_f16_e64 vcc, v30, 0
v_cndmask_b32_e32 v30, v30, v210, vcc
v_add_f16_e64 v33, v33, s92
v_mul_f16_e64 v210, v33, s36
v_cmp_lt_f16_e64 vcc, v33, 0
v_cndmask_b32_e32 v33, v33, v210, vcc
buffer_store_short v30, v200, s[80:83], 0 idxen
buffer_store_short v33, v206, s[80:83], 0 idxen
v_add_f16_e64 v36, v36, s92
v_mul_f16_e64 v210, v36, s36
v_cmp_lt_f16_e64 vcc, v36, 0
v_cndmask_b32_e32 v36, v36, v210, vcc
v_add_f16_e64 v39, v39, s92
v_mul_f16_e64 v210, v39, s36
v_cmp_lt_f16_e64 vcc, v39, 0
v_cndmask_b32_e32 v39, v39, v210, vcc
buffer_store_short v36, v203, s[80:83], 0 idxen
buffer_store_short v39, v209, s[80:83], 0 idxen
s_lshl_b32 s92, s46, 1
s_add_u32 s80, s80, s92
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
v_readlane_b32 s92, v196, 5
v_add_f16_e64 v40, v40, s92
v_mul_f16_e64 v210, v40, s36
v_cmp_lt_f16_e64 vcc, v40, 0
v_cndmask_b32_e32 v40, v40, v210, vcc
v_add_f16_e64 v43, v43, s92
v_mul_f16_e64 v210, v43, s36
v_cmp_lt_f16_e64 vcc, v43, 0
v_cndmask_b32_e32 v43, v43, v210, vcc
buffer_store_short v40, v198, s[80:83], 0 idxen
buffer_store_short v43, v204, s[80:83], 0 idxen
v_add_f16_e64 v46, v46, s92
v_mul_f16_e64 v210, v46, s36
v_cmp_lt_f16_e64 vcc, v46, 0
v_cndmask_b32_e32 v46, v46, v210, vcc
v_add_f16_e64 v49, v49, s92
v_mul_f16_e64 v210, v49, s36
v_cmp_lt_f16_e64 vcc, v49, 0
v_cndmask_b32_e32 v49, v49, v210, vcc
buffer_store_short v46, v201, s[80:83], 0 idxen
buffer_store_short v49, v207, s[80:83], 0 idxen
v_add_f16_e64 v41, v41, s92
v_mul_f16_e64 v210, v41, s36
v_cmp_lt_f16_e64 vcc, v41, 0
v_cndmask_b32_e32 v41, v41, v210, vcc
v_add_f16_e64 v44, v44, s92
v_mul_f16_e64 v210, v44, s36
v_cmp_lt_f16_e64 vcc, v44, 0
v_cndmask_b32_e32 v44, v44, v210, vcc
buffer_store_short v41, v199, s[80:83], 0 idxen
buffer_store_short v44, v205, s[80:83], 0 idxen
v_add_f16_e64 v47, v47, s92
v_mul_f16_e64 v210, v47, s36
v_cmp_lt_f16_e64 vcc, v47, 0
v_cndmask_b32_e32 v47, v47, v210, vcc
v_add_f16_e64 v50, v50, s92
v_mul_f16_e64 v210, v50, s36
v_cmp_lt_f16_e64 vcc, v50, 0
v_cndmask_b32_e32 v50, v50, v210, vcc
buffer_store_short v47, v202, s[80:83], 0 idxen
buffer_store_short v50, v208, s[80:83], 0 idxen
v_add_f16_e64 v42, v42, s92
v_mul_f16_e64 v210, v42, s36
v_cmp_lt_f16_e64 vcc, v42, 0
v_cndmask_b32_e32 v42, v42, v210, vcc
v_add_f16_e64 v45, v45, s92
v_mul_f16_e64 v210, v45, s36
v_cmp_lt_f16_e64 vcc, v45, 0
v_cndmask_b32_e32 v45, v45, v210, vcc
buffer_store_short v42, v200, s[80:83], 0 idxen
buffer_store_short v45, v206, s[80:83], 0 idxen
v_add_f16_e64 v48, v48, s92
v_mul_f16_e64 v210, v48, s36
v_cmp_lt_f16_e64 vcc, v48, 0
v_cndmask_b32_e32 v48, v48, v210, vcc
v_add_f16_e64 v51, v51, s92
v_mul_f16_e64 v210, v51, s36
v_cmp_lt_f16_e64 vcc, v51, 0
v_cndmask_b32_e32 v51, v51, v210, vcc
buffer_store_short v48, v203, s[80:83], 0 idxen
buffer_store_short v51, v209, s[80:83], 0 idxen
s_lshl_b32 s92, s46, 1
s_add_u32 s80, s80, s92
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 1
s_cselect_b32 s83, 0, s83
s_lshl_b32 s92, s46, 2
s_add_u32 s80, s80, s92
s_addc_u32 s81, s81, 0
s_lshl_b32 s92, s92, 2
s_add_u32 s80, s80, s92
s_addc_u32 s81, s81, 0
s_sub_u32 s72, s72, 10
s_cselect_b32 s83, 0, s83
s_bitcmp1_b32 s18, 21
s_cselect_b32 s83, s83, s93
s_cselect_b32 s80, s80, s94
s_cselect_b32 s81, s81, s95
s_cselect_b32 s93, 0, 16
s_cselect_b32 s94, 16, 0
s_lshl_b32 s95, s94, 1
s_add_u32 s72, s72, s93
s_add_u32 s84, s84, s95
s_addc_u32 s85, s85, 0
s_sub_u32 s86, s86, s94
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
s_bitcmp1_b32 s13, 0
s_addc_u32 s88, s13, 0
s_bitcmp0_b32 s18, 21
s_addc_u32 s73, s9, 0
s_lshr_b32 s73, s73, 1
s_mul_i32 s73, s73, s10
s_lshr_b32 s73, s73, 1
s_mul_i32 s73, s73, s88
s_cmp_eq_u32 s73, 0
s_cbranch_scc1 63832
s_add_u32 s88, s72, s71
s_cmp_lt_i32 s88, 0
s_cbranch_scc0 461
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
ds_read_b32 v197, v155 offset:256
ds_read_b32 v154, v154
s_add_u32 s76, s76, 0x18c
s_cmp_eq_u32 s76, 0x10000
s_cselect_b32 s76, 0xc220, s76
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s74, v154
v_readlane_b32 s90, v197, 0
s_bitcmp1_b32 s90, 18
s_cbranch_scc1 435
v_readlane_b32 s88, v197, 1
v_readlane_b32 s89, v197, 2
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
s_lshl_b64 s[80:81], s[80:81], 1
s_add_u32 s80, s80, s24
s_addc_u32 s81, s81, s25
s_mul_i32 s78, s46, s72
s_lshl_b32 s78, s78, 1
s_add_u32 s80, s80, s78
s_addc_u32 s81, s81, 0
s_add_u32 s81, s81, 0x20000
s_mov_b32 s83, 0x11014000
s_bitcmp1_b32 s18, 7
s_cselect_b32 s87, 0x11014000, 0
s_lshl_b32 s77, s72, 1
s_add_u32 s84, s34, s77
s_addc_u32 s85, s35, 0
s_add_u32 s85, s85, 0x20000
s_sub_u32 s86, s88, s72
s_cselect_b32 s87, 0, s87
s_sub_u32 s72, s88, s89
s_sub_u32 s72, s72, 1
s_sub_u32 s72, s72, s71
s_cselect_b32 s83, 0, s83
v_bfe_u32 v197, v154, 16, 16
v_bfe_u32 v198, v154, 0, 16
v_and_b32_e64 v199, v1, 7
v_sub_co_u32 v200, vcc, 7, v199
v_min_u32_e32 v199, v199, v200
v_bfe_u32 v200, v199, 1, 1
v_bfe_u32 v199, v199, 0, 1
v_mov_b32_dpp v197, v197 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v198, v198 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
v_add_co_u32 v197, vcc, v197, v200
v_add_co_u32 v198, vcc, v198, v199
v_mov_b32_dpp v199, v154 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[78:79], v199, s12
v_sub_co_u32 v199, vcc, v199, s74
v_mul_lo_u32 v199, v199, s45
v_xor_b32_dpp v200, v1, v1 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v200, v1, v1 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v201, v1, v1 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v201, v1, v1 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_add_co_u32 v201, vcc, v198, v201
v_add_co_u32 v200, vcc, v197, v200
v_lshlrev_b32_e32 v201, 1, v201
s_and_b32 s92, 1, s31
v_add_co_u32 v201, vcc, s92, v201
v_lshlrev_b32_e32 v200, 1, v200
s_and_b32 s92, 1, s30
v_subrev_co_u32 v200, vcc, s92, v200
v_mad_i32_i24 v166, v200, s33, v201
v_add_co_u32 v166, vcc, v166, v199
v_subrev_co_u32 v169, vcc, 1, v166
v_add_co_u32 v172, vcc, s33, v166
v_add_co_u32 v175, vcc, s33, v169
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[94:95], s[96:97], s[78:79]
v_subrev_co_u32 v201, vcc, 1, v201
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[96:97], s[96:97], s[78:79]
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v166, v166, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v169, v169, -1, s[98:99]
v_add_co_u32 v200, vcc, 1, v200
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v172, v172, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v175, v175, -1, s[98:99]
v_xor_b32_dpp v200, v1, v1 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v200, v1, v1 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v201, v1, v1 quad_perm:[1,1,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32 v201, vcc, v198, v201
v_add_co_u32 v200, vcc, v197, v200
v_lshlrev_b32_e32 v201, 1, v201
s_and_b32 s92, 1, s31
v_add_co_u32 v201, vcc, s92, v201
v_lshlrev_b32_e32 v200, 1, v200
s_and_b32 s92, 1, s30
v_subrev_co_u32 v200, vcc, s92, v200
v_mad_i32_i24 v167, v200, s33, v201
v_add_co_u32 v167, vcc, v167, v199
v_subrev_co_u32 v170, vcc, 1, v167
v_add_co_u32 v173, vcc, s33, v167
v_add_co_u32 v176, vcc, s33, v170
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[94:95], s[96:97], s[78:79]
v_subrev_co_u32 v201, vcc, 1, v201
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[96:97], s[96:97], s[78:79]
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v167, v167, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v170, v170, -1, s[98:99]
v_add_co_u32 v200, vcc, 1, v200
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v173, v173, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v176, v176, -1, s[98:99]
v_xor_b32_dpp v200, v1, v1 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v200, v1, v1 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v201, v1, v1 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v201, v1, v1 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xa
v_add_co_u32 v201, vcc, v198, v201
v_add_co_u32 v200, vcc, v197, v200
v_lshlrev_b32_e32 v201, 1, v201
s_and_b32 s92, 1, s31
v_add_co_u32 v201, vcc, s92, v201
v_lshlrev_b32_e32 v200, 1, v200
s_and_b32 s92, 1, s30
v_subrev_co_u32 v200, vcc, s92, v200
v_mad_i32_i24 v168, v200, s33, v201
v_add_co_u32 v168, vcc, v168, v199
v_subrev_co_u32 v171, vcc, 1, v168
v_add_co_u32 v174, vcc, s33, v168
v_add_co_u32 v177, vcc, s33, v171
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[94:95], s[96:97], s[78:79]
v_subrev_co_u32 v201, vcc, 1, v201
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[96:97], s[96:97], s[78:79]
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v168, v168, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v171, v171, -1, s[98:99]
v_add_co_u32 v200, vcc, 1, v200
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v174, v174, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v177, v177, -1, s[98:99]
v_bfe_u32 v197, v154, 16, 16
v_bfe_u32 v198, v154, 0, 16
v_and_b32_e64 v199, v1, 7
v_sub_co_u32 v200, vcc, 7, v199
v_min_u32_e32 v199, v199, v200
v_bfe_u32 v200, v199, 1, 1
v_bfe_u32 v199, v199, 0, 1
v_mov_b32_dpp v197, v197 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v198, v198 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_add_co_u32 v197, vcc, v197, v200
v_add_co_u32 v198, vcc, v198, v199
v_mov_b32_dpp v199, v154 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[78:79], v199, s12
v_sub_co_u32 v199, vcc, v199, s74
v_mul_lo_u32 v199, v199, s45
v_xor_b32_dpp v200, v1, v1 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v200, v1, v1 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v201, v1, v1 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v201, v1, v1 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_add_co_u32 v201, vcc, v198, v201
v_add_co_u32 v200, vcc, v197, v200
v_lshlrev_b32_e32 v201, 1, v201
s_and_b32 s92, 1, s31
v_add_co_u32 v201, vcc, s92, v201
v_lshlrev_b32_e32 v200, 1, v200
s_and_b32 s92, 1, s30
v_subrev_co_u32 v200, vcc, s92, v200
v_mad_i32_i24 v154, v200, s33, v201
v_add_co_u32 v154, vcc, v154, v199
v_subrev_co_u32 v157, vcc, 1, v154
v_add_co_u32 v160, vcc, s33, v154
v_add_co_u32 v163, vcc, s33, v157
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[94:95], s[96:97], s[78:79]
v_subrev_co_u32 v201, vcc, 1, v201
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[96:97], s[96:97], s[78:79]
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v154, v154, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v157, v157, -1, s[98:99]
v_add_co_u32 v200, vcc, 1, v200
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v160, v160, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v163, v163, -1, s[98:99]
v_xor_b32_dpp v200, v1, v1 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v200, v1, v1 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v201, v1, v1 quad_perm:[1,1,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32 v201, vcc, v198, v201
v_add_co_u32 v200, vcc, v197, v200
v_lshlrev_b32_e32 v201, 1, v201
s_and_b32 s92, 1, s31
v_add_co_u32 v201, vcc, s92, v201
v_lshlrev_b32_e32 v200, 1, v200
s_and_b32 s92, 1, s30
v_subrev_co_u32 v200, vcc, s92, v200
v_mad_i32_i24 v155, v200, s33, v201
v_add_co_u32 v155, vcc, v155, v199
v_subrev_co_u32 v158, vcc, 1, v155
v_add_co_u32 v161, vcc, s33, v155
v_add_co_u32 v164, vcc, s33, v158
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[94:95], s[96:97], s[78:79]
v_subrev_co_u32 v201, vcc, 1, v201
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[96:97], s[96:97], s[78:79]
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v155, v155, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v158, v158, -1, s[98:99]
v_add_co_u32 v200, vcc, 1, v200
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v161, v161, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v164, v164, -1, s[98:99]
v_xor_b32_dpp v200, v1, v1 quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v200, v1, v1 quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v201, v1, v1 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v201, v1, v1 quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xa
v_add_co_u32 v201, vcc, v198, v201
v_add_co_u32 v200, vcc, v197, v200
v_lshlrev_b32_e32 v201, 1, v201
s_and_b32 s92, 1, s31
v_add_co_u32 v201, vcc, s92, v201
v_lshlrev_b32_e32 v200, 1, v200
s_and_b32 s92, 1, s30
v_subrev_co_u32 v200, vcc, s92, v200
v_mad_i32_i24 v156, v200, s33, v201
v_add_co_u32 v156, vcc, v156, v199
v_subrev_co_u32 v159, vcc, 1, v156
v_add_co_u32 v162, vcc, s33, v156
v_add_co_u32 v165, vcc, s33, v159
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[94:95], s[96:97], s[78:79]
v_subrev_co_u32 v201, vcc, 1, v201
v_cmp_ge_u32_e64 s[96:97], v201, s33
s_or_b64 s[96:97], s[96:97], s[78:79]
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v156, v156, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v159, v159, -1, s[98:99]
v_add_co_u32 v200, vcc, 1, v200
v_cmp_ge_u32_e64 s[92:93], v200, s32
s_or_b64 s[98:99], s[94:95], s[92:93]
v_cndmask_b32_e64 v162, v162, -1, s[98:99]
s_or_b64 s[98:99], s[96:97], s[92:93]
v_cndmask_b32_e64 v165, v165, -1, s[98:99]
v_and_b32_e64 v196, v1, 63
buffer_load_ushort v196, v196, s[84:87], 0 idxen
s_mov_b64 vcc, s[2:3]
s_branch 62602
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
s_code_end
s_code_end
s_code_end
s_code_end
