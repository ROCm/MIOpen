/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

.macro _v_add_f16_e32_1_gfx1x vdst:req, vsrc:req
    .long  0x640000FF + (\vdst << 17) + (\vsrc << 9)
    .long  0x00003C00
.endm

.macro _v_pk_add_f16_gfx1x vdst:req, vsrc0:req, vsrc1:req
    .long  0xCC0F0000 + (\vdst << 0)
    .long  0x18020100 + (\vsrc0 << 0) + (\vsrc1 << 9)
.endm

.macro _v_pk_add_f16_0_gfx1x vdst:req, vsrc0:req
    .long  0xCC0F0000 + (\vdst << 0)
    .long  0x00010100 + (\vsrc0 << 0)
.endm

.macro _v_pk_mul_f16_05_gfx1x vdst:req, vsrc0:req
    .long  0xCC100000 + (\vdst << 0)
    .long  0x0801E100 + (\vsrc0 << 0)
.endm

s_version 0x2004
s_inst_prefetch 0x3
v_mov_b32_e32 v1, v0
s_mov_b32 s0, 0
v_mov_b32_e32 v180, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s79, 0xc220
s_mov_b32 s78, 0xc220
v_and_b32_e32 v181, 0xc0, v0
v_add_co_u32 v1, vcc, v0, v181
v_readfirstlane_b32 s84, v1
s_lshr_b32 s84, s84, 5
s_add_u32 s84, s84, 8
s_and_b32 s74, s84, 20
s_mov_b64 s[84:85], s[2:3]
s_load_dwordx16 s[8:23], s[84:85], null
s_load_dwordx4 s[24:27], s[84:85], 0x40
s_load_dwordx2 s[28:29], s[84:85], 0x50
s_waitcnt lgkmcnt(0)
s_and_b32 s14, s14, 0xffff
s_bitcmp1_b32 s14, 6
s_cbranch_scc0 12
s_and_b32 s17, s17, 0xffff
s_and_b32 s19, s19, 0xffff
s_and_b32 s21, s21, 0xffff
s_load_dwordx2 s[16:17], s[16:17], null
s_load_dwordx2 s[18:19], s[18:19], null
s_load_dwordx2 s[20:21], s[20:21], null
s_bitcmp1_b32 s14, 7
s_cbranch_scc0 2
s_load_dwordx2 s[30:31], s[84:85], 0x58
s_mov_b32 s32, 1.0
s_mov_b32 s33, 1.0
s_getpc_b64 s[82:83]
s_add_u32 s82, s82, 0x6844
s_addc_u32 s83, s83, 0
s_bitcmp1_b32 s14, 14
s_cbranch_scc1 3
s_bitcmp1_b32 s14, 8
s_cbranch_scc0 12
s_branch 16
s_load_dword s34, s[84:85], 0xc8
s_waitcnt lgkmcnt(0)
s_and_b32 s34, s34, 0xff
s_cmp_eq_u32 s34, 2
s_cbranch_scc1 16
s_cmp_eq_u32 s34, 3
s_cbranch_scc1 19
s_cmp_eq_u32 s34, 1
s_cbranch_scc1 5
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x6cac
s_addc_u32 s81, s81, 0
s_branch 18
s_load_dword s32, s[84:85], 0x60
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x6e90
s_addc_u32 s81, s81, 0
s_branch 11
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x71dc
s_addc_u32 s81, s81, 0
s_branch 6
s_load_dwordx2 s[32:33], s[84:85], 0x60
s_getpc_b64 s[80:81]
s_add_u32 s80, s80, 0x7540
s_addc_u32 s81, s81, 0
s_bitcmp1_b32 s14, 7
s_cbranch_scc0 7
s_bitcmp1_b32 s14, 6
s_cbranch_scc0 5
s_waitcnt lgkmcnt(0)
s_and_b32 s31, s31, 0xffff
s_load_dwordx2 s[30:31], s[30:31], null
s_bitcmp1_b32 s14, 9
s_cbranch_scc0 83
s_mov_b32 s86, 0x8c
s_mov_b32 s87, 0x9c
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cmp_eq_u64 0, vcc
s_cselect_b32 s86, s87, s86
s_load_dword s44, s[84:85], 0x88
s_load_dword s49, s[84:85], 0x90
s_load_dword s72, s[84:85], 0x98
s_load_dword s48, s[84:85], s86
s_load_dword s73, s[84:85], 0xa0
s_load_dword s45, s[84:85], 0xa8
s_load_dword s46, s[84:85], 0xac
s_load_dword s50, s[84:85], 0xb0
s_bitcmp1_b32 s14, 10
s_cbranch_scc0 79
s_load_dwordx4 s[92:95], s[84:85], 0xb8
v_ffbh_u32_e32 v184, s13
v_lshlrev_b32_e64 v185, v184, s13
v_and_b32_e32 v183, 0xffffff00, v185
v_cmp_eq_u32_e32 vcc, 0x80000000, v185
v_cvt_f32_u32_e32 v183, v183
v_rcp_f32_e32 v181, v183
v_sub_co_ci_u32_e32 v182, vcc, 32, v184, vcc
v_cvt_f32_ubyte0_e32 v184, v185
v_fma_f32 v183, v183, v181, -1.0
v_fma_f32 v183, v184, v181, v183
v_fmaak_f32 v183, v183, v181, 0x9f000000
v_mul_f32_e32 v183, 0x5f800000, v183
v_mov_b32_e32 v184, 0
v_cvt_flr_i32_f32_e64 v183, -v183
v_lshl_add_u32 v181, v181, 9, v183
v_mad_u64_u32 v[184:185], vcc, v185, v181, v[184:185]
v_sub_co_ci_u32_e64 v181, vcc, v181, -1, vcc
v_mul_hi_u32 v183, s4, v181
v_add_co_u32 v181, vcc, v183, s4
v_add_co_ci_u32_e64 v183, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v182
v_cndmask_b32_e32 v181, v181, v183, vcc
v_alignbit_b32 v181, v183, v181, v182
s_nop 0
v_readfirstlane_b32 s88, v181
s_mul_i32 s89, s88, s13
s_sub_u32 s4, s4, s89
s_waitcnt lgkmcnt(0)
s_lshl_b32 s93, s93, 1
s_lshl_b64 s[94:95], s[94:95], 1
s_mul_i32 s89, s93, s88
s_add_u32 s16, s16, s89
s_addc_u32 s17, s17, 0
s_mul_i32 s89, s94, s88
s_add_u32 s18, s18, s89
s_addc_u32 s19, s19, 0
s_mul_i32 s89, s95, s88
s_add_u32 s20, s20, s89
s_addc_u32 s21, s21, 0
s_branch 22
s_mov_b32 s49, s11
s_mul_i32 s48, s10, s11
s_mul_i32 s44, s48, s9
s_mov_b32 s50, s29
s_mul_i32 s46, s28, s29
s_mul_i32 s45, s46, s12
s_bitcmp1_b32 s14, 13
s_cbranch_scc0 2
s_load_dwordx8 s[92:99], s[84:85], 0x68
s_mov_b32 s73, s25
s_mul_i32 s86, s24, s25
s_bitcmp1_b32 s14, 2
s_cselect_b32 s87, s12, s9
s_mul_i32 s87, s86, s87
s_bitcmp1_b32 s14, 2
s_cselect_b32 s100, s87, s86
s_cselect_b32 s72, s86, s87
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cmp_eq_u64 0, vcc
s_cselect_b32 s48, s100, s48
s_waitcnt lgkmcnt(0)
s_lshl_b32 s47, s48, 1
s_and_b32 s17, s17, 0xffff
s_and_b32 s19, s19, 0xffff
s_and_b32 s21, s21, 0xffff
s_and_b32 s31, s31, 0xffff
s_bitcmp1_b32 s14, 13
s_cbranch_scc0 8
s_add_u32 s16, s16, s92
s_addc_u32 s17, s17, s93
s_add_u32 s18, s18, s94
s_addc_u32 s19, s19, s95
s_add_u32 s20, s20, s96
s_addc_u32 s21, s21, s97
s_add_u32 s30, s30, s98
s_addc_u32 s31, s31, s99
v_cvt_f16_f32_e32 v181, s32
v_cvt_f16_f32_e32 v182, s33
v_readfirstlane_b32 s32, v181
v_readfirstlane_b32 s33, v182
s_and_b32 s88, 1, s26
s_addc_u32 s88, s28, 1
s_ashr_i32 s88, s88, 1
s_add_u32 s86, s88, 1
v_mov_b32_e32 v182, 0x80000000
v_mul_hi_u32 v182, v182, s86
s_nop 0
v_readfirstlane_b32 s86, v182
s_andn2_b32 s88, 1, s27
s_addc_u32 s88, s29, 1
s_ashr_i32 s88, s88, 1
s_add_u32 s87, s88, 1
v_mov_b32_e32 v182, 0x80000000
v_mul_hi_u32 v182, v182, s87
s_nop 0
v_readfirstlane_b32 s87, v182
s_sub_u32 s57, 0, s87
s_sub_u32 s56, 0, s86
s_add_u32 s1, s24, 2
v_mov_b32_e32 v182, 0x55555556
v_mul_hi_u32 v182, v182, s1
s_nop 0
v_readfirstlane_b32 s1, v182
s_add_u32 s5, s25, 2
v_mov_b32_e32 v182, 0x55555556
v_mul_hi_u32 v182, v182, s5
s_nop 0
v_readfirstlane_b32 s5, v182
v_mad_i32_i24 v181, 3, s1, -2
v_sub_co_u32 v181, vcc, v181, s24
v_add_co_ci_u32_e64 v181, vcc, 0, 0, vcc
s_nop 0
v_readfirstlane_b32 s88, v181
s_and_b32 s88, s88, 1
s_and_b32 s88, s88, s1
s_add_u32 s1, s1, s88
v_readfirstlane_b32 s89, v1
s_and_b32 s90, s89, 64
s_cselect_b32 s90, 0x80000, 0
s_or_b32 s14, s14, s90
s_lshl_b32 s51, s47, 1
s_sub_u32 s52, 0, s51
s_subb_u32 s53, 0, 0
s_bitset1_b32 s14, 23
s_mov_b32 s51, s47
s_mov_b32 s52, s47
s_mov_b32 s53, 0
s_add_u32 s5, s5, 1
s_and_b32 s5, s5, -2
s_branch 16
s_and_b32 s90, s9, 3
s_cselect_b32 s90, 0, 0x1000000
s_bitcmp1_b32 s14, 2
s_cselect_b32 s90, 0, s90
s_or_b32 s14, s14, s90
s_cmp_eq_u32 s90, 0
s_cselect_b32 s51, s47, s51
s_cselect_b32 s52, s47, s52
s_cselect_b32 s53, 0, s53
s_bitcmp0_b32 s89, 8
s_cselect_b32 s90, s90, 0
s_cmp_eq_u32 s90, 0
s_cselect_b32 s90, 0, 0x80000
s_andn2_b32 s14, s14, s90
s_add_u32 s52, s52, s51
s_addc_u32 s53, s53, 0
s_add_u32 s51, s51, s51
v_bfe_u32 v182, v1, 2, 6
v_lshrrev_b32_e32 v175, 1, v182
s_bitcmp0_b32 s89, 8
s_cselect_b32 s90, 0x1000000, 0
s_or_b32 s90, s90, 0x100000
s_and_b32 s90, s14, s90
s_cselect_b32 s90, 0, 15
v_bfi_b32 v175, s90, v182, v175
v_bfe_u32 v182, s89, 8, 1
v_xor_b32_e64 v182, v182, 1
v_lshrrev_b32_e32 v175, v182, v175
s_mul_i32 s70, s8, s86
s_sub_u32 s70, s70, 1
s_lshr_b32 s70, s70, 0
s_add_u32 s70, s70, 1
s_lshr_b32 s89, -1, 16
s_and_b32 s89, s89, s70
s_lshr_b32 s90, s70, 16
s_mul_i32 s90, s90, s87
s_mul_i32 s70, s89, s87
s_lshl_b32 s89, s90, 16
s_lshr_b32 s90, s90, 16
s_add_u32 s70, s89, s70
s_addc_u32 s71, s90, 0
s_sub_u32 s70, s70, 1
s_subb_u32 s71, s71, 0
s_lshr_b64 s[70:71], s[70:71], 5
s_add_u32 s70, s70, 1
s_addc_u32 s71, s71, 0
v_mov_b32_e32 v182, s4
v_mov_b32_e32 v183, s13
v_and_b32_e32 v184, 3, v1
v_cmp_eq_u32_e32 vcc, 2, v184
v_cndmask_b32_e32 v182, v182, v183, vcc
v_cmp_eq_u32_e32 vcc, 1, v184
v_cndmask_b32_e32 v185, 0, v175, vcc
s_bitcmp1_b32 s14, 20
s_cbranch_scc0 4
v_add_co_u32 v183, vcc, v175, 8
v_cmp_eq_u32_e32 vcc, 0, v184
v_cndmask_b32_e32 v185, v185, v183, vcc
v_cmp_eq_u32_e64 s[90:91], 3, v184
v_bfe_u32 v173, v185, 0, 5
v_mad_u32_u24 v173, v182, 32, v173
v_ffbh_u32_e32 v188, s87
v_lshlrev_b32_e64 v189, v188, s87
v_and_b32_e32 v187, 0xffffff00, v189
v_cmp_eq_u32_e32 vcc, 0x80000000, v189
v_cvt_f32_u32_e32 v187, v187
v_rcp_f32_e32 v174, v187
v_sub_co_ci_u32_e32 v186, vcc, 32, v188, vcc
v_cvt_f32_ubyte0_e32 v188, v189
v_fma_f32 v187, v187, v174, -1.0
v_fma_f32 v187, v188, v174, v187
v_fmaak_f32 v187, v187, v174, 0x9f000000
v_mul_f32_e32 v187, 0x5f800000, v187
v_mov_b32_e32 v188, 0
v_cvt_flr_i32_f32_e64 v187, -v187
v_lshl_add_u32 v174, v174, 9, v187
v_mad_u64_u32 v[188:189], vcc, v189, v174, v[188:189]
v_sub_co_ci_u32_e64 v174, vcc, v174, -1, vcc
v_mul_hi_u32 v187, v173, v174
v_add_co_u32 v174, vcc, v187, v173
v_add_co_ci_u32_e64 v187, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v186
v_cndmask_b32_e32 v174, v174, v187, vcc
v_alignbit_b32 v174, v187, v174, v186
v_mad_i32_i24 v172, v174, s57, v173
v_lshrrev_b32_e32 v173, 5, v185
v_mad_u32_u24 v173, v174, 1, v173
v_cndmask_b32_e64 v173, v173, 1, s[90:91]
v_ffbh_u32_e32 v188, s86
v_lshlrev_b32_e64 v189, v188, s86
v_and_b32_e32 v187, 0xffffff00, v189
v_cmp_eq_u32_e32 vcc, 0x80000000, v189
v_cvt_f32_u32_e32 v187, v187
v_rcp_f32_e32 v174, v187
v_sub_co_ci_u32_e32 v186, vcc, 32, v188, vcc
v_cvt_f32_ubyte0_e32 v188, v189
v_fma_f32 v187, v187, v174, -1.0
v_fma_f32 v187, v188, v174, v187
v_fmaak_f32 v187, v187, v174, 0x9f000000
v_mul_f32_e32 v187, 0x5f800000, v187
v_mov_b32_e32 v188, 0
v_cvt_flr_i32_f32_e64 v187, -v187
v_lshl_add_u32 v174, v174, 9, v187
v_mad_u64_u32 v[188:189], vcc, v189, v174, v[188:189]
v_sub_co_ci_u32_e64 v174, vcc, v174, -1, vcc
v_mul_hi_u32 v187, v173, v174
v_add_co_u32 v174, vcc, v187, v173
v_add_co_ci_u32_e64 v187, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v186
v_cndmask_b32_e32 v174, v174, v187, vcc
v_alignbit_b32 v174, v187, v174, v186
v_mad_i32_i24 v173, v174, s56, v173
v_readlane_b32 s58, v172, 2
v_readlane_b32 s59, v173, 2
v_readlane_b32 s60, v174, 2
v_readlane_b32 s61, v173, 3
v_readlane_b32 s62, v174, 3
v_add_co_u32 v172, vcc, v172, s57
v_add_co_u32 v173, vcc, v173, s56
v_mov_b32_dpp v174, v174 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v172, v172 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v173, v173 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x11014000
s_mov_b32 s86, 0x80000000
s_mov_b32 s87, 0x11014000
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 8
v_xor_b32_dpp v176, v1, v1 quad_perm:[1,3,2,2] row_mask:0xf bank_mask:0xf
v_subrev_co_u32 v176, vcc, 1, v176
v_cvt_f16_i16_e32 v176, v176
_v_pk_add_f16_0_gfx1x 176, 176
s_branch 7
v_xor_b32_dpp v176, v1, v1 quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32 v176, vcc, 1, v176
v_cvt_f16_i16_e32 v176, v176
_v_pk_add_f16_0_gfx1x 176, 176
v_mov_b32_e32 v177, 1
v_xor_b32_dpp v177, v1, v1 quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v177, v1, v1 quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32 v177, vcc, 1, v177
v_mov_b32_e32 v178, 1
v_xor_b32_dpp v178, v1, v1 quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v178, v1, v1 quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32 v178, vcc, 1, v178
v_cvt_f32_i32_e32 v177, v177
v_cvt_f32_i32_e32 v178, v178
v_lshrrev_b32_e64 v181, 2, s74
v_and_b32_e32 v182, 3, v1
v_bfe_u32 v183, v1, 4, 3
v_mad_u32_u24 v171, v183, 4, v182
v_lshlrev_b32_e32 v171, 4, v171
v_mad_u32_u24 v162, v181, 4, v182
v_lshlrev_b32_e32 v162, 4, v162
v_bfe_u32 v181, v1, 2, 2
v_and_b32_e32 v182, 1, v181
v_mad_u32_u24 v184, v181, 16, v182
v_lshlrev_b32_e32 v184, 6, v184
v_xor_b32_e32 v162, v162, v184
v_mul_u32_u24_e32 v184, 0x400, v181
v_xor_b32_e32 v171, v171, v184
s_lshr_b32 s74, s74, 1
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 64
s_and_b32 s3, s14, 0x1100000
s_addc_u32 s3, 0, 0
v_lshrrev_b32_e32 v184, 1, v1
s_mul_i32 s2, 60, s3
s_sub_u32 s2, 63, s2
v_bfi_b32 v184, s2, v1, v184
v_and_b32_e32 v181, 1, v184
v_bfe_u32 v182, v184, 1, 1
v_xor_b32_e32 v181, v181, v182
v_bfe_u32 v183, v184, 3, 1
v_mad_u32_u24 v182, v182, 2, v183
v_mul_u32_u24_e32 v181, 0x118, v181
v_bfe_u32 v183, v184, 2, 1
v_mad_u32_u24 v182, v182, 2, v181
v_xor_b32_e32 v182, v182, v183
v_and_b32_e32 v183, 0xf0, v184
v_xor_b32_e32 v182, v182, v183
s_mul_i32 s2, 4, s3
s_sub_u32 s2, 6, s2
v_bfe_u32 v184, v1, s2, 1
v_mul_u32_u24_e32 v184, 0x1040, v184
v_xor_b32_e32 v164, 0x314, v182
v_xor_b32_e32 v165, 0x31c, v182
v_xor_b32_e32 v166, 8, v182
s_bitcmp1_b32 s14, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v163, v182, v166, vcc
v_cndmask_b32_e32 v166, v166, v182, vcc
v_mad_u32_u24 v163, 4, v163, v184
v_mad_u32_u24 v164, 4, v164, v184
v_mad_u32_u24 v165, 4, v165, v184
v_mad_u32_u24 v166, 4, v166, v184
s_mov_b32 s2, 0x1040
s_and_b32 s3, s14, 0x1100000
s_cselect_b32 s2, 0x80, s2
v_add_co_u32 v167, vcc, v163, s2
v_add_co_u32 v168, vcc, v164, s2
v_add_co_u32 v169, vcc, v165, s2
v_add_co_u32 v170, vcc, v166, s2
s_branch 57
s_bfe_u32 s3, s14, 0x10014
v_lshrrev_b32_e32 v184, 1, v1
s_mul_i32 s2, 60, s3
s_sub_u32 s2, 63, s2
v_bfi_b32 v184, s2, v1, v184
v_and_b32_e32 v181, 1, v184
v_bfe_u32 v182, v184, 1, 1
v_bfe_u32 v183, v184, 3, 1
v_xor_b32_e32 v181, v181, v182
v_mad_u32_u24 v182, v182, 2, v183
v_mul_u32_u24_e32 v181, 0x109, v181
v_bfe_u32 v183, v184, 2, 1
v_mad_u32_u24 v182, v182, 2, v181
v_xor_b32_e32 v182, v182, v183
v_and_b32_e32 v183, 0xf0, v184
v_or_b32_e32 v182, v182, v183
s_mul_i32 s2, 4, s3
s_sub_u32 s2, 6, s2
v_bfe_u32 v184, v1, s2, 1
v_mul_u32_u24_e32 v184, 0x1040, v184
v_mad_u32_u24 v163, 4, v182, v184
v_xor_b32_e32 v164, 0x307, v182
v_mad_u32_u24 v164, 4, v164, v184
v_xor_b32_e32 v165, 0x30f, v182
v_mad_u32_u24 v165, 4, v165, v184
v_xor_b32_e32 v166, 8, v182
v_mad_u32_u24 v166, 4, v166, v184
s_mov_b32 s2, 0x1040
s_bitcmp1_b32 s14, 20
s_cselect_b32 s2, 0x80, s2
v_add_co_u32 v167, vcc, v163, s2
v_add_co_u32 v168, vcc, v164, s2
v_add_co_u32 v169, vcc, v165, s2
v_add_co_u32 v170, vcc, v166, s2
v_subrev_co_u32 v172, vcc, s58, v172
v_mov_b32_e32 v182, s57
v_cmp_lt_i32_e32 vcc, v172, v182
v_sub_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_mad_i32_i24 v172, v181, s57, v172
v_mad_i32_i24 v174, v181, s62, v174
v_mad_i32_i24 v173, v181, s61, v173
v_mov_b32_e32 v182, s56
v_cmp_lt_i32_e32 vcc, v173, v182
v_sub_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v181
v_mad_i32_i24 v173, v181, v182, v173
v_subrev_co_u32 v173, vcc, s59, v173
v_cmp_lt_i32_e32 vcc, v173, v182
v_sub_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v181
v_mad_i32_i24 v173, v181, s56, v173
v_subrev_co_u32 v174, vcc, s60, v174
s_mov_b32 s35, 0
s_mov_b32 s38, s24
s_mov_b32 s39, 1
s_mov_b32 s66, 0
s_mov_b32 s67, s12
s_mov_b32 s65, s67
s_sub_u32 s75, -1, s74
s_sub_u32 s75, s75, 16
s_bitset1_b32 s14, 21
s_mov_b32 s87, 0
s_mov_b32 s91, 0
v_add_co_u32 v181, vcc, 2, v1
v_bfe_u32 v181, v181, 2, 1
v_cmp_ne_u32_e64 vcc, v181, 1
s_mov_b64 s[6:7], vcc
s_mov_b32 s76, 38
s_mov_b32 s64, 0
s_bitset1_b32 s14, 26
s_call_b64 s[36:37], 4903
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 1
s_branch 2402
s_mov_b64 vcc, s[6:7]
s_nop 0
s_nop 0
v_pk_fma_f16 v116, v118, -1.0, v116 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
_v_pk_mul_f16_05_gfx1x 116, 116
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v119, v117, -1.0, v119 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
_v_pk_mul_f16_05_gfx1x 119, 119
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_add_f16_gfx1x 117, 118, 117
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_mul_f16_05_gfx1x 117, 117
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v118, v117, -1.0, v118 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 116, 116, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 117, 117, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 118, 118, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 119, 119, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v104, v2, s[40:43], 0 idxen
buffer_load_short_d16 v106, v68, s[40:43], 0 idxen
buffer_load_short_d16 v105, v3, s[40:43], 0 idxen
buffer_load_short_d16 v107, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v104, v2, s[92:95], 0 idxen
buffer_load_short_d16_hi v106, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v105, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v107, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v108
ds_read_b128 v[70:73], v171 offset:29440
ds_write_b32 v168, v109
ds_read_b128 v[74:77], v171 offset:29696
ds_write_b32 v169, v110
ds_read_b128 v[86:89], v162 offset:28928
ds_write_b32 v170, v111
ds_read_b128 v[90:93], v162 offset:29056
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 4
s_call_b64 s[36:37], 4699
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v120, v122, -1.0, v120 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
_v_pk_mul_f16_05_gfx1x 120, 120
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v123, v121, -1.0, v123 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
_v_pk_mul_f16_05_gfx1x 123, 123
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_add_f16_gfx1x 121, 122, 121
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_mul_f16_05_gfx1x 121, 121
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v122, v121, -1.0, v122 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 120, 120, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 121, 121, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 122, 122, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 123, 123, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v108, v102, s[40:43], 0 idxen
buffer_load_short_d16 v110, v152, s[40:43], 0 idxen
buffer_load_short_d16 v109, v103, s[40:43], 0 idxen
buffer_load_short_d16 v111, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v108, v102, s[92:95], 0 idxen
buffer_load_short_d16_hi v110, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v109, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v111, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v112 offset:8256
ds_read_b128 v[78:81], v171 offset:33536
ds_write_b32 v164, v113 offset:8256
ds_read_b128 v[82:85], v171 offset:33792
ds_write_b32 v165, v114 offset:8256
ds_read_b128 v[94:97], v162 offset:33024
ds_write_b32 v166, v115 offset:8256
ds_read_b128 v[98:101], v162 offset:33152
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 4
s_call_b64 s[36:37], 4499
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v124, v126, -1.0, v124 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
_v_pk_mul_f16_05_gfx1x 124, 124
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v127, v125, -1.0, v127 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
_v_pk_mul_f16_05_gfx1x 127, 127
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_add_f16_gfx1x 125, 126, 125
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_mul_f16_05_gfx1x 125, 125
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v126, v125, -1.0, v126 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 124, 124, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 125, 125, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 126, 126, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 127, 127, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v112, v2, s[40:43], 0 idxen
buffer_load_short_d16 v114, v68, s[40:43], 0 idxen
buffer_load_short_d16 v113, v3, s[40:43], 0 idxen
buffer_load_short_d16 v115, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v112, v2, s[92:95], 0 idxen
buffer_load_short_d16_hi v114, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v113, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v115, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v116 offset:8256
ds_read_b128 v[70:73], v171 offset:37696
ds_write_b32 v168, v117 offset:8256
ds_read_b128 v[74:77], v171 offset:37952
ds_write_b32 v169, v118 offset:8256
ds_read_b128 v[86:89], v162 offset:37184
ds_write_b32 v170, v119 offset:8256
ds_read_b128 v[90:93], v162 offset:37312
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 4
s_call_b64 s[36:37], 4299
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v128, v130, -1.0, v128 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
_v_pk_mul_f16_05_gfx1x 128, 128
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v131, v129, -1.0, v131 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
_v_pk_mul_f16_05_gfx1x 131, 131
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_add_f16_gfx1x 129, 130, 129
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_mul_f16_05_gfx1x 129, 129
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v130, v129, -1.0, v130 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 128, 128, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 129, 129, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 130, 130, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 131, 131, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v116, v102, s[40:43], 0 idxen
buffer_load_short_d16 v118, v152, s[40:43], 0 idxen
buffer_load_short_d16 v117, v103, s[40:43], 0 idxen
buffer_load_short_d16 v119, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v116, v102, s[92:95], 0 idxen
buffer_load_short_d16_hi v118, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v117, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v119, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v120 offset:16512
ds_read_b128 v[78:81], v171 offset:41792
ds_write_b32 v164, v121 offset:16512
ds_read_b128 v[82:85], v171 offset:42048
ds_write_b32 v165, v122 offset:16512
ds_read_b128 v[94:97], v162 offset:41280
ds_write_b32 v166, v123 offset:16512
ds_read_b128 v[98:101], v162 offset:41408
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 3
s_call_b64 s[36:37], 4098
s_nop 0
s_nop 0
v_pk_fma_f16 v132, v134, -1.0, v132 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
_v_pk_mul_f16_05_gfx1x 132, 132
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v135, v133, -1.0, v135 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
_v_pk_mul_f16_05_gfx1x 135, 135
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_add_f16_gfx1x 133, 134, 133
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_mul_f16_05_gfx1x 133, 133
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v134, v133, -1.0, v134 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 132, 132, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 133, 133, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 134, 134, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 135, 135, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v120, v2, s[40:43], 0 idxen
buffer_load_short_d16 v122, v68, s[40:43], 0 idxen
buffer_load_short_d16 v121, v3, s[40:43], 0 idxen
buffer_load_short_d16 v123, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v120, v2, s[92:95], 0 idxen
buffer_load_short_d16_hi v122, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v121, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v123, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v124 offset:16512
ds_read_b128 v[70:73], v171 offset:45952
ds_write_b32 v168, v125 offset:16512
ds_read_b128 v[74:77], v171 offset:46208
ds_write_b32 v169, v126 offset:16512
ds_read_b128 v[86:89], v162 offset:45440
ds_write_b32 v170, v127 offset:16512
ds_read_b128 v[90:93], v162 offset:45568
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 4
s_call_b64 s[36:37], 3899
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v136, v138, -1.0, v136 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
_v_pk_mul_f16_05_gfx1x 136, 136
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v139, v137, -1.0, v139 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
_v_pk_mul_f16_05_gfx1x 139, 139
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_add_f16_gfx1x 137, 138, 137
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_mul_f16_05_gfx1x 137, 137
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v138, v137, -1.0, v138 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 136, 136, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 137, 137, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 138, 138, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 139, 139, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v124, v102, s[40:43], 0 idxen
buffer_load_short_d16 v126, v152, s[40:43], 0 idxen
buffer_load_short_d16 v125, v103, s[40:43], 0 idxen
buffer_load_short_d16 v127, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v124, v102, s[92:95], 0 idxen
buffer_load_short_d16_hi v126, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v125, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v127, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v128 offset:24768
ds_read_b128 v[78:81], v171 offset:512
ds_write_b32 v164, v129 offset:24768
ds_read_b128 v[82:85], v171 offset:768
ds_write_b32 v165, v130 offset:24768
ds_read_b128 v[94:97], v162
ds_write_b32 v166, v131 offset:24768
ds_read_b128 v[98:101], v162 offset:128
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 4
s_call_b64 s[36:37], 3699
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v140, v142, -1.0, v140 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
_v_pk_mul_f16_05_gfx1x 140, 140
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v143, v141, -1.0, v143 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
_v_pk_mul_f16_05_gfx1x 143, 143
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_add_f16_gfx1x 141, 142, 141
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_mul_f16_05_gfx1x 141, 141
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v142, v141, -1.0, v142 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 140, 140, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 141, 141, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 142, 142, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 143, 143, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v128, v2, s[40:43], 0 idxen
buffer_load_short_d16 v130, v68, s[40:43], 0 idxen
buffer_load_short_d16 v129, v3, s[40:43], 0 idxen
buffer_load_short_d16 v131, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v128, v2, s[92:95], 0 idxen
buffer_load_short_d16_hi v130, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v129, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v131, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v132 offset:24768
ds_read_b128 v[70:73], v171 offset:4672
ds_write_b32 v168, v133 offset:24768
ds_read_b128 v[74:77], v171 offset:4928
ds_write_b32 v169, v134 offset:24768
ds_read_b128 v[86:89], v162 offset:4160
ds_write_b32 v170, v135 offset:24768
ds_read_b128 v[90:93], v162 offset:4288
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 4
s_call_b64 s[36:37], 3499
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v144, v146, -1.0, v144 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
_v_pk_mul_f16_05_gfx1x 144, 144
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v147, v145, -1.0, v147 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
_v_pk_mul_f16_05_gfx1x 147, 147
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_add_f16_gfx1x 145, 146, 145
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_mul_f16_05_gfx1x 145, 145
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v146, v145, -1.0, v146 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 144, 144, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 145, 145, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 146, 146, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 147, 147, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v132, v102, s[40:43], 0 idxen
buffer_load_short_d16 v134, v152, s[40:43], 0 idxen
buffer_load_short_d16 v133, v103, s[40:43], 0 idxen
buffer_load_short_d16 v135, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v132, v102, s[92:95], 0 idxen
buffer_load_short_d16_hi v134, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v133, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v135, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v136 offset:33024
ds_read_b128 v[78:81], v171 offset:8768
ds_write_b32 v164, v137 offset:33024
ds_read_b128 v[82:85], v171 offset:9024
ds_write_b32 v165, v138 offset:33024
ds_read_b128 v[94:97], v162 offset:8256
ds_write_b32 v166, v139 offset:33024
ds_read_b128 v[98:101], v162 offset:8384
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 3
s_call_b64 s[36:37], 3298
s_nop 0
s_nop 0
v_pk_fma_f16 v148, v150, -1.0, v148 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
_v_pk_mul_f16_05_gfx1x 148, 148
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v151, v149, -1.0, v151 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
_v_pk_mul_f16_05_gfx1x 151, 151
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_add_f16_gfx1x 149, 150, 149
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_mul_f16_05_gfx1x 149, 149
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v150, v149, -1.0, v150 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 148, 148, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 149, 149, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 150, 150, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 151, 151, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v136, v2, s[40:43], 0 idxen
buffer_load_short_d16 v138, v68, s[40:43], 0 idxen
buffer_load_short_d16 v137, v3, s[40:43], 0 idxen
buffer_load_short_d16 v139, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v136, v2, s[92:95], 0 idxen
buffer_load_short_d16_hi v138, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v137, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v139, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v140 offset:33024
ds_read_b128 v[70:73], v171 offset:12928
ds_write_b32 v168, v141 offset:33024
ds_read_b128 v[74:77], v171 offset:13184
ds_write_b32 v169, v142 offset:33024
ds_read_b128 v[86:89], v162 offset:12416
ds_write_b32 v170, v143 offset:33024
ds_read_b128 v[90:93], v162 offset:12544
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 4
s_call_b64 s[36:37], 3099
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v104, v106, -1.0, v104 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
_v_pk_mul_f16_05_gfx1x 104, 104
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v107, v105, -1.0, v107 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
_v_pk_mul_f16_05_gfx1x 107, 107
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_add_f16_gfx1x 105, 106, 105
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_mul_f16_05_gfx1x 105, 105
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v106, v105, -1.0, v106 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 104, 104, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 105, 105, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 106, 106, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 107, 107, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v140, v102, s[40:43], 0 idxen
buffer_load_short_d16 v142, v152, s[40:43], 0 idxen
buffer_load_short_d16 v141, v103, s[40:43], 0 idxen
buffer_load_short_d16 v143, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v140, v102, s[92:95], 0 idxen
buffer_load_short_d16_hi v142, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v141, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v143, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v144 offset:41280
ds_read_b128 v[78:81], v171 offset:17024
ds_write_b32 v164, v145 offset:41280
ds_read_b128 v[82:85], v171 offset:17280
ds_write_b32 v165, v146 offset:41280
ds_read_b128 v[94:97], v162 offset:16512
ds_write_b32 v166, v147 offset:41280
ds_read_b128 v[98:101], v162 offset:16640
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 4
s_call_b64 s[36:37], 2899
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v108, v110, -1.0, v108 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v70, v86, v4
v_dot2_f32_f16 v5, v71, v86, v5
v_dot2_f32_f16 v6, v72, v86, v6
v_dot2_f32_f16 v7, v73, v86, v7
_v_pk_mul_f16_05_gfx1x 108, 108
v_dot2_f32_f16 v8, v74, v86, v8
v_dot2_f32_f16 v9, v75, v86, v9
v_dot2_f32_f16 v10, v76, v86, v10
v_dot2_f32_f16 v11, v77, v86, v11
v_pk_fma_f16 v111, v109, -1.0, v111 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v70, v87, v12
v_dot2_f32_f16 v13, v71, v87, v13
v_dot2_f32_f16 v14, v72, v87, v14
v_dot2_f32_f16 v15, v73, v87, v15
_v_pk_mul_f16_05_gfx1x 111, 111
v_dot2_f32_f16 v16, v74, v87, v16
v_dot2_f32_f16 v17, v75, v87, v17
v_dot2_f32_f16 v18, v76, v87, v18
v_dot2_f32_f16 v19, v77, v87, v19
_v_pk_add_f16_gfx1x 109, 110, 109
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_mul_f16_05_gfx1x 109, 109
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
v_pk_fma_f16 v110, v109, -1.0, v110 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
_v_pk_fmac_f16_gfx10_quad_perm 108, 108, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_fmac_f16_gfx10_quad_perm 109, 109, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 110, 110, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
_v_pk_fmac_f16_gfx10_quad_perm 111, 111, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v144, v2, s[40:43], 0 idxen
buffer_load_short_d16 v146, v68, s[40:43], 0 idxen
buffer_load_short_d16 v145, v3, s[40:43], 0 idxen
buffer_load_short_d16 v147, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v144, v2, s[92:95], 0 idxen
buffer_load_short_d16_hi v146, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v145, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v147, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v148 offset:41280
ds_read_b128 v[70:73], v171 offset:21184
ds_write_b32 v168, v149 offset:41280
ds_read_b128 v[74:77], v171 offset:21440
ds_write_b32 v169, v150 offset:41280
ds_read_b128 v[86:89], v162 offset:20672
ds_write_b32 v170, v151 offset:41280
ds_read_b128 v[90:93], v162 offset:20800
s_waitcnt vmcnt(56) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 4
s_call_b64 s[36:37], 2699
s_nop 0
s_nop 0
s_nop 0
v_pk_fma_f16 v112, v114, -1.0, v112 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v4, v78, v94, v4
v_dot2_f32_f16 v5, v79, v94, v5
v_dot2_f32_f16 v6, v80, v94, v6
v_dot2_f32_f16 v7, v81, v94, v7
_v_pk_mul_f16_05_gfx1x 112, 112
v_dot2_f32_f16 v8, v82, v94, v8
v_dot2_f32_f16 v9, v83, v94, v9
v_dot2_f32_f16 v10, v84, v94, v10
v_dot2_f32_f16 v11, v85, v94, v11
v_pk_fma_f16 v115, v113, -1.0, v115 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v12, v78, v95, v12
v_dot2_f32_f16 v13, v79, v95, v13
v_dot2_f32_f16 v14, v80, v95, v14
v_dot2_f32_f16 v15, v81, v95, v15
_v_pk_mul_f16_05_gfx1x 115, 115
v_dot2_f32_f16 v16, v82, v95, v16
v_dot2_f32_f16 v17, v83, v95, v17
v_dot2_f32_f16 v18, v84, v95, v18
v_dot2_f32_f16 v19, v85, v95, v19
_v_pk_add_f16_gfx1x 113, 114, 113
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_mul_f16_05_gfx1x 113, 113
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
v_pk_fma_f16 v114, v113, -1.0, v114 op_sel_hi:[1,0,1]
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
_v_pk_fmac_f16_gfx10_quad_perm 112, 112, 176, 2, 2, 1, 1
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_fmac_f16_gfx10_quad_perm 113, 113, 176, 2, 2, 1, 1
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 114, 114, 176, 2, 2, 1, 1
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
_v_pk_fmac_f16_gfx10_quad_perm 115, 115, 176, 2, 2, 1, 1
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
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x7
buffer_load_short_d16 v148, v102, s[40:43], 0 idxen
buffer_load_short_d16 v150, v152, s[40:43], 0 idxen
buffer_load_short_d16 v149, v103, s[40:43], 0 idxen
buffer_load_short_d16 v151, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v148, v102, s[92:95], 0 idxen
buffer_load_short_d16_hi v150, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v149, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v151, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v104
ds_read_b128 v[78:81], v171 offset:25280
ds_write_b32 v164, v105
ds_read_b128 v[82:85], v171 offset:25536
ds_write_b32 v165, v106
ds_read_b128 v[94:97], v162 offset:24768
ds_write_b32 v166, v107
ds_read_b128 v[98:101], v162 offset:24896
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 63139
s_call_b64 s[36:37], 2498
s_branch 63137
s_mov_b64 vcc, s[6:7]
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
v_mov_b32_dpp v116, v117 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_add_f16_gfx1x 116, 116, 117
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
_v_pk_fmac_f16_gfx10_quad_perm 116, 117, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_mov_b32_dpp v117, v119 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_add_f16_gfx1x 117, 117, 119
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 117, 119, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_mov_b32_dpp v119, v118 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
_v_pk_add_f16_gfx1x 119, 119, 118
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
_v_pk_fmac_f16_gfx10_quad_perm 119, 118, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
_v_pk_add_f16_gfx1x 118, 116, 119
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
_v_pk_add_f16_gfx1x 117, 117, 118
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
_v_pk_mul_f16_05_gfx1x 117, 117
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
v_pk_fma_f16 v118, -1.0, v117, v118 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v106, v68, s[40:43], 0 idxen
buffer_load_short_d16 v105, v3, s[40:43], 0 idxen
buffer_load_short_d16 v107, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v106, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v105, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v107, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v108
ds_read_b128 v[70:73], v171 offset:29440
ds_write_b32 v168, v109
ds_read_b128 v[74:77], v171 offset:29696
ds_write_b32 v169, v110
ds_read_b128 v[86:89], v162 offset:28928
ds_write_b32 v170, v111
ds_read_b128 v[90:93], v162 offset:29056
s_waitcnt vmcnt(42) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 3
s_call_b64 s[36:37], 2290
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
v_mov_b32_dpp v120, v121 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_add_f16_gfx1x 120, 120, 121
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
_v_pk_fmac_f16_gfx10_quad_perm 120, 121, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_mov_b32_dpp v121, v123 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_add_f16_gfx1x 121, 121, 123
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 121, 123, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_mov_b32_dpp v123, v122 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
_v_pk_add_f16_gfx1x 123, 123, 122
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
_v_pk_fmac_f16_gfx10_quad_perm 123, 122, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
_v_pk_add_f16_gfx1x 122, 120, 123
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
_v_pk_add_f16_gfx1x 121, 121, 122
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
_v_pk_mul_f16_05_gfx1x 121, 121
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
v_pk_fma_f16 v122, -1.0, v121, v122 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v110, v152, s[40:43], 0 idxen
buffer_load_short_d16 v109, v103, s[40:43], 0 idxen
buffer_load_short_d16 v111, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v110, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v109, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v111, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v112 offset:8256
ds_read_b128 v[78:81], v171 offset:33536
ds_write_b32 v164, v113 offset:8256
ds_read_b128 v[82:85], v171 offset:33792
ds_write_b32 v165, v114 offset:8256
ds_read_b128 v[94:97], v162 offset:33024
ds_write_b32 v166, v115 offset:8256
ds_read_b128 v[98:101], v162 offset:33152
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 3
s_call_b64 s[36:37], 2082
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
v_mov_b32_dpp v124, v125 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_add_f16_gfx1x 124, 124, 125
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
_v_pk_fmac_f16_gfx10_quad_perm 124, 125, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_mov_b32_dpp v125, v127 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_add_f16_gfx1x 125, 125, 127
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 125, 127, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_mov_b32_dpp v127, v126 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
_v_pk_add_f16_gfx1x 127, 127, 126
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
_v_pk_fmac_f16_gfx10_quad_perm 127, 126, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
_v_pk_add_f16_gfx1x 126, 124, 127
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
_v_pk_add_f16_gfx1x 125, 125, 126
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
_v_pk_mul_f16_05_gfx1x 125, 125
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
v_pk_fma_f16 v126, -1.0, v125, v126 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v114, v68, s[40:43], 0 idxen
buffer_load_short_d16 v113, v3, s[40:43], 0 idxen
buffer_load_short_d16 v115, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v114, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v113, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v115, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v116 offset:8256
ds_read_b128 v[70:73], v171 offset:37696
ds_write_b32 v168, v117 offset:8256
ds_read_b128 v[74:77], v171 offset:37952
ds_write_b32 v169, v118 offset:8256
ds_read_b128 v[86:89], v162 offset:37184
ds_write_b32 v170, v119 offset:8256
ds_read_b128 v[90:93], v162 offset:37312
s_waitcnt vmcnt(42) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 3
s_call_b64 s[36:37], 1874
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
v_mov_b32_dpp v128, v129 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_add_f16_gfx1x 128, 128, 129
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
_v_pk_fmac_f16_gfx10_quad_perm 128, 129, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_mov_b32_dpp v129, v131 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_add_f16_gfx1x 129, 129, 131
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 129, 131, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_mov_b32_dpp v131, v130 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
_v_pk_add_f16_gfx1x 131, 131, 130
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
_v_pk_fmac_f16_gfx10_quad_perm 131, 130, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
_v_pk_add_f16_gfx1x 130, 128, 131
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
_v_pk_add_f16_gfx1x 129, 129, 130
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
_v_pk_mul_f16_05_gfx1x 129, 129
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
v_pk_fma_f16 v130, -1.0, v129, v130 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v118, v152, s[40:43], 0 idxen
buffer_load_short_d16 v117, v103, s[40:43], 0 idxen
buffer_load_short_d16 v119, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v118, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v117, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v119, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v120 offset:16512
ds_read_b128 v[78:81], v171 offset:41792
ds_write_b32 v164, v121 offset:16512
ds_read_b128 v[82:85], v171 offset:42048
ds_write_b32 v165, v122 offset:16512
ds_read_b128 v[94:97], v162 offset:41280
ds_write_b32 v166, v123 offset:16512
ds_read_b128 v[98:101], v162 offset:41408
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 2
s_call_b64 s[36:37], 1665
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
v_mov_b32_dpp v132, v133 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_add_f16_gfx1x 132, 132, 133
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
_v_pk_fmac_f16_gfx10_quad_perm 132, 133, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_mov_b32_dpp v133, v135 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_add_f16_gfx1x 133, 133, 135
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 133, 135, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_mov_b32_dpp v135, v134 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
_v_pk_add_f16_gfx1x 135, 135, 134
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
_v_pk_fmac_f16_gfx10_quad_perm 135, 134, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
_v_pk_add_f16_gfx1x 134, 132, 135
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
_v_pk_add_f16_gfx1x 133, 133, 134
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
_v_pk_mul_f16_05_gfx1x 133, 133
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
v_pk_fma_f16 v134, -1.0, v133, v134 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v122, v68, s[40:43], 0 idxen
buffer_load_short_d16 v121, v3, s[40:43], 0 idxen
buffer_load_short_d16 v123, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v122, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v121, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v123, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v124 offset:16512
ds_read_b128 v[70:73], v171 offset:45952
ds_write_b32 v168, v125 offset:16512
ds_read_b128 v[74:77], v171 offset:46208
ds_write_b32 v169, v126 offset:16512
ds_read_b128 v[86:89], v162 offset:45440
ds_write_b32 v170, v127 offset:16512
ds_read_b128 v[90:93], v162 offset:45568
s_waitcnt vmcnt(42) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 3
s_call_b64 s[36:37], 1458
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
v_mov_b32_dpp v136, v137 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_add_f16_gfx1x 136, 136, 137
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
_v_pk_fmac_f16_gfx10_quad_perm 136, 137, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_mov_b32_dpp v137, v139 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_add_f16_gfx1x 137, 137, 139
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 137, 139, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_mov_b32_dpp v139, v138 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
_v_pk_add_f16_gfx1x 139, 139, 138
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
_v_pk_fmac_f16_gfx10_quad_perm 139, 138, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
_v_pk_add_f16_gfx1x 138, 136, 139
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
_v_pk_add_f16_gfx1x 137, 137, 138
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
_v_pk_mul_f16_05_gfx1x 137, 137
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
v_pk_fma_f16 v138, -1.0, v137, v138 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v126, v152, s[40:43], 0 idxen
buffer_load_short_d16 v125, v103, s[40:43], 0 idxen
buffer_load_short_d16 v127, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v126, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v125, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v127, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v128 offset:24768
ds_read_b128 v[78:81], v171 offset:512
ds_write_b32 v164, v129 offset:24768
ds_read_b128 v[82:85], v171 offset:768
ds_write_b32 v165, v130 offset:24768
ds_read_b128 v[94:97], v162
ds_write_b32 v166, v131 offset:24768
ds_read_b128 v[98:101], v162 offset:128
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 3
s_call_b64 s[36:37], 1250
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
v_mov_b32_dpp v140, v141 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_add_f16_gfx1x 140, 140, 141
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
_v_pk_fmac_f16_gfx10_quad_perm 140, 141, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_mov_b32_dpp v141, v143 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_add_f16_gfx1x 141, 141, 143
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 141, 143, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_mov_b32_dpp v143, v142 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
_v_pk_add_f16_gfx1x 143, 143, 142
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
_v_pk_fmac_f16_gfx10_quad_perm 143, 142, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
_v_pk_add_f16_gfx1x 142, 140, 143
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
_v_pk_add_f16_gfx1x 141, 141, 142
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
_v_pk_mul_f16_05_gfx1x 141, 141
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
v_pk_fma_f16 v142, -1.0, v141, v142 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v130, v68, s[40:43], 0 idxen
buffer_load_short_d16 v129, v3, s[40:43], 0 idxen
buffer_load_short_d16 v131, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v130, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v129, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v131, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v132 offset:24768
ds_read_b128 v[70:73], v171 offset:4672
ds_write_b32 v168, v133 offset:24768
ds_read_b128 v[74:77], v171 offset:4928
ds_write_b32 v169, v134 offset:24768
ds_read_b128 v[86:89], v162 offset:4160
ds_write_b32 v170, v135 offset:24768
ds_read_b128 v[90:93], v162 offset:4288
s_waitcnt vmcnt(42) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 3
s_call_b64 s[36:37], 1042
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
v_mov_b32_dpp v144, v145 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_add_f16_gfx1x 144, 144, 145
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
_v_pk_fmac_f16_gfx10_quad_perm 144, 145, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_mov_b32_dpp v145, v147 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_add_f16_gfx1x 145, 145, 147
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 145, 147, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_mov_b32_dpp v147, v146 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
_v_pk_add_f16_gfx1x 147, 147, 146
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
_v_pk_fmac_f16_gfx10_quad_perm 147, 146, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
_v_pk_add_f16_gfx1x 146, 144, 147
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
_v_pk_add_f16_gfx1x 145, 145, 146
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
_v_pk_mul_f16_05_gfx1x 145, 145
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
v_pk_fma_f16 v146, -1.0, v145, v146 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v134, v152, s[40:43], 0 idxen
buffer_load_short_d16 v133, v103, s[40:43], 0 idxen
buffer_load_short_d16 v135, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v134, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v133, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v135, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v136 offset:33024
ds_read_b128 v[78:81], v171 offset:8768
ds_write_b32 v164, v137 offset:33024
ds_read_b128 v[82:85], v171 offset:9024
ds_write_b32 v165, v138 offset:33024
ds_read_b128 v[94:97], v162 offset:8256
ds_write_b32 v166, v139 offset:33024
ds_read_b128 v[98:101], v162 offset:8384
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 2
s_call_b64 s[36:37], 833
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
v_mov_b32_dpp v148, v149 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_add_f16_gfx1x 148, 148, 149
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
_v_pk_fmac_f16_gfx10_quad_perm 148, 149, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_mov_b32_dpp v149, v151 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_add_f16_gfx1x 149, 149, 151
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 149, 151, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_mov_b32_dpp v151, v150 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
_v_pk_add_f16_gfx1x 151, 151, 150
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
_v_pk_fmac_f16_gfx10_quad_perm 151, 150, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
_v_pk_add_f16_gfx1x 150, 148, 151
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
_v_pk_add_f16_gfx1x 149, 149, 150
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
_v_pk_mul_f16_05_gfx1x 149, 149
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
v_pk_fma_f16 v150, -1.0, v149, v150 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v138, v68, s[40:43], 0 idxen
buffer_load_short_d16 v137, v3, s[40:43], 0 idxen
buffer_load_short_d16 v139, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v138, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v137, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v139, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v140 offset:33024
ds_read_b128 v[70:73], v171 offset:12928
ds_write_b32 v168, v141 offset:33024
ds_read_b128 v[74:77], v171 offset:13184
ds_write_b32 v169, v142 offset:33024
ds_read_b128 v[86:89], v162 offset:12416
ds_write_b32 v170, v143 offset:33024
ds_read_b128 v[90:93], v162 offset:12544
s_waitcnt vmcnt(42) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 3
s_call_b64 s[36:37], 626
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
v_mov_b32_dpp v104, v105 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_add_f16_gfx1x 104, 104, 105
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
_v_pk_fmac_f16_gfx10_quad_perm 104, 105, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_mov_b32_dpp v105, v107 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_add_f16_gfx1x 105, 105, 107
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 105, 107, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_mov_b32_dpp v107, v106 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
_v_pk_add_f16_gfx1x 107, 107, 106
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
_v_pk_fmac_f16_gfx10_quad_perm 107, 106, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
_v_pk_add_f16_gfx1x 106, 104, 107
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
_v_pk_add_f16_gfx1x 105, 105, 106
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
_v_pk_mul_f16_05_gfx1x 105, 105
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
v_pk_fma_f16 v106, -1.0, v105, v106 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v142, v152, s[40:43], 0 idxen
buffer_load_short_d16 v141, v103, s[40:43], 0 idxen
buffer_load_short_d16 v143, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v142, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v141, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v143, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v144 offset:41280
ds_read_b128 v[78:81], v171 offset:17024
ds_write_b32 v164, v145 offset:41280
ds_read_b128 v[82:85], v171 offset:17280
ds_write_b32 v165, v146 offset:41280
ds_read_b128 v[94:97], v162 offset:16512
ds_write_b32 v166, v147 offset:41280
ds_read_b128 v[98:101], v162 offset:16640
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 3
s_call_b64 s[36:37], 418
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
v_mov_b32_dpp v108, v109 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v70, v88, v20
v_dot2_f32_f16 v21, v71, v88, v21
v_dot2_f32_f16 v22, v72, v88, v22
v_dot2_f32_f16 v23, v73, v88, v23
_v_pk_add_f16_gfx1x 108, 108, 109
v_dot2_f32_f16 v24, v74, v88, v24
v_dot2_f32_f16 v25, v75, v88, v25
v_dot2_f32_f16 v26, v76, v88, v26
v_dot2_f32_f16 v27, v77, v88, v27
_v_pk_fmac_f16_gfx10_quad_perm 108, 109, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v70, v89, v28
v_dot2_f32_f16 v29, v71, v89, v29
v_dot2_f32_f16 v30, v72, v89, v30
v_dot2_f32_f16 v31, v73, v89, v31
v_mov_b32_dpp v109, v111 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v74, v89, v32
v_dot2_f32_f16 v33, v75, v89, v33
v_dot2_f32_f16 v34, v76, v89, v34
v_dot2_f32_f16 v35, v77, v89, v35
_v_pk_add_f16_gfx1x 109, 109, 111
v_dot2_f32_f16 v36, v70, v90, v36
v_dot2_f32_f16 v37, v71, v90, v37
v_dot2_f32_f16 v38, v72, v90, v38
v_dot2_f32_f16 v39, v73, v90, v39
_v_pk_fmac_f16_gfx10_quad_perm 109, 111, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v74, v90, v40
v_dot2_f32_f16 v41, v75, v90, v41
v_dot2_f32_f16 v42, v76, v90, v42
v_dot2_f32_f16 v43, v77, v90, v43
v_mov_b32_dpp v111, v110 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v70, v91, v44
v_dot2_f32_f16 v45, v71, v91, v45
v_dot2_f32_f16 v46, v72, v91, v46
v_dot2_f32_f16 v47, v73, v91, v47
_v_pk_add_f16_gfx1x 111, 111, 110
v_dot2_f32_f16 v48, v74, v91, v48
v_dot2_f32_f16 v49, v75, v91, v49
v_dot2_f32_f16 v50, v76, v91, v50
v_dot2_f32_f16 v51, v77, v91, v51
_v_pk_fmac_f16_gfx10_quad_perm 111, 110, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v70, v92, v52
v_dot2_f32_f16 v53, v71, v92, v53
v_dot2_f32_f16 v54, v72, v92, v54
v_dot2_f32_f16 v55, v73, v92, v55
_v_pk_add_f16_gfx1x 110, 108, 111
v_dot2_f32_f16 v56, v74, v92, v56
v_dot2_f32_f16 v57, v75, v92, v57
v_dot2_f32_f16 v58, v76, v92, v58
v_dot2_f32_f16 v59, v77, v92, v59
_v_pk_add_f16_gfx1x 109, 109, 110
v_dot2_f32_f16 v60, v70, v93, v60
v_dot2_f32_f16 v61, v71, v93, v61
v_dot2_f32_f16 v62, v72, v93, v62
v_dot2_f32_f16 v63, v73, v93, v63
_v_pk_mul_f16_05_gfx1x 109, 109
v_dot2_f32_f16 v64, v74, v93, v64
v_dot2_f32_f16 v65, v75, v93, v65
v_dot2_f32_f16 v66, v76, v93, v66
v_dot2_f32_f16 v67, v77, v93, v67
v_pk_fma_f16 v110, -1.0, v109, v110 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s51
s_addc_u32 s41, s41, 0
s_add_u32 s92, s92, s51
s_addc_u32 s93, s93, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v146, v68, s[40:43], 0 idxen
buffer_load_short_d16 v145, v3, s[40:43], 0 idxen
buffer_load_short_d16 v147, v69, s[40:43], 0 idxen
buffer_load_short_d16_hi v146, v68, s[92:95], 0 idxen
buffer_load_short_d16_hi v145, v3, s[92:95], 0 idxen
buffer_load_short_d16_hi v147, v69, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v167, v148 offset:41280
ds_read_b128 v[70:73], v171 offset:21184
ds_write_b32 v168, v149 offset:41280
ds_read_b128 v[74:77], v171 offset:21440
ds_write_b32 v169, v150 offset:41280
ds_read_b128 v[86:89], v162 offset:20672
ds_write_b32 v170, v151 offset:41280
ds_read_b128 v[90:93], v162 offset:20800
s_waitcnt vmcnt(42) lgkmcnt(8)
s_bitset0_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 3
s_call_b64 s[36:37], 210
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
v_mov_b32_dpp v112, v113 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v20, v78, v96, v20
v_dot2_f32_f16 v21, v79, v96, v21
v_dot2_f32_f16 v22, v80, v96, v22
v_dot2_f32_f16 v23, v81, v96, v23
_v_pk_add_f16_gfx1x 112, 112, 113
v_dot2_f32_f16 v24, v82, v96, v24
v_dot2_f32_f16 v25, v83, v96, v25
v_dot2_f32_f16 v26, v84, v96, v26
v_dot2_f32_f16 v27, v85, v96, v27
_v_pk_fmac_f16_gfx10_quad_perm 112, 113, 176, 0, 2, 1, 3
v_dot2_f32_f16 v28, v78, v97, v28
v_dot2_f32_f16 v29, v79, v97, v29
v_dot2_f32_f16 v30, v80, v97, v30
v_dot2_f32_f16 v31, v81, v97, v31
v_mov_b32_dpp v113, v115 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v32, v82, v97, v32
v_dot2_f32_f16 v33, v83, v97, v33
v_dot2_f32_f16 v34, v84, v97, v34
v_dot2_f32_f16 v35, v85, v97, v35
_v_pk_add_f16_gfx1x 113, 113, 115
v_dot2_f32_f16 v36, v78, v98, v36
v_dot2_f32_f16 v37, v79, v98, v37
v_dot2_f32_f16 v38, v80, v98, v38
v_dot2_f32_f16 v39, v81, v98, v39
_v_pk_fmac_f16_gfx10_quad_perm 113, 115, 176, 0, 2, 1, 3
v_dot2_f32_f16 v40, v82, v98, v40
v_dot2_f32_f16 v41, v83, v98, v41
v_dot2_f32_f16 v42, v84, v98, v42
v_dot2_f32_f16 v43, v85, v98, v43
v_mov_b32_dpp v115, v114 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_dot2_f32_f16 v44, v78, v99, v44
v_dot2_f32_f16 v45, v79, v99, v45
v_dot2_f32_f16 v46, v80, v99, v46
v_dot2_f32_f16 v47, v81, v99, v47
_v_pk_add_f16_gfx1x 115, 115, 114
v_dot2_f32_f16 v48, v82, v99, v48
v_dot2_f32_f16 v49, v83, v99, v49
v_dot2_f32_f16 v50, v84, v99, v50
v_dot2_f32_f16 v51, v85, v99, v51
_v_pk_fmac_f16_gfx10_quad_perm 115, 114, 176, 0, 2, 1, 3
v_dot2_f32_f16 v52, v78, v100, v52
v_dot2_f32_f16 v53, v79, v100, v53
v_dot2_f32_f16 v54, v80, v100, v54
v_dot2_f32_f16 v55, v81, v100, v55
_v_pk_add_f16_gfx1x 114, 112, 115
v_dot2_f32_f16 v56, v82, v100, v56
v_dot2_f32_f16 v57, v83, v100, v57
v_dot2_f32_f16 v58, v84, v100, v58
v_dot2_f32_f16 v59, v85, v100, v59
_v_pk_add_f16_gfx1x 113, 113, 114
v_dot2_f32_f16 v60, v78, v101, v60
v_dot2_f32_f16 v61, v79, v101, v61
v_dot2_f32_f16 v62, v80, v101, v62
v_dot2_f32_f16 v63, v81, v101, v63
_v_pk_mul_f16_05_gfx1x 113, 113
v_dot2_f32_f16 v64, v82, v101, v64
v_dot2_f32_f16 v65, v83, v101, v65
v_dot2_f32_f16 v66, v84, v101, v66
v_dot2_f32_f16 v67, v85, v101, v67
v_pk_fma_f16 v114, -1.0, v113, v114 op_sel_hi:[0,1,1]
s_setprio 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, s53
s_add_u32 s92, s92, s52
s_addc_u32 s93, s93, s53
s_sub_u32 s55, s55, 1
s_cselect_b32 s43, 0x11014000, s43
s_clause 0x5
buffer_load_short_d16 v150, v152, s[40:43], 0 idxen
buffer_load_short_d16 v149, v103, s[40:43], 0 idxen
buffer_load_short_d16 v151, v153, s[40:43], 0 idxen
buffer_load_short_d16_hi v150, v152, s[92:95], 0 idxen
buffer_load_short_d16_hi v149, v103, s[92:95], 0 idxen
buffer_load_short_d16_hi v151, v153, s[92:95], 0 idxen
s_clause 0x7
ds_write_b32 v163, v104
ds_read_b128 v[78:81], v171 offset:25280
ds_write_b32 v164, v105
ds_read_b128 v[82:85], v171 offset:25536
ds_write_b32 v165, v106
ds_read_b128 v[94:97], v162 offset:24768
ds_write_b32 v166, v107
ds_read_b128 v[98:101], v162 offset:24896
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s14, 26
s_add_u32 s54, s54, -2
s_cbranch_scc1 63042
s_call_b64 s[36:37], 1
s_branch 63040
v_nop
s_cmp_eq_u32 s64, 0
s_cbranch_scc0 8
s_branch 740
s_add_u32 s64, s64, 1
s_andn2_b32 s64, s64, 1
s_bitcmp1_b32 s14, 26
s_cselect_b32 s92, s51, s52
s_cselect_b32 s93, 0, s53
s_sub_u32 s40, s40, s92
s_subb_u32 s41, s41, s93
s_cmp_eq_u32 s76, 0
s_cbranch_scc0 5
s_cbranch_scc1 754
s_nop 0
s_nop 0
s_add_u32 s76, s76, 1
s_andn2_b32 s76, s76, 1
s_min_u32 s54, s64, s76
s_sub_u32 s64, s64, s54
s_sub_u32 s76, s76, s54
s_sub_u32 s54, s54, 2
s_lshr_b32 s92, s51, 1
s_add_u32 s92, s40, s92
s_addc_u32 s93, s41, 0
s_mov_b64 s[94:95], s[42:43]
s_bitcmp1_b32 s14, 18
s_cselect_b32 s95, 0, 0x11014000
s_setpc_b64 s[36:37]
s_nop 0
s_nop 0
s_bitcmp1_b32 s14, 17
s_cbranch_scc1 253
s_add_u32 s70, s70, s13
s_cmp_eq_u32 s70, 0
s_cbranch_scc1 250
s_mov_b32 s71, 0
s_bitcmp1_b32 s14, 16
s_cbranch_scc1 239
s_add_u32 s69, s12, 15
s_lshr_b32 s69, s69, 4
v_mov_b32_e32 v182, s70
v_mul_u32_u24_e32 v182, s69, v182
v_add_co_u32 v182, vcc, s13, v182
v_sub_co_u32 v182, vcc, v182, 1
v_ffbh_u32_e32 v186, s13
v_lshlrev_b32_e64 v187, v186, s13
v_and_b32_e32 v185, 0xffffff00, v187
v_cmp_eq_u32_e32 vcc, 0x80000000, v187
v_cvt_f32_u32_e32 v185, v185
v_rcp_f32_e32 v181, v185
v_sub_co_ci_u32_e32 v184, vcc, 32, v186, vcc
v_cvt_f32_ubyte0_e32 v186, v187
v_fma_f32 v185, v185, v181, -1.0
v_fma_f32 v185, v186, v181, v185
v_fmaak_f32 v185, v185, v181, 0x9f000000
v_mul_f32_e32 v185, 0x5f800000, v185
v_mov_b32_e32 v186, 0
v_cvt_flr_i32_f32_e64 v185, -v185
v_lshl_add_u32 v181, v181, 9, v185
v_mad_u64_u32 v[186:187], vcc, v187, v181, v[186:187]
v_sub_co_ci_u32_e64 v181, vcc, v181, -1, vcc
v_mul_hi_u32 v185, v182, v181
v_add_co_u32 v181, vcc, v185, v182
v_add_co_ci_u32_e64 v185, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v184
v_cndmask_b32_e32 v181, v181, v185, vcc
v_alignbit_b32 v181, v185, v181, v184
s_nop 0
v_readfirstlane_b32 s68, v181
v_mul_u32_u24_e64 v181, v181, s4
v_ffbh_u32_e32 v186, s69
v_lshlrev_b32_e64 v187, v186, s69
v_and_b32_e32 v185, 0xffffff00, v187
v_cmp_eq_u32_e32 vcc, 0x80000000, v187
v_cvt_f32_u32_e32 v185, v185
v_rcp_f32_e32 v182, v185
v_sub_co_ci_u32_e32 v184, vcc, 32, v186, vcc
v_cvt_f32_ubyte0_e32 v186, v187
v_fma_f32 v185, v185, v182, -1.0
v_fma_f32 v185, v186, v182, v185
v_fmaak_f32 v185, v185, v182, 0x9f000000
v_mul_f32_e32 v185, 0x5f800000, v185
v_mov_b32_e32 v186, 0
v_cvt_flr_i32_f32_e64 v185, -v185
v_lshl_add_u32 v182, v182, 9, v185
v_mad_u64_u32 v[186:187], vcc, v187, v182, v[186:187]
v_sub_co_ci_u32_e64 v182, vcc, v182, -1, vcc
v_mul_hi_u32 v185, v181, v182
v_add_co_u32 v182, vcc, v185, v181
v_add_co_ci_u32_e64 v185, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v184
v_cndmask_b32_e32 v182, v182, v185, vcc
v_alignbit_b32 v182, v185, v182, v184
v_readfirstlane_b32 s2, v181
v_readfirstlane_b32 s66, v182
s_mul_i32 s66, s66, s69
s_sub_u32 s66, s2, s66
v_sub_co_u32 v182, vcc, s4, v182
v_sub_co_u32 v182, vcc, s13, v182
v_and_b32_e64 v184, v1, 63
v_cmp_eq_u32_e64 vcc, v184, 0
v_cndmask_b32_e32 v182, 1, v182, vcc
s_sub_u32 s3, 0, s57
s_sub_u32 s15, 0, s56
v_mul_u32_u24_e64 v186, v182, 32
v_ffbh_u32_e32 v188, s3
v_lshlrev_b32_e64 v189, v188, s3
v_and_b32_e32 v190, 0xffffff00, v189
v_cmp_eq_u32_e32 vcc, 0x80000000, v189
v_cvt_f32_u32_e32 v190, v190
v_rcp_f32_e32 v184, v190
v_sub_co_ci_u32_e32 v187, vcc, 32, v188, vcc
v_cvt_f32_ubyte0_e32 v188, v189
v_fma_f32 v190, v190, v184, -1.0
v_fma_f32 v190, v188, v184, v190
v_fmaak_f32 v190, v190, v184, 0x9f000000
v_mul_f32_e32 v190, 0x5f800000, v190
v_mov_b32_e32 v188, 0
v_cvt_flr_i32_f32_e64 v190, -v190
v_lshl_add_u32 v184, v184, 9, v190
v_mad_u64_u32 v[188:189], vcc, v189, v184, v[188:189]
v_sub_co_ci_u32_e64 v184, vcc, v184, -1, vcc
v_mul_hi_u32 v188, v186, v184
v_add_co_u32 v184, vcc, v188, v186
v_add_co_ci_u32_e64 v188, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v187
v_cndmask_b32_e32 v184, v184, v188, vcc
v_alignbit_b32 v184, v188, v184, v187
v_mad_i32_i24 v185, v184, s57, v186
v_mul_u32_u24_e64 v186, v184, 1
v_ffbh_u32_e32 v188, s15
v_lshlrev_b32_e64 v189, v188, s15
v_and_b32_e32 v190, 0xffffff00, v189
v_cmp_eq_u32_e32 vcc, 0x80000000, v189
v_cvt_f32_u32_e32 v190, v190
v_rcp_f32_e32 v184, v190
v_sub_co_ci_u32_e32 v187, vcc, 32, v188, vcc
v_cvt_f32_ubyte0_e32 v188, v189
v_fma_f32 v190, v190, v184, -1.0
v_fma_f32 v190, v188, v184, v190
v_fmaak_f32 v190, v190, v184, 0x9f000000
v_mul_f32_e32 v190, 0x5f800000, v190
v_mov_b32_e32 v188, 0
v_cvt_flr_i32_f32_e64 v190, -v190
v_lshl_add_u32 v184, v184, 9, v190
v_mad_u64_u32 v[188:189], vcc, v189, v184, v[188:189]
v_sub_co_ci_u32_e64 v184, vcc, v184, -1, vcc
v_mul_hi_u32 v188, v186, v184
v_add_co_u32 v184, vcc, v188, v186
v_add_co_ci_u32_e64 v188, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v187
v_cndmask_b32_e32 v184, v184, v188, vcc
v_alignbit_b32 v184, v188, v184, v187
v_mad_i32_i24 v186, v184, s56, v186
v_readfirstlane_b32 s58, v185
v_readfirstlane_b32 s59, v186
v_readfirstlane_b32 s60, v184
v_add_co_u32 v172, vcc, s58, v172
v_add_co_ci_u32_e64 v187, vcc, 0, 0, vcc
v_mad_i32_i24 v172, v187, s57, v172
v_mad_i32_i24 v174, v187, s62, v174
v_mad_i32_i24 v173, v187, s61, v173
v_cmp_ge_i32_e64 vcc, v173, 0
v_add_co_ci_u32_e64 v187, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v187
v_mad_i32_i24 v173, v187, s56, v173
v_add_co_u32 v173, vcc, s59, v173
v_add_co_ci_u32_e64 v187, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v187
v_mad_i32_i24 v173, v187, s56, v173
v_add_co_u32 v174, vcc, s60, v174
v_readlane_b32 s58, v185, 1
v_readlane_b32 s59, v186, 1
v_readlane_b32 s60, v184, 1
s_add_u32 s67, s66, s68
s_cmp_le_u32 s67, s69
s_cselect_b32 s92, 0x20000, 0
s_cselect_b32 s67, s67, s69
s_or_b32 s14, s14, s92
s_lshl_b32 s66, s66, 4
s_lshl_b32 s67, s67, 4
s_min_u32 s67, s67, s12
s_cmp_eq_u32 s4, s13
s_cselect_b32 s92, 0x20000, 0
s_or_b32 s14, s14, s92
s_bitset1_b32 s14, 16
s_branch 48
s_lshr_b32 s66, s66, 4
s_add_u32 s67, s66, s68
s_sub_u32 s67, s67, s69
s_mov_b32 s66, 0
s_lshl_b32 s67, s67, 4
s_min_u32 s67, s67, s12
s_bitset1_b32 s14, 17
s_branch 12
s_bitset1_b32 s14, 18
s_mov_b32 s43, 0
s_mov_b32 s55, -1
s_mov_b32 s64, 40
s_branch 36
s_add_u32 s65, s65, 16
s_cmp_ge_u32 s65, s67
s_cbranch_scc0 33
s_bitset1_b32 s14, 22
s_sub_u32 s70, s70, s13
s_subb_u32 s71, s71, 0
s_cbranch_scc1 65269
v_add_co_u32 v172, vcc, s58, v172
v_add_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_mad_i32_i24 v172, v181, s57, v172
v_mad_i32_i24 v174, v181, s62, v174
v_mad_i32_i24 v173, v181, s61, v173
v_cmp_ge_i32_e64 vcc, v173, 0
v_add_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v181
v_mad_i32_i24 v173, v181, s56, v173
v_add_co_u32 v173, vcc, s59, v173
v_add_co_ci_u32_e64 v181, vcc, 0, 0, vcc
v_add_co_u32 v174, vcc, v174, v181
v_mad_i32_i24 v173, v181, s56, v173
v_add_co_u32 v174, vcc, s60, v174
s_mov_b32 s65, s66
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccz 257
v_subrev_co_u32 v181, vcc, s57, v172
v_subrev_co_u32 v182, vcc, s56, v173
s_bitcmp1_b32 s14, 22
s_cbranch_scc0 66
s_bitset0_b32 s14, 22
s_bfe_u32 s2, s14, 0x10014
v_mul_u32_u24_e32 v184, 2, v181
v_mul_u32_u24_e32 v185, 2, v182
v_cvt_pk_u16_u32 v187, v184, v185
v_and_b32_e64 v184, v1, 1
v_cmp_eq_u32_e64 vcc, v184, 1
v_cndmask_b32_e32 v187, v174, v187, vcc
v_lshrrev_b32_e32 v183, 1, v1
v_bfe_u32 v188, v183, s2, 1
v_lshrrev_b32_e32 v183, 1, v1
v_bfi_b32 v183, 1, v1, v183
v_lshrrev_b32_e32 v184, 2, v1
v_bfi_b32 v184, 1, v1, v184
v_cmp_eq_u32_e64 vcc, s2, 0
v_cndmask_b32_e32 v183, v184, v183, vcc
s_sub_u32 s2, 1, s2
v_lshrrev_b32_e32 v184, s2, v183
v_bfi_b32 v183, 32, v184, v183
v_and_b32_e32 v183, 63, v183
v_add_co_u32 v184, vcc, 16, v183
v_and_b32_e64 v185, v1, 2
v_cmp_eq_u32_e64 vcc, v185, 0
v_cndmask_b32_e32 v184, v184, v183, vcc
v_lshlrev_b32_e32 v185, 14, v188
v_mad_u32_u24 v184, 4, v184, v185
v_add_co_u32 v183, vcc, s78, v184
ds_write_b32 v183, v187
v_writelane_b32 v185, s14, 0
v_writelane_b32 v185, s67, 1
v_writelane_b32 v185, s66, 2
v_and_b32_e64 v183, v1, 63
v_cmp_ge_u32_e64 vcc, v183, 3
v_mov_b32_e32 v186, 0x4000
v_cndmask_b32_e32 v183, v183, v186, vcc
v_mad_i32_i24 v183, v183, 4, s78
ds_write_b32 v183, v185 offset:256
s_add_u32 s78, s78, 0x18c
s_cmp_eq_u32 s78, 0x10000
s_cselect_b32 s78, 0xc220, s78
v_mov_b32_dpp v183, v174 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v181, v181 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v182, v182 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_readfirstlane_b32 s63, v183
v_sub_co_u32 v184, vcc, v183, s63
v_mul_lo_u32 v184, v184, s44
v_and_b32_e64 v188, v1, 3
v_ashrrev_i32_e64 v189, 1, s27
v_subrev_co_u32 v188, vcc, v189, v188
v_ashrrev_i32_e64 v189, 1, s35
v_mad_i32_i24 v185, v189, 3, v188
s_bfe_u32 s2, s14, 0x10014
v_lshrrev_b32_e32 v187, 2, v1
v_and_b32_e32 v187, s2, v187
v_mad_i32_i24 v185, v187, 3, v185
v_add_co_u32 v186, vcc, 1, s38
v_ashrrev_i32_e32 v186, 1, v186
v_add_co_u32 v187, vcc, 1, s26
v_ashrrev_i32_e32 v187, 1, v187
v_sub_nc_i32 v186, v186, v187
v_cmp_ge_u32_e64 s[2:3], v183, s8
v_mad_i32_i24 v181, v181, 2, v185
v_cmp_ge_u32_e64 s[22:23], v181, s11
v_add_co_u32 v181, vcc, v181, v184
s_or_b64 s[22:23], s[22:23], s[2:3]
v_mad_i32_i24 v182, v182, 2, v186
v_cmp_ge_u32_e64 s[96:97], v182, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v2, v182, s49, v181
v_cndmask_b32_e64 v2, v2, -1, s[96:97]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[96:97], v182, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v3, v182, s49, v181
v_cndmask_b32_e64 v3, v3, -1, s[96:97]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[96:97], v182, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v68, v182, s49, v181
v_cndmask_b32_e64 v68, v68, -1, s[96:97]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[96:97], v182, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v69, v182, s49, v181
v_cndmask_b32_e64 v69, v69, -1, s[96:97]
s_bitcmp1_b32 s14, 20
s_cbranch_scc0 60
v_mov_b32_dpp v183, v174 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v181, v172 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v182, v173 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[2:3], v183, s8
v_sub_co_u32 v184, vcc, v183, s63
v_mul_lo_u32 v184, v184, s44
v_sub_co_u32 v181, vcc, v181, s57
v_sub_co_u32 v182, vcc, v182, s56
v_mad_i32_i24 v181, v181, 2, v185
v_cmp_ge_u32_e64 s[22:23], v181, s11
v_add_co_u32 v181, vcc, v181, v184
s_or_b64 s[22:23], s[22:23], s[2:3]
v_mad_i32_i24 v182, v182, 2, v186
v_cmp_ge_u32_e64 s[96:97], v182, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v102, v182, s49, v181
v_cndmask_b32_e64 v102, v102, -1, s[96:97]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[96:97], v182, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v103, v182, s49, v181
v_cndmask_b32_e64 v103, v103, -1, s[96:97]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[96:97], v182, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v152, v182, s49, v181
v_cndmask_b32_e64 v152, v152, -1, s[96:97]
v_add_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[96:97], v182, s10
s_or_b64 s[96:97], s[22:23], s[96:97]
v_mad_u32_u24 v153, v182, s49, v181
v_cndmask_b32_e64 v153, v153, -1, s[96:97]
s_branch 26
s_bitcmp1_b32 s14, 24
s_cselect_b32 s15, s48, 0
v_add_co_u32 v187, vcc, v2, s15
v_cmp_eq_u32_e64 vcc, v2, -1
v_cndmask_b32_e64 v102, v187, -1, vcc
v_add_co_u32 v187, vcc, v3, s15
v_cmp_eq_u32_e64 vcc, v3, -1
v_cndmask_b32_e64 v103, v187, -1, vcc
v_add_co_u32 v187, vcc, v68, s15
v_cmp_eq_u32_e64 vcc, v68, -1
v_cndmask_b32_e64 v152, v187, -1, vcc
v_add_co_u32 v187, vcc, v69, s15
v_cmp_eq_u32_e64 vcc, v69, -1
v_cndmask_b32_e64 v153, v187, -1, vcc
s_bitcmp1_b32 s14, 18
s_cbranch_scc1 167
s_lshr_b32 s15, -1, 16
s_and_b32 s15, s15, s44
s_lshr_b32 s22, s44, 16
s_mul_i32 s22, s22, s63
s_mul_i32 s40, s15, s63
s_lshl_b32 s15, s22, 16
s_lshr_b32 s22, s22, 16
s_add_u32 s40, s15, s40
s_addc_u32 s41, s22, 0
s_lshl_b64 s[40:41], s[40:41], 1
s_add_u32 s40, s40, s16
s_addc_u32 s41, s41, s17
s_add_u32 s41, s41, 0x20000
s_branch 131
s_bitcmp1_b32 s14, 18
s_cbranch_scc1 150
v_mad_u32_u24 v183, 5, v1, 2
v_lshlrev_b32_e32 v181, 1, v1
v_bfi_b32 v183, 4, v183, v181
v_bfe_u32 v181, v183, 2, 2
v_min_u32_e32 v181, 2, v181
v_bfe_u32 v183, v1, 1, 1
v_mad_u32_u24 v181, 2, v181, v183
v_mad_u32_u24 v181, s35, 3, v181
v_sub_co_u32 v183, vcc, s25, v181
v_sub_co_u32 v183, vcc, v183, 1
s_bfe_u32 s2, s14, 0x10001
v_cmp_eq_u32_e64 vcc, s2, 1
v_cndmask_b32_e32 v181, v181, v183, vcc
v_cmp_ge_u32_e64 s[2:3], v181, s25
s_bfe_u32 s15, s14, 0x10018
v_bfe_u32 v184, v1, 2, s15
v_mul_lo_u32 v184, s48, v184
v_add_co_u32 v181, vcc, v181, v184
v_mul_lo_u32 v182, s72, v175
v_add_co_u32 v182, vcc, v182, v181
s_sub_u32 s15, s24, s38
s_sub_u32 s15, s15, 5
s_bitcmp1_b32 s14, 0
s_cselect_b32 s15, s15, s38
v_mov_b32_e32 v184, s15
v_cmp_ge_u32_e64 s[22:23], v184, s24
v_mad_i32_i24 v2, v184, s73, v182
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v2, v2, -1, s[22:23]
v_mov_b32_e32 v3, v2
v_add_co_u32 v184, vcc, v184, 2
v_cmp_ge_u32_e64 s[22:23], v184, s24
v_mad_i32_i24 v69, v184, s73, v182
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v69, v69, -1, s[22:23]
v_add_co_u32 v184, vcc, v184, 2
v_cmp_ge_u32_e64 s[22:23], v184, s24
v_mad_i32_i24 v68, v184, s73, v182
s_or_b64 s[22:23], s[22:23], s[2:3]
v_cndmask_b32_e64 v68, v68, -1, s[22:23]
s_lshl_b32 s22, s72, 3
s_and_b32 s23, s14, 0x1100000
s_cselect_b32 s22, s22, 0
v_add_co_u32 v181, vcc, v2, s22
v_cmp_eq_u32_e64 vcc, v2, -1
v_cndmask_b32_e64 v102, v181, -1, vcc
v_add_co_u32 v181, vcc, v3, s22
v_cmp_eq_u32_e64 vcc, v3, -1
v_cndmask_b32_e64 v103, v181, -1, vcc
v_add_co_u32 v181, vcc, v68, s22
v_cmp_eq_u32_e64 vcc, v68, -1
v_cndmask_b32_e64 v152, v181, -1, vcc
v_add_co_u32 v181, vcc, v69, s22
v_cmp_eq_u32_e64 vcc, v69, -1
v_cndmask_b32_e64 v153, v181, -1, vcc
v_add_co_u32 v181, vcc, v175, s65
v_cmp_lt_u32_e64 vcc, v181, s12
v_cndmask_b32_e32 v2, -1, v2, vcc
v_cndmask_b32_e32 v3, -1, v3, vcc
v_cndmask_b32_e32 v68, -1, v68, vcc
v_cndmask_b32_e32 v69, -1, v69, vcc
s_and_b32 s2, s14, 0x1100000
s_cbranch_scc0 4
v_add_co_u32 v181, vcc, v181, 8
v_cmp_lt_u32_e64 vcc, v181, s12
v_cndmask_b32_e32 v102, -1, v102, vcc
v_cndmask_b32_e32 v103, -1, v103, vcc
v_cndmask_b32_e32 v152, -1, v152, vcc
v_cndmask_b32_e32 v153, -1, v153, vcc
s_lshr_b32 s15, -1, 16
s_and_b32 s15, s15, s72
s_lshr_b32 s22, s72, 16
s_mul_i32 s22, s22, s65
s_mul_i32 s40, s15, s65
s_lshl_b32 s15, s22, 16
s_lshr_b32 s22, s22, 16
s_add_u32 s40, s15, s40
s_addc_u32 s41, s22, 0
s_lshl_b64 s[40:41], s[40:41], 1
s_add_u32 s40, s40, s18
s_addc_u32 s41, s41, s19
s_add_u32 s41, s41, 0x20000
s_mov_b32 s43, 0x11014000
s_mov_b32 s55, -1
s_bitcmp0_b32 s9, 0
s_cbranch_scc1 5
s_mov_b32 s43, 0
s_bitcmp1_b32 s14, 20
s_addc_u32 s55, 0, 1
s_sub_u32 s40, s40, s47
s_subb_u32 s41, s41, 0
s_add_u32 s93, s9, 1
s_and_b32 s93, s93, -2
s_bfe_u32 s92, s14, 0x10014
s_lshl_b32 s64, s93, s92
s_bitcmp1_b32 s14, 20
s_cselect_b32 s92, 0, 0x2000000
s_bitcmp1_b32 s93, 1
s_cselect_b32 s92, s92, 0
s_xor_b32 s14, s14, s92
s_mov_b64 vcc, s[6:7]
s_branch 64798
s_nop 0
s_nop 0
s_and_b32 s92, 0x900000, s14
s_subb_u32 s35, s35, 1
s_cbranch_scc0 65116
s_and_b32 s92, 0x900000, s14
s_subb_u32 s35, s5, 1
s_add_u32 s38, s38, 6
s_cmp_ge_u32 s38, s24
s_cbranch_scc0 65110
s_mov_b32 s38, 1
s_cmp_ge_u32 s38, s24
s_addc_u32 s39, s39, 1
s_cmp_gt_u32 s39, 1
s_cbranch_scc0 65105
s_mov_b32 s39, 0
s_mov_b32 s38, 0
s_branch 65066
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_dpp v6, v6, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v7, v7, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v4, v4, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v5, v5, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v5, v6, v5 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v4, v7, v4 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v5, v5, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v4, v4, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v4, v5, v4 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v4, v4
v_fmac_f32_dpp v10, v10, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v11, v11, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v8, v8, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v9, v9, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v9, v10, v9 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v8, v11, v8 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v9, v9, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v8, v8, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v5, v9, v8 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v5, v5
v_fmac_f32_dpp v14, v14, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v15, v15, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v12, v12, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v13, v13, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v13, v14, v13 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v12, v15, v12 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v13, v13, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v12, v12, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v6, v13, v12 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v6, v6
v_fmac_f32_dpp v18, v18, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v19, v19, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v16, v16, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v17, v17, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v17, v18, v17 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v16, v19, v16 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v17, v17, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v16, v16, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v7, v17, v16 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v7, v7
v_fmac_f32_dpp v22, v22, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v23, v23, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v20, v20, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v21, v21, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v21, v22, v21 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v20, v23, v20 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v21, v21, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v20, v20, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v8, v21, v20 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v8, v8
v_fmac_f32_dpp v26, v26, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v27, v27, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v24, v24, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v25, v25, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v25, v26, v25 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v24, v27, v24 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v25, v25, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v24, v24, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v9, v25, v24 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v9, v9
v_fmac_f32_dpp v30, v30, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v31, v31, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v28, v28, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v29, v29, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v29, v30, v29 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v28, v31, v28 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v29, v29, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v28, v28, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v10, v29, v28 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v10, v10
s_setprio 1
v_fmac_f32_dpp v34, v34, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v35, v35, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v32, v32, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v33, v33, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v33, v34, v33 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v32, v35, v32 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v33, v33, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v32, v32, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v11, v33, v32 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v11, v11
v_fmac_f32_dpp v38, v38, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v39, v39, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v36, v36, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v37, v37, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v37, v38, v37 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v36, v39, v36 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v37, v37, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v36, v36, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v12, v37, v36 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v12, v12
v_fmac_f32_dpp v42, v42, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v43, v43, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v40, v40, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v41, v41, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v41, v42, v41 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v40, v43, v40 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v41, v41, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v40, v40, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v13, v41, v40 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v13, v13
v_fmac_f32_dpp v46, v46, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v47, v47, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v44, v44, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v45, v45, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v45, v46, v45 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v44, v47, v44 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v45, v45, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v44, v44, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v14, v45, v44 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v14, v14
v_fmac_f32_dpp v50, v50, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v51, v51, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v48, v48, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v49, v49, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v49, v50, v49 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v48, v51, v48 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v49, v49, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v48, v48, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v15, v49, v48 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v15, v15
v_fmac_f32_dpp v54, v54, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v55, v55, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v52, v52, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v53, v53, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v53, v54, v53 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v52, v55, v52 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v53, v53, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v52, v52, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v16, v53, v52 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v16, v16
v_fmac_f32_dpp v58, v58, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v59, v59, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v56, v56, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v57, v57, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v57, v58, v57 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v56, v59, v56 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v57, v57, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v56, v56, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v17, v57, v56 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v17, v17
v_fmac_f32_dpp v62, v62, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v63, v63, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v60, v60, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v61, v61, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v61, v62, v61 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v60, v63, v60 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v61, v61, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v60, v60, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v18, v61, v60 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v18, v18
v_fmac_f32_dpp v66, v66, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v67, v67, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v64, v64, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v65, v65, v177 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v65, v66, v65 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v64, v67, v64 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_fmac_f32_dpp v65, v65, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_fmac_f32_dpp v64, v64, v178 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v19, v65, v64 row_half_mirror row_mask:0xf bank_mask:0xf
v_cvt_f16_f32_e32 v19, v19
s_waitcnt vmcnt(0)
s_setpc_b64 s[80:81]
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
s_xor_b32 s14, s14, 0x200000
s_bitcmp1_b32 s9, 0
s_addc_u32 s92, s9, 0
s_bitcmp0_b32 s14, 21
s_addc_u32 s76, s1, 0
s_lshr_b32 s76, s76, 1
s_mul_i32 s76, s76, s5
s_lshr_b32 s76, s76, 1
s_mul_i32 s76, s76, s92
s_cmp_eq_u32 s76, 0
s_cbranch_scc1 65121
s_add_u32 s92, s75, s74
s_cmp_lt_i32 s92, 0
s_cbranch_scc0 217
v_and_b32_e32 v154, 0x7f, v1
v_lshrrev_b32_e32 v154, 1, v154
v_bfi_b32 v154, 1, v1, v154
v_and_b32_e64 v155, v1, 2
v_mad_u32_u24 v154, v155, 16, v154
v_lshlrev_b32_e32 v154, 2, v154
v_add_co_u32 v154, vcc, v154, s79
v_and_b32_e32 v155, 3, v1
v_lshlrev_b32_e32 v155, 2, v155
v_add_co_u32 v155, vcc, v155, s79
ds_read_b32 v181, v155 offset:256
ds_read_b32 v154, v154
s_add_u32 s79, s79, 0x18c
s_cmp_eq_u32 s79, 0x10000
s_cselect_b32 s79, 0xc220, s79
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s77, v154
v_readlane_b32 s94, v181, 0
s_bitcmp1_b32 s94, 18
s_cbranch_scc1 1111
v_readlane_b32 s92, v181, 1
v_readlane_b32 s93, v181, 2
s_add_u32 s75, s74, s93
s_lshr_b32 s3, -1, 16
s_and_b32 s3, s3, s45
s_lshr_b32 s15, s45, 16
s_mul_i32 s15, s15, s77
s_mul_i32 s84, s3, s77
s_lshl_b32 s3, s15, 16
s_lshr_b32 s15, s15, 16
s_add_u32 s84, s3, s84
s_addc_u32 s85, s15, 0
s_lshl_b64 s[84:85], s[84:85], 1
s_add_u32 s84, s84, s20
s_addc_u32 s85, s85, s21
s_mul_i32 s22, s46, s75
s_lshl_b32 s22, s22, 1
s_add_u32 s84, s84, s22
s_addc_u32 s85, s85, 0
s_add_u32 s85, s85, 0x20000
s_mov_b32 s87, 0x11014000
s_bitcmp1_b32 s14, 7
s_cselect_b32 s91, 0x11014000, 0
s_lshl_b32 s2, s75, 1
s_add_u32 s88, s30, s2
s_addc_u32 s89, s31, 0
s_add_u32 s89, s89, 0x20000
s_sub_u32 s90, s92, s75
s_cselect_b32 s91, 0, s91
s_sub_u32 s75, s92, s93
s_sub_u32 s75, s75, 1
s_sub_u32 s75, s75, s74
s_cselect_b32 s87, 0, s87
v_bfe_u32 v181, v154, 16, 16
v_bfe_u32 v182, v154, 0, 16
v_and_b32_e64 v183, v1, 7
v_sub_co_u32 v184, vcc, 7, v183
v_min_u32_e32 v183, v183, v184
v_bfe_u32 v184, v183, 1, 1
v_bfe_u32 v183, v183, 0, 1
v_mov_b32_dpp v181, v181 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v182, v182 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
v_add_co_u32 v181, vcc, v181, v184
v_add_co_u32 v182, vcc, v182, v183
v_mov_b32_dpp v183, v154 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[2:3], v183, s8
v_sub_co_u32 v183, vcc, v183, s77
v_mul_lo_u32 v183, v183, s45
v_lshlrev_b32_e32 v182, 1, v182
s_and_b32 s22, 1, s27
v_add_co_u32 v182, vcc, s22, v182
v_lshlrev_b32_e32 v181, 1, v181
s_and_b32 s22, 1, s26
v_subrev_co_u32 v181, vcc, s22, v181
v_mad_i32_i24 v158, v181, s50, v182
v_add_co_u32 v158, vcc, v158, v183
v_subrev_co_u32 v159, vcc, 1, v158
v_add_co_u32 v160, vcc, s50, v158
v_add_co_u32 v161, vcc, s50, v159
v_cmp_ge_u32_e64 s[98:99], v182, s29
s_or_b64 s[96:97], s[98:99], s[2:3]
v_subrev_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[98:99], v182, s29
s_or_b64 s[98:99], s[98:99], s[2:3]
v_cmp_ge_u32_e64 s[22:23], v181, s28
s_or_b64 s[2:3], s[96:97], s[22:23]
v_cndmask_b32_e64 v158, v158, -1, s[2:3]
s_or_b64 s[2:3], s[98:99], s[22:23]
v_cndmask_b32_e64 v159, v159, -1, s[2:3]
v_add_co_u32 v181, vcc, 1, v181
v_cmp_ge_u32_e64 s[22:23], v181, s28
s_or_b64 s[2:3], s[96:97], s[22:23]
v_cndmask_b32_e64 v160, v160, -1, s[2:3]
s_or_b64 s[2:3], s[98:99], s[22:23]
v_cndmask_b32_e64 v161, v161, -1, s[2:3]
v_bfe_u32 v181, v154, 16, 16
v_bfe_u32 v182, v154, 0, 16
v_and_b32_e64 v183, v1, 7
v_sub_co_u32 v184, vcc, 7, v183
v_min_u32_e32 v183, v183, v184
v_bfe_u32 v184, v183, 1, 1
v_bfe_u32 v183, v183, 0, 1
v_mov_b32_dpp v181, v181 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v182, v182 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_add_co_u32 v181, vcc, v181, v184
v_add_co_u32 v182, vcc, v182, v183
v_mov_b32_dpp v183, v154 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[2:3], v183, s8
v_sub_co_u32 v183, vcc, v183, s77
v_mul_lo_u32 v183, v183, s45
v_lshlrev_b32_e32 v182, 1, v182
s_and_b32 s22, 1, s27
v_add_co_u32 v182, vcc, s22, v182
v_lshlrev_b32_e32 v181, 1, v181
s_and_b32 s22, 1, s26
v_subrev_co_u32 v181, vcc, s22, v181
v_mad_i32_i24 v154, v181, s50, v182
v_add_co_u32 v154, vcc, v154, v183
v_subrev_co_u32 v155, vcc, 1, v154
v_add_co_u32 v156, vcc, s50, v154
v_add_co_u32 v157, vcc, s50, v155
v_cmp_ge_u32_e64 s[98:99], v182, s29
s_or_b64 s[96:97], s[98:99], s[2:3]
v_subrev_co_u32 v182, vcc, 1, v182
v_cmp_ge_u32_e64 s[98:99], v182, s29
s_or_b64 s[98:99], s[98:99], s[2:3]
v_cmp_ge_u32_e64 s[22:23], v181, s28
s_or_b64 s[2:3], s[96:97], s[22:23]
v_cndmask_b32_e64 v154, v154, -1, s[2:3]
s_or_b64 s[2:3], s[98:99], s[22:23]
v_cndmask_b32_e64 v155, v155, -1, s[2:3]
v_add_co_u32 v181, vcc, 1, v181
v_cmp_ge_u32_e64 s[22:23], v181, s28
s_or_b64 s[2:3], s[96:97], s[22:23]
v_cndmask_b32_e64 v156, v156, -1, s[2:3]
s_or_b64 s[2:3], s[98:99], s[22:23]
v_cndmask_b32_e64 v157, v157, -1, s[2:3]
v_and_b32_e64 v180, v1, 63
buffer_load_ushort v180, v180, s[88:91], 0 idxen
s_mov_b64 vcc, s[6:7]
s_branch 64143
s_mov_b64 s[98:99], s[84:85]
s_mov_b32 s97, s87
v_bfe_u32 v181, s14, 21, 1
v_sub_co_u32 v181, vcc, v181, 1
v_cndmask_b32_e32 v182, v156, v154, vcc
v_cndmask_b32_e32 v183, v157, v155, vcc
v_cndmask_b32_e32 v184, v160, v158, vcc
v_cndmask_b32_e32 v185, v161, v159, vcc
v_readlane_b32 s96, v180, 0
v_add_f16_e64 v4, v4, s96
v_add_f16_e64 v5, v5, s96
buffer_store_short v4, v182, s[84:87], 0 idxen
buffer_store_short v5, v184, s[84:87], 0 idxen
v_add_f16_e64 v6, v6, s96
v_add_f16_e64 v7, v7, s96
buffer_store_short v6, v183, s[84:87], 0 idxen
buffer_store_short v7, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 1
v_add_f16_e64 v8, v8, s96
v_add_f16_e64 v9, v9, s96
buffer_store_short v8, v182, s[84:87], 0 idxen
buffer_store_short v9, v184, s[84:87], 0 idxen
v_add_f16_e64 v10, v10, s96
v_add_f16_e64 v11, v11, s96
buffer_store_short v10, v183, s[84:87], 0 idxen
buffer_store_short v11, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s96, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 2
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 4
v_add_f16_e64 v12, v12, s96
v_add_f16_e64 v13, v13, s96
buffer_store_short v12, v182, s[84:87], 0 idxen
buffer_store_short v13, v184, s[84:87], 0 idxen
v_add_f16_e64 v14, v14, s96
v_add_f16_e64 v15, v15, s96
buffer_store_short v14, v183, s[84:87], 0 idxen
buffer_store_short v15, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 5
v_add_f16_e64 v16, v16, s96
v_add_f16_e64 v17, v17, s96
buffer_store_short v16, v182, s[84:87], 0 idxen
buffer_store_short v17, v184, s[84:87], 0 idxen
v_add_f16_e64 v18, v18, s96
v_add_f16_e64 v19, v19, s96
buffer_store_short v18, v183, s[84:87], 0 idxen
buffer_store_short v19, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 10
s_cselect_b32 s87, 0, s87
s_bitcmp1_b32 s14, 21
s_cselect_b32 s87, s87, s97
s_cselect_b32 s84, s84, s98
s_cselect_b32 s85, s85, s99
s_cselect_b32 s97, 0, 16
s_cselect_b32 s98, 16, 0
s_lshl_b32 s99, s98, 1
s_add_u32 s75, s75, s97
s_add_u32 s88, s88, s99
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, s98
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
s_mov_b64 s[98:99], s[84:85]
s_mov_b32 s97, s87
v_bfe_u32 v181, s14, 21, 1
v_sub_co_u32 v181, vcc, v181, 1
v_cndmask_b32_e32 v182, v156, v154, vcc
v_cndmask_b32_e32 v183, v157, v155, vcc
v_cndmask_b32_e32 v184, v160, v158, vcc
v_cndmask_b32_e32 v185, v161, v159, vcc
v_readlane_b32 s96, v180, 0
v_add_f16_e64 v4, v4, s96
v_add_f16_e64 v5, v5, s96
v_mul_f16_e64 v186, v4, s32
v_mul_f16_e64 v188, v5, s32
v_cmp_lt_f16_e64 vcc, v4, 0
v_cmp_lt_f16_e64 s[2:3], v5, 0
v_cndmask_b32_e32 v4, v4, v186, vcc
v_cndmask_b32_e64 v5, v5, v188, s[2:3]
buffer_store_short v4, v182, s[84:87], 0 idxen
buffer_store_short v5, v184, s[84:87], 0 idxen
v_add_f16_e64 v6, v6, s96
v_add_f16_e64 v7, v7, s96
v_mul_f16_e64 v186, v6, s32
v_mul_f16_e64 v188, v7, s32
v_cmp_lt_f16_e64 vcc, v6, 0
v_cmp_lt_f16_e64 s[2:3], v7, 0
v_cndmask_b32_e32 v6, v6, v186, vcc
v_cndmask_b32_e64 v7, v7, v188, s[2:3]
buffer_store_short v6, v183, s[84:87], 0 idxen
buffer_store_short v7, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 1
v_add_f16_e64 v8, v8, s96
v_add_f16_e64 v9, v9, s96
v_mul_f16_e64 v186, v8, s32
v_mul_f16_e64 v188, v9, s32
v_cmp_lt_f16_e64 vcc, v8, 0
v_cmp_lt_f16_e64 s[2:3], v9, 0
v_cndmask_b32_e32 v8, v8, v186, vcc
v_cndmask_b32_e64 v9, v9, v188, s[2:3]
buffer_store_short v8, v182, s[84:87], 0 idxen
buffer_store_short v9, v184, s[84:87], 0 idxen
v_add_f16_e64 v10, v10, s96
v_add_f16_e64 v11, v11, s96
v_mul_f16_e64 v186, v10, s32
v_mul_f16_e64 v188, v11, s32
v_cmp_lt_f16_e64 vcc, v10, 0
v_cmp_lt_f16_e64 s[2:3], v11, 0
v_cndmask_b32_e32 v10, v10, v186, vcc
v_cndmask_b32_e64 v11, v11, v188, s[2:3]
buffer_store_short v10, v183, s[84:87], 0 idxen
buffer_store_short v11, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s96, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 2
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 4
v_add_f16_e64 v12, v12, s96
v_add_f16_e64 v13, v13, s96
v_mul_f16_e64 v186, v12, s32
v_mul_f16_e64 v188, v13, s32
v_cmp_lt_f16_e64 vcc, v12, 0
v_cmp_lt_f16_e64 s[2:3], v13, 0
v_cndmask_b32_e32 v12, v12, v186, vcc
v_cndmask_b32_e64 v13, v13, v188, s[2:3]
buffer_store_short v12, v182, s[84:87], 0 idxen
buffer_store_short v13, v184, s[84:87], 0 idxen
v_add_f16_e64 v14, v14, s96
v_add_f16_e64 v15, v15, s96
v_mul_f16_e64 v186, v14, s32
v_mul_f16_e64 v188, v15, s32
v_cmp_lt_f16_e64 vcc, v14, 0
v_cmp_lt_f16_e64 s[2:3], v15, 0
v_cndmask_b32_e32 v14, v14, v186, vcc
v_cndmask_b32_e64 v15, v15, v188, s[2:3]
buffer_store_short v14, v183, s[84:87], 0 idxen
buffer_store_short v15, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 5
v_add_f16_e64 v16, v16, s96
v_add_f16_e64 v17, v17, s96
v_mul_f16_e64 v186, v16, s32
v_mul_f16_e64 v188, v17, s32
v_cmp_lt_f16_e64 vcc, v16, 0
v_cmp_lt_f16_e64 s[2:3], v17, 0
v_cndmask_b32_e32 v16, v16, v186, vcc
v_cndmask_b32_e64 v17, v17, v188, s[2:3]
buffer_store_short v16, v182, s[84:87], 0 idxen
buffer_store_short v17, v184, s[84:87], 0 idxen
v_add_f16_e64 v18, v18, s96
v_add_f16_e64 v19, v19, s96
v_mul_f16_e64 v186, v18, s32
v_mul_f16_e64 v188, v19, s32
v_cmp_lt_f16_e64 vcc, v18, 0
v_cmp_lt_f16_e64 s[2:3], v19, 0
v_cndmask_b32_e32 v18, v18, v186, vcc
v_cndmask_b32_e64 v19, v19, v188, s[2:3]
buffer_store_short v18, v183, s[84:87], 0 idxen
buffer_store_short v19, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 10
s_cselect_b32 s87, 0, s87
s_bitcmp1_b32 s14, 21
s_cselect_b32 s87, s87, s97
s_cselect_b32 s84, s84, s98
s_cselect_b32 s85, s85, s99
s_cselect_b32 s97, 0, 16
s_cselect_b32 s98, 16, 0
s_lshl_b32 s99, s98, 1
s_add_u32 s75, s75, s97
s_add_u32 s88, s88, s99
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, s98
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
s_mov_b64 s[98:99], s[84:85]
s_mov_b32 s97, s87
v_bfe_u32 v181, s14, 21, 1
v_sub_co_u32 v181, vcc, v181, 1
v_cndmask_b32_e32 v182, v156, v154, vcc
v_cndmask_b32_e32 v183, v157, v155, vcc
v_cndmask_b32_e32 v184, v160, v158, vcc
v_cndmask_b32_e32 v185, v161, v159, vcc
v_readlane_b32 s96, v180, 0
v_add_f16_e64 v4, v4, s96
v_add_f16_e64 v5, v5, s96
v_mul_f16_e32 v4, 0xbdc5, v4
v_mul_f16_e32 v5, 0xbdc5, v5
v_exp_f16_e32 v4, v4
v_exp_f16_e32 v5, v5
_v_add_f16_e32_1_gfx1x 4, 4
_v_add_f16_e32_1_gfx1x 5, 5
v_rcp_f16_e32 v4, v4
v_rcp_f16_e32 v5, v5
buffer_store_short v4, v182, s[84:87], 0 idxen
buffer_store_short v5, v184, s[84:87], 0 idxen
v_add_f16_e64 v6, v6, s96
v_add_f16_e64 v7, v7, s96
v_mul_f16_e32 v6, 0xbdc5, v6
v_mul_f16_e32 v7, 0xbdc5, v7
v_exp_f16_e32 v6, v6
v_exp_f16_e32 v7, v7
_v_add_f16_e32_1_gfx1x 6, 6
_v_add_f16_e32_1_gfx1x 7, 7
v_rcp_f16_e32 v6, v6
v_rcp_f16_e32 v7, v7
buffer_store_short v6, v183, s[84:87], 0 idxen
buffer_store_short v7, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 1
v_add_f16_e64 v8, v8, s96
v_add_f16_e64 v9, v9, s96
v_mul_f16_e32 v8, 0xbdc5, v8
v_mul_f16_e32 v9, 0xbdc5, v9
v_exp_f16_e32 v8, v8
v_exp_f16_e32 v9, v9
_v_add_f16_e32_1_gfx1x 8, 8
_v_add_f16_e32_1_gfx1x 9, 9
v_rcp_f16_e32 v8, v8
v_rcp_f16_e32 v9, v9
buffer_store_short v8, v182, s[84:87], 0 idxen
buffer_store_short v9, v184, s[84:87], 0 idxen
v_add_f16_e64 v10, v10, s96
v_add_f16_e64 v11, v11, s96
v_mul_f16_e32 v10, 0xbdc5, v10
v_mul_f16_e32 v11, 0xbdc5, v11
v_exp_f16_e32 v10, v10
v_exp_f16_e32 v11, v11
_v_add_f16_e32_1_gfx1x 10, 10
_v_add_f16_e32_1_gfx1x 11, 11
v_rcp_f16_e32 v10, v10
v_rcp_f16_e32 v11, v11
buffer_store_short v10, v183, s[84:87], 0 idxen
buffer_store_short v11, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s96, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 2
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 4
v_add_f16_e64 v12, v12, s96
v_add_f16_e64 v13, v13, s96
v_mul_f16_e32 v12, 0xbdc5, v12
v_mul_f16_e32 v13, 0xbdc5, v13
v_exp_f16_e32 v12, v12
v_exp_f16_e32 v13, v13
_v_add_f16_e32_1_gfx1x 12, 12
_v_add_f16_e32_1_gfx1x 13, 13
v_rcp_f16_e32 v12, v12
v_rcp_f16_e32 v13, v13
buffer_store_short v12, v182, s[84:87], 0 idxen
buffer_store_short v13, v184, s[84:87], 0 idxen
v_add_f16_e64 v14, v14, s96
v_add_f16_e64 v15, v15, s96
v_mul_f16_e32 v14, 0xbdc5, v14
v_mul_f16_e32 v15, 0xbdc5, v15
v_exp_f16_e32 v14, v14
v_exp_f16_e32 v15, v15
_v_add_f16_e32_1_gfx1x 14, 14
_v_add_f16_e32_1_gfx1x 15, 15
v_rcp_f16_e32 v14, v14
v_rcp_f16_e32 v15, v15
buffer_store_short v14, v183, s[84:87], 0 idxen
buffer_store_short v15, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 5
v_add_f16_e64 v16, v16, s96
v_add_f16_e64 v17, v17, s96
v_mul_f16_e32 v16, 0xbdc5, v16
v_mul_f16_e32 v17, 0xbdc5, v17
v_exp_f16_e32 v16, v16
v_exp_f16_e32 v17, v17
_v_add_f16_e32_1_gfx1x 16, 16
_v_add_f16_e32_1_gfx1x 17, 17
v_rcp_f16_e32 v16, v16
v_rcp_f16_e32 v17, v17
buffer_store_short v16, v182, s[84:87], 0 idxen
buffer_store_short v17, v184, s[84:87], 0 idxen
v_add_f16_e64 v18, v18, s96
v_add_f16_e64 v19, v19, s96
v_mul_f16_e32 v18, 0xbdc5, v18
v_mul_f16_e32 v19, 0xbdc5, v19
v_exp_f16_e32 v18, v18
v_exp_f16_e32 v19, v19
_v_add_f16_e32_1_gfx1x 18, 18
_v_add_f16_e32_1_gfx1x 19, 19
v_rcp_f16_e32 v18, v18
v_rcp_f16_e32 v19, v19
buffer_store_short v18, v183, s[84:87], 0 idxen
buffer_store_short v19, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 10
s_cselect_b32 s87, 0, s87
s_bitcmp1_b32 s14, 21
s_cselect_b32 s87, s87, s97
s_cselect_b32 s84, s84, s98
s_cselect_b32 s85, s85, s99
s_cselect_b32 s97, 0, 16
s_cselect_b32 s98, 16, 0
s_lshl_b32 s99, s98, 1
s_add_u32 s75, s75, s97
s_add_u32 s88, s88, s99
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, s98
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
s_mov_b64 s[98:99], s[84:85]
s_mov_b32 s97, s87
v_bfe_u32 v181, s14, 21, 1
v_sub_co_u32 v181, vcc, v181, 1
v_cndmask_b32_e32 v182, v156, v154, vcc
v_cndmask_b32_e32 v183, v157, v155, vcc
v_cndmask_b32_e32 v184, v160, v158, vcc
v_cndmask_b32_e32 v185, v161, v159, vcc
v_readlane_b32 s96, v180, 0
v_add_f16_e64 v4, v4, s96
v_add_f16_e64 v5, v5, s96
v_mul_f16_e64 v4, v4, s33
v_mul_f16_e64 v5, v5, s33
v_mul_f16_e32 v186, 0x3dc5, v4
v_mul_f16_e32 v188, 0x3dc5, v5
v_mul_f16_e32 v187, 0xbdc5, v4
v_mul_f16_e32 v189, 0xbdc5, v5
v_exp_f16_e32 v186, v186
v_exp_f16_e32 v188, v188
v_exp_f16_e32 v187, v187
v_exp_f16_e32 v189, v189
v_add_f16_e32 v4, v186, v187
v_add_f16_e32 v5, v188, v189
v_sub_f16_e32 v186, v186, v187
v_sub_f16_e32 v188, v188, v189
v_rcp_f16_e32 v4, v4
v_rcp_f16_e32 v5, v5
v_mul_f16_e32 v4, v4, v186
v_mul_f16_e32 v5, v5, v188
v_mul_f16_e64 v4, v4, s32
v_mul_f16_e64 v5, v5, s32
buffer_store_short v4, v182, s[84:87], 0 idxen
buffer_store_short v5, v184, s[84:87], 0 idxen
v_add_f16_e64 v6, v6, s96
v_add_f16_e64 v7, v7, s96
v_mul_f16_e64 v6, v6, s33
v_mul_f16_e64 v7, v7, s33
v_mul_f16_e32 v186, 0x3dc5, v6
v_mul_f16_e32 v188, 0x3dc5, v7
v_mul_f16_e32 v187, 0xbdc5, v6
v_mul_f16_e32 v189, 0xbdc5, v7
v_exp_f16_e32 v186, v186
v_exp_f16_e32 v188, v188
v_exp_f16_e32 v187, v187
v_exp_f16_e32 v189, v189
v_add_f16_e32 v6, v186, v187
v_add_f16_e32 v7, v188, v189
v_sub_f16_e32 v186, v186, v187
v_sub_f16_e32 v188, v188, v189
v_rcp_f16_e32 v6, v6
v_rcp_f16_e32 v7, v7
v_mul_f16_e32 v6, v6, v186
v_mul_f16_e32 v7, v7, v188
v_mul_f16_e64 v6, v6, s32
v_mul_f16_e64 v7, v7, s32
buffer_store_short v6, v183, s[84:87], 0 idxen
buffer_store_short v7, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 1
v_add_f16_e64 v8, v8, s96
v_add_f16_e64 v9, v9, s96
v_mul_f16_e64 v8, v8, s33
v_mul_f16_e64 v9, v9, s33
v_mul_f16_e32 v186, 0x3dc5, v8
v_mul_f16_e32 v188, 0x3dc5, v9
v_mul_f16_e32 v187, 0xbdc5, v8
v_mul_f16_e32 v189, 0xbdc5, v9
v_exp_f16_e32 v186, v186
v_exp_f16_e32 v188, v188
v_exp_f16_e32 v187, v187
v_exp_f16_e32 v189, v189
v_add_f16_e32 v8, v186, v187
v_add_f16_e32 v9, v188, v189
v_sub_f16_e32 v186, v186, v187
v_sub_f16_e32 v188, v188, v189
v_rcp_f16_e32 v8, v8
v_rcp_f16_e32 v9, v9
v_mul_f16_e32 v8, v8, v186
v_mul_f16_e32 v9, v9, v188
v_mul_f16_e64 v8, v8, s32
v_mul_f16_e64 v9, v9, s32
buffer_store_short v8, v182, s[84:87], 0 idxen
buffer_store_short v9, v184, s[84:87], 0 idxen
v_add_f16_e64 v10, v10, s96
v_add_f16_e64 v11, v11, s96
v_mul_f16_e64 v10, v10, s33
v_mul_f16_e64 v11, v11, s33
v_mul_f16_e32 v186, 0x3dc5, v10
v_mul_f16_e32 v188, 0x3dc5, v11
v_mul_f16_e32 v187, 0xbdc5, v10
v_mul_f16_e32 v189, 0xbdc5, v11
v_exp_f16_e32 v186, v186
v_exp_f16_e32 v188, v188
v_exp_f16_e32 v187, v187
v_exp_f16_e32 v189, v189
v_add_f16_e32 v10, v186, v187
v_add_f16_e32 v11, v188, v189
v_sub_f16_e32 v186, v186, v187
v_sub_f16_e32 v188, v188, v189
v_rcp_f16_e32 v10, v10
v_rcp_f16_e32 v11, v11
v_mul_f16_e32 v10, v10, v186
v_mul_f16_e32 v11, v11, v188
v_mul_f16_e64 v10, v10, s32
v_mul_f16_e64 v11, v11, s32
buffer_store_short v10, v183, s[84:87], 0 idxen
buffer_store_short v11, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s96, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 2
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 4
v_add_f16_e64 v12, v12, s96
v_add_f16_e64 v13, v13, s96
v_mul_f16_e64 v12, v12, s33
v_mul_f16_e64 v13, v13, s33
v_mul_f16_e32 v186, 0x3dc5, v12
v_mul_f16_e32 v188, 0x3dc5, v13
v_mul_f16_e32 v187, 0xbdc5, v12
v_mul_f16_e32 v189, 0xbdc5, v13
v_exp_f16_e32 v186, v186
v_exp_f16_e32 v188, v188
v_exp_f16_e32 v187, v187
v_exp_f16_e32 v189, v189
v_add_f16_e32 v12, v186, v187
v_add_f16_e32 v13, v188, v189
v_sub_f16_e32 v186, v186, v187
v_sub_f16_e32 v188, v188, v189
v_rcp_f16_e32 v12, v12
v_rcp_f16_e32 v13, v13
v_mul_f16_e32 v12, v12, v186
v_mul_f16_e32 v13, v13, v188
v_mul_f16_e64 v12, v12, s32
v_mul_f16_e64 v13, v13, s32
buffer_store_short v12, v182, s[84:87], 0 idxen
buffer_store_short v13, v184, s[84:87], 0 idxen
v_add_f16_e64 v14, v14, s96
v_add_f16_e64 v15, v15, s96
v_mul_f16_e64 v14, v14, s33
v_mul_f16_e64 v15, v15, s33
v_mul_f16_e32 v186, 0x3dc5, v14
v_mul_f16_e32 v188, 0x3dc5, v15
v_mul_f16_e32 v187, 0xbdc5, v14
v_mul_f16_e32 v189, 0xbdc5, v15
v_exp_f16_e32 v186, v186
v_exp_f16_e32 v188, v188
v_exp_f16_e32 v187, v187
v_exp_f16_e32 v189, v189
v_add_f16_e32 v14, v186, v187
v_add_f16_e32 v15, v188, v189
v_sub_f16_e32 v186, v186, v187
v_sub_f16_e32 v188, v188, v189
v_rcp_f16_e32 v14, v14
v_rcp_f16_e32 v15, v15
v_mul_f16_e32 v14, v14, v186
v_mul_f16_e32 v15, v15, v188
v_mul_f16_e64 v14, v14, s32
v_mul_f16_e64 v15, v15, s32
buffer_store_short v14, v183, s[84:87], 0 idxen
buffer_store_short v15, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
v_readlane_b32 s96, v180, 5
v_add_f16_e64 v16, v16, s96
v_add_f16_e64 v17, v17, s96
v_mul_f16_e64 v16, v16, s33
v_mul_f16_e64 v17, v17, s33
v_mul_f16_e32 v186, 0x3dc5, v16
v_mul_f16_e32 v188, 0x3dc5, v17
v_mul_f16_e32 v187, 0xbdc5, v16
v_mul_f16_e32 v189, 0xbdc5, v17
v_exp_f16_e32 v186, v186
v_exp_f16_e32 v188, v188
v_exp_f16_e32 v187, v187
v_exp_f16_e32 v189, v189
v_add_f16_e32 v16, v186, v187
v_add_f16_e32 v17, v188, v189
v_sub_f16_e32 v186, v186, v187
v_sub_f16_e32 v188, v188, v189
v_rcp_f16_e32 v16, v16
v_rcp_f16_e32 v17, v17
v_mul_f16_e32 v16, v16, v186
v_mul_f16_e32 v17, v17, v188
v_mul_f16_e64 v16, v16, s32
v_mul_f16_e64 v17, v17, s32
buffer_store_short v16, v182, s[84:87], 0 idxen
buffer_store_short v17, v184, s[84:87], 0 idxen
v_add_f16_e64 v18, v18, s96
v_add_f16_e64 v19, v19, s96
v_mul_f16_e64 v18, v18, s33
v_mul_f16_e64 v19, v19, s33
v_mul_f16_e32 v186, 0x3dc5, v18
v_mul_f16_e32 v188, 0x3dc5, v19
v_mul_f16_e32 v187, 0xbdc5, v18
v_mul_f16_e32 v189, 0xbdc5, v19
v_exp_f16_e32 v186, v186
v_exp_f16_e32 v188, v188
v_exp_f16_e32 v187, v187
v_exp_f16_e32 v189, v189
v_add_f16_e32 v18, v186, v187
v_add_f16_e32 v19, v188, v189
v_sub_f16_e32 v186, v186, v187
v_sub_f16_e32 v188, v188, v189
v_rcp_f16_e32 v18, v18
v_rcp_f16_e32 v19, v19
v_mul_f16_e32 v18, v18, v186
v_mul_f16_e32 v19, v19, v188
v_mul_f16_e64 v18, v18, s32
v_mul_f16_e64 v19, v19, s32
buffer_store_short v18, v183, s[84:87], 0 idxen
buffer_store_short v19, v185, s[84:87], 0 idxen
s_lshl_b32 s96, s46, 1
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 1
s_cselect_b32 s87, 0, s87
s_lshl_b32 s96, s46, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_lshl_b32 s96, s96, 2
s_add_u32 s84, s84, s96
s_addc_u32 s85, s85, 0
s_sub_u32 s75, s75, 10
s_cselect_b32 s87, 0, s87
s_bitcmp1_b32 s14, 21
s_cselect_b32 s87, s87, s97
s_cselect_b32 s84, s84, s98
s_cselect_b32 s85, s85, s99
s_cselect_b32 s97, 0, 16
s_cselect_b32 s98, 16, 0
s_lshl_b32 s99, s98, 1
s_add_u32 s75, s75, s97
s_add_u32 s88, s88, s99
s_addc_u32 s89, s89, 0
s_sub_u32 s90, s90, s98
s_cselect_b32 s91, 0, s91
s_setpc_b64 s[82:83]
s_endpgm
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
s_code_end
