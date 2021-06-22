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
s_version 0x2004
s_inst_prefetch 0x3
v_mov_b32_e32 v1, v0
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_mov_b32_e32 v187, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s97, 0xc1e0
s_mov_b32 s96, 0xc1e0
v_and_b32_e32 v189, 0xc0, v0
v_add_co_u32_e64 v1, vcc, v0, v189
s_mov_b32 s91, 0
v_lshlrev_b32_e32 v190, 2, v1
v_add_co_u32_e64 v190, vcc, 0xffc0, v190
v_cmp_ge_u32_e32 vcc, 12, v1
s_cbranch_vccz 5
v_mov_b32_e32 v189, 0
v_cndmask_b32_e32 v190, -1, v190, vcc
ds_write_b32 v190, v189
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
s_lshl_b32 s65, s44, 2
s_lshl_b32 s68, s42, 2
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
s_lshl_b32 s66, s44, 2
s_lshl_b32 s67, s43, 2
s_bitcmp1_b32 s18, 13
s_cbranch_scc0 2
s_load_dwordx8 s[48:55], s[40:41], 0x68
s_mul_i32 s42, s28, s29
s_lshl_b32 s42, s42, 2
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
s_mov_b64 s[70:71], 0
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
s_and_b32 s48, s13, 1
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
v_bfe_u32 v3, v1, 2, 6
v_lshrrev_b32_e32 v182, 1, v3
s_bitcmp0_b32 s45, 8
s_cselect_b32 s48, 0x1000000, 0
s_or_b32 s48, s48, 0x100000
s_and_b32 s48, s18, s48
s_cselect_b32 s48, 0, 15
v_bfi_b32 v182, s48, v3, v182
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
v_cndmask_b32_e32 v7, 0, v182, vcc
s_bitcmp1_b32 s18, 20
s_cbranch_scc0 4
v_add_co_u32_e64 v5, vcc, v182, 8
v_cmp_eq_u32_e32 vcc, 0, v6
v_cndmask_b32_e32 v7, v7, v5, vcc
v_cmp_eq_u32_e64 s[46:47], 3, v6
v_bfe_u32 v180, v7, 0, 5
v_mad_u32_u24 v180, v4, 32, v180
v_ffbh_u32_e32 v9, s43
v_lshlrev_b32_e64 v10, v9, s43
v_and_b32_e32 v11, 0xffffff00, v10
v_cmp_eq_u32_e32 vcc, 0x80000000, v10
v_cvt_f32_u32_e32 v11, v11
v_rcp_f32_e32 v181, v11
v_sub_co_ci_u32_e32 v8, vcc, 32, v9, vcc
v_cvt_f32_ubyte0_e32 v9, v10
v_fma_f32 v11, v11, v181, -1.0
v_fma_f32 v11, v9, v181, v11
v_fmaak_f32 v11, v11, v181, 0x9f000000
v_mul_f32_e32 v11, 0x5f800000, v11
v_mov_b32_e32 v9, 0
v_cvt_flr_i32_f32_e64 v11, -v11
v_lshl_add_u32 v181, v181, 9, v11
v_mad_u64_u32 v[9:10], vcc, v10, v181, v[9:10]
v_sub_co_ci_u32_e64 v181, vcc, v181, -1, vcc
v_mul_hi_u32 v9, v180, v181
v_add_co_u32_e64 v181, vcc, v9, v180
v_add_co_ci_u32_e64 v9, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v8
v_cndmask_b32_e32 v181, v181, v9, vcc
v_alignbit_b32 v181, v9, v181, v8
v_mad_i32_i24 v179, v181, s75, v180
v_lshrrev_b32_e32 v180, 5, v7
v_mad_u32_u24 v180, v181, 1, v180
v_cndmask_b32_e64 v180, v180, 1, s[46:47]
v_ffbh_u32_e32 v9, s42
v_lshlrev_b32_e64 v10, v9, s42
v_and_b32_e32 v11, 0xffffff00, v10
v_cmp_eq_u32_e32 vcc, 0x80000000, v10
v_cvt_f32_u32_e32 v11, v11
v_rcp_f32_e32 v181, v11
v_sub_co_ci_u32_e32 v8, vcc, 32, v9, vcc
v_cvt_f32_ubyte0_e32 v9, v10
v_fma_f32 v11, v11, v181, -1.0
v_fma_f32 v11, v9, v181, v11
v_fmaak_f32 v11, v11, v181, 0x9f000000
v_mul_f32_e32 v11, 0x5f800000, v11
v_mov_b32_e32 v9, 0
v_cvt_flr_i32_f32_e64 v11, -v11
v_lshl_add_u32 v181, v181, 9, v11
v_mad_u64_u32 v[9:10], vcc, v10, v181, v[9:10]
v_sub_co_ci_u32_e64 v181, vcc, v181, -1, vcc
v_mul_hi_u32 v9, v180, v181
v_add_co_u32_e64 v181, vcc, v9, v180
v_add_co_ci_u32_e64 v9, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v8
v_cndmask_b32_e32 v181, v181, v9, vcc
v_alignbit_b32 v181, v9, v181, v8
v_mad_i32_i24 v180, v181, s74, v180
v_readlane_b32 s76, v179, 2
v_readlane_b32 s77, v180, 2
v_readlane_b32 s78, v181, 2
v_readlane_b32 s79, v180, 3
v_readlane_b32 s80, v181, 3
v_add_co_u32_e64 v179, vcc, v179, s75
v_add_co_u32_e64 v180, vcc, v180, s74
v_mov_b32_dpp v181, v181  quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v179, v179  quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v180, v180  quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x31014000
s_mov_b32 s46, 0x80000000
s_mov_b32 s47, 0x31014000
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 6
v_xor_b32_dpp v183, v1, v1  quad_perm:[2,3,2,1] row_mask:0xf bank_mask:0xf
v_subrev_co_u32_e64 v183, vcc, 1, v183
v_cvt_f32_i32_e32 v183, v183
s_branch 5
v_xor_b32_dpp v183, v1, v1  quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32_e64 v183, vcc, 1, v183
v_cvt_f32_i32_e32 v183, v183
v_mov_b32_e32 v184, 1
v_xor_b32_dpp v184, v1, v1  quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v184, v1, v1  quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32_e64 v184, vcc, 1, v184
v_mov_b32_e32 v185, 1
v_xor_b32_dpp v185, v1, v1  quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v185, v1, v1  quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32_e64 v185, vcc, 1, v185
v_cvt_f32_i32_e32 v184, v184
v_cvt_f32_i32_e32 v185, v185
v_lshrrev_b32_e64 v189, 2, s92
v_and_b32_e32 v190, 3, v1
v_bfe_u32 v191, v1, 4, 3
v_mad_u32_u24 v175, v191, 4, v190
v_lshlrev_b32_e32 v175, 4, v175
v_mad_u32_u24 v165, v189, 4, v190
v_lshlrev_b32_e32 v165, 4, v165
v_bfe_u32 v189, v1, 2, 2
v_and_b32_e32 v190, 1, v189
v_mad_u32_u24 v192, v189, 16, v190
v_lshlrev_b32_e32 v192, 6, v192
v_xor_b32_e32 v165, v165, v192
v_mul_u32_u24_e32 v192, 0x400, v189
v_xor_b32_e32 v175, v175, v192
s_lshr_b32 s92, s92, 0
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 61
s_and_b32 s53, s18, 0x1100000
s_addc_u32 s53, 0, 0
v_lshrrev_b32_e32 v192, 1, v1
s_mul_i32 s52, 60, s53
s_sub_u32 s52, 63, s52
v_bfi_b32 v192, s52, v1, v192
v_and_b32_e32 v189, 1, v192
v_bfe_u32 v190, v192, 1, 1
v_xor_b32_e32 v189, v189, v190
v_bfe_u32 v191, v192, 3, 1
v_mad_u32_u24 v190, v190, 2, v191
v_mul_u32_u24_e32 v189, 0x118, v189
v_bfe_u32 v191, v192, 2, 1
v_mad_u32_u24 v190, v190, 2, v189
v_xor_b32_e32 v190, v190, v191
v_and_b32_e32 v191, 0xf0, v192
v_xor_b32_e32 v190, v190, v191
s_mul_i32 s52, 4, s53
s_sub_u32 s52, 6, s52
v_bfe_u32 v192, v1, s52, 1
v_mul_u32_u24_e32 v192, 0x1040, v192
v_xor_b32_e32 v168, 0x314, v190
v_xor_b32_e32 v169, 0x31c, v190
v_xor_b32_e32 v170, 8, v190
v_mov_b32_e32 v167, v190
v_mad_u32_u24 v167, 4, v167, v192
v_mad_u32_u24 v168, 4, v168, v192
v_mad_u32_u24 v169, 4, v169, v192
v_mad_u32_u24 v170, 4, v170, v192
s_mov_b32 s52, 0x1040
s_and_b32 s53, s18, 0x1100000
s_cselect_b32 s52, 0x80, s52
v_add_co_u32_e64 v171, vcc, v167, s52
v_add_co_u32_e64 v172, vcc, v168, s52
v_add_co_u32_e64 v173, vcc, v169, s52
v_add_co_u32_e64 v174, vcc, v170, s52
s_branch 57
s_bfe_u32 s53, s18, 0x10014
v_lshrrev_b32_e32 v192, 1, v1
s_mul_i32 s52, 60, s53
s_sub_u32 s52, 63, s52
v_bfi_b32 v192, s52, v1, v192
v_and_b32_e32 v189, 1, v192
v_bfe_u32 v190, v192, 1, 1
v_bfe_u32 v191, v192, 3, 1
v_xor_b32_e32 v189, v189, v190
v_mad_u32_u24 v190, v190, 2, v191
v_mul_u32_u24_e32 v189, 0x109, v189
v_bfe_u32 v191, v192, 2, 1
v_mad_u32_u24 v190, v190, 2, v189
v_xor_b32_e32 v190, v190, v191
v_and_b32_e32 v191, 0xf0, v192
v_or_b32_e32 v190, v190, v191
s_mul_i32 s52, 4, s53
s_sub_u32 s52, 6, s52
v_bfe_u32 v192, v1, s52, 1
v_mul_u32_u24_e32 v192, 0x1040, v192
v_mad_u32_u24 v167, 4, v190, v192
v_xor_b32_e32 v168, 0x307, v190
v_mad_u32_u24 v168, 4, v168, v192
v_xor_b32_e32 v169, 0x30f, v190
v_mad_u32_u24 v169, 4, v169, v192
v_xor_b32_e32 v170, 8, v190
v_mad_u32_u24 v170, 4, v170, v192
s_mov_b32 s52, 0x1040
s_bitcmp1_b32 s18, 20
s_cselect_b32 s52, 0x80, s52
v_add_co_u32_e64 v171, vcc, v167, s52
v_add_co_u32_e64 v172, vcc, v168, s52
v_add_co_u32_e64 v173, vcc, v169, s52
v_add_co_u32_e64 v174, vcc, v170, s52
v_subrev_co_u32_e64 v179, vcc, s76, v179
v_mov_b32_e32 v190, s75
v_cmp_lt_i32_e32 vcc, v179, v190
v_sub_co_ci_u32_e64 v189, vcc, 0, 0, vcc
v_mad_i32_i24 v179, v189, s75, v179
v_mad_i32_i24 v181, v189, s80, v181
v_mad_i32_i24 v180, v189, s79, v180
v_mov_b32_e32 v190, s74
v_cmp_lt_i32_e32 vcc, v180, v190
v_sub_co_ci_u32_e64 v189, vcc, 0, 0, vcc
v_add_co_u32_e64 v181, vcc, v181, v189
v_mad_i32_i24 v180, v189, v190, v180
v_subrev_co_u32_e64 v180, vcc, s77, v180
v_cmp_lt_i32_e32 vcc, v180, v190
v_sub_co_ci_u32_e64 v189, vcc, 0, 0, vcc
v_add_co_u32_e64 v181, vcc, v181, v189
v_mad_i32_i24 v180, v189, s74, v180
v_subrev_co_u32_e64 v181, vcc, s78, v181
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
s_mov_b32 s94, 19
s_mov_b32 s82, 0
s_bitset1_b32 s18, 26
s_call_b64 s[38:39], 2771
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccnz 65
s_branch 1508
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v113, v115, v113 div:2
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_subrev_f32_e64 v116, v114, v116 div:2
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v114, v115, v114 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_fma_f32 v115, v115, 1.0, -v114
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_dpp v113, v113, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_dpp v114, v114, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_dpp v115, v115, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_dpp v116, v116, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v105
ds_read_b128 v[69:72], v175 offset:29440
ds_write_b32 v172, v106
ds_read_b128 v[73:76], v175 offset:29696
ds_write_b32 v173, v107
ds_read_b128 v[85:88], v165 offset:28928
ds_write_b32 v174, v108
ds_read_b128 v[89:92], v165 offset:29056
s_setprio 0
s_add_u32 s91, s91, 0x100
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v101, v149, s[40:43], 0 offen
buffer_load_dword v103, v151, s[40:43], 0 offen
buffer_load_dword v102, v150, s[40:43], 0 offen
buffer_load_dword v104, v152, s[40:43], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 2581
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v117, v119, v117 div:2
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_subrev_f32_e64 v120, v118, v120 div:2
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v118, v119, v118 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_fma_f32 v119, v119, 1.0, -v118
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_dpp v117, v117, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_dpp v118, v118, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_dpp v119, v119, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_dpp v120, v120, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
ds_write_b32 v167, v109 offset:8256
ds_read_b128 v[77:80], v175 offset:33536
ds_write_b32 v168, v110 offset:8256
ds_read_b128 v[81:84], v175 offset:33792
ds_write_b32 v169, v111 offset:8256
ds_read_b128 v[93:96], v165 offset:33024
ds_write_b32 v170, v112 offset:8256
ds_read_b128 v[97:100], v165 offset:33152
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v105, v153, s[40:43], 0 offen
buffer_load_dword v107, v155, s[40:43], 0 offen
buffer_load_dword v106, v154, s[40:43], 0 offen
buffer_load_dword v108, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v188 offset:65472
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 2461
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v121, v123, v121 div:2
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_subrev_f32_e64 v124, v122, v124 div:2
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v122, v123, v122 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_fma_f32 v123, v123, 1.0, -v122
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_dpp v121, v121, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_dpp v122, v122, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_dpp v123, v123, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_dpp v124, v124, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v113 offset:8256
ds_read_b128 v[69:72], v175 offset:37696
ds_write_b32 v172, v114 offset:8256
ds_read_b128 v[73:76], v175 offset:37952
ds_write_b32 v173, v115 offset:8256
ds_read_b128 v[85:88], v165 offset:37184
ds_write_b32 v174, v116 offset:8256
ds_read_b128 v[89:92], v165 offset:37312
s_setprio 0
s_mov_b32 m0, 0x2ffc0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v109, v149, s[40:43], 0 offen
buffer_load_dword v111, v151, s[40:43], 0 offen
buffer_load_dword v110, v150, s[40:43], 0 offen
buffer_load_dword v112, v152, s[40:43], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 2341
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v125, v127, v125 div:2
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_subrev_f32_e64 v128, v126, v128 div:2
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v126, v127, v126 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_fma_f32 v127, v127, 1.0, -v126
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_dpp v125, v125, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_dpp v126, v126, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_dpp v127, v127, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_dpp v128, v128, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v167, v117 offset:16512
ds_read_b128 v[77:80], v175 offset:41792
ds_write_b32 v168, v118 offset:16512
ds_read_b128 v[81:84], v175 offset:42048
ds_write_b32 v169, v119 offset:16512
ds_read_b128 v[93:96], v165 offset:41280
ds_write_b32 v170, v120 offset:16512
ds_read_b128 v[97:100], v165 offset:41408
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v113, v153, s[40:43], 0 offen
buffer_load_dword v115, v155, s[40:43], 0 offen
buffer_load_dword v114, v154, s[40:43], 0 offen
buffer_load_dword v116, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 5
s_call_b64 s[38:39], 2220
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v129, v131, v129 div:2
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_subrev_f32_e64 v132, v130, v132 div:2
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v130, v131, v130 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_fma_f32 v131, v131, 1.0, -v130
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_dpp v129, v129, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_dpp v130, v130, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_dpp v131, v131, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_dpp v132, v132, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v121 offset:16512
ds_read_b128 v[69:72], v175 offset:45952
ds_write_b32 v172, v122 offset:16512
ds_read_b128 v[73:76], v175 offset:46208
ds_write_b32 v173, v123 offset:16512
ds_read_b128 v[85:88], v165 offset:45440
ds_write_b32 v174, v124 offset:16512
ds_read_b128 v[89:92], v165 offset:45568
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v117, v149, s[40:43], 0 offen
buffer_load_dword v119, v151, s[40:43], 0 offen
buffer_load_dword v118, v150, s[40:43], 0 offen
buffer_load_dword v120, v152, s[40:43], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 8
s_call_b64 s[38:39], 2103
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v133, v135, v133 div:2
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_subrev_f32_e64 v136, v134, v136 div:2
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v134, v135, v134 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_fma_f32 v135, v135, 1.0, -v134
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_dpp v133, v133, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_dpp v134, v134, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_dpp v135, v135, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_dpp v136, v136, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
ds_write_b32 v167, v125 offset:24768
ds_read_b128 v[77:80], v175 offset:512
ds_write_b32 v168, v126 offset:24768
ds_read_b128 v[81:84], v175 offset:768
ds_write_b32 v169, v127 offset:24768
ds_read_b128 v[93:96], v165
ds_write_b32 v170, v128 offset:24768
ds_read_b128 v[97:100], v165 offset:128
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v121, v153, s[40:43], 0 offen
buffer_load_dword v123, v155, s[40:43], 0 offen
buffer_load_dword v122, v154, s[40:43], 0 offen
buffer_load_dword v124, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v188 offset:65476
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 1981
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v137, v139, v137 div:2
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_subrev_f32_e64 v140, v138, v140 div:2
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v138, v139, v138 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_fma_f32 v139, v139, 1.0, -v138
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_dpp v137, v137, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_dpp v138, v138, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_dpp v139, v139, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_dpp v140, v140, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v129 offset:24768
ds_read_b128 v[69:72], v175 offset:4672
ds_write_b32 v172, v130 offset:24768
ds_read_b128 v[73:76], v175 offset:4928
ds_write_b32 v173, v131 offset:24768
ds_read_b128 v[85:88], v165 offset:4160
ds_write_b32 v174, v132 offset:24768
ds_read_b128 v[89:92], v165 offset:4288
s_setprio 0
s_mov_b32 m0, 0x2ffc4
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v125, v149, s[40:43], 0 offen
buffer_load_dword v127, v151, s[40:43], 0 offen
buffer_load_dword v126, v150, s[40:43], 0 offen
buffer_load_dword v128, v152, s[40:43], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 1861
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v141, v143, v141 div:2
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_subrev_f32_e64 v144, v142, v144 div:2
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v142, v143, v142 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_fma_f32 v143, v143, 1.0, -v142
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_dpp v141, v141, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_dpp v142, v142, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_dpp v143, v143, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_dpp v144, v144, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v167, v133 offset:33024
ds_read_b128 v[77:80], v175 offset:8768
ds_write_b32 v168, v134 offset:33024
ds_read_b128 v[81:84], v175 offset:9024
ds_write_b32 v169, v135 offset:33024
ds_read_b128 v[93:96], v165 offset:8256
ds_write_b32 v170, v136 offset:33024
ds_read_b128 v[97:100], v165 offset:8384
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v129, v153, s[40:43], 0 offen
buffer_load_dword v131, v155, s[40:43], 0 offen
buffer_load_dword v130, v154, s[40:43], 0 offen
buffer_load_dword v132, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 5
s_call_b64 s[38:39], 1740
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v145, v147, v145 div:2
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_subrev_f32_e64 v148, v146, v148 div:2
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v146, v147, v146 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_fma_f32 v147, v147, 1.0, -v146
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_dpp v145, v145, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_dpp v146, v146, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_dpp v147, v147, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_dpp v148, v148, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v137 offset:33024
ds_read_b128 v[69:72], v175 offset:12928
ds_write_b32 v172, v138 offset:33024
ds_read_b128 v[73:76], v175 offset:13184
ds_write_b32 v173, v139 offset:33024
ds_read_b128 v[85:88], v165 offset:12416
ds_write_b32 v174, v140 offset:33024
ds_read_b128 v[89:92], v165 offset:12544
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v133, v149, s[40:43], 0 offen
buffer_load_dword v135, v151, s[40:43], 0 offen
buffer_load_dword v134, v150, s[40:43], 0 offen
buffer_load_dword v136, v152, s[40:43], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 8
s_call_b64 s[38:39], 1623
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v101, v103, v101 div:2
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_subrev_f32_e64 v104, v102, v104 div:2
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v102, v103, v102 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_fma_f32 v103, v103, 1.0, -v102
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_dpp v101, v101, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_dpp v102, v102, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_dpp v103, v103, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_dpp v104, v104, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
ds_write_b32 v167, v141 offset:41280
ds_read_b128 v[77:80], v175 offset:17024
ds_write_b32 v168, v142 offset:41280
ds_read_b128 v[81:84], v175 offset:17280
ds_write_b32 v169, v143 offset:41280
ds_read_b128 v[93:96], v165 offset:16512
ds_write_b32 v170, v144 offset:41280
ds_read_b128 v[97:100], v165 offset:16640
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v137, v153, s[40:43], 0 offen
buffer_load_dword v139, v155, s[40:43], 0 offen
buffer_load_dword v138, v154, s[40:43], 0 offen
buffer_load_dword v140, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v188 offset:65480
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 1501
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v105, v107, v105 div:2
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_subrev_f32_e64 v108, v106, v108 div:2
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v106, v107, v106 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_fma_f32 v107, v107, 1.0, -v106
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_dpp v105, v105, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_dpp v106, v106, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_dpp v107, v107, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_dpp v108, v108, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v145 offset:41280
ds_read_b128 v[69:72], v175 offset:21184
ds_write_b32 v172, v146 offset:41280
ds_read_b128 v[73:76], v175 offset:21440
ds_write_b32 v173, v147 offset:41280
ds_read_b128 v[85:88], v165 offset:20672
ds_write_b32 v174, v148 offset:41280
ds_read_b128 v[89:92], v165 offset:20800
s_setprio 0
s_mov_b32 m0, 0x2ffc8
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v141, v149, s[40:43], 0 offen
buffer_load_dword v143, v151, s[40:43], 0 offen
buffer_load_dword v142, v150, s[40:43], 0 offen
buffer_load_dword v144, v152, s[40:43], 0 offen
s_waitcnt vmcnt(28) lgkmcnt(8)
s_setprio 1
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 1381
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_subrev_f32_e64 v109, v111, v109 div:2
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_subrev_f32_e64 v112, v110, v112 div:2
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v110, v111, v110 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_fma_f32 v111, v111, 1.0, -v110
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_dpp v109, v109, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_dpp v110, v110, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_dpp v111, v111, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_dpp v112, v112, v183  quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v167, v101
ds_read_b128 v[77:80], v175 offset:25280
ds_write_b32 v168, v102
ds_read_b128 v[81:84], v175 offset:25536
ds_write_b32 v169, v103
ds_read_b128 v[93:96], v165 offset:24768
ds_write_b32 v170, v104
ds_read_b128 v[97:100], v165 offset:24896
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v145, v153, s[40:43], 0 offen
buffer_load_dword v147, v155, s[40:43], 0 offen
buffer_load_dword v146, v154, s[40:43], 0 offen
buffer_load_dword v148, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
s_setprio 1
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 64101
s_call_b64 s[38:39], 1260
s_branch 64099
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_dpp v113, v113, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_fmac_f32_dpp v116, v116, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
s_setprio 1
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v114, v113, v116 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_add_f32_e64 v115, v113, -v116 div:2
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v105
ds_read_b128 v[69:72], v175 offset:29440
ds_write_b32 v172, v106
ds_read_b128 v[73:76], v175 offset:29696
ds_write_b32 v173, v107
ds_read_b128 v[85:88], v165 offset:28928
ds_write_b32 v174, v108
ds_read_b128 v[89:92], v165 offset:29056
s_setprio 0
s_add_u32 s91, s91, 0x100
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v101, v149, s[40:43], 0 offen
buffer_load_dword v104, v152, s[40:43], 0 offen
s_waitcnt vmcnt(14) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 2
s_call_b64 s[38:39], 1153
s_nop 0
v_fmac_f32_dpp v117, v117, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_fmac_f32_dpp v120, v120, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
s_setprio 1
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v118, v117, v120 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_add_f32_e64 v119, v117, -v120 div:2
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
ds_write_b32 v167, v109 offset:8256
ds_read_b128 v[77:80], v175 offset:33536
ds_write_b32 v168, v110 offset:8256
ds_read_b128 v[81:84], v175 offset:33792
ds_write_b32 v169, v111 offset:8256
ds_read_b128 v[93:96], v165 offset:33024
ds_write_b32 v170, v112 offset:8256
ds_read_b128 v[97:100], v165 offset:33152
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v105, v153, s[40:43], 0 offen
buffer_load_dword v108, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v188 offset:65472
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 2
s_call_b64 s[38:39], 1049
s_nop 0
v_fmac_f32_dpp v121, v121, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_fmac_f32_dpp v124, v124, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
s_setprio 1
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v122, v121, v124 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_add_f32_e64 v123, v121, -v124 div:2
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v113 offset:8256
ds_read_b128 v[69:72], v175 offset:37696
ds_write_b32 v172, v114 offset:8256
ds_read_b128 v[73:76], v175 offset:37952
ds_write_b32 v173, v115 offset:8256
ds_read_b128 v[85:88], v165 offset:37184
ds_write_b32 v174, v116 offset:8256
ds_read_b128 v[89:92], v165 offset:37312
s_setprio 0
s_mov_b32 m0, 0x2ffc0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v109, v149, s[40:43], 0 offen
buffer_load_dword v112, v152, s[40:43], 0 offen
s_waitcnt vmcnt(14) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 2
s_call_b64 s[38:39], 945
s_nop 0
v_fmac_f32_dpp v125, v125, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_fmac_f32_dpp v128, v128, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
s_setprio 1
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v126, v125, v128 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_add_f32_e64 v127, v125, -v128 div:2
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v167, v117 offset:16512
ds_read_b128 v[77:80], v175 offset:41792
ds_write_b32 v168, v118 offset:16512
ds_read_b128 v[81:84], v175 offset:42048
ds_write_b32 v169, v119 offset:16512
ds_read_b128 v[93:96], v165 offset:41280
ds_write_b32 v170, v120 offset:16512
ds_read_b128 v[97:100], v165 offset:41408
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v113, v153, s[40:43], 0 offen
buffer_load_dword v116, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 1
s_call_b64 s[38:39], 840
v_fmac_f32_dpp v129, v129, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_fmac_f32_dpp v132, v132, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
s_setprio 1
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v130, v129, v132 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_add_f32_e64 v131, v129, -v132 div:2
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v121 offset:16512
ds_read_b128 v[69:72], v175 offset:45952
ds_write_b32 v172, v122 offset:16512
ds_read_b128 v[73:76], v175 offset:46208
ds_write_b32 v173, v123 offset:16512
ds_read_b128 v[85:88], v165 offset:45440
ds_write_b32 v174, v124 offset:16512
ds_read_b128 v[89:92], v165 offset:45568
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v117, v149, s[40:43], 0 offen
buffer_load_dword v120, v152, s[40:43], 0 offen
s_waitcnt vmcnt(14) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 4
s_call_b64 s[38:39], 739
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_dpp v133, v133, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_fmac_f32_dpp v136, v136, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
s_setprio 1
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v134, v133, v136 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_add_f32_e64 v135, v133, -v136 div:2
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
ds_write_b32 v167, v125 offset:24768
ds_read_b128 v[77:80], v175 offset:512
ds_write_b32 v168, v126 offset:24768
ds_read_b128 v[81:84], v175 offset:768
ds_write_b32 v169, v127 offset:24768
ds_read_b128 v[93:96], v165
ds_write_b32 v170, v128 offset:24768
ds_read_b128 v[97:100], v165 offset:128
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v121, v153, s[40:43], 0 offen
buffer_load_dword v124, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v188 offset:65476
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 2
s_call_b64 s[38:39], 633
s_nop 0
v_fmac_f32_dpp v137, v137, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_fmac_f32_dpp v140, v140, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
s_setprio 1
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v138, v137, v140 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_add_f32_e64 v139, v137, -v140 div:2
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v129 offset:24768
ds_read_b128 v[69:72], v175 offset:4672
ds_write_b32 v172, v130 offset:24768
ds_read_b128 v[73:76], v175 offset:4928
ds_write_b32 v173, v131 offset:24768
ds_read_b128 v[85:88], v165 offset:4160
ds_write_b32 v174, v132 offset:24768
ds_read_b128 v[89:92], v165 offset:4288
s_setprio 0
s_mov_b32 m0, 0x2ffc4
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v125, v149, s[40:43], 0 offen
buffer_load_dword v128, v152, s[40:43], 0 offen
s_waitcnt vmcnt(14) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 2
s_call_b64 s[38:39], 529
s_nop 0
v_fmac_f32_dpp v141, v141, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_fmac_f32_dpp v144, v144, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
s_setprio 1
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v142, v141, v144 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_add_f32_e64 v143, v141, -v144 div:2
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v167, v133 offset:33024
ds_read_b128 v[77:80], v175 offset:8768
ds_write_b32 v168, v134 offset:33024
ds_read_b128 v[81:84], v175 offset:9024
ds_write_b32 v169, v135 offset:33024
ds_read_b128 v[93:96], v165 offset:8256
ds_write_b32 v170, v136 offset:33024
ds_read_b128 v[97:100], v165 offset:8384
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v129, v153, s[40:43], 0 offen
buffer_load_dword v132, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 1
s_call_b64 s[38:39], 424
v_fmac_f32_dpp v145, v145, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_fmac_f32_dpp v148, v148, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
s_setprio 1
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v146, v145, v148 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_add_f32_e64 v147, v145, -v148 div:2
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v137 offset:33024
ds_read_b128 v[69:72], v175 offset:12928
ds_write_b32 v172, v138 offset:33024
ds_read_b128 v[73:76], v175 offset:13184
ds_write_b32 v173, v139 offset:33024
ds_read_b128 v[85:88], v165 offset:12416
ds_write_b32 v174, v140 offset:33024
ds_read_b128 v[89:92], v165 offset:12544
s_setprio 0
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v133, v149, s[40:43], 0 offen
buffer_load_dword v136, v152, s[40:43], 0 offen
s_waitcnt vmcnt(14) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 4
s_call_b64 s[38:39], 323
s_nop 0
s_nop 0
s_nop 0
v_fmac_f32_dpp v101, v101, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_fmac_f32_dpp v104, v104, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
s_setprio 1
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v102, v101, v104 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_add_f32_e64 v103, v101, -v104 div:2
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
ds_write_b32 v167, v141 offset:41280
ds_read_b128 v[77:80], v175 offset:17024
ds_write_b32 v168, v142 offset:41280
ds_read_b128 v[81:84], v175 offset:17280
ds_write_b32 v169, v143 offset:41280
ds_read_b128 v[93:96], v165 offset:16512
ds_write_b32 v170, v144 offset:41280
ds_read_b128 v[97:100], v165 offset:16640
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v137, v153, s[40:43], 0 offen
buffer_load_dword v140, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
ds_append v188 offset:65480
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 2
s_call_b64 s[38:39], 217
s_nop 0
v_fmac_f32_dpp v105, v105, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v69, v85
v_fmac_f32_e32 v3, v70, v85
v_fmac_f32_e32 v4, v71, v85
v_fmac_f32_e32 v5, v72, v85
v_fmac_f32_e32 v6, v73, v85
v_fmac_f32_e32 v7, v74, v85
v_fmac_f32_e32 v8, v75, v85
v_fmac_f32_e32 v9, v76, v85
v_fmac_f32_dpp v108, v108, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v69, v86
v_fmac_f32_e32 v11, v70, v86
v_fmac_f32_e32 v12, v71, v86
s_setprio 1
v_fmac_f32_e32 v13, v72, v86
v_fmac_f32_e32 v14, v73, v86
v_fmac_f32_e32 v15, v74, v86
v_fmac_f32_e32 v16, v75, v86
v_fmac_f32_e32 v17, v76, v86
v_add_f32_e64 v106, v105, v108 div:2
v_fmac_f32_e32 v18, v69, v87
v_fmac_f32_e32 v19, v70, v87
v_fmac_f32_e32 v20, v71, v87
v_fmac_f32_e32 v21, v72, v87
v_fmac_f32_e32 v22, v73, v87
v_fmac_f32_e32 v23, v74, v87
v_fmac_f32_e32 v24, v75, v87
v_fmac_f32_e32 v25, v76, v87
v_add_f32_e64 v107, v105, -v108 div:2
v_fmac_f32_e32 v26, v69, v88
v_fmac_f32_e32 v27, v70, v88
v_fmac_f32_e32 v28, v71, v88
v_fmac_f32_e32 v29, v72, v88
v_fmac_f32_e32 v30, v73, v88
v_fmac_f32_e32 v31, v74, v88
v_fmac_f32_e32 v32, v75, v88
v_fmac_f32_e32 v33, v76, v88
v_fmac_f32_e32 v34, v69, v89
v_fmac_f32_e32 v35, v70, v89
v_fmac_f32_e32 v36, v71, v89
v_fmac_f32_e32 v37, v72, v89
v_fmac_f32_e32 v38, v73, v89
v_fmac_f32_e32 v39, v74, v89
v_fmac_f32_e32 v40, v75, v89
v_fmac_f32_e32 v41, v76, v89
v_fmac_f32_e32 v42, v69, v90
v_fmac_f32_e32 v43, v70, v90
v_fmac_f32_e32 v44, v71, v90
v_fmac_f32_e32 v45, v72, v90
v_fmac_f32_e32 v46, v73, v90
v_fmac_f32_e32 v47, v74, v90
v_fmac_f32_e32 v48, v75, v90
v_fmac_f32_e32 v49, v76, v90
v_fmac_f32_e32 v50, v69, v91
v_fmac_f32_e32 v51, v70, v91
v_fmac_f32_e32 v52, v71, v91
v_fmac_f32_e32 v53, v72, v91
v_fmac_f32_e32 v54, v73, v91
v_fmac_f32_e32 v55, v74, v91
v_fmac_f32_e32 v56, v75, v91
v_fmac_f32_e32 v57, v76, v91
v_fmac_f32_e32 v58, v69, v92
v_fmac_f32_e32 v59, v70, v92
v_fmac_f32_e32 v60, v71, v92
v_fmac_f32_e32 v61, v72, v92
v_fmac_f32_e32 v62, v73, v92
v_fmac_f32_e32 v63, v74, v92
v_fmac_f32_e32 v64, v75, v92
v_fmac_f32_e32 v65, v76, v92
ds_write_b32 v171, v145 offset:41280
ds_read_b128 v[69:72], v175 offset:21184
ds_write_b32 v172, v146 offset:41280
ds_read_b128 v[73:76], v175 offset:21440
ds_write_b32 v173, v147 offset:41280
ds_read_b128 v[85:88], v165 offset:20672
ds_write_b32 v174, v148 offset:41280
ds_read_b128 v[89:92], v165 offset:20800
s_setprio 0
s_mov_b32 m0, 0x2ffc8
s_add_u32 s40, s40, s69
s_addc_u32 s41, s41, 0
buffer_load_dword v141, v149, s[40:43], 0 offen
buffer_load_dword v144, v152, s[40:43], 0 offen
s_waitcnt vmcnt(14) lgkmcnt(8)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 2
s_call_b64 s[38:39], 113
s_nop 0
v_fmac_f32_dpp v109, v109, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v2, v77, v93
v_fmac_f32_e32 v3, v78, v93
v_fmac_f32_e32 v4, v79, v93
v_fmac_f32_e32 v5, v80, v93
v_fmac_f32_e32 v6, v81, v93
v_fmac_f32_e32 v7, v82, v93
v_fmac_f32_e32 v8, v83, v93
v_fmac_f32_e32 v9, v84, v93
v_fmac_f32_dpp v112, v112, v183  quad_perm:[0,0,1,1] row_mask:0xf bank_mask:0xf
v_fmac_f32_e32 v10, v77, v94
v_fmac_f32_e32 v11, v78, v94
v_fmac_f32_e32 v12, v79, v94
s_setprio 1
v_fmac_f32_e32 v13, v80, v94
v_fmac_f32_e32 v14, v81, v94
v_fmac_f32_e32 v15, v82, v94
v_fmac_f32_e32 v16, v83, v94
v_fmac_f32_e32 v17, v84, v94
v_add_f32_e64 v110, v109, v112 div:2
v_fmac_f32_e32 v18, v77, v95
v_fmac_f32_e32 v19, v78, v95
v_fmac_f32_e32 v20, v79, v95
v_fmac_f32_e32 v21, v80, v95
v_fmac_f32_e32 v22, v81, v95
v_fmac_f32_e32 v23, v82, v95
v_fmac_f32_e32 v24, v83, v95
v_fmac_f32_e32 v25, v84, v95
v_add_f32_e64 v111, v109, -v112 div:2
v_fmac_f32_e32 v26, v77, v96
v_fmac_f32_e32 v27, v78, v96
v_fmac_f32_e32 v28, v79, v96
v_fmac_f32_e32 v29, v80, v96
v_fmac_f32_e32 v30, v81, v96
v_fmac_f32_e32 v31, v82, v96
v_fmac_f32_e32 v32, v83, v96
v_fmac_f32_e32 v33, v84, v96
v_fmac_f32_e32 v34, v77, v97
v_fmac_f32_e32 v35, v78, v97
v_fmac_f32_e32 v36, v79, v97
v_fmac_f32_e32 v37, v80, v97
v_fmac_f32_e32 v38, v81, v97
v_fmac_f32_e32 v39, v82, v97
v_fmac_f32_e32 v40, v83, v97
v_fmac_f32_e32 v41, v84, v97
v_fmac_f32_e32 v42, v77, v98
v_fmac_f32_e32 v43, v78, v98
v_fmac_f32_e32 v44, v79, v98
v_fmac_f32_e32 v45, v80, v98
v_fmac_f32_e32 v46, v81, v98
v_fmac_f32_e32 v47, v82, v98
v_fmac_f32_e32 v48, v83, v98
v_fmac_f32_e32 v49, v84, v98
v_fmac_f32_e32 v50, v77, v99
v_fmac_f32_e32 v51, v78, v99
v_fmac_f32_e32 v52, v79, v99
v_fmac_f32_e32 v53, v80, v99
v_fmac_f32_e32 v54, v81, v99
v_fmac_f32_e32 v55, v82, v99
v_fmac_f32_e32 v56, v83, v99
v_fmac_f32_e32 v57, v84, v99
v_fmac_f32_e32 v58, v77, v100
v_fmac_f32_e32 v59, v78, v100
v_fmac_f32_e32 v60, v79, v100
v_fmac_f32_e32 v61, v80, v100
v_fmac_f32_e32 v62, v81, v100
v_fmac_f32_e32 v63, v82, v100
v_fmac_f32_e32 v64, v83, v100
v_fmac_f32_e32 v65, v84, v100
v_cmp_eq_u32_e64 vcc, src_lds_direct, s91
s_cbranch_vccz 65533
ds_write_b32 v167, v101
ds_read_b128 v[77:80], v175 offset:25280
ds_write_b32 v168, v102
ds_read_b128 v[81:84], v175 offset:25536
ds_write_b32 v169, v103
ds_read_b128 v[93:96], v165 offset:24768
ds_write_b32 v170, v104
ds_read_b128 v[97:100], v165 offset:24896
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v145, v153, s[40:43], 0 offen
buffer_load_dword v148, v156, s[40:43], 0 offen
s_waitcnt lgkmcnt(8)
s_bitset1_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 64289
s_call_b64 s[38:39], 8
s_branch 64287
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_nop
s_cmp_eq_u32 s82, 0
s_cbranch_scc0 6
s_branch 724
s_bitcmp1_b32 s18, 26
s_cselect_b32 s52, s69, s70
s_cselect_b32 s53, 0, s71
s_sub_u32 s40, s40, s52
s_subb_u32 s41, s41, s53
s_cmp_eq_u32 s94, 0
s_cbranch_scc0 5
s_cbranch_scc1 748
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_min_u32 s72, s82, s94
s_sub_u32 s82, s82, s72
s_sub_u32 s94, s94, s72
s_sub_u32 s72, s72, 1
s_setpc_b64 s[38:39]
s_nop 0
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
v_mov_b32_e32 v190, s88
v_mul_u32_u24_e32 v190, s87, v190
v_add_co_u32_e64 v190, vcc, s17, v190
v_sub_co_u32_e64 v190, vcc, v190, 1
v_ffbh_u32_e32 v193, s17
v_lshlrev_b32_e64 v194, v193, s17
v_and_b32_e32 v195, 0xffffff00, v194
v_cmp_eq_u32_e32 vcc, 0x80000000, v194
v_cvt_f32_u32_e32 v195, v195
v_rcp_f32_e32 v189, v195
v_sub_co_ci_u32_e32 v192, vcc, 32, v193, vcc
v_cvt_f32_ubyte0_e32 v193, v194
v_fma_f32 v195, v195, v189, -1.0
v_fma_f32 v195, v193, v189, v195
v_fmaak_f32 v195, v195, v189, 0x9f000000
v_mul_f32_e32 v195, 0x5f800000, v195
v_mov_b32_e32 v193, 0
v_cvt_flr_i32_f32_e64 v195, -v195
v_lshl_add_u32 v189, v189, 9, v195
v_mad_u64_u32 v[193:194], vcc, v194, v189, v[193:194]
v_sub_co_ci_u32_e64 v189, vcc, v189, -1, vcc
v_mul_hi_u32 v193, v190, v189
v_add_co_u32_e64 v189, vcc, v193, v190
v_add_co_ci_u32_e64 v193, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v192
v_cndmask_b32_e32 v189, v189, v193, vcc
v_alignbit_b32 v189, v193, v189, v192
v_readfirstlane_b32 s86, v189
v_mul_u32_u24_e64 v189, v189, s8
v_ffbh_u32_e32 v193, s87
v_lshlrev_b32_e64 v194, v193, s87
v_and_b32_e32 v195, 0xffffff00, v194
v_cmp_eq_u32_e32 vcc, 0x80000000, v194
v_cvt_f32_u32_e32 v195, v195
v_rcp_f32_e32 v190, v195
v_sub_co_ci_u32_e32 v192, vcc, 32, v193, vcc
v_cvt_f32_ubyte0_e32 v193, v194
v_fma_f32 v195, v195, v190, -1.0
v_fma_f32 v195, v193, v190, v195
v_fmaak_f32 v195, v195, v190, 0x9f000000
v_mul_f32_e32 v195, 0x5f800000, v195
v_mov_b32_e32 v193, 0
v_cvt_flr_i32_f32_e64 v195, -v195
v_lshl_add_u32 v190, v190, 9, v195
v_mad_u64_u32 v[193:194], vcc, v194, v190, v[193:194]
v_sub_co_ci_u32_e64 v190, vcc, v190, -1, vcc
v_mul_hi_u32 v193, v189, v190
v_add_co_u32_e64 v190, vcc, v193, v189
v_add_co_ci_u32_e64 v193, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v192
v_cndmask_b32_e32 v190, v190, v193, vcc
v_alignbit_b32 v190, v193, v190, v192
v_readfirstlane_b32 s52, v189
v_readfirstlane_b32 s84, v190
s_mul_i32 s84, s84, s87
s_sub_u32 s84, s52, s84
v_sub_co_u32_e64 v190, vcc, s8, v190
v_sub_co_u32_e64 v190, vcc, s17, v190
v_and_b32_e64 v192, v1, 63
v_cmp_eq_u32_e64 vcc, v192, 0
v_cndmask_b32_e32 v190, 1, v190, vcc
s_sub_u32 s58, 0, s75
s_sub_u32 s59, 0, s74
v_mul_u32_u24_e64 v194, v190, 32
v_ffbh_u32_e32 v196, s58
v_lshlrev_b32_e64 v197, v196, s58
v_and_b32_e32 v198, 0xffffff00, v197
v_cmp_eq_u32_e32 vcc, 0x80000000, v197
v_cvt_f32_u32_e32 v198, v198
v_rcp_f32_e32 v192, v198
v_sub_co_ci_u32_e32 v195, vcc, 32, v196, vcc
v_cvt_f32_ubyte0_e32 v196, v197
v_fma_f32 v198, v198, v192, -1.0
v_fma_f32 v198, v196, v192, v198
v_fmaak_f32 v198, v198, v192, 0x9f000000
v_mul_f32_e32 v198, 0x5f800000, v198
v_mov_b32_e32 v196, 0
v_cvt_flr_i32_f32_e64 v198, -v198
v_lshl_add_u32 v192, v192, 9, v198
v_mad_u64_u32 v[196:197], vcc, v197, v192, v[196:197]
v_sub_co_ci_u32_e64 v192, vcc, v192, -1, vcc
v_mul_hi_u32 v196, v194, v192
v_add_co_u32_e64 v192, vcc, v196, v194
v_add_co_ci_u32_e64 v196, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v195
v_cndmask_b32_e32 v192, v192, v196, vcc
v_alignbit_b32 v192, v196, v192, v195
v_mad_i32_i24 v193, v192, s75, v194
v_mul_u32_u24_e64 v194, v192, 1
v_ffbh_u32_e32 v196, s59
v_lshlrev_b32_e64 v197, v196, s59
v_and_b32_e32 v198, 0xffffff00, v197
v_cmp_eq_u32_e32 vcc, 0x80000000, v197
v_cvt_f32_u32_e32 v198, v198
v_rcp_f32_e32 v192, v198
v_sub_co_ci_u32_e32 v195, vcc, 32, v196, vcc
v_cvt_f32_ubyte0_e32 v196, v197
v_fma_f32 v198, v198, v192, -1.0
v_fma_f32 v198, v196, v192, v198
v_fmaak_f32 v198, v198, v192, 0x9f000000
v_mul_f32_e32 v198, 0x5f800000, v198
v_mov_b32_e32 v196, 0
v_cvt_flr_i32_f32_e64 v198, -v198
v_lshl_add_u32 v192, v192, 9, v198
v_mad_u64_u32 v[196:197], vcc, v197, v192, v[196:197]
v_sub_co_ci_u32_e64 v192, vcc, v192, -1, vcc
v_mul_hi_u32 v196, v194, v192
v_add_co_u32_e64 v192, vcc, v196, v194
v_add_co_ci_u32_e64 v196, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v195
v_cndmask_b32_e32 v192, v192, v196, vcc
v_alignbit_b32 v192, v196, v192, v195
v_mad_i32_i24 v194, v192, s74, v194
v_readfirstlane_b32 s76, v193
v_readfirstlane_b32 s77, v194
v_readfirstlane_b32 s78, v192
v_add_co_u32_e64 v179, vcc, s76, v179
v_add_co_ci_u32_e64 v195, vcc, 0, 0, vcc
v_mad_i32_i24 v179, v195, s75, v179
v_mad_i32_i24 v181, v195, s80, v181
v_mad_i32_i24 v180, v195, s79, v180
v_cmp_ge_i32_e64 vcc, v180, 0
v_add_co_ci_u32_e64 v195, vcc, 0, 0, vcc
v_add_co_u32_e64 v181, vcc, v181, v195
v_mad_i32_i24 v180, v195, s74, v180
v_add_co_u32_e64 v180, vcc, s77, v180
v_add_co_ci_u32_e64 v195, vcc, 0, 0, vcc
v_add_co_u32_e64 v181, vcc, v181, v195
v_mad_i32_i24 v180, v195, s74, v180
v_add_co_u32_e64 v181, vcc, s78, v181
v_readlane_b32 s76, v193, 1
v_readlane_b32 s77, v194, 1
v_readlane_b32 s78, v192, 1
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
v_add_co_u32_e64 v179, vcc, s76, v179
v_add_co_ci_u32_e64 v189, vcc, 0, 0, vcc
v_mad_i32_i24 v179, v189, s75, v179
v_mad_i32_i24 v181, v189, s80, v181
v_mad_i32_i24 v180, v189, s79, v180
v_cmp_ge_i32_e64 vcc, v180, 0
v_add_co_ci_u32_e64 v189, vcc, 0, 0, vcc
v_add_co_u32_e64 v181, vcc, v181, v189
v_mad_i32_i24 v180, v189, s74, v180
v_add_co_u32_e64 v180, vcc, s77, v180
v_add_co_ci_u32_e64 v189, vcc, 0, 0, vcc
v_add_co_u32_e64 v181, vcc, v181, v189
v_mad_i32_i24 v180, v189, s74, v180
v_add_co_u32_e64 v181, vcc, s78, v181
s_mov_b32 s83, s84
v_cmp_le_u32_e32 vcc, 0x100, v1
s_cbranch_vccz 255
v_subrev_co_u32_e64 v189, vcc, s75, v179
v_subrev_co_u32_e64 v190, vcc, s74, v180
s_bitcmp1_b32 s18, 22
s_cbranch_scc0 66
s_bitset0_b32 s18, 22
s_bfe_u32 s52, s18, 0x10014
v_mul_u32_u24_e32 v194, 3, v189
v_mul_u32_u24_e32 v195, 3, v190
v_cvt_pk_u16_u32_e64 v197, v194, v195
v_and_b32_e64 v194, v1, 1
v_cmp_eq_u32_e64 vcc, v194, 1
v_cndmask_b32_e32 v197, v181, v197, vcc
v_lshrrev_b32_e32 v193, 1, v1
v_bfe_u32 v198, v193, s52, 1
v_lshrrev_b32_e32 v193, 1, v1
v_bfi_b32 v193, 1, v1, v193
v_lshrrev_b32_e32 v194, 2, v1
v_bfi_b32 v194, 1, v1, v194
v_cmp_eq_u32_e64 vcc, s52, 0
v_cndmask_b32_e32 v193, v194, v193, vcc
s_sub_u32 s52, 1, s52
v_lshrrev_b32_e32 v194, s52, v193
v_bfi_b32 v193, 32, v194, v193
v_and_b32_e32 v193, 63, v193
v_add_co_u32_e64 v194, vcc, 16, v193
v_and_b32_e64 v195, v1, 2
v_cmp_eq_u32_e64 vcc, v195, 0
v_cndmask_b32_e32 v194, v194, v193, vcc
v_lshlrev_b32_e32 v195, 14, v198
v_mad_u32_u24 v194, 4, v194, v195
v_add_co_u32_e64 v193, vcc, s96, v194
ds_write_b32 v193, v197
v_writelane_b32 v195, s18, 0
v_writelane_b32 v195, s85, 1
v_writelane_b32 v195, s84, 2
v_and_b32_e64 v193, v1, 63
v_cmp_ge_u32_e64 vcc, v193, 3
v_mov_b32_e32 v196, 0x4000
v_cndmask_b32_e32 v193, v193, v196, vcc
v_mad_i32_i24 v193, v193, 4, s96
ds_write_b32 v193, v195 offset:256
s_add_u32 s96, s96, 0x18c
s_cmp_eq_u32 s96, 0xffc0
s_cselect_b32 s96, 0xc1e0, s96
v_mov_b32_dpp v191, v181  quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v189, v189  quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v190, v190  quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_readfirstlane_b32 s81, v191
v_sub_co_u32_e64 v192, vcc, v191, s81
v_mul_lo_u32 v192, v192, s65
v_and_b32_e64 v196, v1, 3
v_ashrrev_i32_e64 v197, 0, s31
v_subrev_co_u32_e64 v196, vcc, v197, v196
v_ashrrev_i32_e64 v197, 0, s62
v_mad_i32_i24 v193, v197, 2, v196
s_bfe_u32 s52, s18, 0x10014
v_lshrrev_b32_e32 v195, 2, v1
v_and_b32_e32 v195, s52, v195
v_mad_i32_i24 v193, v195, 2, v193
v_add_co_u32_e64 v194, vcc, 0, s63
v_ashrrev_i32_e32 v194, 0, v194
v_add_co_u32_e64 v195, vcc, 0, s30
v_ashrrev_i32_e32 v195, 0, v195
v_sub_nc_i32 v194, v194, v195
s_lshl_b32 s54, s15, 2
v_cmp_ge_u32_e64 s[52:53], v191, s12
v_mad_i32_i24 v189, v189, 3, v193
v_cmp_ge_u32_e64 s[56:57], v189, s15
v_mad_i32_i24 v189, 4, v189, v192
s_or_b64 s[56:57], s[56:57], s[52:53]
v_mad_i32_i24 v190, v190, 3, v194
v_cmp_ge_u32_e64 s[58:59], v190, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v149, v190, s54, v189
v_cndmask_b32_e64 v149, v149, -1, s[58:59]
v_add_co_u32_e64 v190, vcc, 1, v190
v_cmp_ge_u32_e64 s[58:59], v190, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v150, v190, s54, v189
v_cndmask_b32_e64 v150, v150, -1, s[58:59]
v_add_co_u32_e64 v190, vcc, 1, v190
v_cmp_ge_u32_e64 s[58:59], v190, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v151, v190, s54, v189
v_cndmask_b32_e64 v151, v151, -1, s[58:59]
v_add_co_u32_e64 v190, vcc, 1, v190
v_cmp_ge_u32_e64 s[58:59], v190, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v152, v190, s54, v189
v_cndmask_b32_e64 v152, v152, -1, s[58:59]
s_bitcmp1_b32 s18, 20
s_cbranch_scc0 60
v_mov_b32_dpp v191, v181  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v189, v179  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v190, v180  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[52:53], v191, s12
v_sub_co_u32_e64 v192, vcc, v191, s81
v_mul_lo_u32 v192, v192, s65
v_sub_co_u32_e64 v189, vcc, v189, s75
v_sub_co_u32_e64 v190, vcc, v190, s74
v_mad_i32_i24 v189, v189, 3, v193
v_cmp_ge_u32_e64 s[56:57], v189, s15
v_mad_i32_i24 v189, 4, v189, v192
s_or_b64 s[56:57], s[56:57], s[52:53]
v_mad_i32_i24 v190, v190, 3, v194
v_cmp_ge_u32_e64 s[58:59], v190, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v153, v190, s54, v189
v_cndmask_b32_e64 v153, v153, -1, s[58:59]
v_add_co_u32_e64 v190, vcc, 1, v190
v_cmp_ge_u32_e64 s[58:59], v190, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v154, v190, s54, v189
v_cndmask_b32_e64 v154, v154, -1, s[58:59]
v_add_co_u32_e64 v190, vcc, 1, v190
v_cmp_ge_u32_e64 s[58:59], v190, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v155, v190, s54, v189
v_cndmask_b32_e64 v155, v155, -1, s[58:59]
v_add_co_u32_e64 v190, vcc, 1, v190
v_cmp_ge_u32_e64 s[58:59], v190, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v156, v190, s54, v189
v_cndmask_b32_e64 v156, v156, -1, s[58:59]
s_branch 26
s_bitcmp1_b32 s18, 24
s_cselect_b32 s52, s68, 0
v_add_co_u32_e64 v189, vcc, v149, s52
v_cmp_eq_u32_e64 vcc, v149, -1
v_cndmask_b32_e64 v153, v189, -1, vcc
v_add_co_u32_e64 v189, vcc, v150, s52
v_cmp_eq_u32_e64 vcc, v150, -1
v_cndmask_b32_e64 v154, v189, -1, vcc
v_add_co_u32_e64 v189, vcc, v151, s52
v_cmp_eq_u32_e64 vcc, v151, -1
v_cndmask_b32_e64 v155, v189, -1, vcc
v_add_co_u32_e64 v189, vcc, v152, s52
v_cmp_eq_u32_e64 vcc, v152, -1
v_cndmask_b32_e64 v156, v189, -1, vcc
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
s_branch 130
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 140
s_bfe_u32 s52, s18, 0x10014
v_xor_b32_dpp v189, v1, v1  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_bfe_u32 v191, v1, 2, s52
v_mad_u32_u24 v189, v191, 2, v189
v_mad_u32_u24 v189, s62, 2, v189
v_sub_co_u32_e64 v191, vcc, s29, v189
v_sub_co_u32_e64 v191, vcc, v191, 1
s_bfe_u32 s54, s18, 0x10001
v_cmp_eq_u32_e64 vcc, s54, 1
v_cndmask_b32_e32 v189, v189, v191, vcc
v_cmp_ge_u32_e64 s[52:53], v189, s29
v_lshlrev_b32_e32 v189, 2, v189
s_bfe_u32 s54, s18, 0x10018
v_bfe_u32 v192, v1, 2, s54
v_mul_lo_u32 v192, s68, v192
v_add_co_u32_e64 v189, vcc, v189, v192
v_mul_lo_u32 v190, s90, v182
v_add_co_u32_e64 v190, vcc, v190, v189
s_sub_u32 s54, s28, s63
s_sub_u32 s54, s54, 2
s_bitcmp1_b32 s18, 0
s_cselect_b32 s54, s54, s63
v_mov_b32_e32 v192, s54
s_lshl_b32 s57, s29, 2
v_cmp_ge_u32_e64 s[54:55], v192, s28
v_mad_i32_i24 v149, v192, s57, v190
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v149, v149, -1, s[54:55]
v_mov_b32_e32 v150, v149
v_add_co_u32_e64 v192, vcc, v192, 1
v_cmp_ge_u32_e64 s[54:55], v192, s28
v_mad_i32_i24 v152, v192, s57, v190
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v152, v152, -1, s[54:55]
v_add_co_u32_e64 v192, vcc, v192, 1
v_cmp_ge_u32_e64 s[54:55], v192, s28
v_mad_i32_i24 v151, v192, s57, v190
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v151, v151, -1, s[54:55]
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v149, v150, v152, vcc
v_cndmask_b32_e32 v152, v152, v150, vcc
s_lshl_b32 s52, s90, 3
s_and_b32 s53, s18, 0x1100000
s_cselect_b32 s52, s52, 0
v_add_co_u32_e64 v189, vcc, v149, s52
v_cmp_eq_u32_e64 vcc, v149, -1
v_cndmask_b32_e64 v153, v189, -1, vcc
v_add_co_u32_e64 v189, vcc, v150, s52
v_cmp_eq_u32_e64 vcc, v150, -1
v_cndmask_b32_e64 v154, v189, -1, vcc
v_add_co_u32_e64 v189, vcc, v151, s52
v_cmp_eq_u32_e64 vcc, v151, -1
v_cndmask_b32_e64 v155, v189, -1, vcc
v_add_co_u32_e64 v189, vcc, v152, s52
v_cmp_eq_u32_e64 vcc, v152, -1
v_cndmask_b32_e64 v156, v189, -1, vcc
v_add_co_u32_e64 v189, vcc, v182, s83
v_cmp_lt_u32_e64 vcc, v189, s16
v_cndmask_b32_e32 v149, -1, v149, vcc
v_cndmask_b32_e32 v150, -1, v150, vcc
v_cndmask_b32_e32 v151, -1, v151, vcc
v_cndmask_b32_e32 v152, -1, v152, vcc
s_and_b32 s52, s18, 0x1100000
s_cbranch_scc0 4
v_add_co_u32_e64 v189, vcc, v189, 8
v_cmp_lt_u32_e64 vcc, v189, s16
v_cndmask_b32_e32 v153, -1, v153, vcc
v_cndmask_b32_e32 v154, -1, v154, vcc
v_cndmask_b32_e32 v155, -1, v155, vcc
v_cndmask_b32_e32 v156, -1, v156, vcc
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
s_bfe_u32 s52, s18, 0x10014
s_lshl_b32 s82, s13, s52
s_bitcmp1_b32 s18, 20
s_cselect_b32 s52, 0, 0x2000000
s_bitcmp1_b32 s13, 0
s_cselect_b32 s52, s52, 0
s_xor_b32 s18, s18, s52
s_branch 64819
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_and_b32 s52, 0x900000, s18
s_subb_u32 s62, s62, 1
s_cbranch_scc0 65124
s_and_b32 s52, 0x900000, s18
s_subb_u32 s62, s61, 1
s_add_u32 s63, s63, 2
s_cmp_ge_u32 s63, s28
s_cbranch_scc0 65118
s_mov_b32 s63, 0
s_branch 65080
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
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
v_mov_b32_e32 v190, v3
v_mov_b32_e32 v191, v4
v_mov_b32_e32 v192, v5
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v2
v_add_f32_dpp v189, v2, v2  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v3, v3  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v4, v4  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v5, v5  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v4, v4, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v5, v5, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v2, v2, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v3, v3, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v3, v4  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v2, v5  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v3, v3  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v3, v3, v3  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v2, v2  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v2, v2, v2  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v5, v190
v_add_f32_dpp v5, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v4, v189
v_add_f32_dpp v4, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v3, v3, v2  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v2, v5, v4  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v4, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v4, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v3, v191, v3, s[52:53]
v_mov_b32_dpp v4, v4  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v4, v4  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v7
v_mov_b32_e32 v191, v8
v_mov_b32_e32 v192, v9
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v6
v_add_f32_dpp v189, v6, v6  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v7, v7  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v8, v8  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v9, v9  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v8, v8, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v9, v9, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v6, v6, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v7, v7, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v7, v8  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v6, v9  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v7, v7  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v7, v7, v7  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v6, v6  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v6, v6, v6  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v9, v190
v_add_f32_dpp v9, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v8, v189
v_add_f32_dpp v8, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v7, v7, v6  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v5, v9, v8  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v8, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v8, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v6, v191, v7, s[52:53]
v_mov_b32_dpp v7, v8  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v7, v8  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v11
v_mov_b32_e32 v191, v12
v_mov_b32_e32 v192, v13
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v10
v_add_f32_dpp v189, v10, v10  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v11, v11  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v12, v12  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v13, v13  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v12, v12, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v13, v13, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v10, v10, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v11, v11, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v11, v12  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v10, v13  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v11, v11  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v11, v11, v11  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v10, v10  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v10, v10, v10  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v13, v190
v_add_f32_dpp v13, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v12, v189
v_add_f32_dpp v12, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v11, v11, v10  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v8, v13, v12  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v12, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v12, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v9, v191, v11, s[52:53]
v_mov_b32_dpp v10, v12  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v10, v12  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v15
v_mov_b32_e32 v191, v16
v_mov_b32_e32 v192, v17
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v14
v_add_f32_dpp v189, v14, v14  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v15, v15  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v16, v16  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v17, v17  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v16, v16, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v17, v17, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v14, v14, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v15, v15, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v15, v16  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v14, v17  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v15, v15  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v15, v15, v15  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v14, v14  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v14, v14, v14  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v17, v190
v_add_f32_dpp v17, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v16, v189
v_add_f32_dpp v16, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v15, v15, v14  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v11, v17, v16  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v16, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v16, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v12, v191, v15, s[52:53]
v_mov_b32_dpp v13, v16  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v13, v16  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v19
v_mov_b32_e32 v191, v20
v_mov_b32_e32 v192, v21
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v18
v_add_f32_dpp v189, v18, v18  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v19, v19  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v20, v20  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v21, v21  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v20, v20, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v21, v21, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v18, v18, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v19, v19, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v19, v20  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v18, v21  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v19, v19  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v19, v19, v19  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v18, v18  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v18, v18, v18  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v21, v190
v_add_f32_dpp v21, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v20, v189
v_add_f32_dpp v20, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v19, v19, v18  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v14, v21, v20  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v20, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v20, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v15, v191, v19, s[52:53]
v_mov_b32_dpp v16, v20  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v16, v20  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v23
v_mov_b32_e32 v191, v24
v_mov_b32_e32 v192, v25
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v22
v_add_f32_dpp v189, v22, v22  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v23, v23  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v24, v24  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v25, v25  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v24, v24, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v25, v25, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v22, v22, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v23, v23, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v23, v24  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v22, v25  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v23, v23  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v23, v23, v23  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v22, v22  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v22, v22, v22  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v25, v190
v_add_f32_dpp v25, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v24, v189
v_add_f32_dpp v24, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v23, v23, v22  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v17, v25, v24  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v24, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v24, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v18, v191, v23, s[52:53]
v_mov_b32_dpp v19, v24  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v19, v24  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v27
v_mov_b32_e32 v191, v28
v_mov_b32_e32 v192, v29
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v26
v_add_f32_dpp v189, v26, v26  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v27, v27  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v28, v28  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v29, v29  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v28, v28, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v29, v29, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v26, v26, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v27, v27, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v27, v28  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v26, v29  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v27, v27  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v27, v27, v27  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v26, v26  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v26, v26, v26  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v29, v190
v_add_f32_dpp v29, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v28, v189
v_add_f32_dpp v28, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v27, v27, v26  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v20, v29, v28  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v28, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v28, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v21, v191, v27, s[52:53]
v_mov_b32_dpp v22, v28  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v22, v28  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v31
v_mov_b32_e32 v191, v32
v_mov_b32_e32 v192, v33
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v30
v_add_f32_dpp v189, v30, v30  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v31, v31  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v32, v32  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v33, v33  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v32, v32, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v33, v33, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v30, v30, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v31, v31, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v31, v32  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v30, v33  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v31, v31  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v31, v31, v31  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v30, v30  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v30, v30, v30  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v33, v190
v_add_f32_dpp v33, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v32, v189
v_add_f32_dpp v32, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v31, v31, v30  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v23, v33, v32  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v32, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v32, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v24, v191, v31, s[52:53]
v_mov_b32_dpp v25, v32  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v25, v32  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v35
v_mov_b32_e32 v191, v36
v_mov_b32_e32 v192, v37
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v34
v_add_f32_dpp v189, v34, v34  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v35, v35  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v36, v36  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v37, v37  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v36, v36, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v37, v37, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v34, v34, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v35, v35, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v35, v36  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v34, v37  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v35, v35  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v35, v35, v35  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v34, v34  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v34, v34, v34  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v37, v190
v_add_f32_dpp v37, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v36, v189
v_add_f32_dpp v36, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v35, v35, v34  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v26, v37, v36  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v36, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v36, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v27, v191, v35, s[52:53]
v_mov_b32_dpp v28, v36  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v28, v36  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v39
v_mov_b32_e32 v191, v40
v_mov_b32_e32 v192, v41
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v38
v_add_f32_dpp v189, v38, v38  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v39, v39  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v40, v40  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v41, v41  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v40, v40, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v41, v41, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v38, v38, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v39, v39, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v39, v40  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v38, v41  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v39, v39  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v39, v39, v39  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v38, v38  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v38, v38, v38  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v41, v190
v_add_f32_dpp v41, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v40, v189
v_add_f32_dpp v40, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v39, v39, v38  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v29, v41, v40  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v40, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v40, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v30, v191, v39, s[52:53]
v_mov_b32_dpp v31, v40  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v31, v40  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v43
v_mov_b32_e32 v191, v44
v_mov_b32_e32 v192, v45
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v42
v_add_f32_dpp v189, v42, v42  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v43, v43  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v44, v44  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v45, v45  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v44, v44, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v45, v45, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v42, v42, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v43, v43, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v43, v44  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v42, v45  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v43, v43  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v43, v43, v43  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v42, v42  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v42, v42, v42  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v45, v190
v_add_f32_dpp v45, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v44, v189
v_add_f32_dpp v44, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v43, v43, v42  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v32, v45, v44  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v44, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v44, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v33, v191, v43, s[52:53]
v_mov_b32_dpp v34, v44  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v34, v44  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v47
v_mov_b32_e32 v191, v48
v_mov_b32_e32 v192, v49
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v46
v_add_f32_dpp v189, v46, v46  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v47, v47  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v48, v48  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v49, v49  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v48, v48, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v49, v49, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v46, v46, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v47, v47, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v47, v48  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v46, v49  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v47, v47  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v47, v47, v47  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v46, v46  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v46, v46, v46  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v49, v190
v_add_f32_dpp v49, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v48, v189
v_add_f32_dpp v48, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v47, v47, v46  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v35, v49, v48  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v48, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v48, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v36, v191, v47, s[52:53]
v_mov_b32_dpp v37, v48  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v37, v48  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v51
v_mov_b32_e32 v191, v52
v_mov_b32_e32 v192, v53
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v50
v_add_f32_dpp v189, v50, v50  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v51, v51  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v52, v52  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v53, v53  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v52, v52, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v53, v53, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v50, v50, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v51, v51, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v51, v52  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v50, v53  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v51, v51  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v51, v51, v51  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v50, v50  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v50, v50, v50  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v53, v190
v_add_f32_dpp v53, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v52, v189
v_add_f32_dpp v52, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v51, v51, v50  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v38, v53, v52  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v52, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v52, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v39, v191, v51, s[52:53]
v_mov_b32_dpp v40, v52  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v40, v52  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v55
v_mov_b32_e32 v191, v56
v_mov_b32_e32 v192, v57
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v54
v_add_f32_dpp v189, v54, v54  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v55, v55  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v56, v56  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v57, v57  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v56, v56, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v57, v57, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v54, v54, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v55, v55, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v55, v56  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v54, v57  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v55, v55  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v55, v55, v55  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v54, v54  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v54, v54, v54  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v57, v190
v_add_f32_dpp v57, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v56, v189
v_add_f32_dpp v56, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v55, v55, v54  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v41, v57, v56  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v56, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v56, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v42, v191, v55, s[52:53]
v_mov_b32_dpp v43, v56  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v43, v56  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v59
v_mov_b32_e32 v191, v60
v_mov_b32_e32 v192, v61
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v58
v_add_f32_dpp v189, v58, v58  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v59, v59  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v60, v60  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v61, v61  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v60, v60, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v61, v61, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v58, v58, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v59, v59, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v59, v60  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v58, v61  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v59, v59  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v59, v59, v59  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v58, v58  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v58, v58, v58  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v61, v190
v_add_f32_dpp v61, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v60, v189
v_add_f32_dpp v60, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v59, v59, v58  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v44, v61, v60  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v60, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v60, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v45, v191, v59, s[52:53]
v_mov_b32_dpp v46, v60  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v46, v60  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_mov_b32_e32 v190, v63
v_mov_b32_e32 v191, v64
v_mov_b32_e32 v192, v65
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v189, v62
v_add_f32_dpp v189, v62, v62  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v190, v63, v63  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v191, v64, v64  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v192, v65, v65  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v64, v64, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v65, v65, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v62, v62, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_fmac_f32_dpp v63, v63, v184  quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mov_b32_dpp v63, v64  row_mirror row_mask:0xf bank_mask:0x3
v_mov_b32_dpp v62, v65  row_mirror row_mask:0xf bank_mask:0x3
v_add_f32_dpp v190, v191, v190  row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v189, v192, v189  row_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v192, v63, v63  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v63, v63, v63  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_sub_f32_dpp v191, v62, v62  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v62, v62, v62  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_e32 v65, v190
v_add_f32_dpp v65, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v192  row_ror:12 row_mask:0xf bank_mask:0x1
v_mov_b32_dpp v191, v192  row_ror:4 row_mask:0xf bank_mask:0x8
v_mov_b32_e32 v64, v189
v_add_f32_dpp v64, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v191, v191  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0x3
v_sub_f32_dpp v192, v190, v190  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_add_f32_dpp v63, v63, v62  row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v47, v65, v64  row_half_mirror row_mask:0xf bank_mask:0xf
v_sub_f32_dpp v64, v189, v189  quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mov_b32_dpp v64, v192  row_half_mirror row_mask:0xf bank_mask:0x9
v_mov_b32_dpp v191, v191  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xc
v_cndmask_b32_e64 v48, v191, v63, s[52:53]
v_mov_b32_dpp v49, v64  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0x5
s_nop 1
v_mov_b32_dpp v49, v64  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
s_waitcnt vmcnt(0)
v_readlane_b32 s55, v187, 0
v_add_f32_e64 v2, v2, s55
v_mul_f32_e64 v189, v2, s36
v_cmp_lt_f32_e64 vcc, v2, 0
v_cndmask_b32_e32 v2, v2, v189, vcc
v_add_f32_e64 v5, v5, s55
v_mul_f32_e64 v189, v5, s36
v_cmp_lt_f32_e64 vcc, v5, 0
v_cndmask_b32_e32 v5, v5, v189, vcc
buffer_store_dword v2, v157, s[44:47], 0 offen
buffer_store_dword v5, v161, s[44:47], 0 offen
v_add_f32_e64 v3, v3, s55
v_mul_f32_e64 v189, v3, s36
v_cmp_lt_f32_e64 vcc, v3, 0
v_cndmask_b32_e32 v3, v3, v189, vcc
v_add_f32_e64 v6, v6, s55
v_mul_f32_e64 v189, v6, s36
v_cmp_lt_f32_e64 vcc, v6, 0
v_cndmask_b32_e32 v6, v6, v189, vcc
buffer_store_dword v3, v158, s[44:47], 0 offen
buffer_store_dword v6, v162, s[44:47], 0 offen
v_add_f32_e64 v4, v4, s55
v_mul_f32_e64 v189, v4, s36
v_cmp_lt_f32_e64 vcc, v4, 0
v_cndmask_b32_e32 v4, v4, v189, vcc
v_add_f32_e64 v7, v7, s55
v_mul_f32_e64 v189, v7, s36
v_cmp_lt_f32_e64 vcc, v7, 0
v_cndmask_b32_e32 v7, v7, v189, vcc
buffer_store_dword v4, v159, s[44:47], 0 offen
buffer_store_dword v7, v163, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v187, 1
v_add_f32_e64 v8, v8, s55
v_mul_f32_e64 v189, v8, s36
v_cmp_lt_f32_e64 vcc, v8, 0
v_cndmask_b32_e32 v8, v8, v189, vcc
v_add_f32_e64 v11, v11, s55
v_mul_f32_e64 v189, v11, s36
v_cmp_lt_f32_e64 vcc, v11, 0
v_cndmask_b32_e32 v11, v11, v189, vcc
buffer_store_dword v8, v157, s[44:47], 0 offen
buffer_store_dword v11, v161, s[44:47], 0 offen
v_add_f32_e64 v9, v9, s55
v_mul_f32_e64 v189, v9, s36
v_cmp_lt_f32_e64 vcc, v9, 0
v_cndmask_b32_e32 v9, v9, v189, vcc
v_add_f32_e64 v12, v12, s55
v_mul_f32_e64 v189, v12, s36
v_cmp_lt_f32_e64 vcc, v12, 0
v_cndmask_b32_e32 v12, v12, v189, vcc
buffer_store_dword v9, v158, s[44:47], 0 offen
buffer_store_dword v12, v162, s[44:47], 0 offen
v_add_f32_e64 v10, v10, s55
v_mul_f32_e64 v189, v10, s36
v_cmp_lt_f32_e64 vcc, v10, 0
v_cndmask_b32_e32 v10, v10, v189, vcc
v_add_f32_e64 v13, v13, s55
v_mul_f32_e64 v189, v13, s36
v_cmp_lt_f32_e64 vcc, v13, 0
v_cndmask_b32_e32 v13, v13, v189, vcc
buffer_store_dword v10, v159, s[44:47], 0 offen
buffer_store_dword v13, v163, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v187, 2
v_add_f32_e64 v14, v14, s55
v_mul_f32_e64 v189, v14, s36
v_cmp_lt_f32_e64 vcc, v14, 0
v_cndmask_b32_e32 v14, v14, v189, vcc
v_add_f32_e64 v17, v17, s55
v_mul_f32_e64 v189, v17, s36
v_cmp_lt_f32_e64 vcc, v17, 0
v_cndmask_b32_e32 v17, v17, v189, vcc
buffer_store_dword v14, v157, s[44:47], 0 offen
buffer_store_dword v17, v161, s[44:47], 0 offen
v_add_f32_e64 v15, v15, s55
v_mul_f32_e64 v189, v15, s36
v_cmp_lt_f32_e64 vcc, v15, 0
v_cndmask_b32_e32 v15, v15, v189, vcc
v_add_f32_e64 v18, v18, s55
v_mul_f32_e64 v189, v18, s36
v_cmp_lt_f32_e64 vcc, v18, 0
v_cndmask_b32_e32 v18, v18, v189, vcc
buffer_store_dword v15, v158, s[44:47], 0 offen
buffer_store_dword v18, v162, s[44:47], 0 offen
v_add_f32_e64 v16, v16, s55
v_mul_f32_e64 v189, v16, s36
v_cmp_lt_f32_e64 vcc, v16, 0
v_cndmask_b32_e32 v16, v16, v189, vcc
v_add_f32_e64 v19, v19, s55
v_mul_f32_e64 v189, v19, s36
v_cmp_lt_f32_e64 vcc, v19, 0
v_cndmask_b32_e32 v19, v19, v189, vcc
buffer_store_dword v16, v159, s[44:47], 0 offen
buffer_store_dword v19, v163, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v187, 3
v_add_f32_e64 v20, v20, s55
v_mul_f32_e64 v189, v20, s36
v_cmp_lt_f32_e64 vcc, v20, 0
v_cndmask_b32_e32 v20, v20, v189, vcc
v_add_f32_e64 v23, v23, s55
v_mul_f32_e64 v189, v23, s36
v_cmp_lt_f32_e64 vcc, v23, 0
v_cndmask_b32_e32 v23, v23, v189, vcc
buffer_store_dword v20, v157, s[44:47], 0 offen
buffer_store_dword v23, v161, s[44:47], 0 offen
v_add_f32_e64 v21, v21, s55
v_mul_f32_e64 v189, v21, s36
v_cmp_lt_f32_e64 vcc, v21, 0
v_cndmask_b32_e32 v21, v21, v189, vcc
v_add_f32_e64 v24, v24, s55
v_mul_f32_e64 v189, v24, s36
v_cmp_lt_f32_e64 vcc, v24, 0
v_cndmask_b32_e32 v24, v24, v189, vcc
buffer_store_dword v21, v158, s[44:47], 0 offen
buffer_store_dword v24, v162, s[44:47], 0 offen
v_add_f32_e64 v22, v22, s55
v_mul_f32_e64 v189, v22, s36
v_cmp_lt_f32_e64 vcc, v22, 0
v_cndmask_b32_e32 v22, v22, v189, vcc
v_add_f32_e64 v25, v25, s55
v_mul_f32_e64 v189, v25, s36
v_cmp_lt_f32_e64 vcc, v25, 0
v_cndmask_b32_e32 v25, v25, v189, vcc
buffer_store_dword v22, v159, s[44:47], 0 offen
buffer_store_dword v25, v163, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
s_lshl_b32 s52, s67, 2
s_add_u32 s44, s44, s52
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 4
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v187, 8
v_add_f32_e64 v26, v26, s55
v_mul_f32_e64 v189, v26, s36
v_cmp_lt_f32_e64 vcc, v26, 0
v_cndmask_b32_e32 v26, v26, v189, vcc
v_add_f32_e64 v29, v29, s55
v_mul_f32_e64 v189, v29, s36
v_cmp_lt_f32_e64 vcc, v29, 0
v_cndmask_b32_e32 v29, v29, v189, vcc
buffer_store_dword v26, v157, s[44:47], 0 offen
buffer_store_dword v29, v161, s[44:47], 0 offen
v_add_f32_e64 v27, v27, s55
v_mul_f32_e64 v189, v27, s36
v_cmp_lt_f32_e64 vcc, v27, 0
v_cndmask_b32_e32 v27, v27, v189, vcc
v_add_f32_e64 v30, v30, s55
v_mul_f32_e64 v189, v30, s36
v_cmp_lt_f32_e64 vcc, v30, 0
v_cndmask_b32_e32 v30, v30, v189, vcc
buffer_store_dword v27, v158, s[44:47], 0 offen
buffer_store_dword v30, v162, s[44:47], 0 offen
v_add_f32_e64 v28, v28, s55
v_mul_f32_e64 v189, v28, s36
v_cmp_lt_f32_e64 vcc, v28, 0
v_cndmask_b32_e32 v28, v28, v189, vcc
v_add_f32_e64 v31, v31, s55
v_mul_f32_e64 v189, v31, s36
v_cmp_lt_f32_e64 vcc, v31, 0
v_cndmask_b32_e32 v31, v31, v189, vcc
buffer_store_dword v28, v159, s[44:47], 0 offen
buffer_store_dword v31, v163, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v187, 9
v_add_f32_e64 v32, v32, s55
v_mul_f32_e64 v189, v32, s36
v_cmp_lt_f32_e64 vcc, v32, 0
v_cndmask_b32_e32 v32, v32, v189, vcc
v_add_f32_e64 v35, v35, s55
v_mul_f32_e64 v189, v35, s36
v_cmp_lt_f32_e64 vcc, v35, 0
v_cndmask_b32_e32 v35, v35, v189, vcc
buffer_store_dword v32, v157, s[44:47], 0 offen
buffer_store_dword v35, v161, s[44:47], 0 offen
v_add_f32_e64 v33, v33, s55
v_mul_f32_e64 v189, v33, s36
v_cmp_lt_f32_e64 vcc, v33, 0
v_cndmask_b32_e32 v33, v33, v189, vcc
v_add_f32_e64 v36, v36, s55
v_mul_f32_e64 v189, v36, s36
v_cmp_lt_f32_e64 vcc, v36, 0
v_cndmask_b32_e32 v36, v36, v189, vcc
buffer_store_dword v33, v158, s[44:47], 0 offen
buffer_store_dword v36, v162, s[44:47], 0 offen
v_add_f32_e64 v34, v34, s55
v_mul_f32_e64 v189, v34, s36
v_cmp_lt_f32_e64 vcc, v34, 0
v_cndmask_b32_e32 v34, v34, v189, vcc
v_add_f32_e64 v37, v37, s55
v_mul_f32_e64 v189, v37, s36
v_cmp_lt_f32_e64 vcc, v37, 0
v_cndmask_b32_e32 v37, v37, v189, vcc
buffer_store_dword v34, v159, s[44:47], 0 offen
buffer_store_dword v37, v163, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v187, 10
v_add_f32_e64 v38, v38, s55
v_mul_f32_e64 v189, v38, s36
v_cmp_lt_f32_e64 vcc, v38, 0
v_cndmask_b32_e32 v38, v38, v189, vcc
v_add_f32_e64 v41, v41, s55
v_mul_f32_e64 v189, v41, s36
v_cmp_lt_f32_e64 vcc, v41, 0
v_cndmask_b32_e32 v41, v41, v189, vcc
buffer_store_dword v38, v157, s[44:47], 0 offen
buffer_store_dword v41, v161, s[44:47], 0 offen
v_add_f32_e64 v39, v39, s55
v_mul_f32_e64 v189, v39, s36
v_cmp_lt_f32_e64 vcc, v39, 0
v_cndmask_b32_e32 v39, v39, v189, vcc
v_add_f32_e64 v42, v42, s55
v_mul_f32_e64 v189, v42, s36
v_cmp_lt_f32_e64 vcc, v42, 0
v_cndmask_b32_e32 v42, v42, v189, vcc
buffer_store_dword v39, v158, s[44:47], 0 offen
buffer_store_dword v42, v162, s[44:47], 0 offen
v_add_f32_e64 v40, v40, s55
v_mul_f32_e64 v189, v40, s36
v_cmp_lt_f32_e64 vcc, v40, 0
v_cndmask_b32_e32 v40, v40, v189, vcc
v_add_f32_e64 v43, v43, s55
v_mul_f32_e64 v189, v43, s36
v_cmp_lt_f32_e64 vcc, v43, 0
v_cndmask_b32_e32 v43, v43, v189, vcc
buffer_store_dword v40, v159, s[44:47], 0 offen
buffer_store_dword v43, v163, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s55, v187, 11
v_add_f32_e64 v44, v44, s55
v_mul_f32_e64 v189, v44, s36
v_cmp_lt_f32_e64 vcc, v44, 0
v_cndmask_b32_e32 v44, v44, v189, vcc
v_add_f32_e64 v47, v47, s55
v_mul_f32_e64 v189, v47, s36
v_cmp_lt_f32_e64 vcc, v47, 0
v_cndmask_b32_e32 v47, v47, v189, vcc
buffer_store_dword v44, v157, s[44:47], 0 offen
buffer_store_dword v47, v161, s[44:47], 0 offen
v_add_f32_e64 v45, v45, s55
v_mul_f32_e64 v189, v45, s36
v_cmp_lt_f32_e64 vcc, v45, 0
v_cndmask_b32_e32 v45, v45, v189, vcc
v_add_f32_e64 v48, v48, s55
v_mul_f32_e64 v189, v48, s36
v_cmp_lt_f32_e64 vcc, v48, 0
v_cndmask_b32_e32 v48, v48, v189, vcc
buffer_store_dword v45, v158, s[44:47], 0 offen
buffer_store_dword v48, v162, s[44:47], 0 offen
v_add_f32_e64 v46, v46, s55
v_mul_f32_e64 v189, v46, s36
v_cmp_lt_f32_e64 vcc, v46, 0
v_cndmask_b32_e32 v46, v46, v189, vcc
v_add_f32_e64 v49, v49, s55
v_mul_f32_e64 v189, v49, s36
v_cmp_lt_f32_e64 vcc, v49, 0
v_cndmask_b32_e32 v49, v49, v189, vcc
buffer_store_dword v46, v159, s[44:47], 0 offen
buffer_store_dword v49, v163, s[44:47], 0 offen
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
s_add_u32 s48, s48, 0x80
s_addc_u32 s49, s49, 0
s_sub_u32 s50, s50, 0x80
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
s_mul_i32 s94, s60, s61
s_mul_i32 s94, s94, s13
s_add_u32 s52, s93, s92
s_cmp_lt_i32 s52, 0
s_cbranch_scc0 270
v_and_b32_e32 v157, 0x7f, v1
v_lshrrev_b32_e32 v157, 1, v157
v_bfi_b32 v157, 1, v1, v157
v_and_b32_e64 v158, v1, 2
v_mad_u32_u24 v157, v158, 16, v157
v_lshlrev_b32_e32 v157, 2, v157
v_add_co_u32_e64 v157, vcc, v157, s97
v_and_b32_e32 v158, 3, v1
v_lshlrev_b32_e32 v158, 2, v158
v_add_co_u32_e64 v158, vcc, v158, s97
ds_read_b32 v191, v158 offset:256
ds_read_b32 v157, v157
s_add_u32 s97, s97, 0x18c
s_cmp_eq_u32 s97, 0xffc0
s_cselect_b32 s97, 0xc1e0, s97
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s95, v157
v_readlane_b32 s54, v191, 0
s_bitcmp1_b32 s54, 18
s_cbranch_scc1 245
v_readlane_b32 s52, v191, 1
v_readlane_b32 s53, v191, 2
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
s_lshl_b32 s55, s93, 2
s_add_u32 s48, s34, s55
s_addc_u32 s49, s35, 0
s_lshl_b32 s56, s52, 2
s_sub_u32 s50, s56, s55
s_cselect_b32 s51, 0, s51
s_sub_u32 s93, s52, s53
s_sub_u32 s93, s93, 1
s_sub_u32 s93, s93, s92
s_cselect_b32 s47, 0, s47
v_bfe_u32 v189, v157, 16, 16
v_bfe_u32 v190, v157, 0, 16
v_and_b32_e64 v191, v1, 7
v_sub_co_u32_e64 v192, vcc, 7, v191
v_min_u32_e32 v191, v191, v192
v_bfe_u32 v192, v191, 1, 1
v_bfe_u32 v191, v191, 0, 1
v_mov_b32_dpp v189, v189  quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v190, v190  quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf
v_add_co_u32_e64 v189, vcc, v189, v192
v_add_co_u32_e64 v190, vcc, v190, v191
v_mov_b32_dpp v191, v157  quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[52:53], v191, s12
v_sub_co_u32_e64 v191, vcc, v191, s95
v_mul_lo_u32 v191, v191, s66
v_xor_b32_dpp v192, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v192, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v164, v1, v1  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v164, v1, v1  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_add_co_u32_e64 v164, vcc, v190, v164
v_add_co_u32_e64 v192, vcc, v189, v192
v_mad_i32_i24 v161, v192, s33, v164
v_lshlrev_b32_e32 v161, 2, v161
v_add_co_u32_e64 v161, vcc, v161, v191
v_cmp_ge_u32_e64 s[56:57], v164, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v192, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v161, v161, -1, s[56:57]
v_xor_b32_dpp v192, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v192, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v164, v1, v1  quad_perm:[1,1,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32_e64 v164, vcc, v190, v164
v_add_co_u32_e64 v192, vcc, v189, v192
v_mad_i32_i24 v162, v192, s33, v164
v_lshlrev_b32_e32 v162, 2, v162
v_add_co_u32_e64 v162, vcc, v162, v191
v_cmp_ge_u32_e64 s[56:57], v164, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v192, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v162, v162, -1, s[56:57]
v_xor_b32_dpp v192, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v192, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v164, v1, v1  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v164, v1, v1  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xa
v_add_co_u32_e64 v164, vcc, v190, v164
v_add_co_u32_e64 v192, vcc, v189, v192
v_mad_i32_i24 v163, v192, s33, v164
v_lshlrev_b32_e32 v163, 2, v163
v_add_co_u32_e64 v163, vcc, v163, v191
v_cmp_ge_u32_e64 s[56:57], v164, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v192, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v163, v163, -1, s[56:57]
v_bfe_u32 v189, v157, 16, 16
v_bfe_u32 v190, v157, 0, 16
v_and_b32_e64 v191, v1, 7
v_sub_co_u32_e64 v192, vcc, 7, v191
v_min_u32_e32 v191, v191, v192
v_bfe_u32 v192, v191, 1, 1
v_bfe_u32 v191, v191, 0, 1
v_mov_b32_dpp v189, v189  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v190, v190  quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_add_co_u32_e64 v189, vcc, v189, v192
v_add_co_u32_e64 v190, vcc, v190, v191
v_mov_b32_dpp v191, v157  quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[52:53], v191, s12
v_sub_co_u32_e64 v191, vcc, v191, s95
v_mul_lo_u32 v191, v191, s66
v_xor_b32_dpp v192, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v192, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v160, v1, v1  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v160, v1, v1  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xa
v_add_co_u32_e64 v160, vcc, v190, v160
v_add_co_u32_e64 v192, vcc, v189, v192
v_mad_i32_i24 v157, v192, s33, v160
v_lshlrev_b32_e32 v157, 2, v157
v_add_co_u32_e64 v157, vcc, v157, v191
v_cmp_ge_u32_e64 s[56:57], v160, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v192, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v157, v157, -1, s[56:57]
v_xor_b32_dpp v192, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v192, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v160, v1, v1  quad_perm:[1,1,2,2] row_mask:0xf bank_mask:0xf
v_add_co_u32_e64 v160, vcc, v190, v160
v_add_co_u32_e64 v192, vcc, v189, v192
v_mad_i32_i24 v158, v192, s33, v160
v_lshlrev_b32_e32 v158, 2, v158
v_add_co_u32_e64 v158, vcc, v158, v191
v_cmp_ge_u32_e64 s[56:57], v160, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v192, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v158, v158, -1, s[56:57]
v_xor_b32_dpp v192, v1, v1  quad_perm:[0,1,3,2] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v192, v1, v1  quad_perm:[1,0,2,3] row_mask:0xf bank_mask:0xa
v_xor_b32_dpp v160, v1, v1  quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf
v_xor_b32_dpp v160, v1, v1  quad_perm:[0,0,2,2] row_mask:0xf bank_mask:0xa
v_add_co_u32_e64 v160, vcc, v190, v160
v_add_co_u32_e64 v192, vcc, v189, v192
v_mad_i32_i24 v159, v192, s33, v160
v_lshlrev_b32_e32 v159, 2, v159
v_add_co_u32_e64 v159, vcc, v159, v191
v_cmp_ge_u32_e64 s[56:57], v160, s33
s_or_b64 s[56:57], s[56:57], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v192, s32
s_or_b64 s[56:57], s[56:57], s[54:55]
v_cndmask_b32_e64 v159, v159, -1, s[56:57]
v_and_b32_e64 v187, v1, 63
v_lshlrev_b32_e32 v187, 2, v187
s_barrier
buffer_load_dword v187, v187, s[48:51], 0 offen
s_branch 62853
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
