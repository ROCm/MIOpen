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

v_mov_b32_e32 v0, v0
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_mov_b32 s3, 0
v_mov_b32_e32 v104, 0
s_mov_b32 m0, 0x1ffff
s_mov_b32 s97, 0xc1e0
s_mov_b32 s96, 0xc1e0
s_mov_b32 s91, 0
v_lshlrev_b32_e32 v107, 2, v0
v_add_co_u32_e32 v107, vcc, 0xffc0, v107
v_cmp_ge_u32_e32 vcc, 12, v0
s_cbranch_vccz 5
v_mov_b32_e32 v106, 0
v_cndmask_b32_e32 v107, -1, v107, vcc
ds_write_b32 v107, v106
s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
s_barrier
v_readfirstlane_b32 s52, v0
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
v_cmp_le_u32_e32 vcc, 0x100, v0
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
v_subb_co_u32_e32 v3, vcc, 32, v4, vcc
v_cvt_f32_ubyte0_e32 v4, v5
v_fma_f32 v6, v6, v2, -1.0
v_fma_f32 v6, v4, v2, v6
v_madak_f32 v6, v6, v2, 0x9f000000
v_mul_f32_e32 v6, 0x5f800000, v6
v_mov_b32_e32 v4, 0
v_cvt_flr_i32_f32_e64 v6, -v6
v_lshl_add_u32 v2, v2, 9, v6
v_mad_u64_u32 v[4:5], vcc, v5, v2, v[4:5]
v_subb_co_u32_e64 v2, vcc, v2, -1, vcc
v_mul_hi_u32 v4, s8, v2
v_add_co_u32_e64 v2, vcc, v4, s8
v_addc_co_u32_e64 v4, vcc, 0, 0, vcc
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
v_cmp_le_u32_e32 vcc, 0x100, v0
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
s_and_b32 s44, 1, s30
s_addc_u32 s44, s32, 1
s_ashr_i32 s44, s44, 1
s_add_u32 s42, s44, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s42
v_readfirstlane_b32 s42, v2
s_andn2_b32 s44, 1, s31
s_addc_u32 s44, s33, 1
s_ashr_i32 s44, s44, 1
s_add_u32 s43, s44, 1
v_mov_b32_e32 v2, 0x80000000
v_mul_hi_u32 v2, v2, s43
v_readfirstlane_b32 s43, v2
s_sub_u32 s75, 0, s43
s_sub_u32 s74, 0, s42
s_add_u32 s60, s28, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s60
v_readfirstlane_b32 s60, v2
s_add_u32 s61, s29, 2
v_mov_b32_e32 v2, 0x55555556
v_mul_hi_u32 v2, v2, s61
v_readfirstlane_b32 s61, v2
v_mad_i32_i24 v2, 3, s60, -2
v_sub_co_u32_e64 v2, vcc, v2, s28
v_addc_co_u32_e64 v2, vcc, 0, 0, vcc
v_readfirstlane_b32 s44, v2
s_and_b32 s44, s44, 1
s_and_b32 s44, s44, s60
s_add_u32 s60, s60, s44
v_readfirstlane_b32 s45, v0
s_and_b32 s48, s45, 64
s_cselect_b32 s48, 0x80000, 0
s_or_b32 s18, s18, s48
s_lshl_b32 s69, s68, 1
s_mov_b64 s[70:71], 0
s_bitset1_b32 s18, 23
s_mov_b32 s69, s68
s_mov_b32 s70, s68
s_mov_b32 s71, 0
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
s_add_u32 s70, s70, s69
s_addc_u32 s71, s71, 0
v_bfe_u32 v3, v0, 2, 6
v_lshrrev_b32_e32 v99, 1, v3
s_bitcmp0_b32 s45, 8
s_cselect_b32 s48, 0x1000000, 0
s_or_b32 s48, s48, 0x100000
s_and_b32 s48, s18, s48
s_cselect_b32 s48, 0, 15
v_bfi_b32 v99, s48, v3, v99
v_bfe_u32 v3, s45, 8, 1
v_xor_b32_e64 v3, v3, 1
v_lshrrev_b32_e32 v99, v3, v99
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
v_and_b32_e32 v6, 3, v0
v_cmp_eq_u32_e32 vcc, 2, v6
v_cndmask_b32_e32 v4, v4, v5, vcc
v_cmp_eq_u32_e32 vcc, 1, v6
v_cndmask_b32_e32 v7, 0, v99, vcc
s_bitcmp1_b32 s18, 20
s_cbranch_scc0 4
v_add_co_u32_e64 v5, vcc, v99, 8
v_cmp_eq_u32_e32 vcc, 0, v6
v_cndmask_b32_e32 v7, v7, v5, vcc
v_cmp_eq_u32_e64 s[46:47], 3, v6
v_bfe_u32 v97, v7, 0, 5
v_mad_u32_u24 v97, v4, 32, v97
v_ffbh_u32_e32 v10, s43
v_lshlrev_b32_e64 v11, v10, s43
v_and_b32_e32 v9, 0xffffff00, v11
v_cmp_eq_u32_e32 vcc, 0x80000000, v11
v_cvt_f32_u32_e32 v9, v9
v_rcp_f32_e32 v98, v9
v_subb_co_u32_e32 v8, vcc, 32, v10, vcc
v_cvt_f32_ubyte0_e32 v10, v11
v_fma_f32 v9, v9, v98, -1.0
v_fma_f32 v9, v10, v98, v9
v_madak_f32 v9, v9, v98, 0x9f000000
v_mul_f32_e32 v9, 0x5f800000, v9
v_mov_b32_e32 v10, 0
v_cvt_flr_i32_f32_e64 v9, -v9
v_lshl_add_u32 v98, v98, 9, v9
v_mad_u64_u32 v[10:11], vcc, v11, v98, v[10:11]
v_subb_co_u32_e64 v98, vcc, v98, -1, vcc
v_mul_hi_u32 v10, v97, v98
v_add_co_u32_e32 v98, vcc, v10, v97
v_addc_co_u32_e64 v10, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v8
v_cndmask_b32_e32 v98, v98, v10, vcc
v_alignbit_b32 v98, v10, v98, v8
v_mad_i32_i24 v96, v98, s75, v97
v_lshrrev_b32_e32 v97, 5, v7
v_mad_u32_u24 v97, v98, 1, v97
v_cndmask_b32_e64 v97, v97, 1, s[46:47]
v_ffbh_u32_e32 v10, s42
v_lshlrev_b32_e64 v11, v10, s42
v_and_b32_e32 v9, 0xffffff00, v11
v_cmp_eq_u32_e32 vcc, 0x80000000, v11
v_cvt_f32_u32_e32 v9, v9
v_rcp_f32_e32 v98, v9
v_subb_co_u32_e32 v8, vcc, 32, v10, vcc
v_cvt_f32_ubyte0_e32 v10, v11
v_fma_f32 v9, v9, v98, -1.0
v_fma_f32 v9, v10, v98, v9
v_madak_f32 v9, v9, v98, 0x9f000000
v_mul_f32_e32 v9, 0x5f800000, v9
v_mov_b32_e32 v10, 0
v_cvt_flr_i32_f32_e64 v9, -v9
v_lshl_add_u32 v98, v98, 9, v9
v_mad_u64_u32 v[10:11], vcc, v11, v98, v[10:11]
v_subb_co_u32_e64 v98, vcc, v98, -1, vcc
v_mul_hi_u32 v10, v97, v98
v_add_co_u32_e32 v98, vcc, v10, v97
v_addc_co_u32_e64 v10, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v8
v_cndmask_b32_e32 v98, v98, v10, vcc
v_alignbit_b32 v98, v10, v98, v8
v_mad_i32_i24 v97, v98, s74, v97
v_readlane_b32 s76, v96, 2
v_readlane_b32 s77, v97, 2
v_readlane_b32 s78, v98, 2
v_readlane_b32 s79, v97, 3
v_readlane_b32 s80, v98, 3
v_add_co_u32_e64 v96, vcc, v96, s75
v_add_co_u32_e64 v97, vcc, v97, s74
v_mov_b32_dpp v98, v98 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v96, v96 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v97, v97 quad_perm:[1,1,0,0] row_mask:0xf bank_mask:0xf
s_mov_b32 s42, 0x80000000
s_mov_b32 s43, 0x20000
s_mov_b32 s46, 0x80000000
s_mov_b32 s47, 0x20000
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 5
v_xor_b32_dpp v100, v0, v0 quad_perm:[1,3,2,2] row_mask:0xf bank_mask:0xf
v_subrev_co_u32_e32 v100, vcc, 1, v100
v_cvt_f32_i32_e32 v100, v100
s_branch 4
v_xor_b32_dpp v100, v0, v0 quad_perm:[2,1,0,1] row_mask:0xf bank_mask:0xf
v_sub_co_u32_e32 v100, vcc, 1, v100
v_cvt_f32_i32_e32 v100, v100
v_mov_b32_e32 v101, 1
v_xor_b32_dpp v101, v0, v0 quad_perm:[2,3,2,3] row_mask:0xf bank_mask:0x4
v_xor_b32_dpp v101, v0, v0 quad_perm:[0,1,0,1] row_mask:0xf bank_mask:0x8
v_subrev_co_u32_e32 v101, vcc, 1, v101
v_mov_b32_e32 v102, 1
v_xor_b32_dpp v102, v0, v0 quad_perm:[0,3,2,1] row_mask:0xf bank_mask:0x2
v_xor_b32_dpp v102, v0, v0 quad_perm:[2,1,0,3] row_mask:0xf bank_mask:0x4
v_subrev_co_u32_e32 v102, vcc, 1, v102
v_cvt_f32_i32_e32 v101, v101
v_cvt_f32_i32_e32 v102, v102
v_lshrrev_b32_e64 v106, 2, s92
v_and_b32_e32 v107, 3, v0
v_bfe_u32 v108, v0, 4, 3
v_mad_u32_u24 v95, v108, 4, v107
v_lshlrev_b32_e32 v95, 4, v95
v_mad_u32_u24 v90, v106, 4, v107
v_lshlrev_b32_e32 v90, 4, v90
v_bfe_u32 v106, v0, 2, 2
v_and_b32_e32 v107, 1, v106
v_mad_u32_u24 v109, v106, 16, v107
v_lshlrev_b32_e32 v109, 6, v109
v_xor_b32_e32 v90, v90, v109
v_mul_u32_u24_e32 v109, 0x400, v106
v_xor_b32_e32 v95, v95, v109
s_lshr_b32 s92, s92, 1
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 50
s_and_b32 s53, s18, 0x1100000
s_addc_u32 s53, 0, 0
v_lshrrev_b32_e32 v109, 1, v0
s_mul_i32 s52, 60, s53
s_sub_u32 s52, 63, s52
v_bfi_b32 v109, s52, v0, v109
v_and_b32_e32 v106, 1, v109
v_bfe_u32 v107, v109, 1, 1
v_xor_b32_e32 v106, v106, v107
v_bfe_u32 v108, v109, 3, 1
v_mad_u32_u24 v107, v107, 2, v108
v_mul_u32_u24_e32 v106, 0x118, v106
v_bfe_u32 v108, v109, 2, 1
v_mad_u32_u24 v107, v107, 2, v106
v_xor_b32_e32 v107, v107, v108
v_and_b32_e32 v108, 0xf0, v109
v_xor_b32_e32 v107, v107, v108
s_mul_i32 s52, 4, s53
s_sub_u32 s52, 6, s52
v_bfe_u32 v109, v0, s52, 1
v_mul_u32_u24_e32 v109, 0x1040, v109
v_xor_b32_e32 v92, 0x314, v107
v_xor_b32_e32 v93, 0x31c, v107
v_xor_b32_e32 v94, 8, v107
s_bitcmp1_b32 s18, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e32 v91, v107, v94, vcc
v_cndmask_b32_e32 v94, v94, v107, vcc
v_mad_u32_u24 v91, 4, v91, v109
v_mad_u32_u24 v92, 4, v92, v109
v_mad_u32_u24 v93, 4, v93, v109
v_mad_u32_u24 v94, 4, v94, v109
s_branch 44
s_bfe_u32 s53, s18, 0x10014
v_lshrrev_b32_e32 v109, 1, v0
s_mul_i32 s52, 60, s53
s_sub_u32 s52, 63, s52
v_bfi_b32 v109, s52, v0, v109
v_and_b32_e32 v106, 1, v109
v_bfe_u32 v107, v109, 1, 1
v_bfe_u32 v108, v109, 3, 1
v_xor_b32_e32 v106, v106, v107
v_mad_u32_u24 v107, v107, 2, v108
v_mul_u32_u24_e32 v106, 0x109, v106
v_bfe_u32 v108, v109, 2, 1
v_mad_u32_u24 v107, v107, 2, v106
v_xor_b32_e32 v107, v107, v108
v_and_b32_e32 v108, 0xf0, v109
v_or_b32_e32 v107, v107, v108
s_mul_i32 s52, 4, s53
s_sub_u32 s52, 6, s52
v_bfe_u32 v109, v0, s52, 1
v_mul_u32_u24_e32 v109, 0x1040, v109
v_mad_u32_u24 v91, 4, v107, v109
v_xor_b32_e32 v92, 0x307, v107
v_mad_u32_u24 v92, 4, v92, v109
v_xor_b32_e32 v93, 0x30f, v107
v_mad_u32_u24 v93, 4, v93, v109
v_xor_b32_e32 v94, 8, v107
v_mad_u32_u24 v94, 4, v94, v109
v_subrev_co_u32_e32 v96, vcc, s76, v96
v_mov_b32_e32 v107, s75
v_cmp_lt_i32_e32 vcc, v96, v107
v_subb_co_u32_e64 v106, vcc, 0, 0, vcc
v_mad_i32_i24 v96, v106, s75, v96
v_mad_i32_i24 v98, v106, s80, v98
v_mad_i32_i24 v97, v106, s79, v97
v_mov_b32_e32 v107, s74
v_cmp_lt_i32_e32 vcc, v97, v107
v_subb_co_u32_e64 v106, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v106
v_mad_i32_i24 v97, v106, v107, v97
v_subrev_co_u32_e32 v97, vcc, s77, v97
v_cmp_lt_i32_e32 vcc, v97, v107
v_subb_co_u32_e64 v106, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v106
v_mad_i32_i24 v97, v106, s74, v97
v_subrev_co_u32_e32 v98, vcc, s78, v98
s_mov_b32 s62, 0
s_mov_b32 s63, s28
s_mov_b32 s64, 1
s_mov_b32 s84, 0
s_mov_b32 s85, s16
s_mov_b32 s83, s85
s_sub_u32 s93, -1, s92
s_sub_u32 s93, s93, 16
s_bitset1_b32 s18, 21
s_mov_b32 s47, 0
s_mov_b32 s51, 0
v_add_co_u32_e32 v106, vcc, 2, v0
v_bfe_u32 v106, v106, 2, 1
v_cmp_ne_u32_e64 vcc, v106, 1
s_mov_b64 s[10:11], vcc
s_mov_b32 s94, 17
s_mov_b32 s82, 0
s_bitset1_b32 s18, 26
s_call_b64 s[38:39], 1783
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccnz 65
s_branch 881
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_subrev_f32_e64 v66, v68, v66 div:2
v_subrev_f32_e64 v69, v67, v69 div:2
v_add_f32_e64 v67, v68, v67 div:2
v_mad_f32 v68, v68, 1.0, -v67
s_setprio 0
s_nop 0
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:29440
ds_read_b128 v[42:45], v90 offset:28928
ds_read_b128 v[46:49], v90 offset:29056
ds_write_b32 v91, v62
ds_write_b32 v92, v63
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v58, v82, s[40:43], 0 offen
buffer_load_dword v60, v84, s[40:43], 0 offen
buffer_load_dword v59, v83, s[40:43], 0 offen
buffer_load_dword v61, v85, s[40:43], 0 offen
s_add_u32 s91, s91, 0x200
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 1
s_call_b64 s[38:39], 1640
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_mac_f32_dpp v66, v66, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v67, v67, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v68, v68, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v69, v69, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_setprio 0
s_nop 0
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
s_nop 0
ds_read_b128 v[38:41], v95 offset:33536
ds_read_b128 v[50:53], v90 offset:33024
ds_read_b128 v[54:57], v90 offset:33152
ds_write_b32 v93, v68 offset:8256
ds_write_b32 v94, v69 offset:8256
s_setprio 1
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
ds_append v105 offset:65472
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 3
s_call_b64 s[38:39], 1578
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_subrev_f32_e64 v70, v72, v70 div:2
v_subrev_f32_e64 v73, v71, v73 div:2
v_add_f32_e64 v71, v72, v71 div:2
v_mad_f32 v72, v72, 1.0, -v71
s_setprio 0
s_nop 0
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:37696
ds_read_b128 v[42:45], v90 offset:37184
ds_read_b128 v[46:49], v90 offset:37312
ds_write_b32 v91, v66 offset:8256
ds_write_b32 v92, v67 offset:8256
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v62, v82, s[40:43], 0 offen
buffer_load_dword v64, v84, s[40:43], 0 offen
buffer_load_dword v63, v83, s[40:43], 0 offen
buffer_load_dword v65, v85, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc0
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 1
s_call_b64 s[38:39], 1504
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_dpp v70, v70, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v71, v71, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v72, v72, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v73, v73, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_setprio 0
s_nop 0
v_mac_f32_e32 v33, v41, v57
s_barrier
s_nop 0
ds_read_b128 v[38:41], v95 offset:41792
ds_read_b128 v[50:53], v90 offset:41280
ds_read_b128 v[54:57], v90 offset:41408
ds_write_b32 v93, v72 offset:16512
ds_write_b32 v94, v73 offset:16512
s_setprio 1
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 4
s_call_b64 s[38:39], 1443
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_subrev_f32_e64 v74, v76, v74 div:2
v_subrev_f32_e64 v77, v75, v77 div:2
v_add_f32_e64 v75, v76, v75 div:2
v_mad_f32 v76, v76, 1.0, -v75
s_setprio 0
s_nop 0
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:45952
ds_read_b128 v[42:45], v90 offset:45440
ds_read_b128 v[46:49], v90 offset:45568
ds_write_b32 v91, v70 offset:16512
ds_write_b32 v92, v71 offset:16512
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v66, v82, s[40:43], 0 offen
buffer_load_dword v68, v84, s[40:43], 0 offen
buffer_load_dword v67, v83, s[40:43], 0 offen
buffer_load_dword v69, v85, s[40:43], 0 offen
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 3
s_call_b64 s[38:39], 1370
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_mac_f32_dpp v74, v74, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v75, v75, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v76, v76, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v77, v77, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_setprio 0
s_nop 0
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
s_nop 0
ds_read_b128 v[38:41], v95 offset:512
ds_read_b128 v[50:53], v90
ds_read_b128 v[54:57], v90 offset:128
ds_write_b32 v93, v76 offset:24768
ds_write_b32 v94, v77 offset:24768
s_setprio 1
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
ds_append v105 offset:65476
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 3
s_call_b64 s[38:39], 1306
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_subrev_f32_e64 v78, v80, v78 div:2
v_subrev_f32_e64 v81, v79, v81 div:2
v_add_f32_e64 v79, v80, v79 div:2
v_mad_f32 v80, v80, 1.0, -v79
s_setprio 0
s_nop 0
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:4672
ds_read_b128 v[42:45], v90 offset:4160
ds_read_b128 v[46:49], v90 offset:4288
ds_write_b32 v91, v74 offset:24768
ds_write_b32 v92, v75 offset:24768
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v70, v82, s[40:43], 0 offen
buffer_load_dword v72, v84, s[40:43], 0 offen
buffer_load_dword v71, v83, s[40:43], 0 offen
buffer_load_dword v73, v85, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc4
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 1
s_call_b64 s[38:39], 1232
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_dpp v78, v78, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v79, v79, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v80, v80, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v81, v81, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_setprio 0
s_nop 0
v_mac_f32_e32 v33, v41, v57
s_barrier
s_nop 0
ds_read_b128 v[38:41], v95 offset:8768
ds_read_b128 v[50:53], v90 offset:8256
ds_read_b128 v[54:57], v90 offset:8384
ds_write_b32 v93, v80 offset:33024
ds_write_b32 v94, v81 offset:33024
s_setprio 1
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 4
s_call_b64 s[38:39], 1171
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_subrev_f32_e64 v58, v60, v58 div:2
v_subrev_f32_e64 v61, v59, v61 div:2
v_add_f32_e64 v59, v60, v59 div:2
v_mad_f32 v60, v60, 1.0, -v59
s_setprio 0
s_nop 0
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:12928
ds_read_b128 v[42:45], v90 offset:12416
ds_read_b128 v[46:49], v90 offset:12544
ds_write_b32 v91, v78 offset:33024
ds_write_b32 v92, v79 offset:33024
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v74, v82, s[40:43], 0 offen
buffer_load_dword v76, v84, s[40:43], 0 offen
buffer_load_dword v75, v83, s[40:43], 0 offen
buffer_load_dword v77, v85, s[40:43], 0 offen
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 3
s_call_b64 s[38:39], 1098
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_mac_f32_dpp v58, v58, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v59, v59, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v60, v60, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v61, v61, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_setprio 0
s_nop 0
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
s_nop 0
ds_read_b128 v[38:41], v95 offset:17024
ds_read_b128 v[50:53], v90 offset:16512
ds_read_b128 v[54:57], v90 offset:16640
ds_write_b32 v93, v60 offset:41280
ds_write_b32 v94, v61 offset:41280
s_setprio 1
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
ds_append v105 offset:65480
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 3
s_call_b64 s[38:39], 1034
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_subrev_f32_e64 v62, v64, v62 div:2
v_subrev_f32_e64 v65, v63, v65 div:2
v_add_f32_e64 v63, v64, v63 div:2
v_mad_f32 v64, v64, 1.0, -v63
s_setprio 0
s_nop 0
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:21184
ds_read_b128 v[42:45], v90 offset:20672
ds_read_b128 v[46:49], v90 offset:20800
ds_write_b32 v91, v58 offset:41280
ds_write_b32 v92, v59 offset:41280
s_setprio 1
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v78, v82, s[40:43], 0 offen
buffer_load_dword v80, v84, s[40:43], 0 offen
buffer_load_dword v79, v83, s[40:43], 0 offen
buffer_load_dword v81, v85, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc8
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 1
s_call_b64 s[38:39], 960
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_mac_f32_dpp v62, v62, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v63, v63, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v64, v64, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v65, v65, v100 quad_perm:[2,2,1,1] row_mask:0xf bank_mask:0xf
s_setprio 0
s_nop 0
v_mac_f32_e32 v33, v41, v57
s_barrier
s_nop 0
ds_read_b128 v[38:41], v95 offset:25280
ds_read_b128 v[50:53], v90 offset:24768
ds_read_b128 v[54:57], v90 offset:24896
ds_write_b32 v93, v64
ds_write_b32 v94, v65
s_setprio 1
s_nop 0
s_waitcnt vmcnt(12) lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 64724
s_call_b64 s[38:39], 899
s_branch 64722
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_cndmask_b32_dpp v66, v66, v66, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v67, v67, v67, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v68, v68, v68, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v69, v69, v69, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v66, v67, v67 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v66, v67, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:29440
ds_read_b128 v[42:45], v90 offset:28928
ds_read_b128 v[46:49], v90 offset:29056
ds_write_b32 v91, v62
ds_write_b32 v92, v63
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v60, v84, s[40:43], 0 offen
buffer_load_dword v59, v83, s[40:43], 0 offen
buffer_load_dword v61, v85, s[40:43], 0 offen
s_add_u32 s91, s91, 0x200
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 7
s_call_b64 s[38:39], 822
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_add_f32_dpp v103, v69, v69 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v103, v69, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v69, v68, v68 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v69, v68, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e32 v68, v66, v69
v_add_f32_e64 v67, v103, v68 div:2
v_add_f32_e64 v68, -v103, v68 div:2
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
s_nop 0
ds_read_b128 v[38:41], v95 offset:33536
ds_read_b128 v[50:53], v90 offset:33024
ds_read_b128 v[54:57], v90 offset:33152
ds_write_b32 v93, v68 offset:8256
ds_write_b32 v94, v69 offset:8256
s_setprio 0
s_nop 0
s_waitcnt vmcnt(9) lgkmcnt(5)
ds_append v105 offset:65472
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 749
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_cndmask_b32_dpp v70, v70, v70, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v71, v71, v71, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v72, v72, v72, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v73, v73, v73, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v70, v71, v71 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v70, v71, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:37696
ds_read_b128 v[42:45], v90 offset:37184
ds_read_b128 v[46:49], v90 offset:37312
ds_write_b32 v91, v66 offset:8256
ds_write_b32 v92, v67 offset:8256
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v64, v84, s[40:43], 0 offen
buffer_load_dword v63, v83, s[40:43], 0 offen
buffer_load_dword v65, v85, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc0
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 7
s_call_b64 s[38:39], 670
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_add_f32_dpp v103, v73, v73 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v103, v73, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v73, v72, v72 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v73, v72, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e32 v72, v70, v73
v_add_f32_e64 v71, v103, v72 div:2
v_add_f32_e64 v72, -v103, v72 div:2
v_mac_f32_e32 v33, v41, v57
s_barrier
s_mov_b64 vcc, s[10:11]
s_nop 0
ds_read_b128 v[38:41], v95 offset:41792
ds_read_b128 v[50:53], v90 offset:41280
ds_read_b128 v[54:57], v90 offset:41408
ds_write_b32 v93, v72 offset:16512
ds_write_b32 v94, v73 offset:16512
s_setprio 0
s_nop 0
s_waitcnt vmcnt(9) lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 597
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_cndmask_b32_dpp v74, v74, v74, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v75, v75, v75, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v76, v76, v76, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v77, v77, v77, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v74, v75, v75 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v74, v75, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:45952
ds_read_b128 v[42:45], v90 offset:45440
ds_read_b128 v[46:49], v90 offset:45568
ds_write_b32 v91, v70 offset:16512
ds_write_b32 v92, v71 offset:16512
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v68, v84, s[40:43], 0 offen
buffer_load_dword v67, v83, s[40:43], 0 offen
buffer_load_dword v69, v85, s[40:43], 0 offen
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 1
s_call_b64 s[38:39], 520
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_add_f32_dpp v103, v77, v77 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v103, v77, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v77, v76, v76 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v77, v76, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e32 v76, v74, v77
v_add_f32_e64 v75, v103, v76 div:2
v_add_f32_e64 v76, -v103, v76 div:2
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
s_nop 0
ds_read_b128 v[38:41], v95 offset:512
ds_read_b128 v[50:53], v90
ds_read_b128 v[54:57], v90 offset:128
ds_write_b32 v93, v76 offset:24768
ds_write_b32 v94, v77 offset:24768
s_setprio 0
s_nop 0
s_waitcnt vmcnt(9) lgkmcnt(5)
ds_append v105 offset:65476
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 453
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_cndmask_b32_dpp v78, v78, v78, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v79, v79, v79, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v80, v80, v80, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v81, v81, v81, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v78, v79, v79 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v78, v79, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:4672
ds_read_b128 v[42:45], v90 offset:4160
ds_read_b128 v[46:49], v90 offset:4288
ds_write_b32 v91, v74 offset:24768
ds_write_b32 v92, v75 offset:24768
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v72, v84, s[40:43], 0 offen
buffer_load_dword v71, v83, s[40:43], 0 offen
buffer_load_dword v73, v85, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc4
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 7
s_call_b64 s[38:39], 374
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_add_f32_dpp v103, v81, v81 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v103, v81, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v81, v80, v80 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v81, v80, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e32 v80, v78, v81
v_add_f32_e64 v79, v103, v80 div:2
v_add_f32_e64 v80, -v103, v80 div:2
v_mac_f32_e32 v33, v41, v57
s_barrier
s_mov_b64 vcc, s[10:11]
s_nop 0
ds_read_b128 v[38:41], v95 offset:8768
ds_read_b128 v[50:53], v90 offset:8256
ds_read_b128 v[54:57], v90 offset:8384
ds_write_b32 v93, v80 offset:33024
ds_write_b32 v94, v81 offset:33024
s_setprio 0
s_nop 0
s_waitcnt vmcnt(9) lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 301
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_cndmask_b32_dpp v58, v58, v58, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v59, v59, v59, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v60, v60, v60, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v61, v61, v61, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v58, v59, v59 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v58, v59, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:12928
ds_read_b128 v[42:45], v90 offset:12416
ds_read_b128 v[46:49], v90 offset:12544
ds_write_b32 v91, v78 offset:33024
ds_write_b32 v92, v79 offset:33024
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v76, v84, s[40:43], 0 offen
buffer_load_dword v75, v83, s[40:43], 0 offen
buffer_load_dword v77, v85, s[40:43], 0 offen
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 1
s_call_b64 s[38:39], 224
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_add_f32_dpp v103, v61, v61 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v103, v61, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v61, v60, v60 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v61, v60, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e32 v60, v58, v61
v_add_f32_e64 v59, v103, v60 div:2
v_add_f32_e64 v60, -v103, v60 div:2
v_mac_f32_e32 v32, v40, v57
v_mac_f32_e32 v33, v41, v57
s_nop 0
ds_read_b128 v[38:41], v95 offset:17024
ds_read_b128 v[50:53], v90 offset:16512
ds_read_b128 v[54:57], v90 offset:16640
ds_write_b32 v93, v60 offset:41280
ds_write_b32 v94, v61 offset:41280
s_setprio 0
s_nop 0
s_waitcnt vmcnt(9) lgkmcnt(5)
ds_append v105 offset:65480
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 6
s_call_b64 s[38:39], 157
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v34, v42
v_mac_f32_e32 v3, v35, v42
v_mac_f32_e32 v4, v36, v42
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v37, v42
v_mac_f32_e32 v6, v34, v43
v_mac_f32_e32 v7, v35, v43
v_mac_f32_e32 v8, v36, v43
v_mac_f32_e32 v9, v37, v43
v_mac_f32_e32 v10, v34, v44
v_mac_f32_e32 v11, v35, v44
v_mac_f32_e32 v12, v36, v44
v_mac_f32_e32 v13, v37, v44
v_mac_f32_e32 v14, v34, v45
v_mac_f32_e32 v15, v35, v45
v_mac_f32_e32 v16, v36, v45
v_mac_f32_e32 v17, v37, v45
v_mac_f32_e32 v18, v34, v46
v_mac_f32_e32 v19, v35, v46
v_mac_f32_e32 v20, v36, v46
v_mac_f32_e32 v21, v37, v46
v_mac_f32_e32 v22, v34, v47
v_mac_f32_e32 v23, v35, v47
v_mac_f32_e32 v24, v36, v47
v_mac_f32_e32 v25, v37, v47
v_mac_f32_e32 v26, v34, v48
v_mac_f32_e32 v27, v35, v48
v_mac_f32_e32 v28, v36, v48
v_mac_f32_e32 v29, v37, v48
v_mac_f32_e32 v30, v34, v49
v_mac_f32_e32 v31, v35, v49
v_cndmask_b32_dpp v62, v62, v62, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v63, v63, v63, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v64, v64, v64, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_cndmask_b32_dpp v65, v65, v65, vcc row_half_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v62, v63, v63 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v62, v63, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_mac_f32_e32 v32, v36, v49
v_mac_f32_e32 v33, v37, v49
s_nop 0
ds_read_b128 v[34:37], v95 offset:21184
ds_read_b128 v[42:45], v90 offset:20672
ds_read_b128 v[46:49], v90 offset:20800
ds_write_b32 v91, v58 offset:41280
ds_write_b32 v92, v59 offset:41280
s_setprio 0
s_add_u32 s40, s40, s70
s_addc_u32 s41, s41, s71
buffer_load_dword v80, v84, s[40:43], 0 offen
buffer_load_dword v79, v83, s[40:43], 0 offen
buffer_load_dword v81, v85, s[40:43], 0 offen
s_mov_b32 m0, 0x2ffc8
s_nop 0
s_waitcnt lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 7
s_call_b64 s[38:39], 78
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_e32 v2, v38, v50
v_mac_f32_e32 v3, v39, v50
v_mac_f32_e32 v4, v40, v50
s_setprio 1
s_nop 0
v_mac_f32_e32 v5, v41, v50
v_mac_f32_e32 v6, v38, v51
v_mac_f32_e32 v7, v39, v51
v_mac_f32_e32 v8, v40, v51
v_mac_f32_e32 v9, v41, v51
v_mac_f32_e32 v10, v38, v52
v_mac_f32_e32 v11, v39, v52
v_mac_f32_e32 v12, v40, v52
v_mac_f32_e32 v13, v41, v52
v_mac_f32_e32 v14, v38, v53
v_mac_f32_e32 v15, v39, v53
v_mac_f32_e32 v16, v40, v53
v_mac_f32_e32 v17, v41, v53
v_mac_f32_e32 v18, v38, v54
v_mac_f32_e32 v19, v39, v54
v_mac_f32_e32 v20, v40, v54
v_mac_f32_e32 v21, v41, v54
v_mac_f32_e32 v22, v38, v55
v_mac_f32_e32 v23, v39, v55
v_mac_f32_e32 v24, v40, v55
v_mac_f32_e32 v25, v41, v55
v_mac_f32_e32 v26, v38, v56
v_mac_f32_e32 v27, v39, v56
v_mac_f32_e32 v28, v40, v56
v_mac_f32_e32 v29, v41, v56
v_mac_f32_e32 v30, v38, v57
v_mac_f32_e32 v31, v39, v57
v_mac_f32_e32 v32, v40, v57
v_add_f32_dpp v103, v65, v65 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v103, v65, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v65, v64, v64 quad_perm:[0,0,0,2] row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v65, v64, v100 quad_perm:[0,2,1,3] row_mask:0xf bank_mask:0xf
v_add_f32_e32 v64, v62, v65
v_add_f32_e64 v63, v103, v64 div:2
v_add_f32_e64 v64, -v103, v64 div:2
v_mac_f32_e32 v33, v41, v57
s_barrier
s_mov_b64 vcc, s[10:11]
s_nop 0
ds_read_b128 v[38:41], v95 offset:25280
ds_read_b128 v[50:53], v90 offset:24768
ds_read_b128 v[54:57], v90 offset:24896
ds_write_b32 v93, v64
ds_write_b32 v94, v65
s_setprio 0
s_nop 0
s_waitcnt vmcnt(9) lgkmcnt(5)
s_bitset0_b32 s18, 26
s_add_u32 s72, s72, -1
s_cbranch_scc1 64646
s_call_b64 s[38:39], 5
s_branch 64644
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_nop
s_cmp_eq_u32 s82, 0
s_cbranch_scc0 8
s_branch 596
s_add_u32 s82, s82, 1
s_andn2_b32 s82, s82, 1
s_bitcmp1_b32 0, 26
s_cselect_b32 s52, s69, s70
s_cselect_b32 s53, 0, s71
s_sub_u32 s40, s40, s52
s_subb_u32 s41, s41, s53
s_cmp_eq_u32 s94, 0
s_cbranch_scc0 3
s_cbranch_scc1 626
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
s_cbranch_scc1 241
s_add_u32 s88, s88, s17
s_cmp_eq_u32 s88, 0
s_cbranch_scc1 238
s_mov_b32 s89, 0
s_bitcmp1_b32 s18, 16
s_cbranch_scc1 227
s_add_u32 s87, s16, 15
s_lshr_b32 s87, s87, 4
v_mov_b32_e32 v107, s88
v_mul_u32_u24_e32 v107, s87, v107
v_add_co_u32_e32 v107, vcc, s17, v107
v_sub_co_u32_e64 v107, vcc, v107, 1
v_ffbh_u32_e32 v110, s17
v_lshlrev_b32_e64 v111, v110, s17
v_and_b32_e32 v112, 0xffffff00, v111
v_cmp_eq_u32_e32 vcc, 0x80000000, v111
v_cvt_f32_u32_e32 v112, v112
v_rcp_f32_e32 v106, v112
v_subb_co_u32_e32 v109, vcc, 32, v110, vcc
v_cvt_f32_ubyte0_e32 v110, v111
v_fma_f32 v112, v112, v106, -1.0
v_fma_f32 v112, v110, v106, v112
v_madak_f32 v112, v112, v106, 0x9f000000
v_mul_f32_e32 v112, 0x5f800000, v112
v_mov_b32_e32 v110, 0
v_cvt_flr_i32_f32_e64 v112, -v112
v_lshl_add_u32 v106, v106, 9, v112
v_mad_u64_u32 v[110:111], vcc, v111, v106, v[110:111]
v_subb_co_u32_e64 v106, vcc, v106, -1, vcc
v_mul_hi_u32 v110, v107, v106
v_add_co_u32_e32 v106, vcc, v110, v107
v_addc_co_u32_e64 v110, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v109
v_cndmask_b32_e32 v106, v106, v110, vcc
v_alignbit_b32 v106, v110, v106, v109
v_readfirstlane_b32 s86, v106
v_mul_u32_u24_e64 v106, v106, s8
v_ffbh_u32_e32 v110, s87
v_lshlrev_b32_e64 v111, v110, s87
v_and_b32_e32 v112, 0xffffff00, v111
v_cmp_eq_u32_e32 vcc, 0x80000000, v111
v_cvt_f32_u32_e32 v112, v112
v_rcp_f32_e32 v107, v112
v_subb_co_u32_e32 v109, vcc, 32, v110, vcc
v_cvt_f32_ubyte0_e32 v110, v111
v_fma_f32 v112, v112, v107, -1.0
v_fma_f32 v112, v110, v107, v112
v_madak_f32 v112, v112, v107, 0x9f000000
v_mul_f32_e32 v112, 0x5f800000, v112
v_mov_b32_e32 v110, 0
v_cvt_flr_i32_f32_e64 v112, -v112
v_lshl_add_u32 v107, v107, 9, v112
v_mad_u64_u32 v[110:111], vcc, v111, v107, v[110:111]
v_subb_co_u32_e64 v107, vcc, v107, -1, vcc
v_mul_hi_u32 v110, v106, v107
v_add_co_u32_e32 v107, vcc, v110, v106
v_addc_co_u32_e64 v110, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v109
v_cndmask_b32_e32 v107, v107, v110, vcc
v_alignbit_b32 v107, v110, v107, v109
v_readfirstlane_b32 s52, v106
v_readfirstlane_b32 s84, v107
s_mul_i32 s84, s84, s87
s_sub_u32 s84, s52, s84
v_sub_co_u32_e32 v107, vcc, s8, v107
v_sub_co_u32_e32 v107, vcc, s17, v107
v_and_b32_e64 v109, v0, 63
v_cmp_eq_u32_e64 vcc, v109, 0
v_cndmask_b32_e32 v107, 1, v107, vcc
s_sub_u32 s58, 0, s75
s_sub_u32 s59, 0, s74
v_mul_u32_u24_e64 v111, v107, 32
v_ffbh_u32_e32 v114, s58
v_lshlrev_b32_e64 v115, v114, s58
v_and_b32_e32 v113, 0xffffff00, v115
v_cmp_eq_u32_e32 vcc, 0x80000000, v115
v_cvt_f32_u32_e32 v113, v113
v_rcp_f32_e32 v109, v113
v_subb_co_u32_e32 v112, vcc, 32, v114, vcc
v_cvt_f32_ubyte0_e32 v114, v115
v_fma_f32 v113, v113, v109, -1.0
v_fma_f32 v113, v114, v109, v113
v_madak_f32 v113, v113, v109, 0x9f000000
v_mul_f32_e32 v113, 0x5f800000, v113
v_mov_b32_e32 v114, 0
v_cvt_flr_i32_f32_e64 v113, -v113
v_lshl_add_u32 v109, v109, 9, v113
v_mad_u64_u32 v[114:115], vcc, v115, v109, v[114:115]
v_subb_co_u32_e64 v109, vcc, v109, -1, vcc
v_mul_hi_u32 v114, v111, v109
v_add_co_u32_e32 v109, vcc, v114, v111
v_addc_co_u32_e64 v114, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v112
v_cndmask_b32_e32 v109, v109, v114, vcc
v_alignbit_b32 v109, v114, v109, v112
v_mad_i32_i24 v110, v109, s75, v111
v_mul_u32_u24_e64 v111, v109, 1
v_ffbh_u32_e32 v114, s59
v_lshlrev_b32_e64 v115, v114, s59
v_and_b32_e32 v113, 0xffffff00, v115
v_cmp_eq_u32_e32 vcc, 0x80000000, v115
v_cvt_f32_u32_e32 v113, v113
v_rcp_f32_e32 v109, v113
v_subb_co_u32_e32 v112, vcc, 32, v114, vcc
v_cvt_f32_ubyte0_e32 v114, v115
v_fma_f32 v113, v113, v109, -1.0
v_fma_f32 v113, v114, v109, v113
v_madak_f32 v113, v113, v109, 0x9f000000
v_mul_f32_e32 v113, 0x5f800000, v113
v_mov_b32_e32 v114, 0
v_cvt_flr_i32_f32_e64 v113, -v113
v_lshl_add_u32 v109, v109, 9, v113
v_mad_u64_u32 v[114:115], vcc, v115, v109, v[114:115]
v_subb_co_u32_e64 v109, vcc, v109, -1, vcc
v_mul_hi_u32 v114, v111, v109
v_add_co_u32_e32 v109, vcc, v114, v111
v_addc_co_u32_e64 v114, vcc, 0, 0, vcc
v_cmp_eq_u32_e32 vcc, 32, v112
v_cndmask_b32_e32 v109, v109, v114, vcc
v_alignbit_b32 v109, v114, v109, v112
v_mad_i32_i24 v111, v109, s74, v111
v_readfirstlane_b32 s76, v110
v_readfirstlane_b32 s77, v111
v_readfirstlane_b32 s78, v109
v_add_co_u32_e32 v96, vcc, s76, v96
v_addc_co_u32_e64 v112, vcc, 0, 0, vcc
v_mad_i32_i24 v96, v112, s75, v96
v_mad_i32_i24 v98, v112, s80, v98
v_mad_i32_i24 v97, v112, s79, v97
v_cmp_ge_i32_e64 vcc, v97, 0
v_addc_co_u32_e64 v112, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v112
v_mad_i32_i24 v97, v112, s74, v97
v_add_co_u32_e32 v97, vcc, s77, v97
v_addc_co_u32_e64 v112, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v112
v_mad_i32_i24 v97, v112, s74, v97
v_add_co_u32_e32 v98, vcc, s78, v98
v_readlane_b32 s76, v110, 1
v_readlane_b32 s77, v111, 1
v_readlane_b32 s78, v109, 1
s_add_u32 s85, s84, s86
s_cmp_le_u32 s85, s87
s_cselect_b32 s52, 0x20000, 0
s_cselect_b32 s85, s85, s87
s_or_b32 s18, s18, s52
s_lshl_b32 s84, s84, 4
s_lshl_b32 s85, s85, 4
s_min_u32 s85, s85, s16
s_cmp_eq_u32 s8, s17
s_cselect_b32 s52, 0x20000, 0
s_or_b32 s18, s18, s52
s_or_b32 s18, s18, s52
s_bitset1_b32 s18, 16
s_branch 43
s_lshr_b32 s84, s84, 4
s_add_u32 s85, s84, s86
s_sub_u32 s85, s85, s87
s_mov_b32 s84, 0
s_lshl_b32 s85, s85, 4
s_min_u32 s85, s85, s16
s_bitset1_b32 s18, 17
s_branch 12
s_bitset1_b32 s18, 18
s_mov_b32 s43, 0
s_mov_b32 s73, -1
s_mov_b32 s82, 40
s_branch 31
s_add_u32 s83, s83, 16
s_cmp_ge_u32 s83, s85
s_cbranch_scc0 28
s_bitset1_b32 s18, 22
s_sub_u32 s88, s88, s17
s_subb_u32 s89, s89, 0
s_cbranch_scc1 65281
v_add_co_u32_e32 v96, vcc, s76, v96
v_addc_co_u32_e64 v106, vcc, 0, 0, vcc
v_mad_i32_i24 v96, v106, s75, v96
v_mad_i32_i24 v98, v106, s80, v98
v_mad_i32_i24 v97, v106, s79, v97
v_cmp_ge_i32_e64 vcc, v97, 0
v_addc_co_u32_e64 v106, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v106
v_mad_i32_i24 v97, v106, s74, v97
v_add_co_u32_e32 v97, vcc, s77, v97
v_addc_co_u32_e64 v106, vcc, 0, 0, vcc
v_add_co_u32_e32 v98, vcc, v98, v106
v_mad_i32_i24 v97, v106, s74, v97
v_add_co_u32_e32 v98, vcc, s78, v98
s_mov_b32 s83, s84
v_cmp_le_u32_e32 vcc, 0x100, v0
s_cbranch_vccz 166
v_subrev_co_u32_e32 v106, vcc, s75, v96
v_subrev_co_u32_e32 v107, vcc, s74, v97
s_bitcmp1_b32 s18, 22
s_cbranch_scc0 64
s_bitset0_b32 s18, 22
s_bfe_u32 s52, s18, 0x10014
v_mul_u32_u24_e32 v111, 2, v106
v_mul_u32_u24_e32 v112, 2, v107
v_cvt_pk_u16_u32 v114, v111, v112
v_and_b32_e64 v111, v0, 1
v_cmp_eq_u32_e64 vcc, v111, 1
v_cndmask_b32_e32 v114, v98, v114, vcc
v_lshrrev_b32_e32 v110, 1, v0
v_bfe_u32 v115, v110, s52, 1
v_lshrrev_b32_e32 v110, 1, v0
v_bfi_b32 v110, 1, v0, v110
v_lshrrev_b32_e32 v111, 2, v0
v_bfi_b32 v111, 1, v0, v111
v_cmp_eq_u32_e64 vcc, s52, 0
v_cndmask_b32_e32 v110, v111, v110, vcc
s_sub_u32 s52, 1, s52
v_lshrrev_b32_e32 v111, s52, v110
v_bfi_b32 v110, 32, v111, v110
v_and_b32_e32 v110, 63, v110
v_add_co_u32_e32 v111, vcc, 16, v110
v_and_b32_e64 v112, v0, 2
v_cmp_eq_u32_e64 vcc, v112, 0
v_cndmask_b32_e32 v111, v111, v110, vcc
v_lshlrev_b32_e32 v112, 14, v115
v_mad_u32_u24 v111, 4, v111, v112
v_add_co_u32_e32 v110, vcc, s96, v111
ds_write_b32 v110, v114
v_writelane_b32 v112, s18, 0
v_writelane_b32 v112, s85, 1
v_writelane_b32 v112, s84, 2
v_and_b32_e64 v110, v0, 63
v_cmp_ge_u32_e64 vcc, v110, 3
v_mov_b32_e32 v113, 0x4000
v_cndmask_b32_e32 v110, v110, v113, vcc
v_mad_i32_i24 v110, v110, 4, s96
ds_write_b32 v110, v112 offset:256
s_add_u32 s96, s96, 0x18c
s_cmp_eq_u32 s96, 0xffc0
s_cselect_b32 s96, 0xc1e0, s96
v_mov_b32_dpp v108, v98 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v106, v106 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v107, v107 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_readfirstlane_b32 s81, v108
v_sub_co_u32_e64 v109, vcc, v108, s81
v_mul_lo_u32 v109, v109, s65
v_and_b32_e64 v113, v0, 3
v_ashrrev_i32_e64 v114, 1, s31
v_subrev_co_u32_e32 v113, vcc, v114, v113
v_ashrrev_i32_e64 v114, 1, s62
v_mad_i32_i24 v110, v114, 3, v113
s_bfe_u32 s52, s18, 0x10014
v_lshrrev_b32_e32 v112, 2, v0
v_and_b32_e32 v112, s52, v112
v_mad_i32_i24 v110, v112, 3, v110
v_add_co_u32_e64 v111, vcc, 1, s63
v_ashrrev_i32_e32 v111, 1, v111
v_add_co_u32_e64 v112, vcc, 1, s30
v_ashrrev_i32_e32 v112, 1, v112
v_sub_i32 v111, v111, v112
s_lshl_b32 s54, s15, 2
v_cmp_ge_u32_e64 s[52:53], v108, s12
v_mad_i32_i24 v106, v106, 2, v110
v_cmp_ge_u32_e64 s[56:57], v106, s15
v_mad_i32_i24 v106, 4, v106, v109
s_or_b64 s[56:57], s[56:57], s[52:53]
v_mad_i32_i24 v107, v107, 2, v111
v_cmp_ge_u32_e64 s[58:59], v107, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v82, v107, s54, v106
v_cndmask_b32_e64 v82, v82, -1, s[58:59]
v_add_co_u32_e32 v107, vcc, 1, v107
v_cmp_ge_u32_e64 s[58:59], v107, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v83, v107, s54, v106
v_cndmask_b32_e64 v83, v83, -1, s[58:59]
v_add_co_u32_e32 v107, vcc, 1, v107
v_cmp_ge_u32_e64 s[58:59], v107, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v84, v107, s54, v106
v_cndmask_b32_e64 v84, v84, -1, s[58:59]
v_add_co_u32_e32 v107, vcc, 1, v107
v_cmp_ge_u32_e64 s[58:59], v107, s14
s_or_b64 s[58:59], s[56:57], s[58:59]
v_mad_u32_u24 v85, v107, s54, v106
v_cndmask_b32_e64 v85, v85, -1, s[58:59]
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 138
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
s_lshr_b32 s52, s18, 6
s_xor_b32 s52, s52, s18
s_and_b32 s52, s52, 0x80000
s_cselect_b32 s52, s68, 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, 0
s_branch 95
s_bitcmp1_b32 s18, 18
s_cbranch_scc1 117
v_mad_u32_u24 v108, 5, v0, 2
v_lshlrev_b32_e32 v106, 1, v0
v_bfi_b32 v108, 4, v108, v106
v_bfe_u32 v106, v108, 2, 2
v_min_u32_e32 v106, 2, v106
v_bfe_u32 v108, v0, 1, 1
v_mad_u32_u24 v106, 2, v106, v108
v_mad_u32_u24 v106, s62, 3, v106
v_sub_co_u32_e32 v108, vcc, s29, v106
v_sub_co_u32_e64 v108, vcc, v108, 1
s_bfe_u32 s54, s18, 0x10001
v_cmp_eq_u32_e64 vcc, s54, 1
v_cndmask_b32_e32 v106, v106, v108, vcc
v_cmp_ge_u32_e64 s[52:53], v106, s29
v_lshlrev_b32_e32 v106, 2, v106
s_bfe_u32 s54, s18, 0x10018
v_bfe_u32 v109, v0, 2, s54
v_mul_lo_u32 v109, s68, v109
v_add_co_u32_e32 v106, vcc, v106, v109
v_mul_lo_u32 v107, s90, v99
v_add_co_u32_e32 v107, vcc, v107, v106
s_sub_u32 s54, s28, s63
s_sub_u32 s54, s54, 5
s_bitcmp1_b32 s18, 0
s_cselect_b32 s54, s54, s63
v_mov_b32_e32 v109, s54
s_lshl_b32 s57, s29, 2
v_cmp_ge_u32_e64 s[54:55], v109, s28
v_mad_i32_i24 v82, v109, s57, v107
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v82, v82, -1, s[54:55]
v_mov_b32_e32 v83, v82
v_add_co_u32_e64 v109, vcc, v109, 2
v_cmp_ge_u32_e64 s[54:55], v109, s28
v_mad_i32_i24 v85, v109, s57, v107
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v85, v85, -1, s[54:55]
v_add_co_u32_e64 v109, vcc, v109, 2
v_cmp_ge_u32_e64 s[54:55], v109, s28
v_mad_i32_i24 v84, v109, s57, v107
s_or_b64 s[54:55], s[54:55], s[52:53]
v_cndmask_b32_e64 v84, v84, -1, s[54:55]
v_add_co_u32_e64 v106, vcc, v99, s83
v_cmp_lt_u32_e64 vcc, v106, s16
v_cndmask_b32_e32 v82, -1, v82, vcc
v_cndmask_b32_e32 v83, -1, v83, vcc
v_cndmask_b32_e32 v84, -1, v84, vcc
v_cndmask_b32_e32 v85, -1, v85, vcc
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
s_lshr_b32 s52, s18, 6
s_xor_b32 s52, s52, s18
s_and_b32 s52, s52, 0x80000
s_cselect_b32 s52, s68, 0
s_add_u32 s40, s40, s52
s_addc_u32 s41, s41, 0
s_mov_b32 s43, 0x20000
s_mov_b32 s73, -1
s_bfe_u32 s52, s18, 0x10014
s_lshl_b32 s82, s13, s52
s_bfe_u32 s52, s18, 0x10013
s_bfe_u32 s54, s18, 0x10019
s_xor_b32 s52, s52, s54
s_cselect_b32 s52, 1, 0
s_cselect_b32 s43, 0x20000, s43
s_and_b32 s52, s52, s82
s_sub_u32 s82, s82, s52
s_bitcmp1_b32 s18, 20
s_cselect_b32 s52, 0, 0x2000000
s_bitcmp1_b32 s13, 0
s_cselect_b32 s52, s52, 0
s_xor_b32 s18, s18, s52
s_cmp_eq_u32 s82, 0
s_cbranch_scc1 9
s_mov_b64 vcc, s[10:11]
s_branch 64947
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_and_b32 s52, 0x900000, s18
s_subb_u32 s62, s62, 1
s_cbranch_scc0 65235
s_and_b32 s52, 0x900000, s18
s_subb_u32 s62, s61, 1
s_add_u32 s63, s63, 6
s_cmp_ge_u32 s63, s28
s_cbranch_scc0 65229
s_mov_b32 s63, 1
s_cmp_ge_u32 s63, s28
s_addc_u32 s64, s64, 1
s_cmp_gt_u32 s64, 1
s_cbranch_scc0 65224
s_mov_b32 s64, 0
s_mov_b32 s63, 0
s_branch 65190
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
v_mac_f32_dpp v4, v4, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v5, v5, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v2, v2, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v3, v3, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v3, v4, v3 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v2, v5, v2 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v3, v3, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v2, v2, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v2, v3, v2 row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v8, v8, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v9, v9, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v6, v6, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v7, v7, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v7, v8, v7 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v6, v9, v6 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v7, v7, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v6, v6, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v3, v7, v6 row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v12, v12, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v13, v13, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v10, v10, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v11, v11, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v11, v12, v11 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v10, v13, v10 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v11, v11, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v10, v10, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v4, v11, v10 row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v16, v16, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v17, v17, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v14, v14, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v15, v15, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v15, v16, v15 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v14, v17, v14 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v15, v15, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v14, v14, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v5, v15, v14 row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v20, v20, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v21, v21, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v18, v18, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v19, v19, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v19, v20, v19 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v18, v21, v18 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v19, v19, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v18, v18, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v6, v19, v18 row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v24, v24, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v25, v25, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v22, v22, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v23, v23, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v23, v24, v23 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v22, v25, v22 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v23, v23, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v22, v22, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v7, v23, v22 row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v28, v28, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v29, v29, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v26, v26, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v27, v27, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v27, v28, v27 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v26, v29, v26 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v27, v27, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v26, v26, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v8, v27, v26 row_half_mirror row_mask:0xf bank_mask:0xf
v_mac_f32_dpp v32, v32, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v33, v33, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v30, v30, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v31, v31, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_add_f32_dpp v31, v32, v31 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v30, v33, v30 row_mirror row_mask:0xf bank_mask:0xf
s_nop 0
v_mac_f32_dpp v31, v31, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v30, v30, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
s_nop 0
v_add_f32_dpp v9, v31, v30 row_half_mirror row_mask:0xf bank_mask:0xf
s_waitcnt vmcnt(0)
s_mov_b64 s[54:55], s[44:45]
s_mov_b32 s53, s47
v_bfe_u32 v106, s18, 21, 1
v_sub_co_u32_e64 v106, vcc, v106, 1
v_cndmask_b32_e32 v110, v88, v86, vcc
v_cndmask_b32_e32 v111, v89, v87, vcc
v_readlane_b32 s52, v104, 0
v_add_f32_e64 v2, v2, s52
v_mul_f32_e64 v106, v2, s36
v_cmp_lt_f32_e64 vcc, v2, 0
v_cndmask_b32_e32 v2, v2, v106, vcc
buffer_store_dword v2, v110, s[44:47], 0 offen
v_add_f32_e64 v3, v3, s52
v_mul_f32_e64 v106, v3, s36
v_cmp_lt_f32_e64 vcc, v3, 0
v_cndmask_b32_e32 v3, v3, v106, vcc
buffer_store_dword v3, v111, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s52, v104, 1
v_add_f32_e64 v4, v4, s52
v_mul_f32_e64 v106, v4, s36
v_cmp_lt_f32_e64 vcc, v4, 0
v_cndmask_b32_e32 v4, v4, v106, vcc
buffer_store_dword v4, v110, s[44:47], 0 offen
v_add_f32_e64 v5, v5, s52
v_mul_f32_e64 v106, v5, s36
v_cmp_lt_f32_e64 vcc, v5, 0
v_cndmask_b32_e32 v5, v5, v106, vcc
buffer_store_dword v5, v111, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
s_lshl_b32 s52, s67, 1
s_add_u32 s44, s44, s52
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 2
s_cselect_b32 s47, 0, s47
v_readlane_b32 s52, v104, 4
v_add_f32_e64 v6, v6, s52
v_mul_f32_e64 v106, v6, s36
v_cmp_lt_f32_e64 vcc, v6, 0
v_cndmask_b32_e32 v6, v6, v106, vcc
buffer_store_dword v6, v110, s[44:47], 0 offen
v_add_f32_e64 v7, v7, s52
v_mul_f32_e64 v106, v7, s36
v_cmp_lt_f32_e64 vcc, v7, 0
v_cndmask_b32_e32 v7, v7, v106, vcc
buffer_store_dword v7, v111, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
v_readlane_b32 s52, v104, 5
v_add_f32_e64 v8, v8, s52
v_mul_f32_e64 v106, v8, s36
v_cmp_lt_f32_e64 vcc, v8, 0
v_cndmask_b32_e32 v8, v8, v106, vcc
buffer_store_dword v8, v110, s[44:47], 0 offen
v_add_f32_e64 v9, v9, s52
v_mul_f32_e64 v106, v9, s36
v_cmp_lt_f32_e64 vcc, v9, 0
v_cndmask_b32_e32 v9, v9, v106, vcc
buffer_store_dword v9, v111, s[44:47], 0 offen
s_add_u32 s44, s44, s67
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 1
s_cselect_b32 s47, 0, s47
s_lshl_b32 s52, s67, 1
s_add_u32 s44, s44, s52
s_addc_u32 s45, s45, 0
s_lshl_b32 s52, s52, 2
s_add_u32 s44, s44, s52
s_addc_u32 s45, s45, 0
s_sub_u32 s93, s93, 10
s_cselect_b32 s47, 0, s47
s_bitcmp1_b32 s18, 21
s_cselect_b32 s47, s47, s53
s_cselect_b32 s44, s44, s54
s_cselect_b32 s45, s45, s55
s_cselect_b32 s53, 0, 16
s_cselect_b32 s54, 64, 0
s_add_u32 s93, s93, s53
s_add_u32 s48, s48, s54
s_addc_u32 s49, s49, 0
s_sub_u32 s50, s50, s54
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
s_xor_b32 s18, s18, 0x200000
s_bitcmp0_b32 s18, 21
s_addc_u32 s94, s60, 0
s_lshr_b32 s94, s94, 1
s_mul_i32 s94, s94, s61
s_lshr_b32 s94, s94, 1
s_mul_i32 s94, s94, s13
s_cmp_eq_u32 s94, 0
s_cbranch_scc1 65205
s_add_u32 s52, s93, s92
s_cmp_lt_i32 s52, 0
s_cbranch_scc0 131
v_and_b32_e32 v86, 0x7f, v0
v_lshrrev_b32_e32 v86, 1, v86
v_bfi_b32 v86, 1, v0, v86
v_and_b32_e64 v87, v0, 2
v_mad_u32_u24 v86, v87, 16, v86
v_lshlrev_b32_e32 v86, 2, v86
v_add_co_u32_e64 v86, vcc, v86, s97
v_and_b32_e32 v87, 3, v0
v_lshlrev_b32_e32 v87, 2, v87
v_add_co_u32_e64 v87, vcc, v87, s97
ds_read_b32 v108, v87 offset:256
ds_read_b32 v86, v86
s_add_u32 s97, s97, 0x18c
s_cmp_eq_u32 s97, 0xffc0
s_cselect_b32 s97, 0xc1e0, s97
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s95, v86
v_readlane_b32 s54, v108, 0
s_bitcmp1_b32 s54, 18
s_cbranch_scc1 107
v_readlane_b32 s52, v108, 1
v_readlane_b32 s53, v108, 2
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
s_mov_b32 s47, 0x20000
s_bitcmp1_b32 s18, 7
s_cselect_b32 s51, 0x20000, 0
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
v_bfe_u32 v106, v86, 16, 16
v_bfe_u32 v107, v86, 0, 16
v_and_b32_e64 v108, v0, 7
v_sub_co_u32_e32 v109, vcc, 7, v108
v_min_u32_e32 v108, v108, v109
v_bfe_u32 v109, v108, 1, 1
v_bfe_u32 v108, v108, 0, 1
v_mov_b32_dpp v106, v106 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v107, v107 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
v_add_co_u32_e32 v106, vcc, v106, v109
v_add_co_u32_e32 v107, vcc, v107, v108
v_mov_b32_dpp v108, v86 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf
v_cmp_ge_u32_e64 s[52:53], v108, s12
v_sub_co_u32_e64 v108, vcc, v108, s95
v_mul_lo_u32 v108, v108, s66
v_lshlrev_b32_e32 v107, 1, v107
s_and_b32 s54, 1, s31
v_add_co_u32_e32 v107, vcc, s54, v107
v_lshlrev_b32_e32 v106, 1, v106
s_and_b32 s54, 1, s30
v_subrev_co_u32_e32 v106, vcc, s54, v106
v_mad_i32_i24 v86, v106, s33, v107
v_lshlrev_b32_e32 v86, 2, v86
v_add_co_u32_e32 v86, vcc, v86, v108
v_subrev_co_u32_e32 v87, vcc, 4, v86
v_mad_i32_i24 v88, 4, s33, v86
v_mad_i32_i24 v89, 4, s33, v87
v_cmp_ge_u32_e64 s[58:59], v107, s33
s_or_b64 s[56:57], s[58:59], s[52:53]
v_subrev_co_u32_e32 v107, vcc, 1, v107
v_cmp_ge_u32_e64 s[58:59], v107, s33
s_or_b64 s[58:59], s[58:59], s[52:53]
v_cmp_ge_u32_e64 s[54:55], v106, s32
s_or_b64 s[52:53], s[56:57], s[54:55]
v_cndmask_b32_e64 v86, v86, -1, s[52:53]
s_or_b64 s[52:53], s[58:59], s[54:55]
v_cndmask_b32_e64 v87, v87, -1, s[52:53]
v_add_co_u32_e32 v106, vcc, 1, v106
v_cmp_ge_u32_e64 s[54:55], v106, s32
s_or_b64 s[52:53], s[56:57], s[54:55]
v_cndmask_b32_e64 v88, v88, -1, s[52:53]
s_or_b64 s[52:53], s[58:59], s[54:55]
v_cndmask_b32_e64 v89, v89, -1, s[52:53]
v_and_b32_e64 v104, v0, 63
v_lshlrev_b32_e32 v104, 2, v104
s_barrier
buffer_load_dword v104, v104, s[48:51], 0 offen
s_mov_b64 vcc, s[10:11]
s_branch 64439
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
