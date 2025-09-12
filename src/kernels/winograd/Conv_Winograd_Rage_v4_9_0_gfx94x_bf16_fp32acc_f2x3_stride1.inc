/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

v_lshlrev_b32 v1, 7, v0
s_getpc_b64 [s8, s9]
label_6:
s_mov_b32 s10, label_0 - label_6
s_mov_b32 s11, 131072
buffer_load_dword v2, v1, [s8, s9, s10, s11], 0 offen
s_waitcnt vmcnt(0)
s_mov_b32 s1, 50464518
s_getpc_b64 [s6, s7]
label_1:
label_2:
s_load_dwordx2 [s22, s23], [s2, s3], 0
s_load_dwordx16 [s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39], [s2, s3], 8
s_load_dwordx8 [s40, s41, s42, s43, s44, s45, s46, s47], [s2, s3], 72
v_readfirstlane_b32 s5, v0
v_and_b32 v6, 255, v0
v_lshrrev_b32 v7, 1, v6
v_and_b32 v8, 1, v0
v_and_b32 v1, 15, v6
v_mul_u32_u24 v3, 4160, v8
v_lshlrev_b32 v1, 3, v1
v_and_b32 v4, 31, v7
v_bfe_u32 v2, v6, 4, 2
v_lshl_add_u32 v3, v4, 3, v3
v_lshl_add_u32 v1, v2, 10, v1
v_bfe_u32 v4, v6, 6, 2
v_bfe_u32 v2, v6, 6, 2
v_bfe_u32 v5, v0, 8, 0
v_lshl_add_u32 v1, v2, 8, v1
v_lshlrev_b32 v4, v5, v4
v_bfe_u32 v2, v0, 8, 1
v_lshl_add_u32 v3, v4, 10, v3
v_lshl_add_u32 v1, v2, 7, v1
s_lshr_b32 s5, s5, 6
s_cmp_ge_u32 s5, 8
s_cbranch_scc0 label_3
s_setprio 3
label_3:
s_waitcnt lgkmcnt(0)
s_and_b32 s28, s28, 262143
s_and_b32 s8, s28, 8192
s_cbranch_scc0 label_7
s_load_dwordx8 [s48, s49, s50, s51, s52, s53, s54, s55], [s2, s3], 104
label_7:
s_and_b32 s8, s28, 512
s_cbranch_scc0 label_8
s_load_dwordx8 [s56, s57, s58, s59, s60, s61, s62, s63], [s2, s3], 136
s_load_dwordx4 [s64, s65, s66, s67], [s2, s3], 168
label_8:
s_and_b32 s8, s28, 1024
s_cbranch_scc0 label_9
s_load_dwordx4 [s68, s69, s70, s71], [s2, s3], 184
label_9:
s_and_b32 s8, s28, 81920
s_cbranch_scc0 label_10
s_load_dword s72, [s2, s3], 200
label_10:
s_and_b32 s8, s28, 65536
s_cbranch_scc0 label_11
s_load_dword s73, [s2, s3], 204
s_load_dwordx2 [s74, s75], [s2, s3], 208
label_11:
s_and_b32 s8, s28, 131072
s_cbranch_scc0 label_12
s_load_dwordx4 [s76, s77, s78, s79], [s2, s3], 216
label_12:
.long 0xba7f0241
.long 0x00000000 // encoding fixup: s_setreg_imm32_b32 hwreg(1, 9, 1), 0
s_bitcmp1_b32 s28, 3
s_cbranch_scc0 label_13
.long 0xba7f1901
.long 0x0000000f // encoding fixup: s_setreg_imm32_b32 hwreg(1, 4, 4), 15
.long 0xba7f1801
.long 0x00000000 // encoding fixup: s_setreg_imm32_b32 hwreg(1, 0, 4), 0
label_13:
s_waitcnt lgkmcnt(0)
s_bitcmp1_b32 s28, 6
s_cbranch_scc0 label_17
s_and_b32 s31, s31, 65535
s_and_b32 s33, s33, 65535
s_and_b32 s35, s35, 65535
s_and_b32 s45, s45, 65535
s_and_b32 s77, s77, 65535
s_and_b32 s75, s75, 65535
s_load_dwordx2 [s30, s31], [s30, s31], 0
s_load_dwordx2 [s32, s33], [s32, s33], 0
s_load_dwordx2 [s34, s35], [s34, s35], 0
s_bitcmp1_b32 s28, 7
s_cbranch_scc0 label_14
s_load_dwordx2 [s44, s45], [s44, s45], 0
label_14:
s_bitcmp1_b32 s28, 17
s_cbranch_scc0 label_15
s_load_dwordx2 [s76, s77], [s76, s77], 0
label_15:
s_bitcmp1_b32 s28, 16
s_cbranch_scc0 label_16
s_load_dwordx2 [s74, s75], [s74, s75], 0
label_16:
label_17:
s_bitcmp1_b32 s28, 9
s_cbranch_scc1 label_18
s_mov_b32 s58, s25
s_mul_i32 s57, s24, s25
s_mul_i32 s56, s57, s23
s_mov_b32 s66, s43
s_mul_i32 s65, s42, s43
s_mul_i32 s64, s65, s26
s_mul_i32 s8, s38, s39
s_bitcmp1_b32 s28, 2
s_cselect_b32 s9, s26, s23
s_mul_i32 s9, s8, s9
s_mov_b32 s62, s39
s_bitcmp1_b32 s28, 2
s_cselect_b32 s61, s9, s8
s_cselect_b32 s60, s8, s9
label_18:
s_and_b32 s8, s72, -256
s_bitcmp1_b32 s28, 8
s_cselect_b32 s9, 1, 0
s_bitcmp1_b32 s28, 14
s_cselect_b32 s72, s72, s9
s_or_b32 s72, s72, s8
s_cmp_lt_u32 s60, s61
s_cselect_b32 s8, 0, 2147483648
s_or_b32 s28, s28, s8
s_waitcnt lgkmcnt(0)
s_bitcmp1_b32 s28, 13
s_cbranch_scc0 label_19
s_add_u32 s30, s30, s48
s_addc_u32 s31, s31, s49
s_add_u32 s34, s34, s52
s_addc_u32 s35, s35, s53
s_add_u32 s32, s32, s50
s_addc_u32 s33, s33, s51
s_add_u32 s44, s44, s54
s_addc_u32 s45, s45, s55
s_add_u32 s76, s76, s78
s_addc_u32 s77, s77, s79
label_19:
s_bitcmp1_b32 s28, 10
s_cbranch_scc0 label_20
v_ffbh_u32 v6, s27
v_lshlrev_b32 v7, v6, s27
v_and_b32 v5, -256, v7
v_cmp_eq_u32 [vcc_lo, vcc_hi], 2147483648, v7
v_cvt_f32_u32 v5, v5
v_rcp_f32 v2, v5
v_subb_co_u32 v4, [vcc_lo, vcc_hi], 32, v6, [vcc_lo, vcc_hi]
v_cvt_f32_ubyte0 v6, v7
v_fma_f32 v5, v5, v2, -1.0
v_fma_f32 v5, v6, v2, v5
v_fmaak_f32 v5, v5, v2, 2667577344
v_mul_f32 v5, 1602224128, v5
v_mov_b32 v6, 0
v_cvt_flr_i32_f32 v5, neg(v5)
v_lshl_add_u32 v2, v2, 9, v5
v_mad_u64_u32 [v6, v7], [vcc_lo, vcc_hi], v7, v2, [v6, v7]
v_subb_co_u32 v2, [vcc_lo, vcc_hi], v2, -1, [vcc_lo, vcc_hi]
v_mul_hi_u32 v5, s4, v2
v_add_co_u32 v2, [vcc_lo, vcc_hi], v5, s4
v_addc_co_u32 v5, [vcc_lo, vcc_hi], 0, 0, [vcc_lo, vcc_hi]
v_cmp_eq_u32 [vcc_lo, vcc_hi], 32, v4
v_cndmask_b32 v2, v2, v5, [vcc_lo, vcc_hi]
v_alignbit_b32 v2, v5, v2, v4
s_nop 0
v_readfirstlane_b32 s8, v2
s_mul_i32 s10, s8, s27
s_sub_u32 s4, s4, s10
s_mul_i32 s10, s69, s8
s_mul_hi_u32 s11, s69, s8
s_lshl_b64 [s10, s11], [s10, s11], 1
s_add_u32 s30, s30, s10
s_addc_u32 s31, s31, s11
s_mul_i32 s10, s70, s8
s_mul_hi_u32 s11, s70, s8
s_lshl_b64 [s10, s11], [s10, s11], 1
s_add_u32 s32, s32, s10
s_addc_u32 s33, s33, s11
s_mul_i32 s10, s71, s8
s_mul_hi_u32 s11, s71, s8
s_lshl_b64 [s10, s11], [s10, s11], 1
s_add_u32 s34, s34, s10
s_addc_u32 s35, s35, s11
s_mul_i32 s10, s26, s8
s_lshl_b32 s10, s10, 1
s_add_u32 s44, s44, s10
s_addc_u32 s45, s45, 0
label_20:
s_cmp_eq_u32 s27, 304
s_cbranch_scc0 label_21
s_and_b32 s8, s4, 7
s_lshr_b32 s9, s4, 3
s_mul_i32 s4, s8, 38
s_add_u32 s4, s4, s9
label_21:
s_add_u32 s10, s26, 31
s_lshr_b32 s10, s10, 5
v_ffbh_u32 v6, s10
v_lshlrev_b32 v7, v6, s10
v_and_b32 v8, -256, v7
v_cmp_eq_u32 [vcc_lo, vcc_hi], 2147483648, v7
v_cvt_f32_u32 v8, v8
v_rcp_f32 v2, v8
v_subb_co_u32 v5, [vcc_lo, vcc_hi], 32, v6, [vcc_lo, vcc_hi]
v_cvt_f32_ubyte0 v6, v7
v_fma_f32 v8, v8, v2, -1.0
v_fma_f32 v8, v6, v2, v8
v_fmaak_f32 v8, v8, v2, 2667577344
v_mul_f32 v8, 1602224128, v8
v_mov_b32 v6, 0
v_cvt_flr_i32_f32 v8, neg(v8)
v_lshl_add_u32 v2, v2, 9, v8
v_mad_u64_u32 [v6, v7], [vcc_lo, vcc_hi], v7, v2, [v6, v7]
v_subb_co_u32 v2, [vcc_lo, vcc_hi], v2, -1, [vcc_lo, vcc_hi]
v_mul_hi_u32 v6, s27, v2
v_add_co_u32 v2, [vcc_lo, vcc_hi], v6, s27
v_addc_co_u32 v6, [vcc_lo, vcc_hi], 0, 0, [vcc_lo, vcc_hi]
v_cmp_eq_u32 [vcc_lo, vcc_hi], 32, v5
v_cndmask_b32 v2, v2, v6, [vcc_lo, vcc_hi]
v_alignbit_b32 v2, v6, v2, v5
v_ffbh_u32 v6, v2
v_lshlrev_b32 v7, v6, v2
v_and_b32 v8, -256, v7
v_cmp_eq_u32 [vcc_lo, vcc_hi], 2147483648, v7
v_cvt_f32_u32 v8, v8
v_rcp_f32 v4, v8
v_subb_co_u32 v5, [vcc_lo, vcc_hi], 32, v6, [vcc_lo, vcc_hi]
v_cvt_f32_ubyte0 v6, v7
v_fma_f32 v8, v8, v4, -1.0
v_fma_f32 v8, v6, v4, v8
v_fmaak_f32 v8, v8, v4, 2667577344
v_mul_f32 v8, 1602224128, v8
v_mov_b32 v6, 0
v_cvt_flr_i32_f32 v8, neg(v8)
v_lshl_add_u32 v4, v4, 9, v8
v_mad_u64_u32 [v6, v7], [vcc_lo, vcc_hi], v7, v4, [v6, v7]
v_subb_co_u32 v4, [vcc_lo, vcc_hi], v4, -1, [vcc_lo, vcc_hi]
v_mul_hi_u32 v6, s4, v4
v_add_co_u32 v4, [vcc_lo, vcc_hi], v6, s4
v_addc_co_u32 v6, [vcc_lo, vcc_hi], 0, 0, [vcc_lo, vcc_hi]
v_cmp_eq_u32 [vcc_lo, vcc_hi], 32, v5
v_cndmask_b32 v4, v4, v6, [vcc_lo, vcc_hi]
v_alignbit_b32 v4, v6, v4, v5
v_readfirstlane_b32 s8, v2
v_readfirstlane_b32 s11, v4
s_cmp_ge_u32 s11, s10
s_cbranch_scc1 label_0
s_mul_i32 s27, s8, s10
s_mul_i32 s9, s11, s8
s_sub_u32 s9, s4, s9
s_mul_i32 s12, s11, 32
s_sub_u32 s26, s26, s12
s_min_u32 s26, s26, 32
s_lshl_b32 s12, s12, 1
s_mul_i32 s14, s12, s60
s_mul_hi_u32 s15, s12, s60
s_add_u32 s32, s32, s14
s_addc_u32 s33, s33, s15
s_mul_i32 s14, s12, s65
s_mul_hi_u32 s15, s12, s65
s_add_u32 s34, s34, s14
s_addc_u32 s35, s35, s15
s_add_u32 s44, s44, s12
s_addc_u32 s45, s45, 0
s_cmp_ge_u32 s5, 8
s_cbranch_scc1 label_5
label_4:
v_and_b32 v19, 255, v0
s_bitcmp1_b32 s28, 31
s_cselect_b32 s37, 2, 0
s_cselect_b32 s50, 0, 2
v_bfe_u32 v17, v19, 2, s37
v_bfe_u32 v20, v19, 6, s50
v_add_u32 v20, v17, v20
v_lshrrev_b32 v21, 2, v19
v_lshrrev_b32 v21, s37, v21
v_and_b32 v17, 3, v21
v_bfe_u32 v4, v21, 2, 1
v_lshl_add_u32 v17, v4, 2, v17
v_bfe_u32 v4, v21, 3, 1
v_and_b32 v18, 15, v19
v_lshl_add_u32 v17, v4, 3, v17
v_bfe_u32 v2, v19, 6, 2
v_mul_u32_u24 v4, 24, v17
v_lshl_add_u32 v18, v2, 4, v18
v_xor_b32 v17, v19, v19 quad_perm:[0, 0, 0, 1]
v_mul_u32_u24 v2, 24, v18
v_lshl_add_u32 v4, v17, 3, v4
v_bfe_u32 v18, v19, 4, 2
v_mov_b32 v22, 1568
v_mad_u32_u16 v4, v20, v22, v4 op_sel:[0, 0, 0, 0]
v_mov_b32 v22, 1568
v_mad_u32_u16 v2, v18, v22, v2 op_sel:[0, 0, 0, 0]
s_cmp_eq_u32 s5, 9
s_cselect_b32 s51, 1024, 0
s_cselect_b32 s52, 4, 0
s_cmp_ge_u32 s5, 10
s_cselect_b32 s51, 65536, s51
s_cselect_b32 s52, 65536, s52
s_and_b32 s54, s5, 3
s_mul_i32 s53, s54, 2048
v_and_b32 v17, 63, v0
v_lshl_add_u32 v5, v17, 4, s51
v_mov_b32 v7, s52
v_lshl_add_u32 v6, v17, 4, s53
s_mov_b32 s20, 0
s_mov_b32 s21, 0
s_lshl_b32 s37, s57, 1
s_lshl_b32 s50, s61, 1
s_and_b32 s51, s5, 3
s_mul_i32 s52, 1, s51
s_sub_u32 s10, s23, s52
s_add_u32 s10, s10, 3
s_lshl_b32 s52, s37, 0
s_mul_i32 s52, s52, s51
s_add_u32 s30, s30, s52
s_addc_u32 s31, s31, 0
s_lshl_b32 s37, s37, 2
s_lshl_b32 s50, s50, 2
s_and_b32 s31, s31, 65535
s_add_u32 s31, s31, 131072
s_and_b32 s33, s33, 65535
s_add_u32 s33, s33, 131072
s_mov_b64 [s12, s13], [s30, s31]
s_mov_b32 s14, 0
s_mov_b32 s15, 131072
s_mov_b64 [s16, s17], [s30, s31]
s_mov_b32 s18, 0
s_mov_b32 s19, 131072
s_mov_b32 s11, 2147483648
s_cmp_ge_u32 s5, 4
s_cbranch_scc0 label_22
s_branch label_34
.align 32
label_50:
ds_read_b32 v16, v7 offset:55808
ds_read_b128 [v8, v9, v10, v11], v6 offset:45568
ds_read_b128 [v12, v13, v14, v15], v6 offset:46592
s_sub_u32 s20, s23, 1
s_add_u32 s48, s6, s36
s_addc_u32 s49, s7, 0
s_waitcnt lgkmcnt(2)
v_readfirstlane_b32 s29, v16
s_bitcmp1_b32 s29, 0
s_cmov_b32 s11, 0
s_add_u32 s21, s21, s29
s_mov_b32 s2, s10
s_mov_b64 [s12, s13], [s32, s33]
s_mov_b32 s14, s11
s_mov_b32 s3, s10
s_mov_b64 [s16, s17], [s32, s33]
s_mov_b32 s18, s11
s_setpc_b64 [s48, s49]
.align 32
label_51:
ds_read_b32 v16, v7 offset:55808
ds_read_b128 [v8, v9, v10, v11], v5 offset:53760
ds_read_b128 [v12, v13, v14, v15], v5 offset:54784
s_sub_u32 s20, s23, 1
s_add_u32 s48, s6, s36
s_addc_u32 s49, s7, 0
s_waitcnt lgkmcnt(2)
v_readfirstlane_b32 s29, v16
s_bitcmp1_b32 s29, 0
s_cmov_b32 s11, 0
s_add_u32 s21, s21, s29
s_mov_b32 s2, s10
s_mov_b64 [s12, s13], [s30, s31]
s_mov_b32 s14, s11
s_mov_b32 s3, s10
s_mov_b64 [s16, s17], [s30, s31]
s_mov_b32 s18, s11
s_setpc_b64 [s48, s49]
.align 32
label_22:
s_lshl_b32 s21, s21, 2
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(24)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_perm_b32 v154, v17, v129, s1
v_perm_b32 v155, v106, v130, s1
v_perm_b32 v156, v112, v136, s1
v_perm_b32 v157, v111, v135, s1
v_mfma_f32_16x16x16_bf16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_add_f32 v112, v17, v112
v_add_f32 v111, v106, v111
v_add_f32 v136, v129, v136
v_add_f32 v135, v130, v135
v_mfma_f32_16x16x16_bf16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_add_f32 v109, v112, v109
v_add_f32 v110, v111, v110
v_add_f32 v133, v136, v133
v_add_f32 v134, v135, v134
v_mfma_f32_16x16x16_bf16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_53
label_52:
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16_hi v17, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16_hi v106, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16_hi v107, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16_hi v129, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16_hi v130, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v131, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_23
s_mov_b32 s36, label_23 - label_1
s_branch label_46
.align 32
label_53:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16_hi v17, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16_hi v106, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16_hi v107, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16_hi v129, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16_hi v130, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v131, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_23
s_mov_b32 s36, label_23 - label_1
s_branch label_46
.align 32
label_23:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v92
v_lshlrev_b32 v159, 16, v93
v_lshlrev_b32 v160, 16, v96
v_lshlrev_b32 v161, 16, v97
v_lshlrev_b32 v162, 16, v104
v_lshlrev_b32 v163, 16, v105
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v92
v_and_b32 v159, 4294901760, v93
v_and_b32 v96, 4294901760, v96
v_and_b32 v97, 4294901760, v97
v_and_b32 v100, 4294901760, v104
v_and_b32 v101, 4294901760, v105
v_pk_add_f32 [v100, v101], [v158, v159], [v100, v101] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v100, v101], [v100, v101], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v100, v101], [v96, v97], -0.5, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v96, v97], [v96, v97], 1.0, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v96, v160, v96, s1
v_perm_b32 v97, v161, v97, s1
v_perm_b32 v100, v162, v100, s1
v_perm_b32 v101, v163, v101, s1
v_mfma_f32_16x16x16_bf16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_mul_f32 v109, v109, 0.5
v_mul_f32 v110, v110, 0.5
v_mul_f32 v133, v133, 0.5
v_mul_f32 v134, v134, 0.5
v_mfma_f32_16x16x16_bf16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_fma_f32 v112, v109, -1.0, v112
v_fma_f32 v111, v110, -1.0, v111
v_fma_f32 v136, v133, -1.0, v136
v_fma_f32 v135, v134, -1.0, v135
v_mfma_f32_16x16x16_bf16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_perm_b32 v154, v109, v133, s1
v_perm_b32 v155, v110, v134, s1
v_perm_b32 v156, v112, v136, s1
v_perm_b32 v157, v111, v135, s1
v_mfma_f32_16x16x16_bf16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_55
label_54:
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16_hi v109, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16_hi v110, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16_hi v111, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16_hi v133, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16_hi v134, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16_hi v135, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_24
s_mov_b32 s36, label_24 - label_1
s_branch label_47
.align 32
label_55:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16_hi v109, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16_hi v110, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16_hi v111, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16_hi v133, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16_hi v134, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16_hi v135, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_24
s_mov_b32 s36, label_24 - label_1
s_branch label_47
.align 32
label_24:
s_mov_b32 s36, label_56 - label_1
s_sub_u32 s20, s20, 16
s_cbranch_scc1 label_50
label_56:
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(24)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_perm_b32 v154, v113, v137, s1
v_perm_b32 v155, v114, v138, s1
v_perm_b32 v156, v120, v144, s1
v_perm_b32 v157, v119, v143, s1
v_mfma_f32_16x16x16_bf16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_add_f32 v120, v113, v120
v_add_f32 v119, v114, v119
v_add_f32 v144, v137, v144
v_add_f32 v143, v138, v143
v_mfma_f32_16x16x16_bf16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_add_f32 v117, v120, v117
v_add_f32 v118, v119, v118
v_add_f32 v141, v144, v141
v_add_f32 v142, v143, v142
v_mfma_f32_16x16x16_bf16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_58
label_57:
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16_hi v113, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16_hi v114, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16_hi v115, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16_hi v137, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16_hi v138, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v139, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_25
s_mov_b32 s36, label_25 - label_1
s_branch label_48
.align 32
label_58:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16_hi v113, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16_hi v114, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16_hi v115, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16_hi v137, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16_hi v138, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v139, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_25
s_mov_b32 s36, label_25 - label_1
s_branch label_48
.align 32
label_25:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v90
v_lshlrev_b32 v159, 16, v91
v_lshlrev_b32 v160, 16, v94
v_lshlrev_b32 v161, 16, v95
v_lshlrev_b32 v162, 16, v102
v_lshlrev_b32 v163, 16, v103
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v90
v_and_b32 v159, 4294901760, v91
v_and_b32 v94, 4294901760, v94
v_and_b32 v95, 4294901760, v95
v_and_b32 v98, 4294901760, v102
v_and_b32 v99, 4294901760, v103
v_pk_add_f32 [v98, v99], [v158, v159], [v98, v99] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v98, v99], [v98, v99], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v98, v99], [v94, v95], -0.5, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v94, v95], [v94, v95], 1.0, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v94, v160, v94, s1
v_perm_b32 v95, v161, v95, s1
v_perm_b32 v98, v162, v98, s1
v_perm_b32 v99, v163, v99, s1
v_mfma_f32_16x16x16_bf16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_mul_f32 v117, v117, 0.5
v_mul_f32 v118, v118, 0.5
v_mul_f32 v141, v141, 0.5
v_mul_f32 v142, v142, 0.5
v_mfma_f32_16x16x16_bf16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_fma_f32 v120, v117, -1.0, v120
v_fma_f32 v119, v118, -1.0, v119
v_fma_f32 v144, v141, -1.0, v144
v_fma_f32 v143, v142, -1.0, v143
v_mfma_f32_16x16x16_bf16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_perm_b32 v154, v117, v141, s1
v_perm_b32 v155, v118, v142, s1
v_perm_b32 v156, v120, v144, s1
v_perm_b32 v157, v119, v143, s1
v_mfma_f32_16x16x16_bf16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_60
label_59:
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16_hi v117, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16_hi v118, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16_hi v119, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16_hi v141, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16_hi v142, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16_hi v143, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_26
s_mov_b32 s36, label_26 - label_1
s_branch label_49
.align 32
label_60:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16_hi v117, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16_hi v118, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16_hi v119, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16_hi v141, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16_hi v142, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16_hi v143, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_26
s_mov_b32 s36, label_26 - label_1
s_branch label_49
.align 32
label_26:
s_lshl_b32 s21, s21, 2
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(24)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_perm_b32 v154, v122, v146, s1
v_perm_b32 v155, v124, v148, s1
v_perm_b32 v156, v127, v151, s1
v_perm_b32 v157, v125, v149, s1
v_mfma_f32_16x16x16_bf16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_add_f32 v127, v122, v127
v_add_f32 v125, v124, v125
v_add_f32 v151, v146, v151
v_add_f32 v149, v148, v149
v_mfma_f32_16x16x16_bf16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_add_f32 v126, v127, v126
v_add_f32 v128, v125, v128
v_add_f32 v150, v151, v150
v_add_f32 v152, v149, v152
v_mfma_f32_16x16x16_bf16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_62
label_61:
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16_hi v122, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16_hi v124, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16_hi v121, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16_hi v146, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16_hi v148, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v145, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_27
s_mov_b32 s36, label_27 - label_1
s_branch label_46
.align 32
label_62:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16_hi v122, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16_hi v124, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16_hi v121, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16_hi v146, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16_hi v148, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v145, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_27
s_mov_b32 s36, label_27 - label_1
s_branch label_46
.align 32
label_27:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v92
v_lshlrev_b32 v159, 16, v93
v_lshlrev_b32 v160, 16, v96
v_lshlrev_b32 v161, 16, v97
v_lshlrev_b32 v162, 16, v104
v_lshlrev_b32 v163, 16, v105
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v92
v_and_b32 v159, 4294901760, v93
v_and_b32 v96, 4294901760, v96
v_and_b32 v97, 4294901760, v97
v_and_b32 v100, 4294901760, v104
v_and_b32 v101, 4294901760, v105
v_pk_add_f32 [v100, v101], [v158, v159], [v100, v101] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v100, v101], [v100, v101], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v100, v101], [v96, v97], -0.5, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v96, v97], [v96, v97], 1.0, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v96, v160, v96, s1
v_perm_b32 v97, v161, v97, s1
v_perm_b32 v100, v162, v100, s1
v_perm_b32 v101, v163, v101, s1
v_mfma_f32_16x16x16_bf16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_mul_f32 v126, v126, 0.5
v_mul_f32 v128, v128, 0.5
v_mul_f32 v150, v150, 0.5
v_mul_f32 v152, v152, 0.5
v_mfma_f32_16x16x16_bf16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_fma_f32 v127, v126, -1.0, v127
v_fma_f32 v125, v128, -1.0, v125
v_fma_f32 v151, v150, -1.0, v151
v_fma_f32 v149, v152, -1.0, v149
v_mfma_f32_16x16x16_bf16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_perm_b32 v154, v126, v150, s1
v_perm_b32 v155, v128, v152, s1
v_perm_b32 v156, v127, v151, s1
v_perm_b32 v157, v125, v149, s1
v_mfma_f32_16x16x16_bf16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_64
label_63:
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16_hi v126, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16_hi v128, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16_hi v125, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16_hi v150, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16_hi v152, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16_hi v149, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_28
s_mov_b32 s36, label_28 - label_1
s_branch label_47
.align 32
label_64:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16_hi v126, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16_hi v128, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16_hi v125, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16_hi v150, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16_hi v152, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16_hi v149, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_28
s_mov_b32 s36, label_28 - label_1
s_branch label_47
.align 32
label_28:
s_mov_b32 s36, label_65 - label_1
s_sub_u32 s20, s20, 16
s_cbranch_scc1 label_50
label_65:
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(24)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_perm_b32 v154, v17, v129, s1
v_perm_b32 v155, v109, v133, s1
v_perm_b32 v156, v107, v131, s1
v_perm_b32 v157, v111, v135, s1
v_mfma_f32_16x16x16_bf16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_add_f32 v107, v17, v107
v_add_f32 v111, v109, v111
v_add_f32 v131, v129, v131
v_add_f32 v135, v133, v135
v_mfma_f32_16x16x16_bf16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_add_f32 v106, v107, v106
v_add_f32 v110, v111, v110
v_add_f32 v130, v131, v130
v_add_f32 v134, v135, v134
v_mfma_f32_16x16x16_bf16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_67
label_66:
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16_hi v17, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16_hi v109, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16_hi v112, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16_hi v129, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16_hi v133, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v136, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_29
s_mov_b32 s36, label_29 - label_1
s_branch label_48
.align 32
label_67:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16_hi v17, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16_hi v109, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16_hi v112, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16_hi v129, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16_hi v133, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v136, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_29
s_mov_b32 s36, label_29 - label_1
s_branch label_48
.align 32
label_29:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v90
v_lshlrev_b32 v159, 16, v91
v_lshlrev_b32 v160, 16, v94
v_lshlrev_b32 v161, 16, v95
v_lshlrev_b32 v162, 16, v102
v_lshlrev_b32 v163, 16, v103
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v90
v_and_b32 v159, 4294901760, v91
v_and_b32 v94, 4294901760, v94
v_and_b32 v95, 4294901760, v95
v_and_b32 v98, 4294901760, v102
v_and_b32 v99, 4294901760, v103
v_pk_add_f32 [v98, v99], [v158, v159], [v98, v99] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v98, v99], [v98, v99], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v98, v99], [v94, v95], -0.5, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v94, v95], [v94, v95], 1.0, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v94, v160, v94, s1
v_perm_b32 v95, v161, v95, s1
v_perm_b32 v98, v162, v98, s1
v_perm_b32 v99, v163, v99, s1
v_mfma_f32_16x16x16_bf16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_mul_f32 v106, v106, 0.5
v_mul_f32 v110, v110, 0.5
v_mul_f32 v130, v130, 0.5
v_mul_f32 v134, v134, 0.5
v_mfma_f32_16x16x16_bf16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_fma_f32 v107, v106, -1.0, v107
v_fma_f32 v111, v110, -1.0, v111
v_fma_f32 v131, v130, -1.0, v131
v_fma_f32 v135, v134, -1.0, v135
v_mfma_f32_16x16x16_bf16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_perm_b32 v154, v106, v130, s1
v_perm_b32 v155, v110, v134, s1
v_perm_b32 v156, v107, v131, s1
v_perm_b32 v157, v111, v135, s1
v_mfma_f32_16x16x16_bf16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_69
label_68:
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16_hi v106, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16_hi v110, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16_hi v111, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16_hi v130, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16_hi v134, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16_hi v135, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_30
s_mov_b32 s36, label_30 - label_1
s_branch label_49
.align 32
label_69:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16_hi v106, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16_hi v110, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16_hi v111, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16_hi v130, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16_hi v134, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16_hi v135, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_30
s_mov_b32 s36, label_30 - label_1
s_branch label_49
.align 32
label_30:
s_lshl_b32 s21, s21, 2
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(24)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_perm_b32 v154, v113, v137, s1
v_perm_b32 v155, v117, v141, s1
v_perm_b32 v156, v115, v139, s1
v_perm_b32 v157, v119, v143, s1
v_mfma_f32_16x16x16_bf16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_add_f32 v115, v113, v115
v_add_f32 v119, v117, v119
v_add_f32 v139, v137, v139
v_add_f32 v143, v141, v143
v_mfma_f32_16x16x16_bf16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_add_f32 v114, v115, v114
v_add_f32 v118, v119, v118
v_add_f32 v138, v139, v138
v_add_f32 v142, v143, v142
v_mfma_f32_16x16x16_bf16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_71
label_70:
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16_hi v113, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16_hi v117, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16_hi v120, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16_hi v137, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16_hi v141, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v144, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_31
s_mov_b32 s36, label_31 - label_1
s_branch label_46
.align 32
label_71:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16_hi v113, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33024
buffer_load_short_d16_hi v117, v13, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16_hi v120, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:34176
buffer_load_short_d16_hi v137, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16_hi v141, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v144, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_31
s_mov_b32 s36, label_31 - label_1
s_branch label_46
.align 32
label_31:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v92
v_lshlrev_b32 v159, 16, v93
v_lshlrev_b32 v160, 16, v96
v_lshlrev_b32 v161, 16, v97
v_lshlrev_b32 v162, 16, v104
v_lshlrev_b32 v163, 16, v105
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v92
v_and_b32 v159, 4294901760, v93
v_and_b32 v96, 4294901760, v96
v_and_b32 v97, 4294901760, v97
v_and_b32 v100, 4294901760, v104
v_and_b32 v101, 4294901760, v105
v_pk_add_f32 [v100, v101], [v158, v159], [v100, v101] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v100, v101], [v100, v101], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v100, v101], [v96, v97], -0.5, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v96, v97], [v96, v97], 1.0, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v96, v160, v96, s1
v_perm_b32 v97, v161, v97, s1
v_perm_b32 v100, v162, v100, s1
v_perm_b32 v101, v163, v101, s1
v_mfma_f32_16x16x16_bf16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_mul_f32 v114, v114, 0.5
v_mul_f32 v118, v118, 0.5
v_mul_f32 v138, v138, 0.5
v_mul_f32 v142, v142, 0.5
v_mfma_f32_16x16x16_bf16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_fma_f32 v115, v114, -1.0, v115
v_fma_f32 v119, v118, -1.0, v119
v_fma_f32 v139, v138, -1.0, v139
v_fma_f32 v143, v142, -1.0, v143
v_mfma_f32_16x16x16_bf16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_perm_b32 v154, v114, v138, s1
v_perm_b32 v155, v118, v142, s1
v_perm_b32 v156, v115, v139, s1
v_perm_b32 v157, v119, v143, s1
v_mfma_f32_16x16x16_bf16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_73
label_72:
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16_hi v114, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16_hi v118, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16_hi v119, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16_hi v138, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16_hi v142, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16_hi v143, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_32
s_mov_b32 s36, label_32 - label_1
s_branch label_47
.align 32
label_73:
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16_hi v114, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16_hi v118, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v154, v155] offset:33408
buffer_load_short_d16_hi v119, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
v_sub_co_u32 v15, [vcc_lo, vcc_hi], v15, 4
v_cndmask_b32 v12, v12, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v13, v13, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v14, v14, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16_hi v138, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16_hi v142, v13, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v4, [v156, v157] offset:33792
buffer_load_short_d16_hi v143, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s50
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_32
s_mov_b32 s36, label_32 - label_1
s_branch label_47
.align 32
label_32:
s_mov_b32 s36, label_74 - label_1
s_sub_u32 s20, s20, 16
s_cbranch_scc1 label_50
label_74:
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(24)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_perm_b32 v154, v122, v146, s1
v_perm_b32 v155, v126, v150, s1
v_perm_b32 v156, v121, v145, s1
v_perm_b32 v157, v125, v149, s1
v_mfma_f32_16x16x16_bf16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_add_f32 v121, v122, v121
v_add_f32 v125, v126, v125
v_add_f32 v145, v146, v145
v_add_f32 v149, v150, v149
v_mfma_f32_16x16x16_bf16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_add_f32 v124, v121, v124
v_add_f32 v128, v125, v128
v_add_f32 v148, v145, v148
v_add_f32 v152, v149, v152
v_mfma_f32_16x16x16_bf16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_76
label_75:
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16_hi v122, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16_hi v126, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16_hi v127, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16_hi v146, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16_hi v150, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v151, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_33
s_mov_b32 s36, label_33 - label_1
s_branch label_48
.align 32
label_76:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16_hi v122, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39296
buffer_load_short_d16_hi v126, v9, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16_hi v127, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_write_b64 v4, [v156, v157] offset:40448
buffer_load_short_d16_hi v146, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16_hi v150, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v151, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_33
s_mov_b32 s36, label_33 - label_1
s_branch label_48
.align 32
label_33:
s_setprio 1
s_waitcnt lgkmcnt(0)
s_barrier
.align 8
v_lshlrev_b32 v158, 16, v90
v_lshlrev_b32 v159, 16, v91
v_lshlrev_b32 v160, 16, v94
v_lshlrev_b32 v161, 16, v95
v_lshlrev_b32 v162, 16, v102
v_lshlrev_b32 v163, 16, v103
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v90
v_and_b32 v159, 4294901760, v91
v_and_b32 v94, 4294901760, v94
v_and_b32 v95, 4294901760, v95
v_and_b32 v98, 4294901760, v102
v_and_b32 v99, 4294901760, v103
v_pk_add_f32 [v98, v99], [v158, v159], [v98, v99] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v98, v99], [v98, v99], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v98, v99], [v94, v95], -0.5, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v94, v95], [v94, v95], 1.0, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v94, v160, v94, s1
v_perm_b32 v95, v161, v95, s1
v_perm_b32 v98, v162, v98, s1
v_perm_b32 v99, v163, v99, s1
v_mfma_f32_16x16x16_bf16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_mul_f32 v124, v124, 0.5
v_mul_f32 v128, v128, 0.5
v_mul_f32 v148, v148, 0.5
v_mul_f32 v152, v152, 0.5
v_mfma_f32_16x16x16_bf16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_fma_f32 v121, v124, -1.0, v121
v_fma_f32 v125, v128, -1.0, v125
v_fma_f32 v145, v148, -1.0, v145
v_fma_f32 v149, v152, -1.0, v149
v_mfma_f32_16x16x16_bf16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_perm_b32 v154, v124, v148, s1
v_perm_b32 v155, v128, v152, s1
v_perm_b32 v156, v121, v145, s1
v_perm_b32 v157, v125, v149, s1
v_mfma_f32_16x16x16_bf16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_setprio 1
s_cmp_lt_u32 s20, 15
s_cbranch_scc1 label_78
label_77:
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16_hi v124, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16_hi v128, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16_hi v125, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16_hi v148, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16_hi v152, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16_hi v149, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_22
s_mov_b32 s36, label_22 - label_1
s_branch label_49
.align 32
label_78:
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16_hi v124, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16_hi v128, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v154, v155] offset:39680
buffer_load_short_d16_hi v125, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
v_sub_co_u32 v11, [vcc_lo, vcc_hi], v11, 4
v_cndmask_b32 v8, v8, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v9, v9, -1, [vcc_lo, vcc_hi]
v_cndmask_b32 v10, v10, -1, [vcc_lo, vcc_hi]
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16_hi v148, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16_hi v152, v9, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v4, [v156, v157] offset:40064
buffer_load_short_d16_hi v149, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s50
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_22
s_mov_b32 s36, label_22 - label_1
s_branch label_49
.align 32
label_34:
s_lshl_b32 s21, s21, 2
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(32)
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_fma_f32 v17, v112, -1.0, v17
v_fma_f32 v106, v111, -1.0, v106
v_fma_f32 v129, v136, -1.0, v129
v_fma_f32 v130, v135, -1.0, v130
v_mfma_f32_16x16x16_bf16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_fma_f32 v108, v109, -1.0, v108
v_fma_f32 v107, v110, -1.0, v107
v_fma_f32 v132, v133, -1.0, v132
v_fma_f32 v131, v134, -1.0, v131
v_mfma_f32_16x16x16_bf16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_perm_b32 v154, v17, v129, s1
v_perm_b32 v155, v106, v130, s1
v_perm_b32 v156, v108, v132, s1
v_perm_b32 v157, v107, v131, s1
v_mfma_f32_16x16x16_bf16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
s_barrier
s_setprio 2
label_79:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16_hi v17, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:16512
buffer_load_short_d16_hi v108, v15, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16_hi v106, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v107, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_write_b64 v3, [v156, v157] offset:17280
buffer_load_short_d16_hi v129, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16_hi v132, v15, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v130, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v131, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_35
s_mov_b32 s36, label_35 - label_1
s_branch label_46
.align 32
label_35:
s_setprio 1
s_waitcnt lgkmcnt(0)
.align 8
v_lshlrev_b32 v158, 16, v92
v_lshlrev_b32 v159, 16, v93
v_lshlrev_b32 v160, 16, v96
v_lshlrev_b32 v161, 16, v97
v_lshlrev_b32 v162, 16, v104
v_lshlrev_b32 v163, 16, v105
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v92
v_and_b32 v159, 4294901760, v93
v_and_b32 v96, 4294901760, v96
v_and_b32 v97, 4294901760, v97
v_and_b32 v100, 4294901760, v104
v_and_b32 v101, 4294901760, v105
v_pk_add_f32 [v100, v101], [v158, v159], [v100, v101] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v100, v101], [v100, v101], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v100, v101], [v96, v97], -0.5, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v96, v97], [v96, v97], 1.0, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v96, v160, v96, s1
v_perm_b32 v97, v161, v97, s1
v_perm_b32 v100, v162, v100, s1
v_perm_b32 v101, v163, v101, s1
v_mfma_f32_16x16x16_bf16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_fma_f32 v154, v109, 1.0, v112
v_fma_f32 v155, v110, 1.0, v111
v_fma_f32 v156, v133, 1.0, v136
v_fma_f32 v157, v134, 1.0, v135
v_mfma_f32_16x16x16_bf16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_fma_f32 v109, v109, -1.0, v112
v_fma_f32 v110, v110, -1.0, v111
v_fma_f32 v112, v133, -1.0, v136
v_fma_f32 v111, v134, -1.0, v135
v_mfma_f32_16x16x16_bf16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_perm_b32 v154, v154, v156, s1
v_perm_b32 v155, v155, v157, s1
v_perm_b32 v156, v109, v112, s1
v_perm_b32 v157, v110, v111, s1
v_mfma_f32_16x16x16_bf16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_barrier
s_setprio 2
label_80:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16_hi v109, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16_hi v112, v15, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:16768
buffer_load_short_d16_hi v110, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v111, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16_hi v133, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16_hi v136, v15, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v156, v157] offset:17024
buffer_load_short_d16_hi v134, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v135, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_36
s_mov_b32 s36, label_36 - label_1
s_branch label_47
.align 32
label_36:
s_mov_b32 s36, label_81 - label_1
s_sub_u32 s20, s20, 16
s_cbranch_scc1 label_51
label_81:
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(32)
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_fma_f32 v113, v120, -1.0, v113
v_fma_f32 v114, v119, -1.0, v114
v_fma_f32 v137, v144, -1.0, v137
v_fma_f32 v138, v143, -1.0, v138
v_mfma_f32_16x16x16_bf16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_fma_f32 v116, v117, -1.0, v116
v_fma_f32 v115, v118, -1.0, v115
v_fma_f32 v140, v141, -1.0, v140
v_fma_f32 v139, v142, -1.0, v139
v_mfma_f32_16x16x16_bf16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_perm_b32 v154, v113, v137, s1
v_perm_b32 v155, v114, v138, s1
v_perm_b32 v156, v116, v140, s1
v_perm_b32 v157, v115, v139, s1
v_mfma_f32_16x16x16_bf16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
s_barrier
s_setprio 2
label_82:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16_hi v113, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:24768
buffer_load_short_d16_hi v116, v11, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16_hi v114, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v115, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_write_b64 v3, [v156, v157] offset:25536
buffer_load_short_d16_hi v137, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16_hi v140, v11, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v138, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v139, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_37
s_mov_b32 s36, label_37 - label_1
s_branch label_48
.align 32
label_37:
s_setprio 1
s_waitcnt lgkmcnt(0)
.align 8
v_lshlrev_b32 v158, 16, v90
v_lshlrev_b32 v159, 16, v91
v_lshlrev_b32 v160, 16, v94
v_lshlrev_b32 v161, 16, v95
v_lshlrev_b32 v162, 16, v102
v_lshlrev_b32 v163, 16, v103
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v90
v_and_b32 v159, 4294901760, v91
v_and_b32 v94, 4294901760, v94
v_and_b32 v95, 4294901760, v95
v_and_b32 v98, 4294901760, v102
v_and_b32 v99, 4294901760, v103
v_pk_add_f32 [v98, v99], [v158, v159], [v98, v99] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v98, v99], [v98, v99], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v98, v99], [v94, v95], -0.5, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v94, v95], [v94, v95], 1.0, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v94, v160, v94, s1
v_perm_b32 v95, v161, v95, s1
v_perm_b32 v98, v162, v98, s1
v_perm_b32 v99, v163, v99, s1
v_mfma_f32_16x16x16_bf16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_fma_f32 v154, v117, 1.0, v120
v_fma_f32 v155, v118, 1.0, v119
v_fma_f32 v156, v141, 1.0, v144
v_fma_f32 v157, v142, 1.0, v143
v_mfma_f32_16x16x16_bf16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_fma_f32 v117, v117, -1.0, v120
v_fma_f32 v118, v118, -1.0, v119
v_fma_f32 v120, v141, -1.0, v144
v_fma_f32 v119, v142, -1.0, v143
v_mfma_f32_16x16x16_bf16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_perm_b32 v154, v154, v156, s1
v_perm_b32 v155, v155, v157, s1
v_perm_b32 v156, v117, v120, s1
v_perm_b32 v157, v118, v119, s1
v_mfma_f32_16x16x16_bf16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_barrier
s_setprio 2
label_83:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16_hi v117, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16_hi v120, v11, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:25024
buffer_load_short_d16_hi v118, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v119, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16_hi v141, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16_hi v144, v11, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v156, v157] offset:25280
buffer_load_short_d16_hi v142, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v143, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_38
s_mov_b32 s36, label_38 - label_1
s_branch label_49
.align 32
label_38:
s_lshl_b32 s21, s21, 2
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(32)
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_fma_f32 v122, v127, -1.0, v122
v_fma_f32 v124, v125, -1.0, v124
v_fma_f32 v146, v151, -1.0, v146
v_fma_f32 v148, v149, -1.0, v148
v_mfma_f32_16x16x16_bf16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_fma_f32 v123, v126, -1.0, v123
v_fma_f32 v121, v128, -1.0, v121
v_fma_f32 v147, v150, -1.0, v147
v_fma_f32 v145, v152, -1.0, v145
v_mfma_f32_16x16x16_bf16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_perm_b32 v154, v122, v146, s1
v_perm_b32 v155, v124, v148, s1
v_perm_b32 v156, v123, v147, s1
v_perm_b32 v157, v121, v145, s1
v_mfma_f32_16x16x16_bf16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
s_barrier
s_setprio 2
label_84:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16_hi v122, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:16512
buffer_load_short_d16_hi v123, v15, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16_hi v124, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v121, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_write_b64 v3, [v156, v157] offset:17280
buffer_load_short_d16_hi v146, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16_hi v147, v15, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v148, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v145, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_39
s_mov_b32 s36, label_39 - label_1
s_branch label_46
.align 32
label_39:
s_setprio 1
s_waitcnt lgkmcnt(0)
.align 8
v_lshlrev_b32 v158, 16, v92
v_lshlrev_b32 v159, 16, v93
v_lshlrev_b32 v160, 16, v96
v_lshlrev_b32 v161, 16, v97
v_lshlrev_b32 v162, 16, v104
v_lshlrev_b32 v163, 16, v105
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v92
v_and_b32 v159, 4294901760, v93
v_and_b32 v96, 4294901760, v96
v_and_b32 v97, 4294901760, v97
v_and_b32 v100, 4294901760, v104
v_and_b32 v101, 4294901760, v105
v_pk_add_f32 [v100, v101], [v158, v159], [v100, v101] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v100, v101], [v100, v101], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v100, v101], [v96, v97], -0.5, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v96, v97], [v96, v97], 1.0, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v96, v160, v96, s1
v_perm_b32 v97, v161, v97, s1
v_perm_b32 v100, v162, v100, s1
v_perm_b32 v101, v163, v101, s1
v_mfma_f32_16x16x16_bf16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_fma_f32 v154, v126, 1.0, v127
v_fma_f32 v155, v128, 1.0, v125
v_fma_f32 v156, v150, 1.0, v151
v_fma_f32 v157, v152, 1.0, v149
v_mfma_f32_16x16x16_bf16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_fma_f32 v126, v126, -1.0, v127
v_fma_f32 v128, v128, -1.0, v125
v_fma_f32 v127, v150, -1.0, v151
v_fma_f32 v125, v152, -1.0, v149
v_mfma_f32_16x16x16_bf16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_perm_b32 v154, v154, v156, s1
v_perm_b32 v155, v155, v157, s1
v_perm_b32 v156, v126, v127, s1
v_perm_b32 v157, v128, v125, s1
v_mfma_f32_16x16x16_bf16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_barrier
s_setprio 2
label_85:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16_hi v126, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16_hi v127, v15, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:16768
buffer_load_short_d16_hi v128, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v125, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16_hi v150, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16_hi v151, v15, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v156, v157] offset:17024
buffer_load_short_d16_hi v152, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v149, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_40
s_mov_b32 s36, label_40 - label_1
s_branch label_47
.align 32
label_40:
s_mov_b32 s36, label_86 - label_1
s_sub_u32 s20, s20, 16
s_cbranch_scc1 label_51
label_86:
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(32)
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_fma_f32 v17, v107, -1.0, v17
v_fma_f32 v109, v111, -1.0, v109
v_fma_f32 v129, v131, -1.0, v129
v_fma_f32 v133, v135, -1.0, v133
v_mfma_f32_16x16x16_bf16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_fma_f32 v108, v106, -1.0, v108
v_fma_f32 v112, v110, -1.0, v112
v_fma_f32 v132, v130, -1.0, v132
v_fma_f32 v136, v134, -1.0, v136
v_mfma_f32_16x16x16_bf16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_perm_b32 v154, v17, v129, s1
v_perm_b32 v155, v109, v133, s1
v_perm_b32 v156, v108, v132, s1
v_perm_b32 v157, v112, v136, s1
v_mfma_f32_16x16x16_bf16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
s_barrier
s_setprio 2
label_87:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16_hi v17, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:24768
buffer_load_short_d16_hi v108, v11, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16_hi v109, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v112, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_write_b64 v3, [v156, v157] offset:25536
buffer_load_short_d16_hi v129, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16_hi v132, v11, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v133, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v136, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_41
s_mov_b32 s36, label_41 - label_1
s_branch label_48
.align 32
label_41:
s_setprio 1
s_waitcnt lgkmcnt(0)
.align 8
v_lshlrev_b32 v158, 16, v90
v_lshlrev_b32 v159, 16, v91
v_lshlrev_b32 v160, 16, v94
v_lshlrev_b32 v161, 16, v95
v_lshlrev_b32 v162, 16, v102
v_lshlrev_b32 v163, 16, v103
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v90
v_and_b32 v159, 4294901760, v91
v_and_b32 v94, 4294901760, v94
v_and_b32 v95, 4294901760, v95
v_and_b32 v98, 4294901760, v102
v_and_b32 v99, 4294901760, v103
v_pk_add_f32 [v98, v99], [v158, v159], [v98, v99] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v98, v99], [v98, v99], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v98, v99], [v94, v95], -0.5, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v94, v95], [v94, v95], 1.0, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v94, v160, v94, s1
v_perm_b32 v95, v161, v95, s1
v_perm_b32 v98, v162, v98, s1
v_perm_b32 v99, v163, v99, s1
v_mfma_f32_16x16x16_bf16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_fma_f32 v154, v106, 1.0, v107
v_fma_f32 v155, v110, 1.0, v111
v_fma_f32 v156, v130, 1.0, v131
v_fma_f32 v157, v134, 1.0, v135
v_mfma_f32_16x16x16_bf16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_fma_f32 v106, v106, -1.0, v107
v_fma_f32 v110, v110, -1.0, v111
v_fma_f32 v107, v130, -1.0, v131
v_fma_f32 v111, v134, -1.0, v135
v_mfma_f32_16x16x16_bf16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_perm_b32 v154, v154, v156, s1
v_perm_b32 v155, v155, v157, s1
v_perm_b32 v156, v106, v107, s1
v_perm_b32 v157, v110, v111, s1
v_mfma_f32_16x16x16_bf16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_barrier
s_setprio 2
label_88:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16_hi v106, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16_hi v107, v11, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:25024
buffer_load_short_d16_hi v110, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v111, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16_hi v130, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16_hi v131, v11, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v156, v157] offset:25280
buffer_load_short_d16_hi v134, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v135, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_42
s_mov_b32 s36, label_42 - label_1
s_branch label_49
.align 32
label_42:
s_lshl_b32 s21, s21, 2
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(32)
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v18, v19, v20, v21], [v90, v91], [v34, v35], [v18, v19, v20, v21]
v_fma_f32 v113, v115, -1.0, v113
v_fma_f32 v117, v119, -1.0, v117
v_fma_f32 v137, v139, -1.0, v137
v_fma_f32 v141, v143, -1.0, v141
v_mfma_f32_16x16x16_bf16 [v70, v71, v72, v73], [v102, v103], [v88, v89], [v70, v71, v72, v73]
v_fma_f32 v116, v114, -1.0, v116
v_fma_f32 v120, v118, -1.0, v120
v_fma_f32 v140, v138, -1.0, v140
v_fma_f32 v144, v142, -1.0, v144
v_mfma_f32_16x16x16_bf16 [v52, v53, v54, v55], [v98, v99], [v86, v87], [v52, v53, v54, v55]
v_perm_b32 v154, v113, v137, s1
v_perm_b32 v155, v117, v141, s1
v_perm_b32 v156, v116, v140, s1
v_perm_b32 v157, v120, v144, s1
v_mfma_f32_16x16x16_bf16 [v36, v37, v38, v39], [v94, v95], [v68, v69], [v36, v37, v38, v39]
s_barrier
s_setprio 2
label_89:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v92, v93], v2 offset:39296
buffer_load_short_d16_hi v113, v12, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:16512
buffer_load_short_d16_hi v116, v15, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v96, v97], v2 offset:39304
buffer_load_short_d16_hi v117, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v120, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_write_b64 v3, [v156, v157] offset:17280
buffer_load_short_d16_hi v137, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v104, v105], v2 offset:39312
buffer_load_short_d16_hi v140, v15, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v141, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v144, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_43
s_mov_b32 s36, label_43 - label_1
s_branch label_46
.align 32
label_43:
s_setprio 1
s_waitcnt lgkmcnt(0)
.align 8
v_lshlrev_b32 v158, 16, v92
v_lshlrev_b32 v159, 16, v93
v_lshlrev_b32 v160, 16, v96
v_lshlrev_b32 v161, 16, v97
v_lshlrev_b32 v162, 16, v104
v_lshlrev_b32 v163, 16, v105
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v92
v_and_b32 v159, 4294901760, v93
v_and_b32 v96, 4294901760, v96
v_and_b32 v97, 4294901760, v97
v_and_b32 v100, 4294901760, v104
v_and_b32 v101, 4294901760, v105
v_pk_add_f32 [v100, v101], [v158, v159], [v100, v101] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v100, v101], [v100, v101], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v100, v101], [v96, v97], -0.5, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v96, v97], [v96, v97], 1.0, [v100, v101] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v96, v160, v96, s1
v_perm_b32 v97, v161, v97, s1
v_perm_b32 v100, v162, v100, s1
v_perm_b32 v101, v163, v101, s1
v_mfma_f32_16x16x16_bf16 [v26, v27, v28, v29], [v92, v93], [v34, v35], [v26, v27, v28, v29]
v_fma_f32 v154, v114, 1.0, v115
v_fma_f32 v155, v118, 1.0, v119
v_fma_f32 v156, v138, 1.0, v139
v_fma_f32 v157, v142, 1.0, v143
v_mfma_f32_16x16x16_bf16 [v78, v79, v80, v81], [v104, v105], [v88, v89], [v78, v79, v80, v81]
v_fma_f32 v114, v114, -1.0, v115
v_fma_f32 v118, v118, -1.0, v119
v_fma_f32 v115, v138, -1.0, v139
v_fma_f32 v119, v142, -1.0, v143
v_mfma_f32_16x16x16_bf16 [v60, v61, v62, v63], [v100, v101], [v86, v87], [v60, v61, v62, v63]
v_perm_b32 v154, v154, v156, s1
v_perm_b32 v155, v155, v157, s1
v_perm_b32 v156, v114, v115, s1
v_perm_b32 v157, v118, v119, s1
v_mfma_f32_16x16x16_bf16 [v44, v45, v46, v47], [v96, v97], [v68, v69], [v44, v45, v46, v47]
s_barrier
s_setprio 2
label_90:
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v34, v35], v1 offset:24768
buffer_load_short_d16_hi v114, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v68, v69], v1 offset:28928
buffer_load_short_d16_hi v115, v15, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v154, v155] offset:16768
buffer_load_short_d16_hi v118, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v119, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_sub_u32 s3, s3, 4
s_cselect_b32 s18, 0, s18
ds_read_b64 [v86, v87], v1 offset:24776
buffer_load_short_d16_hi v138, v12, [s16, s17, s18, s19], 0 idxen
ds_read_b64 [v88, v89], v1 offset:28936
buffer_load_short_d16_hi v139, v15, [s16, s17, s18, s19], 0 idxen
ds_write_b64 v3, [v156, v157] offset:17024
buffer_load_short_d16_hi v142, v13, [s16, s17, s18, s19], 0 idxen
buffer_load_short_d16_hi v143, v14, [s16, s17, s18, s19], 0 idxen
s_add_u32 s16, s16, s37
s_addc_u32 s17, s17, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_44
s_mov_b32 s36, label_44 - label_1
s_branch label_47
.align 32
label_44:
s_mov_b32 s36, label_91 - label_1
s_sub_u32 s20, s20, 16
s_cbranch_scc1 label_51
label_91:
s_setprio 1
s_waitcnt lgkmcnt(0) & vmcnt(32)
.align 8
v_lshlrev_b32 v158, 16, v34
v_lshlrev_b32 v159, 16, v35
v_lshlrev_b32 v162, 16, v86
v_lshlrev_b32 v163, 16, v87
v_and_b32 v34, 4294901760, v34
v_and_b32 v35, 4294901760, v35
v_and_b32 v86, 4294901760, v86
v_and_b32 v87, 4294901760, v87
v_pk_fma_f32 [v158, v159], [v162, v163], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v34, v35], [v86, v87], -1.0, [v34, v35] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v34, v158, v34, s1
v_perm_b32 v35, v159, v35, s1
v_lshlrev_b32 v158, 16, v88
v_lshlrev_b32 v159, 16, v89
v_lshlrev_b32 v160, 16, v68
v_lshlrev_b32 v161, 16, v69
v_and_b32 v88, 4294901760, v88
v_and_b32 v89, 4294901760, v89
v_and_b32 v68, 4294901760, v68
v_and_b32 v69, 4294901760, v69
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v88, v89], [v68, v69], -1.0, [v88, v89] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v88, v158, v88, s1
v_perm_b32 v89, v159, v89, s1
v_pk_fma_f32 [v158, v159], [v160, v161], -1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v162, v163], [v68, v69], -1.0, [v86, v87] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v86, v158, v162, s1
v_perm_b32 v87, v159, v163, s1
v_pk_fma_f32 [v158, v159], [v160, v161], 2.0, [v158, v159] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v68, v69], 2.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v68, v158, v160, s1
v_perm_b32 v69, v159, v161, s1
v_mfma_f32_16x16x16_bf16 [v22, v23, v24, v25], [v90, v91], [v34, v35], [v22, v23, v24, v25]
v_fma_f32 v122, v121, -1.0, v122
v_fma_f32 v126, v125, -1.0, v126
v_fma_f32 v146, v145, -1.0, v146
v_fma_f32 v150, v149, -1.0, v150
v_mfma_f32_16x16x16_bf16 [v74, v75, v76, v77], [v102, v103], [v88, v89], [v74, v75, v76, v77]
v_fma_f32 v123, v124, -1.0, v123
v_fma_f32 v127, v128, -1.0, v127
v_fma_f32 v147, v148, -1.0, v147
v_fma_f32 v151, v152, -1.0, v151
v_mfma_f32_16x16x16_bf16 [v56, v57, v58, v59], [v98, v99], [v86, v87], [v56, v57, v58, v59]
v_perm_b32 v154, v122, v146, s1
v_perm_b32 v155, v126, v150, s1
v_perm_b32 v156, v123, v147, s1
v_perm_b32 v157, v127, v151, s1
v_mfma_f32_16x16x16_bf16 [v40, v41, v42, v43], [v94, v95], [v68, v69], [v40, v41, v42, v43]
s_barrier
s_setprio 2
label_92:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v90, v91], v2 offset:33024
buffer_load_short_d16_hi v122, v8, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:24768
buffer_load_short_d16_hi v123, v11, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v94, v95], v2 offset:33032
buffer_load_short_d16_hi v126, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v127, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_write_b64 v3, [v156, v157] offset:25536
buffer_load_short_d16_hi v146, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v102, v103], v2 offset:33040
buffer_load_short_d16_hi v147, v11, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v150, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v151, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_45
s_mov_b32 s36, label_45 - label_1
s_branch label_48
.align 32
label_45:
s_setprio 1
s_waitcnt lgkmcnt(0)
.align 8
v_lshlrev_b32 v158, 16, v90
v_lshlrev_b32 v159, 16, v91
v_lshlrev_b32 v160, 16, v94
v_lshlrev_b32 v161, 16, v95
v_lshlrev_b32 v162, 16, v102
v_lshlrev_b32 v163, 16, v103
v_pk_add_f32 [v162, v163], [v158, v159], [v162, v163] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v162, v163], [v162, v163], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v162, v163], [v160, v161], -0.5, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v160, v161], [v160, v161], 1.0, [v162, v163] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_and_b32 v158, 4294901760, v90
v_and_b32 v159, 4294901760, v91
v_and_b32 v94, 4294901760, v94
v_and_b32 v95, 4294901760, v95
v_and_b32 v98, 4294901760, v102
v_and_b32 v99, 4294901760, v103
v_pk_add_f32 [v98, v99], [v158, v159], [v98, v99] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_mul_f32 [v98, v99], [v98, v99], 0.5 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v98, v99], [v94, v95], -0.5, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_fma_f32 [v94, v95], [v94, v95], 1.0, [v98, v99] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_perm_b32 v94, v160, v94, s1
v_perm_b32 v95, v161, v95, s1
v_perm_b32 v98, v162, v98, s1
v_perm_b32 v99, v163, v99, s1
v_mfma_f32_16x16x16_bf16 [v30, v31, v32, v33], [v92, v93], [v34, v35], [v30, v31, v32, v33]
v_fma_f32 v154, v124, 1.0, v121
v_fma_f32 v155, v128, 1.0, v125
v_fma_f32 v156, v148, 1.0, v145
v_fma_f32 v157, v152, 1.0, v149
v_mfma_f32_16x16x16_bf16 [v82, v83, v84, v85], [v104, v105], [v88, v89], [v82, v83, v84, v85]
v_fma_f32 v124, v124, -1.0, v121
v_fma_f32 v128, v128, -1.0, v125
v_fma_f32 v121, v148, -1.0, v145
v_fma_f32 v125, v152, -1.0, v149
v_mfma_f32_16x16x16_bf16 [v64, v65, v66, v67], [v100, v101], [v86, v87], [v64, v65, v66, v67]
v_perm_b32 v154, v154, v156, s1
v_perm_b32 v155, v155, v157, s1
v_perm_b32 v156, v124, v121, s1
v_perm_b32 v157, v128, v125, s1
v_mfma_f32_16x16x16_bf16 [v48, v49, v50, v51], [v96, v97], [v68, v69], [v48, v49, v50, v51]
s_barrier
s_setprio 2
label_93:
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v34, v35], v1 offset:16512
buffer_load_short_d16_hi v124, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v68, v69], v1 offset:20672
buffer_load_short_d16_hi v121, v11, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v154, v155] offset:25024
buffer_load_short_d16_hi v128, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v125, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_sub_u32 s2, s2, 4
s_cselect_b32 s14, 0, s14
ds_read_b64 [v86, v87], v1 offset:16520
buffer_load_short_d16_hi v148, v8, [s12, s13, s14, s15], 0 idxen
ds_read_b64 [v88, v89], v1 offset:20680
buffer_load_short_d16_hi v145, v11, [s12, s13, s14, s15], 0 idxen
ds_write_b64 v3, [v156, v157] offset:25280
buffer_load_short_d16_hi v152, v9, [s12, s13, s14, s15], 0 idxen
buffer_load_short_d16_hi v149, v10, [s12, s13, s14, s15], 0 idxen
s_add_u32 s12, s12, s37
s_addc_u32 s13, s13, 0
s_bitcmp1_b32 s21, 5
s_cbranch_scc0 label_34
s_mov_b32 s36, label_34 - label_1
s_branch label_49
.align 32
label_46:
v_pk_add_f32 [v160, v161], [v18, v19], [v36, v37] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v158, v159], [v52, v53], -1.0, [v70, v71] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v160, v161], [v160, v161], [v52, v53] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v158, v159], [v158, v159], [v36, v37] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v18, 0
v_mov_b32 v19, 0
v_mov_b32 v36, 0
v_mov_b32 v37, 0
v_mov_b32 v52, 0
v_mov_b32 v53, 0
v_mov_b32 v70, 0
v_mov_b32 v71, 0
ds_write_b64 v1, [v160, v161] offset:4160
ds_write_b64 v1, [v158, v159] offset:8
v_pk_add_f32 [v160, v161], [v20, v21], [v38, v39] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v158, v159], [v54, v55], -1.0, [v72, v73] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v160, v161], [v160, v161], [v54, v55] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v158, v159], [v158, v159], [v38, v39] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v20, 0
v_mov_b32 v21, 0
v_mov_b32 v38, 0
v_mov_b32 v39, 0
v_mov_b32 v54, 0
v_mov_b32 v55, 0
v_mov_b32 v72, 0
v_mov_b32 v73, 0
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b64 v1, [v160, v161] offset:12416
ds_write_b64 v1, [v158, v159] offset:8264
s_add_u32 s48, s6, s36
s_addc_u32 s49, s7, 0
s_setpc_b64 [s48, s49]
.align 32
label_47:
v_pk_add_f32 [v160, v161], [v26, v27], [v44, v45] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v158, v159], [v60, v61], -1.0, [v78, v79] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v160, v161], [v160, v161], [v60, v61] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v158, v159], [v158, v159], [v44, v45] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v26, 0
v_mov_b32 v27, 0
v_mov_b32 v44, 0
v_mov_b32 v45, 0
v_mov_b32 v60, 0
v_mov_b32 v61, 0
v_mov_b32 v78, 0
v_mov_b32 v79, 0
ds_write_b64 v1, [v160, v161] offset:4160
ds_write_b64 v1, [v158, v159] offset:8
v_pk_add_f32 [v160, v161], [v28, v29], [v46, v47] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v158, v159], [v62, v63], -1.0, [v80, v81] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v160, v161], [v160, v161], [v62, v63] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v158, v159], [v158, v159], [v46, v47] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v28, 0
v_mov_b32 v29, 0
v_mov_b32 v46, 0
v_mov_b32 v47, 0
v_mov_b32 v62, 0
v_mov_b32 v63, 0
v_mov_b32 v80, 0
v_mov_b32 v81, 0
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b64 v1, [v160, v161] offset:12416
ds_write_b64 v1, [v158, v159] offset:8264
s_add_u32 s48, s6, s36
s_addc_u32 s49, s7, 0
s_setpc_b64 [s48, s49]
.align 32
label_48:
v_pk_add_f32 [v160, v161], [v22, v23], [v40, v41] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v158, v159], [v56, v57], -1.0, [v74, v75] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v160, v161], [v160, v161], [v56, v57] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v158, v159], [v158, v159], [v40, v41] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v22, 0
v_mov_b32 v23, 0
v_mov_b32 v40, 0
v_mov_b32 v41, 0
v_mov_b32 v56, 0
v_mov_b32 v57, 0
v_mov_b32 v74, 0
v_mov_b32 v75, 0
ds_write_b64 v1, [v160, v161] offset:4160
ds_write_b64 v1, [v158, v159] offset:8
v_pk_add_f32 [v160, v161], [v24, v25], [v42, v43] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v158, v159], [v58, v59], -1.0, [v76, v77] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v160, v161], [v160, v161], [v58, v59] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v158, v159], [v158, v159], [v42, v43] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v24, 0
v_mov_b32 v25, 0
v_mov_b32 v42, 0
v_mov_b32 v43, 0
v_mov_b32 v58, 0
v_mov_b32 v59, 0
v_mov_b32 v76, 0
v_mov_b32 v77, 0
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b64 v1, [v160, v161] offset:12416
ds_write_b64 v1, [v158, v159] offset:8264
s_add_u32 s48, s6, s36
s_addc_u32 s49, s7, 0
s_setpc_b64 [s48, s49]
.align 32
label_49:
v_pk_add_f32 [v160, v161], [v30, v31], [v48, v49] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v158, v159], [v64, v65], -1.0, [v82, v83] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v160, v161], [v160, v161], [v64, v65] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v158, v159], [v158, v159], [v48, v49] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v30, 0
v_mov_b32 v31, 0
v_mov_b32 v48, 0
v_mov_b32 v49, 0
v_mov_b32 v64, 0
v_mov_b32 v65, 0
v_mov_b32 v82, 0
v_mov_b32 v83, 0
ds_write_b64 v1, [v160, v161] offset:4160
ds_write_b64 v1, [v158, v159] offset:8
v_pk_add_f32 [v160, v161], [v32, v33], [v50, v51] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v158, v159], [v66, v67], -1.0, [v84, v85] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v160, v161], [v160, v161], [v66, v67] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v158, v159], [v158, v159], [v50, v51] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mov_b32 v32, 0
v_mov_b32 v33, 0
v_mov_b32 v50, 0
v_mov_b32 v51, 0
v_mov_b32 v66, 0
v_mov_b32 v67, 0
v_mov_b32 v84, 0
v_mov_b32 v85, 0
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b64 v1, [v160, v161] offset:12416
ds_write_b64 v1, [v158, v159] offset:8264
s_bitcmp1_b32 s21, 4
s_cmov_b32 s36, label_0 - label_1
s_add_u32 s48, s6, s36
s_addc_u32 s49, s7, 0
s_setpc_b64 [s48, s49]
label_5:
s_and_b32 s45, s45, 65535
s_add_u32 s45, s45, 131072
s_mov_b64 [s84, s85], [s44, s45]
s_mov_b32 s86, 2147483648
s_mov_b32 s87, 131072
s_bitcmp0_b32 s28, 7
s_cselect_b32 s86, 0, 2147483648
s_and_b32 s88, s5, 3
v_and_b32 v135, v0, 63
v_lshrrev_b32 v134, 1, v135
v_and_b32 v133, v135, 1
v_mad_u32_u24 v133, v134, 8, v133
v_lshlrev_b32 v133, 1, v133
v_mad_u32_u24 v133, 4, s88, v133
v_add_u32 v134, v133, 1
v_cmp_lt_u32 [vcc_lo, vcc_hi], v133, s26
v_cndmask_b32 v133, -1, v133, [vcc_lo, vcc_hi]
v_cmp_lt_u32 [vcc_lo, vcc_hi], v134, s26
v_cndmask_b32 v134, -1, v134, [vcc_lo, vcc_hi]
buffer_load_short_d16_hi v125, v133, [s84, s85, s86, s87], 0 idxen
buffer_load_short_d16_hi v129, v134, [s84, s85, s86, s87], 0 idxen
s_cmp_eq_u32 s5, 9
s_cselect_b32 s44, 1536, 0
s_cselect_b32 s45, 12, 0
s_cmp_ge_u32 s5, 10
s_cselect_b32 s44, 65536, s44
s_cselect_b32 s45, 65536, s45
s_cmp_eq_u32 s5, 9
s_cselect_b32 s84, 1024, 0
s_cselect_b32 s85, 4, 0
s_cmp_ge_u32 s5, 10
s_cselect_b32 s84, 65536, s84
s_cselect_b32 s85, 65536, s85
s_and_b32 s87, s5, 3
s_mul_i32 s86, s87, 2048
v_and_b32 v133, 63, v0
v_lshl_add_u32 v118, v133, 3, s44
v_lshl_add_u32 v119, v133, 3, 0
v_mov_b32 v120, s45
v_mov_b32 v121, 0
v_lshl_add_u32 v122, v133, 4, s84
v_mov_b32 v124, s85
v_lshl_add_u32 v123, v133, 4, s86
s_add_u32 s85, s23, 15
s_and_b32 s85, s85, -16
s_sub_u32 s85, s85, 16
s_sub_u32 s85, s23, s85
s_bitcmp1_b32 s28, 1
s_cselect_b32 s86, s39, 1
s_cselect_b32 s30, -1, 1
s_sub_u32 s84, s86, 1
s_bitcmp1_b32 s28, 0
s_cselect_b32 s86, s38, 1
s_cselect_b32 s29, -1, 1
s_sub_u32 s86, s86, 1
s_mul_i32 s29, s29, s62
s_mul_i32 s86, s86, s62
s_add_u32 s84, s84, s86
s_bitcmp1_b32 s28, 31
s_cselect_b32 s45, 0, 1
s_cselect_b32 s44, 4, 0
s_cselect_b32 s86, 2, 0
s_cselect_b32 s87, 2, 4
v_bfe_u32 v134, v0, 2, 6
v_and_b32 v135, s5, 3
v_bfe_u32 v133, v134, 0, s86
v_mad_u32_u24 v133, v135, s45, v133
v_mov_b32 v10, s87
v_bfe_u32 v10, v134, s86, v10
v_mad_u32_u24 v10, v135, s44, v10
v_mov_b32 v134, s84
v_mad_u32_u24 v5, s61, v133, v134
v_sub_u32 v21, s85, v133
v_add_u32 v17, v21, 3
v_add_u32 v21, v21, 3
s_add_u32 s45, s42, 1
s_lshr_b32 s45, s45, 1
s_lshl_b32 s61, s45, 1
s_add_u32 s62, s43, 1
s_lshr_b32 s62, s62, 1
s_lshl1_add_u32 s62, s62, 2
s_pack_ll_b32_b16 s3, s62, s45
s_pack_ll_b32_b16 s11, s25, s24
s_sub_u32 s12, s61, s40
s_sub_u32 s44, s62, s41
s_pack_ll_b32_b16 s12, s44, s12
s_pack_ll_b32_b16 s14, s43, s42
s_sub_u32 s44, s62, 1
s_pack_ll_b32_b16 s15, s44, s61
v_lshrrev_b32 v136, 16, s3
v_mov_b32 v151, 65535
v_bfi_b32 v137, v151, s3, 0
v_and_b32 v139, 1, v0
v_bfe_u32 v145, v0, 6, 1
v_and_b32 v134, 63, v0
v_mov_b32 v151, 124
v_mad_u32_u16 v140, v151, s8, 0 op_sel:[0, 0, 0, 0]
v_mad_u32_u16 v145, 2, s9, v145 op_sel:[0, 0, 0, 0]
v_mad_u32_u16 v138, v136, v137, 0 op_sel:[0, 0, 0, 0]
v_cmp_eq_u32 [vcc_lo, vcc_hi], 0, v139
v_cndmask_b32 v146, v138, v137, [vcc_lo, vcc_hi]
v_mad_u32_u16 v135, 62, v145, v134 op_sel:[0, 0, 0, 0]
v_cndmask_b32 v135, v140, v135, [vcc_lo, vcc_hi]
v_ffbh_u32 v152, v146
v_lshlrev_b32 v153, v152, v146
v_and_b32 v154, -256, v153
v_cmp_eq_u32 [vcc_lo, vcc_hi], 2147483648, v153
v_cvt_f32_u32 v154, v154
v_rcp_f32 v147, v154
v_subb_co_u32 v148, [vcc_lo, vcc_hi], 32, v152, [vcc_lo, vcc_hi]
v_cvt_f32_ubyte0 v152, v153
v_fma_f32 v154, v154, v147, -1.0
v_fma_f32 v154, v152, v147, v154
v_fmaak_f32 v154, v154, v147, 2667577344
v_mul_f32 v154, 1602224128, v154
v_mov_b32 v152, 0
v_cvt_flr_i32_f32 v154, neg(v154)
v_lshl_add_u32 v147, v147, 9, v154
v_mad_u64_u32 [v152, v153], [vcc_lo, vcc_hi], v153, v147, [v152, v153]
v_subb_co_u32 v147, [vcc_lo, vcc_hi], v147, -1, [vcc_lo, vcc_hi]
v_mov_b32 v150, v148 quad_perm:[1, 1, 1, 1]
v_mov_b32 v148, v148 quad_perm:[0, 0, 0, 0]
v_mov_b32 v149, v147 quad_perm:[1, 1, 1, 1]
v_mov_b32 v147, v147 quad_perm:[0, 0, 0, 0]
v_mul_hi_u32 v152, v135, v147
v_add_co_u32 v133, [vcc_lo, vcc_hi], v152, v135
v_addc_co_u32 v152, [vcc_lo, vcc_hi], 0, 0, [vcc_lo, vcc_hi]
v_cmp_eq_u32 [vcc_lo, vcc_hi], 32, v148
v_cndmask_b32 v133, v133, v152, [vcc_lo, vcc_hi]
v_alignbit_b32 v133, v152, v133, v148
v_mul_hi_u32 v152, v135, v149
v_add_co_u32 v4, [vcc_lo, vcc_hi], v152, v135
v_addc_co_u32 v152, [vcc_lo, vcc_hi], 0, 0, [vcc_lo, vcc_hi]
v_cmp_eq_u32 [vcc_lo, vcc_hi], 32, v150
v_cndmask_b32 v4, v4, v152, [vcc_lo, vcc_hi]
v_alignbit_b32 v4, v152, v4, v150
v_mad_u32_u16 v144, v133, v137, 0 op_sel:[0, 0, 0, 0]
v_mad_u32_u16 v143, v4, v136, 0 op_sel:[0, 0, 0, 0]
v_sub_u32 v144, v135, v144
v_sub_u32 v143, v133, v143
v_readlane_b32 s84, v144, 1
v_sub_u32 v144, v144, v137
v_readlane_b32 s10, v143, 1
v_sub_u32 v143, v143, v136
v_readlane_b32 s2, v4, 1
v_sub_u32 v4, v4, s22
s_lshl_b32 s10, s10, 16
s_and_b32 s84, s84, lit(0xffff)
s_add_u32 s10, s10, s84
v_mov_b32 v144, v144 quad_perm:[0, 0, 2, 2]
v_mov_b32 v143, v143 quad_perm:[0, 0, 2, 2]
v_mov_b32 v4, v4 quad_perm:[0, 0, 2, 2]
v_add_co_u32 v144, [vcc_lo, vcc_hi], v144, v139
v_cndmask_b32 v142, 0, v137, [vcc_lo, vcc_hi]
v_addc_co_u32 v143, [vcc_lo, vcc_hi], v143, 0, [vcc_lo, vcc_hi]
v_cndmask_b32 v141, 0, v136, [vcc_lo, vcc_hi]
v_addc_co_u32 v4, [vcc_lo, vcc_hi], v4, 0, [vcc_lo, vcc_hi]
v_min_u32 v139, v134, 63
v_sub_u32 v144, v144, v142
v_sub_u32 v143, v143, v141
v_cmp_eq_u32 [vcc_lo, vcc_hi], v134, v139
v_lshlrev_b32 v1, 16, v143
v_mov_b32 v151, 65535
v_bfi_b32 v1, v151, v144, v1
v_add_u32 v155, v4, s22
v_med3_u32 v139, v134, 1, 62
v_mul_lo_u32 v2, v155, s56
v_mul_lo_u32 v11, v155, s64
s_mul_i32 s13, s2, s56
s_mul_i32 s16, s2, s64
v_mov_b32 v151, 2147483648
v_cndmask_b32 v2, v151, v2, [vcc_lo, vcc_hi]
v_cmp_eq_u32 [vcc_lo, vcc_hi], v134, v139
v_cndmask_b32 v11, v151, v11, [vcc_lo, vcc_hi]
v_cmp_ge_u32 [s20, s21], v155, s22
v_cndmask_b32 v2, v2, v151, [s20, s21]
v_cndmask_b32 v11, v11, v151, [s20, s21]
s_and_b32 s24, s72, 255
s_cmp_eq_u32 s24, 0
s_cmov_b32 s46, 1.0
s_cbranch_scc1 label_118
s_cmp_eq_u32 s24, 4
s_cbranch_scc1 label_120
s_cmp_eq_u32 s24, 2
s_cbranch_scc1 label_121
s_cmp_eq_u32 s24, 3
s_cbranch_scc1 label_122
v_cmp_gt_f32 [vcc_lo, vcc_hi], s46, 1.0
s_cbranch_vccnz label_119
label_118:
s_mov_b32 s83, label_98 - label_1
s_branch label_123
label_119:
s_mov_b32 s83, label_102 - label_1
s_branch label_123
label_120:
s_mov_b32 s83, label_106 - label_1
s_branch label_123
label_121:
s_mov_b32 s83, label_110 - label_1
s_branch label_123
label_122:
s_mov_b32 s83, label_114 - label_1
label_123:
s_lshl_b32 s73, s65, 1
s_mul_i32 s74, s73, 16
s_mul_hi_u32 s75, s73, 16
s_and_b32 s24, s5, 3
s_lshl_b32 s25, s73, 2
s_mul_i32 s25, s25, s24
s_add_u32 s34, s34, s25
s_addc_u32 s35, s35, 0
s_lshl_b32 s77, s24, 2
s_sub_u32 s77, s26, s77
s_cmov_b32 s77, 0
s_and_b32 s35, s35, 65535
s_add_u32 s35, s35, 131072
s_mov_b64 [s68, s69], [s34, s35]
s_mov_b32 s70, 0
s_mov_b32 s71, 131072
s_mov_b32 s17, 65536
s_mov_b32 s18, 196610
s_mov_b32 s19, 65536
s_add_u32 s31, s38, 2
s_add_u32 s32, s39, 2
s_mul_hi_u32 s25, s31, 1431655766
s_add_u32 s31, s25, s31
s_addc_u32 s25, 0, 0
s_mov_b32 s24, s31
s_lshr_b64 [s24, s25], [s24, s25], 2
s_mov_b32 s31, s24
s_mul_hi_u32 s25, s32, 1431655766
s_add_u32 s32, s25, s32
s_addc_u32 s25, 0, 0
s_mov_b32 s24, s32
s_lshr_b64 [s24, s25], [s24, s25], 2
s_mov_b32 s32, s24
s_sub_u32 s31, 0, s31
s_sub_u32 s32, 0, s32
s_mov_b32 s33, -1
s_mov_b32 s36, -1
s_waitcnt vmcnt(0)
v_readlane_b32 s37, v125, 0
v_readlane_b32 s48, v129, 0
v_readlane_b32 s49, v125, 1
v_readlane_b32 s50, v129, 1
v_readlane_b32 s51, v125, 2
v_readlane_b32 s52, v129, 2
v_readlane_b32 s53, v125, 3
v_readlane_b32 s54, v129, 3
s_mov_b32 s78, 0
s_mov_b32 s79, 0
s_mov_b32 s55, 2
s_lshl_b32 s59, s55, 9
s_lshl_b32 s63, s55, 2
s_branch label_97
.align 32
label_94:
s_sub_u32 s79, s23, 1
s_add_u32 s80, s6, s82
s_addc_u32 s81, s7, 0
s_setprio 3
s_add_u32 s36, s36, 1
s_cselect_b32 s36, s32, s36
s_addc_u32 s33, s33, 0
s_cselect_b32 s33, s31, s33
s_cbranch_scc0 label_124
s_bitcmp1_b32 s20, 0
s_addc_u32 s78, s78, 2
v_add_u32 v125, s59, v118
v_add_u32 v129, s63, v120
v_and_b32 v132, s78, 3
ds_write_b64 v125, [v12, v13] offset:55816
ds_write_b32 v129, v132 offset:58888
label_124:
s_sub_u32 s24, -1, s33
s_sub_u32 s25, -1, s36
s_mul_i32 s24, 3, s24
s_mul_i32 s25, 3, s25
s_pack_ll_b32_b16 s40, s25, s24
s_cmp_eq_u32 s40, 0
s_cbranch_scc1 label_126
label_125:
v_mov_b32 v129, 2147483648
v_mov_b32 v135, 131073
v_pk_mad_u16 v135, v1, v135, s12 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_u16 v135, s40, v135 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_mad_u32_u16 v6, v135, 1, v2 op_sel:[0, 0, 0, 0]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v135, s11
v_cndmask_b32 v6, v129, v6, [vcc_lo, vcc_hi]
v_pk_add_u16 v125, v135, s18 op_sel:[1, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_u16 v136, v135, s17 op_sel:[1, 0, 0] op_sel_hi:[1, 1, 1]
v_mad_u32_u16 v9, v125, s58, v6 op_sel:[1, 0, 0, 0]
v_mad_u32_u16 v7, v136, s58, v6 op_sel:[1, 0, 0, 0]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v125, s11 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v9, v129, v9, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v136, s11 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v7, v129, v7, [vcc_lo, vcc_hi]
v_mad_u32_u16 v8, v125, s58, v6 op_sel:[0, 0, 0, 0]
v_mad_u32_u16 v6, v136, s58, v6 op_sel:[0, 0, 0, 0]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v125, s11 src0_sel:WORD_0 src1_sel:WORD_1
v_cndmask_b32 v8, v129, v8, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v136, s11 src0_sel:WORD_0 src1_sel:WORD_1
v_cndmask_b32 v6, v129, v6, [vcc_lo, vcc_hi]
s_branch label_127
label_126:
v_mov_b32 v139, 2147483648
v_mov_b32 v140, s56
v_mov_b32 v129, s64
v_mov_b32 v138, s2
v_add_co_u32 v125, [vcc_lo, vcc_hi], v1, s10
v_mov_b32 v136, 131073
v_pk_mad_u16 v135, v1, v136, s12 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_mad_u16 v136, v1, v136, s15 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_min_u16 v134, v125, v1 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_cndmask_b32 v133, 0, v140, [vcc_lo, vcc_hi]
v_cndmask_b32 v137, 0, v129, [vcc_lo, vcc_hi]
v_mad_u32_u16 v6, v135, 1, v2 op_sel:[0, 0, 0, 0]
v_mad_u32_u16 v12, v136, 1, v11 op_sel:[0, 0, 0, 0]
v_add3_u32 v2, v2, s13, v133
v_add3_u32 v11, v11, s16, v137
v_pk_sub_u16 v134, v1, v134 op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_addc_co_u32 v4, [s20, s21], v4, v138, [vcc_lo, vcc_hi]
v_cndmask_b32 v2, v2, v139, [s20, s21]
v_cndmask_b32 v11, v11, v139, [s20, s21]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v135, s11
v_cndmask_b32 v6, v139, v6, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v136, s14
v_cndmask_b32 v12, v139, v12, [vcc_lo, vcc_hi]
v_pk_ashrrev_i16 v134, 15, v134 op_sel:[0, 0, 0] op_sel_hi:[0, 1, 1]
v_pk_add_u16 v129, v135, s18 op_sel:[1, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_u16 v140, v135, s17 op_sel:[1, 0, 0] op_sel_hi:[1, 1, 1]
v_mad_u32_u16 v9, v129, s58, v6 op_sel:[1, 0, 0, 0]
v_mad_u32_u16 v7, v140, s58, v6 op_sel:[1, 0, 0, 0]
v_pk_add_u16 v138, v136, s19 op_sel:[1, 0, 0] op_sel_hi:[1, 1, 1]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v129, s11 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v9, v139, v9, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v140, s11 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v7, v139, v7, [vcc_lo, vcc_hi]
v_mad_u32_u16 v13, v138, s66, v12 op_sel:[1, 0, 0, 0]
v_mad_u32_u16 v8, v129, s58, v6 op_sel:[0, 0, 0, 0]
v_mad_u32_u16 v6, v140, s58, v6 op_sel:[0, 0, 0, 0]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v138, s14 src0_sel:WORD_1 src1_sel:WORD_1
v_cndmask_b32 v13, v139, v13, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v129, s11 src0_sel:WORD_0 src1_sel:WORD_1
v_cndmask_b32 v8, v139, v8, [vcc_lo, vcc_hi]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v140, s11 src0_sel:WORD_0 src1_sel:WORD_1
v_cndmask_b32 v6, v139, v6, [vcc_lo, vcc_hi]
v_mad_u32_u16 v12, v138, s66, v12 op_sel:[0, 0, 0, 0]
v_pk_mad_u16 v1, v134, s3, v125 op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_cmp_lt_u16 [vcc_lo, vcc_hi], v138, s14 src0_sel:WORD_0 src1_sel:WORD_1
v_cndmask_b32 v12, v139, v12, [vcc_lo, vcc_hi]
label_127:
v_and_b32 v132, s78, 3
v_xor_b32 v125, v0, v0 quad_perm:[0, 0, 0, 1]
v_add_u32 v125, s25, v125
v_cmp_lt_u32 [s44, s45], v125, s39
v_mul_i32_i24 v125, s30, v125
s_mul_i32 s41, s29, s24
s_sub_u32 s42, s38, 1
s_sub_u32 s42, s42, s24
s_cselect_b64 [s44, s45], 0, [s44, s45]
s_sub_u32 s42, s42, 1
s_cselect_b64 [s84, s85], 0, [s44, s45]
s_sub_u32 s42, s42, 1
s_cselect_b64 [s86, s87], 0, [s84, s85]
v_mov_b32 v129, 3221225472
v_add3_u32 v133, v125, s41, v5
v_mad_u32_u24 v14, s60, v10, v133
v_cmp_lt_u32 [vcc_lo, vcc_hi], v10, s26
v_cndmask_b32 v14, v129, v14, [vcc_lo, vcc_hi]
v_add_u32 v15, s29, v14
v_add_u32 v16, s29, v15
v_cndmask_b32 v14, v129, v14, [s44, s45]
v_cndmask_b32 v15, v129, v15, [s84, s85]
v_cndmask_b32 v16, v129, v16, [s86, s87]
v_add_u32 v134, v10, 16
v_mad_u32_u24 v18, s60, v134, v133
v_cmp_lt_u32 [vcc_lo, vcc_hi], v134, s26
v_cndmask_b32 v18, v129, v18, [vcc_lo, vcc_hi]
v_add_u32 v19, s29, v18
v_add_u32 v20, s29, v19
v_cndmask_b32 v18, v129, v18, [s44, s45]
v_cndmask_b32 v19, v129, v19, [s84, s85]
v_cndmask_b32 v20, v129, v20, [s86, s87]
ds_write_b128 v123, [v14, v15, v16, v17] offset:45568
ds_write_b128 v123, [v18, v19, v20, v21] offset:46592
ds_write_b128 v122, [v6, v7, v8, v9] offset:53760
ds_write_b32 v124, v132 offset:55808
s_setpc_b64 [s80, s81]
label_97:
label_128:
s_barrier
s_lshl_b32 s78, s78, 2
s_mov_b32 s82, label_130 - label_1
s_sub_u32 s79, s79, 16
s_cbranch_scc1 label_94
label_130:
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s78, 7
s_cselect_b32 s80, s83, label_129 - label_1
s_add_u32 s80, s6, s80
s_addc_u32 s81, s7, 0
s_sub_u32 s55, s55, 1
s_cselect_b32 s55, 2, s55
s_lshl_b32 s59, s55, 9
s_lshl_b32 s63, s55, 2
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s80, s81], [s80, s81]
label_129:
s_bitcmp1_b32 s78, 5
s_cbranch_scc1 label_95
label_96:
s_barrier
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s78, 7
s_cselect_b32 s80, s83, label_131 - label_1
s_add_u32 s80, s6, s80
s_addc_u32 s81, s7, 0
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s80, s81], [s80, s81]
label_131:
s_barrier
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s78, 7
s_cselect_b32 s80, s83, label_132 - label_1
s_add_u32 s80, s6, s80
s_addc_u32 s81, s7, 0
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s80, s81], [s80, s81]
label_132:
s_barrier
s_setprio 0
s_setprio 2
s_bitcmp1_b32 s78, 5
s_cselect_b32 s80, s83, label_133 - label_1
s_add_u32 s80, s6, s80
s_addc_u32 s81, s7, 0
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s80, s81], [s80, s81]
label_133:
s_branch label_128
label_134:
s_barrier
s_lshl_b32 s78, s78, 2
s_mov_b32 s82, label_136 - label_1
s_sub_u32 s79, s79, 16
s_cbranch_scc1 label_94
label_136:
s_setprio 0
ds_read_b64 [v60, v61], v3 offset:0
ds_read_b64 [v58, v59], v3 offset:256
ds_read_b64 [v68, v69], v3 offset:512
ds_read_b64 [v66, v67], v3 offset:768
s_setprio 3
s_sub_u32 s55, s55, 1
s_cselect_b32 s55, 2, s55
s_lshl_b32 s59, s55, 9
s_lshl_b32 s63, s55, 2
s_waitcnt lgkmcnt(0)
v_pk_add_f32 [v106, v107], [v60, v61], [v58, v59] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v106, v107], [v106, v107], [v68, v69] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v108, v109], [v68, v69], -1.0, [v58, v59] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v108, v109], [v108, v109], [v66, v67] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_barrier
s_setprio 0
ds_read_b64 [v76, v77], v3 offset:8256
ds_read_b64 [v74, v75], v3 offset:8512
ds_read_b64 [v84, v85], v3 offset:8768
ds_read_b64 [v82, v83], v3 offset:9024
s_setprio 3
s_bitcmp1_b32 s78, 7
s_cselect_b32 s80, s83, label_137 - label_1
s_add_u32 s80, s6, s80
s_addc_u32 s81, s7, 0
s_waitcnt lgkmcnt(0)
v_pk_add_f32 [v114, v115], [v76, v77], [v74, v75] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v114, v115], [v114, v115], [v84, v85] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v116, v117], [v84, v85], -1.0, [v74, v75] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v116, v117], [v116, v117], [v82, v83] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_swappc_b64 [s80, s81], [s80, s81]
label_137:
s_bitcmp1_b32 s78, 5
s_cbranch_scc0 label_96
label_95:
s_barrier
s_setprio 0
ds_read_b64 [v24, v25], v3 offset:0
ds_read_b64 [v22, v23], v3 offset:256
ds_read_b64 [v32, v33], v3 offset:512
ds_read_b64 [v30, v31], v3 offset:768
s_setprio 3
s_waitcnt lgkmcnt(0)
v_pk_add_f32 [v86, v87], [v24, v25], [v22, v23] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v86, v87], [v86, v87], [v32, v33] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v88, v89], [v32, v33], -1.0, [v22, v23] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v88, v89], [v88, v89], [v30, v31] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_barrier
s_setprio 0
ds_read_b64 [v40, v41], v3 offset:8256
ds_read_b64 [v38, v39], v3 offset:8512
ds_read_b64 [v48, v49], v3 offset:8768
ds_read_b64 [v46, v47], v3 offset:9024
s_setprio 3
s_bitcmp1_b32 s78, 7
s_cselect_b32 s80, s83, label_139 - label_1
s_add_u32 s80, s6, s80
s_addc_u32 s81, s7, 0
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s80, s81], [s80, s81]
label_139:
v_pk_add_f32 [v94, v95], [v40, v41], [v38, v39] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v94, v95], [v94, v95], [v48, v49] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v96, v97], [v48, v49], -1.0, [v38, v39] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v96, v97], [v96, v97], [v46, v47] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_barrier
s_setprio 0
ds_read_b64 [v56, v57], v3 offset:0
ds_read_b64 [v54, v55], v3 offset:256
ds_read_b64 [v64, v65], v3 offset:512
ds_read_b64 [v62, v63], v3 offset:768
s_setprio 3
s_waitcnt lgkmcnt(0)
v_pk_add_f32 [v102, v103], [v56, v57], [v54, v55] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v102, v103], [v102, v103], [v64, v65] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v104, v105], [v64, v65], -1.0, [v54, v55] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v104, v105], [v104, v105], [v62, v63] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_barrier
s_setprio 0
ds_read_b64 [v72, v73], v3 offset:8256
ds_read_b64 [v70, v71], v3 offset:8512
ds_read_b64 [v80, v81], v3 offset:8768
ds_read_b64 [v78, v79], v3 offset:9024
s_setprio 3
s_bitcmp1_b32 s78, 7
s_cselect_b32 s80, s83, label_141 - label_1
s_add_u32 s80, s6, s80
s_addc_u32 s81, s7, 0
s_waitcnt lgkmcnt(0)
s_swappc_b64 [s80, s81], [s80, s81]
label_141:
v_pk_add_f32 [v110, v111], [v72, v73], [v70, v71] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v110, v111], [v110, v111], [v80, v81] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v112, v113], [v80, v81], -1.0, [v70, v71] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v112, v113], [v112, v113], [v78, v79] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_add_u32 v131, s59, v119
v_add_u32 v128, s63, v121
s_barrier
s_setprio 0
ds_read_b64 [v126, v127], v131 offset:55816
ds_read_b64 [v130, v131], v131 offset:57352
ds_read_b32 v128, v128 offset:58888
ds_read_b64 [v28, v29], v3 offset:0
ds_read_b64 [v26, v27], v3 offset:256
ds_read_b64 [v36, v37], v3 offset:512
ds_read_b64 [v34, v35], v3 offset:768
s_setprio 3
s_waitcnt lgkmcnt(0)
v_pk_add_f32 [v90, v91], [v28, v29], [v26, v27] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v90, v91], [v90, v91], [v36, v37] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v92, v93], [v36, v37], -1.0, [v26, v27] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v92, v93], [v92, v93], [v34, v35] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_barrier
s_setprio 0
ds_read_b64 [v44, v45], v3 offset:8256
ds_read_b64 [v42, v43], v3 offset:8512
ds_read_b64 [v52, v53], v3 offset:8768
ds_read_b64 [v50, v51], v3 offset:9024
s_setprio 3
s_bitcmp1_b32 s78, 5
s_cselect_b32 s80, s83, label_143 - label_1
s_add_u32 s80, s6, s80
s_addc_u32 s81, s7, 0
s_waitcnt lgkmcnt(0)
v_pk_add_f32 [v98, v99], [v44, v45], [v42, v43] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_add_f32 [v98, v99], [v98, v99], [v52, v53] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
v_pk_fma_f32 [v100, v101], [v52, v53], -1.0, [v42, v43] op_sel:[0, 0, 0, 0] op_sel_hi:[1, 1, 1, 1]
v_pk_add_f32 [v100, v101], [v100, v101], [v50, v51] op_sel:[0, 0, 0] op_sel_hi:[1, 1, 1]
s_swappc_b64 [s80, s81], [s80, s81]
label_143:
s_branch label_134
label_98:
s_setprio 3
v_readfirstlane_b32 s67, v128
v_add_f32 v86, v86, s37
v_add_f32 v87, v87, s48
v_add_f32 v90, v90, s37
v_add_f32 v91, v91, s48
v_add_f32 v88, v88, s37
v_add_f32 v89, v89, s48
v_add_f32 v92, v92, s37
v_add_f32 v93, v93, s48
v_mul_f32 v125, s46, v86
v_mul_f32 v129, s46, v87
v_mul_f32 v133, s46, v90
v_mul_f32 v134, s46, v91
v_max_f32 v86, v86, v125
v_max_f32 v87, v87, v129
v_max_f32 v90, v90, v133
v_max_f32 v91, v91, v134
v_mul_f32 v125, s46, v88
v_mul_f32 v129, s46, v89
v_mul_f32 v133, s46, v92
v_mul_f32 v134, s46, v93
v_max_f32 v88, v88, v125
v_max_f32 v89, v89, v129
v_max_f32 v92, v92, v133
v_max_f32 v93, v93, v134
s_setprio 0
buffer_store_short_d16_hi v86, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v90, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v88, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v92, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v87, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v91, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v89, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v93, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_99 - label_1
s_setpc_b64 [s80, s81]
label_99:
s_setprio 3
v_add_f32 v94, v94, s49
v_add_f32 v95, v95, s50
v_add_f32 v98, v98, s49
v_add_f32 v99, v99, s50
v_add_f32 v96, v96, s49
v_add_f32 v97, v97, s50
v_add_f32 v100, v100, s49
v_add_f32 v101, v101, s50
v_mul_f32 v125, s46, v94
v_mul_f32 v129, s46, v95
v_mul_f32 v133, s46, v98
v_mul_f32 v134, s46, v99
v_max_f32 v94, v94, v125
v_max_f32 v95, v95, v129
v_max_f32 v98, v98, v133
v_max_f32 v99, v99, v134
v_mul_f32 v125, s46, v96
v_mul_f32 v129, s46, v97
v_mul_f32 v133, s46, v100
v_mul_f32 v134, s46, v101
v_max_f32 v96, v96, v125
v_max_f32 v97, v97, v129
v_max_f32 v100, v100, v133
v_max_f32 v101, v101, v134
s_setprio 0
buffer_store_short_d16_hi v94, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v98, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v96, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v100, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v95, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v99, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v97, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v101, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_100 - label_1
s_sub_u32 s76, s77, 17
s_cselect_b32 s70, 0, s70
s_add_u32 s68, s34, s74
s_addc_u32 s69, s35, s75
s_setpc_b64 [s80, s81]
label_100:
s_setprio 3
v_add_f32 v102, v102, s51
v_add_f32 v103, v103, s52
v_add_f32 v106, v106, s51
v_add_f32 v107, v107, s52
v_add_f32 v104, v104, s51
v_add_f32 v105, v105, s52
v_add_f32 v108, v108, s51
v_add_f32 v109, v109, s52
v_mul_f32 v125, s46, v102
v_mul_f32 v129, s46, v103
v_mul_f32 v133, s46, v106
v_mul_f32 v134, s46, v107
v_max_f32 v102, v102, v125
v_max_f32 v103, v103, v129
v_max_f32 v106, v106, v133
v_max_f32 v107, v107, v134
v_mul_f32 v125, s46, v104
v_mul_f32 v129, s46, v105
v_mul_f32 v133, s46, v108
v_mul_f32 v134, s46, v109
v_max_f32 v104, v104, v125
v_max_f32 v105, v105, v129
v_max_f32 v108, v108, v133
v_max_f32 v109, v109, v134
s_setprio 0
buffer_store_short_d16_hi v102, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v106, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v104, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v108, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v103, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v107, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v105, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v109, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_101 - label_1
s_setpc_b64 [s80, s81]
label_101:
s_setprio 3
v_add_f32 v110, v110, s53
v_add_f32 v111, v111, s54
v_add_f32 v114, v114, s53
v_add_f32 v115, v115, s54
v_add_f32 v112, v112, s53
v_add_f32 v113, v113, s54
v_add_f32 v116, v116, s53
v_add_f32 v117, v117, s54
v_mul_f32 v125, s46, v110
v_mul_f32 v129, s46, v111
v_mul_f32 v133, s46, v114
v_mul_f32 v134, s46, v115
v_max_f32 v110, v110, v125
v_max_f32 v111, v111, v129
v_max_f32 v114, v114, v133
v_max_f32 v115, v115, v134
v_mul_f32 v125, s46, v112
v_mul_f32 v129, s46, v113
v_mul_f32 v133, s46, v116
v_mul_f32 v134, s46, v117
v_max_f32 v112, v112, v125
v_max_f32 v113, v113, v129
v_max_f32 v116, v116, v133
v_max_f32 v117, v117, v134
s_setprio 0
buffer_store_short_d16_hi v110, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v114, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v112, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v116, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v111, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v115, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v113, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v117, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_98 - label_1
s_bitcmp1_b32 s67, 0
s_cbranch_scc1 label_0
s_sub_u32 s76, s77, 1
s_cselect_b32 s70, 0, 2147483648
s_mov_b64 [s68, s69], [s34, s35]
s_setpc_b64 [s80, s81]
label_102:
s_setprio 3
v_readfirstlane_b32 s67, v128
v_add_f32 v86, v86, s37
v_add_f32 v87, v87, s48
v_add_f32 v90, v90, s37
v_add_f32 v91, v91, s48
v_add_f32 v88, v88, s37
v_add_f32 v89, v89, s48
v_add_f32 v92, v92, s37
v_add_f32 v93, v93, s48
v_mul_f32 v125, s46, v86
v_mul_f32 v129, s46, v87
v_mul_f32 v133, s46, v90
v_mul_f32 v134, s46, v91
v_min_f32 v86, v86, v125
v_min_f32 v87, v87, v129
v_min_f32 v90, v90, v133
v_min_f32 v91, v91, v134
v_mul_f32 v125, s46, v88
v_mul_f32 v129, s46, v89
v_mul_f32 v133, s46, v92
v_mul_f32 v134, s46, v93
v_min_f32 v88, v88, v125
v_min_f32 v89, v89, v129
v_min_f32 v92, v92, v133
v_min_f32 v93, v93, v134
s_setprio 0
buffer_store_short_d16_hi v86, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v90, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v88, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v92, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v87, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v91, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v89, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v93, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_103 - label_1
s_setpc_b64 [s80, s81]
label_103:
s_setprio 3
v_add_f32 v94, v94, s49
v_add_f32 v95, v95, s50
v_add_f32 v98, v98, s49
v_add_f32 v99, v99, s50
v_add_f32 v96, v96, s49
v_add_f32 v97, v97, s50
v_add_f32 v100, v100, s49
v_add_f32 v101, v101, s50
v_mul_f32 v125, s46, v94
v_mul_f32 v129, s46, v95
v_mul_f32 v133, s46, v98
v_mul_f32 v134, s46, v99
v_min_f32 v94, v94, v125
v_min_f32 v95, v95, v129
v_min_f32 v98, v98, v133
v_min_f32 v99, v99, v134
v_mul_f32 v125, s46, v96
v_mul_f32 v129, s46, v97
v_mul_f32 v133, s46, v100
v_mul_f32 v134, s46, v101
v_min_f32 v96, v96, v125
v_min_f32 v97, v97, v129
v_min_f32 v100, v100, v133
v_min_f32 v101, v101, v134
s_setprio 0
buffer_store_short_d16_hi v94, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v98, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v96, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v100, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v95, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v99, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v97, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v101, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_104 - label_1
s_sub_u32 s76, s77, 17
s_cselect_b32 s70, 0, s70
s_add_u32 s68, s34, s74
s_addc_u32 s69, s35, s75
s_setpc_b64 [s80, s81]
label_104:
s_setprio 3
v_add_f32 v102, v102, s51
v_add_f32 v103, v103, s52
v_add_f32 v106, v106, s51
v_add_f32 v107, v107, s52
v_add_f32 v104, v104, s51
v_add_f32 v105, v105, s52
v_add_f32 v108, v108, s51
v_add_f32 v109, v109, s52
v_mul_f32 v125, s46, v102
v_mul_f32 v129, s46, v103
v_mul_f32 v133, s46, v106
v_mul_f32 v134, s46, v107
v_min_f32 v102, v102, v125
v_min_f32 v103, v103, v129
v_min_f32 v106, v106, v133
v_min_f32 v107, v107, v134
v_mul_f32 v125, s46, v104
v_mul_f32 v129, s46, v105
v_mul_f32 v133, s46, v108
v_mul_f32 v134, s46, v109
v_min_f32 v104, v104, v125
v_min_f32 v105, v105, v129
v_min_f32 v108, v108, v133
v_min_f32 v109, v109, v134
s_setprio 0
buffer_store_short_d16_hi v102, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v106, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v104, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v108, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v103, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v107, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v105, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v109, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_105 - label_1
s_setpc_b64 [s80, s81]
label_105:
s_setprio 3
v_add_f32 v110, v110, s53
v_add_f32 v111, v111, s54
v_add_f32 v114, v114, s53
v_add_f32 v115, v115, s54
v_add_f32 v112, v112, s53
v_add_f32 v113, v113, s54
v_add_f32 v116, v116, s53
v_add_f32 v117, v117, s54
v_mul_f32 v125, s46, v110
v_mul_f32 v129, s46, v111
v_mul_f32 v133, s46, v114
v_mul_f32 v134, s46, v115
v_min_f32 v110, v110, v125
v_min_f32 v111, v111, v129
v_min_f32 v114, v114, v133
v_min_f32 v115, v115, v134
v_mul_f32 v125, s46, v112
v_mul_f32 v129, s46, v113
v_mul_f32 v133, s46, v116
v_mul_f32 v134, s46, v117
v_min_f32 v112, v112, v125
v_min_f32 v113, v113, v129
v_min_f32 v116, v116, v133
v_min_f32 v117, v117, v134
s_setprio 0
buffer_store_short_d16_hi v110, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v114, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v112, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v116, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v111, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v115, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v113, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v117, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_102 - label_1
s_bitcmp1_b32 s67, 0
s_cbranch_scc1 label_0
s_sub_u32 s76, s77, 1
s_cselect_b32 s70, 0, 2147483648
s_mov_b64 [s68, s69], [s34, s35]
s_setpc_b64 [s80, s81]
label_106:
s_setprio 3
v_readfirstlane_b32 s67, v128
v_add_f32 v86, v86, s37
v_add_f32 v87, v87, s48
v_add_f32 v90, v90, s37
v_add_f32 v91, v91, s48
v_add_f32 v88, v88, s37
v_add_f32 v89, v89, s48
v_add_f32 v92, v92, s37
v_add_f32 v93, v93, s48
v_max_f32 v86, 0, v86
v_max_f32 v87, 0, v87
v_max_f32 v90, 0, v90
v_max_f32 v91, 0, v91
v_max_f32 v88, 0, v88
v_max_f32 v89, 0, v89
v_max_f32 v92, 0, v92
v_max_f32 v93, 0, v93
s_setprio 0
buffer_store_short_d16_hi v86, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v90, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v88, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v92, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v87, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v91, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v89, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v93, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_107 - label_1
s_setpc_b64 [s80, s81]
label_107:
s_setprio 3
v_add_f32 v94, v94, s49
v_add_f32 v95, v95, s50
v_add_f32 v98, v98, s49
v_add_f32 v99, v99, s50
v_add_f32 v96, v96, s49
v_add_f32 v97, v97, s50
v_add_f32 v100, v100, s49
v_add_f32 v101, v101, s50
v_max_f32 v94, 0, v94
v_max_f32 v95, 0, v95
v_max_f32 v98, 0, v98
v_max_f32 v99, 0, v99
v_max_f32 v96, 0, v96
v_max_f32 v97, 0, v97
v_max_f32 v100, 0, v100
v_max_f32 v101, 0, v101
s_setprio 0
buffer_store_short_d16_hi v94, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v98, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v96, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v100, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v95, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v99, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v97, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v101, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_108 - label_1
s_sub_u32 s76, s77, 17
s_cselect_b32 s70, 0, s70
s_add_u32 s68, s34, s74
s_addc_u32 s69, s35, s75
s_setpc_b64 [s80, s81]
label_108:
s_setprio 3
v_add_f32 v102, v102, s51
v_add_f32 v103, v103, s52
v_add_f32 v106, v106, s51
v_add_f32 v107, v107, s52
v_add_f32 v104, v104, s51
v_add_f32 v105, v105, s52
v_add_f32 v108, v108, s51
v_add_f32 v109, v109, s52
v_max_f32 v102, 0, v102
v_max_f32 v103, 0, v103
v_max_f32 v106, 0, v106
v_max_f32 v107, 0, v107
v_max_f32 v104, 0, v104
v_max_f32 v105, 0, v105
v_max_f32 v108, 0, v108
v_max_f32 v109, 0, v109
s_setprio 0
buffer_store_short_d16_hi v102, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v106, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v104, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v108, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v103, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v107, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v105, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v109, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_109 - label_1
s_setpc_b64 [s80, s81]
label_109:
s_setprio 3
v_add_f32 v110, v110, s53
v_add_f32 v111, v111, s54
v_add_f32 v114, v114, s53
v_add_f32 v115, v115, s54
v_add_f32 v112, v112, s53
v_add_f32 v113, v113, s54
v_add_f32 v116, v116, s53
v_add_f32 v117, v117, s54
v_max_f32 v110, 0, v110
v_max_f32 v111, 0, v111
v_max_f32 v114, 0, v114
v_max_f32 v115, 0, v115
v_max_f32 v112, 0, v112
v_max_f32 v113, 0, v113
v_max_f32 v116, 0, v116
v_max_f32 v117, 0, v117
s_setprio 0
buffer_store_short_d16_hi v110, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v114, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v112, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v116, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v111, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v115, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v113, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v117, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_106 - label_1
s_bitcmp1_b32 s67, 0
s_cbranch_scc1 label_0
s_sub_u32 s76, s77, 1
s_cselect_b32 s70, 0, 2147483648
s_mov_b64 [s68, s69], [s34, s35]
s_setpc_b64 [s80, s81]
label_110:
s_setprio 3
v_readfirstlane_b32 s67, v128
v_add_f32 v86, v86, s37
v_add_f32 v87, v87, s48
v_add_f32 v90, v90, s37
v_add_f32 v91, v91, s48
v_add_f32 v88, v88, s37
v_add_f32 v89, v89, s48
v_add_f32 v92, v92, s37
v_add_f32 v93, v93, s48
v_cvt_f64_f32 [v134, v135], v86
v_cvt_f64_f32 [v140, v141], v87
v_cvt_f64_f32 [v146, v147], v90
v_cvt_f64_f32 [v152, v153], v91
v_mov_b32 v136, 1700790272
v_mov_b32 v137, 3220641095
v_mov_b32 v142, 1700790272
v_mov_b32 v143, 3220641095
v_mov_b32 v148, 1700790272
v_mov_b32 v149, 3220641095
v_mov_b32 v154, 1700790272
v_mov_b32 v155, 3220641095
v_mul_f64 [v134, v135], [v136, v137], [v134, v135]
v_mul_f64 [v140, v141], [v142, v143], [v140, v141]
v_mul_f64 [v146, v147], [v148, v149], [v146, v147]
v_mul_f64 [v152, v153], [v154, v155], [v152, v153]
v_floor_f64 [v138, v139], [v134, v135]
v_floor_f64 [v144, v145], [v140, v141]
v_floor_f64 [v150, v151], [v146, v147]
v_floor_f64 [v156, v157], [v152, v153]
v_max_f64 [v138, v139], [v138, v139], 0.0
v_max_f64 [v144, v145], [v144, v145], 0.0
v_max_f64 [v150, v151], [v150, v151], 0.0
v_max_f64 [v156, v157], [v156, v157], 0.0
v_mov_b32 v136, 0
v_mov_b32 v137, 1083179008
v_mov_b32 v142, 0
v_mov_b32 v143, 1083179008
v_mov_b32 v148, 0
v_mov_b32 v149, 1083179008
v_mov_b32 v154, 0
v_mov_b32 v155, 1083179008
v_min_f64 [v138, v139], [v138, v139], [v136, v137]
v_min_f64 [v144, v145], [v144, v145], [v142, v143]
v_min_f64 [v150, v151], [v150, v151], [v148, v149]
v_min_f64 [v156, v157], [v156, v157], [v154, v155]
v_add_f64 [v136, v137], [v134, v135], neg([v138, v139])
v_add_f64 [v142, v143], [v140, v141], neg([v144, v145])
v_add_f64 [v148, v149], [v146, v147], neg([v150, v151])
v_add_f64 [v154, v155], [v152, v153], neg([v156, v157])
v_cvt_f32_f64 v136, [v136, v137]
v_cvt_f32_f64 v142, [v142, v143]
v_cvt_f32_f64 v148, [v148, v149]
v_cvt_f32_f64 v154, [v154, v155]
v_cvt_i32_f64 v138, neg([v138, v139])
v_cvt_i32_f64 v144, neg([v144, v145])
v_cvt_i32_f64 v150, neg([v150, v151])
v_cvt_i32_f64 v156, neg([v156, v157])
v_exp_f32 v136, v136
v_exp_f32 v142, v142
v_exp_f32 v148, v148
v_exp_f32 v154, v154
v_ldexp_f32 v138, 1.0, v138
v_ldexp_f32 v144, 1.0, v144
v_ldexp_f32 v150, 1.0, v150
v_ldexp_f32 v156, 1.0, v156
v_add_f32 v134, v136, v138
v_add_f32 v140, v142, v144
v_add_f32 v146, v148, v150
v_add_f32 v152, v154, v156
v_rcp_f32 v134, v134
v_rcp_f32 v140, v140
v_rcp_f32 v146, v146
v_rcp_f32 v152, v152
v_mul_f32 v86, v138, v134
v_mul_f32 v87, v144, v140
v_mul_f32 v90, v150, v146
v_mul_f32 v91, v156, v152
v_cvt_f64_f32 [v134, v135], v88
v_cvt_f64_f32 [v140, v141], v89
v_cvt_f64_f32 [v146, v147], v92
v_cvt_f64_f32 [v152, v153], v93
v_mov_b32 v136, 1700790272
v_mov_b32 v137, 3220641095
v_mov_b32 v142, 1700790272
v_mov_b32 v143, 3220641095
v_mov_b32 v148, 1700790272
v_mov_b32 v149, 3220641095
v_mov_b32 v154, 1700790272
v_mov_b32 v155, 3220641095
v_mul_f64 [v134, v135], [v136, v137], [v134, v135]
v_mul_f64 [v140, v141], [v142, v143], [v140, v141]
v_mul_f64 [v146, v147], [v148, v149], [v146, v147]
v_mul_f64 [v152, v153], [v154, v155], [v152, v153]
v_floor_f64 [v138, v139], [v134, v135]
v_floor_f64 [v144, v145], [v140, v141]
v_floor_f64 [v150, v151], [v146, v147]
v_floor_f64 [v156, v157], [v152, v153]
v_max_f64 [v138, v139], [v138, v139], 0.0
v_max_f64 [v144, v145], [v144, v145], 0.0
v_max_f64 [v150, v151], [v150, v151], 0.0
v_max_f64 [v156, v157], [v156, v157], 0.0
v_mov_b32 v136, 0
v_mov_b32 v137, 1083179008
v_mov_b32 v142, 0
v_mov_b32 v143, 1083179008
v_mov_b32 v148, 0
v_mov_b32 v149, 1083179008
v_mov_b32 v154, 0
v_mov_b32 v155, 1083179008
v_min_f64 [v138, v139], [v138, v139], [v136, v137]
v_min_f64 [v144, v145], [v144, v145], [v142, v143]
v_min_f64 [v150, v151], [v150, v151], [v148, v149]
v_min_f64 [v156, v157], [v156, v157], [v154, v155]
v_add_f64 [v136, v137], [v134, v135], neg([v138, v139])
v_add_f64 [v142, v143], [v140, v141], neg([v144, v145])
v_add_f64 [v148, v149], [v146, v147], neg([v150, v151])
v_add_f64 [v154, v155], [v152, v153], neg([v156, v157])
v_cvt_f32_f64 v136, [v136, v137]
v_cvt_f32_f64 v142, [v142, v143]
v_cvt_f32_f64 v148, [v148, v149]
v_cvt_f32_f64 v154, [v154, v155]
v_cvt_i32_f64 v138, neg([v138, v139])
v_cvt_i32_f64 v144, neg([v144, v145])
v_cvt_i32_f64 v150, neg([v150, v151])
v_cvt_i32_f64 v156, neg([v156, v157])
v_exp_f32 v136, v136
v_exp_f32 v142, v142
v_exp_f32 v148, v148
v_exp_f32 v154, v154
v_ldexp_f32 v138, 1.0, v138
v_ldexp_f32 v144, 1.0, v144
v_ldexp_f32 v150, 1.0, v150
v_ldexp_f32 v156, 1.0, v156
v_add_f32 v134, v136, v138
v_add_f32 v140, v142, v144
v_add_f32 v146, v148, v150
v_add_f32 v152, v154, v156
v_rcp_f32 v134, v134
v_rcp_f32 v140, v140
v_rcp_f32 v146, v146
v_rcp_f32 v152, v152
v_mul_f32 v88, v138, v134
v_mul_f32 v89, v144, v140
v_mul_f32 v92, v150, v146
v_mul_f32 v93, v156, v152
s_setprio 0
buffer_store_short_d16_hi v86, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v90, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v88, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v92, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v87, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v91, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v89, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v93, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_111 - label_1
s_setpc_b64 [s80, s81]
label_111:
s_setprio 3
v_add_f32 v94, v94, s49
v_add_f32 v95, v95, s50
v_add_f32 v98, v98, s49
v_add_f32 v99, v99, s50
v_add_f32 v96, v96, s49
v_add_f32 v97, v97, s50
v_add_f32 v100, v100, s49
v_add_f32 v101, v101, s50
v_cvt_f64_f32 [v134, v135], v94
v_cvt_f64_f32 [v140, v141], v95
v_cvt_f64_f32 [v146, v147], v98
v_cvt_f64_f32 [v152, v153], v99
v_mov_b32 v136, 1700790272
v_mov_b32 v137, 3220641095
v_mov_b32 v142, 1700790272
v_mov_b32 v143, 3220641095
v_mov_b32 v148, 1700790272
v_mov_b32 v149, 3220641095
v_mov_b32 v154, 1700790272
v_mov_b32 v155, 3220641095
v_mul_f64 [v134, v135], [v136, v137], [v134, v135]
v_mul_f64 [v140, v141], [v142, v143], [v140, v141]
v_mul_f64 [v146, v147], [v148, v149], [v146, v147]
v_mul_f64 [v152, v153], [v154, v155], [v152, v153]
v_floor_f64 [v138, v139], [v134, v135]
v_floor_f64 [v144, v145], [v140, v141]
v_floor_f64 [v150, v151], [v146, v147]
v_floor_f64 [v156, v157], [v152, v153]
v_max_f64 [v138, v139], [v138, v139], 0.0
v_max_f64 [v144, v145], [v144, v145], 0.0
v_max_f64 [v150, v151], [v150, v151], 0.0
v_max_f64 [v156, v157], [v156, v157], 0.0
v_mov_b32 v136, 0
v_mov_b32 v137, 1083179008
v_mov_b32 v142, 0
v_mov_b32 v143, 1083179008
v_mov_b32 v148, 0
v_mov_b32 v149, 1083179008
v_mov_b32 v154, 0
v_mov_b32 v155, 1083179008
v_min_f64 [v138, v139], [v138, v139], [v136, v137]
v_min_f64 [v144, v145], [v144, v145], [v142, v143]
v_min_f64 [v150, v151], [v150, v151], [v148, v149]
v_min_f64 [v156, v157], [v156, v157], [v154, v155]
v_add_f64 [v136, v137], [v134, v135], neg([v138, v139])
v_add_f64 [v142, v143], [v140, v141], neg([v144, v145])
v_add_f64 [v148, v149], [v146, v147], neg([v150, v151])
v_add_f64 [v154, v155], [v152, v153], neg([v156, v157])
v_cvt_f32_f64 v136, [v136, v137]
v_cvt_f32_f64 v142, [v142, v143]
v_cvt_f32_f64 v148, [v148, v149]
v_cvt_f32_f64 v154, [v154, v155]
v_cvt_i32_f64 v138, neg([v138, v139])
v_cvt_i32_f64 v144, neg([v144, v145])
v_cvt_i32_f64 v150, neg([v150, v151])
v_cvt_i32_f64 v156, neg([v156, v157])
v_exp_f32 v136, v136
v_exp_f32 v142, v142
v_exp_f32 v148, v148
v_exp_f32 v154, v154
v_ldexp_f32 v138, 1.0, v138
v_ldexp_f32 v144, 1.0, v144
v_ldexp_f32 v150, 1.0, v150
v_ldexp_f32 v156, 1.0, v156
v_add_f32 v134, v136, v138
v_add_f32 v140, v142, v144
v_add_f32 v146, v148, v150
v_add_f32 v152, v154, v156
v_rcp_f32 v134, v134
v_rcp_f32 v140, v140
v_rcp_f32 v146, v146
v_rcp_f32 v152, v152
v_mul_f32 v94, v138, v134
v_mul_f32 v95, v144, v140
v_mul_f32 v98, v150, v146
v_mul_f32 v99, v156, v152
v_cvt_f64_f32 [v134, v135], v96
v_cvt_f64_f32 [v140, v141], v97
v_cvt_f64_f32 [v146, v147], v100
v_cvt_f64_f32 [v152, v153], v101
v_mov_b32 v136, 1700790272
v_mov_b32 v137, 3220641095
v_mov_b32 v142, 1700790272
v_mov_b32 v143, 3220641095
v_mov_b32 v148, 1700790272
v_mov_b32 v149, 3220641095
v_mov_b32 v154, 1700790272
v_mov_b32 v155, 3220641095
v_mul_f64 [v134, v135], [v136, v137], [v134, v135]
v_mul_f64 [v140, v141], [v142, v143], [v140, v141]
v_mul_f64 [v146, v147], [v148, v149], [v146, v147]
v_mul_f64 [v152, v153], [v154, v155], [v152, v153]
v_floor_f64 [v138, v139], [v134, v135]
v_floor_f64 [v144, v145], [v140, v141]
v_floor_f64 [v150, v151], [v146, v147]
v_floor_f64 [v156, v157], [v152, v153]
v_max_f64 [v138, v139], [v138, v139], 0.0
v_max_f64 [v144, v145], [v144, v145], 0.0
v_max_f64 [v150, v151], [v150, v151], 0.0
v_max_f64 [v156, v157], [v156, v157], 0.0
v_mov_b32 v136, 0
v_mov_b32 v137, 1083179008
v_mov_b32 v142, 0
v_mov_b32 v143, 1083179008
v_mov_b32 v148, 0
v_mov_b32 v149, 1083179008
v_mov_b32 v154, 0
v_mov_b32 v155, 1083179008
v_min_f64 [v138, v139], [v138, v139], [v136, v137]
v_min_f64 [v144, v145], [v144, v145], [v142, v143]
v_min_f64 [v150, v151], [v150, v151], [v148, v149]
v_min_f64 [v156, v157], [v156, v157], [v154, v155]
v_add_f64 [v136, v137], [v134, v135], neg([v138, v139])
v_add_f64 [v142, v143], [v140, v141], neg([v144, v145])
v_add_f64 [v148, v149], [v146, v147], neg([v150, v151])
v_add_f64 [v154, v155], [v152, v153], neg([v156, v157])
v_cvt_f32_f64 v136, [v136, v137]
v_cvt_f32_f64 v142, [v142, v143]
v_cvt_f32_f64 v148, [v148, v149]
v_cvt_f32_f64 v154, [v154, v155]
v_cvt_i32_f64 v138, neg([v138, v139])
v_cvt_i32_f64 v144, neg([v144, v145])
v_cvt_i32_f64 v150, neg([v150, v151])
v_cvt_i32_f64 v156, neg([v156, v157])
v_exp_f32 v136, v136
v_exp_f32 v142, v142
v_exp_f32 v148, v148
v_exp_f32 v154, v154
v_ldexp_f32 v138, 1.0, v138
v_ldexp_f32 v144, 1.0, v144
v_ldexp_f32 v150, 1.0, v150
v_ldexp_f32 v156, 1.0, v156
v_add_f32 v134, v136, v138
v_add_f32 v140, v142, v144
v_add_f32 v146, v148, v150
v_add_f32 v152, v154, v156
v_rcp_f32 v134, v134
v_rcp_f32 v140, v140
v_rcp_f32 v146, v146
v_rcp_f32 v152, v152
v_mul_f32 v96, v138, v134
v_mul_f32 v97, v144, v140
v_mul_f32 v100, v150, v146
v_mul_f32 v101, v156, v152
s_setprio 0
buffer_store_short_d16_hi v94, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v98, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v96, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v100, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v95, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v99, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v97, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v101, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_112 - label_1
s_sub_u32 s76, s77, 17
s_cselect_b32 s70, 0, s70
s_add_u32 s68, s34, s74
s_addc_u32 s69, s35, s75
s_setpc_b64 [s80, s81]
label_112:
s_setprio 3
v_add_f32 v102, v102, s51
v_add_f32 v103, v103, s52
v_add_f32 v106, v106, s51
v_add_f32 v107, v107, s52
v_add_f32 v104, v104, s51
v_add_f32 v105, v105, s52
v_add_f32 v108, v108, s51
v_add_f32 v109, v109, s52
v_cvt_f64_f32 [v134, v135], v102
v_cvt_f64_f32 [v140, v141], v103
v_cvt_f64_f32 [v146, v147], v106
v_cvt_f64_f32 [v152, v153], v107
v_mov_b32 v136, 1700790272
v_mov_b32 v137, 3220641095
v_mov_b32 v142, 1700790272
v_mov_b32 v143, 3220641095
v_mov_b32 v148, 1700790272
v_mov_b32 v149, 3220641095
v_mov_b32 v154, 1700790272
v_mov_b32 v155, 3220641095
v_mul_f64 [v134, v135], [v136, v137], [v134, v135]
v_mul_f64 [v140, v141], [v142, v143], [v140, v141]
v_mul_f64 [v146, v147], [v148, v149], [v146, v147]
v_mul_f64 [v152, v153], [v154, v155], [v152, v153]
v_floor_f64 [v138, v139], [v134, v135]
v_floor_f64 [v144, v145], [v140, v141]
v_floor_f64 [v150, v151], [v146, v147]
v_floor_f64 [v156, v157], [v152, v153]
v_max_f64 [v138, v139], [v138, v139], 0.0
v_max_f64 [v144, v145], [v144, v145], 0.0
v_max_f64 [v150, v151], [v150, v151], 0.0
v_max_f64 [v156, v157], [v156, v157], 0.0
v_mov_b32 v136, 0
v_mov_b32 v137, 1083179008
v_mov_b32 v142, 0
v_mov_b32 v143, 1083179008
v_mov_b32 v148, 0
v_mov_b32 v149, 1083179008
v_mov_b32 v154, 0
v_mov_b32 v155, 1083179008
v_min_f64 [v138, v139], [v138, v139], [v136, v137]
v_min_f64 [v144, v145], [v144, v145], [v142, v143]
v_min_f64 [v150, v151], [v150, v151], [v148, v149]
v_min_f64 [v156, v157], [v156, v157], [v154, v155]
v_add_f64 [v136, v137], [v134, v135], neg([v138, v139])
v_add_f64 [v142, v143], [v140, v141], neg([v144, v145])
v_add_f64 [v148, v149], [v146, v147], neg([v150, v151])
v_add_f64 [v154, v155], [v152, v153], neg([v156, v157])
v_cvt_f32_f64 v136, [v136, v137]
v_cvt_f32_f64 v142, [v142, v143]
v_cvt_f32_f64 v148, [v148, v149]
v_cvt_f32_f64 v154, [v154, v155]
v_cvt_i32_f64 v138, neg([v138, v139])
v_cvt_i32_f64 v144, neg([v144, v145])
v_cvt_i32_f64 v150, neg([v150, v151])
v_cvt_i32_f64 v156, neg([v156, v157])
v_exp_f32 v136, v136
v_exp_f32 v142, v142
v_exp_f32 v148, v148
v_exp_f32 v154, v154
v_ldexp_f32 v138, 1.0, v138
v_ldexp_f32 v144, 1.0, v144
v_ldexp_f32 v150, 1.0, v150
v_ldexp_f32 v156, 1.0, v156
v_add_f32 v134, v136, v138
v_add_f32 v140, v142, v144
v_add_f32 v146, v148, v150
v_add_f32 v152, v154, v156
v_rcp_f32 v134, v134
v_rcp_f32 v140, v140
v_rcp_f32 v146, v146
v_rcp_f32 v152, v152
v_mul_f32 v102, v138, v134
v_mul_f32 v103, v144, v140
v_mul_f32 v106, v150, v146
v_mul_f32 v107, v156, v152
v_cvt_f64_f32 [v134, v135], v104
v_cvt_f64_f32 [v140, v141], v105
v_cvt_f64_f32 [v146, v147], v108
v_cvt_f64_f32 [v152, v153], v109
v_mov_b32 v136, 1700790272
v_mov_b32 v137, 3220641095
v_mov_b32 v142, 1700790272
v_mov_b32 v143, 3220641095
v_mov_b32 v148, 1700790272
v_mov_b32 v149, 3220641095
v_mov_b32 v154, 1700790272
v_mov_b32 v155, 3220641095
v_mul_f64 [v134, v135], [v136, v137], [v134, v135]
v_mul_f64 [v140, v141], [v142, v143], [v140, v141]
v_mul_f64 [v146, v147], [v148, v149], [v146, v147]
v_mul_f64 [v152, v153], [v154, v155], [v152, v153]
v_floor_f64 [v138, v139], [v134, v135]
v_floor_f64 [v144, v145], [v140, v141]
v_floor_f64 [v150, v151], [v146, v147]
v_floor_f64 [v156, v157], [v152, v153]
v_max_f64 [v138, v139], [v138, v139], 0.0
v_max_f64 [v144, v145], [v144, v145], 0.0
v_max_f64 [v150, v151], [v150, v151], 0.0
v_max_f64 [v156, v157], [v156, v157], 0.0
v_mov_b32 v136, 0
v_mov_b32 v137, 1083179008
v_mov_b32 v142, 0
v_mov_b32 v143, 1083179008
v_mov_b32 v148, 0
v_mov_b32 v149, 1083179008
v_mov_b32 v154, 0
v_mov_b32 v155, 1083179008
v_min_f64 [v138, v139], [v138, v139], [v136, v137]
v_min_f64 [v144, v145], [v144, v145], [v142, v143]
v_min_f64 [v150, v151], [v150, v151], [v148, v149]
v_min_f64 [v156, v157], [v156, v157], [v154, v155]
v_add_f64 [v136, v137], [v134, v135], neg([v138, v139])
v_add_f64 [v142, v143], [v140, v141], neg([v144, v145])
v_add_f64 [v148, v149], [v146, v147], neg([v150, v151])
v_add_f64 [v154, v155], [v152, v153], neg([v156, v157])
v_cvt_f32_f64 v136, [v136, v137]
v_cvt_f32_f64 v142, [v142, v143]
v_cvt_f32_f64 v148, [v148, v149]
v_cvt_f32_f64 v154, [v154, v155]
v_cvt_i32_f64 v138, neg([v138, v139])
v_cvt_i32_f64 v144, neg([v144, v145])
v_cvt_i32_f64 v150, neg([v150, v151])
v_cvt_i32_f64 v156, neg([v156, v157])
v_exp_f32 v136, v136
v_exp_f32 v142, v142
v_exp_f32 v148, v148
v_exp_f32 v154, v154
v_ldexp_f32 v138, 1.0, v138
v_ldexp_f32 v144, 1.0, v144
v_ldexp_f32 v150, 1.0, v150
v_ldexp_f32 v156, 1.0, v156
v_add_f32 v134, v136, v138
v_add_f32 v140, v142, v144
v_add_f32 v146, v148, v150
v_add_f32 v152, v154, v156
v_rcp_f32 v134, v134
v_rcp_f32 v140, v140
v_rcp_f32 v146, v146
v_rcp_f32 v152, v152
v_mul_f32 v104, v138, v134
v_mul_f32 v105, v144, v140
v_mul_f32 v108, v150, v146
v_mul_f32 v109, v156, v152
s_setprio 0
buffer_store_short_d16_hi v102, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v106, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v104, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v108, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v103, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v107, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v105, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v109, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_113 - label_1
s_setpc_b64 [s80, s81]
label_113:
s_setprio 3
v_add_f32 v110, v110, s53
v_add_f32 v111, v111, s54
v_add_f32 v114, v114, s53
v_add_f32 v115, v115, s54
v_add_f32 v112, v112, s53
v_add_f32 v113, v113, s54
v_add_f32 v116, v116, s53
v_add_f32 v117, v117, s54
v_cvt_f64_f32 [v134, v135], v110
v_cvt_f64_f32 [v140, v141], v111
v_cvt_f64_f32 [v146, v147], v114
v_cvt_f64_f32 [v152, v153], v115
v_mov_b32 v136, 1700790272
v_mov_b32 v137, 3220641095
v_mov_b32 v142, 1700790272
v_mov_b32 v143, 3220641095
v_mov_b32 v148, 1700790272
v_mov_b32 v149, 3220641095
v_mov_b32 v154, 1700790272
v_mov_b32 v155, 3220641095
v_mul_f64 [v134, v135], [v136, v137], [v134, v135]
v_mul_f64 [v140, v141], [v142, v143], [v140, v141]
v_mul_f64 [v146, v147], [v148, v149], [v146, v147]
v_mul_f64 [v152, v153], [v154, v155], [v152, v153]
v_floor_f64 [v138, v139], [v134, v135]
v_floor_f64 [v144, v145], [v140, v141]
v_floor_f64 [v150, v151], [v146, v147]
v_floor_f64 [v156, v157], [v152, v153]
v_max_f64 [v138, v139], [v138, v139], 0.0
v_max_f64 [v144, v145], [v144, v145], 0.0
v_max_f64 [v150, v151], [v150, v151], 0.0
v_max_f64 [v156, v157], [v156, v157], 0.0
v_mov_b32 v136, 0
v_mov_b32 v137, 1083179008
v_mov_b32 v142, 0
v_mov_b32 v143, 1083179008
v_mov_b32 v148, 0
v_mov_b32 v149, 1083179008
v_mov_b32 v154, 0
v_mov_b32 v155, 1083179008
v_min_f64 [v138, v139], [v138, v139], [v136, v137]
v_min_f64 [v144, v145], [v144, v145], [v142, v143]
v_min_f64 [v150, v151], [v150, v151], [v148, v149]
v_min_f64 [v156, v157], [v156, v157], [v154, v155]
v_add_f64 [v136, v137], [v134, v135], neg([v138, v139])
v_add_f64 [v142, v143], [v140, v141], neg([v144, v145])
v_add_f64 [v148, v149], [v146, v147], neg([v150, v151])
v_add_f64 [v154, v155], [v152, v153], neg([v156, v157])
v_cvt_f32_f64 v136, [v136, v137]
v_cvt_f32_f64 v142, [v142, v143]
v_cvt_f32_f64 v148, [v148, v149]
v_cvt_f32_f64 v154, [v154, v155]
v_cvt_i32_f64 v138, neg([v138, v139])
v_cvt_i32_f64 v144, neg([v144, v145])
v_cvt_i32_f64 v150, neg([v150, v151])
v_cvt_i32_f64 v156, neg([v156, v157])
v_exp_f32 v136, v136
v_exp_f32 v142, v142
v_exp_f32 v148, v148
v_exp_f32 v154, v154
v_ldexp_f32 v138, 1.0, v138
v_ldexp_f32 v144, 1.0, v144
v_ldexp_f32 v150, 1.0, v150
v_ldexp_f32 v156, 1.0, v156
v_add_f32 v134, v136, v138
v_add_f32 v140, v142, v144
v_add_f32 v146, v148, v150
v_add_f32 v152, v154, v156
v_rcp_f32 v134, v134
v_rcp_f32 v140, v140
v_rcp_f32 v146, v146
v_rcp_f32 v152, v152
v_mul_f32 v110, v138, v134
v_mul_f32 v111, v144, v140
v_mul_f32 v114, v150, v146
v_mul_f32 v115, v156, v152
v_cvt_f64_f32 [v134, v135], v112
v_cvt_f64_f32 [v140, v141], v113
v_cvt_f64_f32 [v146, v147], v116
v_cvt_f64_f32 [v152, v153], v117
v_mov_b32 v136, 1700790272
v_mov_b32 v137, 3220641095
v_mov_b32 v142, 1700790272
v_mov_b32 v143, 3220641095
v_mov_b32 v148, 1700790272
v_mov_b32 v149, 3220641095
v_mov_b32 v154, 1700790272
v_mov_b32 v155, 3220641095
v_mul_f64 [v134, v135], [v136, v137], [v134, v135]
v_mul_f64 [v140, v141], [v142, v143], [v140, v141]
v_mul_f64 [v146, v147], [v148, v149], [v146, v147]
v_mul_f64 [v152, v153], [v154, v155], [v152, v153]
v_floor_f64 [v138, v139], [v134, v135]
v_floor_f64 [v144, v145], [v140, v141]
v_floor_f64 [v150, v151], [v146, v147]
v_floor_f64 [v156, v157], [v152, v153]
v_max_f64 [v138, v139], [v138, v139], 0.0
v_max_f64 [v144, v145], [v144, v145], 0.0
v_max_f64 [v150, v151], [v150, v151], 0.0
v_max_f64 [v156, v157], [v156, v157], 0.0
v_mov_b32 v136, 0
v_mov_b32 v137, 1083179008
v_mov_b32 v142, 0
v_mov_b32 v143, 1083179008
v_mov_b32 v148, 0
v_mov_b32 v149, 1083179008
v_mov_b32 v154, 0
v_mov_b32 v155, 1083179008
v_min_f64 [v138, v139], [v138, v139], [v136, v137]
v_min_f64 [v144, v145], [v144, v145], [v142, v143]
v_min_f64 [v150, v151], [v150, v151], [v148, v149]
v_min_f64 [v156, v157], [v156, v157], [v154, v155]
v_add_f64 [v136, v137], [v134, v135], neg([v138, v139])
v_add_f64 [v142, v143], [v140, v141], neg([v144, v145])
v_add_f64 [v148, v149], [v146, v147], neg([v150, v151])
v_add_f64 [v154, v155], [v152, v153], neg([v156, v157])
v_cvt_f32_f64 v136, [v136, v137]
v_cvt_f32_f64 v142, [v142, v143]
v_cvt_f32_f64 v148, [v148, v149]
v_cvt_f32_f64 v154, [v154, v155]
v_cvt_i32_f64 v138, neg([v138, v139])
v_cvt_i32_f64 v144, neg([v144, v145])
v_cvt_i32_f64 v150, neg([v150, v151])
v_cvt_i32_f64 v156, neg([v156, v157])
v_exp_f32 v136, v136
v_exp_f32 v142, v142
v_exp_f32 v148, v148
v_exp_f32 v154, v154
v_ldexp_f32 v138, 1.0, v138
v_ldexp_f32 v144, 1.0, v144
v_ldexp_f32 v150, 1.0, v150
v_ldexp_f32 v156, 1.0, v156
v_add_f32 v134, v136, v138
v_add_f32 v140, v142, v144
v_add_f32 v146, v148, v150
v_add_f32 v152, v154, v156
v_rcp_f32 v134, v134
v_rcp_f32 v140, v140
v_rcp_f32 v146, v146
v_rcp_f32 v152, v152
v_mul_f32 v112, v138, v134
v_mul_f32 v113, v144, v140
v_mul_f32 v116, v150, v146
v_mul_f32 v117, v156, v152
s_setprio 0
buffer_store_short_d16_hi v110, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v114, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v112, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v116, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v111, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v115, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v113, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v117, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_110 - label_1
s_bitcmp1_b32 s67, 0
s_cbranch_scc1 label_0
s_sub_u32 s76, s77, 1
s_cselect_b32 s70, 0, 2147483648
s_mov_b64 [s68, s69], [s34, s35]
s_setpc_b64 [s80, s81]
label_114:
s_setprio 3
v_readfirstlane_b32 s67, v128
v_add_f32 v86, v86, s37
v_add_f32 v87, v87, s48
v_add_f32 v90, v90, s37
v_add_f32 v91, v91, s48
v_add_f32 v88, v88, s37
v_add_f32 v89, v89, s48
v_add_f32 v92, v92, s37
v_add_f32 v93, v93, s48
v_mul_f32 v86, v86, s47
v_mul_f32 v87, v87, s47
v_mul_f32 v90, v90, s47
v_mul_f32 v91, v91, s47
v_mul_f32 v133, 1077455419, v86
v_mul_f32 v135, 1077455419, v87
v_mul_f32 v137, 1077455419, v90
v_mul_f32 v139, 1077455419, v91
v_exp_f32 v133, abs(v133)
v_exp_f32 v135, abs(v135)
v_exp_f32 v137, abs(v137)
v_exp_f32 v139, abs(v139)
v_add_f32 v133, 1.0, v133
v_add_f32 v135, 1.0, v135
v_add_f32 v137, 1.0, v137
v_add_f32 v139, 1.0, v139
v_rcp_f32 v133, v133
v_rcp_f32 v135, v135
v_rcp_f32 v137, v137
v_rcp_f32 v139, v139
v_fma_f32 v133, v133, -2.0, 1.0
v_fma_f32 v135, v135, -2.0, 1.0
v_fma_f32 v137, v137, -2.0, 1.0
v_fma_f32 v139, v139, -2.0, 1.0
v_mov_b32 v134, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v86), v134
v_cndmask_b32 v133, v86, v133, [vcc_lo, vcc_hi]
v_mov_b32 v136, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v87), v136
v_cndmask_b32 v135, v87, v135, [vcc_lo, vcc_hi]
v_mov_b32 v138, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v90), v138
v_cndmask_b32 v137, v90, v137, [vcc_lo, vcc_hi]
v_mov_b32 v140, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v91), v140
v_cndmask_b32 v139, v91, v139, [vcc_lo, vcc_hi]
v_mov_b32 v134, 2147483647
v_bfi_b32 v86, v134, v133, v86
v_mov_b32 v136, 2147483647
v_bfi_b32 v87, v136, v135, v87
v_mov_b32 v138, 2147483647
v_bfi_b32 v90, v138, v137, v90
v_mov_b32 v140, 2147483647
v_bfi_b32 v91, v140, v139, v91
v_mul_f32 v86, v86, s46
v_mul_f32 v87, v87, s46
v_mul_f32 v90, v90, s46
v_mul_f32 v91, v91, s46
v_mul_f32 v88, v88, s47
v_mul_f32 v89, v89, s47
v_mul_f32 v92, v92, s47
v_mul_f32 v93, v93, s47
v_mul_f32 v133, 1077455419, v88
v_mul_f32 v135, 1077455419, v89
v_mul_f32 v137, 1077455419, v92
v_mul_f32 v139, 1077455419, v93
v_exp_f32 v133, abs(v133)
v_exp_f32 v135, abs(v135)
v_exp_f32 v137, abs(v137)
v_exp_f32 v139, abs(v139)
v_add_f32 v133, 1.0, v133
v_add_f32 v135, 1.0, v135
v_add_f32 v137, 1.0, v137
v_add_f32 v139, 1.0, v139
v_rcp_f32 v133, v133
v_rcp_f32 v135, v135
v_rcp_f32 v137, v137
v_rcp_f32 v139, v139
v_fma_f32 v133, v133, -2.0, 1.0
v_fma_f32 v135, v135, -2.0, 1.0
v_fma_f32 v137, v137, -2.0, 1.0
v_fma_f32 v139, v139, -2.0, 1.0
v_mov_b32 v134, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v88), v134
v_cndmask_b32 v133, v88, v133, [vcc_lo, vcc_hi]
v_mov_b32 v136, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v89), v136
v_cndmask_b32 v135, v89, v135, [vcc_lo, vcc_hi]
v_mov_b32 v138, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v92), v138
v_cndmask_b32 v137, v92, v137, [vcc_lo, vcc_hi]
v_mov_b32 v140, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v93), v140
v_cndmask_b32 v139, v93, v139, [vcc_lo, vcc_hi]
v_mov_b32 v134, 2147483647
v_bfi_b32 v88, v134, v133, v88
v_mov_b32 v136, 2147483647
v_bfi_b32 v89, v136, v135, v89
v_mov_b32 v138, 2147483647
v_bfi_b32 v92, v138, v137, v92
v_mov_b32 v140, 2147483647
v_bfi_b32 v93, v140, v139, v93
v_mul_f32 v88, v88, s46
v_mul_f32 v89, v89, s46
v_mul_f32 v92, v92, s46
v_mul_f32 v93, v93, s46
s_setprio 0
buffer_store_short_d16_hi v86, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v90, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v88, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v92, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v87, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v91, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v89, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v93, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_115 - label_1
s_setpc_b64 [s80, s81]
label_115:
s_setprio 3
v_add_f32 v94, v94, s49
v_add_f32 v95, v95, s50
v_add_f32 v98, v98, s49
v_add_f32 v99, v99, s50
v_add_f32 v96, v96, s49
v_add_f32 v97, v97, s50
v_add_f32 v100, v100, s49
v_add_f32 v101, v101, s50
v_mul_f32 v94, v94, s47
v_mul_f32 v95, v95, s47
v_mul_f32 v98, v98, s47
v_mul_f32 v99, v99, s47
v_mul_f32 v133, 1077455419, v94
v_mul_f32 v135, 1077455419, v95
v_mul_f32 v137, 1077455419, v98
v_mul_f32 v139, 1077455419, v99
v_exp_f32 v133, abs(v133)
v_exp_f32 v135, abs(v135)
v_exp_f32 v137, abs(v137)
v_exp_f32 v139, abs(v139)
v_add_f32 v133, 1.0, v133
v_add_f32 v135, 1.0, v135
v_add_f32 v137, 1.0, v137
v_add_f32 v139, 1.0, v139
v_rcp_f32 v133, v133
v_rcp_f32 v135, v135
v_rcp_f32 v137, v137
v_rcp_f32 v139, v139
v_fma_f32 v133, v133, -2.0, 1.0
v_fma_f32 v135, v135, -2.0, 1.0
v_fma_f32 v137, v137, -2.0, 1.0
v_fma_f32 v139, v139, -2.0, 1.0
v_mov_b32 v134, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v94), v134
v_cndmask_b32 v133, v94, v133, [vcc_lo, vcc_hi]
v_mov_b32 v136, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v95), v136
v_cndmask_b32 v135, v95, v135, [vcc_lo, vcc_hi]
v_mov_b32 v138, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v98), v138
v_cndmask_b32 v137, v98, v137, [vcc_lo, vcc_hi]
v_mov_b32 v140, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v99), v140
v_cndmask_b32 v139, v99, v139, [vcc_lo, vcc_hi]
v_mov_b32 v134, 2147483647
v_bfi_b32 v94, v134, v133, v94
v_mov_b32 v136, 2147483647
v_bfi_b32 v95, v136, v135, v95
v_mov_b32 v138, 2147483647
v_bfi_b32 v98, v138, v137, v98
v_mov_b32 v140, 2147483647
v_bfi_b32 v99, v140, v139, v99
v_mul_f32 v94, v94, s46
v_mul_f32 v95, v95, s46
v_mul_f32 v98, v98, s46
v_mul_f32 v99, v99, s46
v_mul_f32 v96, v96, s47
v_mul_f32 v97, v97, s47
v_mul_f32 v100, v100, s47
v_mul_f32 v101, v101, s47
v_mul_f32 v133, 1077455419, v96
v_mul_f32 v135, 1077455419, v97
v_mul_f32 v137, 1077455419, v100
v_mul_f32 v139, 1077455419, v101
v_exp_f32 v133, abs(v133)
v_exp_f32 v135, abs(v135)
v_exp_f32 v137, abs(v137)
v_exp_f32 v139, abs(v139)
v_add_f32 v133, 1.0, v133
v_add_f32 v135, 1.0, v135
v_add_f32 v137, 1.0, v137
v_add_f32 v139, 1.0, v139
v_rcp_f32 v133, v133
v_rcp_f32 v135, v135
v_rcp_f32 v137, v137
v_rcp_f32 v139, v139
v_fma_f32 v133, v133, -2.0, 1.0
v_fma_f32 v135, v135, -2.0, 1.0
v_fma_f32 v137, v137, -2.0, 1.0
v_fma_f32 v139, v139, -2.0, 1.0
v_mov_b32 v134, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v96), v134
v_cndmask_b32 v133, v96, v133, [vcc_lo, vcc_hi]
v_mov_b32 v136, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v97), v136
v_cndmask_b32 v135, v97, v135, [vcc_lo, vcc_hi]
v_mov_b32 v138, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v100), v138
v_cndmask_b32 v137, v100, v137, [vcc_lo, vcc_hi]
v_mov_b32 v140, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v101), v140
v_cndmask_b32 v139, v101, v139, [vcc_lo, vcc_hi]
v_mov_b32 v134, 2147483647
v_bfi_b32 v96, v134, v133, v96
v_mov_b32 v136, 2147483647
v_bfi_b32 v97, v136, v135, v97
v_mov_b32 v138, 2147483647
v_bfi_b32 v100, v138, v137, v100
v_mov_b32 v140, 2147483647
v_bfi_b32 v101, v140, v139, v101
v_mul_f32 v96, v96, s46
v_mul_f32 v97, v97, s46
v_mul_f32 v100, v100, s46
v_mul_f32 v101, v101, s46
s_setprio 0
buffer_store_short_d16_hi v94, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v98, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v96, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v100, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v95, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v99, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v97, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v101, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_116 - label_1
s_sub_u32 s76, s77, 17
s_cselect_b32 s70, 0, s70
s_add_u32 s68, s34, s74
s_addc_u32 s69, s35, s75
s_setpc_b64 [s80, s81]
label_116:
s_setprio 3
v_add_f32 v102, v102, s51
v_add_f32 v103, v103, s52
v_add_f32 v106, v106, s51
v_add_f32 v107, v107, s52
v_add_f32 v104, v104, s51
v_add_f32 v105, v105, s52
v_add_f32 v108, v108, s51
v_add_f32 v109, v109, s52
v_mul_f32 v102, v102, s47
v_mul_f32 v103, v103, s47
v_mul_f32 v106, v106, s47
v_mul_f32 v107, v107, s47
v_mul_f32 v133, 1077455419, v102
v_mul_f32 v135, 1077455419, v103
v_mul_f32 v137, 1077455419, v106
v_mul_f32 v139, 1077455419, v107
v_exp_f32 v133, abs(v133)
v_exp_f32 v135, abs(v135)
v_exp_f32 v137, abs(v137)
v_exp_f32 v139, abs(v139)
v_add_f32 v133, 1.0, v133
v_add_f32 v135, 1.0, v135
v_add_f32 v137, 1.0, v137
v_add_f32 v139, 1.0, v139
v_rcp_f32 v133, v133
v_rcp_f32 v135, v135
v_rcp_f32 v137, v137
v_rcp_f32 v139, v139
v_fma_f32 v133, v133, -2.0, 1.0
v_fma_f32 v135, v135, -2.0, 1.0
v_fma_f32 v137, v137, -2.0, 1.0
v_fma_f32 v139, v139, -2.0, 1.0
v_mov_b32 v134, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v102), v134
v_cndmask_b32 v133, v102, v133, [vcc_lo, vcc_hi]
v_mov_b32 v136, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v103), v136
v_cndmask_b32 v135, v103, v135, [vcc_lo, vcc_hi]
v_mov_b32 v138, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v106), v138
v_cndmask_b32 v137, v106, v137, [vcc_lo, vcc_hi]
v_mov_b32 v140, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v107), v140
v_cndmask_b32 v139, v107, v139, [vcc_lo, vcc_hi]
v_mov_b32 v134, 2147483647
v_bfi_b32 v102, v134, v133, v102
v_mov_b32 v136, 2147483647
v_bfi_b32 v103, v136, v135, v103
v_mov_b32 v138, 2147483647
v_bfi_b32 v106, v138, v137, v106
v_mov_b32 v140, 2147483647
v_bfi_b32 v107, v140, v139, v107
v_mul_f32 v102, v102, s46
v_mul_f32 v103, v103, s46
v_mul_f32 v106, v106, s46
v_mul_f32 v107, v107, s46
v_mul_f32 v104, v104, s47
v_mul_f32 v105, v105, s47
v_mul_f32 v108, v108, s47
v_mul_f32 v109, v109, s47
v_mul_f32 v133, 1077455419, v104
v_mul_f32 v135, 1077455419, v105
v_mul_f32 v137, 1077455419, v108
v_mul_f32 v139, 1077455419, v109
v_exp_f32 v133, abs(v133)
v_exp_f32 v135, abs(v135)
v_exp_f32 v137, abs(v137)
v_exp_f32 v139, abs(v139)
v_add_f32 v133, 1.0, v133
v_add_f32 v135, 1.0, v135
v_add_f32 v137, 1.0, v137
v_add_f32 v139, 1.0, v139
v_rcp_f32 v133, v133
v_rcp_f32 v135, v135
v_rcp_f32 v137, v137
v_rcp_f32 v139, v139
v_fma_f32 v133, v133, -2.0, 1.0
v_fma_f32 v135, v135, -2.0, 1.0
v_fma_f32 v137, v137, -2.0, 1.0
v_fma_f32 v139, v139, -2.0, 1.0
v_mov_b32 v134, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v104), v134
v_cndmask_b32 v133, v104, v133, [vcc_lo, vcc_hi]
v_mov_b32 v136, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v105), v136
v_cndmask_b32 v135, v105, v135, [vcc_lo, vcc_hi]
v_mov_b32 v138, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v108), v138
v_cndmask_b32 v137, v108, v137, [vcc_lo, vcc_hi]
v_mov_b32 v140, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v109), v140
v_cndmask_b32 v139, v109, v139, [vcc_lo, vcc_hi]
v_mov_b32 v134, 2147483647
v_bfi_b32 v104, v134, v133, v104
v_mov_b32 v136, 2147483647
v_bfi_b32 v105, v136, v135, v105
v_mov_b32 v138, 2147483647
v_bfi_b32 v108, v138, v137, v108
v_mov_b32 v140, 2147483647
v_bfi_b32 v109, v140, v139, v109
v_mul_f32 v104, v104, s46
v_mul_f32 v105, v105, s46
v_mul_f32 v108, v108, s46
v_mul_f32 v109, v109, s46
s_setprio 0
buffer_store_short_d16_hi v102, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v106, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v104, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v108, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v103, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v107, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v105, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v109, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_117 - label_1
s_setpc_b64 [s80, s81]
label_117:
s_setprio 3
v_add_f32 v110, v110, s53
v_add_f32 v111, v111, s54
v_add_f32 v114, v114, s53
v_add_f32 v115, v115, s54
v_add_f32 v112, v112, s53
v_add_f32 v113, v113, s54
v_add_f32 v116, v116, s53
v_add_f32 v117, v117, s54
v_mul_f32 v110, v110, s47
v_mul_f32 v111, v111, s47
v_mul_f32 v114, v114, s47
v_mul_f32 v115, v115, s47
v_mul_f32 v133, 1077455419, v110
v_mul_f32 v135, 1077455419, v111
v_mul_f32 v137, 1077455419, v114
v_mul_f32 v139, 1077455419, v115
v_exp_f32 v133, abs(v133)
v_exp_f32 v135, abs(v135)
v_exp_f32 v137, abs(v137)
v_exp_f32 v139, abs(v139)
v_add_f32 v133, 1.0, v133
v_add_f32 v135, 1.0, v135
v_add_f32 v137, 1.0, v137
v_add_f32 v139, 1.0, v139
v_rcp_f32 v133, v133
v_rcp_f32 v135, v135
v_rcp_f32 v137, v137
v_rcp_f32 v139, v139
v_fma_f32 v133, v133, -2.0, 1.0
v_fma_f32 v135, v135, -2.0, 1.0
v_fma_f32 v137, v137, -2.0, 1.0
v_fma_f32 v139, v139, -2.0, 1.0
v_mov_b32 v134, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v110), v134
v_cndmask_b32 v133, v110, v133, [vcc_lo, vcc_hi]
v_mov_b32 v136, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v111), v136
v_cndmask_b32 v135, v111, v135, [vcc_lo, vcc_hi]
v_mov_b32 v138, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v114), v138
v_cndmask_b32 v137, v114, v137, [vcc_lo, vcc_hi]
v_mov_b32 v140, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v115), v140
v_cndmask_b32 v139, v115, v139, [vcc_lo, vcc_hi]
v_mov_b32 v134, 2147483647
v_bfi_b32 v110, v134, v133, v110
v_mov_b32 v136, 2147483647
v_bfi_b32 v111, v136, v135, v111
v_mov_b32 v138, 2147483647
v_bfi_b32 v114, v138, v137, v114
v_mov_b32 v140, 2147483647
v_bfi_b32 v115, v140, v139, v115
v_mul_f32 v110, v110, s46
v_mul_f32 v111, v111, s46
v_mul_f32 v114, v114, s46
v_mul_f32 v115, v115, s46
v_mul_f32 v112, v112, s47
v_mul_f32 v113, v113, s47
v_mul_f32 v116, v116, s47
v_mul_f32 v117, v117, s47
v_mul_f32 v133, 1077455419, v112
v_mul_f32 v135, 1077455419, v113
v_mul_f32 v137, 1077455419, v116
v_mul_f32 v139, 1077455419, v117
v_exp_f32 v133, abs(v133)
v_exp_f32 v135, abs(v135)
v_exp_f32 v137, abs(v137)
v_exp_f32 v139, abs(v139)
v_add_f32 v133, 1.0, v133
v_add_f32 v135, 1.0, v135
v_add_f32 v137, 1.0, v137
v_add_f32 v139, 1.0, v139
v_rcp_f32 v133, v133
v_rcp_f32 v135, v135
v_rcp_f32 v137, v137
v_rcp_f32 v139, v139
v_fma_f32 v133, v133, -2.0, 1.0
v_fma_f32 v135, v135, -2.0, 1.0
v_fma_f32 v137, v137, -2.0, 1.0
v_fma_f32 v139, v139, -2.0, 1.0
v_mov_b32 v134, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v112), v134
v_cndmask_b32 v133, v112, v133, [vcc_lo, vcc_hi]
v_mov_b32 v136, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v113), v136
v_cndmask_b32 v135, v113, v135, [vcc_lo, vcc_hi]
v_mov_b32 v138, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v116), v138
v_cndmask_b32 v137, v116, v137, [vcc_lo, vcc_hi]
v_mov_b32 v140, 1000587264
v_cmp_gt_f32 [vcc_lo, vcc_hi], abs(v117), v140
v_cndmask_b32 v139, v117, v139, [vcc_lo, vcc_hi]
v_mov_b32 v134, 2147483647
v_bfi_b32 v112, v134, v133, v112
v_mov_b32 v136, 2147483647
v_bfi_b32 v113, v136, v135, v113
v_mov_b32 v138, 2147483647
v_bfi_b32 v116, v138, v137, v116
v_mov_b32 v140, 2147483647
v_bfi_b32 v117, v140, v139, v117
v_mul_f32 v112, v112, s46
v_mul_f32 v113, v113, s46
v_mul_f32 v116, v116, s46
v_mul_f32 v117, v117, s46
s_setprio 0
buffer_store_short_d16_hi v110, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v114, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v112, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v116, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
buffer_store_short_d16_hi v111, v126, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v115, v130, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v113, v127, [s68, s69, s70, s71], 0 idxen
buffer_store_short_d16_hi v117, v131, [s68, s69, s70, s71], 0 idxen
s_add_u32 s68, s68, s73
s_addc_u32 s69, s69, 0
s_sub_u32 s76, s76, 1
s_cselect_b32 s70, 0, s70
s_setprio 2
s_mov_b32 s83, label_114 - label_1
s_bitcmp1_b32 s67, 0
s_cbranch_scc1 label_0
s_sub_u32 s76, s77, 1
s_cselect_b32 s70, 0, 2147483648
s_mov_b64 [s68, s69], [s34, s35]
s_setpc_b64 [s80, s81]
label_0:
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
